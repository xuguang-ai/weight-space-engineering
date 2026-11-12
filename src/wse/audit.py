"""Structured audit events, controller records, and experiment exports.

Every state change of a managed experiment (creation, previews, candidate
commits, selections) and every rejected mutation is recorded as an event in
the same SQLite database. Events that change state are written in the same
transaction as the change; failures are written in their own transaction
after the failed operation has rolled back, so they remain traceable without
affecting any experiment state.

Each experiment also records the controller that drove it
(``controller_runs``: strategy or agent adapter, exact ID and version,
configuration and its hash, constraint status, tracing setting, and the
termination record) and the controller's normalized trace
(``controller_events``: tool calls and results, model responses with usage,
stop conditions, cancellations).

``export_experiment`` writes an operator export of one experiment: hashes of
the base checkpoint, every patch, candidate, and report; the exact patches;
the reports; the audit events and controller trace; the task, policy, costs,
``constraint_status``, and tracing setting; and, by default, the base model
bundle, so ``replay_export`` can recover byte-identical candidate weights in
a clean workspace by applying the exact stored patches (no editing operator
runs). Final-set inputs, labels, and generation seeds are excluded unless
the operator explicitly asks for them.

A run whose policy grants the sandbox also exports its code execution under
``sandbox/``: each call's code, standard output and error as stored, result
summary, and kept output files; the inputs manifest (what the code could
read, with hashes and sources, not the inputs themselves: the base weights
are in the base bundle); and every stored sandbox array, byte for byte, so
each array's file hashes to the digest in the URI an edit names. Replay
needs none of it, but checks its hashes with every other file's.
"""

from __future__ import annotations

import json
import platform
import sqlite3
import sys
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from wse.database import Database, audit_events, insert_audit_event, utc_now
from wse.policy import sandbox_available
from wse.schemas import (
    SCHEMA_VERSION,
    ErrorCode,
    MaterializedPatch,
    WSEError,
    _Frozen,
    _optional,
    canonical_json,
    content_hash,
    json_hash,
)

if TYPE_CHECKING:
    from wse.api import Engine

__all__ = [
    "ExportManifest",
    "ReplayResult",
    "append_controller_event",
    "controller_events",
    "controller_record",
    "experiment_events",
    "export_experiment",
    "record_controller_end",
    "record_controller_start",
    "record_event",
    "record_failure",
    "replay_export",
]


def record_event(
    conn: sqlite3.Connection,
    event_type: str,
    subject_id: str | None,
    payload: dict[str, Any],
    *,
    experiment_id: str | None = None,
) -> None:
    """Append one event inside the caller's transaction."""
    insert_audit_event(conn, event_type, subject_id, payload, experiment_id=experiment_id)


def record_failure(
    db: Database,
    operation: str,
    error: WSEError,
    *,
    experiment_id: str | None,
    subject_id: str | None = None,
    request_hash: str | None = None,
) -> None:
    """Record a rejected mutation as an ``<operation>.failed`` event (own transaction)."""
    try:
        details = json.loads(canonical_json(error.details))
    except WSEError:
        details = {"unserializable_details": sorted(error.details)}
    with db.transaction() as conn:
        insert_audit_event(
            conn,
            f"{operation}.failed",
            subject_id,
            {
                "code": error.code.value,
                "message": error.message,
                "details": details,
                "request_hash": request_hash,
            },
            experiment_id=experiment_id,
        )


def experiment_events(db: Database, experiment_id: str) -> list[dict[str, Any]]:
    """All events of one experiment, oldest first."""
    return audit_events(db.conn, experiment_id=experiment_id)


# ---------------------------------------------------------------------------
# Controller records
# ---------------------------------------------------------------------------


def record_controller_start(db: Database, experiment_id: str, *, kind: str, controller_id: str,
                            controller_version: str, config: Mapping[str, Any], constraint_status: str,
                            tracing: Mapping[str, Any], constraint_reason: str | None = None) -> dict[str, Any]:
    """Record which controller drives an experiment (once per experiment).

    ``constraint_reason`` says why the run is unverified when its policy
    decides it (a sandbox); the record names it only when it is set.
    """
    config_json = canonical_json(dict(config))
    now = utc_now()
    with db.transaction() as conn:
        existing = conn.execute("SELECT controller_id FROM controller_runs WHERE experiment_id = ?",
                                (experiment_id,)).fetchone()
        if existing is not None:
            raise WSEError(ErrorCode.VERSION_CONFLICT,
                           f"experiment {experiment_id} already has a controller ({existing['controller_id']}); "
                           "each run grant permits one controller run",
                           details={"experiment_id": experiment_id})
        conn.execute(
            "INSERT INTO controller_runs (experiment_id, kind, controller_id, controller_version, config, "
            "config_hash, constraint_status, tracing, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (experiment_id, kind, controller_id, controller_version, config_json,
             json_hash(json.loads(config_json)), constraint_status, canonical_json(dict(tracing)), now))
        if constraint_reason is not None:
            conn.execute("INSERT INTO run_constraints (experiment_id, reason) VALUES (?, ?)",
                         (experiment_id, constraint_reason))
        payload = {"kind": kind, "controller": f"{controller_id}@{controller_version}",
                   "config_hash": json_hash(json.loads(config_json)),
                   "constraint_status": constraint_status, "tracing": dict(tracing)}
        if constraint_reason is not None:
            payload["constraint_reason"] = constraint_reason
        insert_audit_event(conn, "controller.started", experiment_id, payload, experiment_id=experiment_id)
    return controller_record(db, experiment_id)


def append_controller_event(db: Database, experiment_id: str, kind: str, payload: Mapping[str, Any]) -> int:
    """Append one normalized controller event; returns its sequence number."""
    with db.transaction() as conn:
        seq = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM controller_events WHERE experiment_id = ?",
                           (experiment_id,)).fetchone()[0]
        conn.execute("INSERT INTO controller_events (experiment_id, seq, created_at, kind, payload) "
                     "VALUES (?, ?, ?, ?, ?)", (experiment_id, seq, utc_now(), kind, canonical_json(dict(payload))))
    return int(seq)


def record_controller_end(db: Database, experiment_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Record how the controller ended (its termination status, reason, and usage)."""
    with db.transaction() as conn:
        conn.execute("UPDATE controller_runs SET ended_at = ?, result = ? WHERE experiment_id = ?",
                     (utc_now(), canonical_json(dict(result)), experiment_id))
        insert_audit_event(conn, "controller.finished", experiment_id,
                           {"status": result.get("status"), "reason": result.get("reason")},
                           experiment_id=experiment_id)
    return controller_record(db, experiment_id)


def controller_record(db: Database, experiment_id: str) -> dict[str, Any] | None:
    row = db.conn.execute("SELECT * FROM controller_runs WHERE experiment_id = ?", (experiment_id,)).fetchone()
    if row is None:
        return None
    reason = db.conn.execute("SELECT reason FROM run_constraints WHERE experiment_id = ?",
                             (experiment_id,)).fetchone()
    return {
        "experiment_id": row["experiment_id"],
        "kind": row["kind"],
        "controller": {"extension_id": row["controller_id"], "version": row["controller_version"]},
        "config": json.loads(row["config"]),
        "config_hash": row["config_hash"],
        "constraint_status": row["constraint_status"],
        **({"constraint_reason": reason["reason"]} if reason is not None else {}),
        "tracing": json.loads(row["tracing"]),
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "result": None if row["result"] is None else json.loads(row["result"]),
    }


def controller_events(db: Database, experiment_id: str) -> list[dict[str, Any]]:
    rows = db.conn.execute("SELECT seq, created_at, kind, payload FROM controller_events WHERE experiment_id = ? "
                           "ORDER BY seq", (experiment_id,)).fetchall()
    return [{"seq": r["seq"], "created_at": r["created_at"], "kind": r["kind"], **json.loads(r["payload"])}
            for r in rows]


# ---------------------------------------------------------------------------
# Export and replay
# ---------------------------------------------------------------------------

EXPORT_MANIFEST_NAME = "export_manifest.json"
BASE_BUNDLE_DIR = "base_model"
SANDBOX_EXPORT_DIR = "sandbox"
SANDBOX_CALL_FILES = ("code.py", "stdout.txt", "stderr.txt", "result.json")
DEFAULT_EXCLUSIONS = ("final-set inputs", "final-set labels", "final-set generation seed")


class ExportedCheckpoint(_Frozen):
    checkpoint_id: str
    role: Literal["base", "candidate"]
    parent_checkpoint_id: str | None
    patch_id: str | None
    weights_hash: str
    config_hash: str
    created_at: str


class ExportedPatch(_Frozen):
    patch_id: str
    spec_hash: str
    base_checkpoint_id: str
    base_weights_hash: str
    result_weights_hash: str
    exact_patch_hash: str
    applied_checkpoint_id: str | None
    hypothesis: str
    changed_elements: int
    path: str


class ExportedReport(_Frozen):
    report_id: str
    split: str
    checkpoint_id: str
    suite_id: str
    report_hash: str
    metrics: dict[str, float]
    sample_count: int
    target_met: bool | None
    path: str


class ExportManifest(_Frozen):
    """The index of an experiment export (``export_manifest.json``).

    ``files`` maps every other exported file to its content hash.
    ``excluded`` lists what the export deliberately leaves out. ``sandbox``
    indexes the run's code execution when its policy grants the sandbox
    (module docstring); it is absent otherwise.
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    created_at: str
    experiment_id: str
    run_id: str
    task_id: str
    task_hash: str | None
    policy_id: str
    policy_hash: str
    mode: str
    state: str
    management: Literal["managed"] = "managed"
    base: ExportedCheckpoint
    selected_checkpoint_id: str
    checkpoints: list[ExportedCheckpoint]
    patches: list[ExportedPatch]
    reports: list[ExportedReport]
    final: dict[str, Any] | None
    controller: dict[str, Any] | None
    constraint_status: str
    tracing: dict[str, Any]
    costs: dict[str, Any]
    environment: dict[str, Any]
    replay: dict[str, Any]
    excluded: list[str]
    included_final_data: bool = False
    files: dict[str, str] = Field(default_factory=dict)
    sandbox: dict[str, Any] | None = _optional()


class ReplayResult(_Frozen):
    """Outcome of replaying an export: every candidate's recorded and replayed weights hash."""

    base_checkpoint_id: str
    base_weights_hash: str
    candidates: list[dict[str, Any]]
    all_identical: bool


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("wse", "torch", "numpy", "safetensors", "pydantic", "mcp", "openai-agents", "openai"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def environment_record() -> dict[str, Any]:
    """Versions and the CPU environment, recorded with exports and runs."""
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "device": "cpu", "packages": _package_versions()}


def _write_json(root: Path, relative: str, value: Any, files: dict[str, str]) -> str:
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    files[relative] = content_hash(text.encode("utf-8"))
    return relative


def _copy_file(source: Path, root: Path, relative: str, files: dict[str, str]) -> str:
    data = source.read_bytes()
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    files[relative] = content_hash(data)
    return relative


def _export_sandbox(engine: Engine, experiment: Any, root: Path, files: dict[str, str]) -> dict[str, Any]:
    """Copy the run's sandbox record into ``root/sandbox`` (module docstring) and return its index."""
    from wse.sandbox_runs import ARRAY_KIND, MANIFEST, layout_for
    from wse.store import ANALYSIS_URI_PREFIX

    layout = layout_for(engine.experiments, experiment.run_id)
    calls = []
    numbered = sorted((p for p in layout.calls.iterdir() if p.is_dir() and not p.is_symlink() and p.name.isdigit()),
                      key=lambda p: int(p.name)) if layout.calls.is_dir() else []
    for directory in numbered:
        prefix = f"{SANDBOX_EXPORT_DIR}/calls/{directory.name}"
        entry: dict[str, Any] = {"call": int(directory.name)}
        for name in SANDBOX_CALL_FILES:
            source = directory / name
            if source.is_file() and not source.is_symlink():
                entry[name] = _copy_file(source, root, f"{prefix}/{name}", files)
        kept = directory / "outputs"
        entry["outputs"] = {
            path.relative_to(kept).as_posix(): _copy_file(path, root, f"{prefix}/outputs/"
                                                          f"{path.relative_to(kept).as_posix()}", files)
            for path in sorted(kept.rglob("*")) if path.is_file() and not path.is_symlink()
        } if kept.is_dir() else {}
        calls.append(entry)
    inputs = layout.inputs / MANIFEST
    arrays = []
    rows = engine.store.db.conn.execute("SELECT uri FROM analysis_results WHERE experiment_id = ? "
                                        "ORDER BY created_at, rowid", (experiment.experiment_id,)).fetchall()
    for row in rows:
        data = engine.store.read_analysis_result(row["uri"])
        document = json.loads(data)
        if not isinstance(document, dict) or document.get("kind") != ARRAY_KIND:
            continue
        relative = f"{SANDBOX_EXPORT_DIR}/arrays/{row['uri'].removeprefix(ANALYSIS_URI_PREFIX)}.json"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files[relative] = content_hash(data)
        arrays.append({"uri": row["uri"], "call": document["call"], "path": document["path"],
                       "shape": document["shape"], "dtype": document["dtype"], "file": relative})
    return {"calls": calls, "arrays": arrays,
            "inputs_manifest": _copy_file(inputs, root, f"{SANDBOX_EXPORT_DIR}/inputs_manifest.json", files)
            if inputs.is_file() else None}


def export_experiment(engine: Engine, experiment_id: str, output_dir: str | Path, *,
                      include_base_weights: bool = True, include_final_data: bool = False) -> ExportManifest:
    """Write an operator export of ``experiment_id`` into ``output_dir`` (which must be empty or absent).

    Operator only. Final-set inputs, labels, and generation seeds are
    excluded unless ``include_final_data`` is true.
    """
    engine._require_operator("exporting an experiment")
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"export directory {root} is not empty; choose a new directory")
    root.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    db = engine.store.db
    experiment = engine.get_experiment(experiment_id)
    policy = engine.experiment_policy(experiment_id)
    grant = engine.experiments.get_run_grant(experiment.run_id)
    try:
        task = engine.get_task(experiment.task_id)
    except WSEError as exc:
        if exc.code != ErrorCode.NOT_FOUND:
            raise
        task = None

    checkpoints = []
    for owned in engine.list_experiment_checkpoints(experiment_id):
        manifest = engine.load_manifest(owned.checkpoint_id)
        checkpoints.append(ExportedCheckpoint(
            checkpoint_id=owned.checkpoint_id, role=owned.role, parent_checkpoint_id=owned.parent_checkpoint_id,
            patch_id=owned.patch_id, weights_hash=owned.weights_hash, config_hash=manifest.config_hash,
            created_at=owned.created_at))
        _write_json(root, f"checkpoints/{owned.checkpoint_id}.manifest.json", manifest.model_dump(mode="json"), files)
    base = next(c for c in checkpoints if c.role == "base")

    patches = []
    rows = db.conn.execute("SELECT patch_id, applied_checkpoint_id FROM patches WHERE experiment_id = ? "
                           "ORDER BY created_at, rowid", (experiment_id,)).fetchall()
    for row in rows:
        artifact = engine.read_patch(experiment_id, row["patch_id"])
        preview = engine.get_patch(experiment_id, row["patch_id"])
        path = _write_json(root, f"patches/{row['patch_id']}.json",
                           {"preview": preview.model_dump(mode="json"), "artifact": artifact.model_dump(mode="json")},
                           files)
        patches.append(ExportedPatch(
            patch_id=row["patch_id"], spec_hash=preview.spec_hash, base_checkpoint_id=preview.base_checkpoint_id,
            base_weights_hash=preview.base_weights_hash, result_weights_hash=preview.result_weights_hash,
            exact_patch_hash=json_hash(artifact.patch.model_dump(mode="json")),
            applied_checkpoint_id=row["applied_checkpoint_id"], hypothesis=preview.hypothesis,
            changed_elements=preview.changed_elements, path=path))

    reports = []
    for report in engine.list_reports(experiment_id):
        data = report.model_dump(mode="json")
        path = _write_json(root, f"reports/{report.report_id}.json", data, files)
        reports.append(ExportedReport(report_id=report.report_id, split=report.split,
                                      checkpoint_id=report.checkpoint_id, suite_id=report.suite_id,
                                      report_hash=json_hash(data), metrics=dict(report.metrics),
                                      sample_count=report.sample_count, target_met=report.target_met, path=path))
    final = None
    if experiment.state == "CLOSED":
        final_report = engine.get_final_report(experiment_id)
        data = final_report.model_dump(mode="json")
        final = {"path": _write_json(root, "final_report.json", data, files), "report_hash": json_hash(data),
                 "candidate_checkpoint_id": final_report.candidate_checkpoint_id,
                 "control_checkpoint_id": final_report.control_checkpoint_id,
                 "candidate_accuracy": final_report.comparison.candidate_value,
                 "control_accuracy": final_report.comparison.base_value,
                 "difference": final_report.comparison.difference,
                 "candidate_fully_correct": final_report.comparison.candidate_fully_correct.model_dump(mode="json"),
                 "difference_interval": final_report.comparison.difference_interval.model_dump(mode="json")}

    _write_json(root, "experiment.json", {"experiment": experiment.model_dump(mode="json"),
                                          "run_grant": grant.model_dump(mode="json")}, files)
    _write_json(root, "policy.json", policy.model_dump(mode="json"), files)
    if task is not None:
        _write_json(root, "task.json", task.model_dump(mode="json"), files)
    _write_json(root, "audit_events.json", engine.audit_log(experiment_id), files)
    controller = controller_record(db, experiment_id)
    trace = controller_events(db, experiment_id)
    _write_json(root, "controller.json", controller, files)
    _write_json(root, "tool_trace.json", trace, files)

    sandbox_record = _export_sandbox(engine, experiment, root, files) if sandbox_available(policy) else None
    if include_base_weights:
        _export_bundle(engine, base.checkpoint_id, root / BASE_BUNDLE_DIR, files)
    if include_final_data:
        final_suite = engine.evaluation.final_suite(experiment.task_id)
        data = engine.datasets.load(final_suite.dataset)
        _write_json(root, "final_data/final_set.json",
                    {"suite_id": final_suite.suite_id, "data": data.model_dump(mode="json"),
                     "generation": engine.datasets.generation(final_suite.dataset.data_hash)}, files)

    result = (controller or {}).get("result") or {}
    selected_manifest = engine.load_manifest(experiment.selected_checkpoint_id)
    selected_patch = next((p for p in patches if p.applied_checkpoint_id == experiment.selected_checkpoint_id), None)
    costs = {
        "budget": {name: entry.model_dump(mode="json") for name, entry in experiment.budget.items()},
        "candidates": sum(1 for c in checkpoints if c.role == "candidate"),
        "evaluated_examples": experiment.budget["evaluated_examples"].consumed,
        "controller_usage": result.get("usage"),
        "controller_elapsed_seconds": result.get("elapsed_seconds"),
        "selected_changed_elements": selected_patch.changed_elements if selected_patch else 0,
        "parameter_count": selected_manifest.parameter_count,
        "selected_changed_fraction": (selected_patch.changed_elements / selected_manifest.parameter_count
                                      if selected_patch else 0.0),
    }
    if sandbox_record is not None:
        from wse.runner import SANDBOX_TIME_LABEL

        costs["sandbox"] = {"calls": experiment.budget["sandbox_calls"].consumed,
                            "wall_seconds": experiment.budget["sandbox_seconds"].consumed,
                            "label": SANDBOX_TIME_LABEL}
    manifest = ExportManifest(
        created_at=utc_now(), experiment_id=experiment_id, run_id=experiment.run_id, task_id=experiment.task_id,
        task_hash=None if task is None else task.task_hash, policy_id=experiment.policy_id,
        policy_hash=experiment.policy_hash, mode=experiment.mode, state=experiment.state, base=base,
        selected_checkpoint_id=experiment.selected_checkpoint_id, checkpoints=checkpoints, patches=patches,
        reports=reports, final=final, controller=controller,
        constraint_status=(controller or {}).get("constraint_status", "unverified"),
        tracing=(controller or {}).get("tracing", {"sdk_tracing": "no controller recorded"}),
        costs=costs, environment=environment_record(),
        replay={"base_bundle": BASE_BUNDLE_DIR if include_base_weights else None,
                "method": "exact patch replay: stored after-values and hashes, no editing operator runs",
                "recipe_reexecution": "not performed; re-running a recipe needs fresh evaluation"},
        excluded=[] if include_final_data else list(DEFAULT_EXCLUSIONS), included_final_data=include_final_data,
        files=dict(sorted(files.items())), sandbox=sandbox_record)
    (root / EXPORT_MANIFEST_NAME).write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _export_bundle(engine: Engine, checkpoint_id: str, directory: Path, files: dict[str, str]) -> None:
    manifest = engine.load_manifest(checkpoint_id)
    directory.mkdir(parents=True)
    engine.store.export_safetensors(checkpoint_id, directory / "model.safetensors")
    files[f"{BASE_BUNDLE_DIR}/model.safetensors"] = content_hash((directory / "model.safetensors").read_bytes())
    root = directory.parent
    _write_json(root, f"{BASE_BUNDLE_DIR}/config.json", manifest.config, files)
    _write_json(root, f"{BASE_BUNDLE_DIR}/wse_bundle.json", {
        "schema_version": "1", "adapter": str(manifest.adapter), "weights": "model.safetensors",
        "config": "config.json", "license": manifest.license,
        "provenance": {"kind": "import", "source": manifest.provenance.source, "notes": manifest.provenance.notes},
        "input_description": manifest.input_semantics.description,
        "output_description": manifest.output_semantics.description}, files)


def replay_export(export_dir: str | Path, engine: Engine) -> ReplayResult:
    """Replay an export into ``engine``'s workspace and verify byte-identical candidate weights.

    Imports the exported base bundle, checks its weights hash, then rebuilds
    every candidate from its parent by applying the stored exact patch
    (``replay_patch``: recorded after-values and per-step hashes, no editing
    operator) and imports the result. Each candidate's replayed weights hash
    must equal the recorded one.
    """
    from wse.patches import replay_patch
    from wse.schemas import ImportManifest

    engine._require_operator("replaying an export")
    root = Path(export_dir)
    manifest = ExportManifest.model_validate_json((root / EXPORT_MANIFEST_NAME).read_text(encoding="utf-8"))
    for relative, expected in manifest.files.items():
        if content_hash((root / relative).read_bytes()) != expected:
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"exported file {relative} does not match its recorded hash")
    if manifest.replay.get("base_bundle") is None:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "this export has no base model bundle; export with base weights")
    base = engine.import_bundle(root / BASE_BUNDLE_DIR)
    if base.weights_hash != manifest.base.weights_hash:
        raise WSEError(ErrorCode.ARTIFACT_CORRUPT, "the exported base weights do not match the recorded hash")
    handles = {manifest.base.checkpoint_id: engine.open_checkpoint(base.checkpoint_id)}
    patches = {patch.patch_id: patch for patch in manifest.patches}
    candidates = []
    for checkpoint in manifest.checkpoints:
        if checkpoint.role != "candidate":
            continue
        stored = json.loads((root / patches[checkpoint.patch_id].path).read_text(encoding="utf-8"))
        exact = MaterializedPatch.model_validate(stored["artifact"]["patch"])
        replayed = replay_patch(handles[checkpoint.parent_checkpoint_id], exact)
        handles[checkpoint.checkpoint_id] = replayed
        imported = engine.import_model(ImportManifest(
            adapter=replayed.adapter_ref, config=replayed.config,
            provenance={"kind": "import", "source": f"replay of {checkpoint.checkpoint_id}"},
            license="replayed from an experiment export"), replayed.tensors())
        candidates.append({"checkpoint_id": checkpoint.checkpoint_id, "patch_id": checkpoint.patch_id,
                           "recorded_weights_hash": checkpoint.weights_hash,
                           "replayed_weights_hash": imported.weights_hash,
                           "replayed_checkpoint_id": imported.checkpoint_id,
                           "identical": imported.weights_hash == checkpoint.weights_hash})
    return ReplayResult(base_checkpoint_id=base.checkpoint_id, base_weights_hash=base.weights_hash,
                        candidates=candidates, all_identical=all(c["identical"] for c in candidates))
