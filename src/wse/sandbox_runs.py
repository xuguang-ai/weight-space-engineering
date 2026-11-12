"""Code execution in a run: the sandbox's inputs, the ``run_python`` operation, and its arrays (M3 plan Section 2).

``run_python`` is the managed operation behind ``wse_run_python``. Before
each call it brings the run's sandbox inputs up to date (``sync_inputs``),
exporting only data that the run may already read through its tools:

* ``inputs/weights/base.safetensors``: the base checkpoint's tensors that the
  sandbox may read (``readable_tensors``), in their stored dtype: every
  tensor that is neither a member of an alias group nor protected by the
  model's family descriptor, narrowed by ``SandboxPolicy.readable_tensors``;
* ``inputs/passes/<report_id>.jsonl``: one line per output of each of the
  run's development generation passes (problem, item, sample, grade,
  answer, finished, limit hit, generated tokens, and text), when the policy
  sets ``export_pass_texts``;
* ``inputs/probes/<job_id>.json`` and ``.npz``: each succeeded probe's
  document and its vectors (``site<i>_mean``, ``site<i>_group_mean_a`` and
  ``_b``, ``site<i>_contrast``);
* ``inputs/manifest.json``: every input with its size and SHA-256.

Final suites are never exported, and a run-bound engine cannot read them.

A call then checks that one sandbox call and some sandbox time remain,
runs the code (``wse.sandbox.run_python``) with a wall-clock limit of the
policy's ``seconds_per_call`` or the time left, whichever is less, and
charges ``sandbox_calls`` and the measured wall-clock time to
``sandbox_seconds``. Nothing is held while the code runs, so an interrupted
call leaves no reservation behind. ``compute_seconds`` is not charged: WSE
does not meter what the code computes.

Every regular file the code left in ``outputs/`` (emptied before each call)
is listed with its size and SHA-256. A ``.npy`` file holding a finite numeric array of one or two
dimensions, of at most ``MAX_REFERENCE_ELEMENTS`` values, is also stored as a
run-scoped result (``kind: sandbox_array``) whose pointer ``#/values`` (or
``#/values/<row>`` for a row of a matrix) names a stored vector, which edits
such as ``project_out`` and analyses such as ``direction_projection``
accept. Files are read without following links (``wse.sandbox``).

The call is recorded in the audit log (``sandbox.ran``) with its code hash,
ending, times, and output files, and it is idempotent by its key: a retry
with the same code returns the first result without running again.
"""

from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from wse import sandbox
from wse.database import insert_audit_event, utc_now
from wse.policy import ExperimentPolicy, sandbox_available
from wse.sandbox import SandboxLayout
from wse.schemas import (
    IDEMPOTENCY_KEY_PATTERN,
    Experiment,
    ErrorCode,
    SandboxOutput,
    SandboxRun,
    Usage,
    WSEError,
    canonical_json,
    json_hash,
)

if TYPE_CHECKING:
    from wse.experiments import ExperimentService

__all__ = ["MAX_CODE_BYTES", "MAX_REFERENCE_ELEMENTS", "layout_for", "readable_tensors", "run_python",
           "sync_inputs"]

OPERATION = "run_python"
MAX_CODE_BYTES = 256 * 1024
MAX_REFERENCE_ELEMENTS = 65536
MAX_ARRAY_FILE_BYTES = 64 * 1024 * 1024
ARRAY_KIND = "sandbox_array"
MANIFEST = "manifest.json"


def layout_for(service: ExperimentService, run_id: str) -> SandboxLayout:
    """The sandbox directory of a run: ``<workspace>/sandboxes/<run_id>``."""
    return SandboxLayout(Path(service.engine.store.workspace) / "sandboxes" / run_id)


def _alias_members(manifest: Any) -> set[str]:
    members: set[str] = set()
    for group in manifest.alias_groups or ():
        names = getattr(group, "tensors", None)
        if names is None and isinstance(group, (list, tuple)):
            names = group
        members.update(str(name) for name in names or ())
    return members


def readable_tensors(service: ExperimentService, checkpoint_id: str, policy: ExperimentPolicy) -> list[str]:
    """The tensors of ``checkpoint_id`` that the sandbox may read (module docstring)."""
    from wse.adapters.families import model_family

    manifest = service.engine.store.load_manifest(checkpoint_id)
    family = model_family(manifest, registry=service.engine.registry)
    aliased = _alias_members(manifest)
    patterns = (policy.sandbox.readable_tensors if policy.sandbox is not None else None) or None
    names = []
    for info in manifest.tensors:
        name = info.name
        if name in aliased or (family is not None and family.protected_role(name) is not None):
            continue
        if patterns is not None and not any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns):
            continue
        names.append(name)
    return names


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _publish(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _floats(values: Any) -> np.ndarray:
    return np.asarray([float(value) for value in values], dtype=np.float32)


def _probe_arrays(document: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for index, site in enumerate(document.get("sites") or ()):
        if isinstance(site.get("mean"), list):
            arrays[f"site{index}_mean"] = _floats(site["mean"])
        for group, values in (site.get("group_means") or {}).items():
            if isinstance(values, list):
                arrays[f"site{index}_group_mean_{group}"] = _floats(values)
        contrast = site.get("contrast") or {}
        if isinstance(contrast.get("direction"), list):
            arrays[f"site{index}_contrast"] = _floats(contrast["direction"])
    return arrays


def sync_inputs(service: ExperimentService, experiment: Experiment, policy: ExperimentPolicy,
                layout: SandboxLayout) -> list[str]:
    """Export every input the run may read that is not exported yet; return the inputs, relative to ``inputs/``."""
    manifest_path = layout.inputs / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"files": {}}
    files: dict[str, dict[str, Any]] = manifest["files"]

    def record(relative: str, path: Path, source: dict[str, Any]) -> None:
        files[relative] = {"size": path.stat().st_size, "sha256": _sha256(path), "source": source}

    weights = layout.inputs / "weights" / "base.safetensors"
    if "weights/base.safetensors" not in files or not weights.exists():
        from safetensors.torch import save_file

        handle = service.engine._open_checkpoint(experiment.base_checkpoint_id)
        names = readable_tensors(service, experiment.base_checkpoint_id, policy)
        weights.parent.mkdir(parents=True, exist_ok=True)
        temporary = weights.with_name(weights.name + ".tmp")
        save_file({name: handle.tensor(name).contiguous() for name in names}, str(temporary),
                  metadata={"checkpoint_id": experiment.base_checkpoint_id, "weights_hash": handle.weights_hash})
        os.replace(temporary, weights)
        record("weights/base.safetensors", weights, {"checkpoint_id": experiment.base_checkpoint_id,
                                                     "tensors": len(names)})

    read = service.experiment_results(experiment.experiment_id)
    if policy.sandbox is not None and policy.sandbox.export_pass_texts:
        for report in service.engine.evaluation.list_reports(experiment.experiment_id):
            relative = f"passes/{report.report_id}.jsonl"
            if report.split != "development" or not report.problems or relative in files:
                continue
            lines = []
            for problem in report.problems:
                for sample in problem.samples:
                    text = read(sample.output_uri)["text"] if sample.output_uri else None
                    lines.append(json.dumps({
                        "problem": problem.index, "item_id": problem.item_id, "sample": sample.sample,
                        "correct": sample.correct, "answer": sample.answer, "finished": sample.finished,
                        "limit_hit": sample.limit_hit, "generated_tokens": sample.generated_tokens, "text": text,
                    }, ensure_ascii=False, sort_keys=True))
            path = layout.inputs / relative
            _publish(path, ("\n".join(lines) + "\n").encode("utf-8"))
            record(relative, path, {"report_id": report.report_id, "checkpoint_id": report.checkpoint_id,
                                    "suite_id": report.suite_id})

    for job in service.engine.jobs.list_jobs(experiment.experiment_id):
        relative = f"probes/{job.job_id}.npz"
        if job.kind != "probe" or job.state != "succeeded" or not job.result or relative in files:
            continue
        document = read(job.result["result_uri"])
        summary = layout.inputs / "probes" / f"{job.job_id}.json"
        _publish(summary, (json.dumps(document, sort_keys=True) + "\n").encode("utf-8"))
        buffer = io.BytesIO()
        np.savez(buffer, **_probe_arrays(document))
        path = layout.inputs / relative
        _publish(path, buffer.getvalue())
        source = {"job_id": job.job_id, "checkpoint_id": job.checkpoint_id, "result_uri": job.result["result_uri"]}
        record(f"probes/{job.job_id}.json", summary, source)
        record(relative, path, source)

    manifest["updated_at"] = utc_now()
    _publish(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return sorted([*files, MANIFEST])


def _load_array(layout: SandboxLayout, output: sandbox.OutputFile) -> np.ndarray | None:
    """The array in ``output`` when it is a regular, singly linked ``.npy`` file that still has its listed hash."""
    data = sandbox.read_output(layout, output, MAX_ARRAY_FILE_BYTES)
    if data is None:
        return None
    try:
        return np.load(io.BytesIO(data), allow_pickle=False)
    except (ValueError, OSError, EOFError):
        return None


def _referenceable(array: np.ndarray) -> bool:
    return (array.ndim in (1, 2) and 0 < array.size <= MAX_REFERENCE_ELEMENTS
            and np.issubdtype(array.dtype, np.number) and not np.issubdtype(array.dtype, np.complexfloating)
            and bool(np.isfinite(array).all()))


def _outputs(service: ExperimentService, experiment_id: str, layout: SandboxLayout,
             result: sandbox.SandboxResult) -> list[SandboxOutput]:
    outputs = []
    for output in result.outputs:
        shape = dtype = pointer = None
        if output.path.endswith(".npy"):
            array = _load_array(layout, output)
            if array is not None:
                shape, dtype = list(array.shape), str(array.dtype)
                if _referenceable(array):
                    document = {"kind": ARRAY_KIND, "experiment_id": experiment_id, "call": result.call,
                                "path": output.path, "sha256": output.sha256, "shape": shape, "dtype": dtype,
                                "values": array.astype(np.float64).tolist()}
                    uri = service.store.put_analysis_result(canonical_json(document).encode("utf-8"))
                    with service.db.transaction() as conn:
                        conn.execute("INSERT OR IGNORE INTO analysis_results (uri, experiment_id, created_at) "
                                     "VALUES (?, ?, ?)", (uri, experiment_id, utc_now()))
                    pointer = f"{uri}#/values"
        outputs.append(SandboxOutput(path=output.path, size=output.size, sha256=output.sha256, shape=shape,
                                     dtype=dtype, pointer=pointer))
    return outputs


def run_python(service: ExperimentService, experiment_id: str, code: Any, key: Any) -> SandboxRun:
    """Run ``code`` in the run's sandbox, charge it, record it, and return its result (module docstring)."""
    if not isinstance(key, str) or not IDEMPOTENCY_KEY_PATTERN.fullmatch(key):
        raise WSEError(ErrorCode.INVALID_ARGUMENT,
                       f"idempotency_key must match {IDEMPOTENCY_KEY_PATTERN.pattern}; reuse the same key to retry "
                       "the same request")
    if not isinstance(code, str) or not code.strip():
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "code must be a non-empty string of Python")
    if len(code.encode("utf-8")) > MAX_CODE_BYTES:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"code has more than {MAX_CODE_BYTES} bytes",
                       details={"max_bytes": MAX_CODE_BYTES})
    row = service._experiment_row(experiment_id)
    policy = service.experiment_policy(experiment_id)
    if not sandbox_available(policy):
        raise WSEError(ErrorCode.POLICY_DENIED, f"policy {policy.policy_id} grants no code execution; use the "
                       "analysis and probe tools instead", details={"policy_id": policy.policy_id})
    request_hash = json_hash({"code": code})
    with service.db.transaction() as conn:
        recorded = service._lookup(conn, experiment_id, key, OPERATION, request_hash)
    if recorded is not None:
        return SandboxRun.model_validate_json(recorded)
    service._require_open(row)
    ok, reason = sandbox.available()
    if not ok:
        raise WSEError(ErrorCode.UNSUPPORTED_CAPABILITY, f"code execution is not available here: {reason}",
                       details={"reason": reason})
    assert policy.sandbox is not None
    service.ledger.require(service.db.conn, experiment_id, {"sandbox_calls": 1})
    experiment = service.get_experiment(experiment_id)
    seconds_left = float(experiment.budget["sandbox_seconds"].remaining)
    if seconds_left <= 0:
        service.ledger.require(service.db.conn, experiment_id, {"sandbox_seconds": 1})
    wall = min(policy.sandbox.seconds_per_call, seconds_left)
    layout = layout_for(service, experiment.run_id).create()
    inputs = sync_inputs(service, experiment, policy, layout)
    result = sandbox.run_python(layout, code, wall_seconds=wall,
                                cpu_seconds=max(1, math.ceil(wall * (os.cpu_count() or 1))),
                                max_output_bytes=policy.sandbox.max_output_bytes)
    seconds = round(min(result.wall_seconds, wall), 3)
    outputs = _outputs(service, experiment_id, layout, result)
    run = SandboxRun(
        call=result.call, ending=result.ending, exit_code=result.exit_code, signal=result.signal,
        wall_seconds=result.wall_seconds, cpu_seconds=result.cpu_seconds, max_rss_bytes=result.max_rss_bytes,
        stdout=result.stdout, stderr=result.stderr, stdout_bytes=result.stdout_bytes,
        stderr_bytes=result.stderr_bytes, stdout_truncated=result.stdout_truncated,
        stderr_truncated=result.stderr_truncated, outputs=outputs, refused_outputs=list(result.refused_outputs),
        inputs=inputs, code_sha256=result.code_sha256, usage=Usage(sandbox_calls=1, sandbox_seconds=seconds))
    with service.db.transaction() as conn:
        service.ledger.charge(conn, experiment_id, "sandbox_calls", 1)
        service.ledger.charge(conn, experiment_id, "sandbox_seconds", seconds)
        service._remember(conn, experiment_id, key, OPERATION, request_hash, run.model_dump_json())
        insert_audit_event(conn, "sandbox.ran", experiment_id, {
            "call": run.call, "code_sha256": run.code_sha256, "ending": run.ending, "exit_code": run.exit_code,
            "wall_seconds": run.wall_seconds, "cpu_seconds": run.cpu_seconds, "max_rss_bytes": run.max_rss_bytes,
            "outputs": [{"path": o.path, "size": o.size, "sha256": o.sha256} for o in outputs],
            "refused_outputs": run.refused_outputs, "charged": {"sandbox_calls": 1, "sandbox_seconds": seconds},
        }, experiment_id=experiment_id)
    return run
