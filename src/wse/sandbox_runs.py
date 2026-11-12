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
runs the code as the next call of the run's kernel (``wse.sandbox``: a
persistent process, so variables survive between calls) with a wall-clock
limit of the policy's ``seconds_per_call`` or the time left, whichever is
less, and charges ``sandbox_calls`` and the measured wall-clock time to
``sandbox_seconds``. ``restart`` starts a new kernel first. Nothing is held
while the code runs, so an interrupted call leaves no reservation behind.
``compute_seconds`` is not charged: WSE does not meter what the code
computes. A result whose two copies in a tool response would exceed twice
the policy's ``max_result_json_bytes`` has its standard output and error
shortened, from their stored bytes, to the largest common bound that fits,
and at most ``MAX_LISTED_OUTPUTS`` output files are listed and stored.

Every regular file the code left in ``outputs/`` (emptied before each call)
is listed with its size and SHA-256. A ``.safetensors`` file of
floating-point tensors, of at most ``MAX_TENSOR_FILE_BYTES``, is stored as
a run-scoped tensor file (``wse://tensors/<sha256>``, table
``tensor_files``) that ``submit_candidate`` turns into a candidate: WSE
reads it from the descriptor the listing opened it as (never through a
link), checks that it is a regular, singly linked file, copies it into the
store from that descriptor, checks the copy's SHA-256, and only then parses
the copy's header, which must list only floating-point tensors; a file that
fails is never published. An experiment's tensor files hold at most
``MAX_EXPERIMENT_TENSOR_BYTES``, checked before anything is copied. A
``.npy`` file holding a finite numeric array of one or two dimensions, of at
most ``MAX_REFERENCE_ELEMENTS`` values, is also stored as a run-scoped
result (``kind: sandbox_array``) whose pointer ``#/values`` (or
``#/values/<row>`` for a row of a matrix) names a stored vector, which edits
such as ``project_out`` and analyses such as ``direction_projection``
accept; it is parsed from its listing descriptor too (``wse.sandbox``).

The call is recorded in the audit log (``sandbox.ran``) with its code hash,
ending, times, and output files (each ``.safetensors`` output with the
tensor file it was stored as, or ``None``), and it is idempotent by its
key: a retry with the same code returns the first result without running
again.
"""

from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import math
import os
import stat
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import numpy as np

from wse import sandbox
from wse.database import insert_audit_event, utc_now
from wse.policy import ExperimentPolicy, sandbox_available
from wse.sandbox import SandboxLayout
from wse.store import TENSOR_URI_PREFIX
from wse.schemas import (
    IDEMPOTENCY_KEY_PATTERN,
    Experiment,
    ErrorCode,
    SandboxOutput,
    SandboxRun,
    SandboxTensor,
    Usage,
    WSEError,
    canonical_json,
    json_hash,
)

if TYPE_CHECKING:
    from wse.experiments import ExperimentService

__all__ = ["MAX_CODE_BYTES", "MAX_EXPERIMENT_TENSOR_BYTES", "MAX_LISTED_OUTPUTS", "MAX_REFERENCE_ELEMENTS",
           "layout_for", "readable_tensors", "refresh_inputs", "run_python", "store_agent_tensor_file",
           "sync_inputs"]

OPERATION = "run_python"
MAX_CODE_BYTES = 256 * 1024
MAX_REFERENCE_ELEMENTS = 65536
MAX_ARRAY_FILE_BYTES = 64 * 1024 * 1024
MAX_LISTED_OUTPUTS = 32  # output files listed and stored per call; calls/<n>/result.json lists every one
MAX_TENSOR_FILE_BYTES = 8 << 30
MAX_EXPERIMENT_TENSOR_BYTES = 32 << 30  # tensor-file bytes stored per experiment, by sandbox calls and paths alike
MAX_SAFETENSORS_HEADER_BYTES = 100 << 20
MAX_LISTED_TENSORS = 16
SAFETENSORS_FLOAT_DTYPES = {"F16": "float16", "BF16": "bfloat16", "F32": "float32", "F64": "float64"}
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


def refresh_inputs(service: ExperimentService, experiment_id: str) -> list[str] | None:
    """Bring the run's sandbox inputs up to date (``sync_inputs``) when the policy grants the sandbox.

    The agent's own code, which runs outside WSE (for example the Codex
    CLI's commands), reads the same ``inputs/`` that ``wse_run_python``
    calls do: the runner exports them before the agent starts, and the MCP
    server again whenever a job succeeds. ``None`` when the policy grants no
    sandbox.
    """
    policy = service.experiment_policy(experiment_id)
    if not sandbox_available(policy):
        return None
    experiment = service.get_experiment(experiment_id)
    return sync_inputs(service, experiment, policy, layout_for(service, experiment.run_id).create())


def _load_array(descriptor: int, output: sandbox.OutputFile) -> np.ndarray | None:
    """The array in the ``.npy`` file open at ``descriptor`` if it is at most ``MAX_ARRAY_FILE_BYTES`` with its hash."""
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_ARRAY_FILE_BYTES:
        return None
    digest, _, data = sandbox.read_descriptor(descriptor, MAX_ARRAY_FILE_BYTES)
    if data is None or digest != output.sha256:
        return None
    try:
        return np.load(io.BytesIO(data), allow_pickle=False)
    except (ValueError, OSError, EOFError):
        return None


def _safetensors_header(handle: Any) -> dict[str, Any]:
    """The header of a safetensors file open at its start (8-byte little-endian length, then JSON); raise ValueError."""
    prefix = handle.read(8)
    if len(prefix) != 8:
        raise ValueError("the file is shorter than a safetensors header")
    length = int.from_bytes(prefix, "little")
    if not 2 <= length <= MAX_SAFETENSORS_HEADER_BYTES:
        raise ValueError(f"its header length {length} is out of range")
    header = json.loads(handle.read(length))
    if not isinstance(header, dict):
        raise ValueError("its header is not a JSON object")
    return header


def _open_beneath(root: Path, relative: PurePosixPath) -> int:
    """Open the regular file ``root / relative`` for reading: no level below ``root`` may be a link (``open_beneath``).

    Every level is opened without following a link and the file without
    waiting on a FIFO, so a FIFO cannot block the server. Raises ``OSError``
    (``wse.sandbox.NotRegularFile`` for an entry that is not a regular file)
    or ``ValueError`` (a part that cannot name an entry, such as one holding
    a NUL byte).
    """
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        descriptor, _ = sandbox.open_beneath(directory, relative.parts)
    finally:
        os.close(directory)
    return descriptor


def _file_sha256(handle: Any) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1 << 22), b""):
        digest.update(block)
    handle.seek(0)
    return "sha256:" + digest.hexdigest()


class _NotStored(Exception):
    """Why a file is not stored as a tensor file: the ``SandboxOutput.note`` it gets."""


def _header_tensors(path: Path) -> list[SandboxTensor]:
    """The tensors that the safetensors file at ``path`` lists; raise ``_NotStored`` unless all are floating point."""
    try:
        with path.open("rb") as handle:
            header = _safetensors_header(handle)
        tensors = []
        for name, entry in sorted(header.items()):
            if name == "__metadata__":
                continue
            dtype = entry.get("dtype") if isinstance(entry, dict) else None
            shape = entry.get("shape") if isinstance(entry, dict) else None
            if dtype not in SAFETENSORS_FLOAT_DTYPES or not isinstance(shape, list):
                raise _NotStored(f"not stored: tensor {name!r} has dtype {dtype}; tensor files hold floating-point "
                                 f"tensors ({sorted(SAFETENSORS_FLOAT_DTYPES.values())})")
            tensors.append(SandboxTensor(name=name, shape=[int(size) for size in shape],
                                         dtype=SAFETENSORS_FLOAT_DTYPES[dtype]))
    except (ValueError, TypeError) as exc:
        raise _NotStored(f"not stored: not a safetensors file ({exc})") from None
    if not tensors:
        raise _NotStored("not stored: the file holds no tensors")
    return tensors


def _over_the_limit(service: ExperimentService, experiment_id: str, uri: str, size: int) -> str | None:
    """Why storing ``size`` more bytes as ``uri`` would pass ``MAX_EXPERIMENT_TENSOR_BYTES``, or ``None``."""
    conn = service.db.conn
    if conn.execute("SELECT 1 FROM tensor_files WHERE uri = ? AND experiment_id = ?", (uri, experiment_id)).fetchone():
        return None  # already stored for this experiment: no new bytes
    stored = int(conn.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM tensor_files WHERE experiment_id = ?",
                              (experiment_id,)).fetchone()[0])
    if stored + size <= MAX_EXPERIMENT_TENSOR_BYTES:
        return None
    return (f"not stored: this experiment's tensor files hold {stored} bytes, and this file's {size} bytes would "
            f"pass the limit of {MAX_EXPERIMENT_TENSOR_BYTES} bytes ({MAX_EXPERIMENT_TENSOR_BYTES / 2**30:g} GiB) "
            "of tensor files per experiment")


def _store_tensor_file(service: ExperimentService, experiment_id: str, descriptor: int, label: str, call: int,
                       sha256: str | None = None) -> dict[str, Any]:
    """Store the ``.safetensors`` file open at ``descriptor`` as a tensor file; return ``SandboxOutput`` fields.

    The file must be regular, singly linked, at most ``MAX_TENSOR_FILE_BYTES``,
    and within the experiment's ``MAX_EXPERIMENT_TENSOR_BYTES``, all checked
    before anything is copied. It is copied from this descriptor only (which
    the caller closes); the header is parsed from the store's copy after its
    hash check, and a file whose copy fails is never published. ``label`` is
    its path (in ``outputs/`` for a call's output, in the run's directory for
    a path) and ``call`` the sandbox call that listed it (0: stored from a
    path). ``sha256``, when given, is the hash the call listed.
    """
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        return {"note": "not stored: not a regular file with a single link"}
    if info.st_size > MAX_TENSOR_FILE_BYTES:
        return {"note": f"not stored: tensor files hold at most {MAX_TENSOR_FILE_BYTES} bytes"}
    found: dict[str, list[SandboxTensor]] = {}

    def check(copy: Path) -> None:
        found["tensors"] = _header_tensors(copy)

    with os.fdopen(os.dup(descriptor), "rb") as handle:
        handle.seek(0)
        digest = sha256 if sha256 is not None else _file_sha256(handle)
        uri = TENSOR_URI_PREFIX + digest.removeprefix("sha256:")
        refusal = _over_the_limit(service, experiment_id, uri, info.st_size)
        if refusal is not None:
            return {"note": refusal}
        try:
            uri = service.store.publish_tensor_file(handle, digest.removeprefix("sha256:"), check=check)
        except _NotStored as exc:
            return {"note": str(exc)}
        except WSEError as exc:
            if exc.code == ErrorCode.ARTIFACT_CORRUPT:  # the copy did not have the hash
                return {"note": "not stored: the file changed after the call listed it" if sha256 is not None
                        else "not stored: the file changed while WSE copied it"}
            return {"note": f"not stored: {exc.message}"}
    tensors = found["tensors"]
    with service.db.transaction() as conn:
        conn.execute("INSERT OR IGNORE INTO tensor_files (uri, experiment_id, call, path, sha256, size_bytes, "
                     "tensors_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     (uri, experiment_id, call, label, digest, info.st_size,
                      json.dumps([t.model_dump(mode="json") for t in tensors]), utc_now()))
    return {"tensor_file": uri, "tensors": tensors[:MAX_LISTED_TENSORS], "tensor_count": len(tensors)}


def store_agent_tensor_file(service: ExperimentService, experiment_id: str, path: Any) -> str:
    """Store a ``.safetensors`` file under ``outputs/`` or ``work/`` of the run's directory; return its URI.

    ``path`` is relative to the run's sandbox directory. No level of it may
    be a link and the file must be a regular file (``_open_beneath``), so a
    file outside the directory can never be reached through a link and a
    FIFO cannot block the server; the checks of ``_store_tensor_file`` apply.
    A path that cannot name a file (a NUL byte, say) is ``INVALID_ARGUMENT``.
    """
    relative = PurePosixPath(path) if isinstance(path, str) and "\x00" not in path else None
    if (relative is None or relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2
            or relative.parts[0] not in ("outputs", "work") or relative.suffix != ".safetensors"):
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "path must name a .safetensors file under outputs/ or work/ of "
                       "the run's directory, for example outputs/candidate.safetensors",
                       details={"path": str(path)})
    experiment = service.get_experiment(experiment_id)
    layout = layout_for(service, experiment.run_id).create()
    try:
        descriptor = _open_beneath(layout.root, relative)
    except sandbox.NotRegularFile as exc:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"{relative} is not a regular file: {exc.strerror}",
                       details={"path": str(relative)}) from exc
    except ValueError as exc:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"path cannot name a file of the run's directory: {exc}",
                       details={"path": str(relative)}) from exc
    except OSError as exc:
        raise WSEError(ErrorCode.NOT_FOUND, f"{relative} cannot be opened as a file of the run's directory without "
                       f"following a link: {exc.strerror}", details={"path": str(relative)}) from exc
    try:
        fields = _store_tensor_file(service, experiment_id, descriptor, str(relative), 0)
    finally:
        os.close(descriptor)
    if "note" in fields:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"{relative} was {fields['note']}", details={"path": str(relative)})
    return fields["tensor_file"]


def _referenceable(array: np.ndarray) -> bool:
    return (array.ndim in (1, 2) and 0 < array.size <= MAX_REFERENCE_ELEMENTS
            and np.issubdtype(array.dtype, np.number) and not np.issubdtype(array.dtype, np.complexfloating)
            and bool(np.isfinite(array).all()))


def _array_fields(service: ExperimentService, experiment_id: str, call: int, output: sandbox.OutputFile,
                  descriptor: int) -> dict[str, Any]:
    """The ``SandboxOutput`` fields of a ``.npy`` output open at ``descriptor``: its shape, dtype, and pointer."""
    array = _load_array(descriptor, output)
    if array is None:
        return {}
    shape, dtype, pointer = list(array.shape), str(array.dtype), None
    if _referenceable(array):
        document = {"kind": ARRAY_KIND, "experiment_id": experiment_id, "call": call, "path": output.path,
                    "sha256": output.sha256, "shape": shape, "dtype": dtype,
                    "values": array.astype(np.float64).tolist()}
        uri = service.store.put_analysis_result(canonical_json(document).encode("utf-8"))
        with service.db.transaction() as conn:
            conn.execute("INSERT OR IGNORE INTO analysis_results (uri, experiment_id, created_at) VALUES (?, ?, ?)",
                         (uri, experiment_id, utc_now()))
        pointer = f"{uri}#/values"
    return {"shape": shape, "dtype": dtype, "pointer": pointer}


def _output_record(output: SandboxOutput) -> dict[str, Any]:
    """An output as the ``sandbox.ran`` event records it; a ``.safetensors`` output names its tensor file or ``None``."""
    record: dict[str, Any] = {"path": output.path, "size": output.size, "sha256": output.sha256}
    if output.path.endswith(".safetensors"):
        record["tensor_file"] = output.tensor_file
    return record


def _fitted(run: SandboxRun, directory: Path, budget: int, limit: int) -> SandboxRun:
    """``run`` with its streams shortened, from their stored bytes, to the largest bound whose response fits."""
    from wse.policy import response_size

    if response_size(run.model_dump(mode="json")) <= budget:
        return run
    stored = {name: (directory / f"{name}.txt").read_bytes() for name in ("stdout", "stderr")}

    def cut(bound: int) -> SandboxRun:
        stdout, stdout_cut = sandbox.shorten(stored["stdout"], run.stdout_bytes, bound)
        stderr, stderr_cut = sandbox.shorten(stored["stderr"], run.stderr_bytes, bound)
        return run.model_copy(update={"stdout": stdout, "stderr": stderr,
                                      "stdout_truncated": run.stdout_truncated or stdout_cut,
                                      "stderr_truncated": run.stderr_truncated or stderr_cut})

    low, high = 0, limit
    while low < high:  # the largest bound whose response fits
        middle = (low + high + 1) // 2
        if response_size(cut(middle).model_dump(mode="json")) <= budget:
            low = middle
        else:
            high = middle - 1
    return cut(low)


def run_python(service: ExperimentService, experiment_id: str, code: Any, key: Any,
               restart: Any = False) -> SandboxRun:
    """Run ``code`` in the run's kernel, charge it, record it, and return its result (module docstring)."""
    if not isinstance(key, str) or not IDEMPOTENCY_KEY_PATTERN.fullmatch(key):
        raise WSEError(ErrorCode.INVALID_ARGUMENT,
                       f"idempotency_key must match {IDEMPOTENCY_KEY_PATTERN.pattern}; reuse the same key to retry "
                       "the same request")
    if not isinstance(code, str) or not code.strip():
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "code must be a non-empty string of Python")
    if len(code.encode("utf-8")) > MAX_CODE_BYTES:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"code has more than {MAX_CODE_BYTES} bytes",
                       details={"max_bytes": MAX_CODE_BYTES})
    if not isinstance(restart, bool):
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "restart must be true or false")
    row = service._experiment_row(experiment_id)
    policy = service.experiment_policy(experiment_id)
    if not sandbox_available(policy):
        raise WSEError(ErrorCode.POLICY_DENIED, f"policy {policy.policy_id} grants no code execution; use the "
                       "analysis and probe tools instead", details={"policy_id": policy.policy_id})
    request_hash = json_hash({"code": code, "restart": restart})
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
    lifetime_cpu = max(1, math.ceil(float(policy.budget.limit("sandbox_seconds")) * (os.cpu_count() or 1)))
    fields: dict[str, dict[str, Any]] = {}

    def use(call: int, output: sandbox.OutputFile, descriptor: int) -> None:
        # The descriptor the listing hashed the file from: a link, or a file swapped in since, is never read.
        if output.path.endswith(".safetensors"):
            fields[output.path] = _store_tensor_file(service, experiment_id, descriptor, output.path, call,
                                                     output.sha256)
        elif output.path.endswith(".npy"):
            fields[output.path] = _array_fields(service, experiment_id, call, output, descriptor)

    result = sandbox.run_python(layout, code, wall_seconds=wall, cpu_seconds=lifetime_cpu,
                                max_output_bytes=policy.sandbox.max_output_bytes,
                                max_memory_bytes=policy.sandbox.max_memory_bytes, restart=restart,
                                use_output=use, use_first=MAX_LISTED_OUTPUTS)
    seconds = round(min(result.wall_seconds, wall), 3)
    outputs = [SandboxOutput(path=o.path, size=o.size, sha256=o.sha256, **fields.get(o.path, {}))
               for o in result.outputs[:MAX_LISTED_OUTPUTS]]
    run = SandboxRun(
        call=result.call, ending=result.ending, exit_code=result.exit_code, signal=result.signal, error=result.error,
        wall_seconds=result.wall_seconds, cpu_seconds=result.cpu_seconds,
        peak_memory_bytes=result.peak_memory_bytes, stdout=result.stdout, stderr=result.stderr,
        stdout_bytes=result.stdout_bytes, stderr_bytes=result.stderr_bytes, stdout_truncated=result.stdout_truncated,
        stderr_truncated=result.stderr_truncated, outputs=outputs, refused_outputs=list(result.refused_outputs),
        inputs=inputs, code_sha256=result.code_sha256, usage=Usage(sandbox_calls=1, sandbox_seconds=seconds),
        session=result.session, session_call=result.session_call, restarted=result.restarted,
        state_kept=result.state_kept, outputs_not_listed=max(0, len(result.outputs) - MAX_LISTED_OUTPUTS) or None,
        repairs=list(result.repairs) or None)
    run = _fitted(run, layout.calls / str(result.call), 2 * policy.operations.limits.max_result_json_bytes,
                  policy.sandbox.max_output_bytes)
    with service.db.transaction() as conn:
        service.ledger.charge(conn, experiment_id, "sandbox_calls", 1)
        service.ledger.charge(conn, experiment_id, "sandbox_seconds", seconds)
        service._remember(conn, experiment_id, key, OPERATION, request_hash, run.model_dump_json())
        insert_audit_event(conn, "sandbox.ran", experiment_id, {
            "call": run.call, "code_sha256": run.code_sha256, "ending": run.ending, "exit_code": run.exit_code,
            "signal": run.signal, "error": run.error, "wall_seconds": run.wall_seconds,
            "cpu_seconds": run.cpu_seconds, "peak_memory_bytes": run.peak_memory_bytes, "session": run.session,
            "session_call": run.session_call, "restarted": run.restarted, "state_kept": run.state_kept,
            "outputs": [_output_record(o) for o in outputs],
            "refused_outputs": run.refused_outputs, "charged": {"sandbox_calls": 1, "sandbox_seconds": seconds},
            **({"repairs": run.repairs} if run.repairs else {}),
        }, experiment_id=experiment_id)
    return run
