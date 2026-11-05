"""Direct library functions and the managed ``Engine`` facade.

Direct use (``load_model``, ``apply_operator``, ``evaluate_candidate``) works
in memory: it needs no service, database, or MCP, and its handles and results
carry no experiment IDs. Results of direct use are labeled ``unmanaged`` and
cannot support managed-enforcement claims (a verified no-forward or
fixed-budget setting); managed reports are labeled ``managed``.

The ``Engine`` wraps the same implementations with persistence: it imports
models into the content-addressed store, assigns checkpoint IDs, and serves
only READY checkpoints. Storage modules are imported lazily so that direct use
never loads SQLite.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch
from pydantic import ValidationError

from wse.adapters.base import Architecture, ModelAdapter
from wse.registry import Registry, builtin_registry
from wse.schemas import (
    BundleManifest,
    ClusterSummary,
    ErrorCode,
    EvalConfig,
    EvaluationSuite,
    ExecutionConfig,
    ExtensionRef,
    ImportManifest,
    IntervalEstimate,
    Management,
    MaterializedPatch,
    ModelIdentity,
    ModelManifest,
    TensorDiff,
    Usage,
    WSEError,
    canonical_json,
    json_hash,
)
from wse.tensors import normalize_tensors, weights_hash

if TYPE_CHECKING:
    from wse.adapters.base import AdapterCapabilities
    from wse.budget import BudgetLedger, UsageLike
    from wse.experiments import ExperimentService
    from wse.jobs import JobService
    from wse.policy import ExperimentPolicy, OperationPolicy
    from wse.evaluation import EvaluationService
    from wse.evaluators.datasets import DatasetStore
    from wse.schemas import (
        AnalysisResult,
        ComparisonReport,
        DatasetManifest,
        EvaluationReport,
        Experiment,
        ExperimentCheckpoint,
        Job,
        PatchArtifact,
        PatchPreview,
        PatchSpec,
        FinalEvaluationReport,
        Reservation,
        RunGrant,
        SuiteData,
        TaskDefinition,
        TensorSlice,
    )
    from wse.store import ArtifactStore, RecoveryReport

DEFAULT_MAX_MODEL_BYTES = 64 * 1024 * 1024
DEFAULT_EVAL_BATCH_SIZE = 64
MAX_CONFIG_BYTES = 1024 * 1024
BUNDLE_MANIFEST_NAME = "wse_bundle.json"


# ---------------------------------------------------------------------------
# Core handles and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ModelHandle:
    """An in-memory model with content identity and adapter metadata.

    The handle owns private tensor copies; accessors return copies, so callers
    cannot change the weights behind the recorded hash.
    """

    adapter_ref: ExtensionRef
    config_json: str
    config_hash: str
    weights_hash: str
    architecture: Architecture
    adapter: ModelAdapter = field(repr=False)
    _tensors: Mapping[str, torch.Tensor] = field(repr=False)

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(
            adapter=self.adapter_ref, config_hash=self.config_hash, weights_hash=self.weights_hash
        )

    @property
    def config(self) -> dict[str, Any]:
        return json.loads(self.config_json)

    @property
    def parameter_count(self) -> int:
        return self.architecture.parameter_count

    def tensor_names(self) -> tuple[str, ...]:
        return tuple(info.name for info in self.architecture.tensors)

    def tensor(self, name: str) -> torch.Tensor:
        if name not in self._tensors:
            raise WSEError(
                ErrorCode.NOT_FOUND,
                f"tensor {name!r} does not exist; tensor names must match exactly",
                details={"available_tensors": list(self.tensor_names())},
            )
        return self._tensors[name].clone()

    def tensors(self) -> dict[str, torch.Tensor]:
        return {name: self._tensors[name].clone() for name in self.tensor_names()}

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Unmanaged inference through the adapter (direct library use only)."""
        return self.adapter.forward(self._tensors, inputs)


@dataclass(frozen=True)
class EditResult:
    """Outcome of a direct edit: the immutable candidate, exact patch, and diff."""

    candidate: ModelHandle
    patch: MaterializedPatch
    diff: tuple[TensorDiff, ...]
    management: Management = "unmanaged"


@dataclass(frozen=True)
class EvaluationResult:
    """Outcome of an evaluation, bound to its model, data, and configuration identities.

    ``evaluator_config_hash`` is the hash of ``execution`` (evaluator,
    parameters, ``eval_config``, batch size, device). ``scores`` holds one
    score per example (1 = correct) when the evaluator reports them, with
    ``cluster_ids`` the cluster of every example; ``intervals`` holds the
    cluster-bootstrap accuracy interval and the Clopper-Pearson interval on
    fully correct clusters. The managed worker persists this result as an
    ``EvaluationReport``; direct results stay ``unmanaged``.
    """

    metrics: Mapping[str, float]
    sample_count: int
    model: ModelIdentity
    evaluator: ExtensionRef
    evaluator_config_hash: str
    data_hash: str
    usage: Usage
    split: str
    suite_id: str
    execution: ExecutionConfig
    clusters: ClusterSummary | None
    intervals: Mapping[str, IntervalEstimate]
    scores: tuple[int, ...] | None
    cluster_ids: tuple[int, ...] | None
    errors: tuple[str, ...]
    truncation_count: int
    elapsed_seconds: float
    management: Management = "unmanaged"


def _resolve_adapter(registry: Registry, adapter: str | ExtensionRef) -> tuple[ExtensionRef, ModelAdapter]:
    """Resolve an exact adapter reference or a declared alias to its exact version."""
    ref = registry.resolve_reference("adapter", adapter)
    return ref, registry.resolve("adapter", ref.extension_id, ref.version)


def _with_resolved_adapter(value: Any, registry: Registry) -> Any:
    """Replace a version-less adapter alias in a raw manifest with its exact reference."""
    if isinstance(value, Mapping) and isinstance(value.get("adapter"), str) and "@" not in value["adapter"]:
        return {**value, "adapter": str(registry.resolve_reference("adapter", value["adapter"]))}
    return value


def canonical_config(adapter: ModelAdapter, config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the adapter-normalized configuration used for model identity.

    Adapters may implement ``canonical_config(config)`` to fill defaults so
    that configurations describing the same architecture hash identically.
    Adapters without it are hashed over the configuration as supplied.
    """
    normalize = getattr(adapter, "canonical_config", None)
    normalized = normalize(config) if callable(normalize) else config
    if not isinstance(normalized, Mapping):
        raise WSEError(ErrorCode.INVALID_MODEL, "adapter canonical_config() must return a JSON object")
    return json.loads(canonical_json(dict(normalized)))


def _build_handle(
    adapter_ref: ExtensionRef,
    adapter: ModelAdapter,
    manifest: ImportManifest | ModelManifest,
    tensors: dict[str, torch.Tensor],
) -> ModelHandle:
    architecture = adapter.validate(manifest, tensors)
    ordered = {info.name: tensors[info.name] for info in architecture.tensors}
    config = canonical_config(adapter, manifest.config)
    return ModelHandle(
        adapter_ref=adapter_ref,
        config_json=canonical_json(config),
        config_hash=json_hash(config),
        weights_hash=weights_hash(ordered),
        architecture=architecture,
        adapter=adapter,
        _tensors=MappingProxyType(ordered),
    )


def _read_json_file(path: Path, *, what: str) -> Any:
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise WSEError(ErrorCode.INVALID_MODEL, f"{what} {path.name} exceeds {MAX_CONFIG_BYTES} bytes")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WSEError(ErrorCode.INVALID_MODEL, f"{what} {path.name} is not valid JSON: {exc}") from exc


def _load_safetensors(path: Path, max_bytes: int) -> dict[str, torch.Tensor]:
    from safetensors import SafetensorError
    from safetensors.torch import load_file

    size = path.stat().st_size
    if size > max_bytes:
        raise WSEError(
            ErrorCode.INVALID_MODEL,
            f"weights file {path.name} is {size} bytes; the limit is {max_bytes} bytes",
            details={"size_bytes": size, "max_bytes": max_bytes},
        )
    try:
        return load_file(str(path))
    except (SafetensorError, OSError, ValueError) as exc:
        raise WSEError(ErrorCode.INVALID_MODEL, f"cannot read safetensors file {path.name}: {exc}") from exc


def _as_config(config: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    if config is None:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "a model configuration (mapping or JSON path) is required")
    if isinstance(config, (str, Path)):
        config = _read_json_file(Path(config), what="configuration")
    if not isinstance(config, Mapping):
        raise WSEError(ErrorCode.INVALID_MODEL, "a model configuration must be a JSON object")
    return dict(config)


def load_model(
    source: Mapping[str, Any] | str | Path,
    adapter: str | ExtensionRef,
    config: Mapping[str, Any] | str | Path | None = None,
    *,
    registry: Registry | None = None,
    max_bytes: int = DEFAULT_MAX_MODEL_BYTES,
) -> ModelHandle:
    """Load a model in memory through a registered adapter (direct use).

    ``source`` is a mapping of tensor name to ``torch.Tensor``/``numpy.ndarray``
    or the path of a ``.safetensors`` file. ``adapter`` is an exact reference
    such as ``"mlp@1"`` or a version-less alias declared in the registry
    (``"mlp"`` means ``mlp@1`` in the built-in registry); an alias never
    resolves to the latest registered version, and the handle records the
    resolved exact reference. Unknown adapters, aliases, and versions are
    rejected.
    ``config`` is the adapter configuration or the path of a JSON file.
    Requires neither SQLite nor MCP.
    """
    registry = registry or builtin_registry()
    adapter_ref, implementation = _resolve_adapter(registry, adapter)
    if isinstance(source, (str, Path)):
        tensors: Mapping[str, Any] = _load_safetensors(Path(source), max_bytes)
    else:
        tensors = source
    manifest = _import_manifest({"adapter": adapter_ref, "config": _as_config(config)})
    return _build_handle(adapter_ref, implementation, manifest, normalize_tensors(tensors))


def apply_operator(
    model: ModelHandle,
    operator_id: str | ExtensionRef,
    parameters: Mapping[str, Any],
    *,
    registry: Registry | None = None,
    policy: OperationPolicy | None = None,
) -> EditResult:
    """Apply one registered edit operator in memory (direct use, unmanaged).

    ``operator_id`` is an exact reference (``"scale_slice@1"``) or a public
    alias (``"scale_slice"``); ``parameters`` holds ``tensor``, ``selection``,
    and the operator's own parameters. Returns the immutable candidate, the
    materialized patch, and the per-tensor diff; the input model is unchanged.
    """
    from wse.patches import apply_operator as _apply_operator

    return _apply_operator(model, operator_id, parameters, registry=registry, policy=policy)


class _CountingForward:
    """Unmetered forward for direct evaluation: counts calls and examples, enforces the batch size."""

    def __init__(self, model: ModelHandle, batch_size: int) -> None:
        self.model = model
        self.batch_size = batch_size
        self.started_calls = 0
        self.started_examples = 0

    def check(self) -> None:
        return None

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor) or inputs.dim() < 1 or not 1 <= inputs.shape[0] <= self.batch_size:
            raise WSEError(ErrorCode.INVALID_ARGUMENT, f"a forward batch must have between 1 and {self.batch_size} rows")
        self.started_calls += 1
        self.started_examples += int(inputs.shape[0])
        with torch.inference_mode():
            return self.model.forward(inputs)


def _as_suite(evaluator: EvaluationSuite | Mapping[str, Any]) -> EvaluationSuite:
    if isinstance(evaluator, EvaluationSuite):
        return evaluator
    try:
        return EvaluationSuite.model_validate(dict(evaluator))
    except (ValidationError, TypeError, ValueError) as exc:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"invalid evaluation suite: {exc}") from exc


def _as_eval_config(config: EvalConfig | Mapping[str, Any]) -> EvalConfig:
    if isinstance(config, EvalConfig):
        return config
    try:
        return EvalConfig.model_validate(config)
    except (ValidationError, TypeError, ValueError) as exc:
        raise WSEError(ErrorCode.INVALID_ARGUMENT,
                       "eval_config must be an object with exactly sample_count (integer) and seed (integer)") from exc


def _parse_summary(summary: Any, sample_count: int) -> tuple[dict[str, float], list[str], list[int] | None, int]:
    """Validate an evaluator's summary: metrics, errors, optional per-example scores, truncation count."""
    import math

    def bad(reason: str) -> WSEError:
        return WSEError(ErrorCode.WORKER_FAILED, f"the evaluator returned an invalid summary: {reason}")

    if not isinstance(summary, Mapping) or not isinstance(summary.get("metrics"), Mapping):
        raise bad("it must be an object with a metrics object")
    metrics: dict[str, float] = {}
    for name, value in summary["metrics"].items():
        if not isinstance(name, str) or isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value):
            raise bad("metrics must map names to finite numbers")
        metrics[name] = float(value)
    if summary.get("sample_count", sample_count) != sample_count:
        raise bad(f"it covers {summary.get('sample_count')!r} examples, not the requested {sample_count}")
    errors = summary.get("errors", [])
    if not isinstance(errors, list) or not all(isinstance(error, str) for error in errors):
        raise bad("errors must be a list of strings")
    scores = summary.get("scores")
    if scores is not None:
        if not isinstance(scores, list) or len(scores) != sample_count \
                or any(isinstance(s, bool) or s not in (0, 1) for s in scores):
            raise bad("scores must hold one 0 or 1 per evaluated example")
        scores = [int(s) for s in scores]
    truncation = summary.get("truncation_count", 0)
    if isinstance(truncation, bool) or not isinstance(truncation, int) or truncation < 0:
        raise bad("truncation_count must be a nonnegative integer")
    return metrics, list(errors), scores, truncation


def evaluate_candidate(
    model: ModelHandle,
    evaluator: EvaluationSuite | Mapping[str, Any],
    config: EvalConfig | Mapping[str, Any],
    *,
    registry: Registry | None = None,
    forward: Any = None,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
) -> EvaluationResult:
    """Evaluate a model on a suite's examples (direct use, unmanaged).

    ``evaluator`` is an ``EvaluationSuite`` with its examples attached: it
    names the exact registered evaluator, its parameters, and the data.
    ``config`` is the ``eval_config`` (``sample_count`` examples from the
    start of the suite, ending at a cluster boundary, and a ``seed``).
    Batches hold at most ``batch_size`` rows; the managed server fixes it to
    the policy's ``examples_per_batch`` (64 under ``small_model_v1``), which
    is also the default here, so the configuration identities agree.

    The managed worker calls this same function with ``forward`` set to its
    metered forward callable and persists the result as a report. Requires
    neither SQLite, a job service, nor MCP.
    """
    from wse.evaluators.classification import accuracy_interval, fully_correct_interval

    suite = _as_suite(evaluator)
    eval_config = _as_eval_config(config)
    if suite.data is None:
        raise WSEError(ErrorCode.INVALID_ARGUMENT,
                       f"suite {suite.suite_id} has no examples attached; direct evaluation needs inline data")
    count = eval_config.sample_count
    if not 1 <= count <= suite.data.example_count:
        raise WSEError(ErrorCode.INVALID_ARGUMENT,
                       f"sample_count must lie between 1 and the {suite.data.example_count} examples of the suite")
    if suite.data.cuts_clusters(count):
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"sample_count {count} would split a {suite.data.cluster_unit}")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "batch_size must be a positive integer")
    registry = registry or builtin_registry()
    implementation = registry.resolve_ref("evaluator", suite.evaluator)
    meter = forward if forward is not None else _CountingForward(model, batch_size)
    start = time.perf_counter()
    summary = implementation.evaluate(meter, suite, eval_config, batch_size=batch_size)
    elapsed = time.perf_counter() - start
    metrics, errors, scores, truncation = _parse_summary(summary, count)
    execution = ExecutionConfig(evaluator=suite.evaluator, parameters=suite.parameters, eval_config=eval_config,
                                batch_size=batch_size)
    cluster_ids = suite.data.cluster_ids()[:count]
    unit = suite.data.cluster_unit or "example"
    clusters, intervals = None, {}
    if scores is not None:
        fully = fully_correct_interval(scores, cluster_ids, unit)
        clusters = ClusterSummary(unit=unit, count=fully.trials, fully_correct=fully.successes)
        intervals = {"accuracy": accuracy_interval(scores, cluster_ids, unit), "fully_correct_clusters": fully}
    calls, examples = int(meter.started_calls), int(meter.started_examples)
    usage = Usage(evaluation_batches=calls, evaluated_examples=examples, model_forward_calls=calls,
                  compute_seconds=round(elapsed, 6), analysis_calls=0, tensor_read_calls=0, candidate_commits=0,
                  artifact_bytes=0, peak_memory_bytes=None,
                  unavailable={"peak_memory_bytes": "measured only for whole evaluation processes"})
    return EvaluationResult(
        metrics=MappingProxyType(metrics),
        sample_count=count,
        model=model.identity,
        evaluator=suite.evaluator,
        evaluator_config_hash=execution.config_hash,
        data_hash=suite.data.data_hash,
        usage=usage,
        split=suite.split,
        suite_id=suite.suite_id,
        execution=execution,
        clusters=clusters,
        intervals=MappingProxyType(intervals),
        scores=tuple(scores) if scores is not None else None,
        cluster_ids=tuple(cluster_ids) if scores is not None else None,
        errors=tuple(errors),
        truncation_count=truncation,
        elapsed_seconds=round(elapsed, 6),
    )


def _import_manifest(value: ImportManifest | Mapping[str, Any]) -> ImportManifest:
    if isinstance(value, ImportManifest):
        return value
    try:
        return ImportManifest.model_validate(dict(value))
    except ValidationError as exc:
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"invalid import manifest: {exc}",
            details={"errors": json.loads(exc.json(include_url=False, include_input=False))},
        ) from exc


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------


def _bundle_member(root: Path, relative: str, *, what: str) -> Path:
    """Resolve a bundle-relative path, rejecting any escape from the bundle."""
    pure = PurePosixPath(relative)
    if (
        "\\" in relative
        or "\x00" in relative
        or pure.is_absolute()
        or any(part in ("..", "") for part in relative.split("/"))
        or relative.startswith("~")
    ):
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"{what} path {relative!r} must be a plain relative path inside the bundle directory",
            details={"path": relative},
        )
    candidate = root / pure
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, NotADirectoryError):
        raise WSEError(ErrorCode.NOT_FOUND, f"{what} file {relative!r} does not exist in the bundle") from None
    if not resolved.is_relative_to(root):
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"{what} path {relative!r} resolves outside the bundle directory",
            details={"path": relative},
        )
    if not resolved.is_file():
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"{what} path {relative!r} is not a regular file")
    return resolved


def read_bundle(
    path: str | Path, *, max_bytes: int = DEFAULT_MAX_MODEL_BYTES, registry: Registry | None = None
) -> tuple[ImportManifest, dict[str, torch.Tensor]]:
    """Read a bundle directory (or its ``wse_bundle.json``) into an import request.

    The bundle's filesystem location is never recorded in the manifest;
    provenance comes only from the bundle file's operator-supplied labels. A
    version-less adapter alias is resolved through ``registry`` (the built-in
    registry by default) and recorded as its exact reference.
    """
    path = Path(path)
    if not path.exists():
        raise WSEError(
            ErrorCode.NOT_FOUND,
            f"bundle {path} does not exist; pass a directory containing {BUNDLE_MANIFEST_NAME}",
        )
    if path.is_dir():
        path = path / BUNDLE_MANIFEST_NAME
    try:
        root = path.parent.resolve(strict=True)
    except FileNotFoundError:
        raise WSEError(ErrorCode.NOT_FOUND, f"bundle directory {path.parent} does not exist") from None
    bundle_file = _bundle_member(root, path.name, what="bundle manifest")
    try:
        raw = _with_resolved_adapter(_read_json_file(bundle_file, what="bundle manifest"),
                                     registry or builtin_registry())
        bundle = BundleManifest.model_validate(raw)
    except ValidationError as exc:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"invalid {BUNDLE_MANIFEST_NAME}: {exc}") from exc
    config = _as_config(_bundle_member(root, bundle.config, what="config"))
    tensors = _load_safetensors(_bundle_member(root, bundle.weights, what="weights"), max_bytes)
    return _import_manifest(bundle.to_import_manifest(config)), tensors


# ---------------------------------------------------------------------------
# Managed engine
# ---------------------------------------------------------------------------


class Engine:
    """Managed facade over the artifact store, registry, and experiments.

    Opening an engine applies database migrations and runs the startup
    recovery scan. ``run_id`` binds the engine to one persisted run grant (as
    for a server process started for one agent trial). A bound engine, and
    every service it uses, can create and see only that run's experiment and
    can see only the grant's base checkpoint and the candidates created in
    that experiment: listings omit every other checkpoint, and any other
    checkpoint ID fails with the same ``NOT_FOUND`` error as an ID that does
    not exist. Operator actions (imports, policy, task, dataset, and suite
    registration, run grants, direct reservations and settlements, freezing,
    and final evaluation) fail with ``POLICY_DENIED``. Its evaluator storage
    is sealed, so no agent-facing operation can read evaluation examples, and
    final reports stay withheld until the experiment is closed. Opening an
    engine with ``recover=True`` also settles jobs lost in a crash
    (``recover_jobs``).
    """

    def __init__(
        self,
        workspace: str | Path,
        *,
        registry: Registry | None = None,
        recover: bool = True,
        max_bytes: int = DEFAULT_MAX_MODEL_BYTES,
        policy: OperationPolicy | None = None,
        run_id: str | None = None,
    ) -> None:
        from wse.evaluation import EvaluationService
        from wse.evaluators.datasets import DatasetStore
        from wse.experiments import ExperimentService
        from wse.jobs import JobService
        from wse.policy import default_operation_policy
        from wse.store import ArtifactStore

        self.registry = registry or builtin_registry()
        self.policy = policy or default_operation_policy()
        self.max_bytes = max_bytes
        self.bound_run_id = run_id
        self.store: ArtifactStore = ArtifactStore(workspace)
        self.datasets: DatasetStore = DatasetStore(workspace, sealed=run_id is not None)
        try:
            self.last_recovery: RecoveryReport | None = self.store.recover() if recover else None
            self.experiments: ExperimentService = ExperimentService(self, bound_run_id=run_id)
            self.budget: BudgetLedger = self.experiments.ledger
            self.jobs: JobService = JobService(self)
            self.evaluation: EvaluationService = EvaluationService(self)
            if recover:
                self.jobs.recover_lost_jobs()
        except BaseException:
            self.store.close()
            raise

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- run scoping ------------------------------------------------------------

    def _require_operator(self, action: str) -> None:
        if self.bound_run_id is not None:
            raise WSEError(ErrorCode.POLICY_DENIED, f"{action} is an operator action and is not available to a run")

    def visible_checkpoint_ids(self) -> set[str] | None:
        """Checkpoint IDs a bound engine may see, or ``None`` when unbound (operator).

        The set is the grant's base checkpoint plus the candidates created in
        the run's experiment.
        """
        if self.bound_run_id is None:
            return None
        rows = self.store.db.conn.execute(
            "SELECT base_checkpoint_id AS checkpoint_id FROM run_grants WHERE run_id = ? "
            "UNION SELECT c.checkpoint_id FROM experiment_checkpoints c "
            "JOIN experiments e ON e.experiment_id = c.experiment_id WHERE e.run_id = ?",
            (self.bound_run_id, self.bound_run_id),
        ).fetchall()
        return {row["checkpoint_id"] for row in rows}

    def require_visible(self, checkpoint_id: Any) -> str:
        """Fail like a nonexistent ID unless this engine may see ``checkpoint_id``."""
        from wse.store import checkpoint_not_found

        visible = self.visible_checkpoint_ids()
        if visible is not None and not (isinstance(checkpoint_id, str) and checkpoint_id in visible):
            raise checkpoint_not_found(checkpoint_id)
        return checkpoint_id

    def _manifest(self, checkpoint_id: str) -> ModelManifest:
        self.require_visible(checkpoint_id)
        return self.store.load_manifest(checkpoint_id)

    def import_model(
        self, manifest: ImportManifest | Mapping[str, Any], tensors: Mapping[str, Any]
    ) -> ModelManifest:
        """Validate and store a model; return its READY manifest (operator only).

        Importing an identical request again returns the existing checkpoint;
        identical weights are stored once.
        """
        self._require_operator("importing a model")
        request = _import_manifest(_with_resolved_adapter(manifest, self.registry))
        adapter_ref, adapter = _resolve_adapter(self.registry, request.adapter)
        normalized = normalize_tensors(tensors)
        size = sum(t.numel() * t.element_size() for t in normalized.values())
        if size > self.max_bytes:
            raise WSEError(
                ErrorCode.INVALID_MODEL,
                f"model tensors total {size} bytes; the limit is {self.max_bytes} bytes",
            )
        handle = _build_handle(adapter_ref, adapter, request, normalized)
        architecture = handle.architecture
        if architecture.alias_groups:
            raise WSEError(ErrorCode.INVALID_MODEL, "shared parameters are not supported in this milestone")
        config = handle.config
        dedup_key = json_hash(
            {
                "import": {**request.model_dump(mode="json"), "config": config},
                "weights_hash": handle.weights_hash,
            }
        )
        model_manifest = ModelManifest(
            checkpoint_id=self.store.new_checkpoint_id(),
            weights_hash=handle.weights_hash,
            config_hash=handle.config_hash,
            adapter=adapter_ref,
            config=config,
            tensors=architecture.tensors,
            parameter_count=architecture.parameter_count,
            input_semantics=architecture.input_semantics.model_copy(
                update={"description": request.input_description or architecture.input_semantics.description}
            ),
            output_semantics=architecture.output_semantics.model_copy(
                update={"description": request.output_description or architecture.output_semantics.description}
            ),
            alias_groups=architecture.alias_groups,
            provenance=request.provenance.to_provenance(),
            license=request.license,
        )
        return self.store.commit_checkpoint(
            model_manifest,
            dict(handle._tensors),
            dedup_key=dedup_key,
            event_type="checkpoint.imported",
        )

    def import_bundle(self, path: str | Path) -> ModelManifest:
        """Import a bundle directory (operator only)."""
        self._require_operator("importing a model")
        manifest, tensors = read_bundle(path, max_bytes=self.max_bytes, registry=self.registry)
        return self.import_model(manifest, tensors)

    def load_manifest(self, checkpoint_id: str) -> ModelManifest:
        """Return the manifest of a READY checkpoint."""
        return self._manifest(checkpoint_id)

    def list_models(self) -> list[ModelManifest]:
        """Manifests of the READY checkpoints this engine may see, oldest first."""
        manifests = []
        visible = self.visible_checkpoint_ids()
        for checkpoint_id in self.store.list_ready():
            if visible is not None and checkpoint_id not in visible:
                continue
            try:
                manifests.append(self.store.load_manifest(checkpoint_id))
            except WSEError as exc:
                if exc.code != ErrorCode.ARTIFACT_CORRUPT:
                    raise
        return manifests

    def inspect(self, checkpoint_id: str) -> ModelManifest:
        """Return the manifest of a READY checkpoint (no tensor data, no forward)."""
        return self._manifest(checkpoint_id)

    def _capabilities(self, manifest: ModelManifest) -> AdapterCapabilities:
        _, adapter = _resolve_adapter(self.registry, manifest.adapter)
        return adapter.capabilities()

    def _bounded_read(
        self, checkpoint_id: str, tensor: str, selection: Any, policy: OperationPolicy | None = None
    ) -> tuple[TensorSlice, torch.Tensor]:
        from wse.analysis import bounded_read

        manifest = self._manifest(checkpoint_id)
        info = manifest.tensor_info(tensor)
        return bounded_read(
            info=info,
            selection=selection,
            policy=policy or self.policy,
            capabilities=self._capabilities(manifest),
            adapter=manifest.adapter,
            read_region=lambda slices: self.store.read_slice(checkpoint_id, info.name, slices),
            checkpoint_id=checkpoint_id,
            weights_hash=manifest.weights_hash,
            management="managed",
        )

    def read_tensor_slice(self, checkpoint_id: str, tensor: str, selection: Any) -> TensorSlice:
        """Bounded read of an exact-name tensor region, serialized for responses.

        At most 2,048 elements and 32 KiB of canonical JSON; oversized requests
        fail with the largest permitted selection.
        """
        return self._bounded_read(checkpoint_id, tensor, selection)[0]

    def read_tensor(self, checkpoint_id: str, tensor: str, selection: Any) -> torch.Tensor:
        """Read an exact-name tensor region from a READY checkpoint (same bounds)."""
        return self._bounded_read(checkpoint_id, tensor, selection)[1]

    def analyze(
        self,
        checkpoint_id: str,
        kind: str,
        tensor: str,
        selection: Any,
        k: int | None = None,
        *,
        parameters: Mapping[str, Any] | None = None,
    ) -> AnalysisResult:
        """Run a permitted, registered analysis on a READY checkpoint.

        ``kind`` is a public alias resolved by the operation policy (``stats``,
        ``norm``, ``top_k`` by default). An omitted ``k`` for top-k resolves to
        ``min(16, element_count)``; other kinds do not accept ``k``.
        """
        return self._analyze(checkpoint_id, kind, tensor, selection, k, parameters=parameters)

    def _analyze(
        self,
        checkpoint_id: str,
        kind: str,
        tensor: str,
        selection: Any,
        k: int | None = None,
        *,
        parameters: Mapping[str, Any] | None = None,
        policy: OperationPolicy | None = None,
    ) -> AnalysisResult:
        from wse.analysis import run_analysis

        manifest = self._manifest(checkpoint_id)
        return run_analysis(
            registry=self.registry,
            policy=policy or self.policy,
            kind=kind,
            info_for=manifest.tensor_info,
            tensor=tensor,
            selection=selection,
            k=k,
            parameters=parameters,
            capabilities=self._capabilities(manifest),
            model=manifest.identity,
            read_region=lambda name, slices: self.store.read_slice(checkpoint_id, name, slices),
            checkpoint_id=checkpoint_id,
            management="managed",
            store_large=self.store.put_analysis_result,
        )

    def read_analysis_result(self, uri: str) -> dict[str, Any]:
        """Return a large analysis result stored behind ``AnalysisResult.result_uri``."""
        return json.loads(self.store.read_analysis_result(uri))

    def open_checkpoint(self, checkpoint_id: str) -> ModelHandle:
        """Load a READY checkpoint into an in-memory handle (hash-verified).

        Operator use only. The returned handle's ``forward`` runs the adapter
        directly with no policy or budget check, so this method must never be
        reachable from an agent-facing path; managed inference goes through
        the worker.
        """
        manifest = self._manifest(checkpoint_id)
        adapter_ref, adapter = _resolve_adapter(self.registry, manifest.adapter)
        handle = _build_handle(adapter_ref, adapter, manifest, self.store.load_tensors(checkpoint_id))
        if handle.identity != manifest.identity:
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"checkpoint {checkpoint_id} identity mismatch")
        return handle

    # -- experiments ------------------------------------------------------------

    def register_policy(self, policy: ExperimentPolicy | Mapping[str, Any]) -> ExperimentPolicy:
        """Register an immutable experiment policy (operator only)."""
        return self.experiments.register_policy(policy)

    def get_policy(self, policy_id: str) -> ExperimentPolicy:
        return self.experiments.get_policy(policy_id)

    def grant_run(self, task_id: str, policy_id: str, base_checkpoint_id: str) -> RunGrant:
        """Issue a run grant permitting one experiment (operator only)."""
        return self.experiments.grant_run(task_id, policy_id, base_checkpoint_id)

    def create_experiment(self, run_id: str, task_id: str, policy_id: str, key: str) -> Experiment:
        """Create the run's single experiment; retries return the existing one."""
        return self.experiments.create_experiment(run_id, task_id, policy_id, key)

    def get_experiment(self, experiment_id: str) -> Experiment:
        return self.experiments.get_experiment(experiment_id)

    def preview_patch(self, experiment_id: str, spec: PatchSpec | Mapping[str, Any], key: str) -> PatchPreview:
        """Freeze, validate, and materialize a patch without changing the selected model."""
        return self.experiments.preview_patch(experiment_id, spec, key)

    def apply_patch(self, experiment_id: str, patch_id: str, key: str) -> ModelManifest:
        """Create a READY candidate checkpoint from a stored preview."""
        return self.experiments.apply_patch(experiment_id, patch_id, key)

    def select_checkpoint(
        self, experiment_id: str, checkpoint_id: str, expected_selected_id: str, key: str
    ) -> Experiment:
        """Compare-and-swap the selected checkpoint; selecting an ancestor rolls back."""
        return self.experiments.select_checkpoint(experiment_id, checkpoint_id, expected_selected_id, key)

    def get_patch(self, experiment_id: str, patch_id: str) -> PatchPreview:
        return self.experiments.get_patch(experiment_id, patch_id)

    def read_patch(self, experiment_id: str, patch_id: str) -> PatchArtifact:
        return self.experiments.read_patch(experiment_id, patch_id)

    def list_experiment_checkpoints(self, experiment_id: str) -> list[ExperimentCheckpoint]:
        return self.experiments.list_checkpoints(experiment_id)

    def audit_log(self, experiment_id: str) -> list[dict[str, Any]]:
        return self.experiments.audit_log(experiment_id)

    # -- metered experiment reads ---------------------------------------------------

    def inspect_model(self, experiment_id: str, checkpoint_id: str) -> ModelManifest:
        """The manifest of a checkpoint the experiment owns (not charged)."""
        return self.experiments.inspect_model(experiment_id, checkpoint_id)

    def read_model_tensor(self, experiment_id: str, checkpoint_id: str, tensor: str, selection: Any) -> TensorSlice:
        """A bounded read of an owned checkpoint, charged to ``tensor_read_calls``."""
        return self.experiments.read_tensor(experiment_id, checkpoint_id, tensor, selection)

    def analyze_model(self, experiment_id: str, checkpoint_id: str, kind: str, tensor: str, selection: Any,
                      k: int | None = None) -> AnalysisResult:
        """A permitted analysis of an owned checkpoint, charged under its operator's usage contract."""
        return self.experiments.analyze(experiment_id, checkpoint_id, kind, tensor, selection, k)

    # -- budgets, suites, and jobs ----------------------------------------------------

    def reserve(self, experiment_id: str, job_id: str, requested_usage: UsageLike) -> Reservation:
        """Atomically reserve allowance for a job (operator and service use only)."""
        self._require_operator("reserving budget directly")
        return self.budget.reserve(experiment_id, job_id, requested_usage)

    def settle(self, reservation_id: str, actual_usage: UsageLike, terminal_state: str) -> Reservation:
        """Charge a reservation's executed work and release the rest (operator and service use only)."""
        self._require_operator("settling a reservation directly")
        return self.budget.settle(reservation_id, actual_usage, terminal_state)

    def register_suite(self, suite: EvaluationSuite | Mapping[str, Any]) -> EvaluationSuite:
        """Register an immutable evaluation suite (operator only)."""
        self._require_operator("registering an evaluation suite")
        return self.jobs.register_suite(suite)

    def get_suite(self, suite_id: str) -> EvaluationSuite:
        """A registered suite; a run-bound engine sees only development suites of its task."""
        return self.jobs.get_suite(suite_id)

    def submit_evaluation(self, experiment_id: str, checkpoint_id: str, suite_id: str,
                          config: EvalConfig | Mapping[str, Any], key: str) -> Job:
        """Queue a development evaluation; returns the job immediately."""
        return self.jobs.submit_evaluation(experiment_id, checkpoint_id, suite_id, config, key)

    def get_job(self, experiment_id: str, job_id: str, wait_seconds: float = 0) -> Job:
        """A job's state; ``wait_seconds`` (0-10) waits for a terminal state."""
        return self.jobs.get_job(experiment_id, job_id, wait_seconds)

    def cancel_job(self, experiment_id: str, job_id: str, key: str) -> Job:
        """Request a stop; computation already performed remains charged."""
        return self.jobs.cancel_job(experiment_id, job_id, key)

    def list_jobs(self, experiment_id: str) -> list[Job]:
        return self.jobs.list_jobs(experiment_id)

    def recover_jobs(self) -> list[str]:
        """Settle jobs whose worker lease expired; return their IDs."""
        return self.jobs.recover_lost_jobs()

    # -- tasks, datasets, reports, freezing, and final evaluation ------------------------

    def register_task(self, task: TaskDefinition | Mapping[str, Any]) -> TaskDefinition:
        """Register an immutable task definition such as ``task.json`` (operator only)."""
        return self.evaluation.register_task(task)

    def get_task(self, task_id: str) -> TaskDefinition:
        """A registered task definition; a run-bound engine sees only its run's task."""
        return self.evaluation.get_task(task_id)

    def register_dataset(self, data: SuiteData | Mapping[str, Any], *,
                         generation: Mapping[str, Any] | None = None) -> DatasetManifest:
        """Store examples in evaluator storage and return their manifest (operator only)."""
        self._require_operator("storing evaluation data")
        return self.datasets.put(data, generation=generation)

    def get_report(self, experiment_id: str, report_id: str) -> EvaluationReport:
        """An evaluation report of the experiment; final reports are withheld until it is closed."""
        return self.evaluation.get_report(experiment_id, report_id)

    def list_reports(self, experiment_id: str) -> list[EvaluationReport]:
        return self.evaluation.list_reports(experiment_id)

    def compare(self, experiment_id: str, base_report_id: str, candidate_report_id: str) -> ComparisonReport:
        """Paired difference and intervals of two development reports on the same examples and configuration."""
        return self.evaluation.compare(experiment_id, base_report_id, candidate_report_id)

    def freeze(self, experiment_id: str, expected_selected_id: str) -> Experiment:
        """Freeze the selected checkpoint once no job or reservation is outstanding (operator only)."""
        return self.experiments.freeze(experiment_id, expected_selected_id)

    def final_evaluate(self, experiment_id: str, checkpoint_id: str | None = None) -> FinalEvaluationReport:
        """Evaluate the frozen candidate and the base-checkpoint control on the final set, then close (operator only).

        ``checkpoint_id``, when given, must be the frozen selection: final
        evaluation accepts no other checkpoint.
        """
        return self.evaluation.final_evaluate(experiment_id, checkpoint_id)

    def get_final_report(self, experiment_id: str) -> FinalEvaluationReport:
        """The sealed final report of a closed experiment."""
        return self.evaluation.get_final_report(experiment_id)
