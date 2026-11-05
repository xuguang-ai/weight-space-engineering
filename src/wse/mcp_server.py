"""The MCP interface: tools, resources, and protocol mapping (Section 4 of the plan).

This is the only module that imports the MCP SDK (``mcp`` 2.3.0, the v2
``MCPServer``). It is a thin front end to the managed ``Engine``: every
handler validates its arguments against the published input schema, passes
them to one Engine operation, and maps the result or the domain error. No
handler contains an editing, analysis, budget, or evaluation algorithm, so
the Python API, the CLI, and MCP share the same operations, policy
decisions, and usage accounting.

* **Run binding.** A server is started (``wse serve``) for one persisted
  run grant and serves through an Engine bound to that run. Every tool and
  resource therefore sees only the grant's base checkpoint, the run's single
  experiment, and that experiment's candidates, patches, jobs, and reports;
  any other ID fails exactly like an ID that does not exist. Operator
  actions (imports, grants, freezing, final evaluation) are not exposed.
* **Tools.** The 13 tools of Section 4.1. Names use only letters, digits,
  and ``_`` (at most 64 characters), because OpenAI Agents SDK forwards MCP
  tool names verbatim as model function names. Every mutation takes an
  ``idempotency_key``.
* **Results.** A successful call returns ``structuredContent`` validated
  against the tool's ``outputSchema`` and a text content block that is the
  canonical compact JSON of exactly the same object, because OpenAI Agents
  SDK shows the model the text, not ``structuredContent``. Every result has
  a one-line ``summary``, the artifact ``uris`` it refers to, the per-call
  usage where the domain object records one, and the experiment's
  cumulative budget, around the domain ``result``.
* **Errors.** A domain failure is returned as an MCP tool error
  (``isError``) whose text is ``Error executing tool <name>: `` followed by
  the canonical JSON ``{"error": {"code", "message", "retryable",
  "details"}}``. Arguments that fail the input schema and unknown tools are
  reported by the SDK the same way, without the JSON body. Protocol errors
  stay JSON-RPC errors.
* **Resources.** ``wse://models/{checkpoint_id}/manifest``,
  ``wse://experiments/{experiment_id}/policy``, ``wse://reports/{report_id}``,
  and ``wse://patches/{patch_id}``, each with an optional ``?page=N``.
  Resource reads use the same bound Engine and so the same authorization as
  tools; final reports stay withheld until the experiment closes. Every
  read is bounded: a document whose JSON exceeds one page is returned in
  pages (see ``RESOURCE_PAGE_BYTES``). Tools carry inline everything a
  controller needs, so resources are optional.
* **Transport.** Local stdio; logs go to stderr. The protocol version is
  negotiated by the SDK: 2026-07-28 clients use self-contained requests,
  and older clients get the SDK's legacy handshake. No experiment state is
  kept in the connection.
"""

import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Annotated, Any, Generic, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ResourceNotFoundError, ToolError
from mcp_types import CallToolResult, TextContent, ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema

from wse.api import Engine
from wse.jobs import MAX_WAIT_SECONDS
from wse.jobs import _wait_seconds as wait_seconds_value
from wse.store import ANALYSIS_URI_PREFIX
from wse.schemas import (
    AnalysisResult,
    AxisSlice,
    BudgetEntry,
    ComparisonReport,
    ErrorCode,
    EvalConfig,
    Experiment,
    ExperimentDescription,
    ExtensionRef,
    Job,
    ModelInspection,
    ModelManifest,
    PatchPreview,
    PatchSpec,
    TensorSlice,
    Usage,
    WSEError,
    canonical_json,
)

SERVER_NAME = "wse"
MODEL_PAGE_SIZE = 50
RESOURCE_PAGE_BYTES = 32 * 1024
PAGE_ENVELOPE_BYTES = 256
WAIT_POLL_SECONDS = 0.05
READ_COUNTERS = ("tensor_read_calls", "compute_seconds")

T = TypeVar("T")
logger = logging.getLogger("wse.mcp")


# ---------------------------------------------------------------------------
# Output envelope
# ---------------------------------------------------------------------------


class ToolUsage(BaseModel):
    """Usage of this call and the experiment's cumulative budget.

    ``call`` is the usage recorded by the result itself (reads, analyses,
    previews, and settled jobs) and ``null`` when the result records none;
    the call's charges are then visible in ``cumulative``. ``cumulative``
    maps ledger counters to their limit, consumed, reserved, and remaining
    amounts; read and analysis results list only the counters they charge,
    which keeps those responses within the response-size limit.
    """

    model_config = ConfigDict(extra="forbid")

    call: Usage | None
    cumulative: dict[str, BudgetEntry]


class ToolResponse(BaseModel, Generic[T]):
    """A successful tool result: a summary, referenced artifact URIs, usage, and the domain result."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    uris: list[str]
    usage: ToolUsage | None
    result: T


class ModelSummary(BaseModel):
    """A registered checkpoint this run may use, without tensor data."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str
    adapter: ExtensionRef
    weights_hash: str
    parameter_count: int
    tensor_count: int
    input_description: str
    output_description: str
    manifest_uri: str


class ModelList(BaseModel):
    """One page of model summaries; pass ``next_cursor`` back as ``cursor`` for the next page."""

    model_config = ConfigDict(extra="forbid")

    models: list[ModelSummary]
    next_cursor: str | None


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Return ``schema`` with every local ``$ref`` replaced by its definition (no ``$defs`` left)."""
    definitions = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                target = resolve(definitions[node["$ref"].rsplit("/", 1)[-1]])
                return {**target, **{k: resolve(v) for k, v in node.items() if k != "$ref"}}
            return {key: resolve(value) for key, value in node.items() if key != "$defs"}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)


SELECTION_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": _inline_refs(AxisSlice.model_json_schema()),
    "description": "One {start, stop, step} object per tensor axis: nonnegative, nonempty, in-bounds, step 1.",
}
PATCH_SPEC_SCHEMA = _inline_refs(PatchSpec.model_json_schema())
EVAL_CONFIG_SCHEMA = _inline_refs(EvalConfig.model_json_schema())

IdArg = Annotated[str, Field(description="An ID returned by an earlier tool call.")]
KeyArg = Annotated[str, Field(description="A client-chosen key: repeating a request with the same key and input "
                                          "returns the original result without charging again; use a new key for a "
                                          "new request.")]
SelectionArg = Annotated[Any, WithJsonSchema(SELECTION_SCHEMA)]
PatchSpecArg = Annotated[Any, WithJsonSchema(PATCH_SPEC_SCHEMA)]
EvalConfigArg = Annotated[Any, WithJsonSchema({**EVAL_CONFIG_SCHEMA, "description": "A preregistered sample_count "
                                               "and seed of the suite; the server fixes the device and batch size."})]
WaitArg = Annotated[float, Field(json_schema_extra={"minimum": 0, "maximum": MAX_WAIT_SECONDS},
                                 description="Seconds (0-10) to hold the call until the job is terminal.")]

READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, destructive_hint=False,
                            open_world_hint=False)
MUTATION = ToolAnnotations(read_only_hint=False, idempotent_hint=True, destructive_hint=False,
                           open_world_hint=False)
# Reads and analyses change no object but charge the budget on every call.
METERED = ToolAnnotations(read_only_hint=False, idempotent_hint=False, destructive_hint=False,
                          open_world_hint=False)


# ---------------------------------------------------------------------------
# Error and result mapping
# ---------------------------------------------------------------------------


def error_text(error: WSEError) -> str:
    """The canonical JSON body of a domain error, as carried by MCP tool and resource errors."""
    return canonical_json({"error": error.to_domain_error().model_dump(mode="json")})


def _respond(response: ToolResponse[Any]) -> CallToolResult:
    structured = response.model_dump(mode="json")
    return CallToolResult(content=[TextContent(type="text", text=canonical_json(structured))],
                          structured_content=structured)


def _manifest_uri(checkpoint_id: str) -> str:
    return f"wse://models/{checkpoint_id}/manifest"


def _patch_uri(patch_id: str) -> str:
    return f"wse://patches/{patch_id}"


def _report_uri(report_id: str) -> str:
    return f"wse://reports/{report_id}"


def _parse_cursor(cursor: Any) -> int:
    if cursor in (None, ""):
        return 0
    if isinstance(cursor, str) and cursor.startswith("models:") and cursor[7:].isdigit():
        return int(cursor[7:])
    raise WSEError(ErrorCode.INVALID_ARGUMENT, f"cursor {cursor!r} was not returned by wse_list_models; omit it to "
                   "start from the first page")


def _paged(uri: str, document: BaseModel | dict[str, Any], page: Any) -> str:
    """A bounded resource read: the whole document, or one page of its JSON text.

    A document whose whole read fits in ``RESOURCE_PAGE_BYTES`` of canonical
    JSON is returned as ``{"uri", "page": 0, "page_count": 1, "next_uri":
    null, "content": <document>}``. A larger one is split into pages of its
    ASCII JSON text, each returned as ``{"uri", "page", "page_count",
    "next_uri", "chunk"}`` and each at most ``RESOURCE_PAGE_BYTES`` of JSON
    (escaping included); concatenating the chunks of pages
    0..page_count-1 gives the document's JSON.
    """
    if isinstance(page, bool) or not isinstance(page, int) or page < 0:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "page must be a nonnegative integer")
    data = document.model_dump(mode="json") if isinstance(document, BaseModel) else document
    whole = canonical_json({"uri": uri, "page": 0, "page_count": 1, "next_uri": None, "content": data})
    if len(whole.encode("utf-8")) <= RESOURCE_PAGE_BYTES:
        if page != 0:
            raise WSEError(ErrorCode.INVALID_ARGUMENT, f"{uri} has one page; page {page} does not exist")
        return whole
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    bounds = _chunk_bounds(text, RESOURCE_PAGE_BYTES - PAGE_ENVELOPE_BYTES - len(uri))
    if page >= len(bounds):
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"{uri} has {len(bounds)} pages (0-{len(bounds) - 1}); page "
                       f"{page} does not exist")
    start, stop = bounds[page]
    return canonical_json({"uri": uri, "page": page, "page_count": len(bounds),
                           "next_uri": f"{uri}?page={page + 1}" if page + 1 < len(bounds) else None,
                           "chunk": text[start:stop]})


def _chunk_bounds(text: str, budget: int) -> list[tuple[int, int]]:
    """Split ASCII ``text`` into consecutive chunks whose JSON string encodings fit in ``budget`` bytes."""
    bounds, start, size = [], 0, 2  # the surrounding quotes
    for index, character in enumerate(text):
        cost = 2 if character in '"\\' else 1  # ASCII JSON text escapes only quotes and backslashes again
        if size + cost > budget:
            bounds.append((start, index))
            start, size = index, 2
        size += cost
    bounds.append((start, len(text)))
    return bounds


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def server_instructions(engine: Engine) -> str:
    """Neutral connection instructions: the run to use and how results and errors are shaped."""
    grant = engine.experiments.get_run_grant(engine.bound_run_id)
    return (
        f"Weight-space engineering tools for run {grant.run_id} (task {grant.task_id}, policy {grant.policy_id}). "
        "Start with wse_create_experiment using these IDs (retries return the same experiment), then use "
        "wse_get_experiment for the task, objective, permitted operations, development suites, and budget. "
        "Every result is JSON with summary, uris, usage, and result; errors carry code, message, retryable, and "
        "details. Mutations need an idempotency_key."
    )


class WSEMCPServer:
    """MCP tools and resources over one run-bound Engine.

    Engine calls are serialized with a lock: the SDK runs synchronous tool
    handlers on worker threads, and one Engine holds one SQLite connection.
    Waiting in ``wse_get_job`` polls outside the lock, so it never blocks
    other calls.
    """

    def __init__(self, engine: Engine, *, constraint_status: str = "unverified") -> None:
        if engine.bound_run_id is None:
            raise WSEError(ErrorCode.POLICY_DENIED, "the MCP server serves only an Engine bound to one run grant")
        self.engine = engine
        self.constraint_status = constraint_status
        self._lock = threading.Lock()
        self.server: MCPServer = MCPServer(SERVER_NAME, instructions=server_instructions(engine),
                                           version=_package_version(), log_level="INFO")
        self._register_tools()
        self._register_resources()

    # -- plumbing ---------------------------------------------------------------

    def _call(self, operation: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        with self._lock:
            try:
                return operation(*args, **kwargs)
            except WSEError as exc:
                raise ToolError(error_text(exc)) from exc

    def _read(self, operation: Callable[..., T], *args: Any) -> T:
        with self._lock:
            try:
                return operation(*args)
            except WSEError as exc:
                if exc.code == ErrorCode.NOT_FOUND:
                    raise ResourceNotFoundError(error_text(exc)) from exc
                raise ResourceError(error_text(exc)) from exc

    def _usage(self, experiment_id: str, call: Usage | None, counters: tuple[str, ...] | None = None) -> ToolUsage:
        budget = self.engine.get_experiment(experiment_id).budget
        if counters is not None:
            budget = {name: entry for name, entry in budget.items() if name in counters}
        return ToolUsage(call=call, cumulative=budget)

    def _describe(self, experiment_id: str) -> ExperimentDescription:
        return self.engine.describe_experiment(experiment_id, constraint_status=self.constraint_status)

    # -- tools ------------------------------------------------------------------

    def _register_tools(self) -> None:
        tools = [
            (self.wse_list_models, READ_ONLY),
            (self.wse_create_experiment, MUTATION),
            (self.wse_get_experiment, READ_ONLY),
            (self.wse_inspect_model, METERED),
            (self.wse_read_tensor, METERED),
            (self.wse_analyze, METERED),
            (self.wse_preview_patch, MUTATION),
            (self.wse_apply_patch, MUTATION),
            (self.wse_submit_evaluation, MUTATION),
            (self.wse_get_job, READ_ONLY),
            (self.wse_cancel_job, MUTATION),
            (self.wse_compare, READ_ONLY),
            (self.wse_select_checkpoint, MUTATION),
        ]
        for function, annotations in tools:
            self.server.add_tool(function, name=function.__name__, annotations=annotations)

    def wse_list_models(self, cursor: Annotated[str | None, Field(
            description="Omit for the first page; pass next_cursor for the next one.")] = None,
    ) -> Annotated[CallToolResult, ToolResponse[ModelList]]:
        """List the registered checkpoints this run may use (no filesystem paths are scanned)."""
        def run() -> ToolResponse[ModelList]:
            offset = _parse_cursor(cursor)
            manifests = self.engine.list_models()
            page = manifests[offset:offset + MODEL_PAGE_SIZE]
            models = [ModelSummary(checkpoint_id=m.checkpoint_id, adapter=m.adapter, weights_hash=m.weights_hash,
                                   parameter_count=m.parameter_count, tensor_count=len(m.tensors),
                                   input_description=m.input_semantics.description,
                                   output_description=m.output_semantics.description,
                                   manifest_uri=_manifest_uri(m.checkpoint_id)) for m in page]
            more = offset + MODEL_PAGE_SIZE < len(manifests)
            return ToolResponse[ModelList](
                summary=f"{len(models)} of {len(manifests)} models", uris=[m.manifest_uri for m in models],
                usage=None, result=ModelList(models=models,
                                             next_cursor=f"models:{offset + MODEL_PAGE_SIZE}" if more else None))
        return _respond(self._call(run))

    def wse_create_experiment(self, run_id: IdArg, task_id: IdArg, policy_id: IdArg, idempotency_key: KeyArg,
                              ) -> Annotated[CallToolResult, ToolResponse[ExperimentDescription]]:
        """Create this run's single experiment; a retry returns the existing experiment and never resets its budget."""
        def run() -> ToolResponse[ExperimentDescription]:
            experiment = self.engine.create_experiment(run_id, task_id, policy_id, idempotency_key)
            description = self._describe(experiment.experiment_id)
            return ToolResponse[ExperimentDescription](
                summary=f"experiment {experiment.experiment_id} is {experiment.state}; selected checkpoint "
                        f"{experiment.selected_checkpoint_id}",
                uris=[description.policy_uri, _manifest_uri(experiment.base_checkpoint_id)],
                usage=self._usage(experiment.experiment_id, None), result=description)
        return _respond(self._call(run))

    def wse_get_experiment(self, experiment_id: IdArg) -> Annotated[CallToolResult,
                                                                    ToolResponse[ExperimentDescription]]:
        """State, task definition and objective, selected checkpoint, budget, permitted operations, and suites."""
        def run() -> ToolResponse[ExperimentDescription]:
            description = self._describe(experiment_id)
            experiment = description.experiment
            return ToolResponse[ExperimentDescription](
                summary=f"experiment {experiment.experiment_id} is {experiment.state} in {experiment.mode} mode; "
                        f"selected checkpoint {experiment.selected_checkpoint_id}",
                uris=[description.policy_uri, *(_manifest_uri(c.checkpoint_id) for c in description.checkpoints)],
                usage=self._usage(experiment_id, None), result=description)
        return _respond(self._call(run))

    def wse_inspect_model(self, experiment_id: IdArg, checkpoint_id: IdArg,
                          ) -> Annotated[CallToolResult, ToolResponse[ModelInspection]]:
        """The manifest of a checkpoint of this experiment and whole-tensor statistics by layer.

        The manifest is free; computing the statistics is charged as compute time.
        """
        def run() -> ToolResponse[ModelInspection]:
            inspection = self.engine.inspect_model_details(experiment_id, checkpoint_id)
            return ToolResponse[ModelInspection](
                summary=f"{inspection.role} checkpoint {checkpoint_id}: {len(inspection.manifest.tensors)} tensors, "
                        f"{inspection.manifest.parameter_count} parameters",
                uris=[inspection.manifest_uri], usage=self._usage(experiment_id, inspection.usage),
                result=inspection)
        return _respond(self._call(run))

    def wse_read_tensor(self, experiment_id: IdArg, checkpoint_id: IdArg,
                        tensor: Annotated[str, Field(description="Exact tensor name from the manifest.")],
                        selection: SelectionArg) -> Annotated[CallToolResult, ToolResponse[TensorSlice]]:
        """Read a bounded slice (at most 2,048 elements); larger requests fail with the largest permitted selection."""
        def run() -> ToolResponse[TensorSlice]:
            region = self.engine.read_model_tensor(experiment_id, checkpoint_id, tensor, selection)
            return ToolResponse[TensorSlice](
                summary=f"{region.element_count} values of {region.tensor}", uris=[],
                usage=self._usage(experiment_id, region.usage, READ_COUNTERS), result=region)
        return _respond(self._call(run))

    def wse_analyze(self, experiment_id: IdArg, checkpoint_id: IdArg,
                    kind: Annotated[str, Field(description="A permitted analysis kind (see wse_get_experiment).")],
                    tensor: Annotated[str, Field(description="Exact tensor name from the manifest.")],
                    selection: SelectionArg,
                    k: Annotated[int | None, Field(description="top_k only: 1 <= k <= min(64, element count); "
                                                               "defaults to min(16, element count).")] = None,
                    ) -> Annotated[CallToolResult, ToolResponse[AnalysisResult]]:
        """Run a permitted registered analysis (stats, norm, or top_k) on a tensor slice."""
        def run() -> ToolResponse[AnalysisResult]:
            result = self.engine.analyze_model(experiment_id, checkpoint_id, kind, tensor, selection, k)
            counters = tuple(name for name, value in result.usage.model_dump().items()
                             if isinstance(value, (int, float)) and value) + ("compute_seconds",)
            return ToolResponse[AnalysisResult](
                summary=f"{result.kind} of {result.element_count} elements of {result.tensor}",
                uris=[result.result_uri] if result.result_uri else [],
                usage=self._usage(experiment_id, result.usage, counters), result=result)
        return _respond(self._call(run))

    def wse_preview_patch(self, experiment_id: IdArg, patch_spec: PatchSpecArg, idempotency_key: KeyArg,
                          ) -> Annotated[CallToolResult, ToolResponse[PatchPreview]]:
        """Validate a declarative patch and save an immutable preview without changing the selected checkpoint.

        Example patch_spec for a hypothetical 8-input MLP (IDs come from earlier results):
        {"base_checkpoint_id": "ckpt_...", "hypothesis": "Hidden unit 3 dominates the output.",
        "operations": [{"op": "scale_slice", "tensor": "layers.0.weight",
        "selection": [{"start": 3, "stop": 4, "step": 1}, {"start": 0, "stop": 8, "step": 1}], "factor": 0.5}]}
        """
        def run() -> ToolResponse[PatchPreview]:
            preview = self.engine.preview_patch(experiment_id, patch_spec, idempotency_key)
            return ToolResponse[PatchPreview](
                summary=f"patch {preview.patch_id} changes {preview.changed_elements} of {preview.parameter_count} "
                        "parameters", uris=[preview.patch_uri],
                usage=self._usage(experiment_id, preview.usage), result=preview)
        return _respond(self._call(run))

    def wse_apply_patch(self, experiment_id: IdArg, patch_id: IdArg, idempotency_key: KeyArg,
                        ) -> Annotated[CallToolResult, ToolResponse[ModelManifest]]:
        """Create a candidate checkpoint from a previewed patch (by patch_id only); it does not select it."""
        def run() -> ToolResponse[ModelManifest]:
            manifest = self.engine.apply_patch(experiment_id, patch_id, idempotency_key)
            return ToolResponse[ModelManifest](
                summary=f"candidate checkpoint {manifest.checkpoint_id} from patch {patch_id}",
                uris=[_manifest_uri(manifest.checkpoint_id), _patch_uri(patch_id)],
                usage=self._usage(experiment_id, None), result=manifest)
        return _respond(self._call(run))

    def wse_submit_evaluation(self, experiment_id: IdArg, checkpoint_id: IdArg,
                              suite_id: Annotated[str, Field(description="A development suite of the experiment.")],
                              eval_config: EvalConfigArg, idempotency_key: KeyArg,
                              ) -> Annotated[CallToolResult, ToolResponse[Job]]:
        """Queue a development evaluation and return its job immediately; poll it with wse_get_job."""
        def run() -> ToolResponse[Job]:
            job = self.engine.submit_evaluation(experiment_id, checkpoint_id, suite_id, eval_config, idempotency_key)
            return self._job_response(experiment_id, job)
        return _respond(self._call(run))

    def wse_get_job(self, experiment_id: IdArg, job_id: IdArg, wait_seconds: WaitArg = 0,
                    ) -> Annotated[CallToolResult, ToolResponse[Job]]:
        """A job's state, progress, and, once it succeeded, its report summary; can wait up to 10 seconds."""
        # The wait happens here, polling outside the lock; the service validates every argument.
        wait = self._call(wait_seconds_value, wait_seconds)
        deadline = time.monotonic() + wait
        while True:
            job = self._call(self.engine.get_job, experiment_id, job_id, 0)
            left = deadline - time.monotonic()
            if job.terminal or left <= 0:
                return _respond(self._call(self._job_response, experiment_id, job))
            time.sleep(min(WAIT_POLL_SECONDS, left))

    def wse_cancel_job(self, experiment_id: IdArg, job_id: IdArg, idempotency_key: KeyArg,
                       ) -> Annotated[CallToolResult, ToolResponse[Job]]:
        """Request that a job stop; computation already performed remains charged."""
        def run() -> ToolResponse[Job]:
            return self._job_response(experiment_id, self.engine.cancel_job(experiment_id, job_id, idempotency_key))
        return _respond(self._call(run))

    def _job_response(self, experiment_id: str, job: Job) -> ToolResponse[Job]:
        summary = f"job {job.job_id} is {job.state}"
        if job.result and "metrics" in job.result:
            summary += f"; metrics {canonical_json(job.result['metrics'])} on {job.result.get('sample_count')} examples"
        if job.error is not None:
            summary += f"; {job.error.code.value}: {job.error.message}"
        return ToolResponse[Job](summary=summary, uris=[_report_uri(job.report_id)] if job.report_id else [],
                                 usage=self._usage(experiment_id, job.usage), result=job)

    def wse_compare(self, experiment_id: IdArg, base_report_id: IdArg, candidate_report_id: IdArg,
                    ) -> Annotated[CallToolResult, ToolResponse[ComparisonReport]]:
        """Paired difference of two development reports with intervals; mismatched configurations are rejected."""
        def run() -> ToolResponse[ComparisonReport]:
            comparison = self.engine.compare(experiment_id, base_report_id, candidate_report_id)
            return ToolResponse[ComparisonReport](
                summary=f"accuracy {comparison.base_value:g} -> {comparison.candidate_value:g} "
                        f"(difference {comparison.difference:+g})",
                uris=[_report_uri(base_report_id), _report_uri(candidate_report_id)],
                usage=self._usage(experiment_id, None), result=comparison)
        return _respond(self._call(run))

    def wse_select_checkpoint(self, experiment_id: IdArg, checkpoint_id: IdArg,
                              expected_selected_id: Annotated[str, Field(
                                  description="The currently selected checkpoint ID (compare-and-swap).")],
                              idempotency_key: KeyArg) -> Annotated[CallToolResult, ToolResponse[Experiment]]:
        """Select a checkpoint of this experiment by compare-and-swap; selecting an ancestor rolls back."""
        def run() -> ToolResponse[Experiment]:
            experiment = self.engine.select_checkpoint(experiment_id, checkpoint_id, expected_selected_id,
                                                       idempotency_key)
            return ToolResponse[Experiment](
                summary=f"selected checkpoint {experiment.selected_checkpoint_id}",
                uris=[_manifest_uri(experiment.selected_checkpoint_id)],
                usage=self._usage(experiment_id, None), result=experiment)
        return _respond(self._call(run))

    # -- resources --------------------------------------------------------------

    def _register_resources(self) -> None:
        server = self.server

        @server.resource("wse://models/{checkpoint_id}/manifest{?page}", name="model_manifest",
                         description="The manifest of a checkpoint this run may use.", mime_type="application/json")
        def model_manifest(checkpoint_id: str, page: int = 0) -> str:
            uri = _manifest_uri(checkpoint_id)
            return self._read(lambda: _paged(uri, self.engine.load_manifest(checkpoint_id), page))

        @server.resource("wse://experiments/{experiment_id}/policy{?page}", name="experiment_policy",
                         description="The immutable policy of this run's experiment.", mime_type="application/json")
        def experiment_policy(experiment_id: str, page: int = 0) -> str:
            uri = f"wse://experiments/{experiment_id}/policy"
            return self._read(lambda: _paged(uri, self.engine.experiment_policy(experiment_id), page))

        @server.resource("wse://reports/{report_id}{?page}", name="evaluation_report",
                         description="An evaluation report of this run's experiment.", mime_type="application/json")
        def evaluation_report(report_id: str, page: int = 0) -> str:
            return self._read(lambda: _paged(_report_uri(report_id), self.engine.find_report(report_id), page))

        @server.resource("wse://patches/{patch_id}{?page}", name="patch",
                         description="A patch of this run's experiment: its preview, frozen spec, and per-step "
                                     "summaries (the materialized after-values are not included).",
                         mime_type="application/json")
        def patch(patch_id: str, page: int = 0) -> str:
            return self._read(lambda: _paged(_patch_uri(patch_id), self.engine.find_patch(patch_id), page))

        @server.resource("wse://analyses/{digest}{?page}", name="analysis_result",
                         description="A large analysis result of this run's experiment, referenced by an analysis "
                                     "result_uri.", mime_type="application/json")
        def analysis_result(digest: str, page: int = 0) -> str:
            uri = f"{ANALYSIS_URI_PREFIX}{digest}"
            return self._read(lambda: _paged(uri, self.engine.read_analysis_result(uri), page))


def _package_version() -> str:
    from wse import __version__

    return __version__


def build_server(engine: Engine, *, constraint_status: str = "unverified") -> MCPServer:
    """The ``MCPServer`` for a run-bound Engine."""
    return WSEMCPServer(engine, constraint_status=constraint_status).server


def serve(workspace: str, run_id: str, *, registry: Any = None, constraint_status: str = "unverified",
          worker: bool = True, worker_args: tuple[str, ...] = (), worker_env: dict[str, str] | None = None) -> int:
    """Serve one run grant over stdio until the client disconnects (``wse serve``).

    The server process owns the run's job scheduler: unless ``worker`` is
    false it keeps one evaluation worker process running for the workspace
    (``WorkerSupervisor``) and stops it on exit. Logs go to stderr; stdout
    carries only the protocol.
    """
    from wse.worker import WorkerSupervisor

    engine = Engine(workspace, registry=registry, run_id=run_id)
    supervisor = None
    try:
        mcp_server = build_server(engine, constraint_status=constraint_status)
        if worker:
            supervisor = WorkerSupervisor(workspace, extra_args=worker_args, env=worker_env).start()
        logger.info("wse serve: run %s bound; worker %s; constraint_status %s", run_id,
                    "started" if worker else "disabled", constraint_status)
        mcp_server.run("stdio")
    finally:
        if supervisor is not None:
            supervisor.stop(stop_timeout=1.5)
        engine.close()
    return 0


__all__ = ["MODEL_PAGE_SIZE", "RESOURCE_PAGE_BYTES", "SERVER_NAME", "ModelList", "ModelSummary", "ToolResponse",
           "ToolUsage", "WSEMCPServer", "build_server", "error_text", "serve", "server_instructions"]
