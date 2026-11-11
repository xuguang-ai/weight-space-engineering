"""The MCP surface for language-model experiments (M2 plan Section 5.6; Task 11, part 2b).

* **New tools.** ``wse_get_report`` returns a report of the run's experiment
  (an oversized one as a bounded overview that points to the paged
  resource), and ``wse_probe`` queues a probe; ``wse_get_job`` shows a
  probe's result URI and summary and an evaluation's report ID.
* **Extended tools.** ``wse_analyze`` takes ``parameters``,
  ``wse_submit_evaluation`` runs generation passes, ``wse_preview_patch``
  takes the M2 edit operators and vector references, ``wse_compare`` pairs
  two generation reports on accuracy, limit-hit rate, and generated tokens,
  and ``wse_inspect_model`` pages the tensors of a large manifest.
* **Response sizes.** Previews of open-weight models list only the changed
  tensors, inspections of large manifests are paged, and an applied
  candidate's manifest lists only its changed tensors whenever the whole
  would exceed the inline-result limit; a report's tails are shortened when
  its escaped text mirror would not fit; M1 responses are unchanged.
* **Parity.** MCP (a real stdio session with ``wse serve``) and the Python
  and strategy paths agree on results, errors, and usage. An OpenAI Agents
  SDK agent under ``qwen_loop_v1`` sees every tool the policy permits.
* **Carried over from M1.** A flat list is accepted for a selection whose
  other axes have length 1, and a run-bound server defaults the experiment
  ID.

Language models are the tiny offline fixtures of ``tests/hf_fixtures.py``;
jobs run in a worker in the test process (``wse serve --no-worker``), so
scripted outputs (``_script``) reach them.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import anyio
import numpy as np
import pytest

from test_llm_policy_mcp import (  # noqa: F401 - lm is a fixture
    REVISION,
    _budget,
    _open_run,
    _samples,
    _script,
    _suite,
    _worker,
    lm,
)
from test_mcp_e2e import (  # noqa: F401 - trial and plugins are fixtures
    MAX_RESPONSE_BYTES,
    TOOL_NAMES,
    Session,
    ToolFailure,
    _create,
    _wire_bytes,
    connect,
    plugins,
    run,
    trial,
)
from wse.api import Engine, apply_operator, load_model
from wse.policy import QWEN_LOOP_POLICY_ID, ExperimentPolicy, builtin_experiment_policies
from wse.schemas import ErrorCode, WSEError, canonical_json


def _sel(*ranges):
    return [{"start": start, "stop": stop, "step": 1} for start, stop in ranges]


def _expect(code, func, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        func(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


@pytest.fixture
def task_model(signed_sum):
    return load_model(signed_sum.task_tensors(), "mlp@1", signed_sum.CONFIG)


ROW1 = _sel((1, 2), (0, 4))  # a [1, 4] selection of the signed-sum model's [2, 4] layers.1.weight
LARGEST = float(np.finfo(np.float32).max)
SHORTEST = float(str(np.float32(LARGEST)))  # 3.4028235e+38: the float32 maximum's shortest digits
BEYOND = 2.0 ** 128 - 2.0 ** 103  # the smallest magnitude that rounds to infinity in float32


# ---------------------------------------------------------------------------
# Carried over from M1: flat lists
# ---------------------------------------------------------------------------


def test_a_flat_list_is_accepted_when_the_other_axes_have_length_one(task_model, engine, workspace, bundles):
    nested = apply_operator(task_model, "set_slice", {"tensor": "layers.1.weight", "selection": ROW1,
                                                      "values": [[1, -1, 1, -1]]})
    flat = apply_operator(task_model, "set_slice", {"tensor": "layers.1.weight", "selection": ROW1,
                                                    "values": [1, -1, 1, -1]})
    # The same step: the recorded parameters take the selection's nesting.
    assert flat.patch == nested.patch and flat.candidate.weights_hash == nested.candidate.weights_hash
    assert flat.patch.steps[0].parameters == {"values": [[1.0, -1.0, 1.0, -1.0]]}
    # A column ([4, 1] of the [4, 2] layers.0.weight) and a single element.
    column = apply_operator(task_model, "set_slice", {"tensor": "layers.0.weight", "selection": _sel((0, 4), (1, 2)),
                                                      "values": [1, 2, 3, 4]})
    assert column.patch.steps[0].parameters == {"values": [[1.0], [2.0], [3.0], [4.0]]}
    assert column.candidate.tensor("layers.0.weight")[:, 1].tolist() == [1.0, 2.0, 3.0, 4.0]
    single = apply_operator(task_model, "set_slice", {"tensor": "layers.0.weight", "selection": _sel((2, 3), (0, 1)),
                                                      "values": [0.5]})
    assert single.patch.steps[0].parameters == {"values": [[0.5]]}
    # Two axes longer than 1 still need the nesting, which the error names; a wrong count is refused too.
    for values in ([1, -1, 1, -1, 1, -1, 1, -1], [1, -1, 1]):
        error = _expect(ErrorCode.INVALID_ARGUMENT, apply_operator, task_model, "set_slice",
                        {"tensor": "layers.1.weight", "selection": _sel((0, 2), (0, 4)) if len(values) == 8 else ROW1,
                         "values": values})
        assert "flat list" in error.message and "nested" in error.message
    error = _expect(ErrorCode.INVALID_ARGUMENT, apply_operator, task_model, "set_slice",
                    {"tensor": "layers.1.weight", "selection": _sel((0, 2), (0, 4)), "values": list(range(8))})
    assert "a list of 2 lists of 4 numbers" in error.message
    # Managed previews of the two forms freeze the same spec and patch.
    base = engine.import_bundle(bundles["task"]).checkpoint_id
    grant = engine.grant_run("signed_sum_v1", "small_model_v1", base)
    with Engine(workspace, run_id=grant.run_id) as bound:
        eid = bound.create_experiment(grant.run_id, "signed_sum_v1", "small_model_v1", "create").experiment_id
        previews = [bound.preview_patch(eid, {"base_checkpoint_id": base, "hypothesis": "Set one output row.",
                                              "operations": [{"op": "set_slice", "tensor": "layers.1.weight",
                                                              "selection": ROW1, "values": values}]}, key)
                    for key, values in (("nested", [[1, -1, 1, -1]]), ("flat", [1, -1, 1, -1]))]
        assert previews[0].spec_hash == previews[1].spec_hash
        assert previews[0].result_weights_hash == previews[1].result_weights_hash
        assert bound.find_patch(previews[1].patch_id).spec.operations[0].parameters == {
            "values": [[1.0, -1.0, 1.0, -1.0]]}


# ---------------------------------------------------------------------------
# The float32 maximum in its shortest digits
# ---------------------------------------------------------------------------


def test_the_float32_maximum_in_its_shortest_digits_is_accepted_wherever_a_float32_value_is(task_model):
    from wse.analysis import analyze_tensor, resolve_vector_reference
    from wse.policy import default_operation_policy

    assert SHORTEST > LARGEST and np.float32(SHORTEST) == np.float32(LARGEST)
    for value in (SHORTEST, -SHORTEST):
        written = apply_operator(task_model, "set_slice", {"tensor": "layers.1.bias", "selection": _sel((0, 1)),
                                                           "values": [value]})
        assert written.candidate.tensor("layers.1.bias")[0].item() == math.copysign(LARGEST, value)
        scaled = apply_operator(task_model, "scale_slice", {"tensor": "layers.0.weight",
                                                            "selection": _sel((0, 1), (0, 1)), "factor": value})
        assert scaled.candidate.tensor("layers.0.weight")[0, 0].item() == math.copysign(LARGEST, value)
    policy = default_operation_policy().with_analysis("direction_projection", "direction_projection@1")
    projected = analyze_tensor(task_model, "direction_projection", "layers.1.weight", _sel((0, 2), (0, 4)),
                               parameters={"direction": [SHORTEST, 0.0], "axis": "columns"}, policy=policy)
    assert projected.result["direction_norm"] == pytest.approx(LARGEST, rel=1e-7)
    reference = "wse://analyses/" + "0" * 64 + "#/vector"
    vector = resolve_vector_reference(reference, lambda uri: {"vector": [SHORTEST, -SHORTEST, 1.0]},
                                      parameter="direction", max_length=8)
    assert vector.tolist() == [LARGEST, -LARGEST, 1.0]
    # A value that rounds to infinity in float32 is still refused, as a domain error.
    for operator, parameters in (("set_slice", {"tensor": "layers.1.bias", "selection": _sel((0, 1)),
                                                "values": [BEYOND]}),
                                 ("scale_slice", {"tensor": "layers.0.weight", "selection": _sel((0, 1), (0, 1)),
                                                  "factor": BEYOND})):
        assert "float32" in _expect(ErrorCode.INVALID_ARGUMENT, apply_operator, task_model, operator,
                                    parameters).message
    error = _expect(ErrorCode.INVALID_ARGUMENT, analyze_tensor, task_model, "direction_projection", "layers.1.weight",
                    _sel((0, 2), (0, 4)), parameters={"direction": [BEYOND, 0.0], "axis": "columns"}, policy=policy)
    assert "float32" in error.message
    error = _expect(ErrorCode.INVALID_ARGUMENT, resolve_vector_reference, reference,
                    lambda uri: {"vector": [BEYOND, 1.0]}, parameter="direction", max_length=8)
    assert error.details["nonfinite_count"] == 1


def test_the_unused_whole_tensor_scaling_helper_that_named_the_factor_is_gone():
    import wse.policy

    assert not hasattr(wse.policy, "require_no_whole_tensor_scaling")
    assert "require_no_whole_tensor_scaling" not in (wse.policy.__doc__ or "")


# ---------------------------------------------------------------------------
# Shared helpers of the MCP tests
# ---------------------------------------------------------------------------

DOWN0 = "model.layers.0.mlp.down_proj"  # [16, 32] in the Llama fixture: 32 inputs, 16 outputs
FULL = _sel((0, 16), (0, 32))
SITES = ["residual:1", f"{DOWN0}:input", {"role": "mlp.down", "layer": 0, "point": "output"}]
INLINE_RESULT_LIMIT = 31 * 1024  # max_inline_json_bytes (32 KiB) less the 1 KiB response envelope
# Identifiers and run-scoped URIs, which differ between two runs that make the same requests.
_IDS = re.compile(r"\b(?:run|exp|job|resv|report|patch|ckpt)_[0-9a-f]{32}\b|wse://analyses/[0-9a-f]{64}")
# Measured times and timestamps, which differ between any two runs.
_VOLATILE = {"created_at", "started_at", "finished_at", "settled_at", "frozen_at", "ended_at", "compute_seconds",
             "elapsed_seconds", "peak_memory_bytes"}


def _loop_or_answer(prompt: str) -> str:
    """Scripted outputs: even practice problems answer and stop; odd ones, and every other prompt, loop to the limit."""
    match = re.search(r"p(\d+) plus", prompt)
    index = int(match.group(1)) if match else 1
    return f"\\boxed{{{2 * index}}}<STOP>" if index % 2 == 0 else "loop " * 10


def _normalized(value: Any, names: dict[str, str]) -> Any:
    """``value`` with IDs and URIs renamed in order of first appearance and measured times left out."""
    if isinstance(value, dict):
        return {key: _normalized(value[key], names) for key in sorted(value) if key not in _VOLATILE}
    if isinstance(value, list):
        return [_normalized(item, names) for item in value]
    if isinstance(value, str):
        return _IDS.sub(lambda match: names.setdefault(match.group(0), f"<id{len(names)}>"), value)
    return value


def _size(value: Any) -> int:
    data = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    return len(canonical_json(data).encode("utf-8"))


def _small_limit_policy(policy_id: str, inline_bytes: int) -> ExperimentPolicy:
    """``qwen_loop_v1`` with a smaller inline-result limit, so that the tiny fixtures take the bounded forms."""
    loop = builtin_experiment_policies()[QWEN_LOOP_POLICY_ID]
    return loop.model_copy(update={"policy_id": policy_id,
                                   "operations": loop.operations.with_limits(max_inline_json_bytes=inline_bytes)})


async def _outcome(session: Session, tool: str, **arguments: Any) -> tuple[str, Any, Any]:
    """A tool call through MCP as ``("ok", result, response)`` or ``("error", error body, response)``."""
    response = await session.raw(tool, **arguments)
    if response.is_error:
        return "error", ToolFailure(tool, response.content[0].text).error, response
    assert json.loads(response.content[0].text) == response.structured_content
    assert _wire_bytes(response) <= MAX_RESPONSE_BYTES, (tool, _wire_bytes(response))
    return "ok", response.structured_content["result"], response


# ---------------------------------------------------------------------------
# Parity of MCP and the Python and strategy paths
# ---------------------------------------------------------------------------


def _requests(eid: str, base: str):
    """The parity test's requests, as a generator that receives each outcome ``(kind, value)``.

    It yields ``(tool, arguments)`` for a call and ``("job", job_id)`` where
    the test's worker must run a queued job.
    """
    def project_out(direction: str, columns: list[int]) -> dict[str, Any]:
        return {"base_checkpoint_id": base, "hypothesis": "Looping outputs differ along this direction.",
                "operations": [{"op": "project_out", "tensor": f"{DOWN0}.weight", "selection": FULL,
                                "direction": direction, "columns": columns}]}

    def evaluation(checkpoint: str, count: int, key: str) -> tuple[str, dict[str, Any]]:
        return "wse_submit_evaluation", {"experiment_id": eid, "checkpoint_id": checkpoint, "suite_id": "practice",
                                         "eval_config": {"sample_count": count, "seed": 0}, "idempotency_key": key}

    _, job = yield evaluation(base, 4, "pass")
    yield "job", job["job_id"]
    _, done = yield "wse_get_job", {"experiment_id": eid, "job_id": job["job_id"]}
    _, report = yield "wse_get_report", {"experiment_id": eid, "report_id": done["report_id"]}
    samples = [sample for problem in report["problems"] for sample in problem["samples"]]
    groups = {"a": [k for k, s in enumerate(samples) if s["limit_hit"]], "b": [k for k, s in enumerate(samples)
                                                                              if s["finished"]]}
    _, probe = yield "wse_probe", {"experiment_id": eid, "checkpoint_id": base, "sites": SITES, "groups": groups,
                                   "texts": [{"output_uri": s["output_uri"]} for s in samples],
                                   "idempotency_key": "probe"}
    yield "job", probe["job_id"]
    _, probed = yield "wse_get_job", {"experiment_id": eid, "job_id": probe["job_id"]}
    uri = probed["result"]["result_uri"]
    _, analyzed = yield "wse_analyze", {"experiment_id": eid, "checkpoint_id": base, "kind": "direction_projection",
                                        "tensor": f"{DOWN0}.weight", "selection": FULL,
                                        "parameters": {"direction": f"{uri}#/sites/2/contrast/direction",
                                                       "axis": "columns"}}
    yield "wse_analyze", {"experiment_id": eid, "checkpoint_id": base, "kind": "tensor_norms",
                          "tensor": f"{DOWN0}.weight", "selection": FULL, "parameters": {"axis": "rows"}}
    columns = sorted(entry["index"] for entry in analyzed["result"]["top"][:4])
    _, preview = yield "wse_preview_patch", {"experiment_id": eid, "idempotency_key": "project",
                                             "patch_spec": project_out(f"{uri}#/sites/2/contrast/direction",
                                                                       columns)}
    yield "wse_preview_patch", {"experiment_id": eid, "idempotency_key": "fill", "patch_spec": {
        "base_checkpoint_id": base, "hypothesis": "Silencing two inputs changes the outputs.",
        "operations": [{"op": "fill_slice", "tensor": f"{DOWN0}.weight", "selection": _sel((0, 16), (4, 6)),
                        "value": 0.0}]}}
    yield "wse_preview_patch", {"experiment_id": eid, "idempotency_key": "low-rank", "patch_spec": {
        "base_checkpoint_id": base, "hypothesis": "A rank-one change of the projection changes the outputs.",
        "operations": [{"op": "add_low_rank", "tensor": f"{DOWN0}.weight", "selection": _sel((0, 4), (0, 2)),
                        "u": [[0.5, -0.5, 0.25, 1.0]], "v": [[1.0, -1.0]], "alpha": 0.1}]}}
    _, candidate = yield "wse_apply_patch", {"experiment_id": eid, "patch_id": preview["patch_id"],
                                             "idempotency_key": "apply"}
    _, job = yield evaluation(candidate["checkpoint_id"], 4, "candidate-pass")
    yield "job", job["job_id"]
    _, candidate_done = yield "wse_get_job", {"experiment_id": eid, "job_id": job["job_id"]}
    yield "wse_get_report", {"experiment_id": eid, "report_id": candidate_done["report_id"]}
    yield "wse_compare", {"experiment_id": eid, "base_report_id": done["report_id"],
                          "candidate_report_id": candidate_done["report_id"]}
    _, job = yield evaluation(base, 2, "small-pass")
    yield "job", job["job_id"]
    _, small = yield "wse_get_job", {"experiment_id": eid, "job_id": job["job_id"]}
    yield "wse_inspect_model", {"experiment_id": eid, "checkpoint_id": base}
    # Errors: incomparable reports, final-suite items, unknown outputs and reports, bad parameters and cursors, and
    # a direction of the wrong length.
    yield "wse_compare", {"experiment_id": eid, "base_report_id": done["report_id"],
                          "candidate_report_id": small["report_id"]}
    yield "wse_probe", {"experiment_id": eid, "checkpoint_id": base, "sites": ["residual:1"],
                        "texts": [{"suite_id": "final", "item": 0}, {"text": "abc"}], "idempotency_key": "final"}
    yield "wse_probe", {"experiment_id": eid, "checkpoint_id": base, "sites": ["residual:1"],
                        "texts": [{"output_uri": "wse://analyses/" + "0" * 64}], "idempotency_key": "missing"}
    yield "wse_get_report", {"experiment_id": eid, "report_id": "report_" + "0" * 32}
    yield "wse_analyze", {"experiment_id": eid, "checkpoint_id": base, "kind": "tensor_norms",
                          "tensor": f"{DOWN0}.weight", "selection": FULL, "parameters": {"axis": "diagonal"}}
    yield "wse_inspect_model", {"experiment_id": eid, "checkpoint_id": base, "cursor": "tensors:3"}
    yield "wse_preview_patch", {"experiment_id": eid, "idempotency_key": "wrong-length",
                                "patch_spec": project_out(f"{uri}#/sites/1/contrast/direction", columns)}


def test_mcp_and_python_agree_on_new_tools_errors_and_usage(lm, monkeypatch):
    from wse.runner import ExperimentExecutor
    from wse.strategies.base import Action

    _script(monkeypatch, _loop_or_answer)
    runs = {path: _open_run(lm.workspace, lm.base) for path in ("strategy", "mcp")}
    traces: dict[str, list[tuple[str, str, Any]]] = {"strategy": [], "mcp": []}
    responses: list[tuple[str, Any]] = []

    with _worker(lm.workspace) as worker:
        # The strategy dispatch: the run-bound Engine through ExperimentExecutor, by tool name and arguments.
        run_id, eid = runs["strategy"]
        with Engine(lm.workspace, run_id=run_id) as bound:
            executor = ExperimentExecutor(bound, constraint_status="verified")
            requests, outcome = _requests(eid, lm.base), None
            while True:
                try:
                    tool, arguments = requests.send(outcome)
                except StopIteration:
                    break
                if tool == "job":
                    assert worker.run_once() == arguments
                    outcome = None
                    continue
                observation = executor.execute(Action.call(tool, **arguments))
                outcome = ("ok", observation.result) if observation.ok else ("error", observation.error)
                traces["strategy"].append((tool, *outcome))

        # MCP: a real stdio session with wse serve bound to the other run grant.
        run_id, eid = runs["mcp"]

        async def body():
            async with connect(lm.workspace, run_id, lm.root / "serve.err") as session:
                requests, outcome = _requests(eid, lm.base), None
                while True:
                    try:
                        tool, arguments = requests.send(outcome)
                    except StopIteration:
                        break
                    if tool == "job":
                        assert await anyio.to_thread.run_sync(worker.run_once) == arguments
                        outcome = None
                        continue
                    kind, value, response = await _outcome(session, tool, **arguments)
                    outcome = (kind, value)
                    traces["mcp"].append((tool, kind, value))
                    responses.append((tool, response))

        run(body)

    assert [step[:2] for step in traces["mcp"]] == [step[:2] for step in traces["strategy"]]
    assert [step[1] for step in traces["mcp"]].count("error") == 7
    strategy_names: dict[str, str] = {}
    mcp_names: dict[str, str] = {}
    for (tool, kind, mine), (_, _, theirs) in zip(traces["strategy"], traces["mcp"], strict=True):
        if kind == "ok" and tool in ("wse_preview_patch", "wse_analyze"):
            # A preview or analysis by reference names a run-scoped URI, so the hash of its request (spec_hash,
            # input_hash) differs between the runs; each is its own request's hash, and the results are the same.
            for value in (mine, theirs):
                value.pop("spec_hash" if tool == "wse_preview_patch" else "input_hash")
        assert _normalized(theirs, mcp_names) == _normalized(mine, strategy_names), tool
    errors = {step[0]: step[2]["code"] for step in traces["mcp"] if step[1] == "error"}
    assert errors == {"wse_compare": "INCOMPATIBLE_REPORTS", "wse_probe": "NOT_FOUND", "wse_get_report": "NOT_FOUND",
                      "wse_analyze": "INVALID_ARGUMENT", "wse_inspect_model": "INVALID_ARGUMENT",
                      "wse_preview_patch": "INVALID_PATCH"}
    assert [step[2]["code"] for step in traces["mcp"] if step[:2] == ("wse_probe", "error")] == \
        ["POLICY_DENIED", "NOT_FOUND"]

    # Usage: each MCP result's call usage is its domain object's, and the two ledgers agree but for measured time.
    for tool, response in responses:
        if response.is_error:
            continue
        structured = response.structured_content
        recorded = structured["result"].get("usage") if isinstance(structured["result"], dict) else None
        if structured["usage"] is not None and tool not in ("wse_get_report",):
            assert structured["usage"]["call"] == recorded, tool
    with Engine(lm.workspace) as engine:
        ledgers = {path: {name: value for name, value in _budget(engine, eid).items() if name != "compute_seconds"}
                   for path, (_, eid) in runs.items()}
        assert ledgers["mcp"] == ledgers["strategy"]
        consumed = {name: value[1] for name, value in ledgers["mcp"].items()}
        assert {name: consumed[name] for name in ("generation_passes", "probe_calls", "analysis_calls",
                                                  "patch_previews", "candidate_commits")} == {
            "generation_passes": 3, "probe_calls": 1, "analysis_calls": 2, "patch_previews": 3,
            "candidate_commits": 1}
        assert all(_budget(engine, eid)["compute_seconds"][1] > 0 for _, eid in runs.values())

    # wse_get_job shows an evaluation's report and a probe's result: their IDs, URIs, and summaries.
    jobs = [(response, value) for (tool, kind, value), (_, response) in zip(traces["mcp"], responses, strict=True)
            if tool == "wse_get_job"]
    evaluation, probe = jobs[0], jobs[1]
    assert evaluation[1]["report_id"] and evaluation[1]["result"]["report_id"] == evaluation[1]["report_id"]
    assert evaluation[0].structured_content["uris"] == [f"wse://reports/{evaluation[1]['report_id']}"]
    result_uri = probe[1]["result"]["result_uri"]
    assert probe[1]["kind"] == "probe" and probe[1]["result"]["sites"] and probe[1]["result"]["texts"]
    assert probe[0].structured_content["uris"] == [result_uri] and result_uri in probe[0].structured_content["summary"]
    # Every result stays within the inline-result limit.
    for tool, response in responses:
        if not response.is_error:
            assert _size(response.structured_content["result"]) <= INLINE_RESULT_LIMIT, tool


# ---------------------------------------------------------------------------
# Carried over from M1: the experiment ID of a run-bound server
# ---------------------------------------------------------------------------


def test_a_run_bound_server_defaults_the_experiment_id(trial, tmp_path):
    from wse.runner import ExperimentExecutor
    from wse.strategies.base import Action

    workspace, base = tmp_path / "workspace", trial["task_checkpoint_id"]
    spec = {"base_checkpoint_id": base, "hypothesis": "Halve one hidden unit.",
            "operations": [{"op": "scale_slice", "tensor": "layers.0.weight", "selection": _sel((0, 1), (0, 2)),
                            "factor": 0.5}]}
    seen: dict[str, Any] = {}

    async def body():
        async with connect(workspace, trial["run_id"], trial["log"]) as session:
            tools = {tool.name: tool for tool in (await session.client.list_tools()).tools}
            taking = sorted(name for name, tool in tools.items() if "experiment_id" in tool.input_schema["properties"])
            assert taking == sorted(set(TOOL_NAMES) - {"wse_list_models", "wse_create_experiment"})
            for name in taking:
                schema = tools[name].input_schema
                assert "experiment_id" not in schema.get("required", []), name
                assert "omit" in schema["properties"]["experiment_id"]["description"], name
            # Before the run's experiment exists, an omitted ID fails and says how to create it.
            error = await session.fails("NOT_FOUND", "wse_get_experiment")
            assert "wse_create_experiment" in error["message"]
            eid = (await _create(session, trial))["experiment"]["experiment_id"]
            assert await session.call("wse_get_experiment") == await session.call("wse_get_experiment",
                                                                                     experiment_id=eid)
            # Metered reads and mutations use the run's experiment too: the same key is the same request.
            read = await session.call("wse_read_tensor", checkpoint_id=base, tensor="layers.1.weight", selection=ROW1)
            assert read["usage"]["cumulative"]["tensor_read_calls"]["consumed"] == 1
            first = await session.call("wse_preview_patch", patch_spec=spec, idempotency_key="p")
            again = await session.call("wse_preview_patch", experiment_id=eid, patch_spec=spec, idempotency_key="p")
            assert again["result"] == first["result"] and first["result"]["experiment_id"] == eid
            applied = await session.call("wse_apply_patch", patch_id=first["result"]["patch_id"],
                                         idempotency_key="a")
            selected = await session.call("wse_select_checkpoint", checkpoint_id=applied["result"]["checkpoint_id"],
                                          expected_selected_id=base, idempotency_key="s")
            assert selected["result"]["selected_checkpoint_id"] == applied["result"]["checkpoint_id"]
            # An ID that is given is used as given: any other one fails as an unknown one does.
            await session.fails("NOT_FOUND", "wse_get_experiment", experiment_id="exp_" + "0" * 32)
            other = eid[:-1] + ("1" if eid[-1] == "0" else "0")
            await session.fails("NOT_FOUND", "wse_read_tensor", experiment_id=other, checkpoint_id=base,
                                tensor="layers.1.weight", selection=ROW1)
            seen["eid"] = eid

    run(body)
    # The strategy dispatch defaults it the same way.
    with Engine(workspace, run_id=trial["run_id"]) as bound:
        executor = ExperimentExecutor(bound, constraint_status="unverified")
        omitted = executor.execute(Action.call("wse_get_experiment"))
        given = executor.execute(Action.call("wse_get_experiment", experiment_id=seen["eid"]))
        assert omitted.ok and omitted.result == given.result
        assert bound.resolve_experiment_id(None) == bound.resolve_experiment_id(seen["eid"]) == seen["eid"]
    # An operator's Engine, bound to no run, needs the ID.
    with Engine(workspace) as operator:
        assert "experiment_id" in _expect(ErrorCode.INVALID_ARGUMENT, operator.resolve_experiment_id, None).message


# ---------------------------------------------------------------------------
# Tool discovery and M1 responses
# ---------------------------------------------------------------------------


def test_the_language_model_tools_are_listed_with_their_schemas_and_m1_responses_keep_their_forms(trial, tmp_path,
                                                                                                    engine):
    workspace, base = tmp_path / "workspace", trial["task_checkpoint_id"]
    spec = {"base_checkpoint_id": base, "hypothesis": "Halve one hidden unit.",
            "operations": [{"op": "scale_slice", "tensor": "layers.0.weight", "selection": _sel((0, 1), (0, 2)),
                            "factor": 0.5}]}
    seen: dict[str, Any] = {}

    async def body():
        async with connect(workspace, trial["run_id"], trial["log"]) as session:
            tools = {tool.name: tool for tool in (await session.client.list_tools()).tools}
            assert {"wse_probe", "wse_get_report"} <= set(tools)
            probe = tools["wse_probe"].input_schema
            assert set(probe["required"]) == {"checkpoint_id", "texts", "sites", "idempotency_key"}
            assert set(probe["properties"]) == {"experiment_id", "checkpoint_id", "texts", "sites", "groups", "top",
                                                "idempotency_key"}
            assert len(probe["properties"]["texts"]["items"]["anyOf"]) == 3
            report = tools["wse_get_report"]
            assert report.input_schema["required"] == ["report_id"] and report.annotations.read_only_hint
            assert "wse_submit_evaluation" in report.description and "sample_count" in report.description
            assert "parameters" in tools["wse_analyze"].input_schema["properties"]
            assert "cursor" in tools["wse_inspect_model"].input_schema["properties"]
            # M1 responses keep their forms: whole previews, inspections, and candidates, and M1 comparisons.
            eid = (await _create(session, trial))["experiment"]["experiment_id"]
            preview = (await session.call("wse_preview_patch", experiment_id=eid, patch_spec=spec,
                                          idempotency_key="p"))["result"]
            inspection = (await session.call("wse_inspect_model", experiment_id=eid, checkpoint_id=base))["result"]
            candidate = (await session.call("wse_apply_patch", experiment_id=eid, patch_id=preview["patch_id"],
                                            idempotency_key="a"))["result"]
            seen.update(eid=eid, preview=preview, inspection=inspection, candidate=candidate)

    run(body)
    manifest = engine.load_manifest(base)
    preview, inspection, candidate = seen["preview"], seen["inspection"], seen["candidate"]
    assert "unchanged_tensors" not in preview and len(preview["tensors"]) == len(manifest.tensors)
    assert preview == engine.get_patch(seen["eid"], preview["patch_id"]).model_dump(mode="json")
    assert "tensor_page" not in inspection and inspection["manifest"] == manifest.model_dump(mode="json")
    assert len(inspection["tensor_statistics"]) == len(manifest.tensors)
    assert candidate == engine.load_manifest(candidate["checkpoint_id"]).model_dump(mode="json")
    assert "unchanged_tensors" not in candidate


# ---------------------------------------------------------------------------
# wse_get_report
# ---------------------------------------------------------------------------


def test_get_report_returns_the_whole_report_or_a_bounded_overview_and_withholds_final_reports(lm, monkeypatch):
    from wse.schemas import EvaluationReport, ReportOverview

    _script(monkeypatch, _loop_or_answer)
    with Engine(lm.workspace) as engine:  # 4 problems x 64 samples: too many outputs for one inline result
        engine.register_suite(_suite("many", lm.practice, samples=64, max_new_tokens=4, sample_counts=[4]))
    run_id, eid = _open_run(lm.workspace, lm.base)
    with Engine(lm.workspace, run_id=run_id) as bound:
        jobs = [bound.submit_evaluation(eid, lm.base, suite, {"sample_count": 4, "seed": 0}, suite).job_id
                for suite in ("practice", "many")]
    with _worker(lm.workspace) as worker:
        assert [worker.run_once() for _ in jobs] == jobs
    with Engine(lm.workspace, run_id=run_id) as bound:
        fits, large = (bound.get_report(eid, bound.get_job(eid, job_id).report_id) for job_id in jobs)
        assert bound.get_report(eid, fits.report_id, bounded=True) == fits
        overview = bound.get_report(eid, large.report_id, bounded=True)
        assert isinstance(overview, ReportOverview) and isinstance(overview, EvaluationReport)
        assert overview.omitted == ["problems", "scores", "cluster_ids"]
        assert overview.problems is None and overview.scores is None and overview.cluster_ids is None
        assert overview.report_bytes == _size(large) > overview.limit_bytes == INLINE_RESULT_LIMIT >= _size(overview)
        assert overview.resource_uri == large.report_uri
        shown = {name: value for name, value in overview.model_dump(mode="json").items()
                 if name not in ("omitted", "report_bytes", "limit_bytes", "resource_uri")}
        assert shown == {**large.model_dump(mode="json", exclude={"problems"}), "scores": None, "cluster_ids": None}
        assert len(_samples(large)) == 256 and overview.metrics == large.metrics
    # A final report of an open experiment is withheld from the run (inserted here by the operator, as a test).
    hidden = fits.model_copy(update={"report_id": "report_" + "f" * 32, "split": "final", "job_id": None,
                                     "problems": None, "scores": None, "cluster_ids": None, "target_met": None})
    with Engine(lm.workspace) as engine:
        with engine.store.db.transaction() as conn:
            engine.evaluation.insert_report(conn, hidden)
        assert engine.get_report(eid, hidden.report_id) == hidden

    async def body():
        async with connect(lm.workspace, run_id, lm.root / "serve.err") as session:
            whole = await session.call("wse_get_report", experiment_id=eid, report_id=fits.report_id)
            assert whole["result"] == fits.model_dump(mode="json") and whole["uris"] == [fits.report_uri]
            bounded = await session.raw("wse_get_report", experiment_id=eid, report_id=large.report_id)
            assert not bounded.is_error and _wire_bytes(bounded) <= MAX_RESPONSE_BYTES
            assert bounded.structured_content["result"] == overview.model_dump(mode="json")
            assert large.report_uri in bounded.structured_content["summary"]
            # The paged resource holds the whole report.
            page = await session.read(large.report_uri)
            chunks = [page["chunk"]]
            while page["next_uri"] is not None:
                page = await session.read(page["next_uri"])
                chunks.append(page["chunk"])
            assert EvaluationReport.model_validate_json("".join(chunks)) == large
            missing = await session.fails("NOT_FOUND", "wse_get_report", experiment_id=eid,
                                          report_id="report_" + "0" * 32)
            withheld = await session.fails("NOT_FOUND", "wse_get_report", experiment_id=eid,
                                           report_id=hidden.report_id)
            assert withheld["message"].replace(hidden.report_id, "R") == missing["message"].replace(
                "report_" + "0" * 32, "R")

    run(body)
    # Once the experiment is closed, the run reads its final reports.
    with Engine(lm.workspace) as engine:
        engine.freeze(eid, lm.base)
        final = engine.final_evaluate(eid)
    with Engine(lm.workspace, run_id=run_id) as bound:
        assert bound.get_report(eid, final.candidate.report_id, bounded=True) == final.candidate
        assert bound.get_report(eid, hidden.report_id, bounded=True) == hidden


# ---------------------------------------------------------------------------
# wse_compare of two generation reports
# ---------------------------------------------------------------------------


def test_two_generation_reports_compare_on_accuracy_limit_hits_and_generated_tokens(lm):
    from wse.evaluators.generation import compare_generation
    from wse.schemas import GenerationComparison

    run_id, eid = _open_run(lm.workspace, lm.base)
    with Engine(lm.workspace, run_id=run_id) as bound:
        spec = {"base_checkpoint_id": lm.base, "hypothesis": "Scaling one row changes some samples.",
                "operations": [{"op": "scale_slice", "tensor": f"{DOWN0}.weight", "factor": -3.0,
                                "selection": _sel((0, 1), (0, 32))}]}
        candidate = bound.apply_patch(eid, bound.preview_patch(eid, spec, "p").patch_id, "a").checkpoint_id
        jobs = [bound.submit_evaluation(eid, checkpoint, "practice", {"sample_count": count, "seed": 0},
                                        f"{checkpoint}-{count}").job_id
                for checkpoint, count in ((lm.base, 4), (candidate, 4), (lm.base, 2))]
    with _worker(lm.workspace) as worker:
        assert [worker.run_once() for _ in jobs] == jobs
    with Engine(lm.workspace, run_id=run_id) as bound:
        base, other, small = (bound.get_report(eid, bound.get_job(eid, job_id).report_id) for job_id in jobs)
        comparison = bound.compare(eid, base.report_id, other.report_id)
        assert isinstance(comparison, GenerationComparison)
        direct = compare_generation(base, other)
        assert comparison.model_dump(mode="json", exclude={"experiment_id", "base_report_id", "candidate_report_id",
                                                           "base_checkpoint_id", "candidate_checkpoint_id",
                                                           "management"}) == direct.model_dump(mode="json")
        assert (comparison.experiment_id, comparison.base_report_id, comparison.candidate_report_id,
                comparison.base_checkpoint_id, comparison.candidate_checkpoint_id, comparison.management) == (
            eid, base.report_id, other.report_id, lm.base, candidate, "managed")
        assert set(comparison.differences) == {"accuracy", "limit_hit_rate", "mean_generated_tokens"}
        assert comparison.differences["mean_generated_tokens"].estimate == pytest.approx(
            other.metrics["mean_generated_tokens"] - base.metrics["mean_generated_tokens"])
        assert (comparison.problems, comparison.samples) == (4, 8)
        error = _expect(ErrorCode.INCOMPATIBLE_REPORTS, bound.compare, eid, base.report_id, small.report_id)
        assert "sample_count" in error.details["mismatched"]

    async def body():
        async with connect(lm.workspace, run_id, lm.root / "serve.err") as session:
            response = await session.call("wse_compare", experiment_id=eid, base_report_id=base.report_id,
                                          candidate_report_id=other.report_id)
            assert response["result"] == comparison.model_dump(mode="json")
            for name in ("accuracy", "limit_hit_rate", "mean_generated_tokens"):
                assert name in response["summary"]
            await session.fails("INCOMPATIBLE_REPORTS", "wse_compare", experiment_id=eid,
                                base_report_id=base.report_id, candidate_report_id=small.report_id)

    run(body)


# ---------------------------------------------------------------------------
# Response sizes of open-weight models
# ---------------------------------------------------------------------------


def test_open_weight_previews_inspections_and_candidates_take_bounded_forms_when_whole_ones_do_not_fit(
        hf_snapshots, tmp_path):
    from wse.adapters.families import role_tensor

    workspace = tmp_path / "workspace"
    small = _small_limit_policy("small_limits_v1", 8 * 1024)  # results of at most 7 KiB
    limit = small.operations.limits.max_result_json_bytes
    with Engine(workspace) as engine:
        manifest = engine.import_hf_snapshot(hf_snapshots["qwen3_5"], repo="example-org/Qwen3.5-tiny",
                                             revision=REVISION)
        engine.register_policy(small)
    base, count = manifest.checkpoint_id, len(manifest.tensors)
    down = role_tensor(manifest, "mlp.down", 0)
    spec = {"base_checkpoint_id": base, "hypothesis": "Scaling two rows of the projection changes the outputs.",
            "operations": [{"op": "scale_slice", "tensor": down.tensor, "factor": 0.5,
                            "selection": [{"start": 0, "stop": 2, "step": 1},
                                          {"start": 0, "stop": down.shape[1], "step": 1}]}]}
    # Under qwen_loop_v1 the fixture's 28 tensors fit whole, as M1 models do.
    run_id, eid = _open_run(workspace, base)
    with Engine(workspace, run_id=run_id) as bound:
        whole = bound.preview_patch(eid, spec, "p")
        assert len(whole.tensors) == count and whole.unchanged_tensors is None
        inspection = bound.inspect_model_details(eid, base)
        assert inspection.tensor_page is None and len(inspection.tensor_statistics) == count
        full = bound.apply_patch(eid, whole.patch_id, "a")
        assert bound.apply_patch(eid, whole.patch_id, "a", bounded=True) == full
    print(f"Qwen3.5 fixture ({count} tensors) under qwen_loop_v1: preview {_size(whole)} B, inspection "
          f"{_size(inspection)} B, candidate manifest {_size(full)} B")
    # Under an 8 KiB inline limit they take their bounded forms, each within the limit.
    run_id, eid = _open_run(workspace, base, policy_id=small.policy_id)
    with Engine(workspace, run_id=run_id) as bound:
        preview = bound.preview_patch(eid, spec, "p")
        assert [diff.tensor for diff in preview.tensors] == [down.tensor] and preview.unchanged_tensors == count - 1
        assert preview.tensors == [diff for diff in whole.tensors if diff.changed_elements]
        for name in ("changed_elements", "parameter_count", "l2_delta", "max_abs_delta", "relative_l2_delta",
                     "result_weights_hash", "spec_hash", "all_finite"):
            assert getattr(preview, name) == getattr(whole, name), name
        assert _size(preview) <= limit < _size(whole)
        assert bound.find_patch(preview.patch_id).preview == preview
        # The inspection is paged; the pages cover the manifest's tensors in order, and their statistics are the
        # whole inspection's.
        pages, cursor = [], None
        while True:
            page = bound.inspect_model_details(eid, base, cursor=cursor)
            assert _size(page) <= limit, page.tensor_page
            pages.append(page)
            cursor = page.tensor_page.next_cursor
            if cursor is None:
                break
        assert len(pages) == pages[0].tensor_page.page_count > 1
        assert [info for page in pages for info in page.manifest.tensors] == manifest.tensors
        assert [entry for page in pages for entry in page.tensor_statistics] == inspection.tensor_statistics
        assert [page.tensor_page.start for page in pages] == [0, *(page.tensor_page.stop for page in pages[:-1])]
        assert all(page.tensor_page.tensor_count == count and page.manifest.parameter_count ==
                   manifest.parameter_count for page in pages)
        for cursor in ("tensors:1", "tensors:x", "models:0", 7):
            _expect(ErrorCode.INVALID_ARGUMENT, bound.inspect_model_details, eid, base, cursor=cursor)
        # The candidate's manifest lists the tensor the patch changed and counts the others.
        view = bound.apply_patch(eid, preview.patch_id, "a", bounded=True)
        full = bound.apply_patch(eid, preview.patch_id, "a")
        assert [info.name for info in view.tensors] == [down.tensor] and view.unchanged_tensors == count - 1
        assert view.model_dump(mode="json", exclude={"tensors", "unchanged_tensors"}) == full.model_dump(
            mode="json", exclude={"tensors"})
        assert view.tensors == [info for info in full.tensors if info.name == down.tensor]
        assert _size(view) <= limit < _size(full)
    print(f"under a {limit}-byte limit: preview {_size(preview)} B, {len(pages)} inspection pages of at most "
          f"{max(_size(page) for page in pages)} B, candidate manifest {_size(view)} B")

    # Through MCP the same results arrive, within the limits.
    async def body():
        async with connect(workspace, run_id, tmp_path / "serve.err") as session:
            response = await session.call("wse_inspect_model", experiment_id=eid, checkpoint_id=base)
            assert response["result"]["tensor_page"] == pages[0].tensor_page.model_dump(mode="json")
            assert response["result"]["tensor_statistics"] == [
                entry.model_dump(mode="json") for entry in pages[0].tensor_statistics]
            assert pages[0].tensor_page.next_cursor in response["summary"]
            later = await session.call("wse_inspect_model", experiment_id=eid, checkpoint_id=base,
                                       cursor=pages[0].tensor_page.next_cursor)
            assert later["result"]["manifest"]["tensors"] == [
                info.model_dump(mode="json") for info in pages[1].manifest.tensors]
            shown = await session.call("wse_apply_patch", experiment_id=eid, patch_id=preview.patch_id,
                                       idempotency_key="a")
            assert shown["result"] == view.model_dump(mode="json")
            for result in (response, later, shown):
                assert _size(result["result"]) <= limit

    run(body)


def _response_wire_bytes(response) -> int:
    """Bytes of the JSON-RPC response line that carries a ``ToolResponse`` built by the server."""
    from wse.mcp_server import _respond

    return _wire_bytes(_respond(response))


def test_text_heavy_reports_and_the_largest_probe_summaries_fit_one_response(lm, monkeypatch):
    """Both copies count: the text mirror escapes quotes and backslashes, which tails and summaries are full of."""
    from wse.mcp_server import WSEMCPServer
    from wse.policy import response_size
    from wse.probes import PROBE_JOB_BYTES, fit_summary

    # 32 problems with answers at the graders' 200-character limit and tails of backslashes and newlines (as in
    # test_generation_pass_reports_tail_excerpts_within_the_response_limit): the stored report fits 31 KiB, but
    # its escaped text mirror is larger, so the tool shows every tail shortened to the same bound.
    def text_for(prompt: str) -> str:
        index = int(re.search(r"r(\d+) plus", prompt).group(1))
        return "so \\boxed{" + "9" * 230 + "} " + "\\loop {i}\n".format(i=index) * 40

    _script(monkeypatch, text_for)
    from test_llm_policy_mcp import _gsm_rows, _jsonl

    with Engine(lm.workspace) as engine:
        rows = engine.register_text_dataset(_jsonl(lm.root / "rows.jsonl", _gsm_rows(32, "r")), format="gsm8k",
                                            id_prefix="gsm8k_train_99")
        engine.register_suite(_suite("full", rows, max_new_tokens=1000))
    run_id, eid = _open_run(lm.workspace, lm.base)
    with Engine(lm.workspace, run_id=run_id) as bound:
        job = bound.submit_evaluation(eid, lm.base, "full", {"sample_count": 32, "seed": 0}, "full")
        probe = bound.submit_probe(eid, lm.base, [{"text": "abc def"}, {"text": "ghi"}, {"text": "jkl mno"}],
                                   ["residual:1"], "probe", groups={"a": [0], "b": [1, 2]})
    with _worker(lm.workspace) as worker:
        assert [worker.run_once(), worker.run_once()] == [job.job_id, probe.job_id]
    with Engine(lm.workspace, run_id=run_id) as bound:
        server = WSEMCPServer(bound)
        report = bound.get_report(eid, bound.get_job(eid, job.job_id).report_id)
        assert _size(report) <= INLINE_RESULT_LIMIT < response_size(report.model_dump(mode="json")) / 2
        shown = bound.get_report(eid, report.report_id, bounded=True)
        assert type(shown) is type(report) and response_size(shown.model_dump(mode="json")) <= 2 * INLINE_RESULT_LIMIT
        pairs = [(mine, theirs) for problem, other in zip(shown.problems, report.problems, strict=True)
                 for mine, theirs in zip(problem.samples, other.samples, strict=True)]
        assert all(theirs.tail.endswith(mine.tail) and 0 < len(mine.tail) < len(theirs.tail) for mine, theirs in pairs)
        assert shown.model_dump(mode="json", exclude={"problems"}) == report.model_dump(mode="json",
                                                                                         exclude={"problems"})
        wire = _wire_bytes(server.wse_get_report(experiment_id=eid, report_id=report.report_id))
        assert wire <= MAX_RESPONSE_BYTES
        # A probe summary that fit_summary fills to its allowance, at 1 to 128 sites of 64 texts with groups.
        done = bound.get_job(eid, probe.job_id)
        allowance = INLINE_RESULT_LIMIT - PROBE_JOB_BYTES
        rng = np.random.default_rng(0)
        texts = [{"source": "output", "forward_tokens": 1024, "window_start": 3, "window_stop": 1024,
                  "dropped_tokens": 4321, "mean_logprob": -1.2345678912345}] * 64
        largest = 0
        for count in (1, 16, 32, 80, 128):
            sites = [{"site": f"model.language_model.layers.{i % 24}.mlp.down_proj:input{i}", "features": 3584,
                      "group_mean_norms": {"a": 12.3456789, "b": 98.7654321},
                      "contrast": {"norm": 3.14159265, "standardized": rng.normal(size=3584).tolist()}}
                     for i in range(count)]
            document = {"checkpoint_id": lm.base, "tokens_per_text": 1024, "texts": texts, "sites": sites,
                        "groups": {"a": list(range(32)), "b": list(range(32, 64))}}
            summary = fit_summary(document, uri="wse://analyses/" + "a" * 64, size=987654321, top=64,
                                  limit=allowance)
            response = server._job_response(eid, done.model_copy(update={"result": summary}))
            largest = max(largest, _response_wire_bytes(response))
        assert largest <= MAX_RESPONSE_BYTES
    print(f"report: {_size(report)} B stored, {wire} B on the wire with shortened tails; largest probe job response "
          f"{largest} B")


def test_an_sdk_agent_under_qwen_loop_v1_sees_every_permitted_tool_and_may_omit_the_experiment_id(lm, monkeypatch):
    """Through the OpenAI Agents SDK's real Runner and wse serve; only the model is a test double."""
    pytest.importorskip("agents", reason="the optional agents-openai extra is not installed")
    from test_agent_adapter import TESTS_DIR, ToolModel
    from wse.agent_adapters.openai_agents import OpenAIAgentsAdapter
    from wse.runner import ReferenceRunner, allowed_tools

    monkeypatch.setenv("OPENAI_AGENTS_DISABLE_TRACING", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from wse.schemas import TaskDefinition

    with Engine(lm.workspace) as engine:
        engine.register_task(TaskDefinition.model_validate({
            "task_id": "loops_v1", "title": "Finish answers", "description": "Make the model finish its answers.",
            "objective": {"task_id": "loops_v1", "description": "Fewer outputs at the token limit.",
                          "gates": [{"metric": "accuracy", "min_value": 0.5}]},
            "development_suite_id": "practice", "development_examples": 4}))
        grant = engine.grant_run("loops_v1", QWEN_LOOP_POLICY_ID, lm.base)
        policy = engine.get_policy(QWEN_LOOP_POLICY_ID)
    made: dict[str, Any] = {}

    def factory(config, eid):
        made["model"] = ToolModel([("call", "wse_get_experiment", {}), ("say", "done")])
        return OpenAIAgentsAdapter(model=made["model"])

    runner = ReferenceRunner(lm.workspace, worker_args=["--poll-seconds", "0.02", "--device", "cpu"],
                             serve_env={"PYTHONPATH": str(TESTS_DIR)}, adapter_factory=factory,
                             settle_timeout_seconds=60)
    eid = runner.run_experiment(grant.run_id, "loops_v1", QWEN_LOOP_POLICY_ID, {"kind": "agent", "model": "test-double"})
    result = runner.last_result["result"]
    assert result["status"] == "completed", result
    seen = result["provider_metadata"]["tools_seen"]
    assert seen == sorted(allowed_tools(policy)) and {"wse_probe", "wse_get_report"} <= set(seen)
    assert sorted(tool["name"] for tool in made["model"].requests[0]["tools"]) == seen
    assert made["model"].outputs[1]["result"]["experiment"]["experiment_id"] == eid  # the ID was omitted
