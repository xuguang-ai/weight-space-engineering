"""MCP integration through the actual protocol (Task 6).

Every test starts ``wse serve`` as a stdio subprocess and talks to it with an
MCP client, never calling services directly: the official SDK client
(``mcp.Client``) for the protocol tests, and OpenAI Agents SDK's
``MCPServerStdio`` for the actual-host connection test. Operator setup
(importing models, registering the task, policies, and suites, granting
runs, freezing, final evaluation) happens outside the server, as it does in
a trial. Every server and worker process is stopped when its client context
exits, including on failure.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import jsonschema
import numpy as np
import pytest
import torch
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import MCPError

from wse.api import Engine
from wse.cli import main as cli_main
from wse.policy import ExperimentPolicy

TESTS_DIR = Path(__file__).resolve().parent
FINAL_SEED = 5_550_199_223_310_017
TOOL_NAMES = {
    "wse_list_models", "wse_create_experiment", "wse_get_experiment", "wse_inspect_model", "wse_read_tensor",
    "wse_analyze", "wse_preview_patch", "wse_apply_patch", "wse_submit_evaluation", "wse_get_job",
    "wse_cancel_job", "wse_compare", "wse_select_checkpoint",
    "wse_probe", "wse_get_report",  # M2: language-model probes and whole reports (M2 plan Section 5.6)
    "wse_run_python",  # M3: code in the sandbox, offered under a policy that grants it (M3 plan Section 2)
    "wse_submit_candidate",  # tensors the sandbox computed, as a candidate (replace_tensor@1)
}
MUTATIONS = {"wse_create_experiment", "wse_preview_patch", "wse_apply_patch", "wse_submit_evaluation",
             "wse_cancel_job", "wse_select_checkpoint", "wse_probe", "wse_run_python", "wse_submit_candidate"}
RESOURCE_TEMPLATES = {"wse://models/{checkpoint_id}/manifest{?page}", "wse://experiments/{experiment_id}/policy{?page}",
                      "wse://reports/{report_id}{?page}", "wse://patches/{patch_id}{?page}",
                      "wse://analyses/{digest}{?page}"}
ROW1 = [{"start": 1, "stop": 2, "step": 1}, {"start": 0, "stop": 4, "step": 1}]
ORACLE = {"op": "scale_slice", "tensor": "layers.1.weight", "selection": ROW1, "factor": -1.0}
DEV = {"sample_count": 64, "seed": 0}
WIDE_CONFIG = {"layer_sizes": [64, 64, 64, 2], "activation": "relu", "bias": True}
MAX_RESPONSE_BYTES = 64 * 1024


class ToolFailure(Exception):
    """A tool call that returned an MCP tool error; ``error`` is the parsed domain error, if any."""

    def __init__(self, name: str, text: str) -> None:
        super().__init__(text)
        self.name, self.text = name, text
        prefix = f"Error executing tool {name}: "
        body = text[len(prefix):] if text.startswith(prefix) else ""
        try:
            self.error = json.loads(body)["error"]
        except (ValueError, KeyError, TypeError):
            self.error = None

    @property
    def code(self) -> str | None:
        return self.error["code"] if self.error else None


class Session:
    """A connected MCP client plus a transcript of everything the server returned."""

    def __init__(self, client: Client) -> None:
        self.client = client
        self.transcript: list[str] = []
        self.results: list[Any] = []

    async def raw(self, name: str, **arguments: Any):
        result = await self.client.call_tool(name, arguments)
        self.results.append(result)
        self.transcript.extend(block.text for block in result.content if block.type == "text")
        return result

    async def call(self, name: str, **arguments: Any) -> dict[str, Any]:
        """Call a tool; return its structured content, which must equal its text mirror."""
        result = await self.raw(name, **arguments)
        if result.is_error:
            raise ToolFailure(name, result.content[0].text)
        assert json.loads(result.content[0].text) == result.structured_content
        return result.structured_content

    async def fails(self, code: str, name: str, **arguments: Any) -> dict[str, Any]:
        with pytest.raises(ToolFailure) as excinfo:
            await self.call(name, **arguments)
        assert excinfo.value.code == code, excinfo.value.text
        return excinfo.value.error

    async def read(self, uri: str) -> dict[str, Any]:
        contents = (await self.client.read_resource(uri)).contents
        assert len(contents) == 1 and contents[0].mime_type == "application/json"
        self.transcript.append(contents[0].text)
        return json.loads(contents[0].text)

    async def read_fails(self, uri: str) -> MCPError:
        with pytest.raises(MCPError) as excinfo:
            await self.client.read_resource(uri)
        self.transcript.append(str(excinfo.value))
        return excinfo.value


def _serve_args(workspace: Path, run_id: str, *, worker: bool, extra: tuple[str, ...] = ()) -> list[str]:
    args = ["-m", "wse.cli", "--workspace", str(workspace), "serve", "--run-id", run_id, *extra]
    if worker:
        return [*args, "--setup", "wse_test_plugins:setup", "--worker-args", "--poll-seconds 0.02"]
    return [*args, "--no-worker"]


def _env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(TESTS_DIR), "OPENAI_AGENTS_DISABLE_TRACING": "1"}


@asynccontextmanager
async def connect(workspace: Path, run_id: str, log: Path, *, worker: bool = False, extra: tuple[str, ...] = ()):
    """Start ``wse serve`` over stdio and connect the official SDK client; stop both on exit."""
    params = StdioServerParameters(command=sys.executable, args=_serve_args(workspace, run_id, worker=worker,
                                                                           extra=extra), env=_env())
    with log.open("a", encoding="utf-8") as errlog:
        async with Client(stdio_client(params, errlog=errlog)) as client:
            yield Session(client)


def run(body) -> None:
    anyio.run(body)


@pytest.fixture
def plugins():
    import wse_test_plugins

    return wse_test_plugins


@pytest.fixture
def trial(engine, signed_sum, tmp_path, plugins):
    """An operator-prepared signed-sum workspace and one run grant."""
    plugins.setup(engine.registry)
    ids = signed_sum.setup_workspace(engine, tmp_path / "bundles", final_seed=FINAL_SEED, with_reference=True)
    grant = engine.grant_run(ids["task_id"], ids["policy_id"], ids["task_checkpoint_id"])
    return {**ids, "run_id": grant.run_id, "log": tmp_path / "serve.err"}


async def _create(session: Session, trial: dict[str, Any], key: str = "create") -> dict[str, Any]:
    return (await session.call("wse_create_experiment", run_id=trial["run_id"], task_id=trial["task_id"],
                               policy_id=trial["policy_id"], idempotency_key=key))["result"]


def _spec(trial, operations, hypothesis="One output row has the wrong sign."):
    return {"base_checkpoint_id": trial["task_checkpoint_id"], "hypothesis": hypothesis, "operations": operations}


async def _evaluate(session: Session, eid: str, checkpoint_id: str, key: str, suite_id: str = "signed_sum_dev_v1",
                    config: dict[str, int] | None = None) -> dict[str, Any]:
    job = (await session.call("wse_submit_evaluation", experiment_id=eid, checkpoint_id=checkpoint_id,
                              suite_id=suite_id, eval_config=config or DEV, idempotency_key=key))["result"]
    for _ in range(6):
        done = (await session.call("wse_get_job", experiment_id=eid, job_id=job["job_id"], wait_seconds=10))
        if done["result"]["state"] not in ("queued", "running"):
            return done["result"]
    raise AssertionError(f"job {job['job_id']} did not finish")


def _consumed(budget: dict[str, Any]) -> dict[str, Any]:
    return {name: entry["consumed"] for name, entry in budget.items()}


# -- discovery and schemas -------------------------------------------------------------------------------


def test_discovery_lists_every_tool_and_resource_with_schemas(trial, tmp_path):
    async def body():
        async with connect(tmp_path / "workspace", trial["run_id"], trial["log"]) as session:
            client = session.client
            assert client.protocol_version == "2026-07-28"  # self-contained requests, no initialize handshake
            assert trial["run_id"] in client.instructions
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            assert set(tools) == TOOL_NAMES
            for name, tool in tools.items():
                assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)
                assert tool.description and tool.input_schema["type"] == "object"
                assert tool.output_schema is not None and tool.output_schema["type"] == "object"
                assert set(tool.output_schema["required"]) == {"summary", "uris", "usage", "result"}
                jsonschema.Draft202012Validator.check_schema(tool.input_schema)
                jsonschema.Draft202012Validator.check_schema(tool.output_schema)
                assert ("idempotency_key" in tool.input_schema.get("required", [])) == (name in MUTATIONS)
                assert tool.annotations.read_only_hint == (name in {"wse_list_models", "wse_get_experiment",
                                                                    "wse_get_job", "wse_compare", "wse_get_report"})
            patch_schema = tools["wse_preview_patch"].input_schema["properties"]["patch_spec"]
            assert set(patch_schema["required"]) == {"base_checkpoint_id", "hypothesis", "operations"}
            assert "$ref" not in json.dumps(tools["wse_read_tensor"].input_schema)
            templates = (await client.list_resource_templates()).resource_templates
            assert {t.uri_template for t in templates} == RESOURCE_TEMPLATES
            assert all(t.mime_type == "application/json" for t in templates)
            # Every successful result validates against its tool's published output schema.
            created = await session.call("wse_create_experiment", run_id=trial["run_id"], task_id=trial["task_id"],
                                         policy_id=trial["policy_id"], idempotency_key="c1")
            jsonschema.validate(created, tools["wse_create_experiment"].output_schema)
            listed = await session.call("wse_list_models")
            jsonschema.validate(listed, tools["wse_list_models"].output_schema)
            # Arguments that fail the input schema never reach the services.
            bad = await session.raw("wse_read_tensor", experiment_id=created["result"]["experiment"]["experiment_id"])
            assert bad.is_error and "validation error" in bad.content[0].text
            unknown = await session.raw("wse_delete_everything")
            assert unknown.is_error and "wse_delete_everything" in unknown.content[0].text
        log = trial["log"].read_text(encoding="utf-8")
        assert f"wse serve: run {trial['run_id']} bound" in log  # logs go to stderr, never stdout

    run(body)


def test_an_unknown_run_grant_fails_on_stderr_before_serving(trial, tmp_path):
    process = subprocess.run([sys.executable, *_serve_args(tmp_path / "workspace", "run_" + "0" * 32, worker=False)],
                             env=_env(), capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert process.returncode == 2 and process.stdout == ""
    assert json.loads(process.stderr.strip().splitlines()[-1])["error"]["code"] == "NOT_FOUND"


# -- the complete oracle repair sequence ------------------------------------------------------------------


def test_scripted_client_repairs_rolls_back_and_survives_a_reconnect(engine, trial, tmp_path):
    workspace, base = tmp_path / "workspace", trial["task_checkpoint_id"]
    base_hash = engine.load_manifest(base).weights_hash
    state: dict[str, Any] = {}

    async def first_connection():
        async with connect(workspace, trial["run_id"], trial["log"], worker=True) as session:
            created = await _create(session, trial)
            eid = created["experiment"]["experiment_id"]
            state["eid"] = eid
            assert created["experiment"]["state"] == "OPEN" and created["experiment"]["selected_checkpoint_id"] == base
            assert created["constraint_status"] == "unverified" and created["evaluation_available"] is True
            assert [s["suite_id"] for s in created["suites"]] == ["signed_sum_dev_v1"]
            gate = created["task"]["objective"]["gates"][0]
            assert (gate["metric"], gate["min_value"]) == ("accuracy", 1.0) and created["task"]["target_accuracy"] == 1
            # Retrying, even with another key, returns the same experiment and never resets the budget.
            again = await _create(session, trial, key="create-again")
            assert again["experiment"]["experiment_id"] == eid
            models = (await session.call("wse_list_models"))["result"]["models"]
            assert [m["checkpoint_id"] for m in models] == [base]
            inspection = (await session.call("wse_inspect_model", experiment_id=eid, checkpoint_id=base))["result"]
            assert [s["tensor"] for s in inspection["tensor_statistics"]] == [t["name"] for t in
                                                                              inspection["manifest"]["tensors"]]
            read = await session.call("wse_read_tensor", experiment_id=eid, checkpoint_id=base,
                                      tensor="layers.1.weight", selection=[ROW1[0], ROW1[1]])
            assert read["result"]["values"] == [-1.0, 1.0, -1.0, 1.0]
            assert read["usage"]["cumulative"]["tensor_read_calls"]["consumed"] == 1
            stats = await session.call("wse_analyze", experiment_id=eid, checkpoint_id=base, kind="stats",
                                       tensor="layers.1.weight", selection=[{"start": 0, "stop": 2, "step": 1},
                                                                            {"start": 0, "stop": 4, "step": 1}])
            assert stats["result"]["result"]["count"] == 8
            assert stats["usage"]["cumulative"]["analysis_calls"]["consumed"] == 1
            # Preview and apply the oracle; duplicates create nothing and charge nothing.
            preview = (await session.call("wse_preview_patch", experiment_id=eid,
                                          patch_spec=_spec(trial, [ORACLE]), idempotency_key="p1"))["result"]
            duplicate = (await session.call("wse_preview_patch", experiment_id=eid,
                                            patch_spec=_spec(trial, [ORACLE]), idempotency_key="p1"))["result"]
            assert duplicate == preview and preview["changed_elements"] == 4
            await session.fails("IDEMPOTENCY_CONFLICT", "wse_preview_patch", experiment_id=eid,
                                patch_spec=_spec(trial, [ORACLE], "Another hypothesis."), idempotency_key="p1")
            applied = await session.call("wse_apply_patch", experiment_id=eid, patch_id=preview["patch_id"],
                                         idempotency_key="a1")
            candidate = applied["result"]["checkpoint_id"]
            again = await session.call("wse_apply_patch", experiment_id=eid, patch_id=preview["patch_id"],
                                       idempotency_key="a1")
            assert again["result"]["checkpoint_id"] == candidate
            assert again["usage"]["cumulative"]["candidate_commits"]["consumed"] == 1
            # Evaluate base and candidate on the development suite and compare them.
            before = await _evaluate(session, eid, base, "e1")
            after = await _evaluate(session, eid, candidate, "e2")
            assert before["state"] == after["state"] == "succeeded"
            assert (before["result"]["metrics"], after["result"]["metrics"]) == ({"accuracy": 0.5}, {"accuracy": 1.0})
            assert after["result"]["target_met"] is True
            repeated = (await session.call("wse_submit_evaluation", experiment_id=eid, checkpoint_id=candidate,
                                           suite_id="signed_sum_dev_v1", eval_config=DEV, idempotency_key="e2"))
            assert repeated["result"]["job_id"] == after["job_id"]
            assert repeated["usage"]["cumulative"]["evaluation_batches"]["consumed"] == 2
            comparison = (await session.call("wse_compare", experiment_id=eid, base_report_id=before["report_id"],
                                             candidate_report_id=after["report_id"]))["result"]
            assert comparison["difference"] == 0.5 and comparison["clusters_improved"] == 32
            # Select by compare-and-swap; a stale expected ID is a conflict; selecting the base rolls back.
            await session.fails("VERSION_CONFLICT", "wse_select_checkpoint", experiment_id=eid,
                                checkpoint_id=candidate, expected_selected_id=candidate, idempotency_key="s0")
            selected = await session.call("wse_select_checkpoint", experiment_id=eid, checkpoint_id=candidate,
                                          expected_selected_id=base, idempotency_key="s1")
            assert selected["result"]["selected_checkpoint_id"] == candidate
            rolled_back = await session.call("wse_select_checkpoint", experiment_id=eid, checkpoint_id=base,
                                             expected_selected_id=candidate, idempotency_key="s2")
            assert rolled_back["result"]["selected_checkpoint_id"] == base
            restored = await _evaluate(session, eid, base, "e3")
            assert restored["result"]["metrics"] == {"accuracy": 0.5}
            # Resources serve the same objects as the tools.
            report = await session.read(f"wse://reports/{after['report_id']}")
            assert report["content"]["metrics"] == {"accuracy": 1.0} and report["page_count"] == 1
            patch = await session.read(preview["patch_uri"])
            assert patch["content"]["preview"] == preview
            assert patch["content"]["result_weights_hash"] == preview["result_weights_hash"]
            assert all("values" not in step for step in patch["content"]["steps"])  # after-values stay operator-only
            state.update(candidate=candidate, budget=(await session.call("wse_get_experiment",
                                                                         experiment_id=eid))["result"])

    async def second_connection():
        async with connect(workspace, trial["run_id"], trial["log"], worker=True) as session:
            eid = state["eid"]
            described = (await session.call("wse_get_experiment", experiment_id=eid))["result"]
            assert described["experiment"] == state["budget"]["experiment"]  # state lives in WSE, not the connection
            assert described["experiment"]["selected_checkpoint_id"] == base
            assert [c["checkpoint_id"] for c in described["checkpoints"]] == [base, state["candidate"]]
            recreated = await _create(session, trial, key="after-reconnect")
            assert recreated["experiment"] == described["experiment"]
            models = (await session.call("wse_list_models"))["result"]["models"]
            assert {m["checkpoint_id"] for m in models} == {base, state["candidate"]}

    run(first_connection)
    run(second_connection)
    assert engine.load_manifest(base).weights_hash == base_hash  # the original checkpoint is unchanged
    assert engine.store.load_tensors(base)["layers.1.weight"][1].tolist() == [-1.0, 1.0, -1.0, 1.0]


def test_cancellation_and_budget_exhaustion_through_mcp(engine, trial, tmp_path, signed_sum):
    tight = ExperimentPolicy.model_validate({"policy_id": "small_model_tight_v1", "mode": "bounded_eval",
                                             "budget": {"candidate_commits": 1, "tensor_read_calls": 2,
                                                        "evaluation_batches": 2}})
    engine.register_policy(tight)
    grant = engine.grant_run(trial["task_id"], tight.policy_id, trial["task_checkpoint_id"])
    marker = tmp_path / "first-batch"
    engine.register_suite({"suite_id": "signed_sum_dev_slow", "task_id": trial["task_id"], "split": "development",
                           "evaluator": "slow_batches@1", "sample_counts": [64], "seeds": [0],
                           "parameters": {"sleep_seconds": 30, "cooperative": True, "marker": str(marker)},
                           "data": signed_sum.suite_data(32, signed_sum.DEV_SEED)})
    base = trial["task_checkpoint_id"]

    async def body():
        async with connect(tmp_path / "workspace", grant.run_id, trial["log"], worker=True) as session:
            eid = (await session.call("wse_create_experiment", run_id=grant.run_id, task_id=trial["task_id"],
                                      policy_id=tight.policy_id, idempotency_key="c"))["result"]["experiment"][
                "experiment_id"]
            slow = {"experiment_id": eid, "checkpoint_id": base, "suite_id": "signed_sum_dev_slow", "eval_config": DEV}
            running = (await session.call("wse_submit_evaluation", **slow, idempotency_key="slow-1"))["result"]
            queued = (await session.call("wse_submit_evaluation", **slow, idempotency_key="slow-2"))["result"]
            with anyio.fail_after(30):
                while not marker.exists():
                    await anyio.sleep(0.02)
            # Cancelling a queued job releases its whole reservation.
            cancelled = (await session.call("wse_cancel_job", experiment_id=eid, job_id=queued["job_id"],
                                            idempotency_key="x2"))["result"]
            assert cancelled["state"] == "cancelled" and cancelled["usage"]["evaluation_batches"] == 0
            # Cancelling a running job stops it; the batch it started stays charged.
            await session.call("wse_cancel_job", experiment_id=eid, job_id=running["job_id"], idempotency_key="x1")
            stopped = (await session.call("wse_get_job", experiment_id=eid, job_id=running["job_id"],
                                          wait_seconds=10))
            assert stopped["result"]["state"] == "cancelled" and stopped["result"]["usage"]["evaluation_batches"] == 1
            assert stopped["usage"]["cumulative"]["evaluation_batches"]["consumed"] == 1
            again = await session.call("wse_cancel_job", experiment_id=eid, job_id=running["job_id"],
                                       idempotency_key="x1")
            assert again["usage"]["cumulative"]["evaluation_batches"]["consumed"] == 1  # nothing refunded
            # The last batch of the budget runs; the next request is refused with an actionable error.
            last = await _evaluate(session, eid, base, "e1")
            assert last["state"] == "succeeded"
            error = await session.fails("BUDGET_EXCEEDED", "wse_submit_evaluation", experiment_id=eid,
                                        checkpoint_id=base, suite_id="signed_sum_dev_v1", eval_config=DEV,
                                        idempotency_key="e2")
            assert "evaluation_batches" in error["message"]
            selection = [{"start": 0, "stop": 1, "step": 1}, {"start": 0, "stop": 2, "step": 1}]
            for _ in range(2):
                await session.call("wse_read_tensor", experiment_id=eid, checkpoint_id=base, tensor="layers.0.weight",
                                   selection=selection)
            await session.fails("BUDGET_EXCEEDED", "wse_read_tensor", experiment_id=eid, checkpoint_id=base,
                                tensor="layers.0.weight", selection=selection)
            first = (await session.call("wse_preview_patch", experiment_id=eid, patch_spec=_spec(trial, [ORACLE]),
                                        idempotency_key="p1"))["result"]
            other = (await session.call("wse_preview_patch", experiment_id=eid,
                                        patch_spec=_spec(trial, [{**ORACLE, "factor": 0.5}]),
                                        idempotency_key="p2"))["result"]
            await session.call("wse_apply_patch", experiment_id=eid, patch_id=first["patch_id"], idempotency_key="a1")
            await session.fails("BUDGET_EXCEEDED", "wse_apply_patch", experiment_id=eid, patch_id=other["patch_id"],
                                idempotency_key="a2")
            final = (await session.call("wse_get_experiment", experiment_id=eid))["result"]
            assert _consumed(final["experiment"]["budget"]) | {"compute_seconds": 0} == {
                "analysis_calls": 0, "candidate_commits": 1, "compute_seconds": 0, "evaluated_examples": 128,
                "evaluation_batches": 2, "model_forward_calls": 2, "patch_previews": 2, "tensor_read_calls": 2,
                "generation_passes": 0, "generated_tokens": 0, "probe_calls": 0, "teacher_forced_tokens": 0,
                "sandbox_calls": 0, "sandbox_seconds": 0}
            assert [c["role"] for c in final["checkpoints"]] == ["base", "candidate"]  # no partial candidate

    run(body)


# -- object-access boundaries and final-set isolation ----------------------------------------------------


def test_tools_and_resources_enforce_the_same_object_boundaries(engine, trial, tmp_path, signed_sum):
    # Another run's experiment, candidate, patch, and report, created by the operator's Python API.
    other_grant = engine.grant_run(trial["task_id"], trial["policy_id"], trial["task_checkpoint_id"])
    other = engine.create_experiment(other_grant.run_id, trial["task_id"], trial["policy_id"], "other")
    other_eid = other.experiment_id
    other_patch = engine.preview_patch(other_eid, _spec(trial, [ORACLE]), "p")
    other_candidate = engine.apply_patch(other_eid, other_patch.patch_id, "a").checkpoint_id
    job = engine.submit_evaluation(other_eid, other_candidate, "signed_sum_dev_v1", DEV, "e")
    from wse.worker import Worker

    with Worker(tmp_path / "workspace") as worker:
        worker.run_once()
    other_report = engine.get_job(other_eid, job.job_id).report_id
    reference = trial["reference_checkpoint_id"]

    async def body():
        async with connect(tmp_path / "workspace", trial["run_id"], trial["log"]) as session:
            eid = (await _create(session, trial))["experiment"]["experiment_id"]
            own_base = trial["task_checkpoint_id"]
            # Tools: foreign and operator-only objects fail exactly like objects that do not exist.
            missing = "ckpt_" + "0" * 32
            for checkpoint in (other_candidate, reference, missing):
                await session.fails("NOT_FOUND", "wse_inspect_model", experiment_id=eid, checkpoint_id=checkpoint)
                await session.fails("NOT_FOUND", "wse_select_checkpoint", experiment_id=eid,
                                    checkpoint_id=checkpoint, expected_selected_id=own_base, idempotency_key="s")
            await session.fails("NOT_FOUND", "wse_get_experiment", experiment_id=other_eid)
            await session.fails("NOT_FOUND", "wse_apply_patch", experiment_id=eid, patch_id=other_patch.patch_id,
                                idempotency_key="a")
            await session.fails("NOT_FOUND", "wse_get_job", experiment_id=other_eid, job_id=job.job_id)
            await session.fails("NOT_FOUND", "wse_compare", experiment_id=eid, base_report_id=other_report,
                                candidate_report_id=other_report)
            await session.fails("POLICY_DENIED", "wse_create_experiment", run_id=other_grant.run_id,
                                task_id=trial["task_id"], policy_id=trial["policy_id"], idempotency_key="c")
            listed = [m["checkpoint_id"] for m in (await session.call("wse_list_models"))["result"]["models"]]
            assert listed == [own_base]
            # Resources: the same boundaries, as JSON-RPC invalid-params errors carrying the domain error.
            for uri in (f"wse://models/{other_candidate}/manifest", f"wse://models/{reference}/manifest",
                        f"wse://models/{missing}/manifest", f"wse://experiments/{other_eid}/policy",
                        f"wse://patches/{other_patch.patch_id}", f"wse://reports/{other_report}",
                        "wse://analyses/" + "0" * 64):
                error = await session.read_fails(uri)
                assert error.error.code == -32602
                assert json.loads(error.error.message)["error"]["code"] == "NOT_FOUND"
            # Own objects read identically through tools and resources.
            manifest = await session.read(f"wse://models/{own_base}/manifest")
            inspected = (await session.call("wse_inspect_model", experiment_id=eid, checkpoint_id=own_base))
            assert manifest["content"] == inspected["result"]["manifest"]
            policy = await session.read(f"wse://experiments/{eid}/policy")
            assert policy["content"] == engine.get_policy(trial["policy_id"]).model_dump(mode="json")
            bad_page = await session.read_fails(f"wse://experiments/{eid}/policy?page=1")
            assert json.loads(bad_page.error.message)["error"]["code"] == "INVALID_ARGUMENT"
            # Nothing of the other run or the reference reached this client.
            text = "\n".join(session.transcript)
            for secret in (engine.load_manifest(reference).weights_hash, engine.load_manifest(other_candidate)
                           .weights_hash):
                assert secret not in text

    run(body)


def _final_secrets(engine, trial) -> list[str]:
    final = engine.evaluation.final_suite(trial["task_id"])
    data = engine.datasets.load(final.dataset)
    return [final.data_hash, str(FINAL_SEED), json.dumps(data.inputs[0][0])]


def _cli(capsys, workspace: Path, *args: str) -> str:
    """Run an operator CLI command in process (the same ``wse`` entry point) and return its stdout."""
    capsys.readouterr()
    assert cli_main(["--workspace", str(workspace), *args]) == 0
    return capsys.readouterr().out


def test_the_final_set_stays_isolated_through_tools_and_resources(engine, trial, tmp_path, capsys):
    workspace = tmp_path / "workspace"
    secrets = _final_secrets(engine, trial)

    async def body():
        # The runner keeps the agent's server connected while the operator freezes and final-evaluates.
        async with connect(workspace, trial["run_id"], trial["log"], worker=True) as session:
            created = await _create(session, trial)
            eid, base = created["experiment"]["experiment_id"], trial["task_checkpoint_id"]
            assert all(s["split"] == "development" for s in created["suites"])
            await session.fails("POLICY_DENIED", "wse_submit_evaluation", experiment_id=eid, checkpoint_id=base,
                                suite_id=trial["final_suite_id"], eval_config={"sample_count": 1024, "seed": 0},
                                idempotency_key="f")
            preview = (await session.call("wse_preview_patch", experiment_id=eid, patch_spec=_spec(trial, [ORACLE]),
                                          idempotency_key="p"))["result"]
            candidate = (await session.call("wse_apply_patch", experiment_id=eid, patch_id=preview["patch_id"],
                                            idempotency_key="a"))["result"]["checkpoint_id"]
            done = await _evaluate(session, eid, candidate, "e")
            await session.call("wse_select_checkpoint", experiment_id=eid, checkpoint_id=candidate,
                               expected_selected_id=base, idempotency_key="s")
            await session.read(f"wse://reports/{done['report_id']}")
            _cli(capsys, workspace, "freeze", eid, "--expected-selected-id", candidate)
            frozen = (await session.call("wse_get_experiment", experiment_id=eid))["result"]
            assert frozen["experiment"]["state"] == "FROZEN"
            await session.fails("VERSION_CONFLICT", "wse_select_checkpoint", experiment_id=eid, checkpoint_id=base,
                                expected_selected_id=candidate, idempotency_key="s2")
            await session.fails("VERSION_CONFLICT", "wse_submit_evaluation", experiment_id=eid,
                                checkpoint_id=candidate, suite_id="signed_sum_dev_v1", eval_config=DEV,
                                idempotency_key="e2")
            text = "\n".join(session.transcript)
            assert all(secret not in text for secret in secrets)
            _cli(capsys, workspace, "final-evaluate", eid)
            final = engine.get_final_report(eid)
            # Released only now that the experiment is closed, and as aggregates only.
            report = (await session.read(f"wse://reports/{final.candidate.report_id}"))["content"]
            assert report["split"] == "final" and report["scores"] is None and report["target_met"] is None
            assert str(FINAL_SEED) not in json.dumps(report) and "inputs" not in json.dumps(report)
            closed = (await session.call("wse_get_experiment", experiment_id=eid))["result"]
            assert closed["experiment"]["state"] == "CLOSED"

    run(body)


# -- response size --------------------------------------------------------------------------------------


def _wide_tensors() -> dict[str, torch.Tensor]:
    """``layers.0.weight`` holds 14-character values, ``layers.1.weight`` values of about 1e15 (19-20 characters)."""
    sizes, tensors = WIDE_CONFIG["layer_sizes"], {}
    for index, (fan_in, fan_out) in enumerate(zip(sizes, sizes[1:])):
        fill = _huge if index == 1 else _fourteen_characters
        tensors[f"layers.{index}.weight"] = torch.as_tensor(fill((fan_out, fan_in)), dtype=torch.float32)
        tensors[f"layers.{index}.bias"] = torch.as_tensor(_fourteen_characters((fan_out,)), dtype=torch.float32)
    return tensors


def _fourteen_characters(shape):
    rng = np.random.default_rng(3)
    mantissas = rng.integers(11_000_001, 19_999_999, size=shape)
    text = [f"-{m // 10_000_000}.{m % 10_000_000:07d}e-05" for m in mantissas.reshape(-1)]
    return np.array(text, dtype=np.float32).reshape(shape)


def _huge(shape):
    return (-np.random.default_rng(0).uniform(1e15, 9e15, size=shape)).astype(np.float32)


def _wire_bytes(result) -> int:
    """Bytes of the JSON-RPC response line that carries ``result`` (the stdio framing of the SDK)."""
    message = {"jsonrpc": "2.0", "id": 2 ** 31, "result": result.model_dump(mode="json", by_alias=True,
                                                                           exclude_none=True)}
    return len(json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def test_the_largest_permitted_read_stays_within_64_kib_with_its_text_mirror(engine, tmp_path):
    checkpoint_id = engine.import_model({"adapter": "mlp@1", "config": WIDE_CONFIG}, _wide_tensors()).checkpoint_id
    grant = engine.grant_run("wide_task_v1", "small_model_v1", checkpoint_id)
    full = [{"start": 0, "stop": 32, "step": 1}, {"start": 0, "stop": 64, "step": 1}]

    async def read_largest(session: Session, eid: str, tensor: str) -> tuple[int, int, int]:
        arguments = {"experiment_id": eid, "checkpoint_id": checkpoint_id, "tensor": tensor}
        try:
            await session.call("wse_read_tensor", **arguments, selection=full)
            selection = full
        except ToolFailure as failure:
            assert failure.code == "INVALID_ARGUMENT"
            selection = failure.error["details"]["largest_permitted_selection"]
        result = await session.raw("wse_read_tensor", **arguments, selection=selection)
        assert not result.is_error
        assert json.loads(result.content[0].text) == result.structured_content
        slice_bytes = len(json.dumps(result.structured_content["result"], separators=(",", ":")).encode())
        region = result.structured_content["result"]
        # A large patch resource is served in bounded pages that reassemble into the document.
        values = np.array(region["values"], dtype=np.float32).reshape(region["shape"]) / 2
        preview = (await session.call("wse_preview_patch", experiment_id=eid, idempotency_key=tensor, patch_spec={
            "base_checkpoint_id": checkpoint_id, "hypothesis": "Halve a block of weights.",
            "operations": [{"op": "set_slice", "tensor": tensor, "selection": selection,
                            "values": values.tolist()}]}))["result"]
        first = await session.read(preview["patch_uri"])
        assert first["page_count"] > 1 and "content" not in first
        chunks, page = [first["chunk"]], first
        while page["next_uri"] is not None:
            page = await session.read(page["next_uri"])
            chunks.append(page["chunk"])
            assert len(json.dumps(page).encode()) <= 32 * 1024
        document = json.loads("".join(chunks))
        assert document["preview"] == preview and document["steps"][0]["tensor"] == tensor
        return region["element_count"], slice_bytes, _wire_bytes(result)

    sizes = {}

    async def body():
        async with connect(tmp_path / "workspace", grant.run_id, tmp_path / "serve.err") as session:
            eid = (await session.call("wse_create_experiment", run_id=grant.run_id, task_id="wide_task_v1",
                                      policy_id="small_model_v1", idempotency_key="c"))["result"]["experiment"][
                "experiment_id"]
            for name, tensor in (("fourteen", "layers.0.weight"), ("huge", "layers.1.weight")):
                sizes[name] = await read_largest(session, eid, tensor)

    run(body)
    print(f"largest reads (elements, result bytes, JSON-RPC response bytes): {sizes}")
    elements, slice_bytes, wire = sizes["fourteen"]
    assert elements == 2048 and 29 * 1024 < slice_bytes and wire <= MAX_RESPONSE_BYTES
    elements, slice_bytes, wire = sizes["huge"]
    assert elements < 2048 and slice_bytes > 30 * 1024 and wire <= MAX_RESPONSE_BYTES


# -- the agent-observable surfaces reveal no answer ---------------------------------------------------------


FAULT_DESCRIPTIONS = re.compile(
    r"\bfault|\binject|\boracle|\breference\b|\bclean\b|\bflip|\bnegat|\binvert|build_fixture|\bW[12]\b|\brow 1\b|"
    r"multipl(?:y|ied)[^.]*-1|wrong sign|sign error", re.IGNORECASE)


def _walk(value: Any):
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _is_oracle(node: Any) -> bool:
    return isinstance(node, dict) and node.get("tensor") == "layers.1.weight" and (
        node.get("factor") == -1 or node.get("parameters", {}).get("factor") == -1
        or node.get("values") == [1.0, -1.0, 1.0, -1.0])


def test_tool_names_descriptions_schemas_and_outputs_reveal_no_answer(engine, trial, tmp_path):
    reference = engine.load_manifest(trial["reference_checkpoint_id"])
    secrets = [reference.checkpoint_id, reference.weights_hash, *_final_secrets(engine, trial)]

    async def body():
        async with connect(tmp_path / "workspace", trial["run_id"], trial["log"], worker=True) as session:
            client = session.client
            tools = (await client.list_tools()).tools
            templates = (await client.list_resource_templates()).resource_templates
            surfaces = [client.instructions,
                        *(json.dumps(t.model_dump(mode="json", by_alias=True, exclude_none=True)) for t in tools),
                        *(json.dumps(t.model_dump(mode="json", by_alias=True, exclude_none=True)) for t in templates)]
            # A trial that has not found the answer: everything it can observe on the task model.
            created = await _create(session, trial)
            eid, base = created["experiment"]["experiment_id"], trial["task_checkpoint_id"]
            await session.call("wse_get_experiment", experiment_id=eid)
            await session.call("wse_list_models")
            inspection = (await session.call("wse_inspect_model", experiment_id=eid, checkpoint_id=base))["result"]
            for info in inspection["manifest"]["tensors"]:
                selection = [{"start": 0, "stop": extent, "step": 1} for extent in info["shape"]]
                await session.call("wse_read_tensor", experiment_id=eid, checkpoint_id=base, tensor=info["name"],
                                   selection=selection)
                for kind in ("stats", "norm", "top_k"):
                    await session.call("wse_analyze", experiment_id=eid, checkpoint_id=base, kind=kind,
                                       tensor=info["name"], selection=selection)
            preview = (await session.call("wse_preview_patch", experiment_id=eid, patch_spec=_spec(
                trial, [{"op": "scale_slice", "tensor": "layers.0.weight",
                         "selection": [{"start": 0, "stop": 1, "step": 1}, {"start": 0, "stop": 2, "step": 1}],
                         "factor": 0.5}], "Halve the first hidden unit."), idempotency_key="p"))["result"]
            done = await _evaluate(session, eid, base, "e")
            for uri in (f"wse://models/{base}/manifest", f"wse://experiments/{eid}/policy", preview["patch_uri"],
                        f"wse://reports/{done['report_id']}"):
                await session.read(uri)
            surfaces += session.transcript
        text = "\n".join(surfaces)
        for secret in secrets:
            assert secret not in text
        hits = sorted({match.group(0) for match in FAULT_DESCRIPTIONS.finditer(text)})
        assert hits == [], hits
        for surface in surfaces:
            try:
                parsed = json.loads(surface)
            except ValueError:
                continue
            assert not any(_is_oracle(node) for node in _walk(parsed))
        assert inspection["manifest"]["provenance"] == {"kind": "import", "source": "signed_sum model v1",
                                                        "parent_checkpoint_id": None, "notes": ""}

    run(body)


# -- the same semantics through the managed SDK, the CLI, and MCP ---------------------------------------------


def test_managed_sdk_cli_and_mcp_agree_on_patches_policy_decisions_and_usage(engine, trial, tmp_path, capsys):
    workspace = tmp_path / "workspace"
    base = trial["task_checkpoint_id"]
    bad_patch = _spec(trial, [{"op": "scale_slice", "tensor": "layers.9.weight", "selection": ROW1, "factor": 2.0}])
    out_of_bounds = [{"start": 0, "stop": 3, "step": 1}, {"start": 0, "stop": 4, "step": 1}]
    denied_patch = _spec(trial, [{**ORACLE, "op": "add_low_rank"}])
    selection = [{"start": 0, "stop": 2, "step": 1}, {"start": 0, "stop": 4, "step": 1}]

    # The managed Python SDK, bound to its own run grant.
    sdk_grant = engine.grant_run(trial["task_id"], trial["policy_id"], base)
    sdk_errors = []
    with Engine(workspace, run_id=sdk_grant.run_id) as bound:
        sdk_eid = bound.create_experiment(sdk_grant.run_id, trial["task_id"], trial["policy_id"], "c").experiment_id
        for call in (lambda: bound.preview_patch(sdk_eid, bad_patch, "bad"),
                     lambda: bound.preview_patch(sdk_eid, denied_patch, "denied"),
                     lambda: bound.read_model_tensor(sdk_eid, base, "layers.1.weight", out_of_bounds)):
            try:
                call()
            except Exception as exc:  # noqa: BLE001 - every call must fail with a domain error
                sdk_errors.append((exc.code.value, exc.message))
        bound.read_model_tensor(sdk_eid, base, "layers.1.weight", selection)
        bound.analyze_model(sdk_eid, base, "norm", "layers.1.weight", selection)
        sdk_preview = bound.preview_patch(sdk_eid, _spec(trial, [ORACLE]), "p")
        sdk_candidate = bound.apply_patch(sdk_eid, sdk_preview.patch_id, "a")
        sdk_job = bound.submit_evaluation(sdk_eid, sdk_candidate.checkpoint_id, "signed_sum_dev_v1", DEV, "e")
    from wse.worker import Worker

    with Worker(workspace) as worker:
        worker.run_once()
    sdk_report = engine.get_report(sdk_eid, engine.get_job(sdk_eid, sdk_job.job_id).report_id)

    mcp: dict[str, Any] = {}

    async def body():
        async with connect(workspace, trial["run_id"], trial["log"], worker=True) as session:
            eid = (await _create(session, trial))["experiment"]["experiment_id"]
            errors = [await session.fails("INVALID_PATCH", "wse_preview_patch", experiment_id=eid,
                                          patch_spec=bad_patch, idempotency_key="bad"),
                      await session.fails("POLICY_DENIED", "wse_preview_patch", experiment_id=eid,
                                          patch_spec=denied_patch, idempotency_key="denied"),
                      await session.fails("INVALID_ARGUMENT", "wse_read_tensor", experiment_id=eid,
                                          checkpoint_id=base, tensor="layers.1.weight", selection=out_of_bounds)]
            await session.call("wse_read_tensor", experiment_id=eid, checkpoint_id=base, tensor="layers.1.weight",
                               selection=selection)
            await session.call("wse_analyze", experiment_id=eid, checkpoint_id=base, kind="norm",
                               tensor="layers.1.weight", selection=selection)
            preview = (await session.call("wse_preview_patch", experiment_id=eid, patch_spec=_spec(trial, [ORACLE]),
                                          idempotency_key="p"))["result"]
            candidate = (await session.call("wse_apply_patch", experiment_id=eid, patch_id=preview["patch_id"],
                                            idempotency_key="a"))["result"]
            done = await _evaluate(session, eid, candidate["checkpoint_id"], "e")
            mcp.update(eid=eid, errors=[(e["code"], e["message"]) for e in errors], preview=preview,
                       candidate=candidate, report_id=done["report_id"])

    run(body)
    assert mcp["errors"] == sdk_errors  # identical policy decisions and messages

    def cli_json(*args: str) -> Any:
        return json.loads(_cli(capsys, workspace, *args))

    # Identical patch content: the CLI shows the MCP run's stored patch, byte for byte the SDK run's.
    mcp_patch = cli_json("patch", mcp["eid"], mcp["preview"]["patch_id"])
    assert mcp_patch == engine.read_patch(sdk_eid, sdk_preview.patch_id).model_dump(mode="json")
    assert mcp["preview"]["spec_hash"] == sdk_preview.spec_hash
    assert mcp["preview"]["result_weights_hash"] == sdk_preview.result_weights_hash
    assert mcp["candidate"]["weights_hash"] == sdk_candidate.weights_hash
    # Identical usage accounting (compute seconds are measured time, so only their presence is compared).
    sdk_budget, mcp_budget = (cli_json("experiment", eid)["budget"] for eid in (sdk_eid, mcp["eid"]))
    assert {k: v for k, v in _consumed(mcp_budget).items() if k != "compute_seconds"} == \
        {k: v for k, v in _consumed(sdk_budget).items() if k != "compute_seconds"}
    assert mcp_budget["compute_seconds"]["consumed"] > 0 and sdk_budget["compute_seconds"]["consumed"] > 0
    # Reports bind the same model, data, evaluator, and configuration identities.
    mcp_report = cli_json("report", mcp["eid"], mcp["report_id"])
    for field in ("model", "data_hash", "evaluator", "evaluator_config_hash", "suite_hash", "metrics", "scores"):
        assert mcp_report[field] == sdk_report.model_dump(mode="json")[field], field


# -- an actual MCP host: OpenAI Agents SDK's stdio client -------------------------------------------------------


def test_openai_agents_sdk_stdio_client_connects_and_sees_verbatim_tools(trial, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_AGENTS_DISABLE_TRACING", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agents = pytest.importorskip("agents", reason="the optional agents-openai extra is not installed")
    from agents.mcp import MCPServerStdio

    import asyncio
    from importlib.metadata import version

    params = {"command": sys.executable,
              "args": _serve_args(tmp_path / "workspace", trial["run_id"], worker=False), "env": _env()}
    observed: dict[str, Any] = {}

    async def body():
        server = MCPServerStdio(params, name="wse", client_session_timeout_seconds=60)
        try:
            await server.connect()
            tools = await server.list_tools()
            observed["names"] = [tool.name for tool in tools]
            result = await server.call_tool("wse_list_models", {})
            observed["result"] = result
            observed["protocol_version"] = getattr(server.session, "protocol_version", None)
        finally:
            await server.cleanup()

    asyncio.run(body())
    assert sorted(observed["names"]) == sorted(TOOL_NAMES)  # forwarded verbatim
    result = observed["result"]
    assert not result.is_error and result.structured_content is not None
    assert json.loads(result.content[0].text) == result.structured_content
    assert result.structured_content["result"]["models"][0]["checkpoint_id"] == trial["task_checkpoint_id"]
    print(f"actual host: openai-agents {version('openai-agents')} with mcp {version('mcp')}; negotiated protocol "
          f"{observed['protocol_version']}")
    assert observed["protocol_version"] == "2026-07-28"
    assert agents is not None


# -- module boundaries and legacy hosts ---------------------------------------------------------------------------


def test_only_the_mcp_module_imports_the_mcp_sdk_and_only_the_openai_adapter_imports_an_agent_sdk_lazily():
    root = TESTS_DIR.parent / "src" / "wse"
    sources = sorted(root.rglob("*.py"))
    importers = {path.name for path in sources
                 if re.search(r"^\s*(?:from|import)\s+(?:mcp|mcp_types)\b", path.read_text(encoding="utf-8"), re.M)}
    assert importers == {"mcp_server.py"}
    agent_sdk = r"(?:from|import)\s+(?:agents|openai|claude_agent_sdk)\b"
    importers = {path.relative_to(root).as_posix() for path in sources
                 if re.search(r"^\s*" + agent_sdk, path.read_text(encoding="utf-8"), re.M)}
    assert importers == {"agent_adapters/openai_agents.py"}
    adapter = (root / "agent_adapters" / "openai_agents.py").read_text(encoding="utf-8")
    assert not re.search(r"^" + agent_sdk, adapter, re.M)  # only inside functions: imported lazily
    probe = subprocess.run([sys.executable, "-c", "import sys, wse, wse.mcp_server, wse.runner, wse.cli, "
                            "wse.agent_adapters.openai_agents; print(sorted(m for m in sys.modules "
                            "if m.split('.')[0] in ('agents', 'openai', 'claude_agent_sdk')))"],
                           capture_output=True, text=True, env=_env(), timeout=60)
    assert probe.returncode == 0 and probe.stdout.strip() == "[]", probe.stderr


def test_a_legacy_handshake_client_is_served_by_the_sdk(trial, tmp_path):
    params = StdioServerParameters(command=sys.executable, args=_serve_args(tmp_path / "workspace", trial["run_id"],
                                                                           worker=False), env=_env())

    async def body():
        with trial["log"].open("a", encoding="utf-8") as errlog:
            async with Client(stdio_client(params, errlog=errlog), mode="legacy") as client:
                assert client.protocol_version != "2026-07-28"  # negotiated by the SDK's initialize handshake
                assert {tool.name for tool in (await client.list_tools()).tools} == TOOL_NAMES
                result = await client.call_tool("wse_list_models", {})
                assert not result.is_error and json.loads(result.content[0].text) == result.structured_content
                print(f"legacy client: negotiated protocol {client.protocol_version}")

    run(body)
