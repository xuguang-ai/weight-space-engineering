"""Code execution in a run: inputs, the operation, its records, and sandbox arrays as references (M3 Tasks 3 to 5).

``Engine.run_python`` runs agent-written code in the run's sandbox
(``wse.sandbox``). Before each call WSE exports what the run may read: the
base checkpoint's permitted tensors, the outputs of its development
generation passes, and its probe results. After the call it lists the
files the code wrote to ``outputs/`` and stores each numeric array of one or
two dimensions as a run-scoped result whose JSON pointer is a vector
reference, so an edit can name it. Every call is charged to the sandbox
counters, recorded in the audit log, and idempotent by its key.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import HF_SKIP_REASON
from test_mcp_e2e import FINAL_SEED
from test_sandbox_policy import SANDBOX_POLICY_ID, _small_sandbox_policy
from test_llm_policy_mcp import lm  # noqa: F401 - a fixture

from wse import sandbox
from wse.api import Engine
from wse.policy import SMALL_MODEL_POLICY_ID, BudgetLimits, SandboxPolicy, builtin_experiment_policies
from wse.schemas import ErrorCode, WSEError

AVAILABLE, REASON = sandbox.available()
pytestmark = pytest.mark.skipif(not AVAILABLE, reason=REASON)


@pytest.fixture
def run(workspace, signed_sum, tmp_path):
    """A signed-sum run under a sandbox policy: (run_id, experiment_id, base checkpoint)."""
    with Engine(workspace) as engine:
        ids = signed_sum.setup_workspace(engine, tmp_path / "bundles", final_seed=FINAL_SEED)
        engine.register_policy(_small_sandbox_policy())
        grant = engine.grant_run("signed_sum_v1", SANDBOX_POLICY_ID, ids["task_checkpoint_id"])
    with Engine(workspace, run_id=grant.run_id) as bound:
        eid = bound.create_experiment(grant.run_id, "signed_sum_v1", SANDBOX_POLICY_ID, "create").experiment_id
    return grant.run_id, eid, ids["task_checkpoint_id"]


def _run(workspace, run, code: str, key: str = "c1"):
    run_id, eid, _ = run
    with Engine(workspace, run_id=run_id) as bound:
        return bound.run_python(eid, code, key)


def test_run_python_returns_bounded_output_and_its_output_files(workspace, run) -> None:
    code = "import numpy as np\nprint('hello')\nnp.save('outputs/v.npy', np.arange(4, dtype=np.float32))\n"
    result = _run(workspace, run, code)
    assert (result.ending, result.exit_code, result.stdout) == ("completed", None, "hello\n")
    [output] = result.outputs
    assert (output.path, output.shape, output.dtype) == ("v.npy", [4], "float32")
    assert output.pointer.startswith("wse://analyses/") and output.pointer.endswith("#/values")
    assert result.usage.sandbox_calls == 1 and 0 < result.usage.sandbox_seconds <= 20
    with Engine(workspace, run_id=run[0]) as bound:
        budget = bound.get_experiment(run[1]).budget
        assert budget["sandbox_calls"].consumed == 1
        assert budget["sandbox_seconds"].consumed == pytest.approx(result.usage.sandbox_seconds)
        assert bound.read_analysis_result(output.pointer.partition("#")[0])["values"] == [0.0, 1.0, 2.0, 3.0]


def test_run_python_reads_the_base_checkpoint_weights(workspace, run) -> None:
    code = ("import json\nfrom safetensors.numpy import load_file\n"
            "weights = load_file('inputs/weights/base.safetensors')\n"
            "print(json.dumps({name: weights[name].tolist() for name in sorted(weights)}))\n")
    result = _run(workspace, run, code)
    assert result.ending == "completed", result.stderr
    with Engine(workspace) as engine:
        handle = engine.open_checkpoint(run[2])
        expected = {name: handle.tensor(name).tolist() for name in sorted(handle.tensor_names())}
    assert json.loads(result.stdout) == expected
    assert "weights/base.safetensors" in result.inputs and "manifest.json" in result.inputs


def test_run_python_is_refused_without_a_sandbox_grant(workspace, signed_sum, tmp_path) -> None:
    with Engine(workspace) as engine:
        ids = signed_sum.setup_workspace(engine, tmp_path / "bundles", final_seed=FINAL_SEED)
        grant = engine.grant_run("signed_sum_v1", SMALL_MODEL_POLICY_ID, ids["task_checkpoint_id"])
    with Engine(workspace, run_id=grant.run_id) as bound:
        eid = bound.create_experiment(grant.run_id, "signed_sum_v1", SMALL_MODEL_POLICY_ID, "create").experiment_id
        with pytest.raises(WSEError) as caught:
            bound.run_python(eid, "print(1)\n", "c1")
    assert caught.value.code == ErrorCode.POLICY_DENIED


def test_run_python_retries_return_the_first_result_and_other_code_conflicts(workspace, run) -> None:
    first = _run(workspace, run, "print('once')\n", key="same")
    again = _run(workspace, run, "print('once')\n", key="same")
    assert again == first
    with Engine(workspace, run_id=run[0]) as bound:
        assert bound.get_experiment(run[1]).budget["sandbox_calls"].consumed == 1
        with pytest.raises(WSEError) as caught:
            bound.run_python(run[1], "print('other')\n", "same")
    assert caught.value.code == ErrorCode.IDEMPOTENCY_CONFLICT


def test_run_python_is_recorded_in_the_audit_log(workspace, run) -> None:
    result = _run(workspace, run, "open('outputs/a.txt', 'w').write('a')\n")
    with Engine(workspace) as engine:
        events = [event for event in engine.audit_log(run[1]) if event["event_type"] == "sandbox.ran"]
    [event] = events
    payload = event["payload"]
    assert (payload["call"], payload["code_sha256"], payload["ending"]) == (result.call, result.code_sha256,
                                                                          "completed")
    assert payload["outputs"] == [{"path": "a.txt", "size": 1, "sha256": result.outputs[0].sha256}]


def test_exhausted_sandbox_calls_refuse_further_calls(workspace, signed_sum, tmp_path) -> None:
    small = builtin_experiment_policies()[SMALL_MODEL_POLICY_ID]
    budget = BudgetLimits(**{**small.budget.model_dump(), "sandbox_calls": 1, "sandbox_seconds": 60.0})
    policy = small.model_copy(update={"policy_id": "one_sandbox_call", "budget": budget,
                                      "sandbox": SandboxPolicy(seconds_per_call=20.0)})
    with Engine(workspace) as engine:
        ids = signed_sum.setup_workspace(engine, tmp_path / "bundles", final_seed=FINAL_SEED)
        engine.register_policy(policy)
        grant = engine.grant_run("signed_sum_v1", "one_sandbox_call", ids["task_checkpoint_id"])
    with Engine(workspace, run_id=grant.run_id) as bound:
        eid = bound.create_experiment(grant.run_id, "signed_sum_v1", "one_sandbox_call", "create").experiment_id
        bound.run_python(eid, "print(1)\n", "c1")
        with pytest.raises(WSEError) as caught:
            bound.run_python(eid, "print(2)\n", "c2")
    assert caught.value.code == ErrorCode.BUDGET_EXCEEDED


def test_only_finite_arrays_of_one_or_two_dimensions_get_a_reference(workspace, run) -> None:
    code = ("import numpy as np\n"
            "np.save('outputs/rows.npy', np.ones((2, 3)))\n"
            "np.save('outputs/nan.npy', np.array([1.0, np.nan]))\n"
            "np.save('outputs/cube.npy', np.zeros((2, 2, 2)))\n"
            "np.save('outputs/words.npy', np.array(['a', 'b']))\n")
    result = _run(workspace, run, code)
    references = {output.path: output.pointer for output in result.outputs}
    assert references["rows.npy"].endswith("#/values")
    assert references["nan.npy"] is None and references["cube.npy"] is None and references["words.npy"] is None
    with Engine(workspace, run_id=run[0]) as bound:
        stored = bound.read_analysis_result(references["rows.npy"].partition("#")[0])
    assert stored["values"] == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]] and stored["shape"] == [2, 3]


def test_a_large_result_is_shortened_to_fit_its_response(workspace, run) -> None:
    # Quotes and backslashes are escaped once in the JSON copy and twice in the text mirror.
    from wse.policy import builtin_experiment_policies, response_size
    from wse.sandbox_runs import layout_for

    limit = builtin_experiment_policies()[SMALL_MODEL_POLICY_ID].operations.limits.max_result_json_bytes
    result = _run(workspace, run, "import sys\nprint(chr(34) * 20000)\nsys.stderr.write(chr(92) * 20000)\n")
    assert result.stdout_truncated and result.stderr_truncated
    assert response_size(result.model_dump(mode="json")) <= 2 * limit
    assert result.stdout.startswith(chr(34)) and "20001 bytes in all; the middle is not shown" in result.stdout
    assert (result.stdout_bytes, result.stderr_bytes) == (20001, 20000)
    with Engine(workspace) as engine:
        directory = layout_for(engine.experiments, run[0]).calls / str(result.call)
    assert (directory / "stdout.txt").read_bytes() == b'"' * 20000 + b"\n"


def test_at_most_32_output_files_are_listed_and_stored(workspace, run) -> None:
    code = "import numpy as np\nfor i in range(40):\n    np.save(f'outputs/v{i:02d}.npy', np.arange(3.0) + i)\n"
    result = _run(workspace, run, code)
    assert [o.path for o in result.outputs] == [f"v{i:02d}.npy" for i in range(32)]
    assert result.outputs_not_listed == 8 and all(o.pointer for o in result.outputs)


# ---------------------------------------------------------------------------
# A language model: passes, probes, protected tensors, and a direction by reference
# ---------------------------------------------------------------------------


def _sandbox_generation_policy(policy_id: str = "loop_sandbox_test"):
    from wse.policy import QWEN_LOOP_SANDBOX_POLICY_ID

    loop = builtin_experiment_policies()[QWEN_LOOP_SANDBOX_POLICY_ID]
    return loop.model_copy(update={"policy_id": policy_id, "sandbox": SandboxPolicy(seconds_per_call=60.0)})


def test_passes_probes_and_unprotected_weights_reach_the_inputs(lm, monkeypatch) -> None:
    pytest.importorskip("transformers", reason=HF_SKIP_REASON)
    from test_probe_jobs import DOWN0, _pass, _probe

    run_id, eid, report = _pass(lm, monkeypatch, policy=_sandbox_generation_policy())
    samples = [sample for problem in report.problems for sample in problem.samples]
    texts = [{"output_uri": sample.output_uri} for sample in samples]
    job, _ = _probe(lm, run_id, eid, texts, [f"{DOWN0}:output"], groups={"a": [0, 1], "b": [2, 3]})
    code = ("import json, os\nimport numpy as np\nfrom safetensors import safe_open\n"
            "with safe_open('inputs/weights/base.safetensors', framework='pt') as weights:\n"
            "    names = sorted(weights.keys())\n"
            f"lines = open('inputs/passes/{report.report_id}.jsonl').read().splitlines()\n"
            f"probe = np.load('inputs/probes/{job.job_id}.npz')\n"
            "print(json.dumps({'names': names, 'outputs': [json.loads(l) for l in lines], "
            "'arrays': sorted(probe.files), 'contrast': probe['site0_contrast'].tolist()}))\n")
    with Engine(lm.workspace, run_id=run_id) as bound:
        result = bound.run_python(eid, code, "inputs")
        document = bound.read_analysis_result(job.result["result_uri"])
        texts = [bound.read_analysis_result(sample.output_uri)["text"] for sample in samples]
    assert result.ending == "completed", result.stderr
    seen = json.loads(result.stdout)
    assert "model.embed_tokens.weight" not in seen["names"] and "lm_head.weight" not in seen["names"]
    assert f"{DOWN0}.weight" in seen["names"]
    assert [row["text"] for row in seen["outputs"]] == texts
    assert {"problem", "item_id", "sample", "correct", "finished", "generated_tokens", "text"} <= set(seen["outputs"][0])
    assert "site0_contrast" in seen["arrays"] and "site0_group_mean_a" in seen["arrays"]
    assert np.allclose(seen["contrast"], document["sites"][0]["contrast"]["direction"])
    assert not [name for name in result.inputs if "final" in name]


def test_a_sandbox_direction_is_accepted_by_a_project_out_preview(lm, monkeypatch) -> None:
    pytest.importorskip("transformers", reason=HF_SKIP_REASON)
    from test_llm_policy_mcp import _open_run
    from test_probe_jobs import DOWN0, FULL

    run_id, eid = _open_run(lm.workspace, lm.base, _sandbox_generation_policy("loop_sandbox_ref"))
    code = ("import numpy as np\n"
            "d = np.zeros(16, dtype=np.float32)\nd[3] = 1.0\n"
            "np.save('outputs/d.npy', d)\n")
    with Engine(lm.workspace, run_id=run_id) as bound:
        result = bound.run_python(eid, code, "direction")
        [output] = result.outputs
        preview = bound.preview_patch(eid, {
            "base_checkpoint_id": lm.base, "hypothesis": "Remove one output direction.",
            "operations": [{"op": "project_out", "tensor": f"{DOWN0}.weight", "selection": FULL,
                            "direction": output.pointer, "columns": [0, 1, 2, 3], "alpha": 1.0}]}, "preview")
        document = bound.find_patch(preview.patch_id)
    assert preview.changed_elements > 0
    recorded = document.spec.operations[0].parameters["direction"]
    assert (recorded["reference"], recorded["length"]) == (output.pointer, 16)
