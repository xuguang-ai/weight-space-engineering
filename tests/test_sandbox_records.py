"""What a sandboxed run leaves on record: the export, the inputs, and the instructions (M3 Task 7).

``export_experiment`` writes every sandbox call of the run (its code, its
standard output and error as stored, its result summary, and the output
files it kept), the inputs manifest, and each stored sandbox array that an
edit may name, each with its hash; ``replay_export`` checks those hashes and
still rebuilds every candidate from the exact patches alone. The
inputs of a run on a language model hold only the base checkpoint's
permitted tensors and the run's development outputs, and the instructions
describe the sandbox without naming any answer.
"""

from __future__ import annotations

import hashlib
import json
import re

import pytest
import torch
from conftest import HF_SKIP_REASON
from test_llm_policy_mcp import MATH_ROWS, lm  # noqa: F401 - a fixture
from test_mcp_e2e import FAULT_DESCRIPTIONS, FINAL_SEED
from test_sandbox_policy import SANDBOX_POLICY_ID, _small_sandbox_policy
from test_sandbox_tool import _sandbox_generation_policy

from wse import sandbox
from wse.api import Engine
from wse.audit import export_experiment, replay_export
from wse.policy import QWEN_LOOP_SANDBOX_POLICY_ID, builtin_experiment_policies
from wse.runner import SANDBOX_TIME_LABEL, allowed_tools, render_instructions
from wse.schemas import TaskDefinition

AVAILABLE, REASON = sandbox.available()
needs_sandbox = pytest.mark.skipif(not AVAILABLE, reason=REASON)

ROWS = ("import json\nimport numpy as np\nfrom safetensors.numpy import load_file\n"
        "w = load_file('inputs/weights/base.safetensors')['layers.1.weight']\n"
        "np.save('outputs/row.npy', -w[0])\n"
        "open('outputs/note.txt', 'w').write('rows compared')\n"
        "print(json.dumps({'identical': bool((w[0] == w[1]).all()), 'values': (-w[0]).tolist()}))\n")


def _rows_run(workspace, signed_sum, tmp_path):
    """A signed-sum run under a sandbox policy with two sandbox calls and one candidate built from the first's output."""
    with Engine(workspace) as engine:
        ids = signed_sum.setup_workspace(engine, tmp_path / "bundles", final_seed=FINAL_SEED)
        engine.register_policy(_small_sandbox_policy())
        grant = engine.grant_run("signed_sum_v1", SANDBOX_POLICY_ID, ids["task_checkpoint_id"])
    base = ids["task_checkpoint_id"]
    with Engine(workspace, run_id=grant.run_id) as bound:
        eid = bound.create_experiment(grant.run_id, "signed_sum_v1", SANDBOX_POLICY_ID, "create").experiment_id
        first = bound.run_python(eid, ROWS, "rows")
        second = bound.run_python(eid, "import os\nprint(sorted(os.listdir('outputs')))\n"
                                       "os.symlink('/etc/hosts', 'outputs/hosts')\n", "link")
        values = json.loads(first.stdout)["values"]
        preview = bound.preview_patch(eid, {
            "base_checkpoint_id": base, "hypothesis": "Give the second unit of the last layer its own weights.",
            "operations": [{"op": "set_slice", "tensor": "layers.1.weight",
                            "selection": [{"start": 1, "stop": 2, "step": 1}, {"start": 0, "stop": 4, "step": 1}],
                            "values": [values]}]}, "preview")
        candidate = bound.apply_patch(eid, preview.patch_id, "apply")
    return eid, base, (first, second), candidate


@needs_sandbox
def test_the_export_holds_every_sandbox_call_and_still_replays(workspace, signed_sum, tmp_path) -> None:
    eid, base, (first, second), candidate = _rows_run(workspace, signed_sum, tmp_path)
    out = tmp_path / "export"
    with Engine(workspace) as engine:
        manifest = export_experiment(engine, eid, out)
    record = manifest.sandbox
    assert [call["call"] for call in record["calls"]] == [first.call, second.call] == [1, 2]
    for call, code in zip(record["calls"], (ROWS, None)):
        for name in ("code.py", "stdout.txt", "stderr.txt", "result.json"):
            relative = call[name]
            assert manifest.files[relative] == "sha256:" + hashlib.sha256((out / relative).read_bytes()).hexdigest()
        if code is not None:
            assert (out / call["code.py"]).read_text(encoding="utf-8") == code
    assert (out / record["calls"][0]["stdout.txt"]).read_text(encoding="utf-8") == first.stdout
    summary = json.loads((out / record["calls"][0]["result.json"]).read_text(encoding="utf-8"))
    assert summary["code_sha256"] == first.code_sha256 and summary["ending"] == "exited"

    # The stored array an edit may name, byte for byte: its file hashes to the digest in its URI.
    [array] = record["arrays"]
    pointer = next(o.pointer for o in first.outputs if o.path == "row.npy")
    assert array["uri"] == pointer.partition("#")[0] and array["call"] == first.call and array["shape"] == [4]
    digest = array["uri"].rsplit("/", 1)[1]
    assert hashlib.sha256((out / array["file"]).read_bytes()).hexdigest() == digest

    # Each call keeps its own output files; outputs/ starts empty, and the link the code made is never followed.
    assert record["calls"][0]["outputs"] == {"note.txt": "sandbox/calls/1/outputs/note.txt",
                                             "row.npy": "sandbox/calls/1/outputs/row.npy"}
    assert (out / record["calls"][0]["outputs"]["note.txt"]).read_text() == "rows compared"
    assert second.stdout.strip() == "[]" and second.refused_outputs == ["hosts"]
    assert record["calls"][1]["outputs"] == {} and not [path for path in manifest.files if "hosts" in path]
    inputs = json.loads((out / record["inputs_manifest"]).read_text(encoding="utf-8"))
    assert set(inputs["files"]) == {"weights/base.safetensors"}

    assert manifest.costs["sandbox"] == {"calls": 2, "wall_seconds": pytest.approx(
        first.usage.sandbox_seconds + second.usage.sandbox_seconds), "label": SANDBOX_TIME_LABEL}
    with Engine(tmp_path / "replay") as clean:
        replayed = replay_export(out, clean)
    assert replayed.all_identical and [c["checkpoint_id"] for c in replayed.candidates] == [candidate.checkpoint_id]


def test_an_export_without_a_sandbox_has_no_sandbox_record(workspace, signed_sum, tmp_path) -> None:
    from wse.policy import SMALL_MODEL_POLICY_ID

    with Engine(workspace) as engine:
        ids = signed_sum.setup_workspace(engine, tmp_path / "bundles", final_seed=FINAL_SEED)
        grant = engine.grant_run("signed_sum_v1", SMALL_MODEL_POLICY_ID, ids["task_checkpoint_id"])
    with Engine(workspace, run_id=grant.run_id) as bound:
        eid = bound.create_experiment(grant.run_id, "signed_sum_v1", SMALL_MODEL_POLICY_ID, "create").experiment_id
    with Engine(workspace) as engine:
        manifest = export_experiment(engine, eid, tmp_path / "export")
    assert manifest.sandbox is None and "sandbox" not in manifest.costs
    written = json.loads((tmp_path / "export" / "export_manifest.json").read_text(encoding="utf-8"))
    assert "sandbox" not in written and not any(path.startswith("sandbox/") for path in written["files"])


@needs_sandbox
def test_qwen_sandbox_inputs_hold_no_published_edit_or_final_item(lm, monkeypatch) -> None:
    pytest.importorskip("transformers", reason=HF_SKIP_REASON)
    from test_llm_policy_mcp import _open_run
    from test_probe_jobs import DOWN0, _pass

    from wse.sandbox_runs import layout_for

    # The workspace also holds an edit made in another run (a published edit) that this run was not granted.
    target = f"{DOWN0}.weight"
    other_run, other_eid = _open_run(lm.workspace, lm.base)
    with Engine(lm.workspace, run_id=other_run) as bound:
        preview = bound.preview_patch(other_eid, {
            "base_checkpoint_id": lm.base, "hypothesis": "An edit made in another run.",
            "operations": [{"op": "fill_slice", "tensor": target, "value": 1234.5,
                            "selection": [{"start": 0, "stop": 1, "step": 1}, {"start": 0, "stop": 1, "step": 1}]}]},
            "preview")
        published = bound.apply_patch(other_eid, preview.patch_id, "apply").checkpoint_id
    run_id, eid, report = _pass(lm, monkeypatch, policy=_sandbox_generation_policy("loop_sandbox_leaks"))
    with Engine(lm.workspace, run_id=run_id) as bound:
        result = bound.run_python(eid, "import os\nprint(sorted(os.listdir('inputs')))\n", "list")
    assert result.exit_code == 0, result.stderr
    with Engine(lm.workspace) as engine:
        layout = layout_for(engine.experiments, run_id)
        base_hash = engine.load_manifest(lm.base).weights_hash
        handle = engine.open_checkpoint(lm.base)
        files = sorted(str(p.relative_to(layout.inputs)) for p in layout.inputs.rglob("*") if p.is_file())
        assert files == ["manifest.json", f"passes/{report.report_id}.jsonl", "weights/base.safetensors"]
        from safetensors import safe_open

        with safe_open(str(layout.inputs / "weights" / "base.safetensors"), framework="pt") as weights:
            assert weights.metadata() == {"checkpoint_id": lm.base, "weights_hash": base_hash}
            for name in weights.keys():
                assert torch.equal(weights.get_tensor(name), handle.tensor(name)), name
            assert float(weights.get_tensor(target)[0, 0]) != 1234.5
        assert published not in "".join(p.read_bytes().decode("latin-1") for p in layout.inputs.rglob("*")
                                         if p.is_file())
    text = "\n".join(p.read_text(encoding="utf-8") for p in layout.inputs.rglob("*") if p.suffix in (".json", ".jsonl"))
    for row in MATH_ROWS:
        assert row["problem"] not in text and row["unique_id"] not in text
    assert "final" not in json.dumps(json.loads((layout.inputs / "manifest.json").read_text())["files"])


def test_instructions_describe_the_sandbox_and_name_no_answer(signed_sum) -> None:
    policies = builtin_experiment_policies()
    signed = TaskDefinition.model_validate_json(signed_sum.TASK_FILE.read_text(encoding="utf-8"))
    loops = TaskDefinition.model_validate({
        "task_id": "loops_v1", "title": "Finish answers", "description": "Make the model finish its answers.",
        "objective": {"task_id": "loops_v1", "description": "Fewer outputs at the token limit.",
                      "gates": [{"metric": "accuracy", "min_delta": 0.0}]},
        "development_suite_id": "practice", "development_examples": 4})
    for task, policy in ((loops, policies[QWEN_LOOP_SANDBOX_POLICY_ID]), (signed, _small_sandbox_policy())):
        text = render_instructions(task, policy, experiment_id="exp_0123", tools=allowed_tools(policy))
        assert "wse_run_python" in text and "unverified" in text
        hits = sorted({match.group(0) for match in FAULT_DESCRIPTIONS.finditer(text)})
        assert hits == [], (policy.policy_id, hits)
        assert not re.search(r"plan|build_fixture|\.py\b|docs/|p50c", text), policy.policy_id
