"""Candidates from tensors computed in the sandbox: tensor files, ``replace_tensor@1``, and ``submit_candidate``.

A ``.safetensors`` file that the run's sandbox saves to ``outputs/`` is
stored as a tensor file of the run (``wse://tensors/<sha256>``);
``Engine.submit_candidate`` turns its tensors into a candidate of a parent
checkpoint through ``replace_tensor@1``, after checking the file's
ownership, every tensor's name, shape, stored dtype, and finiteness, and the
protected tensors. The candidate is an exact patch whose record names the
file and the sandbox call that saved it, and the export replays it.
"""

from __future__ import annotations

import json

import pytest
import torch
from conftest import HF_SKIP_REASON
from test_llm_policy_mcp import _open_run, lm  # noqa: F401 - a fixture
from test_probe_jobs import DOWN0
from test_sandbox_tool import _sandbox_generation_policy

from wse import sandbox
from wse.api import Engine
from wse.audit import export_experiment, replay_export
from wse.schemas import ErrorCode, WSEError

AVAILABLE, REASON = sandbox.available()
pytestmark = pytest.mark.skipif(not AVAILABLE, reason=REASON)
TARGET = f"{DOWN0}.weight"

SAVE = ("import torch\nfrom safetensors.torch import load_file, save_file\n"
        "weights = load_file('inputs/weights/base.safetensors')\n"
        "new = weights[{name!r}].clone()\n"
        "{change}\n"
        "save_file({{{name!r}: new}}, 'outputs/candidate.safetensors')\n")


@pytest.fixture
def run(lm, monkeypatch):
    """A language-model run under a sandbox policy that permits replace_tensor: (run ID, experiment ID)."""
    pytest.importorskip("transformers", reason=HF_SKIP_REASON)
    return _open_run(lm.workspace, lm.base, _sandbox_generation_policy("loop_sandbox_submit"))


def _saved(lm, run, change: str, name: str = TARGET, key: str = "save"):
    """Run code that saves one changed tensor; return its stored tensor file's output entry."""
    with Engine(lm.workspace, run_id=run[0]) as bound:
        result = bound.run_python(run[1], SAVE.format(name=name, change=change), key)
    assert result.ending == "completed", result.stderr
    [output] = result.outputs
    return result, output


def _rejected(lm, run, tensor_file, *, parent=None, key="submit", **options) -> WSEError:
    with Engine(lm.workspace, run_id=run[0]) as bound, pytest.raises(WSEError) as caught:
        bound.submit_candidate(run[1], parent or lm.base, tensor_file, "Halve one row.", key, **options)
    return caught.value


def test_a_tensor_saved_in_the_sandbox_becomes_an_exact_candidate(lm, run) -> None:
    result, output = _saved(lm, run, "new[0] = new[0] * 0.5")
    assert output.tensor_file.startswith("wse://tensors/") and output.tensor_count == 1
    assert [(t.name, t.dtype) for t in output.tensors] == [(TARGET, "bfloat16")]
    with Engine(lm.workspace, run_id=run[0]) as bound:
        submission = bound.submit_candidate(run[1], lm.base, output.tensor_file, "Halve one row.", "submit")
        again = bound.submit_candidate(run[1], lm.base, output.tensor_file, "Halve one row.", "submit")
        budget = bound.get_experiment(run[1]).budget
        patch = bound.find_patch(submission.patch_id)
    assert again == submission
    assert (budget["patch_previews"].consumed, budget["candidate_commits"].consumed) == (1, 1)
    assert submission.parent_checkpoint_id == lm.base and submission.changed_elements > 0
    assert submission.source["call"] == result.call and submission.source["sha256"] == output.sha256
    assert submission.source["code_sha256"] == result.code_sha256 and submission.source["path"] == "candidate.safetensors"
    [operation] = patch.spec.operations
    assert (operation.op, str(operation.operator), operation.tensor) == ("replace_tensor", "replace_tensor@1", TARGET)
    assert operation.parameters["source"]["reference"] == f"{output.tensor_file}#{TARGET}"
    with Engine(lm.workspace) as engine:
        base = engine.open_checkpoint(lm.base).tensor(TARGET)
        candidate = engine.open_checkpoint(submission.checkpoint_id).tensor(TARGET)
        events = [e for e in engine.audit_log(run[1]) if e["event_type"] == "candidate.submitted"]
    expected = base.clone()
    expected[0] = expected[0] * 0.5
    assert torch.equal(candidate, expected) and candidate.dtype == torch.bfloat16
    [event] = events
    assert event["payload"]["source"]["call"] == result.call and event["payload"]["tensors"] == [TARGET]


@pytest.mark.parametrize(("change", "code", "fragment"), [
    ("new = new.float()", ErrorCode.INVALID_PATCH, "stored as bfloat16"),
    ("new = new[:, :4].contiguous()", ErrorCode.INVALID_PATCH, "has shape"),
    ("new[0, 0] = float('nan')", ErrorCode.INVALID_PATCH, "nonfinite"),
    ("pass", ErrorCode.INVALID_PATCH, "equal the parent"),
])
def test_a_submission_is_checked_before_a_candidate_exists(lm, run, change, code, fragment) -> None:
    _, output = _saved(lm, run, change)
    error = _rejected(lm, run, output.tensor_file)
    assert error.code == code and fragment in error.message, error.message
    with Engine(lm.workspace) as engine:
        assert [c.role for c in engine.list_experiment_checkpoints(run[1])] == ["base"]


def test_unknown_and_protected_tensors_are_refused(lm, run) -> None:
    _, unknown = _saved(lm, run, "pass", name=TARGET)
    error = _rejected(lm, run, unknown.tensor_file, tensors=["model.layers.0.mlp.missing.weight"])
    assert error.code == ErrorCode.INVALID_ARGUMENT
    # The base checkpoint's protected tensors are not even exported to the sandbox, so the code saves its own.
    code = ("import torch\nfrom safetensors.torch import save_file\n"
            "save_file({'model.embed_tokens.weight': torch.zeros(" + ", ".join(
                str(size) for size in _shape(lm, "model.embed_tokens.weight")) + ", dtype=torch.bfloat16)}, "
            "'outputs/embed.safetensors')\n")
    with Engine(lm.workspace, run_id=run[0]) as bound:
        [output] = bound.run_python(run[1], code, "embed").outputs
    error = _rejected(lm, run, output.tensor_file, key="embed-submit")
    assert error.code == ErrorCode.POLICY_DENIED, error.message


def _shape(lm, name: str) -> list[int]:
    with Engine(lm.workspace) as engine:
        return next(info.shape for info in engine.load_manifest(lm.base).tensors if info.name == name)


def test_tensor_files_of_another_experiment_and_other_files_are_not_submittable(lm, run) -> None:
    _, output = _saved(lm, run, "new[0] = new[0] * 0.5")
    other = _open_run(lm.workspace, lm.base, _sandbox_generation_policy("loop_sandbox_other"))
    error = _rejected(lm, other, output.tensor_file)
    assert error.code == ErrorCode.NOT_FOUND
    code = ("import os, torch\nfrom safetensors.torch import save_file\n"
            "save_file({'ids': torch.arange(4)}, 'outputs/ids.safetensors')\n"
            "os.symlink(os.path.abspath('inputs/weights/base.safetensors'), 'outputs/link.safetensors')\n")
    with Engine(lm.workspace, run_id=run[0]) as bound:
        result = bound.run_python(run[1], code, "others")
    [ids] = result.outputs
    assert ids.tensor_file is None and "floating-point" in ids.note
    assert result.refused_outputs == ["link.safetensors"]


def test_a_submitted_candidate_exports_and_replays_byte_identically(lm, run, tmp_path) -> None:
    _, output = _saved(lm, run, "new[1] = -new[1]")
    with Engine(lm.workspace, run_id=run[0]) as bound:
        submission = bound.submit_candidate(run[1], lm.base, output.tensor_file, "Flip one row.", "submit")
    with Engine(lm.workspace) as engine:
        manifest = export_experiment(engine, run[1], tmp_path / "export")
    [listed] = manifest.sandbox["tensor_files"]
    assert listed["uri"] == output.tensor_file and listed["tensors"][0]["name"] == TARGET
    with Engine(tmp_path / "clean") as clean:
        replayed = replay_export(tmp_path / "export", clean)
    assert replayed.all_identical and [c["checkpoint_id"] for c in replayed.candidates] == [submission.checkpoint_id]
    written = json.loads((tmp_path / "export" / "export_manifest.json").read_text(encoding="utf-8"))
    assert written["sandbox"]["tensor_files"][0]["call"] == listed["call"]
