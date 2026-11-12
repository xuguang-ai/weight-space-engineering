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

from wse import sandbox, sandbox_runs
from wse.api import Engine
from wse.audit import export_experiment, replay_export
from wse.schemas import ErrorCode, WSEError

# The writer a submission records when no sandbox call listed exactly the file it used.
NOT_RECORDED_WRITER = ("not recorded: WSE did not record whether the agent's own commands or a sandbox call wrote "
                       "this file")

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


def test_a_file_the_agents_own_code_wrote_is_submitted_by_path(lm, run) -> None:
    from safetensors.torch import save_file

    from wse.sandbox_runs import layout_for

    with Engine(lm.workspace) as engine:
        layout = layout_for(engine.experiments, run[0]).create()
        new = engine.open_checkpoint(lm.base).tensor(TARGET)
    new[0] = new[0] * 0.5
    save_file({TARGET: new}, str(layout.outputs / "mine.safetensors"))  # as a command of the agent's own would
    with Engine(lm.workspace, run_id=run[0]) as bound:
        submission = bound.submit_candidate(run[1], lm.base, None, "Halve one row.", "by-path",
                                            path="outputs/mine.safetensors")
    # No sandbox call listed this file, so WSE cannot tell who wrote it, and says so.
    assert (submission.source["call"], submission.source["written_by"]) == (None, NOT_RECORDED_WRITER)
    assert submission.source["path"] == "outputs/mine.safetensors" and submission.source["code_sha256"] is None
    with Engine(lm.workspace) as engine:
        assert torch.equal(engine.open_checkpoint(submission.checkpoint_id).tensor(TARGET), new)


def test_paths_that_leave_the_run_directory_or_pass_through_links_are_refused(lm, run, tmp_path) -> None:
    import os

    from safetensors.torch import save_file

    from wse.sandbox_runs import layout_for

    with Engine(lm.workspace) as engine:
        layout = layout_for(engine.experiments, run[0]).create()
        tensor = engine.open_checkpoint(lm.base).tensor(TARGET)
    outside = tmp_path / "outside"
    outside.mkdir()
    save_file({TARGET: tensor * 0}, str(outside / "x.safetensors"))
    os.symlink(outside, layout.work / "escape")
    os.symlink(outside / "x.safetensors", layout.outputs / "link.safetensors")
    cases = [("../outside/x.safetensors", ErrorCode.INVALID_ARGUMENT),
             ("inputs/weights/base.safetensors", ErrorCode.INVALID_ARGUMENT),
             (str(outside / "x.safetensors"), ErrorCode.INVALID_ARGUMENT),
             ("work/escape/x.safetensors", ErrorCode.NOT_FOUND),
             ("outputs/link.safetensors", ErrorCode.NOT_FOUND),
             ("outputs/missing.safetensors", ErrorCode.NOT_FOUND)]
    for index, (path, code) in enumerate(cases):
        error = _rejected(lm, run, None, key=f"refused-{index}", path=path)
        assert error.code == code, (path, error.message)
    with Engine(lm.workspace) as engine:
        assert [c.role for c in engine.list_experiment_checkpoints(run[1])] == ["base"]


def _write(lm, run, relative: str, change) -> tuple[str, str]:
    """Write a changed copy of TARGET to ``relative`` in the run's directory, as the agent's own commands would.

    Returns the file's SHA-256 hex digest and the run's directory.
    """
    import hashlib

    from safetensors.torch import save_file

    from wse.sandbox_runs import layout_for

    with Engine(lm.workspace) as engine:
        layout = layout_for(engine.experiments, run[0]).create()
        new = engine.open_checkpoint(lm.base).tensor(TARGET)
    change(new)
    save_file({TARGET: new}, str(layout.root / relative))
    return hashlib.sha256((layout.root / relative).read_bytes()).hexdigest(), str(layout.root)


def _stored(lm, digest: str) -> bool:
    with Engine(lm.workspace) as engine:
        return (engine.store.artifacts_dir / "tensors" / f"{digest}.safetensors").exists()


def test_the_source_comes_from_the_route_the_submission_used(lm, run) -> None:
    import shutil

    from wse.sandbox_runs import layout_for

    code = SAVE.format(name=TARGET, change="new[0] = new[0] * 0.5")
    with Engine(lm.workspace, run_id=run[0]) as bound:
        # Identical bytes, stored first by a path submission (work/ is never listed), then listed by a sandbox call.
        first = bound.run_python(run[1], code.replace("outputs/candidate", "work/shell"), "to-work")
        assert first.ending == "completed" and first.outputs == [], first.stderr
        by_path = bound.submit_candidate(run[1], lm.base, None, "Halve one row.", "by-path",
                                         path="work/shell.safetensors")
        second = bound.run_python(run[1], code, "to-outputs")
        [output] = second.outputs
        by_file = bound.submit_candidate(run[1], lm.base, output.tensor_file, "Halve one row.", "by-file")
        by_listed_path = bound.submit_candidate(run[1], lm.base, None, "Halve one row.", "by-listed-path",
                                                path="outputs/candidate.safetensors")
        # The reverse order: a call stores the bytes first; the agent's commands copy them to work/.
        third = bound.run_python(run[1], SAVE.format(name=TARGET, change="new[1] = new[1] * 0.25"), "call-first")
        [stored] = third.outputs
        layout = layout_for(bound.experiments, run[0])
        shutil.copy(layout.outputs / "candidate.safetensors", layout.work / "copy.safetensors")
        by_copy = bound.submit_candidate(run[1], lm.base, None, "Quarter one row.", "by-copy",
                                         path="work/copy.safetensors")
    assert by_file.source["tensor_file"] == output.tensor_file == by_path.source["tensor_file"]
    assert (by_file.source["call"], by_file.source["code_sha256"], by_file.source["written_by"]) == (
        second.call, second.code_sha256, "a sandbox call")
    assert by_file.source["path"] == "candidate.safetensors"
    assert (by_path.source["call"], by_path.source["code_sha256"], by_path.source["written_by"]) == (
        None, None, NOT_RECORDED_WRITER)
    assert by_path.source["path"] == "work/shell.safetensors"
    assert (by_listed_path.source["call"], by_listed_path.source["code_sha256"]) == (second.call, second.code_sha256)
    assert by_listed_path.source["path"] == "outputs/candidate.safetensors"
    assert by_copy.source["tensor_file"] == stored.tensor_file
    assert (by_copy.source["call"], by_copy.source["code_sha256"], by_copy.source["written_by"]) == (
        None, None, NOT_RECORDED_WRITER)


def test_a_fifo_named_by_path_is_refused_without_blocking_the_server(lm, run) -> None:
    import os
    import threading

    from wse.sandbox_runs import layout_for

    with Engine(lm.workspace) as engine:
        layout = layout_for(engine.experiments, run[0]).create()
    os.mkfifo(layout.work / "cand.safetensors")
    found: dict = {}

    def submit() -> None:
        try:
            with Engine(lm.workspace, run_id=run[0]) as bound:
                bound.submit_candidate(run[1], lm.base, None, "Halve one row.", "fifo", path="work/cand.safetensors")
        except WSEError as exc:
            found["error"] = exc

    thread = threading.Thread(target=submit, daemon=True)
    thread.start()
    thread.join(20)
    blocked = thread.is_alive()
    if blocked:  # release the blocked open so the test can end
        with open(layout.work / "cand.safetensors", "wb"):
            pass
        thread.join(20)
    assert not blocked, "the submission's open of a FIFO blocked"
    assert found["error"].code == ErrorCode.INVALID_ARGUMENT and "not a regular file" in found["error"].message


@pytest.mark.parametrize("path", ["outputs/a\x00b.safetensors", "outputs/\udcff.safetensors",
                                  "work/sub\x00/x.safetensors"])
def test_a_path_that_cannot_name_a_file_is_an_invalid_argument(lm, run, path) -> None:
    error = _rejected(lm, run, None, key="bad-path", path=path)
    assert error.code == ErrorCode.INVALID_ARGUMENT, error.message


def test_a_path_submission_the_budget_cannot_cover_copies_nothing_into_the_store(lm) -> None:
    pytest.importorskip("transformers", reason=HF_SKIP_REASON)
    policy = _sandbox_generation_policy("loop_sandbox_one_commit")
    policy = policy.model_copy(update={"budget": policy.budget.model_copy(update={"candidate_commits": 1})})
    run = _open_run(lm.workspace, lm.base, policy)
    _, output = _saved(lm, run, "new[0] = new[0] * 0.5")
    with Engine(lm.workspace, run_id=run[0]) as bound:
        bound.submit_candidate(run[1], lm.base, output.tensor_file, "Halve one row.", "first")
    digest, _ = _write(lm, run, "outputs/second.safetensors", lambda new: new[2].mul_(0.5))
    error = _rejected(lm, run, None, key="second", path="outputs/second.safetensors")
    assert not _stored(lm, digest), "the file was copied into the store before the budget check"
    assert error.code == ErrorCode.BUDGET_EXCEEDED, error.message


def test_tensor_files_per_experiment_are_capped_before_anything_is_copied(lm, run, monkeypatch) -> None:
    _, first = _saved(lm, run, "new[0] = new[0] * 0.5")
    limit = first.size + first.size // 2  # room for one file of this size, not two
    monkeypatch.setattr(sandbox_runs, "MAX_EXPERIMENT_TENSOR_BYTES", limit, raising=False)
    digest, _ = _write(lm, run, "work/second.safetensors", lambda new: new[2].mul_(0.5))
    error = _rejected(lm, run, None, key="over", path="work/second.safetensors")
    assert not _stored(lm, digest), "the file over the limit was copied into the store"
    assert error.code == ErrorCode.INVALID_ARGUMENT and f"limit of {limit} bytes" in error.message, error.message
    with Engine(lm.workspace, run_id=run[0]) as bound:
        third = bound.run_python(run[1], SAVE.format(name=TARGET, change="new[3] = new[3] * 0.5"), "third")
        again = bound.submit_candidate(run[1], lm.base, first.tensor_file, "Halve one row.", "again")
    [output] = third.outputs
    assert output.tensor_file is None and f"limit of {limit} bytes" in output.note
    assert again.source["tensor_file"] == first.tensor_file  # a file already stored needs no new bytes


def test_the_header_is_parsed_from_the_stored_copy_not_from_the_agents_file(lm, run, monkeypatch) -> None:
    from pathlib import Path

    seen: list = []
    real = sandbox_runs._safetensors_header

    def spy(handle):
        seen.append(getattr(handle, "name", None))
        return real(handle)

    monkeypatch.setattr(sandbox_runs, "_safetensors_header", spy)
    _saved(lm, run, "new[0] = new[0] * 0.5")
    _, sandbox_root = _write(lm, run, "work/mine.safetensors", lambda new: new[1].mul_(0.5))
    with Engine(lm.workspace, run_id=run[0]) as bound:
        bound.submit_candidate(run[1], lm.base, None, "Halve one row.", "mine", path="work/mine.safetensors")
        store = Path(bound.store.artifacts_dir).resolve()
    assert len(seen) == 2, seen
    for name in seen:  # a path in the store (its copy being published, or the stored file), never a descriptor
        assert isinstance(name, str) and Path(name).resolve().is_relative_to(store), name
        assert not Path(name).resolve().is_relative_to(Path(sandbox_root).resolve())
