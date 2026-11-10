"""The high-level entry point ``wse.WSE`` (``wse.highlevel``).

``WSE`` wraps the operator ``Engine`` and the direct-use functions for the
four steps of a weight-space experiment: diagnose (generation and
evaluation), investigate (tensors and probes), intervene (edits and
published patches), and evaluate (paired comparisons). Models are the tiny
offline fixtures of ``tests/hf_fixtures.py``; nothing is downloaded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import HF_SKIP_REASON
from wse.schemas import ErrorCode, WSEError

pytest.importorskip("transformers", reason=HF_SKIP_REASON)

REPO = "example-org/Llama-3.2-1B-Instruct"
REVISION = "0123456789abcdef0123456789abcdef01234567"
DOWN0 = "model.layers.0.mlp.down_proj.weight"  # [16, 32]
# Short problems: the boxed-answer instruction plus 6 new tokens must fit the fixture's 128 positions.
ROWS = [{"problem": f"{a}+{b}?", "solution": "", "answer": str(a + b), "subject": "s", "level": 1,
         "unique_id": f"test/s/{a}{b}.json"} for a, b in ((1, 1), (2, 3), (4, 4), (5, 1), (6, 2), (7, 0))]
EVAL = {"format": "math500", "profile": "qwen35think@1", "grader": "math_boxed@1", "max_new_tokens": 6,
        "device": "cpu", "batch_size": 2}


def _expect(code, func, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        func(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("highlevel")


@pytest.fixture(scope="module")
def llama(hf_snapshots, root):
    from wse import WSE

    model = WSE.from_pretrained(hf_snapshots["llama_untied"], repo=REPO, revision=REVISION,
                                workspace=root / "workspace")
    yield model
    model.close()


@pytest.fixture(scope="module")
def dataset(root) -> Path:
    path = root / "math.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in ROWS), encoding="utf-8")
    return path


# -- loading -------------------------------------------------------------------


def test_wse_is_exported_from_the_package():
    import wse
    from wse.highlevel import WSE

    assert wse.WSE is WSE and "WSE" in wse.__all__


def test_from_pretrained_imports_a_local_snapshot_once(hf_snapshots, root, llama):
    from wse import WSE
    from wse.api import Engine

    assert llama.checkpoint_id is not None and llama.weights_hash.startswith("sha256:")
    with WSE.from_pretrained(hf_snapshots["llama_untied"], repo=REPO, revision=REVISION,
                             workspace=root / "workspace") as again:
        assert again.checkpoint_id == llama.checkpoint_id
    with Engine(root / "workspace") as engine:
        assert [m.checkpoint_id for m in engine.list_models()] == [llama.checkpoint_id]
        assert engine.load_manifest(llama.checkpoint_id).source.repository == REPO


def test_a_local_snapshot_needs_its_source_labels(hf_snapshots, tmp_path):
    from wse import WSE

    error = _expect(ErrorCode.INVALID_ARGUMENT, WSE.from_pretrained, hf_snapshots["llama_untied"],
                    workspace=tmp_path / "ws")
    assert "repo" in str(error) and "revision" in str(error)


def test_a_hub_cache_snapshot_names_its_repository_and_commit(hf_snapshots, tmp_path):
    import shutil

    from wse import WSE

    sha = "fedcba9876543210fedcba9876543210fedcba98"
    snapshot = tmp_path / "hub" / "models--example-org--Tiny-Llama" / "snapshots" / sha
    shutil.copytree(hf_snapshots["llama_untied"], snapshot)
    with WSE.from_pretrained(snapshot, workspace=tmp_path / "ws") as model:
        assert (model.source.repository, model.source.revision) == ("example-org/Tiny-Llama", sha)


def test_hub_ids_download_only_allow_listed_root_files_at_the_resolved_commit(hf_snapshots, tmp_path,
                                                                             monkeypatch):
    import shutil
    from types import SimpleNamespace

    import huggingface_hub

    from wse import WSE

    sha = "00112233445566778899aabbccddeeff00112233"
    cache = tmp_path / "hub" / "models--example-org--Tiny-Llama" / "snapshots" / sha
    listing = sorted(str(p.relative_to(hf_snapshots["llama_untied"]))
                     for p in Path(hf_snapshots["llama_untied"]).rglob("*") if p.is_file())
    calls: dict = {}

    class FakeApi:
        def __init__(self, *args, **kwargs):
            pass

        def model_info(self, repo_id, *, revision=None, **kwargs):
            calls["model_info"] = (repo_id, revision)
            return SimpleNamespace(sha=sha)

        def list_repo_files(self, repo_id, *, revision=None, **kwargs):
            calls["list_repo_files"] = (repo_id, revision)
            return listing

    def fake_download(repo_id, *, revision=None, allow_patterns=None, **kwargs):
        calls["snapshot_download"] = (repo_id, revision, sorted(allow_patterns))
        shutil.copytree(hf_snapshots["llama_untied"], cache)
        return str(cache)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    with WSE.from_pretrained("example-org/Tiny-Llama", revision="main", workspace=tmp_path / "ws") as model:
        assert (model.source.repository, model.source.revision) == ("example-org/Tiny-Llama", sha)
    assert calls["model_info"] == ("example-org/Tiny-Llama", "main")
    assert calls["list_repo_files"] == ("example-org/Tiny-Llama", sha)
    repo_id, revision, patterns = calls["snapshot_download"]
    assert (repo_id, revision) == ("example-org/Tiny-Llama", sha)
    # Only allow-listed files at the root: the decoys (training_args.bin, original/, onnx/) are never fetched.
    assert "config.json" in patterns and "model.safetensors" in patterns
    assert not any(name.endswith((".bin", ".pth", ".onnx")) or "/" in name for name in patterns)


def test_open_reopens_a_stored_checkpoint(llama, root):
    from wse import WSE

    with WSE.open(llama.checkpoint_id, workspace=root / "workspace") as model:
        assert model.weights_hash == llama.weights_hash
    _expect(ErrorCode.NOT_FOUND, WSE.open, "ckpt_" + "0" * 32, workspace=root / "workspace")


def test_the_default_workspace_is_shared_with_the_cli(monkeypatch, tmp_path):
    from wse import cli
    from wse.highlevel import default_workspace

    monkeypatch.delenv("WSE_WORKSPACE", raising=False)
    assert default_workspace() == Path("workspace")
    monkeypatch.setenv("WSE_WORKSPACE", str(tmp_path / "elsewhere"))
    assert default_workspace() == tmp_path / "elsewhere"
    assert cli._parser().parse_args(["experiment", "x"]).workspace == str(tmp_path / "elsewhere")


# -- names and targets -----------------------------------------------------------


def test_short_names_resolve_to_one_tensor(llama):
    assert llama.name("layers.0.mlp.down_proj.weight") == DOWN0
    assert llama.name("layers.0.mlp.down_proj") == DOWN0  # a module name means its weight
    assert llama.name(DOWN0) == DOWN0
    error = _expect(ErrorCode.INVALID_ARGUMENT, llama.name, "mlp.down_proj")
    assert "model.layers.1.mlp.down_proj.weight" in error.details["candidates"]
    error = _expect(ErrorCode.NOT_FOUND, llama.name, "layers.9.mlp.down_proj")
    assert error.details["candidates"]  # close names, to correct a typo
    assert torch.equal(llama.tensor("layers.0.mlp.down_proj"), llama.handle.tensor(DOWN0))
    assert DOWN0 in llama.tensor_names("down_proj") and all("down_proj" in n for n in llama.tensor_names("down_proj"))


@pytest.mark.parametrize(("index", "expected"), [
    ("", [(0, 16), (0, 32)]),
    ("[:, :8]", [(0, 16), (0, 8)]),
    ("[2]", [(2, 3), (0, 32)]),
    ("[-1, 4:]", [(15, 16), (4, 32)]),
    ("[1:3, -4:]", [(1, 3), (28, 32)]),
])
def test_targets_take_numpy_style_indices(llama, index, expected):
    tensor, selection = llama.target("layers.0.mlp.down_proj" + index)
    assert tensor == DOWN0
    assert selection == [{"start": start, "stop": stop, "step": 1} for start, stop in expected]


@pytest.mark.parametrize(("target", "fragment"), [
    ("layers.0.mlp.down_proj[::2]", "step"),
    ("layers.0.mlp.down_proj[1, 2, 3]", "axes"),
    ("layers.0.mlp.down_proj[16]", "outside"),
    ("layers.0.mlp.down_proj[3:3]", "empty"),
    ("layers.0.mlp.down_proj[a]", "index"),
    ("layers.0.mlp.down_proj[:, :8", "bracket"),
])
def test_invalid_targets_are_refused(llama, target, fragment):
    error = _expect(ErrorCode.INVALID_ARGUMENT, llama.target, target)
    assert fragment in str(error)


# -- intervene -------------------------------------------------------------------


def test_edit_returns_a_verifiable_patch_and_leaves_the_base_unchanged(llama):
    before = llama.weights_hash
    edit = llama.edit("layers.0.mlp.down_proj[:, :8]", "scale_slice", factor=0.5)
    assert llama.weights_hash == before and edit.base is llama
    assert edit.model.weights_hash != before and edit.model.checkpoint_id is None
    (step,) = edit.patch.steps
    # The patch records the full name and explicit ranges, as a managed patch does.
    assert step.tensor == DOWN0 and str(step.operator) == "scale_slice@1"
    assert [(s.start, s.stop, s.step) for s in step.selection] == [(0, 16, 1), (0, 8, 1)]
    changed = [d for d in edit.diff if d.changed_elements]
    assert [d.tensor for d in changed] == [DOWN0] and edit.changed_elements == changed[0].changed_elements
    assert edit.verify() is True
    expected = llama.handle.tensor(DOWN0)
    expected[:, :8] = (expected[:, :8].float() * 0.5).to(expected.dtype)
    assert torch.equal(edit.model.tensor(DOWN0), expected)


def test_edits_chain_on_the_edited_model(llama):
    first = llama.edit("layers.0.mlp.down_proj[:, :8]", "scale_slice", factor=0.5)
    second = first.model.edit("layers.1.mlp.down_proj[:4]", "scale_slice", factor=-1.0)
    assert second.base is first.model and second.verify() is True
    assert {d.tensor for d in second.model.diff_from(llama) if d.changed_elements} == {
        DOWN0, "model.layers.1.mlp.down_proj.weight"}


def test_project_out_is_available_by_name(llama):
    rng = np.random.default_rng(0)
    direction = rng.normal(size=16).tolist()
    edit = llama.edit("layers.0.mlp.down_proj", "project_out", direction=direction, columns=[1, 5, 9], alpha=6.0)
    (step,) = edit.patch.steps
    assert str(step.operator) == "project_out@1" and step.parameters["columns"] == [1, 5, 9]
    assert edit.verify() is True


def test_unknown_operators_name_the_available_ones(llama):
    error = _expect(ErrorCode.POLICY_DENIED, llama.edit, "layers.0.mlp.down_proj", "rotate", angle=1.0)
    assert "project_out" in str(error)


def _sparse_edit(directory: Path, base_bits: np.ndarray, count: int = 10) -> np.ndarray:
    """A published edit in the lm-understand-v1 form: ``count`` entries of DOWN0 overwritten."""
    rng = np.random.default_rng(3)
    idx = np.sort(rng.choice(base_bits.size, size=count, replace=False)).astype(np.int32)
    val = (torch.randn(count, generator=torch.Generator().manual_seed(4)) * 0.05).to(torch.bfloat16)
    val = val.view(torch.int16).numpy().view(np.uint16).copy()
    edited = base_bits.reshape(-1).copy()
    edited[idx] = val
    edited = edited.reshape(base_bits.shape)

    def sha(bits):
        return hashlib.sha256(np.ascontiguousarray(bits).tobytes()).hexdigest()

    entry = {"name": DOWN0, "shape": list(base_bits.shape), "dtype": "bfloat16", "form": "sparse",
             "changed_entries": int((edited != base_bits).sum()), "base_sha256": sha(base_bits),
             "edited_sha256": sha(edited)}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps({"tensors": [entry], "changed_tensors": 1}))
    np.savez(directory / "edit.npz", t0_idx=idx, t0_val=val)
    return edited


def _bits(tensor: torch.Tensor) -> np.ndarray:
    return tensor.contiguous().view(torch.int16).numpy().view(np.uint16).copy()


@pytest.fixture(scope="module")
def published(llama, root) -> tuple[Path, np.ndarray]:
    """A published edit directory for the Llama fixture and the edited bits of DOWN0."""
    directory = root / "edits" / "p01" / "edit"
    return directory, _sparse_edit(directory, _bits(llama.handle.tensor(DOWN0)))


def test_apply_patch_imports_a_published_edit_and_stores_its_candidate(llama, root, published):
    from wse import WSE

    directory, edited = published
    edit = llama.apply_patch(directory, repo="example-org/edits", revision="abc123")
    assert edit.patch_id is not None and edit.model.checkpoint_id not in (None, llama.checkpoint_id)
    assert np.array_equal(_bits(edit.model.tensor(DOWN0)), edited)
    assert edit.verify() is True and edit.changed_elements == int((edited != _bits(llama.tensor(DOWN0))).sum())
    with WSE.open(edit.model.checkpoint_id, workspace=root / "workspace") as stored:
        assert stored.weights_hash == edit.model.weights_hash
    again = llama.apply_patch(directory, repo="example-org/edits", revision="abc123")
    assert (again.patch_id, again.model.checkpoint_id) == (edit.patch_id, edit.model.checkpoint_id)


def test_apply_patch_needs_a_stored_base(llama, published):
    edit = llama.edit("layers.0.mlp.down_proj[:, :8]", "scale_slice", factor=0.5)
    error = _expect(ErrorCode.INVALID_ARGUMENT, edit.model.apply_patch, published[0], repo="example-org/edits",
                    revision="abc123")
    assert "stored" in str(error)


# -- diagnose and evaluate -------------------------------------------------------


def test_generate_is_seeded_and_returns_text_and_tokens(llama):
    first = llama.generate("2+2?", max_new_tokens=5, device="cpu")
    again = llama.generate("2+2?", max_new_tokens=5, device="cpu")
    assert first.tokens == again.tokens and 1 <= len(first.tokens) <= 5 and isinstance(first.text, str)
    other = llama.generate("2+2?", max_new_tokens=5, device="cpu", seed=1)
    assert other.seed != first.seed


def test_evaluate_matches_a_direct_generation_evaluation(llama, dataset):
    from wse.evaluators.generation import evaluate_generation
    from wse.evaluators.generation_runs import subset_suite
    from wse.evaluators.text_suites import read_jsonl

    result = llama.evaluate(dataset, problems=4, subset_seed=7, **EVAL)
    data, _ = read_jsonl(dataset, format="math500")
    suite, _ = subset_suite(data, count=4, seed=7, parameters={"profile": "qwen35think@1", "grader": "math_boxed@1",
                                                              "samples_per_problem": 1, "max_new_tokens": 6})
    direct = evaluate_generation(llama.handle, suite, {"sample_count": 4, "seed": 0}, files=llama.files,
                                 device="cpu", batch_size=2)
    assert result.problems == direct.problems and dict(result.metrics) == dict(direct.metrics)
    assert result.evaluator_config_hash == direct.evaluator_config_hash


def test_compare_an_in_memory_edit_with_its_base(llama, dataset):
    from wse.evaluators.generation import compare_generation

    edit = llama.edit("layers.0.mlp.down_proj", "scale_slice", factor=-3.0)
    comparison = llama.compare(edit, dataset, problems=4, subset_seed=7, **EVAL)
    base = llama.evaluate(dataset, problems=4, subset_seed=7, **EVAL)
    candidate = edit.model.evaluate(dataset, problems=4, subset_seed=7, **EVAL)
    assert comparison.result == compare_generation(base, candidate)
    assert comparison.out_dir is None and comparison.base["metrics"] == dict(base.metrics)
    table = comparison.table()
    assert "accuracy" in table and "limit_hit_rate" in table and "mean_generated_tokens" in table


def test_compare_with_out_dir_records_a_resumable_run(llama, dataset, published, tmp_path):
    from wse.evaluators.generation_runs import PROBLEMS_FILE, RUN_FILE, SUMMARY_FILE, read_log

    edit = llama.apply_patch(published[0], repo="example-org/edits", revision="abc123")
    in_memory = llama.compare(edit, dataset, problems=4, subset_seed=7, **EVAL)
    recorded = llama.compare(edit, dataset, problems=4, subset_seed=7, out_dir=tmp_path / "run", **EVAL)
    assert recorded.out_dir == tmp_path / "run" and recorded.result == in_memory.result
    assert all((tmp_path / "run" / name).exists() for name in (RUN_FILE, PROBLEMS_FILE, SUMMARY_FILE))
    assert len(read_log(tmp_path / "run" / PROBLEMS_FILE)) == 8
    # An in-memory edit has no checkpoint, so it cannot be recorded by checkpoint ID.
    scaled = llama.edit("layers.0.mlp.down_proj", "scale_slice", factor=-3.0)
    error = _expect(ErrorCode.INVALID_ARGUMENT, llama.compare, scaled, dataset, problems=4,
                    out_dir=tmp_path / "other", **EVAL)
    assert "stored" in str(error)
