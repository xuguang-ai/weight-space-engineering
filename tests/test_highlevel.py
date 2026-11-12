"""The high-level entry point ``wse.Model`` (``wse.highlevel``).

``Model`` wraps the operator ``Engine`` and the direct-use functions for the
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
    from wse import Model

    model = Model.from_pretrained(hf_snapshots["llama_untied"], repo=REPO, revision=REVISION,
                                workspace=root / "workspace")
    yield model
    model.close()


@pytest.fixture(scope="module")
def dataset(root) -> Path:
    path = root / "math.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in ROWS), encoding="utf-8")
    return path


# -- loading -------------------------------------------------------------------


def test_model_is_exported_from_the_package():
    import wse
    from wse.highlevel import Model

    assert wse.Model is Model and "Model" in wse.__all__ and not hasattr(wse, "WSE")


def test_from_pretrained_imports_a_local_snapshot_once(hf_snapshots, root, llama):
    from wse import Model
    from wse.api import Engine

    assert llama.checkpoint_id is not None and llama.weights_hash.startswith("sha256:")
    with Model.from_pretrained(hf_snapshots["llama_untied"], repo=REPO, revision=REVISION,
                             workspace=root / "workspace") as again:
        assert again.checkpoint_id == llama.checkpoint_id
    with Engine(root / "workspace") as engine:
        assert [m.checkpoint_id for m in engine.list_models()] == [llama.checkpoint_id]
        assert engine.load_manifest(llama.checkpoint_id).source.repository == REPO


def test_a_local_snapshot_needs_its_source_labels(hf_snapshots, tmp_path):
    from wse import Model

    error = _expect(ErrorCode.INVALID_ARGUMENT, Model.from_pretrained, hf_snapshots["llama_untied"],
                    workspace=tmp_path / "ws")
    assert "repo" in str(error) and "revision" in str(error)


def test_a_hub_cache_snapshot_names_its_repository_and_commit(hf_snapshots, tmp_path):
    import shutil

    from wse import Model

    sha = "fedcba9876543210fedcba9876543210fedcba98"
    snapshot = tmp_path / "hub" / "models--example-org--Tiny-Llama" / "snapshots" / sha
    shutil.copytree(hf_snapshots["llama_untied"], snapshot)
    with Model.from_pretrained(snapshot, workspace=tmp_path / "ws") as model:
        assert (model.source.repository, model.source.revision) == ("example-org/Tiny-Llama", sha)


def test_hub_ids_download_only_allow_listed_root_files_at_the_resolved_commit(hf_snapshots, tmp_path,
                                                                             monkeypatch):
    import shutil
    from types import SimpleNamespace

    import huggingface_hub

    from wse import Model

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
    with Model.from_pretrained("example-org/Tiny-Llama", revision="main", workspace=tmp_path / "ws") as model:
        assert (model.source.repository, model.source.revision) == ("example-org/Tiny-Llama", sha)
    assert calls["model_info"] == ("example-org/Tiny-Llama", "main")
    assert calls["list_repo_files"] == ("example-org/Tiny-Llama", sha)
    repo_id, revision, patterns = calls["snapshot_download"]
    assert (repo_id, revision) == ("example-org/Tiny-Llama", sha)
    # Only allow-listed files at the root: the decoys (training_args.bin, original/, onnx/) are never fetched.
    assert "config.json" in patterns and "model.safetensors" in patterns
    assert not any(name.endswith((".bin", ".pth", ".onnx")) or "/" in name for name in patterns)


def test_open_reopens_a_stored_checkpoint(llama, root):
    from wse import Model

    with Model.open(llama.checkpoint_id, workspace=root / "workspace") as model:
        assert model.weights_hash == llama.weights_hash
    _expect(ErrorCode.NOT_FOUND, Model.open, "ckpt_" + "0" * 32, workspace=root / "workspace")


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


def test_roles_name_tensors_sites_and_protections(llama):
    role = llama.role("mlp.down", 0)
    assert (role.tensor, role.site, role.protected) == (DOWN0, "model.layers.0.mlp.down_proj", False)
    assert llama.role("final_norm").protected and llama.role("final_norm").tensor == "model.norm.weight"
    assert {r.role for r in llama.roles() if r.layer == 1} >= {"mlp.down", "attn.o"}
    error = _expect(ErrorCode.POLICY_DENIED, llama.edit, "model.norm", "scale_slice", factor=2.0)
    assert "final_norm" in str(error)


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


def test_fill_and_low_rank_edits_are_available_by_name(llama):
    filled = llama.edit("layers.0.mlp.down_proj[:, 4:8]", "fill_slice", value=0.0)
    assert str(filled.patch.steps[0].operator) == "fill_slice@1" and filled.verify() is True
    assert torch.count_nonzero(filled.model.tensor("layers.0.mlp.down_proj")[:, 4:8]) == 0
    rng = np.random.default_rng(2)
    low_rank = llama.edit("layers.0.mlp.down_proj", "add_low_rank", u=rng.normal(size=(2, 16)),
                          v=rng.normal(size=(2, 32)), alpha=0.01)
    assert str(low_rank.patch.steps[0].operator) == "add_low_rank@1" and low_rank.verify() is True


def test_unknown_operators_name_the_available_ones(llama):
    error = _expect(ErrorCode.NOT_FOUND, llama.edit, "layers.0.mlp.down_proj", "rotate", angle=1.0)
    assert "rotate" in str(error) and "project_out" in str(error)


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
    from wse import Model

    directory, edited = published
    edit = llama.apply_patch(directory, repo="example-org/edits", revision="abc123")
    assert edit.patch_id is not None and edit.model.checkpoint_id not in (None, llama.checkpoint_id)
    assert np.array_equal(_bits(edit.model.tensor(DOWN0)), edited)
    assert edit.verify() is True and edit.changed_elements == int((edited != _bits(llama.tensor(DOWN0))).sum())
    with Model.open(edit.model.checkpoint_id, workspace=root / "workspace") as stored:
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

    evaluation = llama.evaluate(dataset, problems=4, subset_seed=7, **EVAL)
    data, _ = read_jsonl(dataset, format="math500")
    suite, _ = subset_suite(data, count=4, seed=7, parameters={"profile": "qwen35think@1", "grader": "math_boxed@1",
                                                              "samples_per_problem": 1, "max_new_tokens": 6})
    direct = evaluate_generation(llama.handle, suite, {"sample_count": 4, "seed": 0}, files=llama.files,
                                 device="cpu", batch_size=2)
    result = evaluation.result
    assert result.problems == direct.problems and dict(evaluation.metrics) == dict(direct.metrics)
    assert result.evaluator_config_hash == direct.evaluator_config_hash
    # Every sample's output is kept, in plan order, with the tokens its result hashed.
    from wse.evaluators.generation import token_hash

    assert len(evaluation.outputs) == 4
    for output, problem in zip(evaluation.outputs, result.problems):
        assert output.index == problem.index and token_hash(output.tokens) == problem.samples[0].output_sha256
    hits, finished = evaluation.limit_hit(), evaluation.finished()
    assert sorted(hits + finished) == [0, 1, 2, 3]
    assert hits == [k for k, p in enumerate(result.problems) if p.samples[0].limit_hit]
    assert "accuracy" in evaluation.table() and "limit_hit_rate" in evaluation.table()


def test_probe_texts_of_an_evaluation_are_prompt_plus_output(llama, dataset):
    evaluation = llama.evaluate(dataset, problems=4, subset_seed=7, **EVAL)
    texts = evaluation.probe_texts()
    for text, output in zip(texts, evaluation.outputs):
        prompt = evaluation.prompts[output.index]
        generated = tuple(output.tokens[:-1]) if output.finished else tuple(output.tokens)  # no stop token
        assert text.tokens == tuple(prompt) + generated
        assert (text.start, text.stop) == (len(prompt), len(text.tokens))
    tails = evaluation.probe_texts(last=3)
    assert [t.start for t in tails] == [len(t.tokens) - min(3, len(o.tokens) - o.finished)
                                        for t, o in zip(tails, evaluation.outputs)]


def test_compare_an_in_memory_edit_with_its_base(llama, dataset):
    from wse.evaluators.generation import compare_generation

    edit = llama.edit("layers.0.mlp.down_proj", "scale_slice", factor=-3.0)
    comparison = llama.compare(edit, dataset, problems=4, subset_seed=7, **EVAL)
    base = llama.evaluate(dataset, problems=4, subset_seed=7, **EVAL)
    candidate = edit.model.evaluate(dataset, problems=4, subset_seed=7, **EVAL)
    assert comparison.result == compare_generation(base.result, candidate.result)
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
    assert "save()" in str(error)


# -- investigate -------------------------------------------------------------------


def test_sites_take_short_module_names(llama):
    assert llama.site("layers.0.mlp.down_proj:input") == "model.layers.0.mlp.down_proj:input"
    assert llama.site("model.layers.1.mlp.down_proj:output") == "model.layers.1.mlp.down_proj:output"
    assert llama.site("residual:2") == "residual:2"
    error = _expect(ErrorCode.INVALID_ARGUMENT, llama.site, "mlp.down_proj:input")
    assert "model.layers.1.mlp.down_proj" in error.details["candidates"]
    _expect(ErrorCode.NOT_FOUND, llama.site, "layers.7.mlp.down_proj:input")
    _expect(ErrorCode.INVALID_ARGUMENT, llama.site, "layers.0.mlp.down_proj")  # no capture point


def test_probe_matches_the_core_and_resolves_short_names(llama):
    from wse.probes import ProbeText, probe

    texts = [[5, 6, 7, 8, 9, 10], [11, 12, 13, 14], [20, 21, 22, 23, 24]]
    result = llama.probe(texts, sites=["layers.0.mlp.down_proj:input", "residual:1"], device="cpu")
    core = probe(llama.handle, [ProbeText(tuple(t)) for t in texts],
                 sites=["model.layers.0.mlp.down_proj:input", "residual:1"], device="cpu")
    assert result.result.sites == core.sites
    for short, canonical in (("layers.0.mlp.down_proj:input", "model.layers.0.mlp.down_proj:input"),
                             ("residual:1", "residual:1")):
        assert np.array_equal(result.means(short), core.means[canonical])
    contrast = result.contrast("layers.0.mlp.down_proj:input", [0], [1, 2])
    expected = core.contrast("model.layers.0.mlp.down_proj:input", [0], [1, 2])
    assert np.array_equal(contrast.direction, expected.direction)
    assert np.array_equal(contrast.standardized, expected.standardized)
    assert np.array_equal(result.group_mean("residual:1", [1, 2]), core.group_mean("residual:1", [1, 2]))


def test_probe_encodes_strings_as_the_tokenizer_does(llama, monkeypatch):
    from types import SimpleNamespace

    from wse.highlevel import Model

    real = llama._text_model("cpu")

    def tokenizer(text, add_special_tokens=True):  # a tokenizer that prepends BOS (1) by default, as Llama's does
        ids = real.tokenizer(text, add_special_tokens=False)["input_ids"]
        return {"input_ids": ([1] if add_special_tokens else []) + list(ids)}

    monkeypatch.setattr(Model, "_text_model", lambda self, device: SimpleNamespace(runtime=real.runtime,
                                                                               tokenizer=tokenizer))
    ids = tokenizer("abc 12")["input_ids"]
    assert ids[0] == 1
    by_text = llama.probe(["abc 12"], sites=["residual:2"], device="cpu")
    by_ids = llama.probe([ids], sites=["residual:2"], device="cpu")
    assert np.array_equal(by_text.means("residual:2"), by_ids.means("residual:2"))
    assert by_text.result.token_counts == (len(ids),)


def test_probe_reads_an_evaluation_and_its_groups(llama, dataset):
    evaluation = llama.evaluate(dataset, problems=4, subset_seed=7, **EVAL)
    result = llama.probe(evaluation, sites=["layers.1.mlp.down_proj:output"], device="cpu")
    assert result.result.token_counts == tuple(len(o.tokens) - o.finished for o in evaluation.outputs)
    direct = llama.probe(evaluation.probe_texts(), sites=["layers.1.mlp.down_proj:output"], device="cpu")
    assert np.array_equal(result.means("layers.1.mlp.down_proj:output"),
                          direct.means("layers.1.mlp.down_proj:output"))


# -- review fixes ------------------------------------------------------------------


def test_an_edit_can_be_saved_reopened_and_compared_on_record(llama, dataset, root, tmp_path):
    from wse import Model

    edit = llama.edit("layers.0.mlp.down_proj[:, :8]", "scale_slice", factor=-2.0)
    saved = edit.save(label="demo")
    assert saved.patch_id is not None and saved.model.checkpoint_id not in (None, llama.checkpoint_id)
    assert saved.model.weights_hash == edit.model.weights_hash and saved.verify() is True
    again = edit.save(label="demo")
    assert (again.patch_id, again.model.checkpoint_id) == (saved.patch_id, saved.model.checkpoint_id)
    with Model.open(saved.model.checkpoint_id, workspace=root / "workspace") as stored:
        assert stored.weights_hash == edit.model.weights_hash
    patch, blocks = llama.engine.read_imported_patch(saved.patch_id)
    assert patch == edit.patch and dict(blocks) == dict(edit.blocks)
    recorded = llama.compare(saved, dataset, problems=2, subset_seed=7, out_dir=tmp_path / "run", **EVAL)
    assert recorded.out_dir == tmp_path / "run"


def test_a_chained_edit_saves_once_its_base_is_stored(llama):
    first = llama.edit("layers.0.mlp.down_proj[:, :8]", "scale_slice", factor=0.25)
    second = first.model.edit("layers.1.mlp.down_proj[:4]", "scale_slice", factor=-1.0)
    error = _expect(ErrorCode.INVALID_ARGUMENT, second.save)
    assert "save" in str(error)
    first.save()
    stored = second.save()
    assert stored.model.checkpoint_id is not None and stored.verify() is True


def test_store_patch_refuses_a_patch_made_on_another_base(llama):
    edit = llama.edit("layers.0.mlp.down_proj[:, :8]", "scale_slice", factor=0.75)
    other = llama.edit("layers.0.mlp.down_proj[:, :8]", "scale_slice", factor=3.0).save()
    _expect(ErrorCode.VERSION_CONFLICT, llama.engine.store_patch, edit.patch, edit.blocks,
            base_checkpoint_id=other.model.checkpoint_id)


def _sample(index, tokens, finished, prompt_sha="p"):
    from wse.evaluators.generation import SampleOutput

    return SampleOutput(index=index, sample=0, item_id=f"i{index}", seed=index, tokens=list(tokens),
                        finished=finished, text="", prompt_sha256=prompt_sha)


def _evaluation(outputs, prompts, max_new_tokens=6):
    from wse.highlevel import Evaluation

    return Evaluation(result=None, outputs=tuple(outputs), prompts=prompts, max_new_tokens=max_new_tokens)


def test_probe_texts_leave_the_stop_token_out_and_can_bound_the_context():
    evaluation = _evaluation([_sample(0, [7, 8, 9, 2], True), _sample(1, [7, 8, 9, 10, 11, 12], False)],
                             {0: (1, 3, 4), 1: (1, 5)})
    finished, looping = evaluation.probe_texts()
    # A finished output's stop token (2) is neither read nor averaged.
    assert (finished.tokens, finished.start, finished.stop) == ((1, 3, 4, 7, 8, 9), 3, 6)
    assert (looping.tokens, looping.start, looping.stop) == ((1, 5, 7, 8, 9, 10, 11, 12), 2, 8)
    assert [(t.start, t.stop) for t in evaluation.probe_texts(last=2)] == [(4, 6), (6, 8)]
    # context keeps only the last positions before the window's end, which removes earlier context.
    bounded = evaluation.probe_texts(context=4)
    assert [(t.tokens, t.start, t.stop) for t in bounded] == [((4, 7, 8, 9), 1, 4), ((9, 10, 11, 12), 0, 4)]
    for bad in ({"last": 0}, {"context": 0}, {"context": True}):
        _expect(ErrorCode.INVALID_ARGUMENT, evaluation.probe_texts, **bad)


def test_a_missing_local_path_is_not_looked_up_on_the_hub(tmp_path, monkeypatch):
    import huggingface_hub

    from wse import Model

    calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: calls.append(args))
    for source in (Path("missing-org/missing-model"), tmp_path / "nowhere", "./missing/model", "~/missing/model"):
        error = _expect(ErrorCode.NOT_FOUND, Model.from_pretrained, source, workspace=tmp_path / "ws")
        assert "no such directory" in str(error) or "not a snapshot directory" in str(error), str(error)
    assert calls == []


def test_a_single_name_hub_cache_snapshot_is_recognized(hf_snapshots, tmp_path):
    import shutil

    from wse import Model

    sha = "abcdefabcdefabcdefabcdefabcdefabcdefabcd"
    snapshot = tmp_path / "hub" / "models--tinyllama" / "snapshots" / sha
    shutil.copytree(hf_snapshots["llama_untied"], snapshot)
    with Model.from_pretrained(snapshot, workspace=tmp_path / "ws") as model:
        assert (model.source.repository, model.source.revision) == ("tinyllama", sha)


def test_compare_resolves_one_profile_before_generating(hf_snapshots, root, dataset, monkeypatch):
    import wse.evaluators.generation as generation
    from wse import Model

    calls = []
    original = generation.run_generation
    monkeypatch.setattr(generation, "run_generation", lambda *a, **k: calls.append(1) or original(*a, **k))
    with Model.from_pretrained(hf_snapshots["llama_untied"], repo="example-org/Other-Model", revision=REVISION,
                             workspace=root / "workspace") as other:
        options = {key: value for key, value in EVAL.items() if key != "profile"}
        # Each import carries its own default profile; the base's does not exist for the other model.
        _expect(ErrorCode.NOT_FOUND, other.compare, other.edit("layers.0.mlp.down_proj", "scale_slice",
                                                                factor=0.5).model, dataset, problems=2,
                **{**options, "profile": "llama_3_2_1b_instruct_default@1"})
    assert calls == []


def test_an_in_memory_comparison_keeps_both_evaluations_and_logs(llama, dataset):
    lines = []
    edit = llama.edit("layers.0.mlp.down_proj", "scale_slice", factor=-3.0)
    comparison = llama.compare(edit, dataset, problems=2, subset_seed=7, log=lines.append, **EVAL)
    base, candidate = comparison.evaluations
    assert len(base.outputs) == len(candidate.outputs) == 2
    assert len(lines) == 4 and lines[0].startswith("base ") and lines[-1].startswith("candidate ")


def test_edit_accepts_numpy_parameters_and_role_targets(llama):
    direction = np.random.default_rng(1).normal(size=16)
    edit = llama.edit("layers.0.mlp.down_proj", "project_out", direction=direction, columns=np.array([2, 4]),
                      alpha=np.float64(1.0))
    assert edit.patch.steps[0].parameters["columns"] == [2, 4] and edit.verify() is True
    role = llama.role("mlp.down", 0)
    whole = llama.edit(role, "scale_slice", factor=0.5)
    assert [(s.start, s.stop) for s in whole.patch.steps[0].selection] == [(0, 16), (0, 32)]
    head = llama.role("attn.o", 1)
    one_head = llama.edit((head, head.selection(head=1)), "scale_slice", factor=0.0)
    assert [(s.start, s.stop) for s in one_head.patch.steps[0].selection] == [
        tuple(entry.values())[:2] for entry in head.selection(head=1)]


def test_name_and_target_edge_cases(llama):
    assert llama.name("  layers.0.mlp.down_proj  ") == DOWN0
    assert llama.target("layers.0.mlp.down_proj[0,]")[1][0] == {"start": 0, "stop": 1, "step": 1}
    assert llama.target("layers.0.mlp.down_proj[..., :8]")[1] == [{"start": 0, "stop": 16, "step": 1},
                                                                   {"start": 0, "stop": 8, "step": 1}]
    _expect(ErrorCode.INVALID_ARGUMENT, llama.target, "layers.0.mlp.down_proj[\u0663]")
    _expect(ErrorCode.INVALID_ARGUMENT, llama.target, "layers.0.mlp.down_proj[..., ..., 1]")


def test_evaluate_defaults_the_grader_by_format(llama, root):
    path = root / "gsm.jsonl"
    path.write_text(json.dumps({"question": "1+1?", "answer": "1+1=2\n#### 2"}) + "\n", encoding="utf-8")
    evaluation = llama.evaluate(path, format="gsm8k", profile="qwen35think@1", max_new_tokens=6, device="cpu")
    assert evaluation.result.execution.parameters["grader"] == "gsm8k_numeric@1"


def test_closing_frees_the_runtimes_and_refuses_store_use(hf_snapshots, tmp_path):
    from wse import Model

    model = Model.from_pretrained(hf_snapshots["llama_untied"], repo=REPO, revision=REVISION, workspace=tmp_path / "ws")
    edit = model.edit("layers.0.mlp.down_proj[:, :8]", "scale_slice", factor=0.5)
    model.generate("1+1?", max_new_tokens=2, device="cpu")
    assert model.handle._runtimes
    model.close()
    assert not model.handle._runtimes
    for call in (lambda: model.engine, lambda: edit.model.engine, edit.save):
        _expect(ErrorCode.INVALID_ARGUMENT, call)
    assert edit.verify() is True  # in-memory work needs no workspace


def test_the_new_result_classes_are_exported():
    import wse

    assert {"Evaluation", "Probe", "Comparison", "Edit", "Model", "compare_models"} <= set(wse.__all__)


def test_an_output_of_only_the_stop_token_keeps_it_as_its_window():
    evaluation = _evaluation([_sample(0, [2], True), _sample(1, [7, 8], False)], {0: (1, 3), 1: (1, 5)})
    only_stop, looping = evaluation.probe_texts()
    assert (only_stop.tokens, only_stop.start, only_stop.stop) == ((1, 3, 2), 2, 3)
    assert (looping.tokens, looping.start, looping.stop) == ((1, 5, 7, 8), 2, 4)


def test_save_refuses_an_edit_whose_model_is_not_its_patchs_result(llama):
    from wse.highlevel import Edit

    edit = llama.edit("layers.0.mlp.down_proj[:, :8]", "scale_slice", factor=0.125)
    other = llama.edit("layers.0.mlp.down_proj[:, :8]", "scale_slice", factor=8.0)
    forged = Edit(base=llama, model=other.model, patch=edit.patch, diff=edit.diff, blocks=edit.blocks)
    before = llama.engine.store.db.conn.execute("SELECT COUNT(*) AS n FROM imported_patches").fetchone()["n"]
    _expect(ErrorCode.INVALID_ARGUMENT, forged.save)
    after = llama.engine.store.db.conn.execute("SELECT COUNT(*) AS n FROM imported_patches").fetchone()["n"]
    assert after == before  # nothing is stored
