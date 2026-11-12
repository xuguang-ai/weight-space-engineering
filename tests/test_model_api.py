"""The ``Model`` API (``wse.highlevel``): inspection, tensor lists, ``replace_weights``, exports, and comparisons.

``Model`` is one version of an imported open-weight model. These tests use
the tiny Llama fixture (bfloat16) of ``test_highlevel`` and its fixtures.
"""

from __future__ import annotations

import json

import pytest
import torch
from test_highlevel import DOWN0, EVAL, REPO, REVISION, dataset, llama, root  # noqa: F401 - fixtures

from wse.schemas import ErrorCode, WSEError

GENERATION = {"profile": "qwen35think@1", "max_new_tokens": 6, "device": "cpu", "batch_size": 2}


def _expect(code, func, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        func(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


# -- inspection -------------------------------------------------------------------------------------------------


def test_inspect_reports_identity_layers_parameters_and_protections(llama) -> None:
    report = llama.inspect()
    assert report.source == f"{REPO}@{REVISION}" and report.checkpoint_id == llama.checkpoint_id
    assert report.parameter_count == llama.handle.parameter_count and report.tensor_count == len(llama.tensors())
    assert {"model.embed_tokens.weight", "lm_head.weight"} <= set(report.protected)
    assert "model.layers.{i}.mlp.down_proj.weight" in [row[0] for row in report.groups]
    assert report.layers == llama.handle.config["model_config"]["num_hidden_layers"]
    text = str(report)
    assert REPO in text and "model.layers.{i}.mlp.down_proj.weight" in text and "protected" in text


def test_tensors_lists_names_shapes_dtypes_and_protection(llama) -> None:
    mlp = llama.tensors("layers.0.mlp.*")
    assert {entry.name for entry in mlp} == {f"model.layers.0.mlp.{part}_proj.weight" for part in ("gate", "up", "down")}
    down = next(entry for entry in mlp if entry.name == DOWN0)
    assert (down.shape, down.dtype, down.protected) == ((16, 32), "bfloat16", False)
    assert [entry.protected for entry in llama.tensors("embed_tokens")] == [True]
    assert len(llama.tensors()) == len(llama.tensor_names())


def test_tensor_returns_a_copy_that_cannot_change_the_model(llama) -> None:
    before = llama.weights_hash
    weight = llama.tensor("layers.0.mlp.down_proj")
    weight.zero_()
    assert not torch.equal(llama.tensor("layers.0.mlp.down_proj"), weight) and llama.weights_hash == before


def test_generate_reports_generated_tokens_and_the_finish_reason(llama) -> None:
    answer = llama.generate("2+2?", max_new_tokens=5, device="cpu")
    assert answer.generated_tokens == len(answer.tokens) and answer.finish_reason in ("stop", "length")
    assert (answer.finish_reason == "length") == (not answer.finished)


# -- replace_weights --------------------------------------------------------------------------------------------


def test_replace_weights_makes_an_exact_edit_and_save_fills_it_in(llama, root) -> None:
    from wse import Model

    before = llama.weights_hash
    new = llama.tensor("layers.0.mlp.down_proj")
    new[0] = new[0] * -2
    edit = llama.replace_weights({"layers.0.mlp.down_proj": new})
    assert llama.weights_hash == before and edit.model.checkpoint_id is None and edit.patch_id is None
    assert torch.equal(edit.model.tensor(DOWN0), new) and edit.verify()
    assert edit.table().startswith("1 tensor changed, ")
    [step] = edit.patch.steps
    assert (str(step.operator), step.tensor) == ("replace_tensor@1", DOWN0)
    assert step.parameters["source"]["dtype"] == "bfloat16" and step.parameters["source"]["shape"] == [16, 32]
    assert edit.save() is edit and edit.patch_id is not None and edit.model.checkpoint_id is not None
    with Model.open(edit.model.checkpoint_id, workspace=root / "workspace") as stored:
        assert torch.equal(stored.tensor(DOWN0), new)


@pytest.mark.parametrize(("change", "fragment"), [
    (lambda tensor: tensor.float(), "convert them"),
    (lambda tensor: tensor[:, :4], "has shape"),
    (lambda tensor: tensor.index_fill(0, torch.tensor([0]), float("nan")), "nonfinite"),
])
def test_replace_weights_checks_dtype_shape_and_finiteness(llama, change, fragment) -> None:
    error = _expect(ErrorCode.INVALID_ARGUMENT, llama.replace_weights, {DOWN0: change(llama.tensor(DOWN0))})
    assert fragment in error.message


def test_replace_weights_refuses_protected_and_unknown_tensors(llama) -> None:
    embed = llama.tensor("model.embed_tokens.weight")
    _expect(ErrorCode.POLICY_DENIED, llama.replace_weights, {"model.embed_tokens.weight": embed * 0})
    _expect(ErrorCode.NOT_FOUND, llama.replace_weights, {"layers.0.mlp.missing_proj": embed})
    _expect(ErrorCode.INVALID_ARGUMENT, llama.replace_weights, {})


# -- exports ----------------------------------------------------------------------------------------------------


def test_a_model_exports_as_a_hugging_face_directory_that_imports_with_the_same_hash(llama, tmp_path) -> None:
    import transformers

    from wse import Model

    new = llama.tensor(DOWN0)
    new[1] = 0
    edit = llama.replace_weights({DOWN0: new}).save()
    out = edit.model.export(tmp_path / "model")
    names = {path.name for path in out.iterdir()}
    assert {"config.json", "model.safetensors", "wse_provenance.json"} | set(llama.files) == names
    provenance = json.loads((out / "wse_provenance.json").read_text(encoding="utf-8"))
    assert provenance["weights_hash"] == edit.model.weights_hash
    assert provenance["parent_checkpoint_id"] == llama.checkpoint_id and provenance["patch_id"] == edit.patch_id
    with Model.from_pretrained(out, workspace=tmp_path / "other") as again:
        assert again.weights_hash == edit.model.weights_hash
        assert again.source.repository == f"wse-export/{edit.model.checkpoint_id}"
    loaded = transformers.AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16)
    assert torch.equal(loaded.state_dict()[DOWN0], new)
    _expect(ErrorCode.INVALID_ARGUMENT, edit.model.export, out)  # not an empty directory


def test_an_exported_patch_replays_onto_the_same_base_in_another_workspace(llama, hf_snapshots, tmp_path) -> None:
    from wse import Model

    edit = llama.edit("layers.1.mlp.down_proj[:2, :3]", "fill_slice", value=0.5)
    out = edit.export(tmp_path / "patch")
    provenance = json.loads((out / "patch.json").read_text(encoding="utf-8"))
    assert provenance["base"]["weights_hash"] == llama.weights_hash
    assert provenance["result_weights_hash"] == edit.model.weights_hash and provenance["format"] == "wse-patch-v1"
    with Model.from_pretrained(hf_snapshots["llama_untied"], repo=REPO, revision=REVISION,
                               workspace=tmp_path / "ws") as base:
        replayed = base.apply_patch(out, format="wse-patch-v1")
        assert replayed.model.weights_hash == edit.model.weights_hash and replayed.patch_id is not None


# -- evaluation and comparison ----------------------------------------------------------------------------------


def test_evaluate_reads_math_rows_and_reports_the_requested_metrics(llama, dataset) -> None:
    evaluation = llama.evaluate(dataset, evaluator="math", metrics=["accuracy", "limit_hit_rate"], problems=2,
                                subset_seed=7, **GENERATION)
    table = evaluation.table()
    assert "accuracy" in table and "limit_hit_rate" in table and "mean_generated_tokens" not in table
    rows = evaluation.samples()
    assert len(rows) == 2 and {"item_id", "correct", "limit_hit", "generated_tokens", "text"} <= set(rows[0])
    _expect(ErrorCode.INVALID_ARGUMENT, llama.evaluate, dataset, evaluator="code")
    _expect(ErrorCode.INVALID_ARGUMENT, llama.evaluate, dataset, metrics=["repetition_rate"])


def test_compare_models_pairs_two_models_on_the_requested_metrics(llama, dataset) -> None:
    from wse import compare_models

    edit = llama.edit("layers.0.mlp.down_proj[:2, :3]", "fill_slice", value=0.0)
    comparison = compare_models(base=llama, candidate=edit.model, data=dataset, evaluator="math",
                                metrics=["accuracy", "limit_hit_rate"], problems=2, subset_seed=7, **GENERATION)
    table = comparison.table()
    assert "limit_hit_rate" in table and "mean_generated_tokens" not in table
    assert {"accuracy", "limit_hit_rate"} <= set(comparison.result.differences)


def test_a_probe_of_an_evaluation_keeps_each_texts_sample(llama, dataset) -> None:
    evaluation = llama.evaluate(dataset, problems=3, subset_seed=7, **EVAL)
    probe = llama.probe(evaluation, sites=["layers.1.mlp.down_proj:output"], device="cpu")
    assert len(probe.samples) == 3 == probe.means("layers.1.mlp.down_proj:output").shape[0]
    assert [(s.item_id, s.correct, s.limit_hit, s.text) for s in probe.samples] == [
        (row["item_id"], row["correct"], row["limit_hit"], row["text"]) for row in evaluation.samples()]


def test_a_local_snapshot_needs_the_commit_it_was_downloaded_from(hf_snapshots, tmp_path) -> None:
    from wse import Model

    error = _expect(ErrorCode.INVALID_ARGUMENT, Model.from_pretrained, hf_snapshots["llama_untied"], repo=REPO,
                    revision="main", workspace=tmp_path / "ws")
    assert "not a commit SHA" in error.message
