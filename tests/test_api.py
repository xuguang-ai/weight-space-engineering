"""Direct (unmanaged) library use: in-memory loading, handles, and contracts."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from safetensors.torch import save_file

from wse.api import EditResult, Engine, EvaluationResult, load_model
from wse.schemas import ErrorCode, MetricGate, ObjectiveSpec, Usage, WSEError
from wse.tensors import weights_hash

FIXTURE_SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "signed_sum" / "build_fixture.py"


def _run_isolated(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_in_memory_load_needs_neither_sqlite_nor_mcp(tmp_path):
    # Block SQLite and MCP before importing wse; direct use must still work.
    result = _run_isolated(
        f"""
        import sys, os
        for blocked in ("sqlite3", "_sqlite3", "mcp", "agents", "claude_agent_sdk"):
            sys.modules[blocked] = None
        import importlib.util
        spec = importlib.util.spec_from_file_location("fixture", {str(FIXTURE_SCRIPT)!r})
        fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
        os.chdir({str(tmp_path)!r})
        from wse import load_model
        handle = load_model(fixture.task_tensors(), "mlp@1", fixture.CONFIG)
        assert handle.weights_hash.startswith("sha256:")
        from wse import analyze_tensor, read_tensor_slice
        full = [dict(start=0, stop=4, step=1), dict(start=0, stop=2, step=1)]
        for kind in ("stats", "norm", "top_k"):
            assert analyze_tensor(handle, kind, "layers.0.weight", full).management == "unmanaged"
        assert read_tensor_slice(handle, "layers.0.weight", full).element_count == 8
        assert os.listdir(".") == [], os.listdir(".")
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_core_imports_no_mcp_or_agent_sdk():
    result = _run_isolated(
        """
        import sys
        import wse, wse.api, wse.cli, wse.store, wse.database, wse.registry, wse.adapters.mlp
        import wse.analysis, wse.policy, wse.operators.base, wse.operators.tensor
        loaded = sorted(m for m in sys.modules
                        if m.split(".")[0] in ("mcp", "agents", "openai", "claude_agent_sdk"))
        print(loaded)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_load_model_from_mapping(signed_sum):
    handle = load_model(signed_sum.task_tensors(), "mlp@1", signed_sum.CONFIG)
    assert str(handle.adapter_ref) == "mlp@1"
    assert handle.tensor_names() == (
        "layers.0.weight",
        "layers.0.bias",
        "layers.1.weight",
        "layers.1.bias",
    )
    assert handle.parameter_count == 8 + 4 + 8 + 2
    assert handle.config == signed_sum.CONFIG
    assert handle.identity.weights_hash == handle.weights_hash
    w1 = handle.architecture.tensors[0]
    assert w1.axes == ["out_features", "in_features"] and w1.shape == [4, 2]


def test_load_model_from_safetensors_file_and_config_path(tmp_path, signed_sum):
    weights = tmp_path / "model.safetensors"
    save_file(signed_sum.task_tensors(), str(weights))
    config = tmp_path / "config.json"
    config.write_text('{"bias": true, "layer_sizes": [2, 4, 2], "activation": "relu"}')
    from_file = load_model(weights, "mlp@1", config)
    from_memory = load_model(signed_sum.task_tensors(), "mlp@1", signed_sum.CONFIG)
    assert from_file.identity == from_memory.identity


def test_load_model_accepts_numpy_arrays(signed_sum):
    arrays = {name: t.numpy().copy() for name, t in signed_sum.task_tensors().items()}
    handle = load_model(arrays, "mlp@1", signed_sum.CONFIG)
    assert handle.weights_hash == load_model(signed_sum.task_tensors(), "mlp@1", signed_sum.CONFIG).weights_hash


@pytest.mark.parametrize("adapter", ["mlp@2", "mlp@0.9", "other@1"])
def test_load_model_rejects_unknown_adapter_versions(signed_sum, adapter):
    with pytest.raises(WSEError) as excinfo:
        load_model(signed_sum.task_tensors(), adapter, signed_sum.CONFIG)
    assert excinfo.value.code == ErrorCode.NOT_FOUND


@pytest.mark.parametrize("adapter", ["mlp", "mlp@", "mlp@latest", "mlp@1@2"])
def test_load_model_requires_an_exact_adapter_version(signed_sum, adapter):
    with pytest.raises(WSEError) as excinfo:
        load_model(signed_sum.task_tensors(), adapter, signed_sum.CONFIG)
    assert excinfo.value.code == ErrorCode.INVALID_ARGUMENT


def test_weights_hash_ignores_dict_insertion_order(signed_sum):
    tensors = signed_sum.task_tensors()
    reversed_tensors = dict(reversed(list(tensors.items())))
    assert list(reversed_tensors) != list(tensors)
    assert weights_hash(tensors) == weights_hash(reversed_tensors)
    first = load_model(tensors, "mlp@1", signed_sum.CONFIG)
    second = load_model(reversed_tensors, "mlp@1", dict(reversed(list(signed_sum.CONFIG.items()))))
    assert first.identity == second.identity
    assert first.tensor_names() == second.tensor_names()


def test_identity_distinguishes_weights_and_config(signed_sum):
    task = load_model(signed_sum.task_tensors(), "mlp@1", signed_sum.CONFIG)
    reference = load_model(signed_sum.reference_tensors(), "mlp@1", signed_sum.CONFIG)
    assert task.weights_hash != reference.weights_hash
    assert task.config_hash == reference.config_hash


def test_handle_owns_private_copies(signed_sum):
    tensors = signed_sum.task_tensors()
    handle = load_model(tensors, "mlp@1", signed_sum.CONFIG)
    before = handle.weights_hash
    tensors["layers.0.weight"].fill_(7.0)
    handle.tensor("layers.0.weight").fill_(9.0)
    handle.tensors()["layers.1.weight"].zero_()
    assert weights_hash(handle.tensors()) == before
    with pytest.raises(WSEError) as excinfo:
        handle.tensor("layers.0.weight ")
    assert excinfo.value.code == ErrorCode.NOT_FOUND


def test_in_memory_load_rejects_invalid_models(signed_sum):
    tensors = signed_sum.task_tensors()
    tensors["layers.0.bias"][0] = float("nan")
    with pytest.raises(WSEError) as excinfo:
        load_model(tensors, "mlp@1", signed_sum.CONFIG)
    assert excinfo.value.code == ErrorCode.INVALID_MODEL
    with pytest.raises(WSEError):
        load_model(signed_sum.task_tensors(), "mlp@1", {"layer_sizes": [2, 3, 2]})


def test_fixture_forward_matches_analytic_accuracy(signed_sum):
    inputs, labels, pair_ids = signed_sum.generate_examples(32, seed=7)
    assert inputs.shape == (64, 2) and inputs.dtype == np.float32
    assert labels.sum() == 32  # balanced
    assert np.all(np.abs(inputs.sum(axis=1)) >= signed_sum.MARGIN)
    assert np.array_equal(pair_ids[::2], pair_ids[1::2])
    x, y = torch.from_numpy(inputs), torch.from_numpy(labels)
    reference = load_model(signed_sum.reference_tensors(), "mlp@1", signed_sum.CONFIG)
    task = load_model(signed_sum.task_tensors(), "mlp@1", signed_sum.CONFIG)
    assert (reference.forward(x).argmax(dim=1) == y).float().mean().item() == 1.0
    assert (task.forward(x).argmax(dim=1) == y).float().mean().item() == 0.5
    with pytest.raises(WSEError):
        task.forward(x.double())


def test_objective_spec_contract():
    objective = ObjectiveSpec(
        task_id="signed_sum_v1",
        description="Classify the sign of x0 + x1.",
        gates=[MetricGate("dev.accuracy", min_value=1.0), MetricGate("other.accuracy", max_drop=0.01)],
    )
    assert [g.metric for g in objective.acceptance_gates] == ["dev.accuracy"]
    assert [g.metric for g in objective.regression_gates] == ["other.accuracy"]
    assert ObjectiveSpec.model_validate_json(objective.canonical_json()) == objective
    assert objective.schema_version == "1"
    with pytest.raises(ValidationError):
        MetricGate("m", min_value=1.0, max_drop=0.1)
    with pytest.raises(ValidationError):
        MetricGate("m")
    with pytest.raises(ValidationError):
        MetricGate("m", min_delta=float("inf"))
    with pytest.raises(ValidationError):
        ObjectiveSpec(task_id="t", description="d", gates=[MetricGate("m", max_drop=0.0)])
    with pytest.raises(ValidationError):
        ObjectiveSpec(task_id="t", description="d", gates=[])


def test_core_results_carry_no_experiment_ids():
    for result_type in (EditResult, EvaluationResult):
        names = {f.name for f in fields(result_type)}
        assert not any("experiment" in name or "report_id" in name for name in names)
        assert "management" in names
    assert Usage().model_forward_calls is None  # unavailable, not zero


def test_config_hash_uses_the_adapter_normalized_config(signed_sum):
    full = load_model(signed_sum.task_tensors(), "mlp@1", signed_sum.CONFIG)
    minimal = load_model(signed_sum.task_tensors(), "mlp@1", {"layer_sizes": [2, 4, 2]})
    assert minimal.config_hash == full.config_hash
    assert minimal.config == full.config
    other = load_model(
        {k: v for k, v in signed_sum.task_tensors().items() if k.endswith("weight")},
        "mlp@1", {"layer_sizes": [2, 4, 2], "bias": False},
    )
    assert other.config_hash != full.config_hash


def test_equivalent_configs_deduplicate_on_import(tmp_path, signed_sum):
    with Engine(tmp_path / "ws") as engine:
        first = engine.import_model({"adapter": "mlp@1", "config": signed_sum.CONFIG},
                                    signed_sum.task_tensors())
        second = engine.import_model({"adapter": "mlp@1", "config": {"layer_sizes": [2, 4, 2]}},
                                     signed_sum.task_tensors())
    assert second.checkpoint_id == first.checkpoint_id
    assert first.config == signed_sum.CONFIG


def test_numpy_arrays_with_negative_strides_or_foreign_byte_order_load(signed_sum):
    tensors = {name: t.numpy() for name, t in signed_sum.task_tensors().items()}
    expected = load_model(tensors, "mlp@1", signed_sum.CONFIG)
    weight = tensors["layers.0.weight"]
    reversed_rows = np.ascontiguousarray(weight[::-1])
    variants = dict(tensors)
    variants["layers.0.weight"] = reversed_rows[::-1]  # negative stride, same values
    variants["layers.1.weight"] = tensors["layers.1.weight"].astype(">f4")
    handle = load_model(variants, "mlp@1", signed_sum.CONFIG)
    assert handle.weights_hash == expected.weights_hash
    with pytest.raises(WSEError) as excinfo:
        load_model({**tensors, "layers.0.bias": np.array([object()] * 4)}, "mlp@1", signed_sum.CONFIG)
    assert excinfo.value.code == ErrorCode.INVALID_MODEL
