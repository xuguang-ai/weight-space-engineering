"""Managed import, content-addressed storage, READY visibility, and the CLI."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import stat
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from wse.api import Engine, read_bundle
from wse.cli import main as cli_main
from wse.database import LATEST_SCHEMA_VERSION, Database, audit_events
from wse.schemas import CHECKPOINT_ID_PATTERN, ErrorCode, ModelManifest, WSEError


def _files(workspace: Path, kind: str) -> list[Path]:
    return sorted((workspace / "artifacts" / kind).iterdir())


def _checkpoint_count(workspace: Path) -> int:
    with sqlite3.connect(workspace / "metadata.sqlite") as conn:
        return conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]


def _import_request(signed_sum, **overrides):
    request = {"adapter": "mlp@1", "config": signed_sum.CONFIG}
    request.update(overrides)
    return request


def _full_selection(shape):
    return [{"start": 0, "stop": size, "step": 1} for size in shape]


# -- roundtrip ------------------------------------------------------------------


def test_fixture_bundle_roundtrip_is_bitwise_identical(engine, bundles, signed_sum):
    manifest = engine.import_bundle(bundles["task"])
    assert CHECKPOINT_ID_PATTERN.fullmatch(manifest.checkpoint_id)
    assert manifest.schema_version == "1"
    assert manifest.alias_groups == []
    assert manifest.parameter_count == 22
    expected = signed_sum.task_tensors()
    stored = engine.store.load_tensors(manifest.checkpoint_id)
    assert sorted(stored) == sorted(expected)
    for name, tensor in expected.items():
        assert stored[name].dtype == torch.float32
        assert stored[name].numpy().tobytes() == tensor.numpy().tobytes()
        info = manifest.tensor_info(name)
        read = engine.read_tensor(manifest.checkpoint_id, name, _full_selection(info.shape))
        assert read.numpy().tobytes() == tensor.numpy().tobytes()
    handle = engine.open_checkpoint(manifest.checkpoint_id)
    assert handle.identity == manifest.identity


def test_roundtrip_preserves_signed_zero_and_subnormals(engine):
    rng = np.random.default_rng(0)
    w0 = rng.standard_normal((16, 8)).astype(np.float32)
    w0[0, 0] = -0.0
    w0[0, 1] = np.float32(1e-45)  # smallest subnormal
    w0[0, 2] = np.finfo(np.float32).max
    w0[0, 3] = -np.finfo(np.float32).tiny
    tensors = {
        "layers.0.weight": w0,
        "layers.1.weight": rng.standard_normal((3, 16)).astype(np.float32),
    }
    manifest = engine.import_model(
        {"adapter": "mlp@1", "config": {"layer_sizes": [8, 16, 3], "bias": False}}, tensors
    )
    stored = engine.store.load_tensors(manifest.checkpoint_id)
    for name, array in tensors.items():
        assert stored[name].numpy().tobytes() == array.tobytes()
    assert np.signbit(stored["layers.0.weight"].numpy()[0, 0])


def test_manifest_and_weights_persist_across_restart(workspace, bundles):
    with Engine(workspace) as first:
        manifest = first.import_bundle(bundles["task"])
    with Engine(workspace) as second:
        assert second.load_manifest(manifest.checkpoint_id) == manifest
        assert [m.checkpoint_id for m in second.list_models()] == [manifest.checkpoint_id]


def test_read_tensor_returns_requested_region(engine, bundles, signed_sum):
    manifest = engine.import_bundle(bundles["task"])
    region = engine.read_tensor(
        manifest.checkpoint_id,
        "layers.1.weight",
        [{"start": 1, "stop": 2, "step": 1}, {"start": 1, "stop": 3}],
    )
    expected = signed_sum.task_tensors()["layers.1.weight"][1:2, 1:3]
    assert torch.equal(region, expected) and region.shape == (1, 2)


# -- deduplication -----------------------------------------------------------------


def test_reimporting_the_same_bundle_does_not_duplicate(workspace, bundles):
    with Engine(workspace) as engine:
        first = engine.import_bundle(bundles["task"])
        second = engine.import_bundle(bundles["task"])
    with Engine(workspace) as engine:
        third = engine.import_bundle(bundles["task"])
    assert first == second == third
    assert len(_files(workspace, "weights")) == 1
    assert len(_files(workspace, "manifests")) == 1
    assert _checkpoint_count(workspace) == 1


def test_same_weights_with_new_provenance_share_one_weights_file(engine, workspace, signed_sum):
    first = engine.import_model(_import_request(signed_sum), signed_sum.task_tensors())
    second = engine.import_model(
        _import_request(signed_sum, provenance={"source": "second copy"}), signed_sum.task_tensors()
    )
    assert first.checkpoint_id != second.checkpoint_id
    assert first.weights_hash == second.weights_hash
    assert len(_files(workspace, "weights")) == 1
    assert len(_files(workspace, "manifests")) == 2
    reference = engine.import_model(_import_request(signed_sum), signed_sum.reference_tensors())
    assert len(_files(workspace, "weights")) == 2
    assert reference.weights_hash != first.weights_hash


def test_stored_artifacts_are_read_only_and_content_addressed(engine, workspace, bundles):
    manifest = engine.import_bundle(bundles["task"])
    (weights_file,) = _files(workspace, "weights")
    (manifest_file,) = _files(workspace, "manifests")
    assert weights_file.name == manifest.weights_hash.removeprefix("sha256:") + ".safetensors"
    for path in (weights_file, manifest_file):
        assert not path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    assert list((workspace / "artifacts" / "tmp").iterdir()) == []
    with sqlite3.connect(workspace / "metadata.sqlite") as conn:
        events = [row[0] for row in conn.execute("SELECT event_type FROM audit_events")]
    assert events == ["checkpoint.imported"]


# -- validation ----------------------------------------------------------------------


def _expect(code, func, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        func(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


@pytest.mark.parametrize(
    "config",
    [
        {"layer_sizes": [2]},
        {"layer_sizes": [2, 0, 2]},
        {"layer_sizes": [2, 4, 2], "activation": "gelu"},
        {"layer_sizes": [2, 4, 2], "residual": True},
        {"layer_sizes": ["2", 4, 2]},
        {"layer_sizes": [2, 4097, 2]},
    ],
)
def test_invalid_architectures_are_rejected(engine, workspace, signed_sum, config):
    _expect(ErrorCode.INVALID_MODEL, engine.import_model, _import_request(signed_sum, config=config),
            signed_sum.task_tensors())
    assert _checkpoint_count(workspace) == 0


def _mutated(signed_sum, name, value):
    tensors = signed_sum.task_tensors()
    if value is None:
        del tensors[name]
    else:
        tensors[name] = value
    return tensors


@pytest.mark.parametrize(
    "name,value",
    [
        ("layers.0.weight", torch.zeros(2, 4)),  # transposed shape
        ("layers.1.bias", torch.zeros(3)),
        ("layers.0.weight", torch.zeros(4, 2, dtype=torch.float64)),
        ("layers.0.weight", torch.zeros(4, 2, dtype=torch.float16)),
        ("layers.0.weight", None),  # missing tensor
        ("layers.2.weight", torch.zeros(2, 2)),  # unexpected tensor
    ],
)
def test_invalid_shapes_dtypes_and_names_are_rejected(engine, workspace, signed_sum, name, value):
    _expect(ErrorCode.INVALID_MODEL, engine.import_model, _import_request(signed_sum),
            _mutated(signed_sum, name, value))
    assert _checkpoint_count(workspace) == 0
    assert _files(workspace, "weights") == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_are_rejected(engine, workspace, signed_sum, bad):
    tensors = signed_sum.task_tensors()
    tensors["layers.1.weight"][0, 2] = bad
    error = _expect(ErrorCode.INVALID_MODEL, engine.import_model, _import_request(signed_sum), tensors)
    assert error.details == {"tensor": "layers.1.weight", "nonfinite_count": 1}
    assert _checkpoint_count(workspace) == 0


def test_shared_parameters_are_rejected(engine, signed_sum):
    tensors = signed_sum.task_tensors()
    shared = torch.zeros(4)
    tensors["layers.0.bias"] = shared
    tensors["layers.0.weight"] = torch.zeros(4, 2)
    tensors["layers.1.bias"] = shared[:2]
    _expect(ErrorCode.INVALID_MODEL, engine.import_model, _import_request(signed_sum), tensors)


def test_client_cannot_supply_identity_fields(engine, signed_sum):
    for field, value in [
        ("checkpoint_id", "ckpt_" + "0" * 32),
        ("weights_hash", "sha256:" + "0" * 64),
        ("tensors", []),
    ]:
        _expect(ErrorCode.INVALID_ARGUMENT, engine.import_model,
                _import_request(signed_sum, **{field: value}), signed_sum.task_tensors())


def test_oversized_models_are_rejected(workspace, bundles, signed_sum):
    with Engine(workspace, max_bytes=64) as engine:
        _expect(ErrorCode.INVALID_MODEL, engine.import_bundle, bundles["task"])
        _expect(ErrorCode.INVALID_MODEL, engine.import_model, _import_request(signed_sum),
                signed_sum.task_tensors())


# -- path escapes ----------------------------------------------------------------------


def _rewrite_bundle(bundle: Path, **changes) -> None:
    path = bundle / "wse_bundle.json"
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize(
    "field,value",
    [
        ("weights", "../reference_model/model.safetensors"),
        ("weights", "sub/../../reference_model/model.safetensors"),
        ("config", "/etc/hosts"),
        ("config", "..\\config.json"),
        ("weights", "~/model.safetensors"),
    ],
)
def test_bundle_paths_cannot_escape(engine, workspace, bundles, field, value):
    _rewrite_bundle(bundles["task"], **{field: value})
    _expect(ErrorCode.INVALID_ARGUMENT, engine.import_bundle, bundles["task"])
    assert _checkpoint_count(workspace) == 0


def test_bundle_symlinks_cannot_escape(engine, tmp_path, bundles):
    outside = tmp_path / "outside.safetensors"
    shutil.copy(bundles["reference"] / "model.safetensors", outside)
    (bundles["task"] / "link.safetensors").symlink_to(outside)
    _rewrite_bundle(bundles["task"], weights="link.safetensors")
    _expect(ErrorCode.INVALID_ARGUMENT, engine.import_bundle, bundles["task"])


def test_bundle_symlink_inside_bundle_is_allowed(engine, bundles):
    (bundles["task"] / "alias.safetensors").symlink_to(bundles["task"] / "model.safetensors")
    _rewrite_bundle(bundles["task"], weights="alias.safetensors")
    assert engine.import_bundle(bundles["task"]).parameter_count == 22


def test_bundle_with_unknown_fields_or_missing_files_is_rejected(engine, bundles):
    _rewrite_bundle(bundles["task"], weights="absent.safetensors")
    _expect(ErrorCode.NOT_FOUND, engine.import_bundle, bundles["task"])
    _rewrite_bundle(bundles["task"], weights="model.safetensors", checkpoint_id="ckpt_" + "a" * 32)
    _expect(ErrorCode.INVALID_ARGUMENT, engine.import_bundle, bundles["task"])


def test_corrupt_safetensors_file_is_rejected(engine, bundles):
    (bundles["task"] / "model.safetensors").write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{garbage")
    _expect(ErrorCode.INVALID_MODEL, engine.import_bundle, bundles["task"])


# -- read_tensor selections -------------------------------------------------------------


@pytest.mark.parametrize(
    "selection",
    [
        [{"start": 0, "stop": 3, "step": 1}],  # wrong number of axes
        [{"start": -1, "stop": 1, "step": 1}, {"start": 0, "stop": 4, "step": 1}],
        [{"start": 1, "stop": 1, "step": 1}, {"start": 0, "stop": 4, "step": 1}],  # empty
        [{"start": 0, "stop": 3, "step": 1}, {"start": 0, "stop": 4, "step": 1}],  # out of bounds
        [{"start": 0, "stop": 2, "step": 2}, {"start": 0, "stop": 4, "step": 1}],
        [{"start": "0", "stop": 2, "step": 1}, {"start": 0, "stop": 4, "step": 1}],
        [{"start": 0.0, "stop": 2, "step": 1}, {"start": 0, "stop": 4, "step": 1}],
        [{"start": 0, "stop": 2, "step": 1, "extra": 1}, {"start": 0, "stop": 4, "step": 1}],
        "[0:2, 0:4]",
        "W[0]",
    ],
)
def test_invalid_selections_are_rejected(engine, bundles, selection):
    manifest = engine.import_bundle(bundles["task"])
    _expect(ErrorCode.INVALID_ARGUMENT, engine.read_tensor, manifest.checkpoint_id, "layers.1.weight", selection)


def test_unknown_tensors_and_checkpoints_are_not_found(engine, bundles):
    manifest = engine.import_bundle(bundles["task"])
    for name in ["layers.1.Weight", "layers.1", "layers.1.weight[0]", "../layers.1.weight"]:
        _expect(ErrorCode.NOT_FOUND, engine.read_tensor, manifest.checkpoint_id, name, [])
    for checkpoint_id in ["ckpt_" + "0" * 32, "../metadata.sqlite", "", "latest"]:
        _expect(ErrorCode.NOT_FOUND, engine.load_manifest, checkpoint_id)


# -- READY visibility and recovery ------------------------------------------------------


def test_orphaned_artifacts_never_become_models(workspace, bundles):
    with Engine(workspace) as engine:
        published = engine.import_bundle(bundles["task"])
    # Simulate an interrupted import: content files published, no database commit.
    orphan_weights = workspace / "artifacts" / "weights" / ("f" * 64 + ".safetensors")
    shutil.copy(bundles["reference"] / "model.safetensors", orphan_weights)
    orphan_manifest = workspace / "artifacts" / "manifests" / ("e" * 64 + ".json")
    orphan_manifest.write_text(published.model_copy(update={"checkpoint_id": "ckpt_" + "1" * 32}).canonical_json())
    temp_file = workspace / "artifacts" / "tmp" / "interrupted.part"
    temp_file.write_bytes(b"partial")

    with Engine(workspace, recover=False) as engine:
        assert [m.checkpoint_id for m in engine.list_models()] == [published.checkpoint_id]
        _expect(ErrorCode.NOT_FOUND, engine.load_manifest, "ckpt_" + "1" * 32)
        fresh = engine.store.recover()  # within the grace period: nothing removed
        assert fresh.removed_orphans == [] and fresh.removed_temp_files == []
        report = engine.store.recover(grace_seconds=0)
        assert sorted(report.removed_orphans) == [
            "manifests/" + orphan_manifest.name,
            "weights/" + orphan_weights.name,
        ]
        assert report.removed_temp_files == ["interrupted.part"]
        assert report.corrupted_checkpoints == []
        assert engine.load_manifest(published.checkpoint_id) == published
    assert not orphan_weights.exists() and not temp_file.exists()


def _make_writable(path: Path) -> None:
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def test_tampered_weights_are_marked_corrupt_and_unloadable(workspace, bundles, signed_sum):
    with Engine(workspace) as engine:
        manifest = engine.import_bundle(bundles["task"])
    (weights_file,) = _files(workspace, "weights")
    _make_writable(weights_file)
    save_file(signed_sum.reference_tensors(), str(weights_file))
    with Engine(workspace) as engine:  # startup recovery detects the mismatch
        assert engine.last_recovery.corrupted_checkpoints == [manifest.checkpoint_id]
        assert engine.list_models() == []
        _expect(ErrorCode.ARTIFACT_CORRUPT, engine.load_manifest, manifest.checkpoint_id)
        _expect(ErrorCode.ARTIFACT_CORRUPT, engine.read_tensor, manifest.checkpoint_id, "layers.1.weight", [])
        events = audit_events(engine.store.db.conn, manifest.checkpoint_id)
        assert [e["event_type"] for e in events] == ["checkpoint.imported", "checkpoint.corrupt"]


def test_missing_file_detected_on_load_marks_checkpoint_corrupt(workspace, bundles):
    with Engine(workspace) as engine:
        manifest = engine.import_bundle(bundles["task"])
        (weights_file,) = _files(workspace, "weights")
        weights_file.unlink()
        _expect(ErrorCode.ARTIFACT_CORRUPT, engine.read_tensor, manifest.checkpoint_id, "layers.1.weight",
                _full_selection([2, 4]))
        assert engine.store.checkpoint_state(manifest.checkpoint_id) == "CORRUPT"
        _expect(ErrorCode.ARTIFACT_CORRUPT, engine.load_manifest, manifest.checkpoint_id)


# -- migrations ------------------------------------------------------------------------------


def test_migration_1_creates_schema_once(workspace):
    with Engine(workspace) as engine:
        assert engine.store.db.schema_version() == LATEST_SCHEMA_VERSION == 1
        tables = {row[0] for row in engine.store.db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert {"artifacts", "checkpoints", "audit_events"} <= tables
    Database(workspace / "metadata.sqlite").close()  # reapplying is a no-op
    with sqlite3.connect(workspace / "metadata.sqlite") as conn:
        conn.execute("PRAGMA user_version = 99")
    _expect(ErrorCode.VERSION_CONFLICT, Database, workspace / "metadata.sqlite")


# -- agent-visible manifest content ----------------------------------------------------


def test_task_manifest_does_not_reveal_the_fault(workspace, bundles):
    with Engine(workspace) as engine:
        reference = engine.import_bundle(bundles["reference"])
        task = engine.import_bundle(bundles["task"])
    rendered = task.canonical_json().lower()
    assert reference.checkpoint_id not in rendered
    for forbidden in ["fault", "clean", "oracle", "reference", "negat", "flip", "inject", "broken",
                      "repair", "row 1", "task_model", str(bundles["task"]).lower()]:
        assert forbidden not in rendered, forbidden
    manifest, _ = read_bundle(bundles["task"])
    assert manifest.provenance.model_dump() == {
        "kind": "import", "source": "signed_sum model v1", "parent_checkpoint_id": None, "notes": ""
    }


# -- CLI ----------------------------------------------------------------------------------------


def test_cli_import_prints_manifest_and_is_idempotent(workspace, bundles, capsys):
    assert cli_main(["--workspace", str(workspace), "import", str(bundles["task"])]) == 0
    first = ModelManifest.model_validate_json(capsys.readouterr().out)
    assert cli_main(["--workspace", str(workspace), "import", str(bundles["task"])]) == 0
    second = ModelManifest.model_validate_json(capsys.readouterr().out)
    assert first == second
    with Engine(workspace) as engine:
        assert engine.load_manifest(first.checkpoint_id) == first


def test_cli_import_reports_domain_errors(workspace, bundles, capsys):
    _rewrite_bundle(bundles["task"], weights="../reference_model/model.safetensors")
    assert cli_main(["--workspace", str(workspace), "import", str(bundles["task"])]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "INVALID_ARGUMENT" and error["retryable"] is False


def test_concurrent_identical_imports_create_one_checkpoint(workspace, bundles):
    from concurrent.futures import ThreadPoolExecutor

    Engine(workspace).close()  # create the schema once

    def run(_):
        with Engine(workspace, recover=False) as engine:
            return engine.import_bundle(bundles["task"]).checkpoint_id

    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = set(pool.map(run, range(8)))
    assert len(ids) == 1
    assert _checkpoint_count(workspace) == 1
    assert len(_files(workspace, "weights")) == 1
