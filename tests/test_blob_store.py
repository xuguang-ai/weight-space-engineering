"""Per-tensor content-addressed blobs, stored dtypes, and migration 8 (M2 plan Sections 3.1 and 3.2).

A blob holds one tensor's raw little-endian row-major bytes in its stored
dtype; its ID is the SHA-256 of those bytes, so published hashes compare
directly. Checkpoints reference blobs through their weights index; recovery
counts references.
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import struct
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from wse.api import Engine
from wse.database import LATEST_SCHEMA_VERSION, MIGRATIONS
from wse.policy import SMALL_MODEL_POLICY_ID
from wse.schemas import ErrorCode, IOSpec, ModelManifest, Provenance, TensorInfo, WSEError
from wse.store import ArtifactStore
from wse.tensors import blob_id, dtype_name, weights_hash

TASK = "signed_sum_v1"
ROW1 = [{"start": 1, "stop": 2, "step": 1}, {"start": 0, "stop": 4, "step": 1}]
ORACLE = {"op": "scale_slice", "tensor": "layers.1.weight", "selection": ROW1, "factor": -1.0}

# Bit patterns per stored dtype: +0, -0, smallest and largest subnormals (both
# signs), smallest normal, largest finite (both signs), and 1.0.
BIT_PATTERNS = {
    torch.bfloat16: (torch.int16, "<H", [0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F, 0x0080, 0x7F7F, 0xFF7F,
                                         0x3F80]),
    torch.float16: (torch.int16, "<H", [0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF, 0x0400, 0x7BFF, 0xFBFF,
                                        0x3C00]),
    torch.float32: (torch.int32, "<I", [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF,
                                        0x00800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x3F800000]),
}

# Store layouts: M1's single safetensors weights file, and the blob index that
# checkpoints above the single-file limit use.
LAYOUTS = {"single_file": None, "blob_index": 0}


def _from_bits(dtype: torch.dtype, shape: tuple[int, ...]) -> tuple[torch.Tensor, bytes]:
    int_dtype, fmt, patterns = BIT_PATTERNS[dtype]
    count = 1
    for size in shape:
        count *= size
    values = (patterns * (count // len(patterns) + 1))[:count]
    packed = struct.pack("<" + fmt[1] * count, *values)
    tensor = torch.frombuffer(bytearray(packed), dtype=dtype).reshape(shape).clone()
    assert tensor.view(int_dtype).numpy().tobytes() == packed
    return tensor, packed


def _manifest(tensors: dict[str, torch.Tensor], *, parent: str | None = None) -> ModelManifest:
    infos = [
        TensorInfo(name=name, shape=list(t.shape), dtype=dtype_name(t), axes=["unknown"] * t.dim(),
                   element_count=t.numel())
        for name, t in sorted(tensors.items())
    ]
    return ModelManifest(
        checkpoint_id=ArtifactStore.new_checkpoint_id(),
        weights_hash=weights_hash(tensors),
        config_hash="sha256:" + "0" * 64,
        adapter="test_blobs@1",
        config={},
        tensors=infos,
        parameter_count=sum(info.element_count for info in infos),
        input_semantics=IOSpec(shape=["batch"], dtype="float32"),
        output_semantics=IOSpec(shape=["batch"], dtype="float32"),
        provenance=Provenance(kind="import" if parent is None else "patch", parent_checkpoint_id=parent),
        license="test",
    )


def _commit(store: ArtifactStore, tensors: dict[str, torch.Tensor], *, parent: str | None = None) -> ModelManifest:
    return store.commit_checkpoint(_manifest(tensors, parent=parent), tensors, dedup_key=None,
                                   event_type="checkpoint.imported")


def _store(workspace: Path, layout: str) -> ArtifactStore:
    limit = LAYOUTS[layout]
    return ArtifactStore(workspace) if limit is None else ArtifactStore(workspace, single_file_max_bytes=limit)


def _blob_files(workspace: Path) -> list[Path]:
    directory = workspace / "artifacts" / "blobs"
    return sorted(directory.iterdir()) if directory.is_dir() else []


def _blob_rows(store: ArtifactStore) -> int:
    return store.db.conn.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]


def _model() -> dict[str, torch.Tensor]:
    return {
        "a.weight": _from_bits(torch.bfloat16, (4, 6))[0],
        "b.weight": _from_bits(torch.float16, (3, 5))[0],
        "c.bias": _from_bits(torch.float32, (7,))[0],
    }


# -- blob identity ---------------------------------------------------------------------------------


def test_blob_id_is_sha256_of_little_endian_row_major_bytes(workspace):
    subnormal = 2.0 ** -133  # the smallest positive BF16 subnormal, 0x0001
    tensor = torch.tensor([[1.0, -2.0, 0.5], [-0.0, subnormal, 3.0]], dtype=torch.float32).to(torch.bfloat16)
    expected_bits = [0x3F80, 0xC000, 0x3F00, 0x8000, 0x0001, 0x4040]
    packed = struct.pack("<6H", *expected_bits)
    assert tensor.view(torch.int16).flatten().tolist() == [struct.unpack("<h", struct.pack("<H", b))[0]
                                                            for b in expected_bits]
    assert blob_id(tensor) == hashlib.sha256(packed).hexdigest()
    # A non-contiguous view hashes its logical row-major order, not its storage order.
    transposed = struct.pack("<6H", *[expected_bits[i] for i in (0, 3, 1, 4, 2, 5)])
    assert blob_id(tensor.t()) == hashlib.sha256(transposed).hexdigest()

    with ArtifactStore(workspace) as store:
        manifest = _commit(store, {"w": tensor})
        (blob_file,) = _blob_files(workspace)
        assert blob_file.name == hashlib.sha256(packed).hexdigest()
        assert blob_file.read_bytes() == packed
        assert store.tensor_blob_ids(manifest.checkpoint_id) == {"w": blob_file.name}


# -- deduplication ---------------------------------------------------------------------------------


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_a_candidate_differing_in_one_tensor_adds_one_blob(workspace, layout):
    with _store(workspace, layout) as store:
        base_tensors = _model()
        base = _commit(store, base_tensors)
        assert len(_blob_files(workspace)) == _blob_rows(store) == 3
        weights_before = sorted((workspace / "artifacts" / "weights").iterdir())

        changed = dict(base_tensors)
        changed["b.weight"] = -base_tensors["b.weight"]
        candidate = _commit(store, changed, parent=base.checkpoint_id)
        assert len(_blob_files(workspace)) == _blob_rows(store) == 4
        base_ids, candidate_ids = (store.tensor_blob_ids(m.checkpoint_id) for m in (base, candidate))
        assert {name for name in base_ids if base_ids[name] != candidate_ids[name]} == {"b.weight"}
        assert store.blob_reference_counts() == {
            **{base_ids[name]: 2 for name in ("a.weight", "c.bias")},
            base_ids["b.weight"]: 1,
            candidate_ids["b.weight"]: 1,
        }
        new_weights = sorted(set((workspace / "artifacts" / "weights").iterdir()) - set(weights_before))
        assert len(new_weights) == 1
        if layout == "blob_index":
            # The candidate's weights artifact is an index of blob IDs, not a copy of the tensors.
            assert new_weights[0].suffix == ".json"
            assert new_weights[0].stat().st_size < 1024
        else:
            assert new_weights[0].suffix == ".safetensors"

        # Recommitting identical weights adds nothing.
        again = _commit(store, changed, parent=base.checkpoint_id)
        assert again.weights_hash == candidate.weights_hash
        assert len(_blob_files(workspace)) == _blob_rows(store) == 4


def test_tensors_with_identical_bytes_share_one_blob(workspace):
    tensor = _from_bits(torch.bfloat16, (2, 5))[0]
    with ArtifactStore(workspace, single_file_max_bytes=0) as store:
        manifest = _commit(store, {"embed.weight": tensor, "lm_head.weight": tensor.clone()})
        ids = store.tensor_blob_ids(manifest.checkpoint_id)
        assert ids["embed.weight"] == ids["lm_head.weight"]
        assert len(_blob_files(workspace)) == 1
        assert store.blob_reference_counts() == {ids["embed.weight"]: 2}
        loaded = store.load_tensors(manifest.checkpoint_id)
        assert loaded["embed.weight"].data_ptr() != loaded["lm_head.weight"].data_ptr()


# -- stored dtypes ---------------------------------------------------------------------------------


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_bf16_f16_f32_round_trip_bitwise_including_signed_zero_and_subnormals(workspace, layout):
    tensors, packed = {}, {}
    for dtype in BIT_PATTERNS:
        name = f"{dtype_name(torch.empty(0, dtype=dtype))}.weight"
        tensors[name], packed[name] = _from_bits(dtype, (2, 10))
    with _store(workspace, layout) as store:
        manifest = _commit(store, tensors)
        ids = store.tensor_blob_ids(manifest.checkpoint_id)
        loaded = store.load_tensors(manifest.checkpoint_id)
        for name, tensor in tensors.items():
            int_dtype = BIT_PATTERNS[tensor.dtype][0]
            assert (workspace / "artifacts" / "blobs" / ids[name]).read_bytes() == packed[name]
            assert loaded[name].dtype == tensor.dtype and loaded[name].shape == tensor.shape
            assert loaded[name].view(int_dtype).numpy().tobytes() == packed[name]
            region = store.read_slice(manifest.checkpoint_id, name, (slice(1, 2), slice(0, 10)))
            assert region.dtype == tensor.dtype
            assert region.view(int_dtype).numpy().tobytes() == packed[name][len(packed[name]) // 2:]
            whole = store.read_slice(manifest.checkpoint_id, name, ())
            assert whole.view(int_dtype).numpy().tobytes() == packed[name]
        exported = workspace.parent / f"export-{layout}.safetensors"
        store.export_safetensors(manifest.checkpoint_id, exported)
        roundtrip = load_file(str(exported))
        assert sorted(roundtrip) == sorted(tensors)
        for name, tensor in tensors.items():
            int_dtype = BIT_PATTERNS[tensor.dtype][0]
            assert roundtrip[name].view(int_dtype).numpy().tobytes() == packed[name]


def test_zero_dimensional_and_empty_tensors_round_trip(workspace):
    tensors = {"scalar": torch.tensor(-0.0, dtype=torch.bfloat16), "empty": torch.empty((0, 3), dtype=torch.float16)}
    with ArtifactStore(workspace, single_file_max_bytes=0) as store:
        manifest = _commit(store, tensors)
        loaded = store.load_tensors(manifest.checkpoint_id)
        assert loaded["scalar"].shape == () and loaded["scalar"].view(torch.int16).item() == -0x8000
        assert loaded["empty"].shape == (0, 3) and loaded["empty"].dtype == torch.float16
        assert store.tensor_blob_ids(manifest.checkpoint_id)["empty"] == hashlib.sha256(b"").hexdigest()


# -- corruption and recovery -----------------------------------------------------------------------


def _tamper(path: Path) -> None:
    path.chmod(0o644)
    data = bytearray(path.read_bytes())
    data[0] ^= 0x01
    path.write_bytes(bytes(data))


@pytest.mark.parametrize("detected_by", ["load", "recovery"])
@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_a_tampered_blob_marks_every_checkpoint_that_uses_it_corrupt(workspace, layout, detected_by):
    with _store(workspace, layout) as store:
        base_tensors = _model()
        base = _commit(store, base_tensors)
        changed = {**base_tensors, "b.weight": -base_tensors["b.weight"]}
        candidate = _commit(store, changed, parent=base.checkpoint_id)
        unrelated = _commit(store, {"x": _from_bits(torch.float32, (3,))[0]})
        shared = store.tensor_blob_ids(base.checkpoint_id)["a.weight"]
        _tamper(workspace / "artifacts" / "blobs" / shared)
        if detected_by == "load":
            with pytest.raises(WSEError) as excinfo:
                store.load_tensors(candidate.checkpoint_id)
            assert excinfo.value.code == ErrorCode.ARTIFACT_CORRUPT
        else:
            report = store.recover()
            assert sorted(report.corrupted_checkpoints) == sorted([base.checkpoint_id, candidate.checkpoint_id])
        assert store.checkpoint_state(base.checkpoint_id) == "CORRUPT"
        assert store.checkpoint_state(candidate.checkpoint_id) == "CORRUPT"
        assert store.checkpoint_state(unrelated.checkpoint_id) == "READY"
        assert store.list_ready() == [unrelated.checkpoint_id]
        for checkpoint_id in (base.checkpoint_id, candidate.checkpoint_id):
            with pytest.raises(WSEError) as excinfo:
                store.read_slice(checkpoint_id, "c.bias", ())
            assert excinfo.value.code == ErrorCode.ARTIFACT_CORRUPT
        store.load_tensors(unrelated.checkpoint_id)


def test_a_missing_blob_is_detected_when_reading_one_slice(workspace):
    with ArtifactStore(workspace, single_file_max_bytes=0) as store:
        manifest = _commit(store, _model())
        (workspace / "artifacts" / "blobs" / store.tensor_blob_ids(manifest.checkpoint_id)["c.bias"]).unlink()
        assert store.read_slice(manifest.checkpoint_id, "a.weight", ()).shape == (4, 6)
        with pytest.raises(WSEError) as excinfo:
            store.read_slice(manifest.checkpoint_id, "c.bias", ())
        assert excinfo.value.code == ErrorCode.ARTIFACT_CORRUPT
        assert store.checkpoint_state(manifest.checkpoint_id) == "CORRUPT"


def test_recovery_removes_only_unreferenced_blobs(workspace):
    with ArtifactStore(workspace, single_file_max_bytes=0) as store:
        manifest = _commit(store, _model())
        referenced = set(store.tensor_blob_ids(manifest.checkpoint_id).values())
        orphan = workspace / "artifacts" / "blobs" / ("f" * 64)
        orphan.write_bytes(b"left by a crashed commit")
        assert store.recover().removed_blobs == []  # within the grace period
        report = store.recover(grace_seconds=0)
        assert report.removed_blobs == ["f" * 64]
        assert report.removed_orphans == [] and report.corrupted_checkpoints == []
        assert {path.name for path in _blob_files(workspace)} == referenced
        assert store.load_tensors(manifest.checkpoint_id).keys() == _model().keys()


# -- migration 8 -----------------------------------------------------------------------------------


def _m1_schema(path: Path) -> list[tuple[str, str, str]]:
    with sqlite3.connect(path) as conn:
        rows = conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall()
    return rows


def _rows(conn: sqlite3.Connection, table: str) -> list[tuple]:
    return conn.execute(f"SELECT * FROM {table} ORDER BY 1, 2").fetchall()


def _downgrade_to_m1(workspace: Path) -> None:
    """Remove every migration-8 object so the workspace is exactly what M1 (schema 7) wrote."""
    conn = sqlite3.connect(workspace / "metadata.sqlite")
    try:
        conn.execute("DROP TABLE imported_patches")  # migration 9
        conn.execute("DROP TABLE weight_tensors")
        conn.execute("DROP TABLE blobs")
        conn.execute("PRAGMA user_version = 7")
        conn.commit()
    finally:
        conn.close()
    shutil.rmtree(workspace / "artifacts" / "blobs")


def test_migration_8_keeps_m1_identities(workspace, bundles, tmp_path):
    with Engine(workspace) as engine:
        base = engine.import_bundle(bundles["task"])
        grant = engine.grant_run(TASK, SMALL_MODEL_POLICY_ID, base.checkpoint_id)
        experiment = engine.create_experiment(grant.run_id, TASK, SMALL_MODEL_POLICY_ID, "create-1")
        spec = {"base_checkpoint_id": base.checkpoint_id, "hypothesis": "One output row has the wrong sign.",
                "operations": [ORACLE]}
        preview = engine.preview_patch(experiment.experiment_id, spec, "preview-1")
        candidate = engine.apply_patch(experiment.experiment_id, preview.patch_id, "apply-1")
        engine.select_checkpoint(experiment.experiment_id, candidate.checkpoint_id, base.checkpoint_id, "select-1")
    _downgrade_to_m1(workspace)
    _assert_migration_8_keeps_identities(workspace, base, candidate, experiment.experiment_id, tmp_path)


def _assert_migration_8_keeps_identities(workspace: Path, base: ModelManifest, candidate: ModelManifest,
                                         experiment_id: str, tmp_path: Path) -> None:
    """Open an M1 (schema 7) workspace with the current code and check that nothing M1 wrote changed."""
    assert LATEST_SCHEMA_VERSION >= 8 and [v for v, _ in MIGRATIONS] == list(range(1, LATEST_SCHEMA_VERSION + 1))

    # The downgraded database has exactly the M1 schema (migrations 1-7).
    reference = tmp_path / "m1.sqlite"
    conn = sqlite3.connect(reference)
    for _, statements in MIGRATIONS[:7]:
        for statement in statements:
            conn.execute(statement)
    conn.execute("PRAGMA user_version = 7")
    conn.commit()
    conn.close()
    assert _m1_schema(workspace / "metadata.sqlite") == _m1_schema(reference)

    tables = ("artifacts", "checkpoints", "experiments", "experiment_checkpoints", "budget_ledger", "patches")
    with sqlite3.connect(workspace / "metadata.sqlite") as conn:
        before_rows = {table: _rows(conn, table) for table in tables}
    artifact_files = {path.relative_to(workspace): path.read_bytes()
                      for directory in ("weights", "manifests", "patches")
                      for path in (workspace / "artifacts" / directory).iterdir()}
    m1_tensors = {path.name: load_file(str(path)) for path in (workspace / "artifacts" / "weights").iterdir()}

    with Engine(workspace) as engine:
        assert engine.store.db.schema_version() == LATEST_SCHEMA_VERSION
        assert engine.last_recovery.corrupted_checkpoints == []
        with sqlite3.connect(workspace / "metadata.sqlite") as conn:
            assert {table: _rows(conn, table) for table in tables} == before_rows
        assert {path.relative_to(workspace): path.read_bytes()
                for directory in ("weights", "manifests", "patches")
                for path in (workspace / "artifacts" / directory).iterdir()} == artifact_files
        for manifest in (base, candidate):
            assert engine.load_manifest(manifest.checkpoint_id) == manifest
            loaded = engine.store.load_tensors(manifest.checkpoint_id)
            assert weights_hash(loaded) == manifest.weights_hash
            original = m1_tensors[manifest.weights_hash.removeprefix("sha256:") + ".safetensors"]
            ids = engine.store.tensor_blob_ids(manifest.checkpoint_id)
            assert sorted(ids) == sorted(original)
            for name, tensor in original.items():
                assert loaded[name].numpy().tobytes() == tensor.numpy().tobytes()
                assert ids[name] == blob_id(tensor)
                assert (workspace / "artifacts" / "blobs" / ids[name]).read_bytes() == tensor.numpy().tobytes()
        state = engine.get_experiment(experiment_id)
        assert state.selected_checkpoint_id == candidate.checkpoint_id
        assert state.budget["candidate_commits"].consumed == 1
        # Base and candidate differ in one tensor, so they share every other blob.
        assert len(_blob_files(workspace)) == len(m1_tensors[base.weights_hash.removeprefix("sha256:")
                                                             + ".safetensors"]) + 1
        indexed = [e for e in engine.store.db.conn.execute(
            "SELECT event_type FROM audit_events WHERE event_type = 'weights.blobs_indexed'")]
        assert len(indexed) == 2

    # Reopening is a no-op: the migration runs once.
    with Engine(workspace) as engine:
        assert engine.store.db.conn.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = 'weights.blobs_indexed'").fetchone()[0] == 2
        assert engine.store.recover(grace_seconds=0).removed_blobs == []


# The last commit whose code is M1 (schema version 7, single safetensors weights files).
M1_COMMIT = "32dc45d"
REPO_ROOT = Path(__file__).resolve().parents[1]
M1_WRITER = """
import json, sys
from pathlib import Path
import wse
from wse.api import Engine
from wse.policy import SMALL_MODEL_POLICY_ID
m1_src, bundle, workspace, oracle = Path(sys.argv[1]), sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
assert Path(wse.__file__).resolve().is_relative_to(m1_src.resolve()), wse.__file__
with Engine(workspace) as engine:
    assert engine.store.db.schema_version() == 7
    base = engine.import_bundle(bundle)
    grant = engine.grant_run("signed_sum_v1", SMALL_MODEL_POLICY_ID, base.checkpoint_id)
    experiment = engine.create_experiment(grant.run_id, "signed_sum_v1", SMALL_MODEL_POLICY_ID, "create-1")
    spec = {"base_checkpoint_id": base.checkpoint_id, "hypothesis": "One output row has the wrong sign.",
            "operations": [oracle]}
    preview = engine.preview_patch(experiment.experiment_id, spec, "preview-1")
    candidate = engine.apply_patch(experiment.experiment_id, preview.patch_id, "apply-1")
    engine.select_checkpoint(experiment.experiment_id, candidate.checkpoint_id, base.checkpoint_id, "select-1")
print(json.dumps({"base": base.model_dump(mode="json"), "candidate": candidate.model_dump(mode="json"),
                  "experiment_id": experiment.experiment_id}))
"""


def _m1_source(destination: Path) -> Path | None:
    """Extract M1's ``src`` tree from git history, or return None when it is unavailable."""
    import io
    import subprocess
    import tarfile

    git = shutil.which("git")
    if git is None:
        return None
    archive = subprocess.run([git, "-C", str(REPO_ROOT), "archive", "--format=tar", M1_COMMIT, "src"],
                             capture_output=True, timeout=60)
    if archive.returncode != 0:
        return None
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(destination, filter="data")
    return destination / "src"


def test_migration_8_on_a_workspace_written_by_m1_code(workspace, bundles, tmp_path):
    """The M1 workspace comes from M1's own code (git history), not from a downgrade of new output."""
    import json
    import os
    import subprocess
    import sys

    m1_src = _m1_source(tmp_path / "m1")
    if m1_src is None:
        pytest.skip(f"M1 code (commit {M1_COMMIT}) is not available from git history")
    env = {**os.environ, "PYTHONPATH": str(m1_src)}
    written = subprocess.run(
        [sys.executable, "-c", M1_WRITER, str(m1_src), str(bundles["task"]), str(workspace), json.dumps(ORACLE)],
        capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120)
    assert written.returncode == 0, written.stderr
    result = json.loads(written.stdout)
    assert not (workspace / "artifacts" / "blobs").exists()
    with sqlite3.connect(workspace / "metadata.sqlite") as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 7
    base, candidate = (ModelManifest.model_validate(result[key]) for key in ("base", "candidate"))
    _assert_migration_8_keeps_identities(workspace, base, candidate, result["experiment_id"], tmp_path)


def test_slices_of_scalar_and_empty_tensors_and_unknown_names(workspace):
    tensors = {"scalar": torch.tensor(-0.0, dtype=torch.bfloat16), "empty": torch.empty((0, 3), dtype=torch.float16)}
    with ArtifactStore(workspace, single_file_max_bytes=0) as store:
        manifest = _commit(store, tensors)
        scalar = store.read_slice(manifest.checkpoint_id, "scalar", ())
        assert scalar.shape == () and scalar.view(torch.int16).item() == -0x8000
        assert store.read_slice(manifest.checkpoint_id, "empty", ()).shape == (0, 3)
        with pytest.raises(WSEError) as excinfo:
            store.read_slice(manifest.checkpoint_id, "missing", ())
        assert excinfo.value.code == ErrorCode.NOT_FOUND


def test_an_index_is_never_registered_under_another_weights_hash(workspace):
    tensors = _model()
    wrong = _manifest({**tensors, "c.bias": -tensors["c.bias"]})
    with ArtifactStore(workspace, single_file_max_bytes=0) as store:
        with pytest.raises(WSEError) as excinfo:
            store.commit_checkpoint(wrong, tensors, dedup_key=None, event_type="checkpoint.imported")
        assert excinfo.value.code == ErrorCode.ARTIFACT_CORRUPT
        assert store.list_ready() == [] and store.blob_reference_counts() == {}


def test_nonfinite_values_are_counted_across_chunks(monkeypatch):
    import wse.tensors as tensors

    monkeypatch.setattr(tensors, "_FINITE_CHUNK", 4)
    values = torch.zeros(3, 5, dtype=torch.bfloat16)
    values[0, 1], values[1, 3], values[2, 4] = float("nan"), float("inf"), float("-inf")
    assert tensors.nonfinite_count(values) == 3
    assert tensors.nonfinite_count(values.t()) == 3  # a non-contiguous view
    assert tensors.nonfinite_count(torch.zeros(0)) == 0
    assert tensors.nonfinite_count(torch.tensor(float("nan"))) == 1
