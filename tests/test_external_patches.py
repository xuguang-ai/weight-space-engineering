"""Importing published edits in the ``lm-understand-v1`` format (M2 plan Section 3.4, Task 5).

An edit directory holds ``manifest.json`` (per tensor: name, shape, dtype,
form, ``base_sha256``, ``edited_sha256``, changed entries) and ``edit.npz``
(``t{i}_idx`` and ``t{i}_val`` for every tensor, plus ``t{i}_a`` and
``t{i}_b`` for the low-rank form). The importer verifies every base hash,
rebuilds each tensor as the article's ``apply_edit.py`` does, verifies every
edited hash, and only then records an exact patch and its candidate. The
edit directories here are synthetic, written by the tests on the tiny Llama
fixture; the expected tensors are computed independently with torch.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import torch

from wse.api import Engine
from wse.schemas import ErrorCode, WSEError
from wse.tensors import blob_id

REPO = "example-org/tiny-model"
REVISION = "0123456789abcdef0123456789abcdef01234567"
# The provenance every import must name (Section 3.4).
EDIT_SOURCE = {"repo": "example-org/edits", "revision": "abc123"}
SPARSE = "model.layers.0.mlp.down_proj.weight"  # [16, 32]
LOWRANK = "model.layers.1.self_attn.o_proj.weight"  # [16, 16]


def _expect(code, func, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        func(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


def _u16(tensor: torch.Tensor) -> np.ndarray:
    return tensor.contiguous().view(torch.int16).numpy().view(np.uint16).copy()


def _sha(bits: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(bits).tobytes()).hexdigest()


@pytest.fixture
def base(hf_snapshots, workspace):
    with Engine(workspace) as engine:
        manifest = engine.import_hf_snapshot(hf_snapshots["llama_untied"], repo=REPO, revision=REVISION)
        handle = engine.open_checkpoint(manifest.checkpoint_id)
    return manifest.checkpoint_id, handle


def _write_edit(directory: Path, handle, *, break_base: str | None = None, break_edited: str | None = None,
                extra: dict | None = None) -> dict[str, np.ndarray]:
    """A two-tensor edit: a sparse overwrite and a rank-2 update with two fixups. Returns the edited bits."""
    rng = np.random.default_rng(0)
    sparse_base = _u16(handle._tensors[SPARSE])
    lowrank_base = _u16(handle._tensors[LOWRANK])
    # Sparse: 40 entries overwritten with new BF16 bit patterns (one is rewritten to its own value).
    idx = np.sort(rng.choice(sparse_base.size, size=40, replace=False)).astype(np.int32)
    val = _u16((torch.randn(40, generator=torch.Generator().manual_seed(1)) * 0.05).to(torch.bfloat16))
    val[0] = sparse_base.reshape(-1)[idx[0]]
    sparse_edited = sparse_base.reshape(-1).copy()
    sparse_edited[idx] = val
    sparse_edited = sparse_edited.reshape(sparse_base.shape)
    # Low-rank: W' = bf16(f32(f64(W) + a[:, 0] b[0] + a[:, 1] b[1])), then two exact overwrites.
    a = rng.normal(0, 0.02, size=(16, 2))
    b = rng.normal(0, 0.02, size=(2, 16))
    w = handle._tensors[LOWRANK].double()
    for k in range(2):
        w = w + torch.outer(torch.from_numpy(a[:, k]), torch.from_numpy(b[k]))
    lowrank_edited = _u16(w.float().to(torch.bfloat16)).reshape(-1)
    fix_idx = np.array([3, 200], dtype=np.int32)
    fix_val = np.array([0x3F80, 0xBC00], dtype=np.uint16)
    lowrank_edited[fix_idx] = fix_val
    lowrank_edited = lowrank_edited.reshape(lowrank_base.shape)

    entries = [
        {"name": SPARSE, "shape": [16, 32], "dtype": "bfloat16", "form": "sparse",
         "changed_entries": int((sparse_edited != sparse_base).sum()),
         "base_sha256": _sha(sparse_base), "edited_sha256": _sha(sparse_edited)},
        {"name": LOWRANK, "shape": [16, 16], "dtype": "bfloat16", "form": "lowrank", "rank": 2,
         "changed_entries": int((lowrank_edited != lowrank_base).sum()),
         "base_sha256": _sha(lowrank_base), "edited_sha256": _sha(lowrank_edited)},
    ]
    for entry in entries:
        if entry["name"] == break_base:
            entry["base_sha256"] = "0" * 64
        if entry["name"] == break_edited:
            entry["edited_sha256"] = "f" * 64
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps({"tensors": entries, "changed_tensors": 2}, indent=1))
    arrays = {"t0_idx": idx, "t0_val": val, "t1_a": a, "t1_b": b, "t1_idx": fix_idx, "t1_val": fix_val,
              **(extra or {})}
    np.savez(directory / "edit.npz", **arrays)
    return {SPARSE: sparse_edited, LOWRANK: lowrank_edited}


def _state(workspace: Path) -> tuple[list[str], list[tuple], list[tuple]]:
    """Every artifact file, checkpoint row, and imported-patch row of a workspace."""
    files = sorted(str(p.relative_to(workspace)) for p in (workspace / "artifacts").rglob("*") if p.is_file())
    with sqlite3.connect(workspace / "metadata.sqlite") as conn:
        checkpoints = conn.execute("SELECT checkpoint_id FROM checkpoints ORDER BY 1").fetchall()
        patches = conn.execute("SELECT patch_id FROM imported_patches ORDER BY 1").fetchall()
    return files, checkpoints, patches


def test_lm_understand_v1_sparse_and_lowrank_rebuild(base, workspace, tmp_path):
    from wse.external_patches import read_lm_understand, rebuild
    from wse.patches import replay_patch

    checkpoint_id, handle = base
    edit_dir = tmp_path / "runs" / "p00x" / "edit"
    expected = _write_edit(edit_dir, handle)

    # rebuild() alone, on the stored bit patterns, as Section 3.4 states it.
    edit = read_lm_understand(edit_dir)
    for i, name in enumerate((SPARSE, LOWRANK)):
        rebuilt = rebuild(_u16(handle._tensors[name]), edit.entries[i], edit.arrays, i)
        assert rebuilt.dtype == np.uint16 and np.array_equal(rebuilt, expected[name])

    with Engine(workspace) as engine:
        blobs_before = set(engine.store.blob_reference_counts())
        imported = engine.import_patch(edit_dir, base_checkpoint_id=checkpoint_id, repo="example-org/edits",
                                       revision="abc123")
        candidate = engine.open_checkpoint(imported.checkpoint_id)
        # Every edited tensor has its published hash; nothing else changed; two new blobs.
        for name in (SPARSE, LOWRANK):
            assert blob_id(candidate._tensors[name]) == _sha(expected[name])
        assert set(engine.store.blob_reference_counts()) - blobs_before == {_sha(expected[SPARSE]),
                                                                              _sha(expected[LOWRANK])}
        assert all(blob_id(candidate._tensors[n]) == blob_id(handle._tensors[n])
                   for n in handle.tensor_names() if n not in (SPARSE, LOWRANK))
        manifest = engine.load_manifest(imported.checkpoint_id)
        assert manifest.provenance.parent_checkpoint_id == checkpoint_id
        assert manifest.provenance.kind == "patch" and manifest.provenance.source == imported.patch_id
        # The record: per-tensor verification and the source's provenance with file hashes.
        assert [(t.tensor, t.form, t.base_sha256, t.edited_sha256) for t in imported.tensors] == [
            (SPARSE, "sparse", blob_id(handle._tensors[SPARSE]), _sha(expected[SPARSE])),
            (LOWRANK, "lowrank", blob_id(handle._tensors[LOWRANK]), _sha(expected[LOWRANK]))]
        assert [t.changed_elements for t in imported.tensors] == [39, int((expected[LOWRANK] !=
                                                                         _u16(handle._tensors[LOWRANK])).sum())]
        source = imported.source
        assert (source.format, source.repository, source.revision, source.run) == \
            ("lm-understand-v1", "example-org/edits", "abc123", "p00x")
        assert source.files == {name: hashlib.sha256((edit_dir / name).read_bytes()).hexdigest()
                                for name in ("manifest.json", "edit.npz")}
        # The stored exact patch replays onto the base without the edit directory.
        patch, blocks = engine.read_imported_patch(imported.patch_id)
        assert patch.source == source and patch.result_weights_hash == candidate.weights_hash
        assert [step.block.kind for step in patch.steps] == ["sparse", "dense"]
        assert replay_patch(handle, patch, blocks=blocks).weights_hash == candidate.weights_hash
        # Importing the same edit again records nothing new.
        state = _state(workspace)
        again = engine.import_patch(edit_dir, base_checkpoint_id=checkpoint_id, repo="example-org/edits",
                                    revision="abc123")
        assert again == imported and _state(workspace) == state


def test_wse_import_patch_cli(base, workspace, tmp_path, capsys):
    from wse.cli import main

    checkpoint_id, handle = base
    edit_dir = tmp_path / "edit"
    expected = _write_edit(edit_dir, handle)
    command = ["--workspace", str(workspace), "import-patch", "--format", "lm-understand-v1", str(edit_dir),
               "--base", checkpoint_id]
    for missing in (["--repo", "example-org/edits"], ["--revision", "abc123"]):  # provenance is required
        with pytest.raises(SystemExit) as excinfo:
            main(command + missing)
        assert excinfo.value.code == 2
    capsys.readouterr()
    code = main(command + ["--repo", "example-org/edits", "--revision", "abc123"])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["base_checkpoint_id"] == checkpoint_id
    assert [t["edited_sha256"] for t in printed["tensors"]] == [_sha(expected[SPARSE]), _sha(expected[LOWRANK])]
    assert printed["source"]["repository"] == "example-org/edits" and printed["source"]["revision"] == "abc123"
    with pytest.raises(SystemExit) as excinfo:  # an unknown format is a usage error
        main(["--workspace", str(workspace), "import-patch", "--format", "other-v1", str(edit_dir),
              "--base", checkpoint_id, "--repo", "example-org/edits", "--revision", "abc123"])
    assert excinfo.value.code == 2


def test_a_wrong_base_hash_is_refused_before_any_write(base, workspace, tmp_path, monkeypatch):
    import wse.external_patches as external

    checkpoint_id, handle = base
    edit_dir = tmp_path / "edit"
    _write_edit(edit_dir, handle, break_base=LOWRANK)
    calls = []
    real = external.rebuild
    monkeypatch.setattr(external, "rebuild", lambda *a, **k: calls.append(1) or real(*a, **k))
    with Engine(workspace) as engine:
        before = _state(workspace)
        error = _expect(ErrorCode.VERSION_CONFLICT, engine.import_patch, edit_dir,
                        base_checkpoint_id=checkpoint_id, **EDIT_SOURCE)
        assert LOWRANK in error.message and "base" in error.message
        assert _state(workspace) == before
    # Every base hash is checked before any tensor is rebuilt.
    assert calls == []


def test_a_wrong_edited_hash_leaves_no_patch_and_no_candidate(base, workspace, tmp_path):
    checkpoint_id, handle = base
    edit_dir = tmp_path / "edit"
    _write_edit(edit_dir, handle, break_edited=LOWRANK)
    with Engine(workspace) as engine:
        before = _state(workspace)
        error = _expect(ErrorCode.INVALID_PATCH, engine.import_patch, edit_dir, base_checkpoint_id=checkpoint_id, **EDIT_SOURCE)
        assert LOWRANK in error.message and "edited" in error.message
        assert _state(workspace) == before


def test_malformed_edits_are_refused(base, workspace, tmp_path):
    checkpoint_id, handle = base
    cases = {
        "pickled": dict(extra={"t9_obj": np.array([{"x": 1}], dtype=object)}),
    }
    with Engine(workspace) as engine:
        before = _state(workspace)
        for label, kwargs in cases.items():
            edit_dir = tmp_path / label
            _write_edit(edit_dir, handle, **kwargs)
            _expect(ErrorCode.INVALID_PATCH, engine.import_patch, edit_dir, base_checkpoint_id=checkpoint_id, **EDIT_SOURCE)
        # An unknown tensor, an index out of range, and a missing array.
        edit_dir = tmp_path / "edit"
        _write_edit(edit_dir, handle)
        manifest = json.loads((edit_dir / "manifest.json").read_text())
        for change, code in ((lambda m: m["tensors"][0].update(name="model.no_such.weight"), ErrorCode.NOT_FOUND),
                             (lambda m: m["tensors"][0].update(shape=[32, 16]), ErrorCode.INVALID_PATCH),
                             (lambda m: m["tensors"][1].update(form="svd"), ErrorCode.INVALID_PATCH)):
            broken = json.loads(json.dumps(manifest))
            change(broken)
            (edit_dir / "manifest.json").write_text(json.dumps(broken))
            _expect(code, engine.import_patch, edit_dir, base_checkpoint_id=checkpoint_id, **EDIT_SOURCE)
        (edit_dir / "manifest.json").write_text(json.dumps(manifest))
        arrays = dict(np.load(edit_dir / "edit.npz"))
        arrays["t0_idx"] = arrays["t0_idx"].copy()
        arrays["t0_idx"][0] = 16 * 32
        np.savez(edit_dir / "edit.npz", **arrays)
        _expect(ErrorCode.INVALID_PATCH, engine.import_patch, edit_dir, base_checkpoint_id=checkpoint_id, **EDIT_SOURCE)
        del arrays["t1_b"]
        np.savez(edit_dir / "edit.npz", **arrays)
        _expect(ErrorCode.INVALID_PATCH, engine.import_patch, edit_dir, base_checkpoint_id=checkpoint_id, **EDIT_SOURCE)
        assert _state(workspace) == before


def test_a_run_bound_engine_cannot_import_patches(base, workspace, tmp_path):
    from wse.policy import SMALL_MODEL_POLICY_ID

    checkpoint_id, handle = base
    edit_dir = tmp_path / "edit"
    _write_edit(edit_dir, handle)
    with Engine(workspace) as operator:
        grant = operator.grant_run("import_patch_v1", SMALL_MODEL_POLICY_ID, checkpoint_id)
    with Engine(workspace, run_id=grant.run_id) as bound:
        _expect(ErrorCode.POLICY_DENIED, bound.import_patch, edit_dir, base_checkpoint_id=checkpoint_id, **EDIT_SOURCE)


def test_import_requires_the_source_repository_and_revision(base, workspace, tmp_path):
    checkpoint_id, handle = base
    edit_dir = tmp_path / "edit"
    _write_edit(edit_dir, handle)
    with Engine(workspace) as engine:
        before = _state(workspace)
        for source in ({"repo": "", "revision": "abc123"}, {"repo": "example-org/edits", "revision": " "}):
            _expect(ErrorCode.INVALID_ARGUMENT, engine.import_patch, edit_dir, base_checkpoint_id=checkpoint_id,
                    **source)
        with pytest.raises(TypeError):
            engine.import_patch(edit_dir, base_checkpoint_id=checkpoint_id)
        assert _state(workspace) == before


def test_a_changed_entries_mismatch_is_refused(base, workspace, tmp_path):
    checkpoint_id, handle = base
    edit_dir = tmp_path / "edit"
    _write_edit(edit_dir, handle)
    manifest = json.loads((edit_dir / "manifest.json").read_text())
    manifest["tensors"][1]["changed_entries"] += 1
    (edit_dir / "manifest.json").write_text(json.dumps(manifest))
    with Engine(workspace) as engine:
        before = _state(workspace)
        error = _expect(ErrorCode.INVALID_PATCH, engine.import_patch, edit_dir, base_checkpoint_id=checkpoint_id,
                        **EDIT_SOURCE)
        assert LOWRANK in error.message and "changes" in error.message
        assert _state(workspace) == before


def test_archive_members_are_checked_from_their_headers_before_decompression(base, workspace, tmp_path,
                                                                              monkeypatch):
    """A small compressed archive cannot force a large allocation before it is refused."""
    import wse.external_patches as external

    checkpoint_id, handle = base
    loaded = []
    real = external._read_member
    monkeypatch.setattr(external, "_read_member", lambda archive, name: loaded.append(name) or real(archive, name))

    def refused(edit_dir):
        loaded.clear()
        with Engine(workspace) as engine:
            _expect(ErrorCode.INVALID_PATCH, engine.import_patch, edit_dir, base_checkpoint_id=checkpoint_id,
                    **EDIT_SOURCE)
        assert loaded == []

    # An unused member: 256 MiB of zeros compresses to a few hundred KiB.
    edit_dir = tmp_path / "bomb"
    _write_edit(edit_dir, handle)
    arrays = dict(np.load(edit_dir / "edit.npz"))
    np.savez_compressed(edit_dir / "edit.npz", zz=np.zeros(256 * 1024 * 1024, dtype=np.uint8), **arrays)
    assert (edit_dir / "edit.npz").stat().st_size < 2 * 1024 * 1024
    refused(edit_dir)
    # A used member whose header claims more entries than the tensor has.
    big = dict(arrays, t0_idx=np.zeros(16 * 32 + 1, dtype=np.int32), t0_val=np.zeros(16 * 32 + 1, dtype=np.uint16))
    np.savez_compressed(edit_dir / "edit.npz", **big)
    refused(edit_dir)
    # Low-rank factors whose rank exceeds the tensor's smaller side.
    wide = dict(arrays, t1_a=np.zeros((16, 17)), t1_b=np.zeros((17, 16)))
    np.savez_compressed(edit_dir / "edit.npz", **wide)
    refused(edit_dir)
    # A member that is not a .npy array.
    import zipfile
    np.savez(edit_dir / "edit.npz", **arrays)
    with zipfile.ZipFile(edit_dir / "edit.npz", "a") as archive:
        archive.writestr("notes.txt", b"hello")
    refused(edit_dir)
    # The well-formed archive still loads every member once.
    np.savez_compressed(edit_dir / "edit.npz", **arrays)
    with Engine(workspace) as engine:
        engine.import_patch(edit_dir, base_checkpoint_id=checkpoint_id, **EDIT_SOURCE)
    assert sorted(loaded) == sorted(f"{key}.npy" for key in arrays)


def test_recovery_reports_a_missing_or_altered_imported_patch_file(base, workspace, tmp_path):
    checkpoint_id, handle = base
    edit_dir = tmp_path / "edit"
    _write_edit(edit_dir, handle)
    with Engine(workspace) as engine:
        record = engine.import_patch(edit_dir, base_checkpoint_id=checkpoint_id, **EDIT_SOURCE)
    with sqlite3.connect(workspace / "metadata.sqlite") as conn:
        (relpath,) = conn.execute("SELECT relpath FROM imported_patches").fetchone()
    path = workspace / "artifacts" / relpath
    data = path.read_bytes()
    path.chmod(0o644)  # published artifacts are read-only
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with Engine(workspace) as engine:
        assert engine.last_recovery.corrupted_patches == [record.patch_id]
        _expect(ErrorCode.ARTIFACT_CORRUPT, engine.read_imported_patch, record.patch_id)
        engine.open_checkpoint(record.checkpoint_id)  # the candidate's blobs are intact
    path.unlink()
    with Engine(workspace) as engine:
        assert engine.last_recovery.corrupted_patches == [record.patch_id]
    path.write_bytes(data)
    with Engine(workspace) as engine:
        assert engine.last_recovery.corrupted_patches == []
        engine.read_imported_patch(record.patch_id)
