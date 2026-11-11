"""Storing patches made in direct use (``Engine.store_patch``): the audit event, refusals, and rollback.

The patch is replayed on the stored base before anything is stored; a
tampered patch stores nothing, and a failure inside the commit leaves no
checkpoint, record, file reference, or audit event. Models are the tiny
offline fixtures of ``tests/hf_fixtures.py``.
"""

from __future__ import annotations

import json

import pytest

from conftest import HF_SKIP_REASON
from wse.api import Engine, apply_operator
from wse.schemas import ErrorCode, WSEError

pytest.importorskip("transformers", reason=HF_SKIP_REASON)

REPO = "example-org/tiny-model"
REVISION = "0123456789abcdef0123456789abcdef01234567"
DOWN = "model.layers.0.mlp.down_proj.weight"  # [16, 32]


def _expect(code, func, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        func(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


def _counts(engine: Engine) -> dict[str, int]:
    conn = engine.store.db.conn
    return {table: conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
            for table in ("checkpoints", "imported_patches", "audit_events")}


@pytest.fixture
def stored(hf_snapshots, workspace):
    with Engine(workspace) as engine:
        base_id = engine.import_hf_snapshot(hf_snapshots["llama_untied"], repo=REPO, revision=REVISION).checkpoint_id
        base = engine.open_checkpoint(base_id)
        edit = apply_operator(base, "scale_slice", {
            "tensor": DOWN, "selection": [{"start": 0, "stop": 16, "step": 1}, {"start": 0, "stop": 8, "step": 1}],
            "factor": 0.5})
        yield engine, base_id, edit


def test_a_stored_patch_writes_its_candidate_record_and_audit_event(stored):
    engine, base_id, edit = stored
    record = engine.store_patch(edit.patch, edit.blocks, base_checkpoint_id=base_id, label="demo")
    assert record.source.format == "wse-direct-v1" and record.source.run == "demo"
    assert record.result_weights_hash == edit.candidate.weights_hash
    assert [t.tensor for t in record.tensors] == [DOWN] and record.changed_elements == 128
    manifest = engine.load_manifest(record.checkpoint_id)
    assert manifest.provenance.parent_checkpoint_id == base_id and manifest.provenance.source == record.patch_id
    event = engine.store.db.conn.execute(
        "SELECT payload FROM audit_events WHERE event_type = 'patch.stored' AND subject_id = ?",
        (record.patch_id,)).fetchone()
    payload = json.loads(event["payload"])
    assert payload["checkpoint_id"] == record.checkpoint_id and payload["base_checkpoint_id"] == base_id
    patch, blocks = engine.read_imported_patch(record.patch_id)
    assert patch == edit.patch and dict(blocks) == dict(edit.blocks)


def test_a_tampered_patch_stores_nothing(stored):
    engine, base_id, edit = stored
    before = _counts(engine)
    step = edit.patch.steps[0]
    forged = edit.patch.model_copy(update={"steps": [step.model_copy(update={"after_hash": "sha256:" + "0" * 64})]})
    with pytest.raises(WSEError) as caught:
        engine.store_patch(forged, edit.blocks, base_checkpoint_id=base_id)
    assert caught.value.code in (ErrorCode.INVALID_PATCH, ErrorCode.ARTIFACT_CORRUPT)
    tampered_blocks = {block_id: b"\x00" * len(data) for block_id, data in edit.blocks.items()}
    with pytest.raises(WSEError):
        engine.store_patch(edit.patch, tampered_blocks, base_checkpoint_id=base_id)
    assert _counts(engine) == before


def test_a_failure_inside_the_commit_rolls_everything_back(stored, monkeypatch):
    import wse.database as database

    engine, base_id, edit = stored
    before = _counts(engine)

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure inside the commit")

    monkeypatch.setattr(database, "insert_audit_event", fail)
    with pytest.raises(RuntimeError):
        engine.store_patch(edit.patch, edit.blocks, base_checkpoint_id=base_id)
    monkeypatch.undo()
    assert _counts(engine) == before
    # The same patch stores normally afterwards.
    assert engine.store_patch(edit.patch, edit.blocks, base_checkpoint_id=base_id).changed_elements == 128
