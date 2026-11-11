"""Records of stored patches (``Engine.store_patch``): one record per base and patch, also under a race.

* The record of a patch made in direct use is keyed by its base checkpoint
  and the exact patch (with its binary blocks), not by its label: storing it
  again under another label returns the first record, with its first label,
  and creates no second candidate.
* Two processes storing the same patch at once (or importing the same edit)
  both get the one record: the commit that loses sees the winner's record
  inside its own transaction, rolls back, and returns it.

Models are the tiny offline fixtures of ``tests/hf_fixtures.py``.
"""

from __future__ import annotations

import pytest

from conftest import HF_SKIP_REASON
from wse.api import Engine, apply_operator

pytest.importorskip("transformers", reason=HF_SKIP_REASON)

REPO = "example-org/tiny-model"
REVISION = "0123456789abcdef0123456789abcdef01234567"
DOWN = "model.layers.0.mlp.down_proj.weight"  # [16, 32]


def _counts(engine: Engine) -> dict[str, int]:
    conn = engine.store.db.conn
    return {table: conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
            for table in ("checkpoints", "imported_patches")}


@pytest.fixture
def stored(hf_snapshots, workspace):
    with Engine(workspace) as engine:
        base_id = engine.import_hf_snapshot(hf_snapshots["llama_untied"], repo=REPO, revision=REVISION).checkpoint_id
        edit = apply_operator(engine.open_checkpoint(base_id), "scale_slice", {
            "tensor": DOWN, "selection": [{"start": 0, "stop": 16, "step": 1}, {"start": 0, "stop": 8, "step": 1}],
            "factor": 0.5})
        yield engine, base_id, edit


def test_storing_a_patch_again_under_another_label_returns_its_first_record(stored):
    engine, base_id, edit = stored
    before = _counts(engine)
    first = engine.store_patch(edit.patch, edit.blocks, base_checkpoint_id=base_id, label="first")
    again = engine.store_patch(edit.patch, edit.blocks, base_checkpoint_id=base_id, label="second")
    assert again == first and again.source.run == "first"
    assert _counts(engine) == {"checkpoints": before["checkpoints"] + 1,
                               "imported_patches": before["imported_patches"] + 1}
    # Another patch on the same base is another record.
    other = apply_operator(engine.open_checkpoint(base_id), "scale_slice", {
        "tensor": DOWN, "selection": [{"start": 0, "stop": 16, "step": 1}, {"start": 0, "stop": 8, "step": 1}],
        "factor": 0.25})
    assert engine.store_patch(other.patch, other.blocks, base_checkpoint_id=base_id, label="first") != first


def test_a_store_that_loses_a_race_returns_the_winners_record(stored, workspace, monkeypatch):
    engine, base_id, edit = stored
    before = _counts(engine)
    store = engine.store
    original = store.commit_checkpoint
    won = []

    def race(*args, **kwargs):
        # Another process stores the same patch after this one's check and before its commit.
        with Engine(workspace) as other:
            won.append(other.store_patch(edit.patch, edit.blocks, base_checkpoint_id=base_id, label="race"))
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "commit_checkpoint", race)
    # One label, so that the two keys agree whatever the key includes.
    record = engine.store_patch(edit.patch, edit.blocks, base_checkpoint_id=base_id, label="race")
    assert record == won[0]  # the winner's record, with the winner's patch ID
    assert _counts(engine) == {"checkpoints": before["checkpoints"] + 1,
                               "imported_patches": before["imported_patches"] + 1}
    assert engine.load_manifest(record.checkpoint_id).provenance.source == record.patch_id
