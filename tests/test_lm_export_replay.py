"""Export and replay of an experiment on an imported language model.

An open-weight model's patches keep their after-values in binary blocks, and
its checkpoints are imported from snapshots, not bundles. The export
therefore holds each binary patch file and the base checkpoint as its
manifest, its tensors under their canonical names, and its model files;
``replay_export`` restores that base (checking its weights hash and every
file's hash), replays each candidate from its parent with the blocks, and
must reproduce every recorded weights hash.
"""

from __future__ import annotations

import json

import pytest
from conftest import HF_SKIP_REASON
from test_llm_policy_mcp import _open_run, lm  # noqa: F401 - a fixture
from test_probe_jobs import DOWN0

from wse.api import Engine
from wse.audit import EXPORT_MANIFEST_NAME, export_experiment, replay_export

REGION = [{"start": 0, "stop": 2, "step": 1}, {"start": 0, "stop": 3, "step": 1}]


def _edited_run(lm):
    """A run with a candidate and a candidate of that candidate: (experiment ID, child, grandchild)."""
    run_id, eid = _open_run(lm.workspace, lm.base)
    with Engine(lm.workspace, run_id=run_id) as bound:
        first = bound.preview_patch(eid, {
            "base_checkpoint_id": lm.base, "hypothesis": "Fill a region of the first down projection.",
            "operations": [{"op": "fill_slice", "tensor": f"{DOWN0}.weight", "selection": REGION, "value": 0.25}]},
            "p1")
        child = bound.apply_patch(eid, first.patch_id, "a1").checkpoint_id
        second = bound.preview_patch(eid, {
            "base_checkpoint_id": child, "hypothesis": "Double the filled region.",
            "operations": [{"op": "scale_slice", "tensor": f"{DOWN0}.weight", "selection": REGION, "factor": 2.0}]},
            "p2")
        grandchild = bound.apply_patch(eid, second.patch_id, "a2").checkpoint_id
    return eid, child, grandchild


def test_an_open_weight_experiment_exports_and_replays_byte_identically(lm, tmp_path) -> None:
    pytest.importorskip("transformers", reason=HF_SKIP_REASON)
    eid, child, grandchild = _edited_run(lm)
    out = tmp_path / "export"
    with Engine(lm.workspace) as engine:
        manifest = export_experiment(engine, eid, out)
        recorded = {c.checkpoint_id: c.weights_hash for c in manifest.checkpoints}
        files = engine.model_files(lm.base)
        source = engine.load_manifest(lm.base).source
    assert manifest.replay["base_format"] == "checkpoint"
    assert (out / "base_model" / "model.safetensors").is_file()
    for name in files:
        assert (out / "base_model" / "files" / name).read_bytes() == files[name]
    for patch in manifest.patches:
        assert patch.blocks_path is not None and manifest.files[patch.blocks_path].startswith("sha256:")
    with Engine(tmp_path / "clean") as clean:
        replayed = replay_export(out, clean)
        restored = clean.load_manifest(replayed.base_checkpoint_id)
    assert replayed.all_identical and replayed.base_weights_hash == recorded[lm.base]
    assert {c["checkpoint_id"]: c["replayed_weights_hash"] for c in replayed.candidates} == {
        child: recorded[child], grandchild: recorded[grandchild]}
    assert restored.source == source and restored.checkpoint_id != lm.base


def test_a_changed_exported_tensor_fails_the_replay(lm, tmp_path) -> None:
    pytest.importorskip("transformers", reason=HF_SKIP_REASON)
    from safetensors.torch import load_file, save_file

    from wse.schemas import ErrorCode, WSEError

    eid, _, _ = _edited_run(lm)
    out = tmp_path / "export"
    with Engine(lm.workspace) as engine:
        export_experiment(engine, eid, out)
    weights = out / "base_model" / "model.safetensors"
    tensors = load_file(str(weights))
    tensors[f"{DOWN0}.weight"][0, 0] += 1.0
    save_file(tensors, str(weights))
    # The file's hash is recorded, so the replay refuses the export before restoring anything.
    with Engine(tmp_path / "clean") as clean, pytest.raises(WSEError) as caught:
        replay_export(out, clean)
    assert caught.value.code == ErrorCode.ARTIFACT_CORRUPT
    data = json.loads((out / EXPORT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert data["replay"]["base_format"] == "checkpoint"
