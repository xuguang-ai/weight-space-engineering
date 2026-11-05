"""Crash injection around file publication and database commits, and restart recovery.

Each crash runs in a subprocess that dies with ``os._exit`` at the injected
point, so no cleanup handler runs, exactly as in a killed process.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import textwrap

import pytest

from wse.api import Engine
from wse.database import LATEST_SCHEMA_VERSION, MIGRATIONS
from wse.policy import SMALL_MODEL_POLICY_ID
from wse.schemas import ErrorCode, WSEError
from wse.tensors import weights_hash

TASK = "signed_sum_v1"
ROW1 = [{"start": 1, "stop": 2, "step": 1}, {"start": 0, "stop": 4, "step": 1}]
ORACLE = {"op": "scale_slice", "tensor": "layers.1.weight", "selection": ROW1, "factor": -1.0}
CRASH_EXIT = 17

_CRASH_SCRIPT = """
import os, sys
import wse.store as store_module
from wse.api import Engine
from wse.experiments import ExperimentService
from wse.store import ArtifactStore

workspace, experiment_id, argument, key, point = sys.argv[1:6]
real_replace = os.replace

def crash():
    sys.stdout.flush()
    os._exit({exit})

def replace(src, dst, *args, **kwargs):
    target = str(dst)
    if point == "before_rename:weights" and "/weights/" in target:
        crash()
    real_replace(src, dst, *args, **kwargs)
    if point == "after_rename:manifests" and "/manifests/" in target:
        crash()
    if point == "after_rename:patches" and "/patches/" in target:
        crash()

store_module.os.replace = replace

if point == "inside_commit":
    original_remember = ExperimentService._remember
    def remember(*args, **kwargs):
        original_remember(*args, **kwargs)
        crash()  # the write transaction is open and uncommitted
    ExperimentService._remember = staticmethod(remember)

if point == "after_commit:apply":
    original_commit = ArtifactStore.commit_checkpoint
    def commit(self, *args, **kwargs):
        manifest = original_commit(self, *args, **kwargs)
        print(manifest.checkpoint_id)
        crash()
    ArtifactStore.commit_checkpoint = commit

if point == "after_commit:preview":
    original_preview = ExperimentService._preview
    def preview(self, *args, **kwargs):
        result = original_preview(self, *args, **kwargs)
        print(result.patch_id)
        crash()
    ExperimentService._preview = preview

with Engine(workspace, recover=False) as engine:
    if point.endswith("patches") or point == "after_commit:preview":
        import json
        engine.preview_patch(experiment_id, json.loads(argument), key)
    else:
        engine.apply_patch(experiment_id, argument, key)
print("no crash")
""".replace("{exit}", str(CRASH_EXIT))


def _crash(workspace, experiment_id, argument, key, point) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_CRASH_SCRIPT), str(workspace), experiment_id, argument, key, point],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == CRASH_EXIT, (result.returncode, result.stdout, result.stderr)
    return result.stdout.strip()


@pytest.fixture
def setup(workspace, bundles):
    """A base checkpoint, an experiment, and one stored preview; the engine is closed."""
    with Engine(workspace) as engine:
        base = engine.import_bundle(bundles["task"])
        grant = engine.grant_run(TASK, SMALL_MODEL_POLICY_ID, base.checkpoint_id)
        experiment = engine.create_experiment(grant.run_id, TASK, SMALL_MODEL_POLICY_ID, "create-1")
        spec = {"base_checkpoint_id": base.checkpoint_id, "hypothesis": "One output row has the wrong sign.",
                "operations": [ORACLE]}
        preview = engine.preview_patch(experiment.experiment_id, spec, "preview-1")
        base_bytes = engine.store.weights_path(base.checkpoint_id).read_bytes()
    return {"base": base, "experiment": experiment, "preview": preview, "spec": spec, "base_bytes": base_bytes}


def _files(workspace, directory):
    return sorted(p.name for p in (workspace / "artifacts" / directory).iterdir())


def _assert_base_untouched(engine, setup):
    base = setup["base"]
    assert engine.store.weights_path(base.checkpoint_id).read_bytes() == setup["base_bytes"]
    assert weights_hash(engine.store.load_tensors(base.checkpoint_id)) == base.weights_hash


@pytest.mark.parametrize("point", ["before_rename:weights", "after_rename:manifests", "inside_commit"])
def test_crash_before_the_commit_leaves_no_candidate_and_no_charge(workspace, setup, point):
    experiment_id = setup["experiment"].experiment_id
    _crash(workspace, experiment_id, setup["preview"].patch_id, "apply-1", point)
    with Engine(workspace) as engine:
        report = engine.store.recover(grace_seconds=0)
        if point == "before_rename:weights":
            assert len(report.removed_temp_files) == 1
        else:
            assert {name.split("/")[0] for name in report.removed_orphans} == {"weights", "manifests"}
        assert _files(workspace, "tmp") == []
        assert engine.store.list_ready() == [setup["base"].checkpoint_id]
        experiment = engine.get_experiment(experiment_id)
        assert experiment.budget["candidate_commits"].consumed == 0
        assert experiment.selected_checkpoint_id == setup["base"].checkpoint_id
        assert len(engine.list_experiment_checkpoints(experiment_id)) == 1
        _assert_base_untouched(engine, setup)
        # The retry with the same key succeeds and is charged exactly once.
        candidate = engine.apply_patch(experiment_id, setup["preview"].patch_id, "apply-1")
        assert candidate.weights_hash == setup["preview"].result_weights_hash
        assert engine.get_experiment(experiment_id).budget["candidate_commits"].consumed == 1
        assert engine.store.recover(grace_seconds=0).removed_orphans == []


def test_crash_after_the_commit_is_completed_by_the_idempotent_retry(workspace, setup):
    experiment_id = setup["experiment"].experiment_id
    committed = _crash(workspace, experiment_id, setup["preview"].patch_id, "apply-1", "after_commit:apply")
    with Engine(workspace) as engine:
        assert engine.store.recover(grace_seconds=0).removed_orphans == []
        assert engine.store.checkpoint_state(committed) == "READY"
        assert engine.get_experiment(experiment_id).budget["candidate_commits"].consumed == 1
        retried = engine.apply_patch(experiment_id, setup["preview"].patch_id, "apply-1")
        assert retried.checkpoint_id == committed
        assert engine.get_experiment(experiment_id).budget["candidate_commits"].consumed == 1
        assert engine.get_experiment(experiment_id).selected_checkpoint_id == setup["base"].checkpoint_id
        _assert_base_untouched(engine, setup)


def test_crash_after_publishing_a_preview_leaves_an_invisible_orphan(workspace, setup):
    experiment_id = setup["experiment"].experiment_id
    spec = {**setup["spec"], "hypothesis": "A second idea."}
    import json

    _crash(workspace, experiment_id, json.dumps(spec), "preview-2", "after_rename:patches")
    with Engine(workspace) as engine:
        assert len(_files(workspace, "patches")) == 2  # published, but no row references it
        assert engine.store.db.conn.execute("SELECT COUNT(*) FROM patches").fetchone()[0] == 1
        # The retry republishes the same content-addressed file and commits it.
        preview = engine.preview_patch(experiment_id, spec, "preview-2")
        assert engine.get_patch(experiment_id, preview.patch_id) == preview
        assert engine.store.recover(grace_seconds=0).removed_orphans == []
        assert len(_files(workspace, "patches")) == 2
        assert engine.store.db.conn.execute("SELECT COUNT(*) FROM patches").fetchone()[0] == 2


def test_orphaned_preview_file_is_removed_by_recovery(workspace, setup):
    experiment_id = setup["experiment"].experiment_id
    spec = {**setup["spec"], "operations": [{**ORACLE, "factor": 2.0}]}
    import json

    _crash(workspace, experiment_id, json.dumps(spec), "preview-3", "after_rename:patches")
    with Engine(workspace) as engine:
        assert len(_files(workspace, "patches")) == 2
        report = engine.store.recover(grace_seconds=0)
        assert [name.split("/")[0] for name in report.removed_orphans] == ["patches"]
        assert len(_files(workspace, "patches")) == 1
        assert engine.store.db.conn.execute("SELECT COUNT(*) FROM patches").fetchone()[0] == 1


def test_crash_after_a_preview_commit_returns_the_recorded_preview(workspace, setup):
    experiment_id = setup["experiment"].experiment_id
    spec = {**setup["spec"], "hypothesis": "Committed before the crash."}
    import json

    patch_id = _crash(workspace, experiment_id, json.dumps(spec), "preview-4", "after_commit:preview")
    with Engine(workspace) as engine:
        assert engine.preview_patch(experiment_id, spec, "preview-4").patch_id == patch_id


def test_state_is_consistent_after_restart(workspace, setup):
    experiment_id = setup["experiment"].experiment_id
    base_id = setup["base"].checkpoint_id
    with Engine(workspace) as engine:
        candidate = engine.apply_patch(experiment_id, setup["preview"].patch_id, "apply-1")
        engine.select_checkpoint(experiment_id, candidate.checkpoint_id, base_id, "select-1")
        engine.select_checkpoint(experiment_id, base_id, candidate.checkpoint_id, "select-2")
        before = (engine.get_experiment(experiment_id), engine.list_experiment_checkpoints(experiment_id),
                  engine.audit_log(experiment_id))
    with Engine(workspace) as engine:
        after = (engine.get_experiment(experiment_id), engine.list_experiment_checkpoints(experiment_id),
                 engine.audit_log(experiment_id))
        assert after == before
        assert after[0].selected_checkpoint_id == base_id
        assert after[0].budget["candidate_commits"].consumed == 1
        assert engine.select_checkpoint(experiment_id, candidate.checkpoint_id, base_id, "select-1") == \
            engine.experiments.select_checkpoint(experiment_id, candidate.checkpoint_id, base_id, "select-1")


def test_corrupt_patch_artifact_cannot_be_applied(workspace, setup):
    experiment_id = setup["experiment"].experiment_id
    with Engine(workspace) as engine:
        (name,) = _files(workspace, "patches")
        path = workspace / "artifacts" / "patches" / name
        path.chmod(0o644)
        data = path.read_bytes()
        assert b'"values":[1.0,-1.0,1.0,-1.0]' in data
        path.write_bytes(data.replace(b'"values":[1.0,-1.0,1.0,-1.0]', b'"values":[2.0,-1.0,1.0,-1.0]'))
        with pytest.raises(WSEError) as excinfo:
            engine.apply_patch(experiment_id, setup["preview"].patch_id, "apply-1")
        assert excinfo.value.code == ErrorCode.ARTIFACT_CORRUPT
        assert engine.store.list_ready() == [setup["base"].checkpoint_id]
        experiment = engine.get_experiment(experiment_id)
        assert experiment.budget["candidate_commits"].consumed == 0
        assert experiment.selected_checkpoint_id == setup["base"].checkpoint_id
        events = [e["event_type"] for e in engine.audit_log(experiment_id)]
        assert events[-2:] == ["patch.corrupt", "patch.apply.failed"]


def test_recovery_marks_missing_patch_files_corrupt(workspace, setup):
    (name,) = _files(workspace, "patches")
    (workspace / "artifacts" / "patches" / name).unlink()
    with Engine(workspace) as engine:
        assert engine.last_recovery.corrupted_patches == [setup["preview"].patch_id]
        with pytest.raises(WSEError) as excinfo:
            engine.apply_patch(setup["experiment"].experiment_id, setup["preview"].patch_id, "apply-1")
        assert excinfo.value.code == ErrorCode.ARTIFACT_CORRUPT


def test_corrupt_candidate_cannot_be_selected_and_the_selection_is_kept(workspace, setup):
    experiment_id = setup["experiment"].experiment_id
    base_id = setup["base"].checkpoint_id
    with Engine(workspace) as engine:
        candidate = engine.apply_patch(experiment_id, setup["preview"].patch_id, "apply-1")
        weights = engine.store.weights_path(candidate.checkpoint_id)
    weights.chmod(0o644)
    weights.write_bytes(weights.read_bytes()[:-4] + b"\x00\x00\x80\x7f")
    with Engine(workspace) as engine:
        assert engine.last_recovery.corrupted_checkpoints == [candidate.checkpoint_id]
        with pytest.raises(WSEError) as excinfo:
            engine.select_checkpoint(experiment_id, candidate.checkpoint_id, base_id, "select-1")
        assert excinfo.value.code == ErrorCode.ARTIFACT_CORRUPT
        assert engine.get_experiment(experiment_id).selected_checkpoint_id == base_id
        assert engine.audit_log(experiment_id)[-1]["event_type"] == "experiment.select.failed"
        _assert_base_untouched(engine, setup)


def test_version_1_databases_are_migrated_in_place(workspace, bundles):
    workspace.mkdir(parents=True)
    with sqlite3.connect(workspace / "metadata.sqlite") as conn:
        for statement in MIGRATIONS[0][1]:
            conn.execute(statement)
        conn.execute("PRAGMA user_version = 1")
    with Engine(workspace) as engine:
        base = engine.import_bundle(bundles["task"])
        grant = engine.grant_run(TASK, SMALL_MODEL_POLICY_ID, base.checkpoint_id)
        experiment = engine.create_experiment(grant.run_id, TASK, SMALL_MODEL_POLICY_ID, "create-1")
        assert engine.store.db.schema_version() == LATEST_SCHEMA_VERSION
        assert engine.audit_log(experiment.experiment_id)[0]["event_type"] == "experiment.created"
