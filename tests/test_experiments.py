"""Experiment state: run grants, idempotency, ownership, budgets, and CAS selection."""

from __future__ import annotations

import threading

import pytest
import torch
from pydantic import ValidationError

from wse.api import Engine
from wse.policy import (
    SMALL_MODEL_POLICY_ID,
    WEIGHTS_ONLY_POLICY_ID,
    BudgetLimits,
    ExperimentPolicy,
    builtin_experiment_policies,
)
from wse.schemas import ErrorCode, WSEError

TASK = "signed_sum_v1"


def _sel(*ranges):
    return [{"start": start, "stop": stop, "step": 1} for start, stop in ranges]


def _expect(code, func, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        func(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


ROW1 = _sel((1, 2), (0, 4))
ORACLE = {"op": "scale_slice", "tensor": "layers.1.weight", "selection": ROW1, "factor": -1.0}


def _spec(base, *operations, **extra):
    return {"base_checkpoint_id": base, "hypothesis": "One output row has the wrong sign.",
            "operations": list(operations or [ORACLE]), **extra}


def _bias_op(value):
    return {"op": "set_slice", "tensor": "layers.0.bias", "selection": _sel((0, 1)), "values": [value]}


@pytest.fixture
def base(engine, bundles):
    return engine.import_bundle(bundles["task"])


@pytest.fixture
def grant(engine, base):
    return engine.grant_run(TASK, SMALL_MODEL_POLICY_ID, base.checkpoint_id)


@pytest.fixture
def experiment(engine, grant):
    return engine.create_experiment(grant.run_id, TASK, SMALL_MODEL_POLICY_ID, "create-1")


def _in_threads(workspace, count, call):
    """Run ``call(engine, index)`` concurrently, each on its own Engine and connection."""
    barrier = threading.Barrier(count)
    outcomes: list[object] = [None] * count

    def run(index):
        with Engine(workspace, recover=False) as own:
            barrier.wait()
            try:
                outcomes[index] = call(own, index)
            except WSEError as exc:
                outcomes[index] = exc.code

    threads = [threading.Thread(target=run, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return outcomes


def _accuracy(engine, checkpoint_id, signed_sum):
    inputs, labels, _ = signed_sum.generate_examples(32, seed=11)
    handle = engine.open_checkpoint(checkpoint_id)  # operator-side check only
    predictions = handle.forward(torch.from_numpy(inputs)).argmax(dim=1)
    return (predictions == torch.from_numpy(labels)).float().mean().item()


# -- policies -------------------------------------------------------------------------


def test_builtin_policies_have_the_planned_limits(engine):
    bounded = engine.get_policy(SMALL_MODEL_POLICY_ID)
    assert bounded.mode == "bounded_eval"
    assert bounded.budget == BudgetLimits(
        candidate_commits=20, analysis_calls=50, tensor_read_calls=100, evaluation_batches=10,
        examples_per_batch=64, evaluated_examples=640, seconds_per_operation=30, compute_seconds=300,
    )
    weights_only = engine.get_policy(WEIGHTS_ONLY_POLICY_ID)
    assert weights_only.mode == "weights_only"
    assert weights_only.budget.candidate_commits == 1
    assert (weights_only.budget.evaluation_batches, weights_only.budget.examples_per_batch,
            weights_only.budget.evaluated_examples, weights_only.budget.model_forward_calls) == (0, 0, 0, 0)
    same = {"analysis_calls", "tensor_read_calls", "seconds_per_operation", "compute_seconds"}
    assert {k: getattr(weights_only.budget, k) for k in same} == {k: getattr(bounded.budget, k) for k in same}


def test_policies_are_immutable(engine):
    changed = builtin_experiment_policies()[SMALL_MODEL_POLICY_ID].model_copy(
        update={"budget": BudgetLimits(candidate_commits=99)})
    _expect(ErrorCode.VERSION_CONFLICT, engine.register_policy, changed)
    engine.register_policy(builtin_experiment_policies()[SMALL_MODEL_POLICY_ID])  # identical: no-op
    with pytest.raises(ValidationError):
        ExperimentPolicy(policy_id="leaky_v1", mode="weights_only")  # default budget grants evaluation
    _expect(ErrorCode.NOT_FOUND, engine.get_policy, "no_such_policy")


# -- run grants and creation ----------------------------------------------------------


def test_create_experiment_from_a_run_grant(engine, base, grant, experiment):
    assert experiment.run_id == grant.run_id and experiment.task_id == TASK
    assert experiment.policy_id == SMALL_MODEL_POLICY_ID and experiment.mode == "bounded_eval"
    assert experiment.policy_hash == engine.get_policy(SMALL_MODEL_POLICY_ID).policy_hash
    assert experiment.base_checkpoint_id == experiment.selected_checkpoint_id == base.checkpoint_id
    assert experiment.state == "OPEN" and experiment.ended_at is None
    assert experiment.budget["candidate_commits"].limit == 20
    assert experiment.budget["candidate_commits"].remaining == 20
    assert experiment.experiment_id.startswith("exp_") and grant.run_id.startswith("run_")
    assert engine.get_experiment(experiment.experiment_id) == experiment


def test_one_experiment_per_run_and_retries_return_it(engine, grant, experiment):
    again = engine.create_experiment(grant.run_id, TASK, SMALL_MODEL_POLICY_ID, "create-1")
    other_key = engine.create_experiment(grant.run_id, TASK, SMALL_MODEL_POLICY_ID, "create-2")
    assert again.experiment_id == other_key.experiment_id == experiment.experiment_id
    assert engine.store.db.conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 1


def test_task_and_policy_changes_are_rejected(engine, grant, experiment):
    _expect(ErrorCode.POLICY_DENIED, engine.create_experiment, grant.run_id, TASK, WEIGHTS_ONLY_POLICY_ID, "c3")
    _expect(ErrorCode.POLICY_DENIED, engine.create_experiment, grant.run_id, "other_task", SMALL_MODEL_POLICY_ID, "c4")
    # Reusing a key with different input is a conflict, not a new experiment.
    _expect(ErrorCode.IDEMPOTENCY_CONFLICT, engine.create_experiment, grant.run_id, TASK, WEIGHTS_ONLY_POLICY_ID,
            "create-1")
    _expect(ErrorCode.NOT_FOUND, engine.create_experiment, "run_" + "0" * 32, TASK, SMALL_MODEL_POLICY_ID, "c5")
    _expect(ErrorCode.INVALID_ARGUMENT, engine.create_experiment, grant.run_id, TASK, SMALL_MODEL_POLICY_ID, "")
    _expect(ErrorCode.INVALID_ARGUMENT, engine.create_experiment, grant.run_id, TASK, SMALL_MODEL_POLICY_ID, "a b")
    assert engine.get_experiment(experiment.experiment_id).mode == "bounded_eval"


def test_recreating_an_experiment_cannot_reset_its_budget(workspace, engine, base, grant, experiment):
    preview = engine.preview_patch(experiment.experiment_id, _spec(base.checkpoint_id), "p1")
    engine.apply_patch(experiment.experiment_id, preview.patch_id, "a1")
    again = engine.create_experiment(grant.run_id, TASK, SMALL_MODEL_POLICY_ID, "create-new-key")
    assert again.experiment_id == experiment.experiment_id
    assert again.budget["candidate_commits"].consumed == 1
    engine.close()
    with Engine(workspace) as restarted:
        after = restarted.create_experiment(grant.run_id, TASK, SMALL_MODEL_POLICY_ID, "create-after-restart")
        assert after.budget["candidate_commits"].consumed == 1
        assert after.budget["candidate_commits"].remaining == 19


def test_grants_require_registered_policies_and_ready_checkpoints(engine, base):
    _expect(ErrorCode.NOT_FOUND, engine.grant_run, TASK, "no_such_policy", base.checkpoint_id)
    _expect(ErrorCode.NOT_FOUND, engine.grant_run, TASK, SMALL_MODEL_POLICY_ID, "ckpt_" + "0" * 32)
    _expect(ErrorCode.INVALID_ARGUMENT, engine.grant_run, "bad task", SMALL_MODEL_POLICY_ID, base.checkpoint_id)


# -- previews ---------------------------------------------------------------------------


def test_preview_freezes_the_spec_with_the_server_resolved_hash(engine, base, experiment):
    preview = engine.preview_patch(experiment.experiment_id, _spec(base.checkpoint_id, development_suite_id="dev_v1",
                                                                    expected_behavior="Accuracy rises."), "p1")
    assert preview.base_weights_hash == base.weights_hash
    assert preview.patch_id.startswith("patch_") and preview.patch_uri == "wse://patches/" + preview.patch_id
    assert preview.changed_elements == 4 and preview.parameter_count == 22
    assert preview.changed_fraction == pytest.approx(4 / 22)
    assert preview.l2_delta == pytest.approx(4.0) and preview.max_abs_delta == 2.0
    assert preview.all_finite and preview.storage.candidate_bytes == 22 * 4 and preview.storage.patch_bytes > 0
    assert preview.usage.model_forward_calls == 0 and preview.usage.candidate_commits == 0
    assert {diff.tensor for diff in preview.tensors} == {"layers.0.weight", "layers.0.bias", "layers.1.weight",
                                                          "layers.1.bias"}
    artifact = engine.read_patch(experiment.experiment_id, preview.patch_id)
    assert artifact.spec.base_weights_hash == base.weights_hash
    assert artifact.spec.development_suite_id == "dev_v1"
    assert engine.get_patch(experiment.experiment_id, preview.patch_id) == preview
    # Previewing does not switch the selected model or create a checkpoint.
    assert engine.get_experiment(experiment.experiment_id).selected_checkpoint_id == base.checkpoint_id
    assert engine.store.list_ready() == [base.checkpoint_id]


def test_client_supplied_base_hash_is_rejected(engine, base, experiment):
    spec = _spec(base.checkpoint_id, base_weights_hash=base.weights_hash)
    error = _expect(ErrorCode.INVALID_PATCH, engine.preview_patch, experiment.experiment_id, spec, "p1")
    assert "base content hash" in error.message
    _expect(ErrorCode.INVALID_PATCH, engine.preview_patch, experiment.experiment_id,
            {"base_checkpoint_id": base.checkpoint_id, "operations": [ORACLE]}, "p2")  # no hypothesis


def test_preview_is_idempotent(engine, base, experiment):
    first = engine.preview_patch(experiment.experiment_id, _spec(base.checkpoint_id), "p1")
    again = engine.preview_patch(experiment.experiment_id, _spec(base.checkpoint_id), "p1")
    assert again == first
    assert engine.store.db.conn.execute("SELECT COUNT(*) FROM patches").fetchone()[0] == 1
    _expect(ErrorCode.IDEMPOTENCY_CONFLICT, engine.preview_patch, experiment.experiment_id,
            _spec(base.checkpoint_id, _bias_op(2.0)), "p1")
    # A new key creates a new, separate preview of the same spec.
    second = engine.preview_patch(experiment.experiment_id, _spec(base.checkpoint_id), "p2")
    assert second.patch_id != first.patch_id and second.spec_hash == first.spec_hash


# -- applies and the candidate-commit limit ----------------------------------------------


def test_apply_creates_an_immutable_candidate(engine, base, experiment):
    preview = engine.preview_patch(experiment.experiment_id, _spec(base.checkpoint_id), "p1")
    candidate = engine.apply_patch(experiment.experiment_id, preview.patch_id, "a1")
    assert candidate.checkpoint_id != base.checkpoint_id
    assert candidate.weights_hash == preview.result_weights_hash
    assert candidate.provenance.kind == "patch"
    assert candidate.provenance.parent_checkpoint_id == base.checkpoint_id
    assert candidate.config_hash == base.config_hash and candidate.adapter == base.adapter
    assert engine.store.checkpoint_state(candidate.checkpoint_id) == "READY"
    roles = [(c.checkpoint_id, c.role, c.patch_id) for c in engine.list_experiment_checkpoints(experiment.experiment_id)]
    assert roles == [(base.checkpoint_id, "base", None), (candidate.checkpoint_id, "candidate", preview.patch_id)]
    after = engine.get_experiment(experiment.experiment_id)
    assert after.selected_checkpoint_id == base.checkpoint_id  # applying does not select
    assert after.budget["candidate_commits"].consumed == 1


def test_apply_is_idempotent_and_never_double_charges(engine, base, experiment):
    preview = engine.preview_patch(experiment.experiment_id, _spec(base.checkpoint_id), "p1")
    first = engine.apply_patch(experiment.experiment_id, preview.patch_id, "a1")
    assert engine.apply_patch(experiment.experiment_id, preview.patch_id, "a1") == first
    # A new key for an already applied patch returns its candidate without a new version or charge.
    assert engine.apply_patch(experiment.experiment_id, preview.patch_id, "a2").checkpoint_id == first.checkpoint_id
    assert engine.get_experiment(experiment.experiment_id).budget["candidate_commits"].consumed == 1
    assert len(engine.store.list_ready()) == 2
    other = engine.preview_patch(experiment.experiment_id, _spec(base.checkpoint_id, _bias_op(3.0)), "p2")
    _expect(ErrorCode.IDEMPOTENCY_CONFLICT, engine.apply_patch, experiment.experiment_id, other.patch_id, "a1")
    _expect(ErrorCode.IDEMPOTENCY_CONFLICT, engine.apply_patch, experiment.experiment_id, other.patch_id, "p1")


def test_weights_only_policy_permits_exactly_one_candidate_commit(engine, base):
    grant = engine.grant_run(TASK, WEIGHTS_ONLY_POLICY_ID, base.checkpoint_id)
    experiment = engine.create_experiment(grant.run_id, TASK, WEIGHTS_ONLY_POLICY_ID, "c1")
    assert experiment.mode == "weights_only" and experiment.budget["candidate_commits"].limit == 1
    first = engine.preview_patch(experiment.experiment_id, _spec(base.checkpoint_id), "p1")
    second = engine.preview_patch(experiment.experiment_id, _spec(base.checkpoint_id, _bias_op(5.0)), "p2")
    candidate = engine.apply_patch(experiment.experiment_id, first.patch_id, "a1")
    error = _expect(ErrorCode.BUDGET_EXCEEDED, engine.apply_patch, experiment.experiment_id, second.patch_id, "a2")
    assert error.details == {"counter": "candidate_commits", "limit": 1, "consumed": 1, "requested": 1}
    assert engine.store.list_ready() == [base.checkpoint_id, candidate.checkpoint_id]
    after = engine.get_experiment(experiment.experiment_id)
    assert after.budget["candidate_commits"].consumed == 1 and after.budget["candidate_commits"].remaining == 0
    assert after.selected_checkpoint_id == base.checkpoint_id
    events = [event["event_type"] for event in engine.audit_log(experiment.experiment_id)]
    assert events[-1] == "patch.apply.failed"
    # The committed candidate is still retrievable by its original key.
    assert engine.apply_patch(experiment.experiment_id, first.patch_id, "a1") == candidate


def test_the_commit_transaction_enforces_the_limit_without_the_precheck(engine, base, monkeypatch):
    grant = engine.grant_run(TASK, WEIGHTS_ONLY_POLICY_ID, base.checkpoint_id)
    experiment = engine.create_experiment(grant.run_id, TASK, WEIGHTS_ONLY_POLICY_ID, "c1")
    first = engine.preview_patch(experiment.experiment_id, _spec(base.checkpoint_id), "p1")
    second = engine.preview_patch(experiment.experiment_id, _spec(base.checkpoint_id, _bias_op(5.0)), "p2")
    engine.apply_patch(experiment.experiment_id, first.patch_id, "a1")
    # Simulate a racing request that passed the early check before the first commit.
    monkeypatch.setattr(type(engine.experiments), "_remaining", lambda *args: 1)
    _expect(ErrorCode.BUDGET_EXCEEDED, engine.apply_patch, experiment.experiment_id, second.patch_id, "a2")
    assert len(engine.store.list_ready()) == 2
    assert engine.get_experiment(experiment.experiment_id).budget["candidate_commits"].consumed == 1
    assert engine.store.db.conn.execute(
        "SELECT applied_checkpoint_id FROM patches WHERE patch_id = ?", (second.patch_id,)).fetchone()[0] is None
    # The rolled-back candidate's files are invisible orphans that recovery removes.
    assert engine.store.recover(grace_seconds=0).removed_orphans


def test_bounded_eval_policy_enforces_its_candidate_limit(engine, base):
    engine.register_policy(ExperimentPolicy(policy_id="two_candidates_v1", mode="bounded_eval",
                                            budget=BudgetLimits(candidate_commits=2)))
    grant = engine.grant_run(TASK, "two_candidates_v1", base.checkpoint_id)
    experiment = engine.create_experiment(grant.run_id, TASK, "two_candidates_v1", "c1")
    for index in range(2):
        preview = engine.preview_patch(experiment.experiment_id, _spec(base.checkpoint_id, _bias_op(index + 1.0)),
                                       f"p{index}")
        engine.apply_patch(experiment.experiment_id, preview.patch_id, f"a{index}")
    preview = engine.preview_patch(experiment.experiment_id, _spec(base.checkpoint_id, _bias_op(9.0)), "p9")
    _expect(ErrorCode.BUDGET_EXCEEDED, engine.apply_patch, experiment.experiment_id, preview.patch_id, "a9")
    assert len(engine.list_experiment_checkpoints(experiment.experiment_id)) == 3


def test_concurrent_applies_cannot_exceed_the_candidate_limit(workspace, engine, base):
    grant = engine.grant_run(TASK, WEIGHTS_ONLY_POLICY_ID, base.checkpoint_id)
    experiment = engine.create_experiment(grant.run_id, TASK, WEIGHTS_ONLY_POLICY_ID, "c1")
    patches = [engine.preview_patch(experiment.experiment_id, _spec(base.checkpoint_id, _bias_op(v)), f"p{v}")
               for v in (1.0, 2.0, 3.0, 4.0)]
    outcomes = _in_threads(workspace, len(patches), lambda own, i: own.apply_patch(
        experiment.experiment_id, patches[i].patch_id, f"a{i}"))
    assert sum(1 for outcome in outcomes if outcome == ErrorCode.BUDGET_EXCEEDED) == len(patches) - 1
    assert len(engine.list_experiment_checkpoints(experiment.experiment_id)) == 2
    assert engine.get_experiment(experiment.experiment_id).budget["candidate_commits"].consumed == 1


def test_concurrent_identical_applies_create_one_candidate(workspace, engine, base, experiment):
    preview = engine.preview_patch(experiment.experiment_id, _spec(base.checkpoint_id), "p1")
    results = _in_threads(workspace, 3, lambda own, i: own.apply_patch(
        experiment.experiment_id, preview.patch_id, "same-key"))
    assert len({result.checkpoint_id for result in results}) == 1
    assert engine.get_experiment(experiment.experiment_id).budget["candidate_commits"].consumed == 1
    assert len(engine.list_experiment_checkpoints(experiment.experiment_id)) == 2


# -- ownership ----------------------------------------------------------------------------


def test_experiments_cannot_use_each_others_objects(engine, bundles, base, experiment):
    other_grant = engine.grant_run(TASK, SMALL_MODEL_POLICY_ID, base.checkpoint_id)
    other = engine.create_experiment(other_grant.run_id, TASK, SMALL_MODEL_POLICY_ID, "c1")
    foreign_preview = engine.preview_patch(other.experiment_id, _spec(base.checkpoint_id), "p1")
    foreign = engine.apply_patch(other.experiment_id, foreign_preview.patch_id, "a1")
    mine = experiment.experiment_id
    _expect(ErrorCode.NOT_FOUND, engine.apply_patch, mine, foreign_preview.patch_id, "a1")
    _expect(ErrorCode.NOT_FOUND, engine.get_patch, mine, foreign_preview.patch_id)
    _expect(ErrorCode.NOT_FOUND, engine.read_patch, mine, foreign_preview.patch_id)
    _expect(ErrorCode.NOT_FOUND, engine.preview_patch, mine, _spec(foreign.checkpoint_id), "p1")
    _expect(ErrorCode.NOT_FOUND, engine.select_checkpoint, mine, foreign.checkpoint_id, base.checkpoint_id, "s1")
    _expect(ErrorCode.NOT_FOUND, engine.experiments.load_manifest, mine, foreign.checkpoint_id)
    # A READY checkpoint outside the run grant is not visible either.
    reference = engine.import_bundle(bundles["reference"])
    _expect(ErrorCode.NOT_FOUND, engine.select_checkpoint, mine, reference.checkpoint_id, base.checkpoint_id, "s2")
    _expect(ErrorCode.NOT_FOUND, engine.preview_patch, mine, _spec(reference.checkpoint_id), "p2")
    assert [c.checkpoint_id for c in engine.list_experiment_checkpoints(mine)] == [base.checkpoint_id]
    _expect(ErrorCode.NOT_FOUND, engine.get_experiment, "exp_" + "0" * 32)
    _expect(ErrorCode.NOT_FOUND, engine.get_experiment, "../exp")


def test_a_run_bound_engine_sees_only_its_own_run(workspace, engine, base, grant, experiment):
    other_grant = engine.grant_run(TASK, SMALL_MODEL_POLICY_ID, base.checkpoint_id)
    other = engine.create_experiment(other_grant.run_id, TASK, SMALL_MODEL_POLICY_ID, "c1")
    with Engine(workspace, run_id=grant.run_id) as bound:
        assert bound.get_experiment(experiment.experiment_id).experiment_id == experiment.experiment_id
        assert bound.create_experiment(grant.run_id, TASK, SMALL_MODEL_POLICY_ID, "c9").experiment_id == \
            experiment.experiment_id
        _expect(ErrorCode.NOT_FOUND, bound.get_experiment, other.experiment_id)
        _expect(ErrorCode.NOT_FOUND, bound.preview_patch, other.experiment_id, _spec(base.checkpoint_id), "p1")
        _expect(ErrorCode.POLICY_DENIED, bound.create_experiment, other_grant.run_id, TASK, SMALL_MODEL_POLICY_ID, "c1")
        _expect(ErrorCode.POLICY_DENIED, bound.grant_run, TASK, SMALL_MODEL_POLICY_ID, base.checkpoint_id)
        _expect(ErrorCode.POLICY_DENIED, bound.register_policy,
                ExperimentPolicy(policy_id="bigger_v1", mode="bounded_eval", budget=BudgetLimits(candidate_commits=99)))
    with pytest.raises(WSEError) as excinfo:
        Engine(workspace, run_id="run_" + "1" * 32)
    assert excinfo.value.code == ErrorCode.NOT_FOUND


# -- selection and rollback -----------------------------------------------------------------


def test_select_uses_compare_and_swap_and_rollback_keeps_history(engine, base, experiment, signed_sum):
    eid = experiment.experiment_id
    preview = engine.preview_patch(eid, _spec(base.checkpoint_id), "p1")
    candidate = engine.apply_patch(eid, preview.patch_id, "a1")
    assert _accuracy(engine, base.checkpoint_id, signed_sum) == 0.5
    assert _accuracy(engine, candidate.checkpoint_id, signed_sum) == 1.0

    selected = engine.select_checkpoint(eid, candidate.checkpoint_id, base.checkpoint_id, "s1")
    assert selected.selected_checkpoint_id == candidate.checkpoint_id
    # A stale expected ID is a conflict and changes nothing.
    error = _expect(ErrorCode.VERSION_CONFLICT, engine.select_checkpoint, eid, base.checkpoint_id,
                    base.checkpoint_id, "s2")
    assert error.details["selected_checkpoint_id"] == candidate.checkpoint_id
    assert engine.get_experiment(eid).selected_checkpoint_id == candidate.checkpoint_id
    # Rolling back to the ancestor changes only the selected ID.
    rolled = engine.select_checkpoint(eid, base.checkpoint_id, candidate.checkpoint_id, "s3")
    assert rolled.selected_checkpoint_id == base.checkpoint_id
    assert _accuracy(engine, rolled.selected_checkpoint_id, signed_sum) == 0.5
    assert rolled.budget["candidate_commits"].consumed == 1  # no refund
    assert engine.store.checkpoint_state(candidate.checkpoint_id) == "READY"
    assert [c.checkpoint_id for c in engine.list_experiment_checkpoints(eid)] == [base.checkpoint_id,
                                                                                    candidate.checkpoint_id]
    assert weights_unchanged(engine, base)
    # Idempotent: the same key returns the recorded result.
    assert engine.select_checkpoint(eid, candidate.checkpoint_id, base.checkpoint_id, "s1") == selected
    _expect(ErrorCode.IDEMPOTENCY_CONFLICT, engine.select_checkpoint, eid, base.checkpoint_id,
            candidate.checkpoint_id, "s1")
    events = [(e["event_type"], e["payload"].get("rollback")) for e in engine.audit_log(eid)
              if e["event_type"].startswith("experiment.")]
    assert events == [("experiment.created", None), ("experiment.selected", False),
                      ("experiment.select.failed", None), ("experiment.rollback", True)]


def weights_unchanged(engine, base):
    from wse.tensors import weights_hash

    return weights_hash(engine.store.load_tensors(base.checkpoint_id)) == base.weights_hash


def test_concurrent_selections_conflict(workspace, engine, base, experiment):
    eid = experiment.experiment_id
    candidates = []
    for value in (1.0, 2.0):
        preview = engine.preview_patch(eid, _spec(base.checkpoint_id, _bias_op(value)), f"p{value}")
        candidates.append(engine.apply_patch(eid, preview.patch_id, f"a{value}").checkpoint_id)
    outcomes = _in_threads(workspace, 2, lambda own, i: own.select_checkpoint(
        eid, candidates[i], base.checkpoint_id, f"s{i}"))
    assert sorted(str(o) for o in outcomes if isinstance(o, ErrorCode)) == ["VERSION_CONFLICT"]
    winner = next(o for o in outcomes if not isinstance(o, ErrorCode))
    assert engine.get_experiment(eid).selected_checkpoint_id == winner.selected_checkpoint_id


def test_patches_may_build_on_candidates(engine, base, experiment):
    eid = experiment.experiment_id
    first = engine.apply_patch(eid, engine.preview_patch(eid, _spec(base.checkpoint_id), "p1").patch_id, "a1")
    second_preview = engine.preview_patch(eid, _spec(first.checkpoint_id, _bias_op(0.5)), "p2")
    second = engine.apply_patch(eid, second_preview.patch_id, "a2")
    assert second.provenance.parent_checkpoint_id == first.checkpoint_id
    engine.select_checkpoint(eid, second.checkpoint_id, base.checkpoint_id, "s1")
    rolled = engine.select_checkpoint(eid, base.checkpoint_id, second.checkpoint_id, "s2")
    assert engine.audit_log(eid)[-1]["event_type"] == "experiment.rollback"
    assert rolled.selected_checkpoint_id == base.checkpoint_id


def test_closed_states_reject_mutations_but_keep_recorded_results(engine, base, experiment):
    eid = experiment.experiment_id
    preview = engine.preview_patch(eid, _spec(base.checkpoint_id), "p1")
    with engine.store.db.transaction() as conn:
        conn.execute("UPDATE experiments SET state = 'FROZEN' WHERE experiment_id = ?", (eid,))
    _expect(ErrorCode.VERSION_CONFLICT, engine.apply_patch, eid, preview.patch_id, "a1")
    _expect(ErrorCode.VERSION_CONFLICT, engine.preview_patch, eid, _spec(base.checkpoint_id, _bias_op(1.0)), "p2")
    _expect(ErrorCode.VERSION_CONFLICT, engine.select_checkpoint, eid, base.checkpoint_id, base.checkpoint_id, "s1")
    assert engine.preview_patch(eid, _spec(base.checkpoint_id), "p1") == preview
    assert engine.get_experiment(eid).budget["candidate_commits"].consumed == 0


# -- run-bound engines expose only the grant's objects ------------------------------------


def _not_found_message(func, *args):
    """The NOT_FOUND error of ``func(*args)``, with the checkpoint ID masked for comparison."""
    error = _expect(ErrorCode.NOT_FOUND, func, *args)
    return error.message.replace(str(args[0]), "<id>"), {**error.details, "checkpoint_id": "<id>"}


def test_a_run_bound_engine_exposes_only_its_grant_base_and_own_candidates(workspace, engine, bundles, base, grant,
                                                                           experiment):
    reference = engine.import_bundle(bundles["reference"])
    other_grant = engine.grant_run(TASK, SMALL_MODEL_POLICY_ID, base.checkpoint_id)
    other = engine.create_experiment(other_grant.run_id, TASK, SMALL_MODEL_POLICY_ID, "c1")
    foreign = engine.apply_patch(other.experiment_id,
                                 engine.preview_patch(other.experiment_id, _spec(base.checkpoint_id), "p1").patch_id,
                                 "a1")
    eid = experiment.experiment_id
    missing = "ckpt_" + "0" * 32
    weight = "layers.1.weight"
    with Engine(workspace, run_id=grant.run_id) as bound:
        assert [m.checkpoint_id for m in bound.list_models()] == [base.checkpoint_id]
        own = bound.apply_patch(eid, bound.preview_patch(eid, _spec(base.checkpoint_id), "p1").patch_id, "a1")
        assert [m.checkpoint_id for m in bound.list_models()] == [base.checkpoint_id, own.checkpoint_id]
        for visible in (base.checkpoint_id, own.checkpoint_id):
            assert bound.inspect(visible).checkpoint_id == visible
            assert bound.load_manifest(visible).checkpoint_id == visible
            assert bound.read_tensor_slice(visible, weight, ROW1).element_count == 4
            assert bound.analyze(visible, "stats", weight, ROW1).result["count"] == 4
            assert bound.open_checkpoint(visible).weights_hash == bound.load_manifest(visible).weights_hash
        # Other checkpoints fail exactly like an ID that does not exist, so existence is not confirmed.
        calls = [
            lambda c: bound.inspect(c),
            lambda c: bound.load_manifest(c),
            lambda c: bound.read_tensor_slice(c, weight, ROW1),
            lambda c: bound.read_tensor(c, weight, ROW1),
            lambda c: bound.analyze(c, "stats", weight, ROW1),
            lambda c: bound.open_checkpoint(c),
        ]
        for call in calls:
            expected = _not_found_message(call, missing)
            for hidden in (reference.checkpoint_id, foreign.checkpoint_id):
                assert _not_found_message(call, hidden) == expected
        for hidden in (reference.checkpoint_id, foreign.checkpoint_id):
            assert _not_found_message(lambda c: bound.preview_patch(eid, _spec(c), f"p-{c}"), hidden) == \
                _not_found_message(lambda c: bound.preview_patch(eid, _spec(c), f"p-{c}"), missing)
            assert _not_found_message(lambda c: bound.select_checkpoint(eid, c, base.checkpoint_id, f"s-{c}"),
                                      hidden) == \
                _not_found_message(lambda c: bound.select_checkpoint(eid, c, base.checkpoint_id, f"s-{c}"), missing)
        # Operator actions are denied to a run.
        _expect(ErrorCode.POLICY_DENIED, bound.import_bundle, bundles["reference"])
        _expect(ErrorCode.POLICY_DENIED, bound.import_model, {"adapter": "mlp@1", "config": {"layer_sizes": [2, 2]}},
                {"layers.0.weight": torch.eye(2), "layers.0.bias": torch.zeros(2)})
        _expect(ErrorCode.POLICY_DENIED, bound.grant_run, TASK, SMALL_MODEL_POLICY_ID, base.checkpoint_id)
        _expect(ErrorCode.POLICY_DENIED, bound.register_policy,
                ExperimentPolicy(policy_id="bigger_v2", mode="bounded_eval"))
    # The unbound operator engine still sees everything.
    assert {reference.checkpoint_id, foreign.checkpoint_id} <= {m.checkpoint_id for m in engine.list_models()}


def test_a_bound_engine_before_creation_sees_only_the_grant_base(workspace, engine, bundles, base):
    reference = engine.import_bundle(bundles["reference"])
    grant = engine.grant_run(TASK, SMALL_MODEL_POLICY_ID, base.checkpoint_id)
    with Engine(workspace, run_id=grant.run_id) as bound:
        assert [m.checkpoint_id for m in bound.list_models()] == [base.checkpoint_id]
        _expect(ErrorCode.NOT_FOUND, bound.inspect, reference.checkpoint_id)


# -- review fixes: argument types and traceable creation failures ---------------------------


@pytest.mark.parametrize("expected", [{"x": 1}, ["a"], 3, None])
def test_select_rejects_a_non_string_expected_id_as_a_domain_error(engine, base, experiment, expected):
    eid = experiment.experiment_id
    error = _expect(ErrorCode.INVALID_ARGUMENT, engine.select_checkpoint, eid, base.checkpoint_id, expected, "s1")
    assert "expected_selected_id" in error.message
    assert engine.audit_log(eid)[-1]["event_type"] == "experiment.select.failed"
    assert engine.get_experiment(eid).selected_checkpoint_id == base.checkpoint_id


def test_rejected_experiment_creation_is_recorded(engine, base, grant, experiment):
    _expect(ErrorCode.POLICY_DENIED, engine.create_experiment, grant.run_id, TASK, WEIGHTS_ONLY_POLICY_ID, "c3")
    _expect(ErrorCode.IDEMPOTENCY_CONFLICT, engine.create_experiment, grant.run_id, "other_task",
            SMALL_MODEL_POLICY_ID, "create-1")
    failures = [e for e in engine.audit_log(experiment.experiment_id) if e["event_type"] == "experiment.create.failed"]
    assert [f["payload"]["code"] for f in failures] == ["POLICY_DENIED", "IDEMPOTENCY_CONFLICT"]
    assert all(f["subject_id"] == grant.run_id for f in failures)
    # Before any experiment exists, the failure is recorded against the run grant.
    fresh = engine.grant_run(TASK, SMALL_MODEL_POLICY_ID, base.checkpoint_id)
    _expect(ErrorCode.POLICY_DENIED, engine.create_experiment, fresh.run_id, "other_task", SMALL_MODEL_POLICY_ID, "c1")
    from wse.database import audit_events

    events = audit_events(engine.store.db.conn, fresh.run_id)
    assert [e["event_type"] for e in events][-1] == "experiment.create.failed"
    assert events[-1]["experiment_id"] is None
