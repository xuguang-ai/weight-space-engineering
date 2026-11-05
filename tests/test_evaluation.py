"""Development evaluation, reports, comparisons, freezing, and final evaluation (Task 5)."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch

from wse.api import Engine, evaluate_candidate, load_model
from wse.evaluation import render_markdown
from wse.evaluators.classification import (
    ClassificationEvaluator,
    accuracy_interval,
    clopper_pearson,
    fully_correct_interval,
    paired_comparison,
)
from wse.policy import SMALL_MODEL_POLICY_ID, ExperimentPolicy, builtin_experiment_policies
from wse.schemas import ErrorCode, EvalConfig, EvaluationSuite, TaskDefinition, WSEError
from wse.worker import Worker

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "signed_sum"
ROW1 = [{"start": 1, "stop": 2, "step": 1}, {"start": 0, "stop": 4, "step": 1}]
ORACLE = {"op": "scale_slice", "tensor": "layers.1.weight", "selection": ROW1, "factor": -1.0}
FINAL_TEST_SEED = 7_390_519_204_117
DEV_CONFIG = {"sample_count": 64, "seed": 0}


def _expect(code, func, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        func(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


@pytest.fixture
def ids(engine, signed_sum, tmp_path):
    """An operator-prepared workspace: task model, task.json, policy, development and final sets."""
    return signed_sum.setup_workspace(engine, tmp_path / "bundles", final_seed=FINAL_TEST_SEED, with_reference=True)


def _experiment(engine, ids, policy_id=SMALL_MODEL_POLICY_ID):
    grant = engine.grant_run(ids["task_id"], policy_id, ids["task_checkpoint_id"])
    return engine.create_experiment(grant.run_id, ids["task_id"], policy_id, "create")


def _repair(engine, ids, experiment, key="1"):
    eid = experiment.experiment_id
    spec = {"base_checkpoint_id": ids["task_checkpoint_id"], "hypothesis": "One output row has the wrong sign.",
            "operations": [ORACLE]}
    return engine.apply_patch(eid, engine.preview_patch(eid, spec, f"p{key}").patch_id, f"a{key}")


def _evaluate(engine, workspace, experiment_id, checkpoint_id, key, suite_id="signed_sum_dev_v1", config=None):
    job = engine.submit_evaluation(experiment_id, checkpoint_id, suite_id, config or DEV_CONFIG, key)
    with Worker(workspace) as worker:
        worker.run_once()
    done = engine.get_job(experiment_id, job.job_id)
    assert done.state == "succeeded", done.error
    return done


def _repaired_and_frozen(engine, workspace, ids):
    experiment = _experiment(engine, ids)
    eid = experiment.experiment_id
    candidate = _repair(engine, ids, experiment)
    engine.select_checkpoint(eid, candidate.checkpoint_id, ids["task_checkpoint_id"], "s1")
    return eid, candidate, engine.freeze(eid, candidate.checkpoint_id)


# -- exact accuracy on the balanced fixture ------------------------------------------------------------


@pytest.mark.parametrize("pairs, seed", [(32, 20261104), (512, 7)])
def test_clean_broken_and_repaired_models_score_100_50_100(signed_sum, pairs, seed):
    suite = EvaluationSuite.model_validate(signed_sum.development_suite(data=signed_sum.suite_data(pairs, seed))
                                           | {"sample_counts": [2 * pairs]})
    config = {"sample_count": 2 * pairs, "seed": 0}
    clean = load_model(signed_sum.reference_tensors(), "mlp@1", signed_sum.CONFIG)
    broken = load_model(signed_sum.task_tensors(), "mlp@1", signed_sum.CONFIG)
    from wse.api import apply_operator

    repaired = apply_operator(broken, "scale_slice@1", {k: v for k, v in ORACLE.items() if k != "op"}).candidate
    results = [evaluate_candidate(model, suite, config) for model in (clean, broken, repaired)]
    assert [r.metrics["accuracy"] for r in results] == [1.0, 0.5, 1.0]
    assert [r.clusters.fully_correct for r in results] == [pairs, 0, pairs]
    assert all(r.clusters.count == pairs and r.clusters.unit == "pair" for r in results)
    # The faulty model gets exactly one example of every pair right.
    assert set(zip(results[1].scores[0::2], results[1].scores[1::2], strict=True)) <= {(0, 1), (1, 0)}
    assert repaired.weights_hash == clean.weights_hash


def test_ties_select_class_zero_and_non_finite_logits_score_incorrect(signed_sum):
    data = signed_sum.suite_data(4, 1)
    suite = EvaluationSuite.model_validate(signed_sum.development_suite(data=data) | {"sample_counts": [8]})

    class TiedForward:  # equal logits everywhere, and NaN in the first row
        def check(self):
            return None

        def __call__(self, inputs):
            logits = torch.zeros(inputs.shape[0], 2)
            logits[0, 1] = float("nan")
            return logits

    summary = ClassificationEvaluator().evaluate(TiedForward(), suite, EvalConfig(sample_count=8, seed=0),
                                                 batch_size=8)
    expected = [0] + [int(label == 0) for label in data["labels"][1:]]
    assert summary["scores"] == expected and summary["errors"] == [
        "1 examples produced non-finite logits and were scored incorrect"]


# -- intervals -------------------------------------------------------------------------------------------


def test_clopper_pearson_matches_the_plan():
    lower, upper = clopper_pearson(512, 512)
    assert round(100 * lower, 2) == 99.28 and upper == 1.0
    lower, upper = clopper_pearson(511, 512)
    assert round(100 * lower, 2) == 98.92 and upper < 1.0
    assert clopper_pearson(0, 512)[0] == 0.0
    lower, upper = clopper_pearson(16, 32)
    assert (round(lower, 4), round(upper, 4)) == (0.3189, 0.6811)
    for successes in (1, 100, 300):  # symmetry of the exact interval
        a, b = clopper_pearson(successes, 512)
        c, d = clopper_pearson(512 - successes, 512)
        assert a == pytest.approx(1 - d, abs=1e-9) and b == pytest.approx(1 - c, abs=1e-9)
    _expect(ErrorCode.INVALID_ARGUMENT, clopper_pearson, 3, 2)


def test_bootstrap_intervals_collapse_at_the_fixture_boundaries_and_are_reproducible():
    clusters = [i // 2 for i in range(64)]
    broken = [1, 0] * 32
    repaired = [1] * 64
    paired = paired_comparison(broken, repaired, clusters, "pair")
    interval = paired["difference_interval"]
    assert paired["difference"] == 0.5 and (interval.lower, interval.upper) == (0.5, 0.5)
    assert (paired["clusters_improved"], paired["clusters_worsened"]) == (32, 0)
    assert accuracy_interval(broken, clusters, "pair").lower == 0.5
    assert fully_correct_interval(repaired, clusters, "pair").successes == 32
    mixed = [1, 1, 0, 0, 1, 0] * 10 + [1, 1, 1, 1]
    first = accuracy_interval(mixed, clusters, "pair")
    assert first == accuracy_interval(mixed, clusters, "pair")  # fixed seed
    assert first.lower < first.estimate < first.upper
    # Treating correlated pair members as independent narrows the interval.
    independent = accuracy_interval(mixed, list(range(64)), "example")
    assert independent.upper - independent.lower < first.upper - first.lower


# -- task.json, policy.json, and the generated datasets -------------------------------------------------


FAULT_TERMS = ("fault", "inject", "flip", "invert", "negat", "corrupt", "reference", "clean", "oracle", "broken",
               "factor", "-1", "row 1", "layers.", "w1", "w2", "50%", "ckpt_", "build_fixture")


def test_task_json_and_policy_json_reveal_no_answer(signed_sum, ids):
    task = TaskDefinition.model_validate(signed_sum.load_task())
    assert task.target_accuracy == 1.0 and task.development_examples == 64
    assert task.development_suite_id == signed_sum.DEV_SUITE_ID
    policy = ExperimentPolicy.model_validate(signed_sum.load_policy())
    assert policy.policy_hash == builtin_experiment_policies()[SMALL_MODEL_POLICY_ID].policy_hash
    task_text = (EXAMPLE_DIR / "task.json").read_text().lower()
    policy_text = (EXAMPLE_DIR / "policy.json").read_text().lower()
    for term in FAULT_TERMS:
        assert term not in task_text, term
    for term in ("fault", "inject", "flip", "oracle", "reference", "clean", "factor", "-1"):
        assert term not in policy_text, term
    assert "scale_slice" not in task_text and "set_slice" not in task_text
    assert ids["reference_checkpoint_id"] not in task_text


def test_setup_generates_both_sets_into_evaluator_storage(engine, workspace, ids, signed_sum):
    development = engine.jobs.load_suite(signed_sum.DEV_SUITE_ID)
    final = engine.jobs.load_suite(signed_sum.FINAL_SUITE_ID)
    assert development.data is None and final.data is None  # registered suites carry no examples
    assert (development.dataset.example_count, final.dataset.example_count) == (64, 1024)
    assert (development.dataset.cluster_count, final.dataset.cluster_count) == (32, 512)
    for suite in (development, final):
        data = engine.datasets.load(suite.dataset)
        pairs = list(zip(data.inputs[0::2], data.inputs[1::2], strict=True))
        assert all(b == [-a[0], -a[1]] for a, b in pairs)
        assert all(abs(a[0] + a[1]) >= 0.1 for a, _ in pairs)
        assert sum(data.labels) == len(data.labels) // 2
    generation = engine.datasets.generation(final.dataset.data_hash)
    assert generation["seed"] == FINAL_TEST_SEED and generation["pairs"] == 512
    files = sorted(p.relative_to(workspace).as_posix() for p in (workspace / "evaluator").rglob("*") if p.is_file())
    assert len(files) == 4 and all(name.startswith("evaluator/datasets/") for name in files)
    assert not any("evaluator" in str(p) for p in (workspace / "artifacts").rglob("*"))
    assert not [p for d in ("src", "tests", "examples") for p in (REPO_ROOT / d).glob("**/evaluator")]
    # A fresh setup draws a fresh final seed: the final examples differ, the development set does not.
    with Engine(workspace.parent / "other") as other:
        again = signed_sum.setup_workspace(other, workspace.parent / "other_bundles")
        assert other.jobs.load_suite(again["final_suite_id"]).data_hash != final.data_hash
        assert other.jobs.load_suite(again["development_suite_id"]).data_hash == development.data_hash


def test_final_suites_are_fixed_per_task_and_kept_in_evaluator_storage(engine, ids, signed_sum):
    inline = {"suite_id": "signed_sum_final_v2", "task_id": "other_task", "split": "final",
              "evaluator": "classification@1", "sample_counts": [16], "seeds": [0], "data": signed_sum.suite_data(8, 3)}
    registered = engine.register_suite(inline)
    assert registered.data is None and engine.datasets.exists(registered.dataset.data_hash)
    _expect(ErrorCode.VERSION_CONFLICT, engine.register_suite, {**inline, "suite_id": "signed_sum_final_v3"})
    _expect(ErrorCode.INVALID_ARGUMENT, engine.register_suite,
            {**inline, "suite_id": "third", "task_id": "third_task", "sample_counts": [8]})
    _expect(ErrorCode.INVALID_ARGUMENT, engine.register_suite,  # 7 examples would split a pair
            {**inline, "suite_id": "odd", "task_id": "odd_task", "split": "development", "sample_counts": [7]})
    assert engine.register_task(signed_sum.load_task()).task_id == signed_sum.TASK_ID  # idempotent
    _expect(ErrorCode.VERSION_CONFLICT, engine.register_task, {**signed_sum.load_task(), "title": "changed"})


# -- managed development evaluation ------------------------------------------------------------------------


def test_development_reports_bind_identities_and_meet_the_target_only_with_full_coverage(engine, workspace, ids,
                                                                                         signed_sum):
    experiment = _experiment(engine, ids)
    eid, base = experiment.experiment_id, ids["task_checkpoint_id"]
    candidate = _repair(engine, ids, experiment)
    before, after = (_evaluate(engine, workspace, eid, checkpoint, key)
                     for checkpoint, key in ((base, "e1"), (candidate.checkpoint_id, "e2")))
    assert before.result["metrics"] == {"accuracy": 0.5} and after.result["metrics"] == {"accuracy": 1.0}
    assert (before.result["target_met"], after.result["target_met"]) == (False, True)
    report = engine.get_report(eid, after.report_id)
    assert after.result["report_id"] == report.report_id == after.report_id
    suite = engine.jobs.load_suite(signed_sum.DEV_SUITE_ID)
    assert report.model == candidate.identity and report.checkpoint_id == candidate.checkpoint_id
    assert (report.suite_hash, report.data_hash) == (suite.suite_hash, suite.dataset.data_hash)
    assert report.task_hash == engine.get_task(signed_sum.TASK_ID).task_hash
    assert report.evaluator == suite.evaluator and report.execution.batch_size == 64
    assert report.seed == 0 and report.sample_count == 64 and report.split == "development"
    assert report.usage == after.usage and report.usage.model_forward_calls == 1
    assert report.usage.peak_memory_bytes > 0 and report.elapsed_seconds >= 0
    assert report.intervals["fully_correct_clusters"].successes == 32 and report.management == "managed"
    assert [r.report_id for r in engine.list_reports(eid)] == [before.report_id, after.report_id]
    # A partial evaluation can never meet the target.
    engine.register_suite({**signed_sum.development_suite(suite.dataset), "suite_id": "signed_sum_dev_partial",
                           "sample_counts": [32, 64]})
    partial = _evaluate(engine, workspace, eid, candidate.checkpoint_id, "e3", "signed_sum_dev_partial",
                        {"sample_count": 32, "seed": 0})
    assert partial.result["metrics"] == {"accuracy": 1.0} and partial.result["target_met"] is False


def test_only_the_tasks_own_full_development_suite_can_meet_the_target(engine, workspace, ids, signed_sum):
    experiment = _experiment(engine, ids)
    eid = experiment.experiment_id
    candidate = _repair(engine, ids, experiment)
    # A smaller development suite of the same task, evaluated in full, still cannot meet the target.
    engine.register_suite({**signed_sum.development_suite(data=signed_sum.suite_data(2, 5)),
                           "suite_id": "signed_sum_dev_small", "sample_counts": [4]})
    small = _evaluate(engine, workspace, eid, candidate.checkpoint_id, "e1", "signed_sum_dev_small",
                      {"sample_count": 4, "seed": 0})
    assert small.result["metrics"] == {"accuracy": 1.0} and small.result["sample_count"] == 4
    assert small.result["target_met"] is False
    full = _evaluate(engine, workspace, eid, candidate.checkpoint_id, "e2")
    assert full.result["target_met"] is True
    # Final reports carry no development-target verdict, so their Markdown never claims one.
    engine.select_checkpoint(eid, candidate.checkpoint_id, ids["task_checkpoint_id"], "s1")
    engine.freeze(eid, candidate.checkpoint_id)
    final = engine.final_evaluate(eid)
    assert final.candidate.target_met is None and final.control.target_met is None
    assert "target met" not in render_markdown(final.candidate)


def test_compare_reports_paired_differences_and_rejects_mismatched_reports(engine, workspace, ids, signed_sum):
    experiment = _experiment(engine, ids)
    eid, base = experiment.experiment_id, ids["task_checkpoint_id"]
    candidate = _repair(engine, ids, experiment)
    dev = engine.jobs.load_suite(signed_sum.DEV_SUITE_ID)
    engine.register_suite({**signed_sum.development_suite(dev.dataset), "suite_id": "signed_sum_dev_multi",
                           "sample_counts": [32, 64], "seeds": [0, 1]})
    a = _evaluate(engine, workspace, eid, base, "e1").report_id
    b = _evaluate(engine, workspace, eid, candidate.checkpoint_id, "e2").report_id
    comparison = engine.compare(eid, a, b)
    assert comparison.difference == 0.5 and comparison.clusters_improved == 32
    assert (comparison.difference_interval.lower, comparison.difference_interval.upper) == (0.5, 0.5)
    assert comparison.base_fully_correct.successes == 0 and comparison.candidate_fully_correct.successes == 32
    assert engine.compare(eid, b, b).difference == 0.0
    other_suite = _evaluate(engine, workspace, eid, base, "e3", "signed_sum_dev_multi").report_id
    other_seed = _evaluate(engine, workspace, eid, base, "e4", "signed_sum_dev_multi",
                           {"sample_count": 64, "seed": 1}).report_id
    other_count = _evaluate(engine, workspace, eid, base, "e5", "signed_sum_dev_multi",
                            {"sample_count": 32, "seed": 0}).report_id
    assert "suite_id" in _expect(ErrorCode.INCOMPATIBLE_REPORTS, engine.compare, eid, a, other_suite).details[
        "mismatched"]
    mismatch = _expect(ErrorCode.INCOMPATIBLE_REPORTS, engine.compare, eid, other_suite, other_seed).details
    assert {"seed", "evaluator_config_hash"} <= set(mismatch["mismatched"])
    assert "sample_count" in _expect(ErrorCode.INCOMPATIBLE_REPORTS, engine.compare, eid, other_suite,
                                     other_count).details["mismatched"]
    # Reports of another experiment are reported as missing.
    foreign_experiment = _experiment(engine, ids)
    foreign = _evaluate(engine, workspace, foreign_experiment.experiment_id, base, "e1").report_id
    _expect(ErrorCode.NOT_FOUND, engine.compare, eid, a, foreign)
    _expect(ErrorCode.NOT_FOUND, engine.get_report, eid, "report_" + "0" * 32)
    _expect(ErrorCode.NOT_FOUND, engine.get_report, eid, {"report": 1})


def test_reports_without_per_example_scores_cannot_be_paired(engine, workspace, ids, signed_sum):
    experiment = _experiment(engine, ids)
    eid, base = experiment.experiment_id, ids["task_checkpoint_id"]
    data = signed_sum.suite_data(32, signed_sum.DEV_SEED)
    engine.register_suite({"suite_id": "dev_forward_batch", "task_id": ids["task_id"], "split": "development",
                           "evaluator": "forward_batch@1", "sample_counts": [64], "seeds": [0], "data": data})
    a = _evaluate(engine, workspace, eid, base, "e1", "dev_forward_batch").report_id
    b = _evaluate(engine, workspace, eid, base, "e2", "dev_forward_batch").report_id
    _expect(ErrorCode.INCOMPATIBLE_REPORTS, engine.compare, eid, a, b)


# -- freezing -------------------------------------------------------------------------------------------------


def test_active_jobs_and_held_reservations_prevent_freezing(engine, workspace, ids):
    experiment = _experiment(engine, ids)
    eid, base = experiment.experiment_id, ids["task_checkpoint_id"]
    job = engine.submit_evaluation(eid, base, "signed_sum_dev_v1", DEV_CONFIG, "e1")
    error = _expect(ErrorCode.VERSION_CONFLICT, engine.freeze, eid, base)
    assert error.details["active_jobs"] == [job.job_id] and error.retryable
    engine.cancel_job(eid, job.job_id, "c1")
    held = engine.reserve(eid, "job_" + "f" * 32, {"evaluation_batches": 1})
    assert _expect(ErrorCode.VERSION_CONFLICT, engine.freeze, eid, base).details["held_reservations"] == [
        held.reservation_id]
    engine.settle(held.reservation_id, {}, "cancelled")
    _expect(ErrorCode.VERSION_CONFLICT, engine.freeze, eid, "ckpt_" + "0" * 32)  # stale expected selection
    _expect(ErrorCode.INVALID_ARGUMENT, engine.freeze, eid, None)
    frozen = engine.freeze(eid, base)
    assert frozen.state == "FROZEN" and frozen.frozen_at is not None and frozen.selected_checkpoint_id == base
    assert engine.freeze(eid, base) == frozen  # a retry returns the frozen experiment
    events = [e["event_type"] for e in engine.audit_log(eid)]
    assert events.count("experiment.freeze.failed") == 4 and events.count("experiment.frozen") == 1


def test_a_frozen_experiment_accepts_no_edits_selections_jobs_or_metered_reads(engine, workspace, ids):
    experiment = _experiment(engine, ids)
    eid, base = experiment.experiment_id, ids["task_checkpoint_id"]
    candidate = _repair(engine, ids, experiment)
    a = _evaluate(engine, workspace, eid, base, "e1").report_id
    b = _evaluate(engine, workspace, eid, candidate.checkpoint_id, "e2").report_id
    spec = {"base_checkpoint_id": base, "hypothesis": "h", "operations": [ORACLE]}
    preview = engine.preview_patch(eid, spec | {"hypothesis": "another"}, "p2")
    engine.freeze(eid, base)
    before = engine.get_experiment(eid)
    cases = [
        (engine.preview_patch, eid, spec | {"hypothesis": "third"}, "p3"),
        (engine.apply_patch, eid, preview.patch_id, "a2"),
        (engine.select_checkpoint, eid, candidate.checkpoint_id, base, "s1"),
        (engine.submit_evaluation, eid, candidate.checkpoint_id, "signed_sum_dev_v1", DEV_CONFIG, "e3"),
        (engine.read_model_tensor, eid, base, "layers.1.weight", ROW1),
        (engine.analyze_model, eid, base, "stats", "layers.1.weight", ROW1),
    ]
    for func, *args in cases:
        _expect(ErrorCode.VERSION_CONFLICT, func, *args)
    after = engine.get_experiment(eid)
    assert after == before and after.selected_checkpoint_id == base
    # Development outputs stay readable and unchanged.
    assert engine.compare(eid, a, b).difference == 0.5
    assert engine.inspect_model(eid, candidate.checkpoint_id).checkpoint_id == candidate.checkpoint_id
    assert engine.submit_evaluation(eid, base, "signed_sum_dev_v1", DEV_CONFIG, "e1").job_id  # recorded result


def test_freezing_and_submission_cannot_both_win(workspace, engine, ids):
    for attempt in range(4):
        experiment = _experiment(engine, ids)
        eid, base = experiment.experiment_id, ids["task_checkpoint_id"]
        barrier, outcomes = threading.Barrier(2), {}

        def run(name, call):
            with Engine(workspace, recover=False) as own:
                barrier.wait()
                try:
                    outcomes[name] = call(own)
                except WSEError as exc:
                    outcomes[name] = exc.code

        threads = [threading.Thread(target=run, args=("freeze", lambda own: own.freeze(eid, base))),
                   threading.Thread(target=run, args=("submit", lambda own: own.submit_evaluation(
                       eid, base, "signed_sum_dev_v1", DEV_CONFIG, f"e{attempt}")))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        frozen = not isinstance(outcomes["freeze"], ErrorCode)
        submitted = not isinstance(outcomes["submit"], ErrorCode)
        assert frozen != submitted, outcomes
        assert outcomes["freeze" if submitted else "submit"] == ErrorCode.VERSION_CONFLICT
        assert len(engine.list_jobs(eid)) == int(submitted)


# -- final evaluation ------------------------------------------------------------------------------------------


def test_the_final_report_pairs_the_frozen_candidate_with_the_base_control(engine, workspace, ids, signed_sum):
    eid, candidate, frozen = _repaired_and_frozen(engine, workspace, ids)
    final = engine.final_evaluate(eid)
    assert (final.candidate_checkpoint_id, final.control_checkpoint_id) == (candidate.checkpoint_id,
                                                                           ids["task_checkpoint_id"])
    assert not final.same_checkpoint and final.attempts == 1
    assert final.candidate.metrics["accuracy"] == 1.0 and final.control.metrics["accuracy"] == 0.5
    assert final.candidate.sample_count == final.control.sample_count == 1024
    assert final.candidate.data_hash == final.control.data_hash == final.data_hash
    assert final.candidate.evaluator_config_hash == final.control.evaluator_config_hash
    comparison = final.comparison
    assert comparison.difference == 0.5 and comparison.difference_interval.lower == 0.5
    assert comparison.difference_interval.upper == 0.5  # every pair scores the same: zero width
    assert comparison.candidate_fully_correct.successes == 512
    assert round(100 * comparison.candidate_fully_correct.lower, 2) == 99.28
    assert comparison.base_fully_correct.successes == 0
    assert final.candidate.scores is None and final.control.scores is None  # aggregates only
    closed = engine.get_experiment(eid)
    assert closed.state == "CLOSED" and closed.ended_at is not None
    assert closed.selected_checkpoint_id == frozen.selected_checkpoint_id
    # The sealed result is never re-run, and no other checkpoint is accepted.
    assert engine.final_evaluate(eid) == final == engine.get_final_report(eid)
    _expect(ErrorCode.POLICY_DENIED, engine.final_evaluate, eid, ids["task_checkpoint_id"])
    _expect(ErrorCode.POLICY_DENIED, engine.final_evaluate, eid, ids["reference_checkpoint_id"])
    assert list(inspect.signature(engine.final_evaluate).parameters) == ["experiment_id", "checkpoint_id"]
    assert engine.final_evaluate(eid, candidate.checkpoint_id) == final
    assert [r.split for r in engine.list_reports(eid)] == ["final", "final"]


def test_a_frozen_base_selection_is_evaluated_once_as_its_own_control(engine, workspace, ids):
    experiment = _experiment(engine, ids)
    eid, base = experiment.experiment_id, ids["task_checkpoint_id"]
    _repair(engine, ids, experiment)  # a candidate exists but is not selected
    engine.freeze(eid, base)
    final = engine.final_evaluate(eid)
    assert final.same_checkpoint and final.candidate == final.control
    assert final.candidate.checkpoint_id == base and final.comparison.difference == 0.0
    assert len(engine.list_reports(eid)) == 1


def test_a_failed_final_evaluation_can_be_retried_on_the_same_frozen_model(engine, workspace, ids, monkeypatch):
    eid, candidate, _ = _repaired_and_frozen(engine, workspace, ids)
    original = ClassificationEvaluator.evaluate
    calls = []

    def flaky(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("simulated evaluator crash")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ClassificationEvaluator, "evaluate", flaky)
    error = _expect(ErrorCode.WORKER_FAILED, engine.final_evaluate, eid)
    assert error.retryable and engine.get_experiment(eid).state == "FROZEN"
    assert engine.list_reports(eid) == []
    final = engine.final_evaluate(eid)
    assert final.attempts == 2 and final.candidate_checkpoint_id == candidate.checkpoint_id
    events = [e["event_type"] for e in engine.audit_log(eid)]
    assert events.count("final_evaluation.started") == 2 and events.count("final_evaluation.failed") == 1


def test_final_evaluation_needs_a_frozen_experiment_and_an_operator(engine, workspace, ids):
    experiment = _experiment(engine, ids)
    eid, base = experiment.experiment_id, ids["task_checkpoint_id"]
    _expect(ErrorCode.VERSION_CONFLICT, engine.final_evaluate, eid)
    with Engine(workspace, run_id=experiment.run_id) as bound:
        _expect(ErrorCode.POLICY_DENIED, bound.freeze, eid, base)
        engine.freeze(eid, base)
        _expect(ErrorCode.POLICY_DENIED, bound.final_evaluate, eid)
        _expect(ErrorCode.NOT_FOUND, bound.get_final_report, eid)
    assert engine.get_experiment(eid).state == "FROZEN"


# -- isolation of the final set ------------------------------------------------------------------------------------


def _dump(value) -> str:
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump(mode="json"), sort_keys=True)
    if isinstance(value, list):
        return "[" + ",".join(_dump(item) for item in value) + "]"
    return json.dumps(value, sort_keys=True, default=str)


def test_no_agent_facing_operation_can_read_the_final_set(engine, workspace, ids, signed_sum):
    final_suite = engine.jobs.load_suite(signed_sum.FINAL_SUITE_ID)
    final_data = engine.datasets.load(final_suite.dataset)
    secrets_ = [json.dumps(row) for row in final_data.inputs[:: 97]] + [str(FINAL_TEST_SEED)]
    secrets_ += [repr(row[0]) for row in final_data.inputs[:: 131]]
    experiment = _experiment(engine, ids)
    eid, base = experiment.experiment_id, ids["task_checkpoint_id"]
    outputs: list[str] = []
    with Engine(workspace, run_id=experiment.run_id) as bound:
        candidate = _repair(bound, ids, experiment)
        a = _evaluate(bound, workspace, eid, base, "e1")
        b = _evaluate(bound, workspace, eid, candidate.checkpoint_id, "e2")
        bound.select_checkpoint(eid, candidate.checkpoint_id, base, "s1")
        sealed = [
            (bound.get_suite, signed_sum.FINAL_SUITE_ID),
            (bound.jobs.get_suite, signed_sum.FINAL_SUITE_ID),
            (bound.jobs.load_suite, signed_sum.FINAL_SUITE_ID),
        ]
        for func, *args in sealed:
            outputs.append(_expect(ErrorCode.NOT_FOUND, func, *args).message)
        for func, *args in [(bound.datasets.load, final_suite.dataset), (bound.datasets.generation,
                                                                          final_suite.dataset.data_hash),
                            (bound.register_dataset, signed_sum.suite_data(2, 1))]:
            outputs.append(_expect(ErrorCode.POLICY_DENIED, func, *args).message)
        outputs.append(_expect(ErrorCode.POLICY_DENIED, bound.evaluation.final_suite, ids["task_id"]).message)
        denied = _expect(ErrorCode.POLICY_DENIED, bound.submit_evaluation, eid, base, signed_sum.FINAL_SUITE_ID,
                         {"sample_count": 1024, "seed": 0}, "final")
        outputs.append(denied.message + json.dumps(denied.details))
        outputs += [_dump(value) for value in (
            bound.get_experiment(eid), bound.get_task(ids["task_id"]), bound.get_suite(signed_sum.DEV_SUITE_ID),
            bound.list_models(), bound.list_jobs(eid), bound.get_job(eid, a.job_id), bound.list_reports(eid),
            bound.get_report(eid, b.report_id), bound.compare(eid, a.report_id, b.report_id), bound.audit_log(eid),
            bound.list_experiment_checkpoints(eid),
        )]
        engine.freeze(eid, candidate.checkpoint_id)
        engine.final_evaluate(eid)
        # Once the experiment is closed, its final report is readable, with aggregates only.
        outputs += [_dump(value) for value in (bound.get_final_report(eid), bound.list_reports(eid),
                                               bound.audit_log(eid), bound.get_experiment(eid))]
        final_report_id = bound.get_final_report(eid).candidate.report_id
        _expect(ErrorCode.POLICY_DENIED, bound.compare, eid, final_report_id, final_report_id)
        # Final reports are withheld while an experiment is not closed (simulated by reopening it).
        with engine.store.db.transaction() as conn:
            conn.execute("UPDATE experiments SET state = 'FROZEN' WHERE experiment_id = ?", (eid,))
        _expect(ErrorCode.NOT_FOUND, bound.get_report, eid, final_report_id)
        _expect(ErrorCode.NOT_FOUND, bound.get_final_report, eid)
        assert all(report.split == "development" for report in bound.list_reports(eid))
    text = "\n".join(outputs)
    stored = (workspace / "evaluator" / "datasets").glob("*.json")
    stored_text = "".join(path.read_text() for path in stored)
    assert all(secret.replace(" ", "") in stored_text.replace(" ", "") for secret in secrets_)  # positive control
    assert len(outputs) > 20 and final_suite.dataset.data_hash in text  # the final report was among the outputs
    for secret in secrets_:
        assert secret not in text, secret
    # No metadata row anywhere carries a final example or the final seed: they live in evaluator storage only.
    database = (workspace / "metadata.sqlite").read_bytes() + (workspace / "metadata.sqlite-wal").read_bytes() \
        if (workspace / "metadata.sqlite-wal").exists() else (workspace / "metadata.sqlite").read_bytes()
    for secret in secrets_:
        assert secret.encode() not in database, secret


# -- CLI and Markdown ------------------------------------------------------------------------------------------------


def _cli(workspace, *args):
    env = {**os.environ, "WSE_WORKSPACE": str(workspace)}
    return subprocess.run([sys.executable, "-m", "wse.cli", *args], capture_output=True, text=True, env=env,
                          timeout=120)


def test_the_cli_freezes_and_final_evaluates_with_json_and_markdown_reports(engine, workspace, ids, tmp_path):
    experiment = _experiment(engine, ids)
    eid = experiment.experiment_id
    candidate = _repair(engine, ids, experiment)
    report_id = _evaluate(engine, workspace, eid, candidate.checkpoint_id, "e1").report_id
    engine.select_checkpoint(eid, candidate.checkpoint_id, ids["task_checkpoint_id"], "s1")
    stale = _cli(workspace, "freeze", eid, "--expected-selected-id", ids["task_checkpoint_id"])
    assert stale.returncode == 2 and json.loads(stale.stderr)["error"]["code"] == "VERSION_CONFLICT"
    frozen = _cli(workspace, "freeze", eid, "--expected-selected-id", candidate.checkpoint_id)
    assert frozen.returncode == 0 and json.loads(frozen.stdout)["state"] == "FROZEN"
    final = _cli(workspace, "final-evaluate", eid)
    assert final.returncode == 0, final.stderr
    sealed = json.loads(final.stdout)
    assert sealed["candidate"]["metrics"] == {"accuracy": 1.0} and sealed["control"]["metrics"] == {"accuracy": 0.5}
    markdown_path = tmp_path / "final.md"
    again = _cli(workspace, "final-evaluate", eid, "--format", "markdown", "--output", str(markdown_path))
    assert again.returncode == 0 and again.stdout == markdown_path.read_text()
    assert "512/512 pairs fully correct, exact 95% CI [99.28%, 100.00%]" in again.stdout
    assert "Difference (candidate - control): 50.00%" in again.stdout
    assert engine.get_final_report(eid).model_dump(mode="json") == sealed
    report = _cli(workspace, "report", eid, report_id, "--format", "markdown")
    assert report.returncode == 0 and "accuracy: 100.00%" in report.stdout
    assert "Development target met: yes" in report.stdout
    assert json.loads(_cli(workspace, "report", eid, report_id).stdout)["report_id"] == report_id


def test_markdown_renders_comparisons(engine, workspace, ids):
    experiment = _experiment(engine, ids)
    eid = experiment.experiment_id
    candidate = _repair(engine, ids, experiment)
    a = _evaluate(engine, workspace, eid, ids["task_checkpoint_id"], "e1").report_id
    b = _evaluate(engine, workspace, eid, candidate.checkpoint_id, "e2").report_id
    text = render_markdown(engine.compare(eid, a, b))
    assert "Base accuracy 50.00%; candidate accuracy 100.00%" in text
    assert "improved / worsened / unchanged: 32 / 0 / 0" in text


def test_concurrent_final_evaluations_seal_one_result(engine, workspace, ids):
    eid, _, _ = _repaired_and_frozen(engine, workspace, ids)
    barrier, results = threading.Barrier(2), []

    def run():
        with Engine(workspace, recover=False) as own:
            barrier.wait()
            results.append(own.final_evaluate(eid))

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 2 and results[0] == results[1] == engine.get_final_report(eid)
    assert len(engine.list_reports(eid)) == 2  # one candidate and one control report, written once
