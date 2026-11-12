"""``Experiment`` (``wse.experiment``): one run from a model, a goal, development data, and a rule to a frozen choice.

The runs here use the tiny Llama fixture and a stepwise strategy registered
for the test (a toolchain check through the runner's dispatch, not an
agent): it commits one candidate and evaluates it on the full development
set. The runner evaluates the baseline first, applies the selection rule
when the strategy stops, and freezes; ``test`` then registers the test data
as the run's final suite.
"""

from __future__ import annotations

import json
from collections.abc import Generator

import pytest
from test_highlevel import REPO, REVISION, ROWS

from wse.schemas import ErrorCode, WSEError
from wse.strategies.base import Action, Observation

DOWN0 = "model.layers.0.mlp.down_proj.weight"
TEST_ROWS = [{"problem": f"{a}*{b}?", "answer": str(a * b)} for a, b in ((2, 3), (3, 3), (4, 2), (5, 2))]


class _OneCandidate:
    """Commits one fill_slice candidate and evaluates it on the full development suite, then stops."""

    strategy_id = "experiment_check"
    version = "1"

    def __init__(self) -> None:
        self._plan: Generator[Action, Observation, None] | None = None
        self._seen = 0

    def config(self) -> dict:
        return {}

    def next_action(self, objective, observations, remaining_budget) -> Action:
        try:
            if self._plan is None:
                self._seen = len(observations)
                self._plan = self._run(observations[0])
                return next(self._plan)
            new = observations[self._seen:]
            self._seen = len(observations)
            return self._plan.send(new[-1])
        except StopIteration:
            return Action.stop("one candidate evaluated")

    def _run(self, first: Observation) -> Generator[Action, Observation, None]:
        description = first.result
        eid = description["experiment"]["experiment_id"]
        base = description["experiment"]["base_checkpoint_id"]
        suite_id = description["task"]["development_suite_id"]
        suite = next(s for s in description["suites"] if s["suite_id"] == suite_id)
        preview = (yield Action.call("wse_preview_patch", "zero a region", experiment_id=eid, idempotency_key="p",
                                     patch_spec={"base_checkpoint_id": base, "hypothesis": "Zero a small region.",
                                                 "operations": [{"op": "fill_slice", "tensor": DOWN0, "value": 0.0,
                                                                 "selection": [{"start": 0, "stop": 2, "step": 1},
                                                                               {"start": 0, "stop": 3, "step": 1}]}]}))
        candidate = (yield Action.call("wse_apply_patch", "commit", experiment_id=eid,
                                       patch_id=preview.result["patch_id"], idempotency_key="a"))
        job = (yield Action.call("wse_submit_evaluation", "evaluate", experiment_id=eid,
                                 checkpoint_id=candidate.result["checkpoint_id"], suite_id=suite_id,
                                 eval_config={"sample_count": max(suite["sample_counts"]), "seed": suite["seeds"][0]},
                                 idempotency_key="e")).result
        while job["state"] in ("queued", "running"):
            job = (yield Action.call("wse_get_job", "wait", experiment_id=eid, job_id=job["job_id"],
                                     wait_seconds=10)).result


@pytest.fixture
def experiment_setup(hf_snapshots, tmp_path):
    from wse import Experiment, Model

    dev = tmp_path / "dev.jsonl"
    dev.write_text("".join(json.dumps(row) + "\n" for row in ROWS[:4]), encoding="utf-8")
    model = Model.from_pretrained(hf_snapshots["llama_untied"], repo=REPO, revision=REVISION,
                                  workspace=tmp_path / "workspace")
    model.engine.registry.register("strategy", "experiment_check", "1", _OneCandidate)
    experiment = Experiment(model, "Make the model finish more answers.", dev, evaluator="math",
                            minimize="limit_hit_rate", preserve=["accuracy"], max_new_tokens=6,
                            profile="qwen35think@1")
    yield model, experiment, tmp_path
    model.close()


def test_a_run_evaluates_the_baseline_applies_the_rule_and_freezes(experiment_setup) -> None:
    from wse.selection import select_by_rule

    model, experiment, tmp_path = experiment_setup
    result = experiment.run(agent="strategy:experiment_check@1", sandbox=None, max_evaluations=3, device="cpu")
    engine = model.engine
    assert result.state == "FROZEN"
    reports = engine.list_reports(result.experiment_id)
    assert [r.checkpoint_id == model.checkpoint_id for r in reports] == [True, False]  # the baseline came first
    passes = engine.get_experiment(result.experiment_id).budget["generation_passes"]
    assert (passes.consumed, passes.limit) == (2, 3)
    decision = result.selection()
    task = engine.get_task(engine.get_experiment(result.experiment_id).task_id)
    expected = select_by_rule(task.objective.selection, base_checkpoint_id=model.checkpoint_id, reports=reports)
    assert decision["selected_checkpoint_id"] == expected.selected_checkpoint_id == result.selected_checkpoint_id
    assert result.model.checkpoint_id == result.selected_checkpoint_id
    summary = result.summary()
    assert "Behavior on the development set" in summary and "Candidates (1 evaluated)" in summary
    assert "Rule: WSE keeps the candidate with the lowest limit_hit_rate" in summary and "Audit:" in summary
    assert summary.index("Candidates") < summary.index("Audit:")


def test_test_registers_the_test_data_only_now_and_caches_the_report(experiment_setup) -> None:
    model, experiment, tmp_path = experiment_setup
    result = experiment.run(agent="strategy:experiment_check@1", sandbox=None, max_evaluations=3, device="cpu")
    engine = model.engine
    task_id = engine.get_experiment(result.experiment_id).task_id
    with pytest.raises(WSEError):
        engine.evaluation.final_suite(task_id)  # no test data exists before test() is called
    overlapping = tmp_path / "overlap.jsonl"
    overlapping.write_text("".join(json.dumps(row) + "\n" for row in ROWS[:2]), encoding="utf-8")
    with pytest.raises(WSEError) as caught:
        result.test(overlapping)
    assert caught.value.code == ErrorCode.INVALID_ARGUMENT
    test_data = tmp_path / "test.jsonl"
    test_data.write_text("".join(json.dumps(row) + "\n" for row in TEST_ROWS), encoding="utf-8")
    report = result.test(test_data)
    again = result.test(test_data)
    assert again.report == report.report and engine.get_experiment(result.experiment_id).state == "CLOSED"
    assert report.report.candidate_checkpoint_id == result.selected_checkpoint_id
    assert report.report.control_checkpoint_id == model.checkpoint_id
    table = report.compare()
    assert "accuracy" in table and "limit_hit_rate" in table and "paired" in table
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps({"problem": "9*9?", "answer": "81"}) + "\n", encoding="utf-8")
    with pytest.raises(WSEError) as caught:
        result.test(other)
    assert caught.value.code == ErrorCode.VERSION_CONFLICT


def test_export_writes_the_record_the_selected_model_and_the_summary(experiment_setup) -> None:
    from wse.api import Engine
    from wse.audit import replay_export

    model, experiment, tmp_path = experiment_setup
    result = experiment.run(agent="strategy:experiment_check@1", sandbox=None, max_evaluations=3, device="cpu")
    out = result.export(tmp_path / "artifacts")
    assert (out / "record" / "export_manifest.json").is_file() and (out / "model" / "model.safetensors").is_file()
    assert (out / "summary.txt").read_text(encoding="utf-8").startswith(f"Experiment {result.experiment_id}")
    with Engine(tmp_path / "replay") as clean:
        assert replay_export(out / "record", clean).all_identical


def test_invalid_experiments_are_refused_before_anything_runs(hf_snapshots, tmp_path) -> None:
    from wse import Experiment

    dev = tmp_path / "dev.jsonl"
    dev.write_text(json.dumps(ROWS[0]) + "\n", encoding="utf-8")
    for options in ({"minimize": "repetition_rate"}, {"minimize": "limit_hit_rate", "maximize": "accuracy"},
                    {"minimize": "limit_hit_rate", "preserve": "accuracy"}, {"evaluator": "code"}):
        with pytest.raises(WSEError):
            Experiment("unused/model", "goal", dev, **{"minimize": None, **options})
    with pytest.raises(WSEError):
        Experiment("unused/model", " ", dev, minimize="limit_hit_rate")
