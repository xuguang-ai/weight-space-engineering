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
    verdict = "selected" if result.selected_checkpoint_id != model.checkpoint_id else "did not qualify"
    assert f"; {verdict}" in summary and "; kept" not in summary


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


def test_the_agents_inputs_hold_the_baseline_and_never_the_test_data(experiment_setup) -> None:
    from wse.sandbox_runs import layout_for

    model, experiment, tmp_path = experiment_setup
    result = experiment.run(agent="strategy:experiment_check@1", sandbox="python", max_evaluations=3, device="cpu")
    engine = model.engine
    run_id = engine.get_experiment(result.experiment_id).run_id
    layout = layout_for(engine.experiments, run_id)
    baseline = engine.list_reports(result.experiment_id)[0]
    assert (layout.inputs / "weights" / "base.safetensors").is_file()
    assert (layout.inputs / "passes" / f"{baseline.report_id}.jsonl").is_file()
    test_data = tmp_path / "test.jsonl"
    test_data.write_text("".join(json.dumps(row) + "\n" for row in TEST_ROWS), encoding="utf-8")
    result.test(test_data)
    texts = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in layout.root.rglob("*")
                      if path.is_file() and path.suffix in (".json", ".jsonl", ".txt", ".py"))
    assert not any(row["problem"] in texts for row in TEST_ROWS)


def test_a_second_run_on_the_same_data_reuses_the_baseline_on_the_workers_device(experiment_setup) -> None:
    from wse import Experiment
    from wse.runner import render_instructions

    model, experiment, tmp_path = experiment_setup
    first = experiment.run(agent="strategy:experiment_check@1", sandbox=None, max_evaluations=3, device="cpu")
    again = Experiment(model, "Make the model finish more answers.", tmp_path / "dev.jsonl", evaluator="math",
                       minimize="limit_hit_rate", preserve=["accuracy"], max_new_tokens=6, profile="qwen35think@1")
    second = again.run(agent="strategy:experiment_check@1", sandbox=None, max_evaluations=3, device="cpu")
    engine = model.engine
    [baseline] = [r for r in engine.list_reports(second.experiment_id) if r.checkpoint_id == model.checkpoint_id]
    first_baseline = engine.list_reports(first.experiment_id)[0]
    assert baseline.reused_from is not None and baseline.reused_from.report_id == first_baseline.report_id
    assert baseline.metrics == first_baseline.metrics and baseline.execution.device == "cpu"
    passes = engine.get_experiment(second.experiment_id).budget["generation_passes"]
    assert (passes.consumed, passes.limit) == (2, 3)  # the reused baseline is charged as the evaluation it repeats
    experiment_row = engine.get_experiment(second.experiment_id)
    task = engine.get_task(experiment_row.task_id)
    policy = engine.experiments.experiment_policy(second.experiment_id)
    text = render_instructions(task, policy, experiment_id=second.experiment_id, tools=["wse_evaluate"],
                               baseline={"report_id": baseline.report_id, "metrics": dict(baseline.metrics)})
    assert "wse_evaluate on the original model returns this report without charging again" in text


# -- preflight, endings, and the review's other cases ------------------------------------------------------------


def _counts(engine) -> tuple[int, int, int]:
    conn = engine.store.db.conn
    return tuple(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                 for table in ("suites", "tasks", "experiments"))


def test_a_run_that_cannot_start_its_agent_or_sandbox_is_refused_before_anything_runs(experiment_setup,
                                                                                       monkeypatch) -> None:
    import wse.sandbox

    model, experiment, tmp_path = experiment_setup
    before = _counts(model.engine)
    monkeypatch.setenv("WSE_CODEX_PATH", str(tmp_path / "no-such-codex"))
    with pytest.raises(WSEError) as caught:
        experiment.run(agent="codex:some-model", sandbox=None, device="cpu")
    assert caught.value.code == ErrorCode.UNSUPPORTED_CAPABILITY and "no-such-codex" in caught.value.message
    monkeypatch.setattr(wse.sandbox, "available", lambda: (False, "this platform has no sandbox-exec"))
    with pytest.raises(WSEError) as caught:
        experiment.run(agent="strategy:experiment_check@1", sandbox="python", device="cpu")
    assert caught.value.code == ErrorCode.UNSUPPORTED_CAPABILITY and "sandbox-exec" in caught.value.message
    assert _counts(model.engine) == before  # no suite, task, or experiment was registered, nothing was evaluated


def test_a_run_without_its_model_endpoint_is_refused_before_anything_runs(experiment_setup, monkeypatch) -> None:
    from wse.agent_adapters.codex_cli import find_codex

    try:
        binary = find_codex()
    except WSEError:
        pytest.skip("needs a Codex CLI signed by OpenAI")
    model, experiment, tmp_path = experiment_setup
    monkeypatch.setenv("WSE_CODEX_PATH", binary.path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/openai/v1")
    before = _counts(model.engine)
    with pytest.raises(WSEError) as caught:
        experiment.run(agent="codex:some-model", sandbox=None, device="cpu")
    assert caught.value.code == ErrorCode.UNSUPPORTED_CAPABILITY and "OPENAI_API_KEY" in caught.value.message
    assert _counts(model.engine) == before


def test_a_run_whose_agent_never_started_freezes_and_is_not_tested(experiment_setup, monkeypatch) -> None:
    import wse.experiment

    model, experiment, tmp_path = experiment_setup
    monkeypatch.setattr(wse.experiment, "_preflight", lambda config, sandbox: None)  # past the checks on purpose
    monkeypatch.setenv("WSE_CODEX_PATH", str(tmp_path / "no-such-codex"))
    result = experiment.run(agent="codex:some-model", sandbox=None, max_evaluations=3, device="cpu")
    assert result.state == "FROZEN" and not result.agent_started
    assert result.ending["status"] == "connection_failed" and not result.ending["requests"]
    test_data = tmp_path / "test.jsonl"
    test_data.write_text("".join(json.dumps(row) + "\n" for row in TEST_ROWS), encoding="utf-8")
    with pytest.raises(WSEError) as caught:
        result.test(test_data)
    assert caught.value.code == ErrorCode.VERSION_CONFLICT and "never started" in caught.value.message
    task_id = model.engine.get_experiment(result.experiment_id).task_id
    with pytest.raises(WSEError):
        model.engine.evaluation.final_suite(task_id)  # nothing was registered


def test_a_refused_test_set_is_not_stored_and_an_earlier_registration_is_reported(experiment_setup) -> None:
    from wse import Experiment
    from wse.evaluators.text_suites import read_jsonl

    model, experiment, tmp_path = experiment_setup
    first = experiment.run(agent="strategy:experiment_check@1", sandbox=None, max_evaluations=3, device="cpu")
    overlapping = tmp_path / "overlap.jsonl"
    overlapping.write_text("".join(json.dumps(row) + "\n" for row in ROWS[1:3]), encoding="utf-8")
    refused, _ = read_jsonl(overlapping, format="math", id_prefix="test_")
    with pytest.raises(WSEError):
        first.test(overlapping)
    assert not model.engine.datasets.exists(refused.data_hash)
    test_data = tmp_path / "test.jsonl"
    test_data.write_text("".join(json.dumps(row) + "\n" for row in TEST_ROWS), encoding="utf-8")
    assert first.test(test_data, device="cpu").previously_registered == ()
    again = Experiment(model, "Make the model finish more answers.", tmp_path / "dev.jsonl", evaluator="math",
                       minimize="limit_hit_rate", preserve=["accuracy"], max_new_tokens=6, profile="qwen35think@1")
    second = again.run(agent="strategy:experiment_check@1", sandbox=None, max_evaluations=3, device="cpu")
    report = second.test(test_data)
    [earlier] = report.previously_registered
    assert earlier["task_id"] == model.engine.get_experiment(first.experiment_id).task_id
    assert "registered before this run was frozen" in report.compare()
    assert report.report.candidate.execution.device == "cpu"  # the run's device, not the host's default


def test_under_a_rule_the_run_ends_when_no_evaluation_remains_and_the_instructions_agree(experiment_setup) -> None:
    from wse.runner import allowed_tools, render_instructions

    model, experiment, tmp_path = experiment_setup
    result = experiment.run(agent="strategy:experiment_check@1", sandbox=None, max_evaluations=2, device="cpu")
    assert result.ending["stop_condition"].startswith("no_evaluations")
    engine = model.engine
    task = engine.get_task(engine.get_experiment(result.experiment_id).task_id)
    policy = engine.experiments.experiment_policy(result.experiment_id)
    tools = [tool for tool in allowed_tools(policy) if tool != "wse_select_checkpoint"]
    text = render_instructions(task, policy, experiment_id=result.experiment_id, tools=tools)
    assert "wse_select_checkpoint" not in text and "consecutive evaluated candidates" not in text
    assert "Acceptance:" not in text and "you do not select a checkpoint yourself" in text
    assert "no development evaluation remains" in text


def test_the_objective_says_no_worse_for_metrics_where_lower_is_better(hf_snapshots, tmp_path) -> None:
    from wse import Experiment, Model

    dev = tmp_path / "dev.jsonl"
    dev.write_text("".join(json.dumps(row) + "\n" for row in ROWS[:4]), encoding="utf-8")
    with Model.from_pretrained(hf_snapshots["llama_untied"], repo=REPO, revision=REVISION,
                               workspace=tmp_path / "workspace") as model:
        model.engine.registry.register("strategy", "experiment_check", "1", _OneCandidate)
        experiment = Experiment(model, "Make answers right.", dev, evaluator="math", maximize="accuracy",
                                preserve=["limit_hit_rate"], max_new_tokens=6, profile="qwen35think@1")
        result = experiment.run(agent="strategy:experiment_check@1", sandbox=None, max_evaluations=2, device="cpu")
        task = model.engine.get_task(model.engine.get_experiment(result.experiment_id).task_id)
        assert task.objective.description == ("Raise the development accuracy, with limit_hit_rate no worse than the "
                                               "baseline's.")


def test_closing_the_selected_model_keeps_the_workspace_open(experiment_setup) -> None:
    model, experiment, tmp_path = experiment_setup
    result = experiment.run(agent="strategy:experiment_check@1", sandbox=None, max_evaluations=3, device="cpu")
    chosen = result.model
    assert chosen is result.model  # opened once
    with chosen:
        pass
    assert result.summary().startswith(f"Experiment {result.experiment_id}")  # the workspace is still open


def test_a_failure_before_the_agent_starts_still_freezes_and_names_the_experiment(experiment_setup,
                                                                                    monkeypatch) -> None:
    import wse.sandbox_runs

    model, experiment, tmp_path = experiment_setup

    def broken(service, experiment_id):
        raise OSError("disk full")

    monkeypatch.setattr(wse.sandbox_runs, "refresh_inputs", broken)
    with pytest.raises(OSError) as caught:
        experiment.run(agent="strategy:experiment_check@1", sandbox="python", max_evaluations=3, device="cpu")
    [note] = caught.value.__notes__
    experiment_id = note.split()[2]
    assert model.engine.get_experiment(experiment_id).state == "FROZEN" and "ExperimentResult.open" in note
