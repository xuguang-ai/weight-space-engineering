"""``Experiment``: an agent investigates and edits a model under a budget; WSE selects, freezes, and tests.

    from wse import Experiment

    experiment = Experiment(
        model="Qwen/Qwen3.5-0.8B",
        goal="Reduce repetitive reasoning so the model finishes more math answers.",
        dev="data/math_dev.jsonl",
        evaluator="math",
        minimize="limit_hit_rate",
        preserve=["accuracy"],
        max_new_tokens=8192,
        workspace="runs/fix-loops",
    )
    result = experiment.run(agent="codex:YOUR_AGENT_MODEL", sandbox="python", max_evaluations=10)
    print(result.summary())
    report = result.test("data/math_test.jsonl")
    print(report.compare())
    result.export("artifacts/fix-loops")

The user names the model (a ``Model`` or a source for
``Model.from_pretrained``), the question (``goal``, the agent's research
brief), the development data, and how it is scored (``evaluator``), and the
rule by which a candidate is kept (``minimize`` or ``maximize`` one metric,
``preserve`` others at the baseline's point estimate). Everything WSE needs
underneath (the dataset, a development suite, a task with the selection
rule, a policy, a run grant) is registered by ``run``; the Engine,
registry, and MCP stay internal.

**One run.** ``run`` registers the development suite and a task of its own
(each run has its own task, so a test set registered later belongs to that
run alone), builds a policy from ``max_evaluations`` and the data (other
limits take the defaults of ``experiment_policy``, which the result
records), grants a run, and runs the agent through the reference runner:
the base checkpoint is evaluated first (``max_evaluations`` counts it, and
an identical earlier evaluation in the workspace is reused when the engine
supports it), the agent works through the run's MCP server, and when it
stops the runner applies the selection rule (``wse.selection``) and freezes
the choice, the original model when no candidate qualifies. The protocol
(examples, prompts, sampling, seeds, token limit, grader) is fixed by the
suite: the agent cannot change it.

**After the run.** ``ExperimentResult.summary`` reports the behavior change,
every candidate (where it changed the weights, its development metrics,
whether the rule kept it, and why), and then the audit details.
``ExperimentResult.model`` is the frozen choice as a ``Model``.
``ExperimentResult.test`` evaluates only the frozen choice and the original
model, on test data registered at that moment as the run's final suite
(rejected if it overlaps the development data); the agent never had a path
to it. The same request returns the stored report; a different test set
needs a new run. ``ExperimentResult.export`` writes the exact patches, the
reports, the agent's trace and code records, the selected model, and the
summary. ``ExperimentResult.open`` reopens a finished experiment from its
workspace.

**What is verified.** A run whose agent executes its own code is labeled
``unverified``: WSE meters its own calls (evaluations, probes, previews,
reads), not what the agent's code computes, and a general Python
environment holding the weights cannot be shown, by rules alone, not to run
the model itself. The final test's isolation does not depend on that label.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wse.highlevel import EVALUATORS, GRADERS, METRICS, Model, _evaluation_format, _invalid, _new_directory
from wse.schemas import METRIC_DIRECTIONS, ErrorCode, WSEError, json_hash

__all__ = ["Experiment", "ExperimentResult", "TestReport", "experiment_policy", "parse_agent"]

DEFAULT_MAX_EVALUATIONS = 10
DEFAULT_WALL_CLOCK_HOURS = 12.0  # ten evaluations of Qwen3.5-0.8B at 8,192 tokens take about 6.5 hours
DEFAULT_MAX_TURNS = 400
GENERATION_EVALUATOR = "generation@1"
TEST_DOUBLE_PREFIXES = ("test-double:", "wse-test-double:")


# ---------------------------------------------------------------------------
# The policy and the agent of a run
# ---------------------------------------------------------------------------


def experiment_policy(*, problems: int, max_new_tokens: int, max_evaluations: int = DEFAULT_MAX_EVALUATIONS,
                      samples: int = 1, sandbox: str | None = "python") -> Any:
    """The policy of an ``Experiment`` run: ``qwen_loop_v1``'s operations and limits, sized to the run.

    * ``max_evaluations`` generation passes of at most ``problems`` problems
      and ``max_new_tokens`` new tokens per output, enough generated tokens
      for all of them (one evaluation is one checkpoint on the whole
      development set; the baseline is one of them);
    * 25 probe calls of at most 64 texts of up to 1,024 tokens, 50 analyses,
      100 reads, 50 previews, and 20 candidate commits;
    * 24 hours of WSE-metered computation, at most 4 hours per operation, and
      job waits of up to 10 minutes per call;
    * the edits ``scale_slice``, ``set_slice``, ``fill_slice``,
      ``project_out``, ``add_low_rank``, and ``replace_tensor`` (tensors the
      agent computed);
    * with ``sandbox="python"``, code execution in the run's sandbox: 400
      calls of at most 10 minutes each and 8 hours in all.
    """
    from wse.policy import BudgetLimits, ExperimentPolicy, SandboxPolicy, qwen_loop_policy

    if sandbox not in (None, "python"):
        raise _invalid(f"sandbox must be 'python' or None, not {sandbox!r}")
    for name, value in (("problems", problems), ("max_new_tokens", max_new_tokens),
                        ("max_evaluations", max_evaluations), ("samples", samples)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise _invalid(f"{name} must be a positive integer")
    base = qwen_loop_policy()
    limits = {**base.budget.model_dump(),
              "generation_passes": max_evaluations, "problems_per_generation_pass": problems,
              "new_tokens_per_output": max_new_tokens,
              "generated_tokens": max_evaluations * problems * samples * max_new_tokens,
              "seconds_per_operation": 4 * 3600.0, "compute_seconds": 24 * 3600.0}
    fields: dict[str, Any] = {"mode": "bounded_eval",
                              "operations": base.operations.with_edit("replace_tensor", "replace_tensor@1"),
                              "max_job_wait_seconds": 600.0}
    if sandbox == "python":
        limits.update(sandbox_calls=400, sandbox_seconds=8 * 3600.0)
        fields["sandbox"] = SandboxPolicy(seconds_per_call=600.0)
    budget = BudgetLimits(**limits)
    content = {"budget": budget.model_dump(mode="json"), **{k: (v.model_dump(mode="json") if hasattr(v, "model_dump")
                                                                else v) for k, v in fields.items()}}
    policy_id = "experiment_" + json_hash(content).removeprefix("sha256:")[:16]
    return ExperimentPolicy(policy_id=policy_id, budget=budget, **fields)


def parse_agent(agent: str, *, max_turns: int | None = None, wall_clock_seconds: float | None = None) -> Any:
    """The controller configuration of an agent string.

    * ``"codex:MODEL"`` (or a bare ``"MODEL"``): the Codex CLI runs a model
      agent (``codex_cli@1``) with the operator's credentials;
    * ``"test-double:STRATEGY@VERSION"``: a registered stepwise strategy
      chooses every tool call through the same MCP path, offline (a toolchain
      check, not an agent);
    * ``"strategy:STRATEGY@VERSION"``: a registered strategy run by the
      runner's dispatch, without MCP.
    """
    from wse.agent_adapters.base import AgentConfig
    from wse.runner import StrategyConfig

    if not isinstance(agent, str) or not agent.strip():
        raise _invalid("agent must be a string such as 'codex:YOUR_AGENT_MODEL'")
    agent = agent.strip()
    limits: dict[str, Any] = {}
    if max_turns is not None:
        limits["max_turns"] = max_turns
    if wall_clock_seconds is not None:
        limits["wall_clock_seconds"] = wall_clock_seconds
    if agent.startswith("strategy:"):
        options = {"wall_clock_seconds": wall_clock_seconds} if wall_clock_seconds is not None else {}
        return StrategyConfig(strategy=agent.removeprefix("strategy:"), **options)
    for prefix in TEST_DOUBLE_PREFIXES:
        if agent.startswith(prefix):
            return AgentConfig(adapter=_agent_adapter(), model="wse-test-double:" + agent.removeprefix(prefix),
                               **limits)
    model = agent.removeprefix("codex:")
    if not model:
        raise _invalid("name the agent's model, for example 'codex:gpt-5.6-luna'")
    return AgentConfig(adapter=_agent_adapter(), model=model, benchmark=False, **limits)


def _agent_adapter() -> str:
    """The agent runner: the Codex CLI adapter when this WSE has it, else the SDK adapter."""
    from wse.runner import AGENT_ADAPTERS

    return "codex_cli@1" if "codex_cli@1" in AGENT_ADAPTERS else next(iter(AGENT_ADAPTERS))


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


class Experiment:
    """A model, a goal, development data, an evaluator, and a selection rule (module docstring)."""

    def __init__(self, model: Model | str | Path, goal: str, dev: str | Path, *, evaluator: str = "math",
                 minimize: str | None = None, maximize: str | None = None, preserve: Sequence[str] = (),
                 max_new_tokens: int | None = None, workspace: str | Path | None = None, profile: str | None = None,
                 problems: int | None = None, subset_seed: int = 0, samples: int = 1, seed: int = 0,
                 repo: str | None = None, revision: str | None = None) -> None:
        from wse.schemas import SelectionRule

        if not isinstance(goal, str) or not goal.strip():
            raise _invalid("goal must state, in words, what the agent should investigate and change")
        if len(goal) > 4096:
            raise _invalid("goal must be at most 4,096 characters")
        if evaluator not in EVALUATORS:
            raise _invalid(f"unknown evaluator {evaluator!r}; the built-in evaluators are {list(EVALUATORS)}")
        if isinstance(preserve, str):
            raise _invalid("preserve is a list of metric names, for example ['accuracy']")
        try:
            self.rule = SelectionRule(minimize=minimize, maximize=maximize, preserve=list(preserve))
        except ValueError as exc:
            raise _invalid(f"invalid selection rule: {exc}; metrics are {sorted(METRIC_DIRECTIONS)}") from exc
        if max_new_tokens is not None and (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int)
                                           or max_new_tokens < 1):
            raise _invalid("max_new_tokens must be a positive integer")
        self.dev = Path(dev).expanduser()
        self.format = _evaluation_format(self.dev, evaluator, None)
        if isinstance(model, Model):
            if model.checkpoint_id is None:
                raise _invalid("the model is an edit kept in memory; save it first (edit.save().model)")
            own = Path(model.engine.store.workspace).resolve()
            if workspace is not None and Path(workspace).expanduser().resolve() != own:
                raise _invalid("the model belongs to another workspace; omit workspace= to use the model's own")
        self.model_source = model
        self.goal = goal.strip()
        self.evaluator = evaluator
        self.max_new_tokens = max_new_tokens
        self.workspace = None if workspace is None else Path(workspace).expanduser()
        self.profile = profile
        self.problems = problems
        self.subset_seed = subset_seed
        self.samples = samples
        self.seed = seed
        self.repo, self.revision = repo, revision
        self._model: Model | None = model if isinstance(model, Model) else None

    @property
    def model(self) -> Model:
        """The original model: the given ``Model``, or the one ``from_pretrained`` imports on first use."""
        if self._model is None:
            from wse.highlevel import default_workspace

            self._model = Model.from_pretrained(self.model_source, workspace=self.workspace or default_workspace(),
                                                repo=self.repo, revision=self.revision)
        return self._model

    def _development_data(self) -> tuple[Any, int]:
        from wse.evaluators.generation_runs import select_subset
        from wse.evaluators.text_suites import read_jsonl
        from wse.schemas import TextSuiteData

        data, _ = read_jsonl(self.dev, format=self.format, id_prefix="dev_")
        if self.problems is not None:
            indices = select_subset([item.item_id for item in data.items], self.problems, self.subset_seed)
            data = TextSuiteData(items=[data.items[index] for index in indices])
        return data, len(data.items)

    def run(self, agent: str, *, sandbox: str | None = "python", max_evaluations: int = DEFAULT_MAX_EVALUATIONS,
            max_turns: int | None = None, wall_clock_hours: float | None = None, device: str | None = None,
            out_dir: str | Path | None = None) -> ExperimentResult:
        """Run the agent once (module docstring) and return the frozen experiment's result.

        ``agent`` is ``"codex:MODEL"`` (see ``parse_agent``); ``sandbox="python"``
        lets the agent run its own Python on the run's inputs (the run is then
        unverified); ``max_evaluations`` counts the baseline; ``max_turns``
        and ``wall_clock_hours`` bound the agent (defaults 400 turns and 12
        hours, its waits for evaluations included); ``device`` is the worker's engine device (default: MPS when
        available); ``out_dir`` receives the runner's logs (default
        ``<workspace>/experiments/<run_id>``).
        """
        from wse.runner import ReferenceRunner

        model = self.model
        engine = model.engine
        data, count = self._development_data()
        if count < 1:
            raise _invalid(f"{self.dev} holds no problems")
        profile = model._profile_ref(self.profile)
        from wse.evaluators.sampling import resolve_profile

        resolved = resolve_profile(engine.registry, profile, model.source)
        max_new_tokens = self.max_new_tokens or resolved.max_new_tokens
        policy = experiment_policy(problems=count, max_new_tokens=max_new_tokens, max_evaluations=max_evaluations,
                                   samples=self.samples, sandbox=sandbox)
        engine.register_policy(policy)
        label = "exp_" + secrets.token_hex(6)
        dataset = engine.register_dataset(data)
        suite_id, task_id = f"{label}_dev", f"{label}_task"
        engine.register_suite({
            "suite_id": suite_id, "task_id": task_id, "split": "development", "evaluator": GENERATION_EVALUATOR,
            "parameters": {"profile": profile, "grader": GRADERS[self.format], "samples_per_problem": self.samples,
                           "max_new_tokens": max_new_tokens},
            "sample_counts": [count], "seeds": [self.seed], "dataset": dataset.model_dump(mode="json")})
        target = self.rule.minimize or self.rule.maximize
        gates = [{"metric": m, "min_delta": 0.0} for m in self.rule.preserve if METRIC_DIRECTIONS[m] == "higher"] \
            or [{"metric": "accuracy", "min_value": 0.0}]
        engine.register_task({
            "task_id": task_id, "title": self.goal.splitlines()[0][:200], "description": self.goal,
            "input_description": "A math problem, followed by the instruction to reason step by step and put the "
                                 "final answer within \\boxed{}.",
            "output_description": "Reasoning, then the final answer within \\boxed{}.",
            "objective": {"task_id": task_id,
                          "description": (f"{'Lower' if self.rule.minimize else 'Raise'} the development {target}"
                                          + (f" while keeping {', '.join(self.rule.preserve)} at least at the "
                                             "baseline's" if self.rule.preserve else "") + "."),
                          "gates": gates, "selection": self.rule.model_dump(mode="json")},
            "development_suite_id": suite_id, "development_examples": count})
        grant = engine.grant_run(task_id, policy.policy_id, model.checkpoint_id)
        config = parse_agent(agent, max_turns=max_turns if max_turns is not None else DEFAULT_MAX_TURNS,
                             wall_clock_seconds=3600.0 * (wall_clock_hours or DEFAULT_WALL_CLOCK_HOURS))
        root = Path(engine.store.workspace)
        out = Path(out_dir).expanduser() if out_dir is not None else root / "experiments" / grant.run_id
        runner = ReferenceRunner(root, registry=engine.registry, log_dir=out / "logs", baseline=True,
                                 worker_args=["--device", device] if device else ())
        try:
            runner.run_experiment(grant.run_id, task_id, policy.policy_id, config)
        finally:
            if runner.last_experiment_id is not None:
                (out / "experiment.json").parent.mkdir(parents=True, exist_ok=True)
                (out / "experiment.json").write_text(json.dumps({
                    "experiment_id": runner.last_experiment_id, "run_id": grant.run_id, "task_id": task_id,
                    "suite_id": suite_id, "policy_id": policy.policy_id, "agent": agent, "sandbox": sandbox,
                    "max_evaluations": max_evaluations, "dev": str(self.dev), "workspace": str(root)},
                    indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if runner.last_experiment_id is None:  # pragma: no cover - run_experiment raises before an experiment exists
            raise WSEError(ErrorCode.WORKER_FAILED, "the run ended before its experiment existed")
        return ExperimentResult(model, runner.last_experiment_id, out_dir=out)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestReport:
    """The independent test of a frozen experiment: its final report, comparing the choice with the original."""

    __test__ = False  # not a pytest test class

    report: Any
    data: Path | None = None

    def compare(self) -> str:
        """The original and the selected model on the test data: each metric, the difference, and its interval.

        Accuracy's interval is the paired 95% interval of the difference; the
        other metrics give each model's own 95% interval.
        """
        report = self.report
        selected, original, paired = report.candidate, report.control, report.comparison
        same = " (the original model: no candidate was kept)" if report.same_checkpoint else ""
        lines = [f"test data {self.data or report.suite_id}: {selected.sample_count} problems; selected "
                 f"{report.candidate_checkpoint_id}{same}, original {report.control_checkpoint_id}",
                 f"{'metric':<24}{'original':>10}{'selected':>10}   difference [95% interval]"]
        for name in METRICS:
            if name not in selected.metrics or name not in original.metrics:
                continue
            digits = 1 if name == "mean_generated_tokens" else 3
            before, after = original.metrics[name], selected.metrics[name]
            if name == paired.metric:
                interval = f"[{paired.difference_interval.lower:+.3f}, {paired.difference_interval.upper:+.3f}] paired"
            else:
                bounds = [report.intervals.get(name) for report in (original, selected)]
                interval = "; ".join(f"{label} [{b.lower:.{digits}f}, {b.upper:.{digits}f}]"
                                     for label, b in zip(("original", "selected"), bounds) if b is not None)
            lines.append(f"{name:<24}{before:>10.{digits}f}{after:>10.{digits}f}   {after - before:+.{digits}f} "
                         f"{interval}")
        return "\n".join(lines)


class ExperimentResult:
    """A frozen experiment of a run (module docstring): ``summary``, ``model``, ``test``, and ``export``."""

    def __init__(self, original: Model, experiment_id: str, *, out_dir: Path | None = None) -> None:
        self.original = original
        self.experiment_id = experiment_id
        self.out_dir = out_dir

    @classmethod
    def open(cls, experiment_id: str, *, workspace: str | Path) -> ExperimentResult:
        """Reopen a finished experiment of ``workspace``."""
        from wse.api import Engine

        with Engine(workspace) as engine:
            base = engine.get_experiment(experiment_id).base_checkpoint_id
        return cls(Model.open(base, workspace=workspace), experiment_id)

    @property
    def engine(self) -> Any:
        return self.original.engine

    @property
    def experiment(self) -> Any:
        return self.engine.get_experiment(self.experiment_id)

    @property
    def state(self) -> str:
        return self.experiment.state

    @property
    def selected_checkpoint_id(self) -> str:
        return self.experiment.selected_checkpoint_id

    @property
    def model(self) -> Model:
        """The frozen choice as a ``Model``: a candidate the rule kept, or the original model."""
        selected = self.selected_checkpoint_id
        if selected == self.original.checkpoint_id:
            return self.original
        return Model._open(self.engine, selected)

    def _controller(self) -> dict[str, Any]:
        from wse.audit import controller_record

        return controller_record(self.engine.store.db, self.experiment_id) or {}

    def selection(self) -> dict[str, Any] | None:
        """The selection rule's decision (``wse.selection.SelectionDecision`` as JSON), if the run made one."""
        result = self._controller().get("result") or {}
        return ((result.get("finalization") or {}).get("selection"))

    def summary(self) -> str:
        """What changed and why first (behavior, candidates, choice), then the audit details."""
        from wse.runner import summarize_trial

        engine, eid = self.engine, self.experiment_id
        experiment = engine.get_experiment(eid)
        task = engine.get_task(experiment.task_id)
        decision = self.selection()
        trial = summarize_trial(engine, eid)
        lines = [f"Experiment {eid} ({experiment.state}): {task.description}"]
        rule = task.objective.selection
        if rule is not None:
            from wse.selection import describe_rule

            lines.append("Rule: " + describe_rule(rule))
        if decision is not None and decision.get("baseline_metrics") is not None:
            base = decision["baseline_metrics"]
            chosen = next((c for c in decision["candidates"]
                           if c["checkpoint_id"] == decision["selected_checkpoint_id"]), None)
            lines.append("")
            lines.append("Behavior on the development set (point estimates):")
            names = [n for n in METRICS if n in base]
            lines.append(f"  {'metric':<24}{'original':>10}{'selected':>10}")
            for name in names:
                digits = 1 if name == "mean_generated_tokens" else 3
                after = chosen["metrics"].get(name, base[name]) if chosen else base[name]
                lines.append(f"  {name:<24}{base[name]:>10.{digits}f}{after:>10.{digits}f}")
            lines.append("Choice: " + decision["reason"])
            lines.append("")
            lines.append(f"Candidates ({len(decision['candidates'])} evaluated):")
            for candidate in decision["candidates"]:
                lines.append(self._candidate_line(candidate))
        elif decision is not None:
            lines.append("Choice: " + decision["reason"])
        unevaluated = self._unevaluated_candidates(decision)
        if unevaluated:
            lines.append(f"Candidates without a full development evaluation ({len(unevaluated)}): "
                         + ", ".join(unevaluated))
        result = self._controller().get("result") or {}
        if result.get("final_message"):
            lines.append("")
            lines.append("The agent's final message: " + str(result["final_message"])[:2000])
        lines.append("")
        lines.append("Audit:")
        lines.append(f"  controller {trial.get('controller')} model {trial.get('model')}, ended "
                     f"{trial.get('status')} ({trial.get('reason')})")
        budget = experiment.budget
        passes = budget.get("generation_passes")
        if passes is not None:
            lines.append(f"  evaluations {passes.consumed:g} of {passes.limit:g} (the baseline included)")
        lines.append(f"  constraint status {trial.get('constraint_status')}"
                     + (f" ({trial.get('constraint_reason')})" if trial.get("constraint_reason") else ""))
        lines.append(f"  WSE-metered computation {trial.get('compute_seconds')} s; agent tokens "
                     f"{(trial.get('tokens') or {}).get('total_tokens')}")
        if trial.get("sandbox"):
            lines.append(f"  sandbox calls {trial['sandbox']['calls']}, {trial['sandbox']['wall_seconds']:.1f} s of "
                         f"{trial['sandbox']['label']}")
        if self.out_dir is not None:
            lines.append(f"  logs {self.out_dir / 'logs'}")
        return "\n".join(lines)

    def _candidate_line(self, candidate: dict[str, Any]) -> str:
        engine, eid = self.engine, self.experiment_id
        checkpoint = next((c for c in engine.list_experiment_checkpoints(eid)
                           if c.checkpoint_id == candidate["checkpoint_id"]), None)
        where, hypothesis = "", ""
        if checkpoint is not None and checkpoint.patch_id is not None:
            preview = engine.get_patch(eid, checkpoint.patch_id)
            tensors = sorted({diff.tensor for diff in preview.tensors if diff.changed_elements})
            where = f"{preview.changed_elements:,} elements in {', '.join(tensors[:4])}" + (
                f" and {len(tensors) - 4} more" if len(tensors) > 4 else "")
            hypothesis = preview.hypothesis
        metrics = ", ".join(f"{name} {candidate['metrics'][name]:.4g}" for name in METRICS
                            if name in candidate["metrics"])
        verdict = "kept" if candidate["qualifies"] else "not kept: " + "; ".join(candidate["reasons"])
        return (f"  - {candidate['checkpoint_id']}: {metrics}; {where}; {verdict}"
                + (f"\n    hypothesis: {hypothesis}" if hypothesis else ""))

    def _unevaluated_candidates(self, decision: dict[str, Any] | None) -> list[str]:
        judged = {c["checkpoint_id"] for c in (decision or {}).get("candidates", [])}
        return [c.checkpoint_id for c in self.engine.list_experiment_checkpoints(self.experiment_id)
                if c.role == "candidate" and c.checkpoint_id not in judged]

    def test(self, data: str | Path) -> TestReport:
        """Evaluate the frozen choice and the original model on independent test data (module docstring).

        The data is registered now, as this run's final suite, with the
        development suite's protocol and grader (the evaluator reads its
        format); a test set that shares a problem with the development data
        is refused. The same request again returns the stored report; another
        test set for the same run is refused.
        """
        from wse.evaluators.text_suites import read_jsonl

        engine, eid = self.engine, self.experiment_id
        experiment = engine.get_experiment(eid)
        if experiment.state == "OPEN":
            raise WSEError(ErrorCode.VERSION_CONFLICT, "the experiment is still open; test it after its run froze it")
        path = Path(data).expanduser()
        task = engine.get_task(experiment.task_id)
        development = engine.get_suite(task.development_suite_id)
        test_format = _evaluation_format(path, "math", None)
        items, _ = read_jsonl(path, format=test_format, id_prefix="test_")
        dataset = engine.register_dataset(items)
        suite_id = f"{experiment.task_id.removesuffix('_task')}_test"
        try:
            existing = engine.evaluation.final_suite(experiment.task_id)
        except WSEError as exc:
            if exc.code != ErrorCode.NOT_FOUND:
                raise
            existing = None
        if existing is not None and existing.dataset.data_hash != dataset.data_hash:
            raise WSEError(ErrorCode.VERSION_CONFLICT, "this run was tested on other data; a new test set needs a new "
                           "run", details={"tested_data_hash": existing.dataset.data_hash})
        if existing is None:
            parameters = dict(development.parameters)
            parameters["grader"] = GRADERS[test_format]
            engine.register_suite({
                "suite_id": suite_id, "task_id": experiment.task_id, "split": "final",
                "evaluator": development.evaluator, "parameters": parameters,
                "sample_counts": [len(items.items)], "seeds": [development.seeds[0]],
                "dataset": dataset.model_dump(mode="json")})
        report = engine.final_evaluate(eid)
        return TestReport(report=report, data=path)

    def export(self, path: str | Path) -> Path:
        """Write the experiment's record, the selected model, and the summary into ``path`` (new or empty).

        ``record/`` is the experiment export (exact patches, reports, audit
        events, the agent's trace and code records, the base model), which
        ``wse replay`` rebuilds; ``model/`` is the selected model as a Hugging
        Face directory (``Model.export``); ``summary.txt`` is ``summary()``.
        """
        from wse.audit import export_experiment

        directory = _new_directory(path)
        export_experiment(self.engine, self.experiment_id, directory / "record")
        self.model.export(directory / "model")
        (directory / "summary.txt").write_text(self.summary() + "\n", encoding="utf-8")
        return directory
