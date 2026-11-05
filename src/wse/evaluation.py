"""Tasks, evaluation reports, paired comparisons, and the final evaluation (Sections 3, 4, and 7 of the plan).

* **Tasks.** The operator registers each task definition (``task.json``)
  once; it is immutable and identified by its ``task_hash``. A run-bound
  engine sees only its run's task.
* **Reports.** The worker persists every successful development evaluation
  as an ``EvaluationReport`` in the transaction that settles its job. A
  report binds the checkpoint's model identity, the task, suite, and data
  hashes, the evaluator and its execution configuration, the seed and sample
  count, metrics with cluster-aware intervals, usage, and errors.
* **Comparisons.** ``compare`` pairs two development reports of one
  experiment. It rejects reports whose suite, data, configuration, sample,
  or clusters differ (``INCOMPATIBLE_REPORTS``) and reports the paired
  accuracy difference with a cluster-bootstrap interval, plus exact
  Clopper-Pearson intervals on fully correct clusters.
* **Final evaluation.** An operator action on a ``FROZEN`` experiment: it
  evaluates the frozen selected checkpoint and, as a fixed control, the
  experiment's base checkpoint on the task's final suite with one fixed
  configuration, pairs them, and closes the experiment in one transaction.
  It accepts no other checkpoint. A failed attempt leaves the experiment
  frozen, so it can be retried on the same candidate. Final examples are
  read from evaluator storage only here; final reports carry aggregates only
  (no per-example scores, examples, or generation seed) and are withheld
  from run-bound engines until the experiment is closed.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from wse.audit import record_event
from wse.database import utc_now
from wse.evaluators.classification import fully_correct_interval, paired_comparison
from wse.schemas import (
    OBJECT_ID_PATTERNS,
    ComparisonReport,
    ErrorCode,
    EvalConfig,
    EvaluationReport,
    EvaluationSuite,
    FinalEvaluationReport,
    TaskDefinition,
    Usage,
    WSEError,
    json_hash,
)

if TYPE_CHECKING:
    from wse.api import Engine, EvaluationResult

FINAL_BATCH_SIZE = 64


def _new_report_id() -> str:
    import secrets

    return f"report_{secrets.token_hex(16)}"


class EvaluationService:
    """Reports, comparisons, and final evaluation of an ``Engine``, scoped like its experiment service."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.db = engine.store.db
        self.experiments = engine.experiments

    # -- tasks ------------------------------------------------------------------

    def register_task(self, task: TaskDefinition | Mapping[str, Any]) -> TaskDefinition:
        """Register an immutable task definition (operator only)."""
        self.experiments._require_operator("registering a task")
        try:
            parsed = task if isinstance(task, TaskDefinition) else TaskDefinition.model_validate(dict(task))
        except (ValidationError, TypeError) as exc:
            raise WSEError(ErrorCode.INVALID_ARGUMENT, f"invalid task definition: {exc}") from exc
        with self.db.transaction() as conn:
            row = conn.execute("SELECT task_hash FROM tasks WHERE task_id = ?", (parsed.task_id,)).fetchone()
            if row is None:
                conn.execute("INSERT INTO tasks (task_id, task_hash, task_json, created_at) VALUES (?, ?, ?, ?)",
                             (parsed.task_id, parsed.task_hash, parsed.canonical_json(), utc_now()))
                record_event(conn, "task.registered", parsed.task_id, {"task_hash": parsed.task_hash})
            elif row["task_hash"] != parsed.task_hash:
                raise WSEError(
                    ErrorCode.VERSION_CONFLICT,
                    f"task {parsed.task_id} is already registered with different content; task definitions are "
                    "immutable, so register a new task ID instead",
                    details={"task_id": parsed.task_id, "registered_hash": row["task_hash"]},
                )
        return parsed

    def _task_definition(self, task_id: str, conn: sqlite3.Connection | None = None) -> TaskDefinition | None:
        """The registered definition of ``task_id`` (unscoped), or ``None``."""
        row = (conn or self.db.conn).execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row is None:
            return None
        task = TaskDefinition.model_validate_json(row["task_json"])
        if task.task_hash != row["task_hash"]:
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"stored task {task_id} does not match its hash")
        return task

    def get_task(self, task_id: Any) -> TaskDefinition:
        """A registered task; a run-bound service sees only its run's task."""
        bound = self.experiments.bound_run_id
        visible = bound is None or task_id == self.experiments._grant_row(bound)["task_id"]
        task = self._task_definition(task_id) if isinstance(task_id, str) and visible else None
        if task is None:
            raise WSEError(ErrorCode.NOT_FOUND, f"task {task_id!r} is not registered",
                           details={"task_id": str(task_id)})
        return task

    # -- reports ----------------------------------------------------------------

    def build_report(self, result: EvaluationResult, *, experiment_id: str, job_id: str | None,
                     checkpoint_id: str, suite: EvaluationSuite, usage: Usage, keep_scores: bool) -> EvaluationReport:
        """Bind an evaluation result to its experiment, checkpoint, task, and suite as a managed report."""
        task = self._task_definition(suite.task_id)
        target_met = None
        # The development target is a property of the task's own development suite only: a
        # report of another (for example smaller) suite cannot meet it, and final reports carry
        # no development-target verdict at all.
        if suite.split == "development" and task is not None and task.target_accuracy is not None \
                and "accuracy" in result.metrics:
            target_met = (suite.suite_id == task.development_suite_id
                          and result.sample_count == suite.example_count == task.development_examples
                          and result.metrics["accuracy"] >= task.target_accuracy and not result.errors)
        return EvaluationReport(
            report_id=_new_report_id(),
            experiment_id=experiment_id,
            job_id=job_id,
            checkpoint_id=checkpoint_id,
            model=result.model,
            task_id=suite.task_id,
            task_hash=task.task_hash if task is not None else None,
            suite_id=suite.suite_id,
            suite_hash=suite.suite_hash,
            split=suite.split,
            data_hash=result.data_hash,
            evaluator=result.evaluator,
            evaluator_config_hash=result.evaluator_config_hash,
            execution=result.execution,
            seed=result.execution.eval_config.seed,
            sample_count=result.sample_count,
            metrics=dict(result.metrics),
            clusters=result.clusters,
            intervals=dict(result.intervals),
            scores=list(result.scores) if keep_scores and result.scores is not None else None,
            cluster_ids=list(result.cluster_ids) if keep_scores and result.cluster_ids is not None else None,
            target_met=target_met,
            elapsed_seconds=result.elapsed_seconds,
            usage=usage,
            errors=list(result.errors),
            truncation_count=result.truncation_count,
            created_at=utc_now(),
        )

    @staticmethod
    def insert_report(conn: sqlite3.Connection, report: EvaluationReport) -> None:
        """Persist a report inside the caller's transaction."""
        payload = report.canonical_json()
        conn.execute(
            "INSERT INTO reports (report_id, experiment_id, job_id, checkpoint_id, suite_id, split, report_hash, "
            "report_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (report.report_id, report.experiment_id, report.job_id, report.checkpoint_id, report.suite_id,
             report.split, json_hash(report.model_dump(mode="json")), payload, report.created_at),
        )
        record_event(conn, "report.created", report.report_id,
                     {"job_id": report.job_id, "checkpoint_id": report.checkpoint_id, "split": report.split,
                      "suite_id": report.suite_id, "metrics": report.metrics},
                     experiment_id=report.experiment_id)

    def _withheld(self, row: sqlite3.Row, split: str) -> bool:
        """Final reports stay hidden from a run-bound service until the experiment is closed."""
        return split == "final" and self.experiments.bound_run_id is not None and row["state"] != "CLOSED"

    def _load_report(self, experiment: sqlite3.Row, report_id: Any) -> EvaluationReport:
        row = None
        if isinstance(report_id, str) and OBJECT_ID_PATTERNS["report"].fullmatch(report_id):
            row = self.db.conn.execute("SELECT * FROM reports WHERE report_id = ? AND experiment_id = ?",
                                       (report_id, experiment["experiment_id"])).fetchone()
        if row is None or self._withheld(experiment, row["split"]):
            raise WSEError(ErrorCode.NOT_FOUND,
                           f"report {report_id!r} does not exist in experiment {experiment['experiment_id']}",
                           details={"report_id": str(report_id), "experiment_id": experiment["experiment_id"]})
        report = EvaluationReport.model_validate_json(row["report_json"])
        if json_hash(report.model_dump(mode="json")) != row["report_hash"]:
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"stored report {report_id} does not match its hash")
        return report

    def get_report(self, experiment_id: str, report_id: str) -> EvaluationReport:
        return self._load_report(self.experiments._experiment_row(experiment_id), report_id)

    def list_reports(self, experiment_id: str) -> list[EvaluationReport]:
        experiment = self.experiments._experiment_row(experiment_id)
        rows = self.db.conn.execute(
            "SELECT report_id, split FROM reports WHERE experiment_id = ? ORDER BY created_at, rowid",
            (experiment_id,),
        ).fetchall()
        return [self._load_report(experiment, row["report_id"]) for row in rows
                if not self._withheld(experiment, row["split"])]

    # -- comparisons ------------------------------------------------------------

    @staticmethod
    def _check_comparable(base: EvaluationReport, candidate: EvaluationReport) -> None:
        mismatched = [name for name in ("split", "suite_id", "suite_hash", "data_hash", "evaluator",
                                        "evaluator_config_hash", "sample_count", "seed")
                      if getattr(base, name) != getattr(candidate, name)]
        if mismatched:
            raise WSEError(
                ErrorCode.INCOMPATIBLE_REPORTS,
                f"the reports differ in {mismatched}; compare only reports of the same suite, data, configuration, "
                "and sample",
                details={"mismatched": mismatched,
                         "base": {name: str(getattr(base, name)) for name in mismatched},
                         "candidate": {name: str(getattr(candidate, name)) for name in mismatched}},
            )

    @staticmethod
    def _comparison(experiment_id: str, base: EvaluationReport, candidate: EvaluationReport,
                    base_scores: list[int], candidate_scores: list[int], cluster_ids: list[int]) -> ComparisonReport:
        unit = (base.clusters.unit if base.clusters is not None else "example")
        paired = paired_comparison(base_scores, candidate_scores, cluster_ids, unit)
        return ComparisonReport(
            experiment_id=experiment_id,
            base_report_id=base.report_id,
            candidate_report_id=candidate.report_id,
            base_checkpoint_id=base.checkpoint_id,
            candidate_checkpoint_id=candidate.checkpoint_id,
            split=base.split,
            suite_id=base.suite_id,
            suite_hash=base.suite_hash,
            data_hash=base.data_hash,
            evaluator_config_hash=base.evaluator_config_hash,
            sample_count=base.sample_count,
            seed=base.seed,
            base_fully_correct=fully_correct_interval(base_scores, cluster_ids, unit),
            candidate_fully_correct=fully_correct_interval(candidate_scores, cluster_ids, unit),
            **paired,
        )

    def compare(self, experiment_id: str, base_report_id: str, candidate_report_id: str) -> ComparisonReport:
        """Paired comparison of two development reports of the experiment (read-only, not charged)."""
        experiment = self.experiments._experiment_row(experiment_id)
        base = self._load_report(experiment, base_report_id)
        candidate = self._load_report(experiment, candidate_report_id)
        for report in (base, candidate):
            if report.split != "development":
                raise WSEError(ErrorCode.POLICY_DENIED,
                               f"report {report.report_id} is a final report; its comparison is part of the final "
                               "evaluation report", details={"report_id": report.report_id})
        self._check_comparable(base, candidate)
        if base.scores is None or candidate.scores is None or base.cluster_ids is None:
            raise WSEError(ErrorCode.INCOMPATIBLE_REPORTS,
                           f"evaluator {base.evaluator} reports no per-example scores, so its reports cannot be paired",
                           details={"evaluator": str(base.evaluator)})
        if base.cluster_ids != candidate.cluster_ids:
            raise WSEError(ErrorCode.INCOMPATIBLE_REPORTS, "the reports cover different clusters of examples",
                           details={"mismatched": ["cluster_ids"]})
        return self._comparison(experiment_id, base, candidate, base.scores, candidate.scores, base.cluster_ids)

    # -- final evaluation (operator) ---------------------------------------------

    def final_suite(self, task_id: str) -> EvaluationSuite:
        """The task's final suite (operator only; exactly one may be registered per task)."""
        self.experiments._require_operator("reading a final suite")
        rows = self.db.conn.execute("SELECT suite_id FROM suites WHERE task_id = ? AND split = 'final'",
                                    (task_id,)).fetchall()
        if not rows:
            raise WSEError(ErrorCode.NOT_FOUND, f"task {task_id} has no registered final suite",
                           details={"task_id": task_id})
        return self.engine.jobs._read_suite(rows[0]["suite_id"])

    def get_final_report(self, experiment_id: str) -> FinalEvaluationReport:
        """The sealed final report; it exists only once the experiment is closed."""
        experiment = self.experiments._experiment_row(experiment_id)
        row = self.db.conn.execute("SELECT * FROM final_evaluations WHERE experiment_id = ?",
                                   (experiment_id,)).fetchone()
        if row is None or self._withheld(experiment, "final"):
            raise WSEError(ErrorCode.NOT_FOUND, f"experiment {experiment_id} has no final report",
                           details={"experiment_id": experiment_id})
        report = FinalEvaluationReport.model_validate_json(row["final_json"])
        if json_hash(report.model_dump(mode="json")) != row["final_hash"]:
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"the final report of {experiment_id} does not match its hash")
        return report

    def final_evaluate(self, experiment_id: str, checkpoint_id: str | None = None) -> FinalEvaluationReport:
        """Evaluate the frozen candidate and the base-checkpoint control, pair them, and close the experiment."""
        self.experiments._require_operator("final evaluation")
        row = self.experiments._experiment_row(experiment_id)
        request_hash = json_hash({"operation": "final_evaluate", "checkpoint_id": str(checkpoint_id)})
        return self.experiments._guarded("final_evaluation", experiment_id, request_hash, None,
                                         lambda: self._final(row, checkpoint_id))

    def _final(self, row: sqlite3.Row, checkpoint_id: Any) -> FinalEvaluationReport:
        experiment_id = row["experiment_id"]
        frozen, base = row["selected_checkpoint_id"], row["base_checkpoint_id"]
        if checkpoint_id is not None and checkpoint_id != frozen:
            raise WSEError(
                ErrorCode.POLICY_DENIED,
                f"final evaluation accepts only the frozen selected checkpoint {frozen}, with the base checkpoint as "
                "its fixed control",
                details={"frozen_checkpoint_id": frozen, "requested_checkpoint_id": str(checkpoint_id)},
            )
        if row["state"] == "CLOSED":
            return self.get_final_report(experiment_id)  # the sealed result is final; nothing is re-run
        if row["state"] != "FROZEN":
            raise WSEError(ErrorCode.VERSION_CONFLICT,
                           f"experiment {experiment_id} is {row['state']}; freeze it before the final evaluation",
                           details={"experiment_id": experiment_id, "state": row["state"]})
        with self.db.transaction() as conn:
            record_event(conn, "final_evaluation.started", experiment_id,
                         {"candidate_checkpoint_id": frozen, "control_checkpoint_id": base},
                         experiment_id=experiment_id)
        try:
            return self._run_final(row)
        except WSEError:
            raise
        except Exception as exc:  # noqa: BLE001 - any evaluation failure leaves the experiment frozen
            raise WSEError(ErrorCode.WORKER_FAILED,
                           f"the final evaluation raised {type(exc).__name__}: {exc}; the experiment stays frozen, "
                           "so it can be retried on the same candidate", retryable=True) from exc

    def _run_final(self, row: sqlite3.Row) -> FinalEvaluationReport:
        from wse.api import evaluate_candidate

        experiment_id = row["experiment_id"]
        frozen, base = row["selected_checkpoint_id"], row["base_checkpoint_id"]
        suite = self.final_suite(row["task_id"])
        resolved = suite.resolved(self.engine.datasets.load(suite.dataset)) if suite.data is None else suite
        config = EvalConfig(sample_count=suite.example_count, seed=suite.seeds[0])
        results = {}
        for checkpoint in dict.fromkeys((frozen, base)):
            results[checkpoint] = evaluate_candidate(self.engine.open_checkpoint(checkpoint), resolved, config,
                                                     registry=self.engine.registry, batch_size=FINAL_BATCH_SIZE)
        candidate_result, control_result = results[frozen], results[base]
        if candidate_result.scores is None:
            raise WSEError(ErrorCode.WORKER_FAILED,
                           f"evaluator {suite.evaluator} reports no per-example scores, so no paired final "
                           "comparison is possible")
        reports = {checkpoint: self.build_report(result, experiment_id=experiment_id, job_id=None,
                                                 checkpoint_id=checkpoint, suite=suite, usage=result.usage,
                                                 keep_scores=False)
                   for checkpoint, result in results.items()}
        candidate, control = reports[frozen], reports[base]
        comparison = self._comparison(experiment_id, control, candidate, list(control_result.scores),
                                      list(candidate_result.scores), list(candidate_result.cluster_ids))
        with self.db.transaction() as conn:
            current = self.experiments._experiment_row(experiment_id, conn)
            if current["state"] == "CLOSED" and current["selected_checkpoint_id"] == frozen:
                sealed_concurrently = True  # another attempt sealed the result first; keep it
            elif current["state"] != "FROZEN" or current["selected_checkpoint_id"] != frozen:
                raise WSEError(ErrorCode.VERSION_CONFLICT,
                               f"experiment {experiment_id} changed during the final evaluation",
                               details={"state": current["state"]})
            else:
                sealed_concurrently = False
                final = self._seal(conn, row, suite, reports, comparison)
        return self.get_final_report(experiment_id) if sealed_concurrently else final

    def _seal(self, conn: sqlite3.Connection, row: sqlite3.Row, suite: EvaluationSuite,
              reports: dict[str, EvaluationReport], comparison: ComparisonReport) -> FinalEvaluationReport:
        """Write the reports and the final report and close the experiment, in the caller's transaction."""
        experiment_id = row["experiment_id"]
        frozen, base = row["selected_checkpoint_id"], row["base_checkpoint_id"]
        candidate, control = reports[frozen], reports[base]
        attempts = conn.execute(
            "SELECT COUNT(*) FROM audit_events WHERE experiment_id = ? AND event_type = 'final_evaluation.started'",
            (experiment_id,),
        ).fetchone()[0]
        final = FinalEvaluationReport(
            experiment_id=experiment_id, run_id=row["run_id"], task_id=row["task_id"],
            task_hash=candidate.task_hash, policy_id=row["policy_id"], suite_id=suite.suite_id,
            suite_hash=suite.suite_hash, data_hash=suite.data_hash, candidate_checkpoint_id=frozen,
            control_checkpoint_id=base, same_checkpoint=frozen == base, candidate=candidate, control=control,
            comparison=comparison, attempts=attempts, created_at=utc_now(),
        )
        for report in reports.values():
            self.insert_report(conn, report)
        conn.execute(
            "INSERT INTO final_evaluations (experiment_id, candidate_report_id, control_report_id, final_hash, "
            "final_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (experiment_id, candidate.report_id, control.report_id, json_hash(final.model_dump(mode="json")),
             final.canonical_json(), final.created_at),
        )
        conn.execute("UPDATE experiments SET state = 'CLOSED', ended_at = ? WHERE experiment_id = ? "
                     "AND state = 'FROZEN'", (final.created_at, experiment_id))
        record_event(conn, "final_evaluation.completed", experiment_id,
                     {"candidate_report_id": candidate.report_id, "control_report_id": control.report_id,
                      "suite_id": suite.suite_id, "data_hash": suite.data_hash, "attempts": attempts},
                     experiment_id=experiment_id)
        record_event(conn, "experiment.closed", experiment_id, {"selected_checkpoint_id": frozen},
                     experiment_id=experiment_id)
        return final


# ---------------------------------------------------------------------------
# Concise Markdown renderings (the JSON forms are the machine-readable reports)
# ---------------------------------------------------------------------------


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def _interval(interval: Any) -> str:
    if interval.method == "clopper_pearson":
        return (f"{interval.successes}/{interval.trials} {interval.cluster_unit}s fully correct, exact 95% CI "
                f"[{_percent(interval.lower)}, {_percent(interval.upper)}]")
    return (f"{_percent(interval.estimate)}, cluster-bootstrap 95% CI [{_percent(interval.lower)}, "
            f"{_percent(interval.upper)}] ({interval.resamples} resamples of {interval.cluster_unit}s, "
            f"seed {interval.seed})")


def evaluation_report_markdown(report: EvaluationReport) -> str:
    """A concise Markdown summary of one evaluation report."""
    lines = [f"# Evaluation report {report.report_id}", "",
             f"- Experiment: `{report.experiment_id}`; checkpoint: `{report.checkpoint_id}` "
             f"(weights `{report.model.weights_hash}`)",
             f"- Suite: `{report.suite_id}` ({report.split}); data `{report.data_hash}`",
             f"- Evaluator: `{report.evaluator}`; configuration `{report.evaluator_config_hash}` "
             f"(sample count {report.sample_count}, seed {report.seed}, batch size {report.execution.batch_size}, "
             f"device {report.execution.device}); {report.management}"]
    for name, value in sorted(report.metrics.items()):
        lines.append(f"- {name}: {_percent(value) if name == 'accuracy' else value}")
    for interval in report.intervals.values():
        lines.append(f"- {_interval(interval)}")
    if report.target_met is not None:
        lines.append(f"- Development target met: {'yes' if report.target_met else 'no'}")
    lines.append(f"- Errors: {'; '.join(report.errors) if report.errors else 'none'}; "
                 f"truncated: {report.truncation_count}; elapsed {report.elapsed_seconds:.3f} s")
    return "\n".join(lines) + "\n"


def comparison_markdown(comparison: ComparisonReport) -> str:
    """A concise Markdown summary of a paired comparison."""
    lines = [f"# Comparison of {comparison.candidate_checkpoint_id} against {comparison.base_checkpoint_id}", "",
             f"- Suite: `{comparison.suite_id}` ({comparison.split}); data `{comparison.data_hash}`; "
             f"{comparison.sample_count} examples, seed {comparison.seed}",
             f"- Base accuracy {_percent(comparison.base_value)}; candidate accuracy "
             f"{_percent(comparison.candidate_value)}",
             f"- Difference (candidate - base): {_interval(comparison.difference_interval)}",
             f"- Base: {_interval(comparison.base_fully_correct)}",
             f"- Candidate: {_interval(comparison.candidate_fully_correct)}",
             f"- Clusters improved / worsened / unchanged: {comparison.clusters_improved} / "
             f"{comparison.clusters_worsened} / {comparison.clusters_unchanged}"]
    return "\n".join(lines) + "\n"


def final_report_markdown(final: FinalEvaluationReport) -> str:
    """A concise Markdown summary of a sealed final evaluation."""
    control = "the same checkpoint (the base was selected)" if final.same_checkpoint else \
        f"`{final.control_checkpoint_id}` (base checkpoint)"
    lines = [f"# Final evaluation of experiment {final.experiment_id}", "",
             f"- Task: `{final.task_id}`; policy: `{final.policy_id}`; run: `{final.run_id}`",
             f"- Final suite: `{final.suite_id}`; data `{final.data_hash}`; attempts: {final.attempts}",
             f"- Frozen candidate: `{final.candidate_checkpoint_id}`; control: {control}",
             f"- Candidate accuracy: {_interval(final.candidate.intervals['accuracy'])}"
             if "accuracy" in final.candidate.intervals else "",
             f"- Control accuracy: {_interval(final.control.intervals['accuracy'])}"
             if "accuracy" in final.control.intervals else "",
             f"- Candidate: {_interval(final.comparison.candidate_fully_correct)}",
             f"- Control: {_interval(final.comparison.base_fully_correct)}",
             f"- Difference (candidate - control): {_interval(final.comparison.difference_interval)}",
             f"- Clusters improved / worsened / unchanged: {final.comparison.clusters_improved} / "
             f"{final.comparison.clusters_worsened} / {final.comparison.clusters_unchanged}"]
    return "\n".join(line for line in lines if line) + "\n"


def render_markdown(value: EvaluationReport | ComparisonReport | FinalEvaluationReport) -> str:
    if isinstance(value, FinalEvaluationReport):
        return final_report_markdown(value)
    if isinstance(value, ComparisonReport):
        return comparison_markdown(value)
    return evaluation_report_markdown(value)


__all__ = ["EvaluationService", "FINAL_BATCH_SIZE", "comparison_markdown", "evaluation_report_markdown",
           "final_report_markdown", "render_markdown"]
