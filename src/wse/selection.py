"""The selection rule of a task: which checkpoint an experiment keeps when its run ends.

A task may state a rule in its objective (``ObjectiveSpec.selection``): one
metric to optimize (``minimize`` or ``maximize``) and metrics to preserve.
When the run ends, before the freeze, the runner applies the rule to the
experiment's full development reports (``select_by_rule``):

* the baseline is the base checkpoint's report; without one, nothing can be
  judged and the base is kept;
* a candidate qualifies when its optimized metric strictly improves on the
  baseline's and none of its preserved metrics is worse than the baseline's
  point estimate (``METRIC_DIRECTIONS`` says which way is better). This is a
  comparison of point estimates on the fixed development set, not a test of
  significance, and it says nothing about other data;
* among qualifying candidates the best optimized value wins; ties go to the
  better preserved metrics in the order listed, then to the fewer changed
  elements, then to the earlier candidate;
* when no candidate qualifies, the base checkpoint is kept.

The decision records the rule, the baseline, every candidate with its
metrics and the reasons it did or did not qualify, the choice, and why.
Differences below ``TOLERANCE`` count as ties: metrics are means over
samples.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import Field

from wse.schemas import METRIC_DIRECTIONS, SelectionRule, _Frozen

__all__ = ["METRIC_DIRECTIONS", "TOLERANCE", "SelectionCandidate", "SelectionDecision", "describe_rule",
           "select_by_rule"]

TOLERANCE = 1e-9


class SelectionCandidate(_Frozen):
    """One evaluated candidate under the rule: its report, metrics, changed elements, and whether it qualified."""

    checkpoint_id: str
    report_id: str
    metrics: dict[str, float]
    changed_elements: int | None = None
    qualifies: bool
    reasons: list[str] = Field(default_factory=list)


class SelectionDecision(_Frozen):
    """What ``select_by_rule`` decided, and why (module docstring)."""

    rule: SelectionRule
    baseline_checkpoint_id: str
    baseline_report_id: str | None
    baseline_metrics: dict[str, float] | None
    candidates: list[SelectionCandidate]
    selected_checkpoint_id: str
    reason: str


def describe_rule(rule: SelectionRule) -> str:
    """The rule in one sentence, for instructions and summaries."""
    target = rule.minimize or rule.maximize
    verb = "lowest" if rule.minimize else "highest"
    kept = (", with " + " and ".join(f"{m} no worse than the baseline's" for m in rule.preserve)) if rule.preserve \
        else ""
    return (f"WSE keeps the candidate with the {verb} {target} on the full development set among those that "
            f"improve on the baseline's {target}{kept}; when none does, it keeps the original model.")


def _better(metric: str, value: float, than: float, direction: str) -> bool:
    return value < than - TOLERANCE if direction == "lower" else value > than + TOLERANCE


def _worse(metric: str, value: float, than: float) -> bool:
    return _better(metric, than, value, METRIC_DIRECTIONS[metric])


def select_by_rule(rule: SelectionRule, *, base_checkpoint_id: str, reports: Sequence[Any],
                   changed_elements: Mapping[str, int] | None = None,
                   order: Sequence[str] | None = None) -> SelectionDecision:
    """Apply ``rule`` to full development ``reports`` (each with ``checkpoint_id``, ``report_id``, ``metrics``).

    Only the first report of each checkpoint counts. ``changed_elements``
    maps candidates to their changed elements (the tie-break); ``order``
    lists candidates in the order they were made (the last tie-break;
    default: the order of their reports).
    """
    target = rule.minimize or rule.maximize
    direction = "lower" if rule.minimize else "higher"
    first: dict[str, Any] = {}
    for report in reports:
        first.setdefault(report.checkpoint_id, report)
    baseline = first.get(base_checkpoint_id)
    changed = dict(changed_elements or {})
    sequence = [c for c in (order or list(first)) if c in first and c != base_checkpoint_id]
    if baseline is None:
        return SelectionDecision(
            rule=rule, baseline_checkpoint_id=base_checkpoint_id, baseline_report_id=None, baseline_metrics=None,
            candidates=[], selected_checkpoint_id=base_checkpoint_id,
            reason="the original model was kept: there is no full development evaluation of it to compare with")
    base = {name: float(value) for name, value in baseline.metrics.items()}
    candidates = []
    for checkpoint_id in sequence:
        report = first[checkpoint_id]
        metrics = {name: float(value) for name, value in report.metrics.items()}
        reasons = []
        if target not in metrics or target not in base:
            reasons.append(f"{target} is missing from a report")
        elif not _better(target, metrics[target], base[target], direction):
            reasons.append(f"{target} {metrics[target]:.4g} does not improve on the baseline's {base[target]:.4g}")
        for metric in rule.preserve:
            if metric not in metrics or metric not in base:
                reasons.append(f"{metric} is missing from a report")
            elif _worse(metric, metrics[metric], base[metric]):
                reasons.append(f"{metric} {metrics[metric]:.4g} is worse than the baseline's {base[metric]:.4g}")
        candidates.append(SelectionCandidate(checkpoint_id=checkpoint_id, report_id=report.report_id,
                                             metrics=metrics, changed_elements=changed.get(checkpoint_id),
                                             qualifies=not reasons, reasons=reasons))
    qualifying = [c for c in candidates if c.qualifies]
    if not qualifying:
        return SelectionDecision(
            rule=rule, baseline_checkpoint_id=base_checkpoint_id, baseline_report_id=baseline.report_id,
            baseline_metrics=base, candidates=candidates, selected_checkpoint_id=base_checkpoint_id,
            reason=f"the original model was kept: none of {len(candidates)} evaluated candidates qualified")

    def key(candidate: SelectionCandidate) -> tuple[Any, ...]:
        sign = 1.0 if direction == "lower" else -1.0
        preserved = tuple((1.0 if METRIC_DIRECTIONS[m] == "lower" else -1.0) * candidate.metrics[m]
                          for m in rule.preserve)
        elements = candidate.changed_elements if candidate.changed_elements is not None else 0
        return (round(sign * candidate.metrics[target] / TOLERANCE),) + tuple(
            round(value / TOLERANCE) for value in preserved) + (elements, sequence.index(candidate.checkpoint_id))

    best = min(qualifying, key=key)
    return SelectionDecision(
        rule=rule, baseline_checkpoint_id=base_checkpoint_id, baseline_report_id=baseline.report_id,
        baseline_metrics=base, candidates=candidates, selected_checkpoint_id=best.checkpoint_id,
        reason=(f"{best.checkpoint_id} was kept: {target} {base[target]:.4g} -> {best.metrics[target]:.4g}"
                + "".join(f", {m} {base[m]:.4g} -> {best.metrics[m]:.4g}" for m in rule.preserve)
                + f"; {len(qualifying)} of {len(candidates)} evaluated candidates qualified"))
