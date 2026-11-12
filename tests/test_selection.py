"""The selection rule of a task (``wse.selection``): which checkpoint an experiment keeps when its run ends."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from wse.schemas import ObjectiveSpec, SelectionRule
from wse.selection import describe_rule, select_by_rule

RULE = SelectionRule(minimize="limit_hit_rate", preserve=["accuracy"])


def _report(checkpoint: str, accuracy: float, limit_hit_rate: float, tokens: float = 1000.0):
    return SimpleNamespace(checkpoint_id=checkpoint, report_id=f"report_{checkpoint}",
                           metrics={"accuracy": accuracy, "limit_hit_rate": limit_hit_rate,
                                    "mean_generated_tokens": tokens})


def test_the_best_qualifying_candidate_is_kept() -> None:
    reports = [_report("base", 0.25, 0.75), _report("a", 0.25, 0.5), _report("b", 0.125, 0.25),
               _report("c", 0.5, 0.625)]
    decision = select_by_rule(RULE, base_checkpoint_id="base", reports=reports)
    assert decision.selected_checkpoint_id == "a"  # b is lower but loses accuracy; c improves less than a
    verdicts = {c.checkpoint_id: c.qualifies for c in decision.candidates}
    assert verdicts == {"a": True, "b": False, "c": True}
    [why] = next(c for c in decision.candidates if c.checkpoint_id == "b").reasons
    assert "accuracy 0.125 is worse than the baseline's 0.25" in why
    assert decision.reason.startswith("a was kept: limit_hit_rate 0.75 -> 0.5")


def test_the_original_is_kept_when_nothing_qualifies_or_nothing_was_measured() -> None:
    reports = [_report("base", 0.25, 0.5), _report("a", 0.25, 0.5), _report("b", 0.0, 0.0)]
    decision = select_by_rule(RULE, base_checkpoint_id="base", reports=reports)
    assert decision.selected_checkpoint_id == "base" and not any(c.qualifies for c in decision.candidates)
    assert "does not improve" in decision.candidates[0].reasons[0]  # an equal value is no improvement
    unmeasured = select_by_rule(RULE, base_checkpoint_id="base", reports=[_report("a", 1.0, 0.0)])
    assert unmeasured.selected_checkpoint_id == "base" and unmeasured.baseline_report_id is None


def test_ties_go_to_preserved_metrics_then_fewer_changes_then_the_earlier_candidate() -> None:
    reports = [_report("base", 0.25, 0.75), _report("a", 0.25, 0.5), _report("b", 0.5, 0.5), _report("c", 0.5, 0.5)]
    assert select_by_rule(RULE, base_checkpoint_id="base", reports=reports).selected_checkpoint_id == "b"
    fewer = select_by_rule(RULE, base_checkpoint_id="base", reports=reports, changed_elements={"b": 90, "c": 10})
    assert fewer.selected_checkpoint_id == "c"
    earlier = select_by_rule(RULE, base_checkpoint_id="base", reports=reports, changed_elements={"b": 10, "c": 10})
    assert earlier.selected_checkpoint_id == "b"


def test_only_the_first_report_of_each_checkpoint_counts_and_maximize_works() -> None:
    reports = [_report("base", 0.25, 0.5), _report("a", 0.5, 0.5), _report("a", 0.0, 0.5)]
    rule = SelectionRule(maximize="accuracy", preserve=["limit_hit_rate"])
    decision = select_by_rule(rule, base_checkpoint_id="base", reports=reports)
    assert decision.selected_checkpoint_id == "a" and len(decision.candidates) == 1
    assert "highest accuracy" in describe_rule(rule) and "limit_hit_rate no worse" in describe_rule(rule)


@pytest.mark.parametrize("fields", [{}, {"minimize": "accuracy", "maximize": "accuracy"},
                                    {"minimize": "repetition_rate"},
                                    {"minimize": "limit_hit_rate", "preserve": ["limit_hit_rate"]},
                                    {"minimize": "limit_hit_rate", "preserve": ["accuracy", "accuracy"]}])
def test_invalid_rules_are_refused(fields) -> None:
    with pytest.raises(ValidationError):
        SelectionRule(**fields)


def test_an_objective_without_a_rule_keeps_its_serialized_form() -> None:
    plain = ObjectiveSpec(task_id="t", description="d", gates=[{"metric": "accuracy", "min_delta": 0.0}])
    assert "selection" not in plain.model_dump(mode="json")
    ruled = plain.model_copy(update={"selection": RULE})
    assert ruled.model_dump(mode="json")["selection"] == {"minimize": "limit_hit_rate", "preserve": ["accuracy"]}
