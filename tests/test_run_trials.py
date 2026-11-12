"""The trial script keeps every trial's report when a trial fails or the operator interrupts (Task 7).

``examples/signed_sum/run_trials.py`` runs on a fresh signed-sum workspace
with the weights-only control policy (no model, no network, no worker); the
runner is wrapped to fail or to be interrupted on chosen trials.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from wse.policy import WEIGHTS_ONLY_POLICY_ID
from wse.runner import ReferenceRunner
from wse.schemas import ErrorCode, WSEError

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "signed_sum" / "run_trials.py"


@pytest.fixture
def run_trials():
    spec = importlib.util.spec_from_file_location("signed_sum_run_trials", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _reports(out: Path, label: str) -> tuple[list[dict], str]:
    rows = json.loads((out / "reports" / f"{label}.json").read_text(encoding="utf-8"))
    return rows, (out / "reports" / f"{label}.md").read_text(encoding="utf-8")


def test_failing_trials_are_reported_and_later_trials_still_run(tmp_path, monkeypatch, run_trials):
    original_run, original_finalize = ReferenceRunner.run_experiment, ReferenceRunner._finalize
    started: list[str] = []

    def run_experiment(self, run_id, *args, **kwargs):
        started.append(run_id)
        if len(started) == 3:
            raise RuntimeError("the runner could not start")  # before the trial's experiment exists
        return original_run(self, run_id, *args, **kwargs)

    def finalize(self, engine, eid):
        if len(started) == 2:  # the controller's finalization cannot freeze, so the experiment stays open
            raise WSEError(ErrorCode.WORKER_FAILED, "1 jobs did not settle within 120 s; the experiment stays open")
        return original_finalize(self, engine, eid)

    monkeypatch.setattr(ReferenceRunner, "run_experiment", run_experiment)
    monkeypatch.setattr(ReferenceRunner, "_finalize", finalize)
    code = run_trials([str(tmp_path), "--label", "batch", "--policy", WEIGHTS_ONLY_POLICY_ID, "--trials", "4"])
    assert code == 1 and len(started) == 4
    rows, markdown = _reports(tmp_path, "batch")
    assert [row["trial"] for row in rows] == [1, 2, 3, 4]
    assert [row["state"] for row in rows] == ["CLOSED", "OPEN", None, "CLOSED"]
    for row in (rows[0], rows[3]):
        assert row["error"] is None and row["final"] is not None and row["replay_identical"] is True
    assert "WORKER_FAILED: 1 jobs did not settle" in rows[1]["error"] and rows[1]["experiment_id"]
    assert rows[2]["error"] == "RuntimeError: the runner could not start" and rows[2]["experiment_id"] is None
    for row in (rows[1], rows[2]):  # a failed trial counts as unsuccessful, with its cause
        assert row["success"] is False and row["failure_cause"] and row["replay_identical"] is None
        assert row["error"] in markdown
    assert "/4 trials found a candidate" in markdown and "not yet final-evaluated" not in markdown


def test_an_interrupt_starts_no_further_trials_and_still_reports_the_frozen_ones(tmp_path, monkeypatch, run_trials):
    original_run = ReferenceRunner.run_experiment
    started: list[str] = []

    def run_experiment(self, run_id, *args, **kwargs):
        started.append(run_id)
        experiment_id = original_run(self, run_id, *args, **kwargs)
        if len(started) == 2:
            raise KeyboardInterrupt  # as the runner re-raises Ctrl-C after settling and freezing the trial
        return experiment_id

    monkeypatch.setattr(ReferenceRunner, "run_experiment", run_experiment)
    try:
        code = run_trials([str(tmp_path), "--label", "batch", "--policy", WEIGHTS_ONLY_POLICY_ID, "--trials", "3"])
    except KeyboardInterrupt:
        pytest.fail("the interrupt escaped the trial script instead of ending it with exit code 130")
    assert code == 130 and len(started) == 2  # the third trial never started
    rows, markdown = _reports(tmp_path, "batch")
    assert [row["trial"] for row in rows] == [1, 2]
    assert all(row["state"] == "CLOSED" and row["final"] is not None for row in rows)
    assert rows[0]["error"] is None and rows[1]["error"].startswith("KeyboardInterrupt")
    assert "Interrupted by the operator: 1 of 3 trials not started." in markdown
