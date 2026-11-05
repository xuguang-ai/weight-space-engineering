"""The deterministic reference controller: ``scripted_search@1``.

A task-agnostic, stepwise search over single-unit edits. It knows nothing
about any particular task or model: it reads the experiment description and
the base model's manifest through the same operations an agent uses, builds
an ordered list of candidate edits, and tries them one at a time.

* **Candidates.** For every two-dimensional weight tensor, taken from the
  output end of the manifest towards the input end, and for every output unit
  (row) in ascending order, one ``scale_slice`` of that row by each of the
  configured ``factors`` (default ``[-1.0]``), optionally restricted to the
  configured ``tensors``. The order depends only on the manifest and the
  configuration.
* **bounded_eval.** Evaluate the base checkpoint once on the full
  development suite, then for each candidate: read the row it edits (evidence
  for the hypothesis), preview the patch, apply it, evaluate the candidate,
  and compare it with the base report. A candidate whose development accuracy
  exceeds the best so far is selected (compare-and-swap on the current
  selection). The search stops when a selected candidate meets the task's
  target, after ``patience`` consecutive evaluated candidates without
  improvement, when the next candidate's operations no longer fit the
  remaining budget, or when the candidates run out.
* **weights_only.** No development feedback exists, so the first candidate is
  previewed, committed, and selected, and the strategy stops: the policy
  permits a single candidate commit.

Every action uses only the run-bound, metered operations; every
idempotency key is derived from the step number, so a run is reproducible.
Hypotheses are brief and state the edit, never private reasoning.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from typing import Any

from wse.schemas import BudgetEntry, ObjectiveSpec
from wse.strategies.base import Action, Observation

__all__ = ["ScriptedSearchStrategy"]

MAX_JOB_POLLS = 30


class _Stop(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ScriptedSearchStrategy:
    """Deterministic single-unit edit search (see the module docstring)."""

    strategy_id = "scripted_search"
    version = "1"

    def __init__(self, factors: Sequence[float] = (-1.0,), patience: int = 3, max_candidates: int | None = None,
                 tensors: Sequence[str] | None = None, wait_seconds: float = 10.0) -> None:
        if not factors or not all(isinstance(f, (int, float)) and not isinstance(f, bool) for f in factors):
            raise ValueError("factors must be a nonempty list of numbers")
        if patience < 1:
            raise ValueError("patience must be at least 1")
        self.factors = [float(f) for f in factors]
        self.patience = int(patience)
        self.max_candidates = None if max_candidates is None else int(max_candidates)
        self.tensors = None if tensors is None else [str(name) for name in tensors]
        self.wait_seconds = float(wait_seconds)
        self._plan: Generator[Action, Observation, None] | None = None
        self._seen = 0
        self._budget: Mapping[str, BudgetEntry] = {}
        self._step = 0

    def config(self) -> dict[str, Any]:
        return {"factors": self.factors, "patience": self.patience, "max_candidates": self.max_candidates,
                "tensors": self.tensors, "wait_seconds": self.wait_seconds}

    # -- Strategy contract -------------------------------------------------------------

    def next_action(self, objective: ObjectiveSpec | None, observations: Sequence[Observation],
                    remaining_budget: Mapping[str, BudgetEntry]) -> Action:
        self._budget = remaining_budget
        try:
            if self._plan is None:
                if not observations:
                    return Action.stop("no experiment description was observed")
                self._seen = len(observations)
                self._plan = self._search(observations[0])
                return next(self._plan)
            new = list(observations[self._seen:])
            self._seen = len(observations)
            if len(new) != 1:
                return Action.stop("expected exactly one new observation per action")
            return self._plan.send(new[0])
        except _Stop as stop:
            return Action.stop(stop.reason)
        except StopIteration:
            return Action.stop("candidates exhausted")

    # -- helpers -----------------------------------------------------------------------

    def _key(self, label: str) -> str:
        self._step += 1
        return f"scripted-{self._step:04d}-{label}"

    def _remaining(self, counter: str) -> float:
        entry = self._budget.get(counter)
        if entry is None:
            return float("inf")
        return entry.remaining if isinstance(entry, BudgetEntry) else entry["remaining"]

    def _require(self, **amounts: float) -> None:
        for counter, amount in amounts.items():
            if self._remaining(counter) < amount:
                raise _Stop(f"insufficient budget: {counter} has {self._remaining(counter):g} left, the next "
                            f"step needs {amount:g}")

    @staticmethod
    def _ok(observation: Observation, what: str) -> Any:
        if not observation.ok:
            code = observation.error_code or "ERROR"
            raise _Stop(f"{what} failed with {code}")
        return observation.result

    def _evaluate(self, eid: str, checkpoint_id: str, suite_id: str, config: dict[str, int]
                  ) -> Generator[Action, Observation, dict[str, Any]]:
        self._require(evaluation_batches=1, evaluated_examples=config["sample_count"])
        job = self._ok((yield Action.call("wse_submit_evaluation", "Measure development accuracy.",
                                          experiment_id=eid, checkpoint_id=checkpoint_id, suite_id=suite_id,
                                          eval_config=config, idempotency_key=self._key("evaluate"))),
                       "evaluation submission")
        for _ in range(MAX_JOB_POLLS):
            if job["state"] not in ("queued", "running"):
                break
            job = self._ok((yield Action.call("wse_get_job", experiment_id=eid, job_id=job["job_id"],
                                              wait_seconds=self.wait_seconds)), "job poll")
        if job["state"] != "succeeded":
            raise _Stop(f"evaluation job {job['job_id']} ended {job['state']}")
        return job

    # -- the search ------------------------------------------------------------------------

    def _candidates(self, manifest: dict[str, Any]) -> list[tuple[str, int, int, float]]:
        matrices = [info for info in manifest["tensors"] if len(info["shape"]) == 2
                    and (self.tensors is None or info["name"] in self.tensors)]
        found = [(info["name"], row, info["shape"][1], factor)
                 for info in reversed(matrices) for row in range(info["shape"][0]) for factor in self.factors]
        return found if self.max_candidates is None else found[: self.max_candidates]

    def _search(self, first: Observation) -> Generator[Action, Observation, None]:
        description = self._ok(first, "reading the experiment")
        experiment = description["experiment"]
        eid, base = experiment["experiment_id"], experiment["base_checkpoint_id"]
        selected = experiment["selected_checkpoint_id"]
        inspection = self._ok((yield Action.call("wse_inspect_model", "Read the architecture.", experiment_id=eid,
                                                 checkpoint_id=base)), "inspection")
        candidates = self._candidates(inspection["manifest"])
        if not candidates:
            raise _Stop("the model has no two-dimensional weight tensors to edit")

        bounded = experiment["mode"] == "bounded_eval" and description["evaluation_available"]
        best: float | None = None
        base_report_id: str | None = None
        task = description.get("task") or {}
        suite_id = task.get("development_suite_id") or (description["suites"][0]["suite_id"]
                                                        if description["suites"] else None)
        config: dict[str, int] = {}
        if bounded:
            suite = next((s for s in description["suites"] if s["suite_id"] == suite_id), None)
            if suite is None:
                raise _Stop("no development suite is available")
            config = {"sample_count": max(suite["sample_counts"]), "seed": suite["seeds"][0]}
            job = yield from self._evaluate(eid, base, suite_id, config)
            best, base_report_id = job["result"]["metrics"].get("accuracy"), job["report_id"]
            if job["result"].get("target_met"):
                raise _Stop("the selected checkpoint already meets the target")

        misses = 0
        for tensor, row, width, factor in candidates:
            selection = [{"start": row, "stop": row + 1, "step": 1}, {"start": 0, "stop": width, "step": 1}]
            needs = {"patch_previews": 1, "candidate_commits": 1, "tensor_read_calls": 1}
            if bounded:
                needs.update(evaluation_batches=1, evaluated_examples=config["sample_count"])
            self._require(**needs)
            self._ok((yield Action.call("wse_read_tensor", "Read the unit the next edit changes.", experiment_id=eid,
                                        checkpoint_id=base, tensor=tensor, selection=selection)), "read")
            hypothesis = (f"Scaling output unit {row} of {tensor} by {factor:g} may move the model toward the "
                          "objective.")
            preview = self._ok((yield Action.call(
                "wse_preview_patch", hypothesis, experiment_id=eid, idempotency_key=self._key("preview"),
                patch_spec={"base_checkpoint_id": base, "hypothesis": hypothesis,
                            "expected_behavior": "Development accuracy rises.",
                            "development_suite_id": suite_id,
                            "operations": [{"op": "scale_slice", "tensor": tensor, "selection": selection,
                                            "factor": factor}]})), "preview")
            if preview["changed_elements"] == 0:
                continue
            candidate = self._ok((yield Action.call("wse_apply_patch", "Commit the previewed candidate.",
                                                    experiment_id=eid, patch_id=preview["patch_id"],
                                                    idempotency_key=self._key("apply"))), "apply")
            checkpoint_id = candidate["checkpoint_id"]
            if not bounded:
                self._ok((yield Action.call("wse_select_checkpoint", "Select the single permitted candidate.",
                                            experiment_id=eid, checkpoint_id=checkpoint_id,
                                            expected_selected_id=selected, idempotency_key=self._key("select"))),
                         "selection")
                raise _Stop("the single permitted candidate is committed and selected")
            job = yield from self._evaluate(eid, checkpoint_id, suite_id, config)
            accuracy = job["result"]["metrics"].get("accuracy")
            self._ok((yield Action.call("wse_compare", "Compare the candidate with the base on the same examples.",
                                        experiment_id=eid, base_report_id=base_report_id,
                                        candidate_report_id=job["report_id"])), "comparison")
            if accuracy is not None and (best is None or accuracy > best):
                best, misses = accuracy, 0
                result = self._ok((yield Action.call(
                    "wse_select_checkpoint", "Select the best candidate so far.", experiment_id=eid,
                    checkpoint_id=checkpoint_id, expected_selected_id=selected,
                    idempotency_key=self._key("select"))), "selection")
                selected = result["selected_checkpoint_id"]
                if job["result"].get("target_met"):
                    raise _Stop("the selected candidate meets the target on the full development suite")
            else:
                misses += 1
                if misses >= self.patience:
                    raise _Stop(f"{misses} consecutive evaluated candidates showed no improvement")
