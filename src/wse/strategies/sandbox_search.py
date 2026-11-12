"""A scripted strategy that analyzes weights in the sandbox and edits with what the sandbox computed (M3 Task 8).

``sandbox_search@1`` checks code execution end to end without a model; it
is a toolchain check, not an agent, and its rule is deliberately narrow. Its
code runs in the sandbox (``wse_run_python``), reads the base checkpoint's
permitted tensors from ``inputs/weights/base.safetensors``, and lists every
pair of identical rows of a two-dimensional tensor: units that a layer
cannot tell apart. For each pair the code computes the first row with every
sign reversed, prints it, and saves it to ``outputs/`` (where it becomes a
stored vector); the strategy passes those values inline to a ``set_slice``
of the second row (``set_slice`` takes values, not stored vectors),
evaluates the candidate on the task's development suite, compares it with
the base, selects it when it improves accuracy, and stops once the selected
candidate meets the target or the pairs run out.

It reads nothing but the experiment's tools and the sandbox's inputs. Every
run of it is unverified, as every run of a policy that grants the sandbox.
"""

from __future__ import annotations

import json
from collections.abc import Generator, Mapping, Sequence
from typing import Any

from wse.schemas import BudgetEntry, ObjectiveSpec
from wse.strategies.base import Action, Observation

__all__ = ["SandboxSearchStrategy"]

MAX_JOB_POLLS = 100

# The code the strategy runs in the sandbox: each pair of identical rows, and the first row with its signs reversed.
ANALYSIS = """\
import json

import numpy as np
from safetensors.torch import load_file

weights = load_file("inputs/weights/base.safetensors")
pairs = []
for name in sorted(weights):
    matrix = weights[name].float().numpy()
    if matrix.ndim != 2:
        continue
    for first in range(matrix.shape[0]):
        for second in range(first + 1, matrix.shape[0]):
            if np.array_equal(matrix[first], matrix[second]):
                values = -matrix[first]
                np.save(f"outputs/values_{len(pairs)}.npy", values)
                pairs.append({"tensor": name, "first": first, "row": second, "columns": int(matrix.shape[1]),
                              "values": values.tolist()})
print(json.dumps(pairs))
"""


class _Stop(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class SandboxSearchStrategy:
    """Identical rows found in the sandbox; each second row gets the first row's values with signs reversed."""

    strategy_id = "sandbox_search"
    version = "1"

    def __init__(self, wait_seconds: float = 10.0) -> None:
        self.wait_seconds = float(wait_seconds)
        self._plan: Generator[Action, Observation, None] | None = None
        self._seen = 0
        self._step = 0

    def config(self) -> dict[str, Any]:
        return {"wait_seconds": self.wait_seconds, "analysis_lines": len(ANALYSIS.splitlines())}

    def next_action(self, objective: ObjectiveSpec | None, observations: Sequence[Observation],
                    remaining_budget: Mapping[str, BudgetEntry]) -> Action:
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
            return Action.stop("the sandbox found no further pair of identical rows")

    def _key(self, label: str) -> str:
        self._step += 1
        return f"sandbox-{self._step:04d}-{label}"

    @staticmethod
    def _ok(observation: Observation, what: str) -> Any:
        if not observation.ok:
            raise _Stop(f"{what} failed with {observation.error_code or 'ERROR'}")
        return observation.result

    def _evaluate(self, eid: str, checkpoint_id: str, suite_id: str, config: dict[str, int]
                  ) -> Generator[Action, Observation, dict[str, Any]]:
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

    def _search(self, first: Observation) -> Generator[Action, Observation, None]:
        description = self._ok(first, "reading the experiment")
        experiment = description["experiment"]
        eid, base = experiment["experiment_id"], experiment["base_checkpoint_id"]
        selected = experiment["selected_checkpoint_id"]
        if not description["evaluation_available"]:
            raise _Stop("this strategy needs development evaluations")
        suite_id = (description.get("task") or {}).get("development_suite_id")
        suite = next((s for s in description["suites"] if s["suite_id"] == suite_id), None)
        if suite is None:
            raise _Stop("no development suite is available")
        config = {"sample_count": max(suite["sample_counts"]), "seed": suite["seeds"][0]}

        ran = self._ok((yield Action.call("wse_run_python", "Look for identical rows in the weight matrices.",
                                          experiment_id=eid, code=ANALYSIS, idempotency_key=self._key("analyze"))),
                       "the sandbox analysis")
        if ran["ending"] != "completed" or ran["stdout_truncated"]:
            raise _Stop(f"the sandbox analysis ended {ran['ending']}")
        pairs = json.loads(ran["stdout"])
        if not pairs:
            raise _Stop("the sandbox found no pair of identical rows")

        base_job = yield from self._evaluate(eid, base, suite_id, config)
        best, base_report = base_job["result"]["metrics"].get("accuracy"), base_job["report_id"]
        for pair in pairs:
            selection = [{"start": pair["row"], "stop": pair["row"] + 1, "step": 1},
                         {"start": 0, "stop": pair["columns"], "step": 1}]
            hypothesis = (f"Rows {pair['first']} and {pair['row']} of {pair['tensor']} are identical, so the layer "
                          f"cannot tell those units apart; giving row {pair['row']} the opposite signs of row "
                          f"{pair['first']} may separate them.")
            preview = self._ok((yield Action.call(
                "wse_preview_patch", hypothesis, experiment_id=eid, idempotency_key=self._key("preview"),
                patch_spec={"base_checkpoint_id": base, "hypothesis": hypothesis,
                            "expected_behavior": "Development accuracy rises.", "development_suite_id": suite_id,
                            "operations": [{"op": "set_slice", "tensor": pair["tensor"], "selection": selection,
                                            "values": [pair["values"]]}]})), "preview")
            candidate = self._ok((yield Action.call("wse_apply_patch", "Commit the previewed candidate.",
                                                    experiment_id=eid, patch_id=preview["patch_id"],
                                                    idempotency_key=self._key("apply"))), "commit")
            job = yield from self._evaluate(eid, candidate["checkpoint_id"], suite_id, config)
            accuracy = job["result"]["metrics"].get("accuracy")
            self._ok((yield Action.call("wse_compare", "Compare the candidate with the base on the same examples.",
                                        experiment_id=eid, base_report_id=base_report,
                                        candidate_report_id=job["report_id"])), "comparison")
            if accuracy is not None and (best is None or accuracy > best):
                best = accuracy
                result = self._ok((yield Action.call(
                    "wse_select_checkpoint", "Select the better candidate.", experiment_id=eid,
                    checkpoint_id=candidate["checkpoint_id"], expected_selected_id=selected,
                    idempotency_key=self._key("select"))), "selection")
                selected = result["selected_checkpoint_id"]
                if job["result"].get("target_met"):
                    raise _Stop("the selected candidate meets the target on the full development suite")
