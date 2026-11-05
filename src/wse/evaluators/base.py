"""The evaluator contract the worker runs.

An evaluator receives a *metered forward* callable instead of the model: the
worker owns the model, counts every forward call and example, refuses calls
beyond the job's reservation or in modes without inference, and stops the job
on cancellation or timeout. Evaluators never load weights or call an adapter
themselves. ``forward(inputs)`` runs one batch of at most ``batch_size`` rows
and returns logits; ``forward.check()`` raises if the job must stop, so long
evaluators can yield between steps.

``evaluate`` returns a JSON summary ``{"metrics": {...}, "sample_count": n,
"errors": [...]}``. A sample count of ``n`` evaluates the suite's first ``n``
examples; the seed is recorded with the configuration.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch

from wse.schemas import ErrorCode, EvalConfig, EvaluationSuite, WSEError


class MeteredForward(Protocol):
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor: ...

    def check(self) -> None: ...


@runtime_checkable
class Evaluator(Protocol):
    evaluator_id: str
    version: str

    def evaluate(
        self, forward: MeteredForward, suite: EvaluationSuite, config: EvalConfig, *, batch_size: int
    ) -> dict[str, Any]: ...


def validate_evaluator(extension_id: str, version: str, implementation: Any) -> None:
    """Registry check: identity matches and ``evaluate`` is implemented."""
    declared = (getattr(implementation, "evaluator_id", None), getattr(implementation, "version", None))
    if declared != (extension_id, version):
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"evaluator declares {declared[0]}@{declared[1]} but is being registered as {extension_id}@{version}",
        )
    if not callable(getattr(implementation, "evaluate", None)):
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"evaluator {extension_id}@{version} does not implement evaluate")
