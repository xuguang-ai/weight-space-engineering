"""The stepwise controller contract (``Strategy``).

A strategy proposes one registered operation at a time, or stops, from the
objective, its permitted observations, and the remaining budget:
``Strategy.next_action(objective, observations, remaining_budget) -> Action``.
It never executes anything itself. The reference runner executes each action
through a run-bound Engine (the same experiment service, policy, budget, and
audit records as the MCP tools) and appends the outcome as an
``Observation``. A strategy therefore cannot alter the evaluator, its run
grant, or its budget, and it sees exactly what an agent sees through the WSE
tools: actions are named after the MCP tools and take the same arguments,
and an observation carries the same ``result`` object a tool returns (or the
same domain error body).

Controllers that run a complete SDK tool loop use the separate
``AgentAdapter.run`` contract (``wse.agent_adapters.base``); a strategy is
never nested inside an SDK loop by the runner.

Strategies are registered extensions (kind ``"strategy"``): the registered
implementation is a factory, usually the strategy class, that accepts the
configured ``parameters`` as keyword arguments and returns a fresh instance
for one experiment. It must declare ``strategy_id`` and ``version``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import Field

from wse.schemas import BudgetEntry, ErrorCode, ObjectiveSpec, WSEError, _Frozen

__all__ = ["STRATEGY_TOOLS", "Action", "Observation", "Strategy", "validate_strategy"]

# The operations a strategy may request: the agent-facing MCP tool names, minus
# experiment creation (the runner persists the experiment before any controller
# starts). The runner rejects any other name, and exposes only those that the
# experiment's policy permits (``wse.runner.allowed_tools``): ``wse_probe``
# (activation probes of language models, M2) only under a policy that grants
# probes, and ``wse_get_report`` (a whole report) with the other evaluation
# tools, whenever development evaluations are permitted.
STRATEGY_TOOLS: tuple[str, ...] = (
    "wse_list_models",
    "wse_get_experiment",
    "wse_inspect_model",
    "wse_read_tensor",
    "wse_analyze",
    "wse_probe",
    "wse_preview_patch",
    "wse_apply_patch",
    "wse_submit_evaluation",
    "wse_get_job",
    "wse_get_report",
    "wse_cancel_job",
    "wse_compare",
    "wse_select_checkpoint",
)


class Action(_Frozen):
    """One proposed step: call a WSE operation, or stop.

    ``rationale`` is a brief, observable justification (never private
    reasoning); ``reason`` names why a ``stop`` was chosen.
    """

    kind: Literal["call", "stop"]
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(default="", max_length=500)
    reason: str | None = Field(default=None, max_length=200)

    @classmethod
    def call(cls, tool: str, rationale: str = "", **arguments: Any) -> Action:
        return cls(kind="call", tool=tool, arguments=arguments, rationale=rationale)

    @classmethod
    def stop(cls, reason: str) -> Action:
        return cls(kind="stop", reason=reason)


class Observation(_Frozen):
    """The outcome of one executed action.

    ``result`` is the domain result as JSON (the ``result`` field of the
    matching MCP tool response); ``error`` is the domain error body
    (``code, message, retryable, details``) when the operation failed.
    """

    tool: str
    arguments: dict[str, Any]
    ok: bool
    result: Any = None
    error: dict[str, Any] | None = None

    @property
    def error_code(self) -> str | None:
        return None if self.error is None else self.error.get("code")


@runtime_checkable
class Strategy(Protocol):
    """A stepwise controller (Section 1.1 of the plan)."""

    strategy_id: str
    version: str

    def config(self) -> dict[str, Any]:
        """The strategy's configuration as plain JSON, recorded with the experiment."""
        ...

    def next_action(self, objective: ObjectiveSpec | None, observations: Sequence[Observation],
                    remaining_budget: Mapping[str, BudgetEntry]) -> Action:
        """Propose the next operation, or stop."""
        ...


def validate_strategy(extension_id: str, version: str, implementation: Any) -> None:
    """Registration check: a callable factory declaring this exact identity and ``next_action``."""
    declared = (getattr(implementation, "strategy_id", None), getattr(implementation, "version", None))
    if declared != (extension_id, version):
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"strategy declares {declared[0]}@{declared[1]} but is being registered as {extension_id}@{version}",
        )
    missing = [name for name in ("next_action", "config") if not callable(getattr(implementation, name, None))]
    if missing or not callable(implementation):
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"strategy {extension_id}@{version} must be a factory (for example its class) whose instances "
            f"implement next_action and config; missing: {missing or ['__call__']}",
            details={"missing_methods": missing},
        )
