"""The vendor-neutral ``AgentAdapter`` contract (Section 1.3 of the plan).

An agent adapter runs one SDK's complete conversation/tool loop against the
WSE MCP server of one run grant, reports normalized events and usage, and
propagates cancellation. It owns no weight versions, evaluation results, or
budgets: it never selects a checkpoint, freezes, or closes an experiment.

The reference runner (``wse.runner``) drives an adapter through its
lifecycle, so the MCP connection and server outlive the SDK loop until the
runner has settled pending jobs and frozen the experiment:

1. ``await adapter.start(connection, agent_config)`` starts the stdio server
   bound to the run grant and connects to it. A failure is remembered and
   reported by ``run`` as ``connection_failed``.
2. ``await adapter.run(experiment_id, agent_config, event_sink)`` runs the
   SDK loop to its end and returns an ``AgentRunResult``.
   ``adapter.cancel(experiment_id, reason)`` (thread-safe) stops a running
   loop, or a loop that has not started yet; the result then reports
   ``stopped`` with that reason.
3. The runner settles jobs and freezes, then ``await adapter.close(...)``.

This module imports no agent SDK; ``wse.agent_adapters.openai_agents``
imports OpenAI Agents SDK lazily, inside its methods.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any, Literal

from pydantic import Field, model_validator

from wse.schemas import _Frozen

__all__ = [
    "AgentAdapter",
    "AgentConfig",
    "AgentEvent",
    "AgentRunResult",
    "AgentUsage",
    "EventSink",
    "RunConnection",
    "TerminationStatus",
    "describe_exception",
]

DEFAULT_MAX_TURNS = 100
DEFAULT_WALL_CLOCK_SECONDS = 900.0

TerminationStatus = Literal["completed", "stopped", "max_turns", "timeout", "cancelled", "connection_failed",
                            "error"]


def describe_exception(exc: BaseException, limit: int = 2000) -> str:
    """``Type: message`` of an exception, or of the innermost exceptions of an exception group.

    SDK and transport failures usually arrive wrapped in task-group
    exception groups whose own message ("unhandled errors in a TaskGroup")
    names no cause, so the leaves are reported instead.
    """
    leaves: list[str] = []

    def walk(error: BaseException) -> None:
        if isinstance(error, BaseExceptionGroup) and error.exceptions:
            for inner in error.exceptions:
                walk(inner)
        else:
            leaves.append(f"{type(error).__name__}: {error}")

    walk(exc)
    return " | ".join(dict.fromkeys(leaves))[:limit]


class AgentConfig(_Frozen):
    """Operator configuration of an SDK-backed controller.

    ``model`` is the operator-chosen model name (credentials come from the
    SDK's normal environment configuration and are never recorded).
    ``max_turns`` and ``wall_clock_seconds`` are the pilot controller limits
    (100 SDK turns, 900 s), independent of WSE's computation budget.
    A ``benchmark`` configuration must disable SDK tracing, because the SDK
    exports traces, including tool inputs and outputs, by default.
    """

    kind: Literal["agent"] = "agent"
    adapter: str = Field(default="openai_agents@1", pattern=r"^[a-z][a-z0-9_]{0,63}@[0-9.]{1,20}$")
    model: str = Field(min_length=1, max_length=200)
    max_turns: int = Field(default=DEFAULT_MAX_TURNS, ge=1, le=10_000)
    wall_clock_seconds: float = Field(default=DEFAULT_WALL_CLOCK_SECONDS, gt=0, allow_inf_nan=False)
    tracing_disabled: bool = True
    benchmark: bool = True
    model_settings: dict[str, Any] = Field(default_factory=dict)
    client_session_timeout_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    connect_timeout_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _benchmark_disables_tracing(self) -> AgentConfig:
        if self.benchmark and not self.tracing_disabled:
            raise ValueError("benchmark runs must disable SDK tracing (tracing_disabled=true)")
        return self


class RunConnection(_Frozen):
    """Everything an adapter needs to reach the run's MCP server, supplied by the runner.

    ``serve_command`` starts ``wse serve`` bound to ``run_id`` on the same
    workspace; ``serve_env`` is the complete (allow-listed) environment of
    that process; ``allowed_tools`` is the client-side allow-list of WSE
    tools; ``instructions`` and ``initial_message`` are generated from the
    task definition and the policy only.
    """

    workspace: str
    run_id: str
    experiment_id: str
    task_id: str
    policy_id: str
    serve_command: list[str]
    serve_env: dict[str, str]
    cwd: str
    allowed_tools: list[str]
    instructions: str
    initial_message: str
    constraint_status: Literal["verified", "unverified"]


class AgentUsage(_Frozen):
    """Normalized controller usage; a field the host cannot report is ``None`` with a reason in ``unavailable``."""

    requests: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None  # part of input_tokens, when the host reports it
    reasoning_output_tokens: int | None = None  # part of output_tokens, when the host reports it
    cost_usd: float | None = None
    provider: str | None = None
    model: str | None = None
    model_version: str | None = None
    unavailable: dict[str, str] = Field(default_factory=dict)


class AgentEvent(_Frozen):
    """One normalized controller event.

    ``kind``: ``run_started``, ``model_response`` (usage, output item counts,
    latency, and the provider's response and request IDs; private reasoning
    content is never stored), ``tool_call``,
    ``tool_result`` (``ok``, ``error_code``, a bounded ``output`` excerpt with
    its full size and hash), ``message``, ``cancel_requested``, and
    ``run_finished``.
    """

    seq: int
    at: str
    elapsed_seconds: float
    kind: Literal["run_started", "model_response", "tool_call", "tool_result", "message", "cancel_requested",
                  "run_finished"]
    tool: str | None = None
    call_id: str | None = None
    arguments: Any = None
    ok: bool | None = None
    error_code: str | None = None
    summary: str | None = None
    output: str | None = None
    output_bytes: int | None = None
    output_hash: str | None = None
    usage: dict[str, Any] | None = None
    details: dict[str, Any] = Field(default_factory=dict)


EventSink = Callable[[AgentEvent], None]


class AgentRunResult(_Frozen):
    """How an SDK loop ended. It selects no checkpoint and closes nothing."""

    status: TerminationStatus
    reason: str
    final_message: str | None = None
    conversation_ref: str | None = None
    turns: int = 0
    tool_calls: int = 0
    elapsed_seconds: float = 0.0
    usage: AgentUsage = Field(default_factory=AgentUsage)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class AgentAdapter(abc.ABC):
    """Runs one SDK's tool loop for one experiment at a time (see the module docstring)."""

    adapter_id: str
    version: str

    @abc.abstractmethod
    async def start(self, connection: RunConnection, agent_config: AgentConfig) -> None:
        """Start and connect the run's MCP server; remember a failure for ``run`` to report."""

    @abc.abstractmethod
    async def run(self, experiment_id: str, agent_config: AgentConfig, event_sink: EventSink) -> AgentRunResult:
        """Run the SDK loop to its end and report how it ended."""

    @abc.abstractmethod
    def cancel(self, experiment_id: str, reason: str = "cancelled") -> None:
        """Stop a running loop (thread-safe); ``run`` then returns ``stopped`` with ``reason``."""

    @abc.abstractmethod
    async def close(self, experiment_id: str) -> None:
        """Disconnect and stop the MCP server (after the runner has frozen the experiment)."""
