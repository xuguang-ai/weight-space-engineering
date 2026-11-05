"""The reference runner: controller selection, records, and experiment finalization.

``run_experiment(run_id, task_id, policy_id, agent_config) -> experiment_id``
drives one trial on an operator-prepared workspace:

1. **Persist first.** The run's single experiment is created through an
   Engine bound to the run grant before any controller starts, and the
   controller (strategy or agent adapter, exact ID and version,
   configuration, constraint status, tracing setting) is recorded with it.
2. **Run one controller, without nesting tool loops.**

   * A stepwise ``Strategy`` (``StrategyConfig``) receives the experiment
     description as its first observation (read task and budget), then
     proposes one action at a time; the runner executes each through the
     run-bound Engine, the same service the MCP tools call, and records it.
     The runner owns the evaluation worker for these runs.
   * An SDK-backed agent (``AgentConfig``, the OpenAI Agents SDK adapter in
     M1) runs its own complete tool loop against ``wse serve`` bound to the
     run grant. That server process is the sole job-scheduler owner; the
     runner starts no worker. The runner only listens to the adapter's
     normalized events, persists them, and checks stop conditions.

3. **Stop conditions** are read from authoritative WSE state after every
   operation: a budget error (insufficient budget for the next operation),
   the selected checkpoint's full development evaluation meeting the task's
   target, 3 consecutive evaluated candidates without improvement, or 3
   consecutive policy denials (bounded_eval); the single permitted candidate
   committed and selected (weights_only, which uses no development-score
   threshold); and the controller stopping voluntarily. A met condition
   cancels the controller.
4. **Finalize after the controller exits.** The runner cancels and settles
   every outstanding job (recovering jobs whose worker was lost), then
   freezes the selected checkpoint, while an SDK adapter's MCP server stays
   connected; only then is the adapter closed. Neither the SDK nor the runner
   closes an experiment: final evaluation (``wse final-evaluate``) is a
   separate operator step.

Agent instructions are generated from the registered task definition
(``task.json``) and the experiment policy only, never from repository
documentation; the agent receives no tool besides the allow-listed WSE MCP
tools, and the server process starts in the workspace with an allow-listed
environment, so the agent cannot reach source code, the plan, fixture
builders, or credentials.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError

from wse.agent_adapters.base import (
    AgentAdapter,
    AgentConfig,
    AgentEvent,
    AgentRunResult,
    AgentUsage,
    RunConnection,
)
from wse.api import Engine
from wse.audit import (
    append_controller_event,
    controller_record,
    environment_record,
    record_controller_end,
    record_controller_start,
)
from wse.policy import ExperimentPolicy
from wse.registry import Registry, builtin_registry
from wse.schemas import (
    BudgetEntry,
    ErrorCode,
    ExtensionRef,
    TaskDefinition,
    WSEError,
    _Frozen,
)
from wse.strategies.base import STRATEGY_TOOLS, Action, Observation

__all__ = [
    "EVALUATION_TOOLS",
    "TEST_DOUBLE_PREFIX",
    "ExperimentExecutor",
    "ReferenceRunner",
    "StopMonitor",
    "StrategyConfig",
    "allowed_tools",
    "initial_message",
    "parse_controller_config",
    "render_instructions",
    "run_experiment",
    "serve_environment",
]

logger = logging.getLogger("wse.runner")

EVALUATION_TOOLS = ("wse_submit_evaluation", "wse_get_job", "wse_cancel_job", "wse_compare")
NO_IMPROVEMENT_PATIENCE = 3
POLICY_DENIAL_LIMIT = 3
TEST_DOUBLE_PREFIX = "wse-test-double:"
SERVE_ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "PYTHONPATH", "VIRTUAL_ENV",
                       "SYSTEMROOT")
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
AGENT_ADAPTERS = {"openai_agents@1": ("wse.agent_adapters.openai_agents", "OpenAIAgentsAdapter")}


class StrategyConfig(_Frozen):
    """Operator configuration of a stepwise controller: an exact registered strategy and its parameters."""

    kind: Literal["strategy"] = "strategy"
    strategy: str = "scripted_search@1"
    parameters: dict[str, Any] = Field(default_factory=dict)
    max_steps: int = Field(default=500, ge=1)
    wall_clock_seconds: float = Field(default=900.0, gt=0, allow_inf_nan=False)


def parse_controller_config(value: Any) -> StrategyConfig | AgentConfig:
    """A ``StrategyConfig`` or ``AgentConfig`` from an object or mapping (``kind`` selects; ``model`` implies agent)."""
    if isinstance(value, (StrategyConfig, AgentConfig)):
        return value
    if not isinstance(value, Mapping):
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "agent_config must be a StrategyConfig, an AgentConfig, or a "
                       "mapping with kind 'strategy' or 'agent'")
    kind = value.get("kind", "agent" if "model" in value else "strategy")
    try:
        return AgentConfig(**value) if kind == "agent" else StrategyConfig(**value)
    except ValidationError as exc:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"invalid controller configuration: {exc.errors()[0]['msg']}",
                       details={"errors": [error["msg"] for error in exc.errors()]}) from exc


# ---------------------------------------------------------------------------
# Agent-facing instructions (generated from task.json and the policy only)
# ---------------------------------------------------------------------------


def allowed_tools(policy: ExperimentPolicy) -> list[str]:
    """The WSE tools a controller of this policy may use; evaluation tools only when evaluation is permitted."""
    evaluation = policy.mode == "bounded_eval" and policy.budget.evaluation_batches > 0
    return [tool for tool in STRATEGY_TOOLS if evaluation or tool not in EVALUATION_TOOLS]


def _gate_text(gate: Any) -> str:
    if gate.min_value is not None:
        return f"{gate.metric} at least {gate.min_value:g}"
    if gate.min_delta is not None:
        return f"{gate.metric} improves on the starting model by at least {gate.min_delta:g}"
    return f"{gate.metric} drops by at most {gate.max_drop:g} from the starting model"


def render_instructions(task: TaskDefinition, policy: ExperimentPolicy, *, experiment_id: str,
                        tools: Sequence[str]) -> str:
    """Agent instructions built only from the task definition, the policy, and the run's IDs and tools."""
    budget, operations = policy.budget, policy.operations
    evaluation = policy.mode == "bounded_eval" and budget.evaluation_batches > 0
    lines = [
        "You control one managed weight-space engineering experiment through the WSE tools listed below. They are "
        "your only tools: you cannot run code, read files, or use the network.",
        "",
        f"Experiment ID: {experiment_id}",
        f"Task {task.task_id}: {task.title}",
        task.description,
    ]
    if task.input_description:
        lines.append(f"Model input: {task.input_description}")
    if task.output_description:
        lines.append(f"Model output: {task.output_description}")
    lines.append(f"Objective: {task.objective.description}")
    lines.append("Acceptance: " + "; ".join(_gate_text(gate) for gate in task.objective.gates) + ".")
    if task.target_accuracy is not None:
        lines.append(f"The target is met only by a successful evaluation of development suite "
                     f"{task.development_suite_id} covering all {task.development_examples} examples with accuracy "
                     f"at least {task.target_accuracy:g}.")
    lines += ["", f"Policy {policy.policy_id} (mode {policy.mode}); budget for the whole experiment:",
              f"- candidate commits: {budget.candidate_commits}",
              f"- patch previews: {budget.patch_previews}",
              f"- tensor analyses: {budget.analysis_calls}",
              f"- tensor reads: {budget.tensor_read_calls}"]
    if evaluation:
        lines += [f"- development evaluation batches: {budget.evaluation_batches}, at most "
                  f"{budget.examples_per_batch} examples each and {budget.evaluated_examples} examples in total"]
    lines += [f"- computation time: {budget.compute_seconds:g} s in total, at most {budget.seconds_per_operation:g} s "
              "per operation",
              "Consumed allowance is never refunded; every result reports usage and the remaining budget.",
              f"Analyses: {', '.join(sorted(operations.analyses))}. Edit operations: "
              f"{', '.join(sorted(operations.edits))}. A read returns at most "
              f"{operations.limits.max_read_elements} elements; a patch has at most "
              f"{operations.limits.max_patch_operations} operations.",
              "",
              "How to work:",
              "1. Call wse_get_experiment with the experiment ID for the task, budget, permitted operations, "
              "development suites, and checkpoints.",
              "2. Gather evidence with wse_inspect_model, wse_read_tensor, and wse_analyze on the experiment's "
              "checkpoints.",
              "3. Propose an edit with wse_preview_patch (base_checkpoint_id, a hypothesis of one or two sentences "
              "stating the observable evidence, and the operations), then create a candidate with wse_apply_patch "
              "by patch_id."]
    if evaluation:
        lines.append("4. Evaluate checkpoints with wse_submit_evaluation on a development suite with a preregistered "
                     "eval_config (sample_count, seed), wait with wse_get_job (wait_seconds up to 10), and compare "
                     "two reports with wse_compare.")
    else:
        lines.append("4. Development evaluation is not available in this mode: no forward pass of the model is "
                     f"permitted, and the policy allows {budget.candidate_commits} candidate commit(s).")
    lines += ["5. Select a checkpoint with wse_select_checkpoint, passing the currently selected ID as "
              "expected_selected_id; selecting an earlier checkpoint rolls back.",
              "6. When you are done, reply with a short final message and no tool call. The selected checkpoint is "
              "then frozen for a separate final evaluation that you cannot see.",
              ""]
    if evaluation:
        lines.append(f"The run also ends automatically once a full development evaluation of the selected checkpoint "
                     f"meets the target, after {NO_IMPROVEMENT_PATIENCE} consecutive evaluated candidates without "
                     "improvement, or when the budget cannot cover the next operation (BUDGET_EXCEEDED).")
    else:
        lines.append("The run also ends automatically once the permitted candidate is committed and selected, or "
                     "when the budget cannot cover the next operation (BUDGET_EXCEEDED).")
    lines += ["Every mutating call needs a new idempotency_key; repeating a key with the same input returns the "
              "original result.",
              "Keep tool arguments and messages to brief hypotheses and observable evidence; do not write out "
              "private reasoning.",
              f"Available tools: {', '.join(tools)}."]
    return "\n".join(lines)


def initial_message(experiment_id: str) -> str:
    return f"Begin work on experiment {experiment_id}."


def serve_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The complete environment of a ``wse serve`` subprocess: allow-listed variables only, tracing disabled.

    Credentials never reach the server: variables whose names look like
    secrets are rejected.
    """
    env = {name: os.environ[name] for name in SERVE_ENV_ALLOWLIST if name in os.environ}
    for name, value in (extra or {}).items():
        if SECRET_NAME.search(name):
            raise WSEError(ErrorCode.POLICY_DENIED, f"environment variable {name} looks like a credential; the WSE "
                           "server never receives credentials")
        env[name] = str(value)
    env["OPENAI_AGENTS_DISABLE_TRACING"] = "1"
    return env


# ---------------------------------------------------------------------------
# Executing strategy actions through the run-bound service
# ---------------------------------------------------------------------------


class ExperimentExecutor:
    """Executes strategy actions through a run-bound Engine, by MCP tool name and arguments.

    The result of each operation is the JSON of the same domain object the
    MCP tool returns as ``result``; a domain failure becomes the same error
    body. Only ``STRATEGY_TOOLS`` are accepted.
    """

    def __init__(self, engine: Engine, *, constraint_status: str, tools: Sequence[str] = STRATEGY_TOOLS) -> None:
        if engine.bound_run_id is None:
            raise WSEError(ErrorCode.POLICY_DENIED, "strategies run only through an Engine bound to their run grant")
        self.engine = engine
        self.constraint_status = constraint_status
        self.tools = tuple(tools)
        e = engine
        self._operations: dict[str, Callable[..., Any]] = {
            "wse_list_models": lambda: {"models": [{"checkpoint_id": m.checkpoint_id, "weights_hash": m.weights_hash,
                                                    "parameter_count": m.parameter_count}
                                                   for m in e.list_models()]},
            "wse_get_experiment": lambda experiment_id: e.describe_experiment(
                experiment_id, constraint_status=self.constraint_status),
            "wse_inspect_model": lambda experiment_id, checkpoint_id: e.inspect_model_details(experiment_id,
                                                                                              checkpoint_id),
            "wse_read_tensor": lambda experiment_id, checkpoint_id, tensor, selection: e.read_model_tensor(
                experiment_id, checkpoint_id, tensor, selection),
            "wse_analyze": lambda experiment_id, checkpoint_id, kind, tensor, selection, k=None: e.analyze_model(
                experiment_id, checkpoint_id, kind, tensor, selection, k),
            "wse_preview_patch": lambda experiment_id, patch_spec, idempotency_key: e.preview_patch(
                experiment_id, patch_spec, idempotency_key),
            "wse_apply_patch": lambda experiment_id, patch_id, idempotency_key: e.apply_patch(
                experiment_id, patch_id, idempotency_key),
            "wse_submit_evaluation": lambda experiment_id, checkpoint_id, suite_id, eval_config, idempotency_key:
                e.submit_evaluation(experiment_id, checkpoint_id, suite_id, eval_config, idempotency_key),
            "wse_get_job": lambda experiment_id, job_id, wait_seconds=0: e.get_job(experiment_id, job_id,
                                                                                    wait_seconds),
            "wse_cancel_job": lambda experiment_id, job_id, idempotency_key: e.cancel_job(experiment_id, job_id,
                                                                                         idempotency_key),
            "wse_compare": lambda experiment_id, base_report_id, candidate_report_id: e.compare(
                experiment_id, base_report_id, candidate_report_id),
            "wse_select_checkpoint": lambda experiment_id, checkpoint_id, expected_selected_id, idempotency_key:
                e.select_checkpoint(experiment_id, checkpoint_id, expected_selected_id, idempotency_key),
        }

    def execute(self, action: Action) -> Observation:
        tool, arguments = action.tool or "", dict(action.arguments)
        if tool not in self.tools:
            error = WSEError(ErrorCode.POLICY_DENIED, f"{tool!r} is not an operation this controller may use",
                             details={"allowed": list(self.tools)})
            return Observation(tool=tool, arguments=arguments, ok=False,
                               error=error.to_domain_error().model_dump(mode="json"))
        try:
            value = self._operations[tool](**arguments)
        except WSEError as exc:
            return Observation(tool=tool, arguments=arguments, ok=False,
                               error=exc.to_domain_error().model_dump(mode="json"))
        except TypeError as exc:
            error = WSEError(ErrorCode.INVALID_ARGUMENT, f"invalid arguments for {tool}: {exc}")
            return Observation(tool=tool, arguments=arguments, ok=False,
                               error=error.to_domain_error().model_dump(mode="json"))
        result = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        return Observation(tool=tool, arguments=arguments, ok=True, result=result)


# ---------------------------------------------------------------------------
# Stop conditions (read from authoritative WSE state)
# ---------------------------------------------------------------------------


class StopMonitor:
    """Checks the stop conditions of an experiment from the operator's view of WSE state."""

    def __init__(self, engine: Engine, experiment_id: str, task: TaskDefinition | None) -> None:
        self.engine = engine
        self.experiment_id = experiment_id
        self.task = task
        self.denials = 0

    def observe(self, error_code: str | None) -> str | None:
        """Record one completed operation (its error code, if any) and return a met stop condition."""
        self.denials = self.denials + 1 if error_code == ErrorCode.POLICY_DENIED.value else 0
        return self.check(error_code)

    def check(self, error_code: str | None = None) -> str | None:
        experiment = self.engine.get_experiment(self.experiment_id)
        if experiment.state != "OPEN":
            return f"experiment_not_open: the experiment is {experiment.state}"
        budget = experiment.budget
        if error_code == ErrorCode.BUDGET_EXCEEDED.value:
            return "insufficient_budget: an operation failed with BUDGET_EXCEEDED"
        if budget["compute_seconds"].remaining <= 0:
            return "insufficient_budget: no computation time remains"
        if self.denials >= POLICY_DENIAL_LIMIT:
            return f"repeated_policy_denials: {self.denials} consecutive operations were denied by the policy"
        if experiment.mode == "weights_only":
            if budget["candidate_commits"].remaining <= 0 and \
                    experiment.selected_checkpoint_id != experiment.base_checkpoint_id:
                return "candidate_committed: the single permitted candidate is committed and selected"
            return None
        if budget["candidate_commits"].remaining <= 0 and budget["evaluation_batches"].remaining <= 0:
            return "insufficient_budget: no candidate commits or evaluation batches remain"
        reports = self._full_development_reports()
        if any(r.checkpoint_id == experiment.selected_checkpoint_id and r.target_met for r in reports):
            return "target_met: a full development evaluation of the selected checkpoint meets the target"
        streak = self.no_improvement_streak(reports, experiment.base_checkpoint_id)
        if streak >= NO_IMPROVEMENT_PATIENCE:
            return f"no_improvement: {streak} consecutive evaluated candidates showed no improvement"
        return None

    def _full_development_reports(self) -> list[Any]:
        task = self.task
        return [r for r in self.engine.list_reports(self.experiment_id)
                if r.split == "development" and (task is None or (r.suite_id == task.development_suite_id
                                                                  and r.sample_count == task.development_examples))]

    @staticmethod
    def no_improvement_streak(reports: Sequence[Any], base_checkpoint_id: str) -> int:
        """Consecutive most recent evaluated candidates whose accuracy did not exceed the best before them."""
        best: float | None = None
        streak, seen = 0, set()
        for report in reports:
            if report.checkpoint_id in seen:
                continue
            seen.add(report.checkpoint_id)
            accuracy = report.metrics.get("accuracy")
            if accuracy is None:
                continue
            if report.checkpoint_id == base_checkpoint_id:
                best = accuracy if best is None else max(best, accuracy)
                continue
            if best is None or accuracy > best:
                best, streak = accuracy, 0
            else:
                streak += 1
        return streak


def _action_shortfall(action: Action, budget: Mapping[str, BudgetEntry]) -> str | None:
    """The counter the next action cannot afford, from the remaining budget (or ``None``)."""
    needs: dict[str, float] = {}
    arguments = action.arguments
    if action.tool == "wse_preview_patch":
        needs["patch_previews"] = 1
    elif action.tool == "wse_apply_patch":
        needs["candidate_commits"] = 1
    elif action.tool == "wse_read_tensor":
        needs["tensor_read_calls"] = 1
    elif action.tool == "wse_analyze":
        needs["analysis_calls"] = 1
    elif action.tool == "wse_submit_evaluation":
        config = arguments.get("eval_config") if isinstance(arguments.get("eval_config"), Mapping) else {}
        needs["evaluation_batches"] = 1
        needs["evaluated_examples"] = config.get("sample_count", 1) if isinstance(config.get("sample_count"),
                                                                                    int) else 1
    if action.tool in ("wse_inspect_model", "wse_read_tensor", "wse_analyze", "wse_preview_patch", "wse_apply_patch",
                       "wse_submit_evaluation"):
        needs["compute_seconds"] = 1e-9
    for counter, amount in needs.items():
        entry = budget.get(counter)
        if entry is not None and entry.remaining < amount:
            return (f"insufficient_budget: {action.tool} needs {counter}, which has {entry.remaining:g} "
                    "remaining")
    return None


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


class _Trace:
    """Persists normalized controller events for one experiment."""

    def __init__(self, engine: Engine, experiment_id: str) -> None:
        self.db = engine.store.db
        self.experiment_id = experiment_id
        self.started = time.monotonic()
        self.seq = 0

    def record(self, event: AgentEvent) -> None:
        append_controller_event(self.db, self.experiment_id, event.kind,
                                event.model_dump(mode="json", exclude={"seq", "kind"}, exclude_none=True))

    def emit(self, kind: str, **fields: Any) -> AgentEvent:
        from wse.database import utc_now

        self.seq += 1
        event = AgentEvent(seq=self.seq, at=utc_now(), elapsed_seconds=round(time.monotonic() - self.started, 6),
                           kind=kind, **fields)
        self.record(event)
        return event


class ReferenceRunner:
    """Runs one controller per run grant and finalizes its experiment (see the module docstring).

    ``setup`` names operator setup hooks (``module:function``) applied to
    the runner's registry and passed to the server and worker processes;
    ``worker_args`` are extra worker options; ``serve_env`` adds variables
    to the allow-listed server environment (never credentials);
    ``adapter_factory(config, experiment_id)`` overrides adapter creation
    (for example to supply an SDK test double).
    """

    def __init__(self, workspace: str | Path, *, registry: Registry | None = None, setup: Sequence[str] = (),
                 worker_args: Sequence[str] = (), serve_env: Mapping[str, str] | None = None,
                 adapter_factory: Callable[[AgentConfig, str], AgentAdapter] | None = None,
                 settle_timeout_seconds: float = 120.0, log_dir: str | Path | None = None) -> None:
        from wse.worker import _load_setup

        self.workspace = Path(workspace).resolve()
        self.setup = tuple(setup)
        self.registry = registry or builtin_registry()
        if registry is None:
            for spec in self.setup:
                _load_setup(spec)(self.registry)
        self.worker_args = tuple(worker_args)
        self.serve_env = dict(serve_env or {})
        self.adapter_factory = adapter_factory
        self.settle_timeout_seconds = settle_timeout_seconds
        self.log_dir = Path(log_dir) if log_dir is not None else self.workspace / "logs"
        self.last_result: dict[str, Any] | None = None

    # -- entry point ---------------------------------------------------------------------

    def run_experiment(self, run_id: str, task_id: str, policy_id: str,
                       agent_config: StrategyConfig | AgentConfig | Mapping[str, Any]) -> str:
        """Run one controller on the run grant's experiment and freeze it; return the experiment ID."""
        config = parse_controller_config(agent_config)
        with Engine(self.workspace, registry=self.registry, run_id=run_id) as bound:
            grant = bound.experiments.get_run_grant(run_id)
            if (grant.task_id, grant.policy_id) != (task_id, policy_id):
                raise WSEError(ErrorCode.POLICY_DENIED, "the run grant is for a different task or policy",
                               details={"task_id": grant.task_id, "policy_id": grant.policy_id})
            experiment = bound.create_experiment(run_id, task_id, policy_id, f"runner-{run_id}")
            eid = experiment.experiment_id
            if experiment.state != "OPEN":
                raise WSEError(ErrorCode.VERSION_CONFLICT, f"experiment {eid} is {experiment.state}; a run grant "
                               "permits one controller run")
            policy = bound.experiment_policy(eid)
            task = bound.get_task(task_id)
        if isinstance(config, StrategyConfig):
            self._run_strategy(eid, task, policy, config)
        else:
            self._run_agent(run_id, eid, task, policy, config)
        return eid

    # -- records -----------------------------------------------------------------------------

    def _start_record(self, engine: Engine, eid: str, *, kind: str, ref: ExtensionRef, config: dict[str, Any],
                      constraint_status: str, tracing: dict[str, Any]) -> None:
        record_controller_start(engine.store.db, eid, kind=kind, controller_id=ref.extension_id,
                                controller_version=ref.version, config=config,
                                constraint_status=constraint_status, tracing=tracing)

    def _end_record(self, engine: Engine, eid: str, result: AgentRunResult, stop_condition: str | None,
                    finalization: dict[str, Any] | None, error: str | None = None) -> None:
        record = {**result.model_dump(mode="json"), "stop_condition": stop_condition,
                  "finalization": finalization, "environment": environment_record()}
        if error is not None:
            record["finalization_error"] = error
        record_controller_end(engine.store.db, eid, record)
        self.last_result = controller_record(engine.store.db, eid)

    # -- finalization ----------------------------------------------------------------------

    def _finalize(self, engine: Engine, eid: str) -> dict[str, Any]:
        """Cancel and settle every outstanding job, then freeze the selected checkpoint."""
        cancelled: list[str] = []
        for _ in range(3):
            deadline = time.monotonic() + self.settle_timeout_seconds
            while True:
                engine.recover_jobs()
                active = [job for job in engine.list_jobs(eid) if not job.terminal]
                if not active:
                    break
                for job in active:
                    if job.job_id not in cancelled:
                        engine.cancel_job(eid, job.job_id, f"runner-cancel-{job.job_id}")
                        cancelled.append(job.job_id)
                left = deadline - time.monotonic()
                if left <= 0:
                    raise WSEError(ErrorCode.WORKER_FAILED,
                                   f"{len(active)} jobs did not settle within {self.settle_timeout_seconds:g} s; "
                                   "the experiment stays open", details={"jobs": [j.job_id for j in active]})
                engine.get_job(eid, active[0].job_id, min(2.0, left))
            selected = engine.get_experiment(eid).selected_checkpoint_id
            try:
                frozen = engine.freeze(eid, selected)
            except WSEError as exc:
                if exc.code != ErrorCode.VERSION_CONFLICT:
                    raise
                continue
            return {"cancelled_jobs": cancelled, "state": frozen.state,
                    "frozen_checkpoint_id": frozen.selected_checkpoint_id}
        raise WSEError(ErrorCode.VERSION_CONFLICT, f"experiment {eid} could not be frozen after three attempts")

    # -- stepwise strategies -----------------------------------------------------------------

    def _run_strategy(self, eid: str, task: TaskDefinition, policy: ExperimentPolicy,
                      config: StrategyConfig) -> None:
        from wse.worker import WorkerSupervisor

        ref = self.registry.resolve_reference("strategy", config.strategy)
        strategy = self.registry.resolve("strategy", ref.extension_id, ref.version)(**config.parameters)
        tools = allowed_tools(policy)
        recorded = {**config.model_dump(mode="json"), "strategy": str(ref), "resolved_config": strategy.config(),
                    "allowed_tools": tools}
        run_id = None
        supervisor = None
        with Engine(self.workspace, registry=self.registry) as operator:
            run_id = operator.get_experiment(eid).run_id
            self._start_record(operator, eid, kind="strategy", ref=ref, config=recorded,
                               constraint_status="verified",
                               tracing={"sdk_tracing": "not applicable: no agent SDK runs"})
            monitor = StopMonitor(operator, eid, task)
            trace = _Trace(operator, eid)
            if policy.mode == "bounded_eval" and policy.budget.evaluation_batches > 0:
                worker_args = [arg for spec in self.setup for arg in ("--setup", spec)] + list(self.worker_args)
                supervisor = WorkerSupervisor(self.workspace, extra_args=worker_args).start()
            try:
                result, stop = self._strategy_loop(strategy, eid, task, config, tools, monitor, trace, run_id)
                try:
                    finalization = self._finalize(operator, eid)
                except WSEError as exc:
                    self._end_record(operator, eid, result, stop, None, error=exc.message)
                    raise
                self._end_record(operator, eid, result, stop, finalization)
            finally:
                if supervisor is not None:
                    supervisor.stop(stop_timeout=5.0)

    def _strategy_loop(self, strategy: Any, eid: str, task: TaskDefinition, config: StrategyConfig,
                       tools: Sequence[str], monitor: StopMonitor, trace: _Trace, run_id: str
                       ) -> tuple[AgentRunResult, str | None]:
        started = time.monotonic()
        trace.emit("run_started", details={"strategy": f"{strategy.strategy_id}@{strategy.version}",
                                           "max_steps": config.max_steps,
                                           "wall_clock_seconds": config.wall_clock_seconds})
        observations: list[Observation] = []
        status, reason, stop = "max_turns", f"the step limit of {config.max_steps} was reached", None
        steps = 0
        with Engine(self.workspace, registry=self.registry, run_id=run_id) as bound:
            executor = ExperimentExecutor(bound, constraint_status="verified", tools=tools)

            def execute(action: Action) -> Observation:
                trace.emit("tool_call", tool=action.tool, arguments=action.arguments,
                           details={"rationale": action.rationale} if action.rationale else {})
                observation = executor.execute(action)
                trace.emit("tool_result", tool=action.tool, ok=observation.ok, error_code=observation.error_code,
                           summary=None if observation.ok else (observation.error or {}).get("message"))
                observations.append(observation)
                return observation

            execute(Action.call("wse_get_experiment", "Read the task and budget.", experiment_id=eid))
            while steps < config.max_steps:
                if time.monotonic() - started > config.wall_clock_seconds:
                    status, reason = "timeout", f"the wall-clock limit of {config.wall_clock_seconds:g} s was reached"
                    break
                stop = monitor.observe(observations[-1].error_code)
                if stop is not None:
                    status, reason = "stopped", stop
                    break
                budget = bound.get_experiment(eid).budget
                action = strategy.next_action(task.objective, list(observations), budget)
                steps += 1
                if action.kind == "stop":
                    status, reason = "completed", f"the strategy stopped: {action.reason}"
                    trace.emit("message", output=action.reason)
                    break
                shortfall = _action_shortfall(action, budget)
                if shortfall is not None:
                    status, reason = "stopped", shortfall
                    stop = shortfall
                    break
                execute(action)
        if stop is not None:
            trace.emit("cancel_requested", details={"reason": stop})
        trace.emit("run_finished", details={"status": status, "reason": reason})
        not_applicable = "not applicable: a scripted strategy calls no language model"
        usage = AgentUsage(provider=None, model=None, unavailable={
            name: not_applicable for name in ("requests", "input_tokens", "output_tokens", "total_tokens",
                                              "cost_usd", "provider", "model", "model_version")})
        return AgentRunResult(status=status, reason=reason, turns=steps,
                              tool_calls=len(observations), elapsed_seconds=round(time.monotonic() - started, 6),
                              usage=usage, provider_metadata={"controller": "strategy",
                                                              "execution_channels": ["wse_engine_run_bound"]}), stop

    # -- SDK-backed agents ---------------------------------------------------------------------

    def _make_adapter(self, config: AgentConfig, eid: str) -> AgentAdapter:
        if self.adapter_factory is not None:
            return self.adapter_factory(config, eid)
        ref = str(ExtensionRef.parse(config.adapter))
        if ref not in AGENT_ADAPTERS:
            raise WSEError(ErrorCode.NOT_FOUND, f"agent adapter {ref} is not available; known: {sorted(AGENT_ADAPTERS)}")
        import importlib

        module_name, class_name = AGENT_ADAPTERS[ref]
        adapter_class = getattr(importlib.import_module(module_name), class_name)
        if config.model.startswith(TEST_DOUBLE_PREFIX):
            from wse.agent_adapters.openai_agents import strategy_model

            strategy_ref = self.registry.resolve_reference("strategy", config.model[len(TEST_DOUBLE_PREFIX):])
            strategy = self.registry.resolve("strategy", strategy_ref.extension_id, strategy_ref.version)()
            return adapter_class(model=strategy_model(strategy, experiment_id=eid))
        return adapter_class()

    def connection(self, run_id: str, eid: str, task: TaskDefinition, policy: ExperimentPolicy,
                   constraint_status: str) -> RunConnection:
        """The MCP connection for a run: ``wse serve`` bound to the grant, allow-listed tools and environment."""
        tools = allowed_tools(policy)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "wse.cli", "--workspace", str(self.workspace), "serve", "--run-id", run_id,
                   "--constraint-status", constraint_status, "--log-file", str(self.log_dir / f"serve-{run_id}.log")]
        for spec in self.setup:
            command += ["--setup", spec]
        if self.worker_args:
            command += ["--worker-args", " ".join(self.worker_args)]
        return RunConnection(
            workspace=str(self.workspace), run_id=run_id, experiment_id=eid, task_id=task.task_id,
            policy_id=policy.policy_id, serve_command=command, serve_env=serve_environment(self.serve_env),
            cwd=str(self.workspace), allowed_tools=tools,
            instructions=render_instructions(task, policy, experiment_id=eid, tools=tools),
            initial_message=initial_message(eid), constraint_status=constraint_status)

    def _run_agent(self, run_id: str, eid: str, task: TaskDefinition, policy: ExperimentPolicy,
                   config: AgentConfig) -> None:
        adapter = self._make_adapter(config, eid)
        constraint_status = "verified" if config.benchmark else "unverified"
        connection = self.connection(run_id, eid, task, policy, constraint_status)
        recorded = {**config.model_dump(mode="json"), "allowed_tools": connection.allowed_tools,
                    "serve_env_names": sorted(connection.serve_env), "cwd": connection.cwd}
        tracing = {"sdk_tracing_disabled": config.tracing_disabled, "benchmark": config.benchmark,
                   "trace_include_sensitive_data": not config.tracing_disabled and not config.benchmark,
                   "server_env_OPENAI_AGENTS_DISABLE_TRACING": "1"}
        ref = ExtensionRef.parse(config.adapter)

        async def main() -> None:
            with Engine(self.workspace, registry=self.registry) as operator:
                self._start_record(operator, eid, kind="agent", ref=ref, config=recorded,
                                   constraint_status=constraint_status, tracing=tracing)
                monitor = StopMonitor(operator, eid, task)
                trace = _Trace(operator, eid)
                stops: list[str] = []

                def sink(event: AgentEvent) -> None:
                    trace.record(event)
                    if event.kind != "tool_result" or stops:
                        return
                    reason = monitor.observe(event.error_code)
                    if reason is not None:
                        stops.append(reason)
                        trace.emit("cancel_requested", details={"reason": reason})
                        adapter.cancel(eid, reason)

                await adapter.start(connection, config)
                try:
                    result = await adapter.run(eid, config, sink)
                    try:
                        finalization = await asyncio.to_thread(self._finalize, operator, eid)
                    except WSEError as exc:
                        self._end_record(operator, eid, result, stops[0] if stops else None, None,
                                         error=exc.message)
                        raise
                    self._end_record(operator, eid, result, stops[0] if stops else None, finalization)
                finally:
                    await adapter.close(eid)

        asyncio.run(main())


def run_experiment(run_id: str, task_id: str, policy_id: str,
                   agent_config: StrategyConfig | AgentConfig | Mapping[str, Any], *, workspace: str | Path,
                   **runner_options: Any) -> str:
    """Run one controller on a run grant's experiment and freeze it (see ``ReferenceRunner``)."""
    return ReferenceRunner(workspace, **runner_options).run_experiment(run_id, task_id, policy_id, agent_config)
