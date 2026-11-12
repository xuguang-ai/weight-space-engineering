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
     The runner owns the evaluation worker for these runs: it starts one
     whenever the policy grants jobs (evaluation batches, generation passes,
     or probes).
   * An agent (``AgentConfig``; the adapter ``codex_cli@1`` runs the Codex
     CLI) runs its own complete tool loop against ``wse serve`` bound to the
     run grant; the adapter owns the run's evaluation worker. The runner only
     listens to the adapter's normalized events, persists them, and checks
     stop conditions.

3. **Stop conditions** are read from authoritative WSE state after every
   operation: a budget error (insufficient budget for the next operation,
   including no candidate commits left while no development evaluation, by
   evaluation batches or generation passes, remains), the selected
   checkpoint's full development evaluation meeting the task's target, 3
   consecutive evaluated candidates without improvement (a higher accuracy
   than the best checkpoint before them or, on a generation suite, a lower
   limit-hit rate at no lower accuracy), or 3 consecutive policy denials
   (bounded_eval); the single permitted candidate committed
   and selected (weights_only, which uses no development-score threshold);
   and the controller stopping voluntarily. A met condition cancels the
   controller.
4. **Finalize after the controller exits.** The runner cancels and settles
   every outstanding job (recovering jobs whose worker was lost), then
   freezes the selected checkpoint, while an agent adapter's worker still
   runs; only then is the adapter closed. Neither the agent nor the runner
   closes an experiment: final evaluation (``wse final-evaluate``) is a
   separate operator step. This holds however the controller ends, including
   a failure inside the runner (status ``error``) and an operator interrupt:
   the first Ctrl-C (SIGINT) while a controller runs stops it at its next
   step or cancels the agent's loop, the trial is recorded as ``cancelled`` with
   the stop condition ``operator_interrupt``, and ``KeyboardInterrupt`` is
   raised to the caller only after the experiment is settled and frozen.
   A second Ctrl-C aborts at once and can leave the experiment open.

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
import signal
import sys
import threading
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
    describe_exception,
)
from wse.api import Engine, model_list_page
from wse.audit import (
    append_controller_event,
    controller_record,
    environment_record,
    record_controller_end,
    record_controller_start,
)
from wse.policy import ExperimentPolicy, sandbox_available
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
    "SANDBOX_TOOLS",
    "SUBMISSION_TOOLS",
    "agent_session_timeout",
    "submissions_available",
    "JOB_TOOLS",
    "OPERATOR_INTERRUPT",
    "PROBE_TOOLS",
    "TEST_DOUBLE_PREFIX",
    "ExperimentExecutor",
    "ReferenceRunner",
    "StopMonitor",
    "StrategyConfig",
    "allowed_tools",
    "initial_message",
    "jobs_available",
    "parse_controller_config",
    "probes_available",
    "render_instructions",
    "render_trials_markdown",
    "run_experiment",
    "serve_environment",
    "summarize_trial",
    "constraint_for",
    "SANDBOX_CONSTRAINT_REASON",
    "SANDBOX_TIME_LABEL",
]

logger = logging.getLogger("wse.runner")

EVALUATION_TOOLS = ("wse_evaluate", "wse_submit_evaluation", "wse_get_job", "wse_get_report", "wse_cancel_job",
                    "wse_compare")
PROBE_TOOLS = ("wse_probe",)
SANDBOX_TOOLS = ("wse_run_python",)  # code execution: only under a policy that grants the sandbox
SUBMISSION_TOOLS = ("wse_submit_candidate",)  # tensors from the sandbox: the sandbox and the replace_tensor edit
JOB_TOOLS = ("wse_get_job", "wse_cancel_job")  # evaluations and probes are both jobs
NO_IMPROVEMENT_PATIENCE = 3
IMPROVEMENT_TOLERANCE = 1e-9
POLICY_DENIAL_LIMIT = 3
TEST_DOUBLE_PREFIX = "wse-test-double:"
OPERATOR_INTERRUPT = "operator_interrupt: the operator interrupted the run (SIGINT)"
# Why a run whose policy grants the sandbox is unverified, whatever its controller (M3 plan Section 2.5).
SANDBOX_CONSTRAINT_REASON = "sandbox code execution: WSE meters only its own calls"
# How reports name the sandbox's usage, apart from the calls WSE meters.
SANDBOX_TIME_LABEL = "sandbox wall-clock time (not metered computation)"
SERVE_ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "PYTHONPATH", "VIRTUAL_ENV",
                       "SYSTEMROOT")
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
AGENT_ADAPTERS = {"codex_cli@1": ("wse.agent_adapters.codex_cli", "CodexCLIAdapter")}


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


def probes_available(policy: ExperimentPolicy) -> bool:
    """Whether runs of ``policy`` may probe activations: a ``bounded_eval`` policy that grants probe calls."""
    return policy.mode == "bounded_eval" and policy.budget.limit("probe_calls") > 0


def jobs_available(policy: ExperimentPolicy) -> bool:
    """Whether runs of ``policy`` submit jobs (evaluations, generation passes, or probes), which a worker runs."""
    return policy.evaluation_available or probes_available(policy)


def constraint_for(policy: ExperimentPolicy, verified: bool) -> tuple[str, str | None]:
    """A run's ``constraint_status`` and, when its policy decides it, the reason.

    A policy that grants the sandbox makes every run unverified, because the
    agent's own code runs computation that WSE does not meter. Otherwise the
    controller decides: ``verified`` when its only channel is the run-bound
    interface (``verified``).
    """
    if sandbox_available(policy):
        return "unverified", SANDBOX_CONSTRAINT_REASON
    return ("verified" if verified else "unverified"), None


# The agent's MCP session must outlast the longest tool call a policy permits (M3 plan Section 2.4).
SESSION_TIMEOUT_MARGIN_SECONDS = 20.0
SANDBOX_CALL_MARGIN_SECONDS = 120.0  # exporting inputs before a call and storing its outputs after it


def _worker_device(worker_args: Sequence[str]) -> str | None:
    """The ``--device`` among the worker's options (``--device cpu`` or ``--device=cpu``), or ``None``."""
    args = list(worker_args)
    for index, arg in enumerate(args):
        if arg == "--device" and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith("--device="):
            return arg.split("=", 1)[1]
    return None


def agent_session_timeout(policy: ExperimentPolicy, configured: float) -> float:
    """The agent's MCP session timeout: ``configured``, raised above the longest tool call of ``policy``.

    The longest calls are a ``wse_get_job`` wait (the policy's cap), a
    ``wse_evaluate`` call when development evaluations are permitted (it
    waits for its job up to ``evaluation_wait_cap``), and, under a sandbox
    policy, a ``wse_run_python`` call (its wall-clock limit, plus the export
    of inputs before it and the storing of outputs after it).
    """
    from wse.jobs import evaluation_wait_cap, wait_cap

    longest = wait_cap(policy) + SESSION_TIMEOUT_MARGIN_SECONDS
    if policy.evaluation_available:
        longest = max(longest, evaluation_wait_cap(policy) + SESSION_TIMEOUT_MARGIN_SECONDS)
    if sandbox_available(policy):
        assert policy.sandbox is not None
        longest = max(longest, policy.sandbox.seconds_per_call + SANDBOX_CALL_MARGIN_SECONDS)
    return max(float(configured), longest)


def submissions_available(policy: ExperimentPolicy) -> bool:
    """Whether runs of ``policy`` may submit tensors computed in the sandbox: the sandbox and ``replace_tensor``."""
    return sandbox_available(policy) and "replace_tensor" in policy.operations.edits


def allowed_tools(policy: ExperimentPolicy) -> list[str]:
    """The WSE tools a controller of this policy may use, in ``STRATEGY_TOOLS`` order.

    Evaluation tools (``EVALUATION_TOOLS``: whole evaluations waited for,
    submission, reports, and comparisons) when the policy permits
    development evaluations (evaluation batches or generation passes,
    ``evaluation_available``), ``wse_probe``
    when it grants probes, ``wse_run_python`` when it grants the sandbox, and
    the job tools for either.
    """
    evaluation, probes = policy.evaluation_available, probes_available(policy)

    def permitted(tool: str) -> bool:
        if tool in SANDBOX_TOOLS:
            return sandbox_available(policy)
        if tool in SUBMISSION_TOOLS:
            return submissions_available(policy)
        if tool in PROBE_TOOLS:
            return probes
        if tool in JOB_TOOLS:
            return evaluation or probes
        return evaluation or tool not in EVALUATION_TOOLS

    return [tool for tool in STRATEGY_TOOLS if permitted(tool)]


def _gate_text(gate: Any) -> str:
    if gate.min_value is not None:
        return f"{gate.metric} at least {gate.min_value:g}"
    if gate.min_delta is not None:
        return f"{gate.metric} improves on the starting model by at least {gate.min_delta:g}"
    return f"{gate.metric} drops by at most {gate.max_drop:g} from the starting model"


def render_instructions(task: TaskDefinition, policy: ExperimentPolicy, *, experiment_id: str,
                        tools: Sequence[str], baseline: Mapping[str, Any] | None = None) -> str:
    """Agent instructions built only from the task definition, the policy, and the run's IDs and tools.

    ``baseline`` (report ID and metrics), when given, is the base checkpoint's
    full development evaluation, made before the controller started.
    """
    budget, operations = policy.budget, policy.operations
    bounded = policy.mode == "bounded_eval"
    batches = bounded and budget.evaluation_batches > 0
    passes = bounded and budget.limit("generation_passes") > 0
    probes = probes_available(policy)
    sandbox = sandbox_available(policy)
    evaluation = batches or passes
    from wse.jobs import wait_cap

    cap = wait_cap(policy)
    lines = [
        "You control one managed weight-space engineering experiment through the WSE tools listed below. They are "
        + ("your only tools. wse_run_python runs your own Python in a sandbox that reads only its own inputs and has "
           "no network; no other tool runs code, reads files, or uses the network." if sandbox else
           "your only tools: you cannot run code, read files, or use the network."),
        "",
        f"Experiment ID: {experiment_id} (the tools use it when you omit experiment_id)",
        f"Task {task.task_id}: {task.title}",
        task.description,
    ]
    if task.input_description:
        lines.append(f"Model input: {task.input_description}")
    if task.output_description:
        lines.append(f"Model output: {task.output_description}")
    lines.append(f"Objective: {task.objective.description}")
    lines.append("Acceptance: " + "; ".join(_gate_text(gate) for gate in task.objective.gates) + ".")
    if task.objective.selection is not None:
        from wse.selection import describe_rule

        lines.append("Selection when the run ends: " + describe_rule(task.objective.selection) + " Evaluate each "
                     "candidate on the full development set so the rule can consider it.")
    if baseline is not None:
        metrics = ", ".join(f"{name} {value:.4g}" for name, value in sorted(baseline["metrics"].items()))
        again = ("wse_evaluate on the original model returns this report without charging again"
                 if "wse_evaluate" in tools else "evaluating the original model again repeats it")
        lines.append(f"Baseline: the original model's full development evaluation is report "
                     f"{baseline['report_id']} ({metrics}); it counts as one of the evaluations, and {again}.")
    if task.target_accuracy is not None:
        lines.append(f"The target is met only by a successful evaluation of development suite "
                     f"{task.development_suite_id} covering all {task.development_examples} examples with accuracy "
                     f"at least {task.target_accuracy:g}.")
    lines += ["", f"Policy {policy.policy_id} (mode {policy.mode}); budget for the whole experiment:",
              f"- candidate commits: {budget.candidate_commits}",
              f"- patch previews: {budget.patch_previews}",
              f"- tensor analyses: {budget.analysis_calls}",
              f"- tensor reads: {budget.tensor_read_calls}"]
    if batches:
        lines += [f"- development evaluation batches: {budget.evaluation_batches}, at most "
                  f"{budget.examples_per_batch} examples each and {budget.evaluated_examples} examples in total"]
    if passes:
        lines += [f"- generation passes: {budget.generation_passes}, each at most "
                  f"{budget.problems_per_generation_pass} problems and {budget.new_tokens_per_output} new tokens per "
                  f"output; {budget.generated_tokens} generated tokens in total"]
    if probes:
        lines += [f"- probe calls: {budget.probe_calls}, each at most {budget.texts_per_probe_call} texts of at most "
                  f"{budget.tokens_per_probe_text} tokens; {budget.teacher_forced_tokens} teacher-forced tokens in "
                  "total"]
    if sandbox:
        assert policy.sandbox is not None
        lines += [f"- sandbox calls: {budget.sandbox_calls}, each at most {policy.sandbox.seconds_per_call:g} s of "
                  f"wall-clock time; {budget.sandbox_seconds:g} s in total. WSE counts the calls and their time; "
                  "what your code computes is not metered, so this run is unverified."]
    lines += [f"- computation time: {budget.compute_seconds:g} s in total, at most {budget.seconds_per_operation:g} s "
              "per operation",
              "Consumed allowance is never refunded; every result reports usage and the remaining budget.",
              f"Analyses: {', '.join(sorted(operations.analyses))}. Edit operations: "
              f"{', '.join(sorted(operations.edits))}. A read returns at most "
              f"{operations.limits.max_read_elements} elements; a patch has at most "
              f"{operations.limits.max_patch_operations} operations.",
              "",
              "How to work:"]
    steps = ["Call wse_get_experiment with the experiment ID for the task, budget, permitted operations, development "
             "suites, and checkpoints.",
             "Gather evidence with wse_inspect_model, wse_read_tensor, and wse_analyze on the experiment's "
             "checkpoints."]
    if probes:
        steps.append("Probe activations with wse_probe: texts (outputs of your generation passes by their output_uri, "
                     "strings, or items of a development suite), sites (residual:<i>, <module path>:input or :output, "
                     "or a role site such as {\"role\": \"attn.o\", \"layer\": 0, \"point\": \"output\"}), "
                     "and optionally two groups of text indices for a contrast; wait for its job with wse_get_job. "
                     "Its vectors are references (<result_uri>#/sites/<i>/contrast/direction) that wse_analyze and "
                     "wse_preview_patch accept.")
    if sandbox:
        steps.append("Run your own Python with wse_run_python (NumPy, PyTorch on the CPU, safetensors) in a "
                     "persistent kernel: variables, imports, and loaded tensors stay between calls, each call has a "
                     "time limit, and restart=true starts a new kernel. It reads inputs/weights/base.safetensors "
                     "(the base "
                     "checkpoint's permitted tensors in their stored dtype; load them with "
                     "safetensors.torch.load_file), inputs/passes/<report_id>.jsonl (every output of your generation "
                     "passes, with its grade and text), and inputs/probes/<job_id>.npz and .json (your probe "
                     "vectors); it writes work/ (kept between calls) and outputs/ (emptied before each call). Each "
                     "numeric vector or matrix you save to outputs/ with numpy.save comes back with a pointer, "
                     "<result_uri>#/values (a row: #/values/<row>), which wse_preview_patch and wse_analyze accept as "
                     "a stored vector."
                     + (" To submit tensors you computed, save them with safetensors.torch.save_file to "
                        "outputs/<name>.safetensors under their full checkpoint names, in the stored dtype: the call "
                        "returns a tensor_file, which wse_submit_candidate turns into a candidate (one preview and one "
                        "commit), recorded with the call that saved it." if submissions_available(policy) else ""))
    steps.append("Propose an edit with wse_preview_patch (base_checkpoint_id, a hypothesis of one or two sentences "
                 "stating the observable evidence, and the operations), then create a candidate with wse_apply_patch "
                 "by patch_id.")
    if evaluation:
        text = ("Evaluate checkpoints with wse_submit_evaluation on a development suite with a preregistered "
                f"eval_config (sample_count, seed), wait with wse_get_job (wait_seconds up to {cap:g}), read a "
                "whole report with wse_get_report, and compare two reports with wse_compare.")
        if "wse_evaluate" in tools:
            text = (f"wse_evaluate evaluates a checkpoint on all of development suite {task.development_suite_id} "
                    "and returns its report and the comparison with the base checkpoint in one call, which waits "
                    "until the evaluation ends; each checkpoint is evaluated once, so asking again returns the same "
                    "result without charging again. " + text)
        if passes:
            text += (" A generation suite's report gives every output's grade, length, whether it reached the token "
                     "limit, the end of its text, and its output_uri; wse_compare pairs two such reports on accuracy, "
                     "limit-hit rate, and generated tokens.")
        steps.append(text)
    elif policy.mode == "weights_only":
        steps.append("Development evaluation is not available in this mode: no forward pass of the model is "
                     f"permitted, and the policy allows {budget.candidate_commits} candidate commit(s).")
    else:
        steps.append("Development evaluation is not available: the policy grants no evaluation batches or "
                     "generation passes.")
    steps += ["Select a checkpoint with wse_select_checkpoint, passing the currently selected ID as "
              "expected_selected_id; selecting an earlier checkpoint rolls back.",
              "When you are done, reply with a short final message and no tool call. The selected checkpoint is "
              "then frozen for a separate final evaluation that you cannot see."]
    lines += [f"{number}. {step}" for number, step in enumerate(steps, start=1)] + [""]
    if evaluation:
        # M1 policies keep their M1 sentence; with generation passes, a lower limit-hit rate also counts.
        improvement = (" (a higher accuracy than the best checkpoint before them, or a lower limit-hit rate at no lower "
                       "accuracy)" if passes else "")
        lines.append(f"The run also ends automatically once a full development evaluation of the selected checkpoint "
                     f"meets the target, after {NO_IMPROVEMENT_PATIENCE} consecutive evaluated candidates without "
                     f"improvement{improvement}, or when the budget cannot cover the next operation (BUDGET_EXCEEDED).")
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
    MCP tool returns as ``result`` (including its bounded forms: reports,
    inspections, and candidates within the inline-result limit); a domain
    failure becomes the same error body. As through MCP, ``experiment_id``
    may be omitted for the run's experiment. Only ``STRATEGY_TOOLS`` are
    accepted.
    """

    def __init__(self, engine: Engine, *, constraint_status: str, tools: Sequence[str] = STRATEGY_TOOLS) -> None:
        if engine.bound_run_id is None:
            raise WSEError(ErrorCode.POLICY_DENIED, "strategies run only through an Engine bound to their run grant")
        self.engine = engine
        self.constraint_status = constraint_status
        self.tools = tuple(tools)
        e, eid = engine, engine.resolve_experiment_id
        self._operations: dict[str, Callable[..., Any]] = {
            "wse_list_models": lambda cursor=None: model_list_page(e.list_models(), cursor),
            "wse_get_experiment": lambda experiment_id=None: e.describe_experiment(
                eid(experiment_id), constraint_status=self.constraint_status),
            "wse_inspect_model": lambda *, checkpoint_id, experiment_id=None, cursor=None: e.inspect_model_details(
                eid(experiment_id), checkpoint_id, cursor=cursor),
            "wse_read_tensor": lambda *, checkpoint_id, tensor, selection, experiment_id=None: e.read_model_tensor(
                eid(experiment_id), checkpoint_id, tensor, selection),
            "wse_analyze": lambda *, checkpoint_id, kind, tensor, selection, k=None, parameters=None,
                experiment_id=None: e.analyze_model(eid(experiment_id), checkpoint_id, kind, tensor, selection, k,
                                                    parameters=parameters),
            "wse_probe": lambda *, checkpoint_id, texts, sites, idempotency_key, groups=None, top=None,
                experiment_id=None: e.submit_probe(eid(experiment_id), checkpoint_id, texts, sites, idempotency_key,
                                                   groups=groups, top=top),
            "wse_run_python": lambda *, code, idempotency_key, experiment_id=None, restart=False: e.run_python(
                eid(experiment_id), code, idempotency_key, restart),
            "wse_submit_candidate": lambda *, parent_checkpoint_id, hypothesis, idempotency_key, tensor_file=None,
            path=None, experiment_id=None, tensors=None: e.submit_candidate(
                eid(experiment_id), parent_checkpoint_id, tensor_file, hypothesis, idempotency_key, tensors, path),
            "wse_preview_patch": lambda *, patch_spec, idempotency_key, experiment_id=None: e.preview_patch(
                eid(experiment_id), patch_spec, idempotency_key),
            "wse_apply_patch": lambda *, patch_id, idempotency_key, experiment_id=None: e.apply_patch(
                eid(experiment_id), patch_id, idempotency_key, bounded=True),
            "wse_evaluate": lambda *, checkpoint_id, idempotency_key, experiment_id=None: e.evaluate_checkpoint(
                eid(experiment_id), checkpoint_id, idempotency_key),
            "wse_submit_evaluation": lambda *, checkpoint_id, suite_id, eval_config, idempotency_key,
                experiment_id=None: e.submit_evaluation(eid(experiment_id), checkpoint_id, suite_id, eval_config,
                                                        idempotency_key),
            "wse_get_job": lambda *, job_id, wait_seconds=0, experiment_id=None: e.get_job(
                eid(experiment_id), job_id, wait_seconds),
            "wse_get_report": lambda *, report_id, experiment_id=None: e.get_report(eid(experiment_id), report_id,
                                                                                    bounded=True),
            "wse_cancel_job": lambda *, job_id, idempotency_key, experiment_id=None: e.cancel_job(
                eid(experiment_id), job_id, idempotency_key),
            "wse_compare": lambda *, base_report_id, candidate_report_id, experiment_id=None: e.compare(
                eid(experiment_id), base_report_id, candidate_report_id),
            "wse_select_checkpoint": lambda *, checkpoint_id, expected_selected_id, idempotency_key,
                experiment_id=None: e.select_checkpoint(eid(experiment_id), checkpoint_id, expected_selected_id,
                                                        idempotency_key),
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
        if budget["candidate_commits"].remaining <= 0 and not self._evaluations_remain(budget):
            return ("insufficient_budget: no candidate commits remain, and no development evaluation (evaluation "
                    "batches or generation passes)")
        reports = self._full_development_reports()
        if any(r.checkpoint_id == experiment.selected_checkpoint_id and r.target_met for r in reports):
            return "target_met: a full development evaluation of the selected checkpoint meets the target"
        if self.task is not None and self.task.objective.selection is not None:
            return None  # the agent decides when to stop; the budget bounds it, and the rule chooses at the end
        streak = self.no_improvement_streak(reports, experiment.base_checkpoint_id)
        if streak >= NO_IMPROVEMENT_PATIENCE:
            return f"no_improvement: {streak} consecutive evaluated candidates showed no improvement"
        return None

    @staticmethod
    def _evaluations_remain(budget: Mapping[str, BudgetEntry]) -> bool:
        """Whether a development evaluation remains: an evaluation batch, or a generation pass with tokens."""
        if budget["evaluation_batches"].remaining > 0:
            return True
        passes, tokens = budget.get("generation_passes"), budget.get("generated_tokens")
        return passes is not None and tokens is not None and passes.remaining > 0 and tokens.remaining > 0

    def _full_development_reports(self) -> list[Any]:
        task = self.task
        return [r for r in self.engine.list_reports(self.experiment_id)
                if r.split == "development" and (task is None or (r.suite_id == task.development_suite_id
                                                                  and r.sample_count == task.development_examples))]

    @staticmethod
    def no_improvement_streak(reports: Sequence[Any], base_checkpoint_id: str) -> int:
        """Consecutive most recent evaluated candidates that did not improve on the best checkpoint before them.

        A checkpoint improves on the best when its accuracy is higher; on a
        generation suite (reports with a ``limit_hit_rate``) also when its
        accuracy is at least the best's and its limit-hit rate is lower, which
        is the objective of a task such as the Qwen loop task. The base sets
        the first best; each report counts once per checkpoint.
        """
        best: tuple[float, float | None] | None = None
        streak, seen = 0, set()
        for report in reports:
            if report.checkpoint_id in seen:
                continue
            seen.add(report.checkpoint_id)
            accuracy = report.metrics.get("accuracy")
            if accuracy is None:
                continue
            current = (accuracy, report.metrics.get("limit_hit_rate"))
            if report.checkpoint_id == base_checkpoint_id:
                if best is None or _improves(best, current):
                    best = current
                continue
            if best is None or _improves(best, current):
                best, streak = current, 0
            else:
                streak += 1
        return streak


def _improves(best: tuple[float, float | None], candidate: tuple[float, float | None]) -> bool:
    """Whether ``(accuracy, limit-hit rate)`` improves on the best: a higher accuracy, or, when both have a limit-hit
    rate, a lower one at no lower accuracy (metrics are means over samples; differences below 1e-9 are ties)."""
    (accuracy, rate), (best_accuracy, best_rate) = candidate, best
    if accuracy > best_accuracy:
        return True
    return (rate is not None and best_rate is not None and accuracy >= best_accuracy - IMPROVEMENT_TOLERANCE
            and rate < best_rate - IMPROVEMENT_TOLERANCE)


def _action_shortfall(action: Action, budget: Mapping[str, BudgetEntry], *, generation: bool | None = False,
                      tokens_per_probe_text: int | None = None) -> str | None:
    """The counter the next action cannot afford, from the remaining budget (or ``None``).

    ``generation`` marks a ``wse_submit_evaluation`` of a generation suite,
    which needs a generation pass and generated tokens instead of evaluation
    batches and examples; ``None`` marks one whose suite the run cannot
    resolve (``_generation_submission``), which needs only compute time here:
    the submission itself then reports the domain error, as it does through
    MCP. A probe needs a probe call and, with ``tokens_per_probe_text`` (the
    policy's), the teacher-forced tokens its texts reserve.
    """
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
    elif action.tool == "wse_submit_evaluation" and generation is None:
        pass  # an unresolvable suite: the engine refuses the submission (NOT_FOUND or POLICY_DENIED)
    elif action.tool == "wse_submit_evaluation" and generation:
        needs["generation_passes"] = 1
        needs["generated_tokens"] = 1
    elif action.tool == "wse_submit_evaluation":
        config = arguments.get("eval_config") if isinstance(arguments.get("eval_config"), Mapping) else {}
        needs["evaluation_batches"] = 1
        needs["evaluated_examples"] = config.get("sample_count", 1) if isinstance(config.get("sample_count"),
                                                                                    int) else 1
    elif action.tool == "wse_probe":
        texts = arguments.get("texts")
        count = len(texts) if isinstance(texts, list) and texts else 1
        needs["probe_calls"] = 1
        needs["teacher_forced_tokens"] = count * (tokens_per_probe_text or 1)
    if action.tool in ("wse_inspect_model", "wse_read_tensor", "wse_analyze", "wse_preview_patch", "wse_apply_patch",
                       "wse_submit_evaluation", "wse_probe"):
        needs["compute_seconds"] = 1e-9
    for counter, amount in needs.items():
        entry = budget.get(counter)
        if entry is not None and entry.remaining < amount:
            return (f"insufficient_budget: {action.tool} needs {counter}, which has {entry.remaining:g} "
                    "remaining")
    return None


def _generation_submission(engine: Engine, action: Action) -> bool | None:
    """Whether ``action`` submits an evaluation of a generation suite the run may see (a generation pass).

    ``None`` when the run cannot resolve the suite (a final suite, another
    task's suite, or a misspelled ID): the submission itself reports that
    error, so ``_action_shortfall`` checks no evaluation counter for it.
    """
    if action.tool != "wse_submit_evaluation":
        return False
    try:
        return engine.jobs.generates_text(engine.get_suite(action.arguments.get("suite_id")))
    except WSEError:
        return None


def _probe_tokens(policy: ExperimentPolicy | None) -> int | None:
    """The teacher-forced tokens a probe reserves per text under ``policy``."""
    from wse.probes import MAX_PROBE_TOKENS

    if policy is None or not probes_available(policy):
        return None
    return min(policy.budget.limit("tokens_per_probe_text"), MAX_PROBE_TOKENS)


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


class _OperatorInterrupt:
    """While a controller runs, the first SIGINT requests a graceful stop; a second one raises at once.

    ``requested`` is polled by the strategy loop; an agent run registers a
    callback (``on_request``) that cancels the agent's loop from the event loop.
    The handler is installed only in the main thread and is restored on exit.
    """

    def __init__(self) -> None:
        self.requested = False
        self._callback: Callable[[], None] | None = None
        self._previous: Any = None
        self._installed = False

    def __enter__(self) -> _OperatorInterrupt:
        if threading.current_thread() is threading.main_thread():
            try:
                self._previous = signal.signal(signal.SIGINT, self._handle)
                self._installed = True
            except ValueError:  # signals unavailable (for example in an embedded interpreter)
                pass
        return self

    def __exit__(self, *exc: object) -> None:
        if self._installed:
            previous = self._previous if self._previous is not None else signal.default_int_handler
            signal.signal(signal.SIGINT, previous)
            self._installed = False

    def on_request(self, callback: Callable[[], None] | None) -> None:
        """Call ``callback`` on the first interrupt, at once if one was already requested."""
        self._callback = callback
        if callback is not None and self.requested:
            callback()

    def _handle(self, signum: int, frame: Any) -> None:
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        logger.warning("interrupted: stopping the controller, then settling jobs and freezing the experiment "
                       "(press Ctrl-C again to abort at once, which can leave the experiment open)")
        if self._callback is not None:
            self._callback()


class ReferenceRunner:
    """Runs one controller per run grant and finalizes its experiment (see the module docstring).

    ``setup`` names operator setup hooks (``module:function``) applied to
    the runner's registry and passed to the server and worker processes;
    ``worker_args`` are extra worker options; ``serve_env`` adds variables
    to the allow-listed server environment (never credentials);
    ``adapter_factory(config, experiment_id)`` overrides adapter creation
    (for example to supply a test double). With ``baseline``, the base
    checkpoint is evaluated on the task's full development suite before the
    controller starts (``Engine.ensure_baseline`` reuses an identical earlier
    evaluation when the engine provides it), and the instructions name that
    report.
    """

    def __init__(self, workspace: str | Path, *, registry: Registry | None = None, setup: Sequence[str] = (),
                 worker_args: Sequence[str] = (), serve_env: Mapping[str, str] | None = None,
                 adapter_factory: Callable[[AgentConfig, str], AgentAdapter] | None = None,
                 settle_timeout_seconds: float = 120.0, log_dir: str | Path | None = None,
                 baseline: bool = False) -> None:
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
        self.last_experiment_id: str | None = None
        self.baseline = bool(baseline)

    # -- entry point ---------------------------------------------------------------------

    def run_experiment(self, run_id: str, task_id: str, policy_id: str,
                       agent_config: StrategyConfig | AgentConfig | Mapping[str, Any]) -> str:
        """Run one controller on the run grant's experiment and freeze it; return the experiment ID.

        ``last_experiment_id`` names the experiment as soon as it exists, so a
        caller can report a run that raised. After an operator interrupt the
        experiment is settled and frozen first, then ``KeyboardInterrupt`` is
        raised.
        """
        config = parse_controller_config(agent_config)
        self.last_result = self.last_experiment_id = None
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
            self.last_experiment_id = eid
            policy = bound.experiment_policy(eid)
            task = bound.get_task(task_id)
        if self.baseline and policy.evaluation_available:
            self._evaluate_baseline(eid, task)
        if sandbox_available(policy):  # the agent's own code reads them too (wse.sandbox_runs.refresh_inputs)
            from wse.sandbox_runs import refresh_inputs

            with Engine(self.workspace, registry=self.registry) as operator:
                refresh_inputs(operator.experiments, eid)
        with _OperatorInterrupt() as interrupt:
            if isinstance(config, StrategyConfig):
                self._run_strategy(eid, task, policy, config, interrupt)
            else:
                self._run_agent(run_id, eid, task, policy, config, interrupt)
        if interrupt.requested:
            raise KeyboardInterrupt(f"experiment {eid} was interrupted by the operator; it is settled and frozen")
        return eid

    # -- records -----------------------------------------------------------------------------

    def _start_record(self, engine: Engine, eid: str, *, kind: str, ref: ExtensionRef, config: dict[str, Any],
                      constraint_status: str, tracing: dict[str, Any], constraint_reason: str | None = None) -> None:
        record_controller_start(engine.store.db, eid, kind=kind, controller_id=ref.extension_id,
                                controller_version=ref.version, config=config,
                                constraint_status=constraint_status, tracing=tracing,
                                constraint_reason=constraint_reason)

    def _end_record(self, engine: Engine, eid: str, result: AgentRunResult, stop_condition: str | None,
                    finalization: dict[str, Any] | None, error: str | None = None,
                    extra: Mapping[str, Any] | None = None) -> None:
        record = {**result.model_dump(mode="json"), "stop_condition": stop_condition,
                  "finalization": finalization, "environment": environment_record(), **(extra or {})}
        if error is not None:
            record["finalization_error"] = error
        record_controller_end(engine.store.db, eid, record)
        self.last_result = controller_record(engine.store.db, eid)

    def _settle_and_record(self, engine: Engine, eid: str, result: AgentRunResult, stop_condition: str | None,
                           extra: Mapping[str, Any] | None = None) -> None:
        """Finalize (settle jobs, freeze) and write the termination record, also when finalization fails."""
        try:
            finalization = self._finalize(engine, eid)
        except WSEError as exc:
            self._end_record(engine, eid, result, stop_condition, None, error=exc.message, extra=extra)
            raise
        self._end_record(engine, eid, result, stop_condition, finalization, extra=extra)

    # -- finalization ----------------------------------------------------------------------

    def _rule_selection(self, engine: Engine, eid: str) -> Any:
        """The task's selection rule applied to the experiment's full development reports (``wse.selection``)."""
        from wse.selection import select_by_rule

        experiment = engine.get_experiment(eid)
        try:
            task = engine.get_task(experiment.task_id)
        except WSEError:
            return None
        if task.objective.selection is None:
            return None
        reports = [r for r in StopMonitor(engine, eid, task)._full_development_reports() if not r.errors]
        checkpoints = engine.list_experiment_checkpoints(eid)
        changed = {}
        for checkpoint in checkpoints:
            if checkpoint.role == "candidate" and checkpoint.patch_id is not None:
                changed[checkpoint.checkpoint_id] = engine.get_patch(eid, checkpoint.patch_id).changed_elements
        return select_by_rule(task.objective.selection, base_checkpoint_id=experiment.base_checkpoint_id,
                              reports=reports, changed_elements=changed,
                              order=[c.checkpoint_id for c in checkpoints if c.role == "candidate"])

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
            decision = self._rule_selection(engine, eid)
            if decision is not None and decision.selected_checkpoint_id != selected:
                engine.select_checkpoint(eid, decision.selected_checkpoint_id, selected, f"runner-rule-{eid}")
                selected = decision.selected_checkpoint_id
            try:
                frozen = engine.freeze(eid, selected)
            except WSEError as exc:
                if exc.code != ErrorCode.VERSION_CONFLICT:
                    raise
                continue
            finalization = {"cancelled_jobs": cancelled, "state": frozen.state,
                            "frozen_checkpoint_id": frozen.selected_checkpoint_id}
            if decision is not None:
                finalization["selection"] = decision.model_dump(mode="json")
            return finalization
        raise WSEError(ErrorCode.VERSION_CONFLICT, f"experiment {eid} could not be frozen after three attempts")

    # -- stepwise strategies -----------------------------------------------------------------

    def _run_strategy(self, eid: str, task: TaskDefinition, policy: ExperimentPolicy,
                      config: StrategyConfig, interrupt: _OperatorInterrupt) -> None:
        from wse.worker import WorkerSupervisor

        ref = self.registry.resolve_reference("strategy", config.strategy)
        strategy = self.registry.resolve("strategy", ref.extension_id, ref.version)(**config.parameters)
        tools = allowed_tools(policy)
        recorded = {**config.model_dump(mode="json"), "strategy": str(ref), "resolved_config": strategy.config(),
                    "allowed_tools": tools}
        run_id = None
        supervisor = None
        constraint_status, constraint_reason = constraint_for(policy, True)
        with Engine(self.workspace, registry=self.registry) as operator:
            run_id = operator.get_experiment(eid).run_id
            self._start_record(operator, eid, kind="strategy", ref=ref, config=recorded,
                               constraint_status=constraint_status, constraint_reason=constraint_reason,
                               tracing={"sdk_tracing": "not applicable: no agent SDK runs"})
            monitor = StopMonitor(operator, eid, task)
            trace = _Trace(operator, eid)
            if jobs_available(policy):
                worker_args = [arg for spec in self.setup for arg in ("--setup", spec)] + list(self.worker_args)
                supervisor = WorkerSupervisor(self.workspace, extra_args=worker_args).start()
            try:
                try:
                    result, stop = self._strategy_loop(strategy, eid, task, config, tools, monitor, trace, run_id,
                                                       interrupt, policy)
                except Exception as exc:  # noqa: BLE001 - a failure inside the runner still settles and freezes
                    logger.exception("strategy run failed")
                    result, stop = AgentRunResult(status="error", reason=describe_exception(exc)), None
                except BaseException as exc:  # a repeated Ctrl-C or SystemExit: settle and freeze, then re-raise
                    self._settle_and_record(operator, eid, AgentRunResult(
                        status="cancelled", reason=f"interrupted: {describe_exception(exc)}"),
                        OPERATOR_INTERRUPT if interrupt.requested else None)
                    raise
                self._settle_and_record(operator, eid, result, stop)
            finally:
                if supervisor is not None:
                    supervisor.stop(stop_timeout=5.0)

    def _strategy_loop(self, strategy: Any, eid: str, task: TaskDefinition, config: StrategyConfig,
                       tools: Sequence[str], monitor: StopMonitor, trace: _Trace, run_id: str,
                       interrupt: _OperatorInterrupt, policy: ExperimentPolicy | None = None
                       ) -> tuple[AgentRunResult, str | None]:
        started = time.monotonic()
        trace.emit("run_started", details={"strategy": f"{strategy.strategy_id}@{strategy.version}",
                                           "max_steps": config.max_steps,
                                           "wall_clock_seconds": config.wall_clock_seconds})
        observations: list[Observation] = []
        status, reason, stop = "max_turns", f"the step limit of {config.max_steps} was reached", None
        steps = 0
        with Engine(self.workspace, registry=self.registry, run_id=run_id) as bound:
            executor = ExperimentExecutor(bound, constraint_status=constraint_for(policy, True)[0]
                                          if policy is not None else "verified", tools=tools)

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
                if interrupt.requested:
                    status, reason = "cancelled", OPERATOR_INTERRUPT
                    stop = OPERATOR_INTERRUPT
                    break
                if time.monotonic() - started > config.wall_clock_seconds:
                    status, reason = "timeout", f"the wall-clock limit of {config.wall_clock_seconds:g} s was reached"
                    break
                stop = monitor.observe(observations[-1].error_code)
                if stop is not None:
                    status, reason = "stopped", stop
                    break
                budget = bound.get_experiment(eid).budget
                try:
                    action = strategy.next_action(task.objective, list(observations), budget)
                except Exception as exc:  # noqa: BLE001 - a strategy failure still settles and freezes
                    logger.exception("strategy failed")
                    status, reason = "error", describe_exception(exc)
                    break
                steps += 1
                if interrupt.requested:  # an action proposed after the interrupt is not executed
                    status, reason = "cancelled", OPERATOR_INTERRUPT
                    stop = OPERATOR_INTERRUPT
                    break
                if action.kind == "stop":
                    status, reason = "completed", f"the strategy stopped: {action.reason}"
                    trace.emit("message", output=action.reason)
                    break
                shortfall = _action_shortfall(action, budget, generation=_generation_submission(bound, action),
                                              tokens_per_probe_text=_probe_tokens(policy))
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

    # -- agents --------------------------------------------------------------------------------

    def _make_adapter(self, config: AgentConfig, eid: str) -> AgentAdapter:
        if self.adapter_factory is not None:
            return self.adapter_factory(config, eid)
        ref = str(ExtensionRef.parse(config.adapter))
        if ref not in AGENT_ADAPTERS:
            raise WSEError(ErrorCode.NOT_FOUND, f"agent adapter {ref} is not available; known: {sorted(AGENT_ADAPTERS)}")
        import importlib

        module_name, class_name = AGENT_ADAPTERS[ref]
        adapter_class = getattr(importlib.import_module(module_name), class_name)
        if config.model.startswith(TEST_DOUBLE_PREFIX):  # the adapter's offline stand-in, driven by a strategy
            strategy_ref = self.registry.resolve_reference("strategy", config.model[len(TEST_DOUBLE_PREFIX):])
            strategy = self.registry.resolve("strategy", strategy_ref.extension_id, strategy_ref.version)()
            return adapter_class.test_double(strategy)
        return adapter_class()

    def serve_log(self, run_id: str) -> str:
        """The log file of the run's ``wse serve`` process and its worker (their stderr)."""
        return str(self.log_dir / f"serve-{run_id}.log")

    def connection(self, run_id: str, eid: str, task: TaskDefinition, policy: ExperimentPolicy,
                   constraint_status: str) -> RunConnection:
        """The MCP connection for a run: ``wse serve`` bound to the grant, allow-listed tools and environment.

        Under a task with a selection rule the runner chooses the checkpoint
        to freeze, so ``wse_select_checkpoint`` is not offered.
        """
        tools = allowed_tools(policy)
        if task.objective.selection is not None:
            tools = [tool for tool in tools if tool != "wse_select_checkpoint"]
        self.log_dir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "wse.cli", "--workspace", str(self.workspace), "serve", "--run-id", run_id,
                   "--constraint-status", constraint_status, "--log-file", self.serve_log(run_id)]
        for spec in self.setup:
            command += ["--setup", spec]
        if self.worker_args:
            command += ["--worker-args", " ".join(self.worker_args)]
        return RunConnection(
            workspace=str(self.workspace), run_id=run_id, experiment_id=eid, task_id=task.task_id,
            policy_id=policy.policy_id, serve_command=command, serve_env=serve_environment(self.serve_env),
            cwd=str(self.workspace), allowed_tools=tools,
            instructions=render_instructions(task, policy, experiment_id=eid, tools=tools,
                                             baseline=self._baseline_summary(eid, task)),
            initial_message=initial_message(eid), constraint_status=constraint_status)

    def _evaluate_baseline(self, eid: str, task: TaskDefinition) -> None:
        """Evaluate the base checkpoint on the full development suite (``baseline``); freeze the experiment if it fails.

        One evaluation of the policy's budget: one generation pass or batch.
        """
        from wse.worker import WorkerSupervisor

        if self._baseline_summary(eid, task) is not None:
            return
        worker_args = [arg for spec in self.setup for arg in ("--setup", spec)] + list(self.worker_args)
        supervisor = WorkerSupervisor(self.workspace, extra_args=worker_args).start()
        try:
            with Engine(self.workspace, registry=self.registry) as operator:
                try:
                    ensure = getattr(operator, "ensure_baseline", None)
                    if ensure is not None:
                        ensure(eid, device=_worker_device(self.worker_args))
                        return
                    suite = operator.get_suite(task.development_suite_id)
                    config = {"sample_count": max(suite.sample_counts), "seed": suite.seeds[0]}
                    base = operator.get_experiment(eid).base_checkpoint_id
                    job = operator.submit_evaluation(eid, base, suite.suite_id, config, f"runner-baseline-{eid}")
                    while not job.terminal:
                        job = operator.get_job(eid, job.job_id, 10)
                    if job.state != "succeeded":
                        raise WSEError(ErrorCode.WORKER_FAILED, f"the baseline evaluation ended {job.state}",
                                       details={"job_id": job.job_id})
                except BaseException:
                    self._finalize(operator, eid)  # an interrupted or failed baseline still settles and freezes
                    raise
        finally:
            supervisor.stop(stop_timeout=5.0)

    def _baseline_summary(self, eid: str, task: TaskDefinition) -> dict[str, Any] | None:
        """The base checkpoint's first full development report of the experiment, if there is one."""
        with Engine(self.workspace, registry=self.registry) as operator:
            base = operator.get_experiment(eid).base_checkpoint_id
            for report in StopMonitor(operator, eid, task)._full_development_reports():
                if report.checkpoint_id == base and not report.errors:
                    return {"report_id": report.report_id, "metrics": dict(report.metrics)}
        return None

    def _run_agent(self, run_id: str, eid: str, task: TaskDefinition, policy: ExperimentPolicy,
                   config: AgentConfig, interrupt: _OperatorInterrupt) -> None:
        timeout = agent_session_timeout(policy, config.client_session_timeout_seconds)
        if timeout != config.client_session_timeout_seconds:
            config = config.model_copy(update={"client_session_timeout_seconds": timeout})
        adapter = self._make_adapter(config, eid)
        constraint_status, constraint_reason = constraint_for(policy, config.benchmark)
        connection = self.connection(run_id, eid, task, policy, constraint_status)
        serve_log = self.serve_log(run_id)
        recorded = {**config.model_dump(mode="json"), "allowed_tools": connection.allowed_tools,
                    "serve_env_names": sorted(connection.serve_env), "cwd": connection.cwd}
        strategy = getattr(getattr(adapter, "model", None), "strategy", None)
        if strategy is not None:  # a strategy test double: record the strategy that chooses the calls
            resolve = getattr(strategy, "config", None)
            recorded["test_double"] = {"strategy": f"{strategy.strategy_id}@{strategy.version}",
                                       "resolved_config": resolve() if callable(resolve) else None}
        tracing = {"sdk_tracing_disabled": config.tracing_disabled, "benchmark": config.benchmark,
                   "trace_include_sensitive_data": not config.tracing_disabled and not config.benchmark,
                   "server_env_OPENAI_AGENTS_DISABLE_TRACING": "1"}
        ref = ExtensionRef.parse(config.adapter)

        async def main() -> None:
            with Engine(self.workspace, registry=self.registry) as operator:
                self._start_record(operator, eid, kind="agent", ref=ref, config=recorded,
                                   constraint_status=constraint_status, constraint_reason=constraint_reason,
                                   tracing=tracing)
                monitor = StopMonitor(operator, eid, task)
                trace = _Trace(operator, eid)
                stops: list[str] = []
                controller_running = [True]

                def request_stop(reason: str) -> None:  # runs in the event loop
                    if stops or not controller_running[0]:
                        return
                    stops.append(reason)
                    trace.emit("cancel_requested", details={"reason": reason})
                    adapter.cancel(eid, reason)

                def sink(event: AgentEvent) -> None:
                    trace.record(event)
                    if event.kind != "tool_result" or stops:
                        return
                    reason = monitor.observe(event.error_code)
                    if reason is not None:
                        request_stop(reason)

                loop = asyncio.get_running_loop()
                interrupt.on_request(lambda: loop.call_soon_threadsafe(request_stop, OPERATOR_INTERRUPT))
                escaped: BaseException | None = None
                try:
                    try:
                        await adapter.start(connection, config)
                        if stops:  # requested before the session existed
                            adapter.cancel(eid, stops[0])
                        result = await adapter.run(eid, config, sink)
                    except Exception as exc:  # noqa: BLE001 - an adapter failure still settles and freezes
                        logger.exception("agent adapter failed")
                        result = AgentRunResult(status="error", reason=describe_exception(exc))
                    except BaseException as exc:  # a repeated Ctrl-C, or this task cancelled: settle, then re-raise
                        escaped = exc
                        task_ = asyncio.current_task()
                        while task_ is not None and task_.cancelling():
                            task_.uncancel()  # let the finalization and the MCP cleanup below run
                        result = AgentRunResult(status="cancelled", reason=f"interrupted: {describe_exception(exc)}")
                    finally:
                        controller_running[0] = False  # no stop request or trace write after the controller ends
                        interrupt.on_request(None)
                    if stops and stops[0] == OPERATOR_INTERRUPT and result.status == "stopped":
                        result = result.model_copy(update={"status": "cancelled"})
                    if result.status in ("connection_failed", "error"):
                        result = result.model_copy(update={"reason": f"{result.reason[:1500]}; see the server log "
                                                                     f"{serve_log}"})
                    await asyncio.to_thread(self._settle_and_record, operator, eid, result,
                                            stops[0] if stops else None, {"serve_log": serve_log})
                finally:
                    await adapter.close(eid)
                if escaped is not None:
                    raise escaped

        asyncio.run(main())


def run_experiment(run_id: str, task_id: str, policy_id: str,
                   agent_config: StrategyConfig | AgentConfig | Mapping[str, Any], *, workspace: str | Path,
                   **runner_options: Any) -> str:
    """Run one controller on a run grant's experiment and freeze it (see ``ReferenceRunner``)."""
    return ReferenceRunner(workspace, **runner_options).run_experiment(run_id, task_id, policy_id, agent_config)


# ---------------------------------------------------------------------------
# Trial summaries (Section 7.2: agent acceptance is reported separately)
# ---------------------------------------------------------------------------

SUCCESS_ACCURACY = 0.99


def summarize_trial(engine: Engine, experiment_id: str, *, threshold: float = SUCCESS_ACCURACY) -> dict[str, Any]:
    """One trial's record for reporting: controller, termination, costs, changed fraction, and final result.

    ``success`` is ``None`` until the experiment is closed by final
    evaluation; then it is whether the frozen candidate's final accuracy is
    at least ``threshold``. ``failure_cause`` explains every unsuccessful
    trial from the controller's termination record and the final result.
    """
    engine._require_operator("summarizing a trial")
    experiment = engine.get_experiment(experiment_id)
    record = controller_record(engine.store.db, experiment_id) or {}
    result = record.get("result") or {}
    usage = result.get("usage") or {}
    checkpoints = engine.list_experiment_checkpoints(experiment_id)
    selected = next(c for c in checkpoints if c.checkpoint_id == experiment.selected_checkpoint_id)
    parameters = engine.load_manifest(selected.checkpoint_id).parameter_count
    changed = engine.get_patch(experiment_id, selected.patch_id).changed_elements if selected.patch_id else 0
    config = record.get("config") or {}
    row: dict[str, Any] = {
        "experiment_id": experiment_id, "run_id": experiment.run_id, "policy_id": experiment.policy_id,
        "mode": experiment.mode, "state": experiment.state,
        "controller": None if not record else f"{record['controller']['extension_id']}@"
                                              f"{record['controller']['version']}",
        "model": config.get("model"), "status": result.get("status"), "reason": result.get("reason"),
        "stop_condition": result.get("stop_condition"),
        "candidates": sum(1 for c in checkpoints if c.role == "candidate"),
        "evaluated_examples": experiment.budget["evaluated_examples"].consumed,
        "compute_seconds": round(float(experiment.budget["compute_seconds"].consumed), 6),
        "elapsed_seconds": result.get("elapsed_seconds"), "turns": result.get("turns"),
        "tool_calls": result.get("tool_calls"),
        "tokens": {name: usage.get(name) for name in ("input_tokens", "output_tokens", "total_tokens")},
        "tokens_unavailable": {name: reason for name, reason in (usage.get("unavailable") or {}).items()
                               if name in ("input_tokens", "output_tokens", "total_tokens")},
        "changed_elements": changed, "parameter_count": parameters,
        "changed_fraction": changed / parameters if parameters else 0.0,
        "constraint_status": record.get("constraint_status"), "constraint_reason": record.get("constraint_reason"),
        "tracing": record.get("tracing"), "sandbox": None,
        "final": None, "success": None, "failure_cause": None,
    }
    if sandbox_available(engine.experiment_policy(experiment_id)):
        row["sandbox"] = {"calls": int(experiment.budget["sandbox_calls"].consumed),
                          "wall_seconds": float(experiment.budget["sandbox_seconds"].consumed),
                          "label": SANDBOX_TIME_LABEL}
    if experiment.state == "CLOSED":
        final = engine.get_final_report(experiment_id)
        comparison = final.comparison
        row["final"] = {"candidate_accuracy": comparison.candidate_value, "control_accuracy": comparison.base_value,
                        "difference": comparison.difference,
                        "fully_correct_clusters": comparison.candidate_fully_correct.successes,
                        "clusters": comparison.candidate_fully_correct.trials,
                        "clopper_pearson_lower": comparison.candidate_fully_correct.lower,
                        "clopper_pearson_upper": comparison.candidate_fully_correct.upper}
        row["success"] = comparison.candidate_value >= threshold
        if not row["success"]:
            row["failure_cause"] = (f"final accuracy {comparison.candidate_value:g} is below {threshold:g}; the "
                                    f"controller ended {row['status']}: {row['reason']}")
    elif experiment.state == "OPEN":
        row["failure_cause"] = "the experiment was not frozen: " + str(result.get("finalization_error") or
                                                                       "the controller did not finish")
    return row


def _trial_name(row: Mapping[str, Any]) -> str:
    """The experiment ID, or the run grant of a trial that failed before its experiment existed."""
    return row["experiment_id"] or f"run {row['run_id']} (no experiment)"


def render_trials_markdown(rows: Sequence[Mapping[str, Any]], *, threshold: float = SUCCESS_ACCURACY) -> str:
    """A concise Markdown report of trials: a table, the success count, and every failure cause."""
    closed = [row for row in rows if row["success"] is not None]
    lines = ["# Trial report", "",
             f"{sum(1 for row in closed if row['success'])}/{len(rows)} trials found a candidate with final accuracy "
             f">= {threshold:.0%}" + ("" if len(closed) == len(rows) else f" ({len(rows) - len(closed)} not yet "
                                       "final-evaluated)") + ".", "",
             "| Experiment | Controller | Model | Policy | Status | Candidates | Evaluated examples | Elapsed s | "
             "Tokens (in/out) | Changed fraction | Final accuracy (control) | CP 95% lower |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        tokens = row["tokens"]
        token_text = ("n/a" if tokens["input_tokens"] is None
                      else f"{tokens['input_tokens']}/{tokens['output_tokens']}")
        final = row["final"]
        final_text = (("pending" if row["success"] is None else "n/a") if final is None
                      else f"{final['candidate_accuracy']:.4f} ({final['control_accuracy']:.4f})")
        lower = "" if final is None else f"{final['clopper_pearson_lower']:.4f}"
        elapsed = "" if row["elapsed_seconds"] is None else f"{row['elapsed_seconds']:.1f}"
        lines.append(f"| {_trial_name(row)} | {row['controller'] or ''} | {row['model'] or ''} | {row['policy_id']} | "
                     f"{row['status'] or 'not recorded'} | {row['candidates']} | {row['evaluated_examples']} | "
                     f"{elapsed} | {token_text} | {row['changed_fraction']:.4f} | {final_text} | {lower} |")
    lines += ["", "Termination and failure causes:", ""]
    for row in rows:
        cause = row["failure_cause"] or "success"
        ending = f"{row['status']} ({row['reason']})" if row["status"] else "no termination record"
        lines.append(f"- {_trial_name(row)}: {ending}; {cause}")
    unavailable = {reason for row in rows for reason in row["tokens_unavailable"].values()}
    if unavailable:
        lines += ["", "Token counts reported as n/a: " + "; ".join(sorted(unavailable)) + "."]
    sandboxed = [row for row in rows if row.get("sandbox")]
    if sandboxed:
        lines += ["", "Evaluated examples, candidates, and costs above are WSE-metered calls. Runs that executed "
                      f"their own code also used {SANDBOX_TIME_LABEL}, which WSE does not meter; these runs are "
                      "unverified:", ""]
        lines += [f"- {_trial_name(row)}: {row['sandbox']['calls']} sandbox calls, "
                  f"{row['sandbox']['wall_seconds']:.1f} s" for row in sandboxed]
    return "\n".join(lines) + "\n"
