"""One budgeted weight-editing experiment in one call (operator use).

``run_agent_experiment`` performs the operator sequence of a single trial
on a prepared workspace: it issues a run grant (``Engine.grant_run``), runs
one controller on the grant's experiment through ``ReferenceRunner`` (an LLM
through the ``openai_agents@1`` adapter and the MCP server, a registered
stepwise strategy, or any controller mapping the runner accepts), and, when
asked, runs the hidden final evaluation of the frozen experiment. It then
summarizes the experiment (``wse.runner.summarize_trial``), exports a closed
experiment, and replays the export in a fresh workspace to check
byte-identical weights. The final evaluation is off by default, because a
series of trials evaluates only after all of them have frozen.
Errors from the engine and the runner propagate unchanged; an experiment
that does not reach ``CLOSED`` is returned in the state it ended in,
without export or replay.

Everything else stays with the lower-level API (``wse.api.Engine``): task,
dataset, and suite registration, base-checkpoint import, policy
registration, the audit log, patches, and exports of other experiments.

A live ``model`` uses the SDK's credentials from the process environment;
this module takes no key and reads no ``.env``. As with
``examples/signed_sum/run_trials.py``, an agent run sets
``OPENAI_AGENTS_DISABLE_TRACING=1`` in the process environment unless the
variable is already set, so the SDK exports no traces of the run.

The wrapper has been used on M1's small-model tasks only. Managed
experiments on open-weight language models run through the same runner and
MCP tools (M2 Q2), but exporting and replaying a whole managed experiment
carries M1's small models only (``docs/models.md``, Known limits); patches
of open-weight models replay exactly.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wse.api import Engine
from wse.runner import ReferenceRunner, parse_controller_config, summarize_trial
from wse.schemas import ErrorCode, WSEError

__all__ = ["AgentRun", "run_agent_experiment"]

OUT_DIR_NAME = "agent_runs"


@dataclass(frozen=True)
class AgentRun:
    """The outcome of one ``run_agent_experiment`` call.

    ``state`` is the experiment's state at the end and ``summary`` the row of
    ``summarize_trial``. ``export_dir`` and ``replay_identical`` are ``None``
    unless the closed experiment was exported and its export replayed.
    ``out_dir`` holds the runner's logs (``logs``: an agent run's server
    log) and, when written, ``export`` and ``replay``.
    """

    run_id: str
    experiment_id: str
    state: str
    summary: dict[str, Any]
    out_dir: Path
    export_dir: Path | None = None
    replay_identical: bool | None = None

    @property
    def success(self) -> bool | None:
        """Whether the final accuracy meets the success threshold; ``None`` until the experiment is closed."""
        return self.summary["success"]

    @property
    def final(self) -> dict[str, Any] | None:
        """The final-evaluation comparison of the summary; ``None`` until the experiment is closed."""
        return self.summary["final"]


def _controller_config(model: str | None, strategy: str | None, controller: Mapping[str, Any] | None,
                       max_turns: int | None, wall_clock_seconds: float | None) -> tuple[Mapping[str, Any], str]:
    """The controller mapping for the runner, validated by ``parse_controller_config``, and its kind."""
    if sum(option is not None for option in (model, strategy, controller)) != 1:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "give exactly one of model (an LLM controller through "
                       "openai_agents@1), strategy (a registered stepwise strategy), or controller (a complete "
                       "controller mapping)")
    limits = {name: value for name, value in (("max_turns", max_turns), ("wall_clock_seconds", wall_clock_seconds))
              if value is not None}
    if controller is not None:
        if limits:
            raise WSEError(ErrorCode.INVALID_ARGUMENT, "with controller, give max_turns and wall_clock_seconds "
                           "inside the controller mapping")
        config: Mapping[str, Any] = controller
    elif model is not None:
        config = {"kind": "agent", "model": model, **limits}
    else:
        config = {"kind": "strategy", "strategy": strategy, **limits}
    return config, parse_controller_config(config).kind


def _require_empty(path: Path, what: str) -> None:
    if path.exists() and any(path.iterdir()):
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"{what} directory {path} is not empty; choose a new out_dir")


def run_agent_experiment(workspace: str | Path, *, task: str, base: str, policy: str, model: str | None = None,
                         strategy: str | None = None, controller: Mapping[str, Any] | None = None,
                         max_turns: int | None = None, wall_clock_seconds: float | None = None,
                         out_dir: str | Path | None = None, final_evaluation: bool = False,
                         verify_replay: bool = True) -> AgentRun:
    """Grant and run one experiment, then summarize, export, and replay it (see the module docstring).

    ``workspace`` must already hold the task, its suites, the policy, and
    the base checkpoint ``base``. Exactly one of ``model``, ``strategy``, and
    ``controller`` selects the controller; ``max_turns`` and
    ``wall_clock_seconds`` override the controller's defaults for ``model``
    and ``strategy``. ``out_dir`` (default ``<workspace>/agent_runs/<run_id>``)
    receives the runner's logs, the export (``export``), and the replay
    workspace (``replay``); both must be absent or empty. Export and replay
    run when the experiment ends ``CLOSED`` and ``verify_replay`` is true.

    The hidden final evaluation runs only with ``final_evaluation=True``, when
    the experiment ends ``FROZEN``; use it for a single experiment. A series of
    trials runs its final evaluations only after every trial has frozen, so
    that no trial can adapt to an earlier trial's final result (M1 plan
    Section 7.3): run the trials, then ``Engine.final_evaluate`` each one.
    """
    config, kind = _controller_config(model, strategy, controller, max_turns, wall_clock_seconds)
    workspace = Path(workspace).resolve()
    out = None if out_dir is None else Path(out_dir).resolve()
    if out is not None and verify_replay:
        _require_empty(out / "export", "export")
        _require_empty(out / "replay", "replay")
    if kind == "agent":
        os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")
    with Engine(workspace) as engine:
        run_id = engine.grant_run(task, policy, base).run_id
    if out is None:
        out = workspace / OUT_DIR_NAME / run_id
    experiment_id = ReferenceRunner(workspace, log_dir=out / "logs").run_experiment(run_id, task, policy, config)
    with Engine(workspace) as engine:
        if final_evaluation and engine.get_experiment(experiment_id).state == "FROZEN":
            engine.final_evaluate(experiment_id)
        summary = summarize_trial(engine, experiment_id)
        if summary["state"] != "CLOSED" or not verify_replay:
            return AgentRun(run_id, experiment_id, summary["state"], summary, out)
        export_dir, replay_dir = out / "export", out / "replay"
        _require_empty(replay_dir, "replay")
        engine.export_experiment(experiment_id, export_dir)
    with Engine(replay_dir) as clean:
        identical = clean.replay_export(export_dir).all_identical
    return AgentRun(run_id, experiment_id, summary["state"], summary, out, export_dir, identical)
