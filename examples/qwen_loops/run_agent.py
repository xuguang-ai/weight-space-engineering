"""Run one LLM agent on Qwen3.5-0.8B's reasoning loops, with code execution (operator use only; M3 plan Task 9).

The live pipeline check of M3's stage S2. An OpenAI Agents SDK agent
(``--model``, default ``gpt-5.6-luna``) works on the existing Qwen loop task
through a real ``wse serve`` MCP session under ``--policy`` (default
``qwen_loop_sandbox_v1``): generation passes and probes run as WSE jobs that
it waits for in long waits, its own Python runs in the sandbox on the run's
inputs, and its edits are exact patches of the registered operators. The
agent is told only the task and the policy: its instructions are generated
from them, and the sandbox reads its run's inputs and the analysis libraries
only, so nothing about p50c or the article reaches it.

Each invocation:

1. prepares the workspace idempotently (``loop_setup.prepare``), as the
   other Qwen loop scripts do: the snapshot, the practice suite of
   ``--problems`` GSM8K-train problems chosen by ``--subset-seed``, and its task;
2. grants a run under ``--policy`` and runs the agent through the reference
   runner, which starts one worker on ``--device`` and settles and freezes
   the experiment however the run ends (the first Ctrl-C stops the agent;
   a second aborts at once);
3. writes ``OUT/runs/<run_id>.json``: the setup, the agent's configuration,
   the trial summary (``summarize_trial``: tokens, WSE-metered calls, and
   sandbox time apart), and the experiment summary
   (``loop_setup.experiment_summary``: every step, job, report, candidate,
   comparison, and the budget); then exports the experiment to
   ``OUT/export/<run_id>`` and replays it into ``OUT/replay/<run_id>``,
   which must rebuild every candidate byte for byte.

No final evaluation runs: the loop task registers no final suite (the
preregistered trials of stage S3 bring one).

The agent's credentials come from the environment (``OPENAI_API_KEY``, and
``OPENAI_BASE_URL`` for an Azure OpenAI resource); load them for this
command only. SDK tracing is disabled. ``--truncation auto`` (the default)
lets the provider drop the oldest conversation items rather than fail when
a long run outgrows the model's context. Keep the Mac awake::

    caffeinate -is uv run --extra models-hf --extra agents-openai --env-file .env \\
        python examples/qwen_loops/run_agent.py ~/Desktop/wse-models/runs/m3-s2 \\
        --workspace ~/Desktop/wse-models/workspaces/m3-s2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import loop_setup  # noqa: E402 - the shared setup next to this script

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_POLICY = "qwen_loop_sandbox_v1"


def _error_text(exc: BaseException) -> str:
    from wse.schemas import WSEError

    if isinstance(exc, WSEError):
        return f"{exc.code.value}: {exc.message}"
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="the output directory for records, logs, exports, and replays")
    loop_setup.add_setup_arguments(parser)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"the agent's model (default {DEFAULT_MODEL})")
    parser.add_argument("--policy", default=DEFAULT_POLICY, choices=("qwen_loop_sandbox_v1", "qwen_loop_v1"),
                        help=f"the run's policy (default {DEFAULT_POLICY})")
    parser.add_argument("--max-turns", type=int, default=400, help="the SDK's turn limit (model calls)")
    parser.add_argument("--wall-clock-hours", type=float, default=7.0, help="the agent's wall-clock limit")
    parser.add_argument("--truncation", choices=("auto", "disabled"), default="auto",
                        help="what the provider does when the conversation outgrows the context (default auto)")
    args = parser.parse_args(argv)
    loop_setup.check_setup_arguments(parser, args)
    if args.via_mcp:
        parser.error("--via-mcp is implied: the agent always works through a wse serve MCP session")
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is not set; load it for this command only, for example with uv run --env-file")
    os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

    from wse.agent_adapters.base import AgentConfig
    from wse.api import Engine
    from wse.audit import export_experiment, replay_export
    from wse.policy import sandbox_available
    from wse.registry import builtin_registry
    from wse.runner import ReferenceRunner, summarize_trial
    from wse.sandbox import available

    out, workspace = args.out.expanduser(), args.workspace.expanduser()
    with Engine(workspace) as engine:
        policy = engine.get_policy(args.policy)
        if sandbox_available(policy):
            ok, reason = available()
            if not ok:
                parser.error(f"{args.policy} grants code execution, which is not available here: {reason}")
        setup = loop_setup.prepare(engine, args)
        grant = engine.grant_run(setup["task_id"], args.policy, setup["base_checkpoint_id"])
    config = AgentConfig(model=args.model, max_turns=args.max_turns, wall_clock_seconds=args.wall_clock_hours * 3600,
                         model_settings={"truncation": args.truncation})
    record = {"script": "examples/qwen_loops/run_agent.py", "workspace": str(workspace), "setup": setup,
              "policy_id": args.policy, "run_id": grant.run_id, "agent_config": config.model_dump(mode="json"),
              "started_at": _stamp()}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "runs.jsonl").open("a", encoding="utf-8") as index:
        index.write(json.dumps({"run_id": grant.run_id, "started_at": record["started_at"], "model": args.model,
                                "policy_id": args.policy, "suite_id": setup["suite_id"]}) + "\n")
    loop_setup.write_json(out / "runs" / f"{grant.run_id}.json", record)
    print(f"run {grant.run_id}: {args.model} under {args.policy} on {setup['base_checkpoint_id']}, suite "
          f"{setup['suite_id']}, device {args.device}", file=sys.stderr)
    runner = ReferenceRunner(workspace, registry=builtin_registry(), worker_args=["--device", args.device],
                             log_dir=out / "logs")
    status, error = 0, None
    started = time.monotonic()
    try:
        runner.run_experiment(grant.run_id, setup["task_id"], args.policy, config)
    except KeyboardInterrupt as exc:
        status, error = 130, _error_text(exc)
    except Exception as exc:  # noqa: BLE001 - the record of a failed run is still written
        status, error = 1, _error_text(exc)
    record.update(elapsed_seconds=round(time.monotonic() - started, 1), error=error, finished_at=_stamp())
    experiment_id = runner.last_experiment_id
    if experiment_id is None:
        loop_setup.write_json(out / "runs" / f"{grant.run_id}.json", record)
        print(f"run {grant.run_id} failed before its experiment existed: {error}", file=sys.stderr)
        return status or 1
    with Engine(workspace) as engine:
        record["trial"] = summarize_trial(engine, experiment_id)
        record["experiment"] = loop_setup.experiment_summary(engine, experiment_id)
        loop_setup.write_json(out / "runs" / f"{grant.run_id}.json", record)
        export_dir = out / "export" / grant.run_id
        try:
            manifest = export_experiment(engine, experiment_id, export_dir)
            record["export"] = {"path": str(export_dir), "files": len(manifest.files),
                                "sandbox_calls": len((manifest.sandbox or {}).get("calls", []))}
        except Exception as exc:  # noqa: BLE001 - recorded, and the replay is skipped
            record["export"] = {"path": str(export_dir), "error": _error_text(exc)}
    if "error" not in record["export"]:
        try:
            with Engine(out / "replay" / grant.run_id) as clean:
                replayed = replay_export(export_dir, clean)
            record["replay"] = {"all_identical": replayed.all_identical, "candidates": len(replayed.candidates)}
        except Exception as exc:  # noqa: BLE001
            record["replay"] = {"error": _error_text(exc)}
    loop_setup.write_json(out / "runs" / f"{grant.run_id}.json", record)
    result = (record["experiment"]["controller"] or {}).get("result") or {}
    print(json.dumps({"experiment_id": experiment_id, "state": record["experiment"]["state"],
                      "status": result.get("status"), "reason": result.get("reason"),
                      "selected_checkpoint_id": record["experiment"]["selected_checkpoint_id"],
                      "export": record["export"], "replay": record.get("replay"),
                      "record": str(out / "runs" / f"{grant.run_id}.json"), "error": error}, indent=2))
    return status


if __name__ == "__main__":
    sys.exit(main())
