"""Run the scripted discovery ``direction_search@1`` on Qwen3.5-0.8B (operator use only; M2 plan Task 12).

The discovery uses only the agent-facing tools and no oracle value: a
generation pass of the base on practice problems, a probe of the outputs that
reached the token limit against those that finished at the input and output
of ``mlp.down`` in every layer, a ``project_out`` of the contrast direction from
the most aligned columns of the layer that separates them most (one candidate
per alpha), and a generation pass of each candidate, then a selection
(``wse.strategies.direction_search``). The layer, the direction, and the
columns come from the run's own pass, probe, and analysis; the edit's form
and its size defaults (64 columns, alphas 1 and 6) come from p50c's
published edit, so the run tests whether the tools can localize an edit of
p50c's shape from behavior. Its outcome is reported whether or not it
improves on the base.

Each invocation:

1. prepares the workspace idempotently (``loop_setup.prepare``): imports the
   snapshot, registers the practice suite of ``--problems`` GSM8K-train
   problems chosen by ``--subset-seed`` and its task;
2. grants a run under ``qwen_loop_v1`` and runs the strategy through the
   reference runner, which starts one worker on ``--device`` and settles and
   freezes the experiment however the run ends (the first Ctrl-C stops the
   strategy at its next step; a second aborts at once). With ``--via-mcp``,
   which the M2 plan's Q2 exit asks for, the OpenAI Agents SDK drives the
   managed tools through a real ``wse serve`` MCP session and the strategy
   chooses every call in place of a model (the runner's test double: no
   network, no credentials); without it the runner dispatches the
   strategy's actions to the run-bound engine. The operations are the same;
3. writes ``OUT/runs/<run_id>.json`` (and a copy as ``OUT/summary.json``):
   the setup, the strategy (its path, ``dispatch`` or ``mcp``, its
   parameters, and its resolved configuration), the strategy's final
   message, every step with its rationale and outcome, every job, report,
   and output, the probe's sites with their separations, the candidates and
   their patches, paired comparisons with the base, the selection, and the
   budget. ``OUT/runs.jsonl`` lists every run.

A run grant permits one experiment, so an interrupted run is not resumed:
invoke the script again and it grants a new run, after the same idempotent
setup; earlier runs and their summaries stay as they are. ``--summarize
EXPERIMENT_ID`` updates the summary of an existing experiment in its run's
record without running anything (for example after the process was killed);
the record's other keys stay.

Default: 32 problems at 8,192 tokens, three passes, about 2 hours on the
operator's MPS. Keep the Mac awake (``caffeinate -is``)::

    caffeinate -is uv run --extra models-hf --extra agents-openai python examples/qwen_loops/run_discovery.py OUT \\
        --workspace ~/Desktop/wse-models/workspaces/q2 \\
        --snapshot ~/Desktop/wse-models/Qwen3.5-0.8B --repo Qwen/Qwen3.5-0.8B \\
        --revision 2fc06364715b967f1860aea9cf38778875588b17 --via-mcp

A short smoke run: ``--problems 4 --max-new-tokens 256``. Strategy options:
``--columns``, ``--alphas``, ``--layers``, ``--last``, and ``--column-rule``
(defaults: 64, ``1.0 6.0``, every layer, the whole output, ``cosine``).
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

STRATEGY = "direction_search@1"


def _error_text(exc: BaseException) -> str:
    from wse.schemas import WSEError

    if isinstance(exc, WSEError):
        return f"{exc.code.value}: {exc.message}"
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


def _parameters(args: argparse.Namespace, suite_id: str) -> dict:
    parameters: dict = {"suite_id": suite_id}
    for name in ("columns", "alphas", "layers", "last", "column_rule"):
        value = getattr(args, name)
        if value is not None:
            parameters[name] = value
    return parameters


def _summarize(out: Path, workspace: Path, experiment_id: str, extra: dict) -> dict:
    """The experiment's summary and ``extra`` merged into its run's record (other keys stay), copied to summary.json."""
    from wse.api import Engine

    with Engine(workspace) as engine:
        experiment = loop_setup.experiment_summary(engine, experiment_id)
    result = (experiment["controller"] or {}).get("result") or {}
    record = loop_setup.merge_json(out / "runs" / f"{experiment['run_id']}.json",
                                   {**extra, "final_message": result.get("final_message"), "experiment": experiment})
    loop_setup.write_json(out / "summary.json", record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="the output directory for summaries and logs")
    loop_setup.add_setup_arguments(parser)
    parser.add_argument("--columns", type=int, default=None, help="columns to edit (default 64)")
    parser.add_argument("--alphas", type=float, nargs="+", default=None, help="one candidate per alpha (default 1 6)")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="layers to probe (default: every layer)")
    parser.add_argument("--last", type=int, default=None, help="probe window: each output's last N tokens")
    parser.add_argument("--column-rule", choices=("cosine", "projection", "input_contrast"), default=None)
    parser.add_argument("--wall-clock-hours", type=float, default=6.5, help="the controller's wall-clock limit")
    parser.add_argument("--max-steps", type=int, default=5000, help="the controller's step limit (job polls count)")
    parser.add_argument("--summarize", metavar="EXPERIMENT_ID", default=None,
                        help="only update the summary of this experiment in its run's record; run nothing")
    args = parser.parse_args(argv)
    loop_setup.check_setup_arguments(parser, args)
    os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

    from wse.api import Engine
    from wse.registry import builtin_registry
    from wse.runner import ReferenceRunner
    from wse.strategies.direction_search import DirectionSearchStrategy

    out, workspace = args.out.expanduser(), args.workspace.expanduser()
    if args.summarize:
        summary = _summarize(out, workspace, args.summarize, {"summarized_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        print(json.dumps({"summary": str(out / "runs" / f"{summary['experiment']['run_id']}.json"),
                          "state": summary["experiment"]["state"], "final_message": summary["final_message"]}))
        return 0
    try:
        DirectionSearchStrategy(**_parameters(args, "suite"))  # refuse bad options before anything is granted
    except ValueError as exc:
        parser.error(str(exc))
    with Engine(workspace) as engine:
        setup = loop_setup.prepare(engine, args)
        grant = engine.grant_run(setup["task_id"], setup["policy_id"], setup["base_checkpoint_id"])
    parameters = _parameters(args, setup["suite_id"])
    registry = builtin_registry()
    config = loop_setup.controller(registry, DirectionSearchStrategy, parameters, via_mcp=args.via_mcp,
                                   max_steps=args.max_steps, wall_clock_seconds=args.wall_clock_hours * 3600)
    strategy = loop_setup.strategy_record(DirectionSearchStrategy, parameters, config)
    record = {"script": "examples/qwen_loops/run_discovery.py", "workspace": str(workspace), "setup": setup,
              "run_id": grant.run_id, "strategy": strategy, "controller_config": config.model_dump(mode="json"),
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "runs.jsonl").open("a", encoding="utf-8") as index:
        index.write(json.dumps({"run_id": grant.run_id, "started_at": record["started_at"],
                                "suite_id": setup["suite_id"], "parameters": parameters,
                                "path": strategy["path"]}) + "\n")
    print(f"run {grant.run_id}: {strategy['controller']} ({STRATEGY}, {strategy['path']}) on "
          f"{setup['base_checkpoint_id']}, suite {setup['suite_id']}, device {args.device}", file=sys.stderr)
    runner = ReferenceRunner(workspace, registry=registry, worker_args=["--device", args.device],
                             log_dir=out / "logs")
    status, error = 0, None
    started = time.monotonic()
    try:
        runner.run_experiment(grant.run_id, setup["task_id"], setup["policy_id"], config)
    except KeyboardInterrupt as exc:
        status, error = 130, _error_text(exc)
    except Exception as exc:  # noqa: BLE001 - the summary of a failed run is still written
        status, error = 1, _error_text(exc)
    record.update(elapsed_seconds=round(time.monotonic() - started, 1), error=error,
                  finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if runner.last_experiment_id is None:
        loop_setup.write_json(out / "runs" / f"{grant.run_id}.json", record)
        print(f"run {grant.run_id} failed before its experiment existed: {error}", file=sys.stderr)
        return status or 1
    summary = _summarize(out, workspace, runner.last_experiment_id, record)
    result = (summary["experiment"]["controller"] or {}).get("result") or {}
    # Through MCP the strategy's own reason is its final message; the SDK reports that the agent stopped.
    print(json.dumps({"experiment_id": runner.last_experiment_id, "state": summary["experiment"]["state"],
                      "status": result.get("status"), "reason": result.get("reason"),
                      "final_message": summary["final_message"],
                      "selected_checkpoint_id": summary["experiment"]["selected_checkpoint_id"],
                      "summary": str(out / "runs" / f"{grant.run_id}.json"), "error": error}, indent=2))
    return status


if __name__ == "__main__":
    sys.exit(main())
