"""Run signed-sum trials end to end and write their reports (operator use).

Each invocation prepares the workspace under ``OUT/workspace`` on first use
(``build_fixture.setup_workspace``), issues one run grant per trial, and runs
each trial through the reference runner with a fresh controller (a new Codex
session and MCP server for an agent). Only after every trial of the
invocation has frozen does it run the final evaluations, so no trial can
adapt to final feedback. It then exports every experiment (final-set data
excluded), replays each export in a clean workspace to confirm
byte-identical weights, and writes ``OUT/reports/LABEL.json`` and
``OUT/reports/LABEL.md``.

A trial that raises is reported with its error and counts as unsuccessful;
the remaining trials still run. Ctrl-C stops the running trial (the runner
settles and freezes it first), starts no further trial, and still
final-evaluates and reports the frozen ones; press it again to abort at
once. The exit status is 0, 1 if any trial or step failed, or 130 after an
interrupt.

Offline (no model, no network):

    uv run python examples/signed_sum/run_trials.py OUT --label scripted
    uv run python examples/signed_sum/run_trials.py OUT --label control \\
        --policy small_model_weights_only_v1
    uv run python examples/signed_sum/run_trials.py OUT --label codex-double \\
        --controller codex --model wse-test-double:scripted_search@1

The last runs the scripted search through the agent's path: Codex's offline
stand-in connects to each run's MCP server, and the strategy chooses every
tool call (a toolchain check, not an agent).

Live agent trials run in the Codex CLI (its path in ``WSE_CODEX_PATH``,
else ``codex`` on PATH, else OpenAI's VS Code extension's). The operator
chooses the model; uv loads the key and endpoint from ``.env``, a copy of
``.env.example``. Only the per-run local model proxy (``wse.model_proxy``)
takes them from this process's environment; nothing prints or stores them,
and neither Codex nor the agent's commands receive them:

    uv run --env-file .env python examples/signed_sum/run_trials.py OUT --label live \\
        --controller codex --model YOUR_AGENT_MODEL --trials 5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _fixture():
    spec = importlib.util.spec_from_file_location("signed_sum_build_fixture", HERE / "build_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _error_text(exc: BaseException) -> str:
    from wse.schemas import WSEError

    if isinstance(exc, WSEError):
        return f"{exc.code.value}: {exc.message}"
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


def _joined(first: str | None, second: str) -> str:
    return second if first is None else f"{first}; {second}"


def _unstarted_row(trial: dict, args: argparse.Namespace) -> dict:
    """The report row of a trial that failed before its experiment existed."""
    return {"experiment_id": None, "run_id": trial["run_id"], "policy_id": args.policy, "mode": None, "state": None,
            "controller": None, "model": args.model, "status": "error", "reason": trial["error"],
            "stop_condition": None, "candidates": 0, "evaluated_examples": 0, "compute_seconds": 0.0,
            "elapsed_seconds": None, "turns": None, "tool_calls": None,
            "tokens": {"input_tokens": None, "output_tokens": None, "total_tokens": None},
            "tokens_unavailable": {}, "changed_elements": 0, "parameter_count": None, "changed_fraction": 0.0,
            "constraint_status": None, "tracing": None, "final": None, "success": False,
            "failure_cause": f"the trial failed before its experiment was created: {trial['error']}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--label", required=True, help="report name for this invocation's trials")
    parser.add_argument("--controller", choices=("scripted", "codex"), default="scripted",
                        help="scripted: scripted_search@1; codex: an agent in the Codex CLI")
    parser.add_argument("--model", default=None,
                        help="codex: the model name (or wse-test-double:STRATEGY@VERSION, Codex's offline stand-in)")
    parser.add_argument("--policy", default="small_model_v1")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=100, help="codex: the limit on the agent's model requests")
    parser.add_argument("--wall-clock-seconds", type=float, default=900.0)
    args = parser.parse_args(argv)
    if args.controller == "codex" and not args.model:
        parser.error("--model is required for --controller codex")

    from wse.api import Engine
    from wse.runner import ReferenceRunner, render_trials_markdown, summarize_trial

    workspace = args.out / "workspace"
    reports = args.out / "reports"
    exports = args.out / "exports"
    with Engine(workspace) as engine:
        ids_file = args.out / "ids.json"
        if ids_file.exists():
            ids = json.loads(ids_file.read_text(encoding="utf-8"))
        else:
            ids = _fixture().setup_workspace(engine, args.out / "bundles")
            ids_file.write_text(json.dumps(ids, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        grants = [engine.grant_run(ids["task_id"], args.policy, ids["task_checkpoint_id"]).run_id
                  for _ in range(args.trials)]
    if args.controller == "scripted":
        config = {"kind": "strategy", "wall_clock_seconds": args.wall_clock_seconds}
    else:
        config = {"kind": "agent", "adapter": "codex_cli@1", "model": args.model, "max_turns": args.max_turns,
                  "wall_clock_seconds": args.wall_clock_seconds}
    trials: list[dict] = []
    interrupted = False
    for number, run_id in enumerate(grants, 1):
        runner = ReferenceRunner(workspace, log_dir=args.out / "logs")
        trial = {"trial": number, "run_id": run_id, "experiment_id": None, "error": None}
        trials.append(trial)
        try:
            runner.run_experiment(run_id, ids["task_id"], args.policy, config)
        except KeyboardInterrupt as exc:
            trial["error"], interrupted = _error_text(exc), True
        except Exception as exc:  # noqa: BLE001 - one failed trial must not lose the other trials' reports
            trial["error"] = _error_text(exc)
        trial["experiment_id"] = runner.last_experiment_id
        result = (runner.last_result or {}).get("result") or {}
        ending = f"{result['status']} ({result['reason']})" if result else "no termination record"
        print(f"trial {number}/{len(grants)}: {trial['experiment_id'] or 'no experiment'} {ending}"
              + (f"; {trial['error']}" if trial["error"] else ""), file=sys.stderr)
        if interrupted:
            print("interrupted: no further trial starts; final-evaluating and reporting the frozen trials (press "
                  "Ctrl-C again to abort)", file=sys.stderr)
            break
    # Final evaluations only after every trial has frozen (or failed).
    rows = []
    replays = {}
    with Engine(workspace) as engine:
        for trial in trials:
            eid = trial["experiment_id"]
            if eid is not None and engine.get_experiment(eid).state == "FROZEN":
                try:
                    engine.final_evaluate(eid)
                except Exception as exc:  # noqa: BLE001
                    trial["error"] = _joined(trial["error"], f"final evaluation: {_error_text(exc)}")
        for trial in trials:
            eid = trial["experiment_id"]
            row = summarize_trial(engine, eid) if eid is not None else _unstarted_row(trial, args)
            row["trial"], row["replay_identical"] = trial["trial"], None
            if row["state"] == "CLOSED":
                try:
                    target = exports / eid
                    engine.export_experiment(eid, target)
                    with Engine(args.out / "replay" / eid) as clean:
                        row["replay_identical"] = replays[eid] = clean.replay_export(target).all_identical
                except Exception as exc:  # noqa: BLE001
                    trial["error"] = _joined(trial["error"], f"export or replay: {_error_text(exc)}")
            row["error"] = trial["error"]
            if row["error"] is not None and row["success"] is None:  # a failed trial is never evaluated later
                row["success"] = False
                row["failure_cause"] = row["failure_cause"] or f"the trial did not complete: {row['error']}"
            rows.append(row)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{args.label}.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_trials_markdown(rows) + "\nExports replayed to byte-identical weights: " + (", ".join(
        f"{eid}: {'yes' if ok else 'NO'}" for eid, ok in replays.items()) or "none") + "\n"
    failed = [row for row in rows if row["error"] is not None]
    if failed:
        markdown += "\nTrial errors:\n\n" + "".join(
            f"- trial {row['trial']} ({row['experiment_id'] or row['run_id']}): {row['error']}\n" for row in failed)
    if interrupted:
        markdown += (f"\nInterrupted by the operator: {len(grants) - len(trials)} of {len(grants)} trials not "
                     "started.\n")
    (reports / f"{args.label}.md").write_text(markdown, encoding="utf-8")
    sys.stdout.write(markdown)
    return 130 if interrupted else 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
