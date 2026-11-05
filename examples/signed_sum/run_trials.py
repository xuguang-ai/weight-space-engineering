"""Run signed-sum trials end to end and write their reports (operator use).

Each invocation prepares the workspace under ``OUT/workspace`` on first use
(``build_fixture.setup_workspace``), issues one run grant per trial, and runs
each trial through the reference runner with a fresh controller (a new MCP
server and SDK conversation for an agent). Only after every trial of the
invocation has frozen does it run the final evaluations, so no trial can
adapt to final feedback. It then exports every experiment (final-set data
excluded), replays each export in a clean workspace to confirm
byte-identical weights, and writes ``OUT/reports/LABEL.json`` and
``OUT/reports/LABEL.md``.

Offline (no model, no network):

    uv run --no-sync python examples/signed_sum/run_trials.py OUT --label scripted
    uv run --no-sync python examples/signed_sum/run_trials.py OUT --label control \\
        --policy small_model_weights_only_v1
    uv run --no-sync python examples/signed_sum/run_trials.py OUT --label sdk-double \\
        --controller openai-agents --model wse-test-double:scripted_search@1

Live agent trials (the operator chooses the model and sets the SDK's
credentials in the environment; nothing here reads, prints, or stores them):

    uv run --no-sync python examples/signed_sum/run_trials.py OUT --label live \\
        --controller openai-agents --model MODEL_NAME --trials 5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _fixture():
    spec = importlib.util.spec_from_file_location("signed_sum_build_fixture", HERE / "build_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--label", required=True, help="report name for this invocation's trials")
    parser.add_argument("--controller", choices=("scripted", "openai-agents"), default="scripted")
    parser.add_argument("--model", default=None, help="openai-agents: the model name (or wse-test-double:STRATEGY)")
    parser.add_argument("--policy", default="small_model_v1")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--wall-clock-seconds", type=float, default=900.0)
    args = parser.parse_args(argv)
    if args.controller == "openai-agents" and not args.model:
        parser.error("--model is required for --controller openai-agents")
    os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

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
        config = {"kind": "agent", "model": args.model, "max_turns": args.max_turns,
                  "wall_clock_seconds": args.wall_clock_seconds}
    experiments = []
    for number, run_id in enumerate(grants, 1):
        runner = ReferenceRunner(workspace, log_dir=args.out / "logs")
        experiment_id = runner.run_experiment(run_id, ids["task_id"], args.policy, config)
        result = runner.last_result["result"]
        print(f"trial {number}/{len(grants)}: {experiment_id} {result['status']} ({result['reason']})", file=sys.stderr)
        experiments.append(experiment_id)
    # Final evaluations only after every trial has frozen.
    with Engine(workspace) as engine:
        for experiment_id in experiments:
            engine.final_evaluate(experiment_id)
        rows = [summarize_trial(engine, experiment_id) for experiment_id in experiments]
        replays = {}
        for experiment_id in experiments:
            target = exports / experiment_id
            engine.export_experiment(experiment_id, target)
            with Engine(args.out / "replay" / experiment_id) as clean:
                replays[experiment_id] = clean.replay_export(target).all_identical
    for row in rows:
        row["replay_identical"] = replays[row["experiment_id"]]
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{args.label}.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_trials_markdown(rows) + "\nExports replayed to byte-identical weights: " + ", ".join(
        f"{eid}: {'yes' if ok else 'NO'}" for eid, ok in replays.items()) + "\n"
    (reports / f"{args.label}.md").write_text(markdown, encoding="utf-8")
    sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
