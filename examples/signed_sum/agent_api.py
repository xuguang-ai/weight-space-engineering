"""Run one signed-sum experiment through ``wse.agent_runs.run_agent_experiment`` and print its result.

The first invocation prepares the workspace under ``OUT/workspace``
(``build_fixture.setup_workspace``, as ``run_trials.py`` does); later ones
reuse it. Each invocation issues a new run grant; its logs, export, and
replay workspace go to ``OUT/workspace/agent_runs/RUN_ID``.

Offline, with the scripted controller (no model, no network):

    uv run --no-sync python examples/signed_sum/agent_api.py OUT

A live LLM controller through ``openai_agents@1`` (the operator chooses the
model and sets the SDK's credentials in the environment; nothing here reads,
prints, or stores them):

    uv run --no-sync python examples/signed_sum/agent_api.py OUT --model MODEL_NAME
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
    parser.add_argument("--model", default=None, help="an LLM controller (default: the scripted_search@1 strategy)")
    parser.add_argument("--policy", default="small_model_v1")
    args = parser.parse_args(argv)

    from wse.agent_runs import run_agent_experiment
    from wse.api import Engine

    workspace = args.out / "workspace"
    ids_file = args.out / "ids.json"
    if ids_file.exists():
        ids = json.loads(ids_file.read_text(encoding="utf-8"))
    else:
        with Engine(workspace) as engine:
            ids = _fixture().setup_workspace(engine, args.out / "bundles")
        ids_file.write_text(json.dumps(ids, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    controller = {"model": args.model} if args.model else {"strategy": "scripted_search@1"}
    # One experiment, so its final evaluation runs at once (a series would wait until every trial froze).
    run = run_agent_experiment(workspace, task=ids["task_id"], base=ids["task_checkpoint_id"], policy=args.policy,
                               final_evaluation=True, **controller)
    print(f"experiment {run.experiment_id} (run {run.run_id}): {run.state}")
    print(f"controller {run.summary['controller']} ended {run.summary['status']}: {run.summary['reason']}")
    print(f"success: {run.success}; final: {json.dumps(run.final, sort_keys=True)}")
    print(f"export: {run.export_dir}; replay byte-identical: {run.replay_identical}")
    return 0 if run.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
