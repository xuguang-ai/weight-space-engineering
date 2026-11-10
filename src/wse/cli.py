"""Operator command-line interface (``wse``).

This milestone provides ``wse import``, ``wse import-hf`` (a local Hugging Face
snapshot through ``hf_causal_lm@1``), ``wse import-patch`` (a published edit
recorded as an exact patch with its candidate; see ``wse.external_patches``), ``wse grant-run``, ``wse serve`` (the
stdio MCP server for one run grant; see ``wse.mcp_server``), ``wse worker``
(the evaluation worker process; see ``wse.worker``), ``wse freeze``,
``wse final-evaluate``, ``wse report``, the read-only operator views
``wse experiment`` and ``wse patch``, ``wse run`` (the reference runner: one
controller per run grant, then settlement and freezing; see ``wse.runner``),
``wse export`` (an experiment export; final-set data excluded by default),
``wse replay`` (replays an export's exact patches into the workspace and
verifies byte-identical weights), ``wse compare-generation`` (a resumable
base-versus-candidate generation comparison on a seeded problem subset; see
``wse.evaluators.generation_runs``), and ``wse trials`` (per-trial summaries for
agent-acceptance reports). Every command calls the same Engine operations as
the Python API and MCP.
Reports print as JSON (machine-readable) or, with ``--format markdown``, as
concise Markdown; ``--output`` also writes the rendering to a file. Errors
print as ``{"error": {...}}`` JSON on stderr with exit code 2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from typing import Any

from wse.schemas import ErrorCode, WSEError

DEFAULT_WORKSPACE = "workspace"
TEST_DOUBLE = "wse-test-double:"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wse", description="Weight-space engineering operator CLI")
    parser.add_argument(
        "--workspace",
        default=os.environ.get("WSE_WORKSPACE", DEFAULT_WORKSPACE),
        help="workspace directory holding artifacts/ and metadata.sqlite "
        "(default: $WSE_WORKSPACE or ./workspace)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import", help="import a model bundle and print its manifest")
    importer.add_argument("bundle", help="bundle directory containing wse_bundle.json")
    import_hf = commands.add_parser(
        "import-hf", help="import a local Hugging Face snapshot (hf_causal_lm@1) and print its manifest")
    import_hf.add_argument("snapshot", help="snapshot directory (for example from 'hf download --local-dir')")
    import_hf.add_argument("--repo", required=True, help="the source repository ID, for example Qwen/Qwen3.5-0.8B")
    import_hf.add_argument("--revision", required=True, help="the source revision (commit SHA) that was downloaded")
    import_hf.add_argument("--family", default=None,
                           help="an exact family descriptor '<id>@<version>' (default: the one registered for the "
                                "configuration's model_type, if any)")
    import_hf.add_argument("--license", default="unspecified", help="the model's license label")
    import_patch = commands.add_parser(
        "import-patch", help="import a published edit as an exact patch, commit its candidate, and print the record")
    import_patch.add_argument("edit_dir", help="the edit directory (manifest.json and edit.npz)")
    import_patch.add_argument("--format", required=True, choices=["lm-understand-v1"], help="the edit format")
    import_patch.add_argument("--base", required=True, help="the checkpoint ID of the base model")
    import_patch.add_argument("--repo", required=True,
                              help="the source repository of the edit, for provenance (e.g. RikiyaT/lm_understand-logs)")
    import_patch.add_argument("--revision", required=True, help="the source revision (commit SHA) of the edit")
    grant = commands.add_parser("grant-run", help="issue a run grant (one experiment) and print it")
    grant.add_argument("task_id")
    grant.add_argument("policy_id")
    grant.add_argument("base_checkpoint_id")
    serve = commands.add_parser("serve", help="serve one run grant over stdio MCP (logs go to stderr)")
    serve.add_argument("--run-id", required=True, help="the persisted run grant this server is bound to")
    serve.add_argument("--setup", action="append", default=[],
                       help="operator setup hook 'module:function' called with the registry of the server and of "
                            "its worker (repeatable)")
    serve.add_argument("--no-worker", action="store_true",
                       help="do not start the evaluation worker (another process owns the job scheduler)")
    serve.add_argument("--worker-args", default="",
                       help="extra options for the worker process, as one shell-quoted string")
    serve.add_argument("--constraint-status", choices=("unverified", "verified"), default="unverified",
                       help="'verified' only when the operator removed every other execution channel of the host")
    serve.add_argument("--log-file", default=None,
                       help="append the server's and its worker's stderr (logs) to this file instead of stderr")
    runner = commands.add_parser("run", help="run one controller on a run grant's experiment, then settle and freeze")
    runner.add_argument("--run-id", required=True, help="the persisted run grant to use")
    runner.add_argument("--controller", choices=("scripted", "openai-agents"), default="scripted")
    runner.add_argument("--strategy", default="scripted_search@1", help="scripted: the registered strategy")
    runner.add_argument("--param", action="append", default=[], metavar="NAME=JSON",
                        help="scripted: a strategy parameter (repeatable)")
    runner.add_argument("--model", default=None,
                        help="openai-agents: the operator-chosen model name (credentials come from the SDK's "
                             f"environment), or '{TEST_DOUBLE}<strategy>' for the offline SDK test double")
    runner.add_argument("--max-turns", type=int, default=100, help="openai-agents: SDK turn limit (default 100)")
    runner.add_argument("--wall-clock-seconds", type=float, default=900.0,
                        help="controller wall-clock limit in seconds (default 900)")
    runner.add_argument("--setup", action="append", default=[],
                        help="operator setup hook 'module:function' for the runner, server, and worker (repeatable)")
    runner.add_argument("--worker-args", default="", help="extra worker options, as one shell-quoted string")
    exporter = commands.add_parser("export", help="export an experiment (hashes, patches, reports, traces, costs)")
    exporter.add_argument("experiment_id")
    exporter.add_argument("--output", required=True, help="a new or empty export directory")
    exporter.add_argument("--no-base-weights", action="store_true", help="omit the base model bundle")
    exporter.add_argument("--include-final-data", action="store_true",
                          help="also export the final set's inputs, labels, and generation seed (excluded by default)")
    replay = commands.add_parser("replay", help="replay an export's exact patches into this workspace")
    replay.add_argument("export_dir")
    trials = commands.add_parser("trials", help="summarize trials: controller, costs, changed fraction, final result")
    trials.add_argument("experiment_ids", nargs="+")
    trials.add_argument("--format", choices=("json", "markdown"), default="json")
    trials.add_argument("--output", default=None, help="also write the rendering to this file")
    experiment = commands.add_parser("experiment", help="print an experiment's state and budget")
    experiment.add_argument("experiment_id")
    patch = commands.add_parser("patch", help="print a stored patch: its frozen spec and exact materialized patch")
    patch.add_argument("experiment_id")
    patch.add_argument("patch_id")
    commands.add_parser("worker", help="run the evaluation worker for the workspace",
                        description="Options after 'worker' (for example --lease-seconds or --setup "
                        "module:function) are passed to python -m wse.worker.")
    freeze = commands.add_parser("freeze", help="freeze an experiment's selected checkpoint")
    freeze.add_argument("experiment_id")
    freeze.add_argument("--expected-selected-id", required=True,
                        help="the selected checkpoint the operator expects (compare-and-swap)")
    final = commands.add_parser("final-evaluate",
                                help="evaluate the frozen candidate and the base-checkpoint control on the final "
                                     "set, then close the experiment (re-running prints the sealed report)")
    final.add_argument("experiment_id")
    report = commands.add_parser("report", help="print an evaluation report of an experiment")
    report.add_argument("experiment_id")
    report.add_argument("report_id")
    for command in (final, report):
        command.add_argument("--format", choices=("json", "markdown"), default="json")
        command.add_argument("--output", default=None, help="also write the rendering to this file")
    compare = commands.add_parser(
        "compare-generation",
        help="evaluate a base and a candidate checkpoint on a seeded subset of a text dataset, resumably "
             "(operator only; see wse.evaluators.generation_runs)")
    compare.add_argument("--base", required=True, help="the base checkpoint ID")
    compare.add_argument("--candidate", required=True, help="the candidate checkpoint ID")
    compare.add_argument("--dataset", required=True, help="a JSON Lines file of problems")
    compare.add_argument("--dataset-format", required=True, choices=["math500", "gsm8k", "math"],
                         help="the file's format (wse.evaluators.text_suites)")
    compare.add_argument("--id-prefix", default=None, help="the item-ID prefix for formats that number lines")
    compare.add_argument("--problems", type=int, required=True, help="how many problems the subset holds")
    compare.add_argument("--subset-seed", type=int, required=True, help="the seed that chooses the subset")
    compare.add_argument("--profile", required=True, help="an exact generation profile, e.g. qwen35think@1")
    compare.add_argument("--grader", required=True, help="an exact grader, e.g. math_boxed@1")
    compare.add_argument("--samples", type=int, default=1, help="samples per problem (default 1)")
    compare.add_argument("--max-new-tokens", type=int, default=None,
                         help="override the profile's token limit (default: the profile's)")
    compare.add_argument("--eval-seed", type=int, default=0, help="the evaluation seed (default 0)")
    compare.add_argument("--device", choices=["mps", "cpu"], default=None,
                         help="the device (default: mps when available)")
    compare.add_argument("--batch-size", type=int, default=1, help="problems per batch (default 1)")
    compare.add_argument("--out", required=True,
                         help="the output directory (run.json, problems.jsonl, summary.json); outside the "
                              "repository; an existing one resumes")
    return parser


def _compare_generation(engine: Any, args: argparse.Namespace) -> None:
    from wse.evaluators.generation_runs import run_comparison

    def progress(line: str) -> None:
        print(line, file=sys.stderr, flush=True)

    summary = run_comparison(
        engine, base=args.base, candidate=args.candidate, dataset=args.dataset, format=args.dataset_format,
        id_prefix=args.id_prefix, count=args.problems, subset_seed=args.subset_seed, profile=args.profile,
        grader=args.grader, samples_per_problem=args.samples, max_new_tokens=args.max_new_tokens,
        eval_seed=args.eval_seed, device=args.device, batch_size=args.batch_size, out_dir=args.out, log=progress)
    print(json.dumps({"comparison": summary["comparison"], "seconds": summary["seconds"],
                      "generated_tokens": summary["generated_tokens"]}, indent=2, sort_keys=True))


def _emit(value, fmt: str, output: str | None) -> None:
    if fmt == "markdown":
        from wse.evaluation import render_markdown

        text = render_markdown(value)
    else:
        text = json.dumps(value.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    if output:
        with open(output, "w", encoding="utf-8") as handle:
            handle.write(text)
    sys.stdout.write(text)


def _error(exc: WSEError) -> int:
    print(json.dumps({"error": exc.to_domain_error().model_dump(mode="json")}), file=sys.stderr)
    return 2


def _serve(args: argparse.Namespace) -> int:
    """``wse serve``: nothing but the protocol may reach stdout, so errors go to stderr."""
    import shlex

    from wse.mcp_server import serve  # the only CLI path that imports MCP
    from wse.registry import builtin_registry
    from wse.worker import _load_setup

    if args.log_file:
        # Redirect fd 2 so the worker process, which inherits it, logs to the same file.
        log = os.open(args.log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(log, 2)
        os.close(log)
    registry = builtin_registry()
    worker_args: list[str] = []
    for spec in args.setup:
        _load_setup(spec)(registry)
        worker_args += ["--setup", spec]
    worker_args += shlex.split(args.worker_args)
    try:
        return serve(args.workspace, args.run_id, registry=registry, constraint_status=args.constraint_status,
                     worker=not args.no_worker, worker_args=tuple(worker_args))
    except WSEError as exc:
        return _error(exc)


def _run(args: argparse.Namespace) -> int:
    """``wse run``: the reference runner; prints the controller record as JSON."""
    import shlex

    from wse.runner import ReferenceRunner

    if args.controller == "scripted":
        parameters = {}
        for item in args.param:
            name, _, value = item.partition("=")
            try:
                parameters[name] = json.loads(value)
            except ValueError:
                parameters[name] = value
        config: dict = {"kind": "strategy", "strategy": args.strategy, "parameters": parameters,
                        "wall_clock_seconds": args.wall_clock_seconds}
    else:
        if not args.model:
            raise WSEError(ErrorCode.INVALID_ARGUMENT, "--model is required for --controller openai-agents")
        config = {"kind": "agent", "model": args.model, "max_turns": args.max_turns,
                  "wall_clock_seconds": args.wall_clock_seconds}
    runner = ReferenceRunner(args.workspace, setup=args.setup, worker_args=shlex.split(args.worker_args))
    from wse.api import Engine

    with Engine(args.workspace, registry=runner.registry) as engine:
        grant = engine.experiments.get_run_grant(args.run_id)
    runner.run_experiment(args.run_id, grant.task_id, grant.policy_id, config)
    print(json.dumps(runner.last_result, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args, extra = parser.parse_known_args(argv)
    if args.command == "worker":
        from wse.worker import main as worker_main

        return worker_main(["--workspace", args.workspace, *extra])
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    if args.command == "serve":
        return _serve(args)
    if args.command == "run":
        try:
            return _run(args)
        except WSEError as exc:
            return _error(exc)
    from wse.api import Engine

    try:
        with Engine(args.workspace) as engine:
            if args.command == "import":
                manifest = engine.import_bundle(args.bundle)
                print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
            elif args.command == "import-hf":
                manifest = engine.import_hf_snapshot(args.snapshot, repo=args.repo, revision=args.revision,
                                                     family=args.family, license=args.license)
                print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
            elif args.command == "import-patch":
                record = engine.import_patch(args.edit_dir, base_checkpoint_id=args.base, format=args.format,
                                             repo=args.repo, revision=args.revision)
                print(json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True))
            elif args.command == "grant-run":
                granted = engine.grant_run(args.task_id, args.policy_id, args.base_checkpoint_id)
                print(json.dumps(granted.model_dump(mode="json"), indent=2, sort_keys=True))
            elif args.command == "experiment":
                print(json.dumps(engine.get_experiment(args.experiment_id).model_dump(mode="json"), indent=2,
                                 sort_keys=True))
            elif args.command == "patch":
                print(json.dumps(engine.read_patch(args.experiment_id, args.patch_id).model_dump(mode="json"),
                                 indent=2, sort_keys=True))
            elif args.command == "freeze":
                frozen = engine.freeze(args.experiment_id, args.expected_selected_id)
                print(json.dumps(frozen.model_dump(mode="json"), indent=2, sort_keys=True))
            elif args.command == "final-evaluate":
                _emit(engine.final_evaluate(args.experiment_id), args.format, args.output)
            elif args.command == "report":
                _emit(engine.get_report(args.experiment_id, args.report_id), args.format, args.output)
            elif args.command == "export":
                exported = engine.export_experiment(args.experiment_id, args.output,
                                                    include_base_weights=not args.no_base_weights,
                                                    include_final_data=args.include_final_data)
                print(json.dumps(exported.model_dump(mode="json"), indent=2, sort_keys=True))
            elif args.command == "trials":
                from wse.runner import render_trials_markdown, summarize_trial

                rows = [summarize_trial(engine, experiment_id) for experiment_id in args.experiment_ids]
                text = (render_trials_markdown(rows) if args.format == "markdown"
                        else json.dumps(rows, indent=2, sort_keys=True) + "\n")
                if args.output:
                    with open(args.output, "w", encoding="utf-8") as handle:
                        handle.write(text)
                sys.stdout.write(text)
            elif args.command == "compare-generation":
                _compare_generation(engine, args)
            elif args.command == "replay":
                print(json.dumps(engine.replay_export(args.export_dir).model_dump(mode="json"), indent=2,
                                 sort_keys=True))
    except WSEError as exc:
        return _error(exc)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
