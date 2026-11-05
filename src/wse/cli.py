"""Operator command-line interface (``wse``).

This milestone provides ``wse import``, ``wse grant-run``, ``wse serve`` (the
stdio MCP server for one run grant; see ``wse.mcp_server``), ``wse worker``
(the evaluation worker process; see ``wse.worker``), ``wse freeze``,
``wse final-evaluate``, ``wse report``, and the read-only operator views
``wse experiment`` and ``wse patch``. A later milestone adds ``export``.
Every command calls the same Engine operations as the Python API and MCP.
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

from wse.schemas import WSEError

DEFAULT_WORKSPACE = "workspace"


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
    return parser


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
    from wse.api import Engine

    try:
        with Engine(args.workspace) as engine:
            if args.command == "import":
                manifest = engine.import_bundle(args.bundle)
                print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
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
    except WSEError as exc:
        return _error(exc)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
