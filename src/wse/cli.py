"""Operator command-line interface (``wse``).

This milestone provides ``wse import``, ``wse worker`` (the evaluation worker
process; see ``wse.worker``), ``wse freeze``, ``wse final-evaluate``, and
``wse report``. Later milestones add ``grant-run``, ``serve``, and ``export``;
every command calls the same Engine operations as the Python API. Reports
print as JSON (machine-readable) or, with ``--format markdown``, as concise
Markdown; ``--output`` also writes the rendering to a file.
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args, extra = parser.parse_known_args(argv)
    if args.command == "worker":
        from wse.worker import main as worker_main

        return worker_main(["--workspace", args.workspace, *extra])
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    from wse.api import Engine

    try:
        with Engine(args.workspace) as engine:
            if args.command == "import":
                manifest = engine.import_bundle(args.bundle)
                print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
            elif args.command == "freeze":
                frozen = engine.freeze(args.experiment_id, args.expected_selected_id)
                print(json.dumps(frozen.model_dump(mode="json"), indent=2, sort_keys=True))
            elif args.command == "final-evaluate":
                _emit(engine.final_evaluate(args.experiment_id), args.format, args.output)
            elif args.command == "report":
                _emit(engine.get_report(args.experiment_id, args.report_id), args.format, args.output)
    except WSEError as exc:
        print(json.dumps({"error": exc.to_domain_error().model_dump(mode="json")}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
