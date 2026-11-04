"""Operator command-line interface (``wse``).

This milestone provides ``wse import``. Later milestones add ``grant-run``,
``serve``, ``freeze``, ``final-evaluate``, and ``export``; every command calls
the same Engine operations as the Python API.
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from wse.api import Engine

    try:
        with Engine(args.workspace) as engine:
            if args.command == "import":
                manifest = engine.import_bundle(args.bundle)
                print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
    except WSEError as exc:
        print(json.dumps({"error": exc.to_domain_error().model_dump(mode="json")}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
