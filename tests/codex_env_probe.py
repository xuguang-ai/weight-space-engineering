"""Record the environment a process receives, then run a command in its place (test helper).

Usage: python codex_env_probe.py RECORD.json TOKEN_FILE -- COMMAND ARGS...

Writes the variable names, and the names of variables whose value equals or
contains the token read from TOKEN_FILE (so the token itself never appears
in a command line), to RECORD.json (never the values), then replaces itself
with COMMAND,
which keeps this environment. Tests put it in front of ``wse serve`` to see
what Codex hands its MCP server.
"""

import json
import os
import sys

record, token_file, separator, *command = sys.argv[1:]
assert separator == "--" and command
with open(token_file, encoding="utf-8") as handle:
    token = handle.read().strip()
with open(record, "w", encoding="utf-8") as handle:
    json.dump({"names": sorted(os.environ), "token_values": sorted(name for name, value in os.environ.items()
                                                                  if value == token),
               "token_substrings": sorted(name for name, value in os.environ.items() if token in value)}, handle)
os.execv(command[0], command)
