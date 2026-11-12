"""The Codex CLI agent adapter (``codex_cli@1``): ``codex exec --json`` against the run's WSE MCP server.

``CodexCLIAdapter`` runs OpenAI's Codex CLI non-interactively as the agent of
one run grant. Codex owns the conversation, the model calls, and its own
sandboxed shell; WSE owns every weight version, budget, evaluation, and
selection, which the agent reaches only through the run's ``wse serve``
(configured as Codex's stdio MCP server ``wse``). The adapter turns Codex's
JSONL events into the normalized ``AgentEvent`` records of
``wse.agent_adapters.base``, and it stops Codex when the runner asks.

* **Per-run Codex home.** ``start`` writes a fresh ``CODEX_HOME`` under the
  run's log directory (``<log_dir>/codex-<run_id>/home``, outside the
  sandbox's readable roots): ``config.toml`` (``codex_config``) and the WSE
  skill at ``skills/wse/SKILL.md``, which Codex lists to the model in its
  "## Skills" catalog. Bundled Codex skills, project documents and project
  roots (``project_doc_max_bytes = 0``, ``project_root_markers = []``: no
  ``AGENTS.md`` or ``.codex`` configuration from the directories around the
  run), plugins, apps, web search, image tools, sub-agents, hooks, memories,
  and analytics are switched off; Codex's own home directory is an empty one
  of the run, and ``--strict-config`` makes Codex reject a key it does not
  know.
* **Model access.** Codex talks to the model only through a provider
  fragment (``provider_config``, for example the local model proxy's
  ``codex_provider_config``) whose ``env_key`` names
  ``WSE_MODEL_PROXY_TOKEN``; the adapter passes that per-run token in
  Codex's environment, which is otherwise built from scratch. Upstream
  credentials never reach Codex, its commands, or the MCP server.
* **Isolation (Codex's native permission profile ``wse``).** Commands the
  model runs read the run's sandbox directory
  (``<workspace>/sandboxes/<run_id>``: ``inputs/`` read only; ``work/``,
  ``outputs/``, and ``tmp/`` writable), Codex's minimal platform paths
  (``:minimal``), the Python runtime, the analysis libraries
  (``wse.sandbox.ANALYSIS_DISTRIBUTIONS`` and their dependencies), and the
  skill. Everything else is unreadable: the workspace's database, store,
  and evaluator storage, the repository, keys, and Codex's own home. A
  permission-profile grant is always a whole subtree, so the environment's
  ``site-packages`` directories are denied and each analysis entry in them is
  allowed again; a symbolic-link directory (``pythonpath/``, on
  ``PYTHONPATH``) makes those entries importable, so ``python`` is the
  environment's interpreter with NumPy, PyTorch, and safetensors, while WSE's
  own distribution, whose metadata holds its README, stays unreadable.
  ``/tmp`` is denied, because Codex otherwise grants writes there, and
  network access for commands is off. The shell environment is set
  explicitly (``shell_environment_policy.inherit = "none"``), and commands run
  in a non-login shell (``allow_login_shell = false``), so macOS's
  ``path_helper`` cannot put ``/usr/bin/python3`` before the environment's
  interpreter.
* **Tools.** The WSE MCP tools are exposed as direct function tools with
  their schemas (``features.code_mode.direct_only_tool_namespaces``); Codex's
  sub-agent tools are removed (``excluded_tool_namespaces``). The WSE tools
  are limited to the runner's allow-list (``enabled_tools``), approved
  without asking (approval policy ``never``), and each call may block for the
  run's wall-clock limit, and at least for the runner's session timeout, which
  covers the policy's longest call (``tool_timeout_sec``), so a blocking
  evaluation needs one call instead of many polls.
* **Jobs.** The adapter runs the workspace's evaluation worker itself
  (``wse.worker.WorkerSupervisor``, with the server's allow-listed
  environment and setup hooks) from ``start`` until ``close``, and starts
  ``wse serve`` with ``--no-worker``: each Codex process starts its own MCP
  server, so jobs outlive a server that ends with its segment, keep running
  across resumes, and are settled by a live worker when the runner
  finalizes.
* **Segments and resume.** ``run`` starts ``codex exec`` in the sandbox
  directory. When Codex ends its turn while the runner has not asked it to
  stop (the runner's stop monitor checks after every tool result) and the
  agent did not stop deliberately (a final message whose first line starts
  with ``WSE-STOP:``), or after a recoverable failure, the adapter continues
  the same Codex session with ``codex exec resume <session_id>``, at most
  ``max_resumes`` times (2 by default). Each segment is recorded.
* **Limits and cancellation.** The wall-clock limit spans all segments; the
  turn limit counts model requests. ``cancel`` (any thread), the wall-clock
  limit, and the turn limit stop Codex's process group with SIGINT, then
  SIGTERM and SIGKILL after grace periods; afterwards every descendant
  process seen while Codex ran (its MCP server among them) is waited for and,
  if still alive, terminated, so nothing outlives the run.
* **Events, usage, and logs.** Each segment's raw JSONL stream and standard
  error are kept in the run's log directory. ``mcp_tool_call`` items become
  ``tool_call``/``tool_result`` events (``ok`` or the WSE error code);
  ``command_execution`` and ``file_change`` items become events of the tools
  ``shell`` and ``apply_patch``; agent messages become ``message`` events.
  Usage comes from Codex's session record (one ``model_response`` event per
  model request, with input, cached input, output, and reasoning tokens);
  when that record is missing, from the cumulative usage of ``turn.completed``.
  Private reasoning is never stored. Cost and the served model snapshot are
  not reported by Codex.

The Codex binary comes from ``codex_path``, else ``WSE_CODEX_PATH``, else a
search of ``PATH`` and then of the VS Code extension bundles
(``~/.vscode*/extensions/openai.chatgpt-*/bin/*/codex``, newest first). On
macOS a binary is used only after ``codesign`` verifies its signature and
reports OpenAI's Team ID ``2DC432GLL2``.

``CodexCLIAdapter.test_double(strategy)`` runs
``wse.agent_adapters.codex_test_double`` instead of Codex: a toolchain check
in which a registered stepwise strategy chooses every tool call through the
same configuration, MCP server, event stream, and process handling. It is
not an agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import shutil
import signal
import site
import subprocess
import sys
import threading
import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from wse.agent_adapters.base import (
    AgentAdapter,
    AgentConfig,
    AgentEvent,
    AgentRunResult,
    AgentUsage,
    EventSink,
    RunConnection,
    describe_exception,
)
from wse.schemas import ErrorCode, WSEError, content_hash

__all__ = [
    "worker_arguments",
    "CODEX_PATH_ENV",
    "CODEX_TEAM_IDENTIFIER",
    "PROFILE_NAME",
    "PROXY_TOKEN_ENV",
    "STOP_MARKER",
    "CodexBinary",
    "CodexCLIAdapter",
    "PythonAccess",
    "codex_config",
    "find_codex",
    "parse_mcp_result",
    "python_access",
    "skill_text",
    "toml_dumps",
]

logger = logging.getLogger("wse.agents.codex")

ADAPTER_ID = "codex_cli"
ADAPTER_VERSION = "1"
CODEX_TEAM_IDENTIFIER = "2DC432GLL2"  # OpenAI OpCo, LLC (Developer ID)
CODEX_PATH_ENV = "WSE_CODEX_PATH"
PROXY_TOKEN_ENV = "WSE_MODEL_PROXY_TOKEN"
PROFILE_NAME = "wse"
MCP_SERVER_NAME = "wse"
MCP_NAMESPACE = f"mcp__{MCP_SERVER_NAME}"
SKILL_NAME = "wse"
STOP_MARKER = "WSE-STOP"
TEST_DOUBLE_PROVIDER = "wse-test-double"
TEST_DOUBLE_MODULE = "wse.agent_adapters.codex_test_double"
DEFAULT_MAX_RESUMES = 2
DEFAULT_INTERRUPT_GRACE_SECONDS = 10.0
DEFAULT_TERMINATE_GRACE_SECONDS = 5.0
DEFAULT_ORPHAN_GRACE_SECONDS = 5.0
SNAPSHOT_SECONDS = 2.0
ROLLOUT_POLL_SECONDS = 0.5
STREAM_DRAIN_SECONDS = 5.0
STREAM_LIMIT_BYTES = 64 << 20
OUTPUT_EXCERPT_CHARS = 2000
MESSAGE_EXCERPT_CHARS = 2000
ERROR_PREFIX = "Error executing tool "
SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
VSCODE_EXTENSION_DIRS = ("~/.vscode/extensions", "~/.vscode-insiders/extensions")
# Codex features that a WSE run does not use; each would add tools, context, or network access.
DISABLED_FEATURES = ("apps", "browser_use", "computer_use", "goals", "hooks", "image_generation", "in_app_browser",
                     "memories", "multi_agent", "plugins", "remote_plugin", "shell_snapshot", "tool_suggest",
                     "view_image")
# Tool namespaces removed from Codex's code-mode tools: sub-agents (``features.multi_agent = false`` alone leaves
# ``multi_agent_v1__spawn_agent`` callable for gpt-5.6-luna in 0.167), which would run model sessions of their own.
EXCLUDED_TOOL_NAMESPACES = ("multi_agent_v1",)
# Failures that a new turn cannot fix; anything else that ends a turn is retried by a resume.
NON_RECOVERABLE = re.compile(r"\b(401|403|404)\b|unauthori[sz]ed|forbidden|invalid[_ ]api[_ ]key|model[_ ]not[_ ]found|"
                             r"context[_ ]window|configuration|config\.toml|not supported", re.IGNORECASE)
MCP_STARTUP_FAILURE = re.compile(r"mcp[^\n]*(failed|error|timed out)[^\n]*(start|initiali[sz]e|handshake)|"
                                 r"(start|initiali[sz]e)[^\n]*mcp[^\n]*(failed|error)", re.IGNORECASE)
MODEL_SETTING_KEYS = {"reasoning_effort": "model_reasoning_effort", "reasoning_summary": "model_reasoning_summary",
                      "verbosity": "model_verbosity"}


# ---------------------------------------------------------------------------
# The Codex binary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodexBinary:
    """How to run Codex: the command prefix, where it came from, and what was verified."""

    command: tuple[str, ...]
    source: str
    path: str | None = None
    version: str | None = None
    team_identifier: str | None = None
    test_double: bool = False


def _version_key(path: Path) -> tuple[int, ...]:
    match = re.search(r"openai\.chatgpt-(\d+(?:\.\d+)*)", str(path))
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def codex_candidates() -> list[tuple[str, Path]]:
    """Where Codex is searched when neither ``codex_path`` nor ``WSE_CODEX_PATH`` names it.

    Every ``codex`` on ``PATH``, in ``PATH`` order, then the binaries bundled
    with OpenAI's VS Code extension (``~/.vscode/extensions`` and
    ``~/.vscode-insiders/extensions``), newest extension version first.
    """
    found: list[tuple[str, Path]] = []
    for directory in os.get_exec_path():
        candidate = Path(directory) / "codex"
        if candidate.is_file():
            found.append(("PATH", candidate))
    bundled: list[Path] = []
    for root in VSCODE_EXTENSION_DIRS:
        bundled += Path(root).expanduser().glob("openai.chatgpt-*/bin/*/codex")
    for candidate in sorted(bundled, key=_version_key, reverse=True):
        if candidate.is_file():
            found.append(("vscode-extension", candidate))
    return found


def code_signature(path: Path) -> tuple[str | None, str | None]:
    """``(team_identifier, problem)`` of a macOS code signature; ``problem`` is ``None`` when it verifies."""
    codesign = "/usr/bin/codesign"
    try:
        verify = subprocess.run([codesign, "--verify", "--strict", str(path)], capture_output=True, text=True,
                                timeout=60)
        shown = subprocess.run([codesign, "-dv", "--verbose=2", str(path)], capture_output=True, text=True,
                               timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"codesign could not run: {exc}"
    match = re.search(r"^TeamIdentifier=(\S+)", shown.stderr + shown.stdout, re.MULTILINE)
    team = match.group(1) if match else None
    if verify.returncode != 0:
        return team, f"the signature does not verify: {(verify.stderr or verify.stdout).strip()[:300]}"
    if team != CODEX_TEAM_IDENTIFIER:
        return team, f"it is signed by Team ID {team or 'none'}, not OpenAI's {CODEX_TEAM_IDENTIFIER}"
    return team, None


def _codex_version(path: Path) -> str | None:
    try:
        result = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=60,
                                stdin=subprocess.DEVNULL, env={"PATH": SYSTEM_PATH, "HOME": str(Path.home())})
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = result.stdout.strip() or result.stderr.strip()
    return text.splitlines()[0][:200] if result.returncode == 0 and text else None


def find_codex(explicit: str | os.PathLike[str] | None = None, *, verify: bool | None = None) -> CodexBinary:
    """The Codex binary to run: ``explicit``, else ``WSE_CODEX_PATH``, else the search of ``codex_candidates``.

    On macOS (``verify`` defaults to ``sys.platform == "darwin"``) a binary is
    accepted only when its code signature verifies and names OpenAI's Team ID;
    a searched candidate that fails is skipped, a named one is an error.
    Nothing is executed before it is verified.
    """
    verify = sys.platform == "darwin" if verify is None else verify
    named = explicit if explicit is not None else os.environ.get(CODEX_PATH_ENV) or None
    candidates = ([("codex_path" if explicit is not None else CODEX_PATH_ENV, Path(named).expanduser())]
                  if named else codex_candidates())
    rejected: list[str] = []
    for source, candidate in candidates:
        path = candidate.resolve() if candidate.exists() else candidate
        if not (path.is_file() and os.access(path, os.X_OK)):
            rejected.append(f"{candidate} ({source}): not an executable file")
            continue
        team = None
        if verify:
            team, problem = code_signature(path)
            if problem is not None:
                rejected.append(f"{candidate} ({source}): {problem}")
                continue
        return CodexBinary(command=(str(path),), source=source, path=str(path), version=_codex_version(path),
                           team_identifier=team)
    where = "; ".join(rejected) if rejected else "no codex binary on PATH or in the VS Code extension bundles"
    raise WSEError(ErrorCode.UNSUPPORTED_CAPABILITY,
                   f"no usable Codex CLI: {where}. Install OpenAI's Codex CLI or VS Code extension, or set "
                   f"{CODEX_PATH_ENV} to a codex binary"
                   + (f" signed by OpenAI (Team ID {CODEX_TEAM_IDENTIFIER})" if verify else ""),
                   details={"rejected": rejected[:16]})


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _toml_key(key: str) -> str:
    return key if _BARE_KEY.match(key) else json.dumps(key)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("TOML values must be finite")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # a JSON string with ASCII escapes is a valid TOML basic string
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, Mapping):
        return "{ " + ", ".join(f"{_toml_key(str(k))} = {_toml_value(v)}" for k, v in value.items()) + " }"
    raise TypeError(f"cannot write {type(value).__name__} as TOML")


def toml_dumps(data: Mapping[str, Any]) -> str:
    """``data`` as TOML: scalars and arrays first, then each nested mapping as a ``[dotted.table]``.

    The text is parsed back with ``tomllib`` and must equal ``data``.
    """
    lines: list[str] = []

    def table(path: list[str], mapping: Mapping[str, Any]) -> None:
        scalars = [(k, v) for k, v in mapping.items() if not isinstance(v, Mapping)]
        tables = [(k, v) for k, v in mapping.items() if isinstance(v, Mapping)]
        if path and (scalars or not tables):
            lines.append("")
            lines.append("[" + ".".join(_toml_key(part) for part in path) + "]")
        for key, value in scalars:
            lines.append(f"{_toml_key(str(key))} = {_toml_value(value)}")
        for key, value in tables:
            table([*path, str(key)], value)

    table([], data)
    text = "\n".join(lines).lstrip("\n") + "\n"
    if tomllib.loads(text) != json.loads(json.dumps(data)):
        raise ValueError("the generated TOML does not round-trip")
    return text


@dataclass(frozen=True)
class PythonAccess:
    """What the agent's commands need to run the environment's Python with the analysis libraries only.

    ``readable`` are subtrees to grant: the base installation's prefix (the
    interpreter and its standard library), a virtual environment's ``bin``
    directory and ``pyvenv.cfg`` (not the rest of the environment, which may
    sit in a repository checkout), each analysis entry, and the link
    directory. ``denied`` are the ``site-packages`` directories, so their
    other entries, WSE's own distribution among them, stay unreadable.
    ``pythonpath`` is the link directory and ``bin_dir`` the directory of the
    environment's ``python``.
    """

    bin_dir: str
    pythonpath: str
    readable: tuple[str, ...]
    denied: tuple[str, ...]
    linked: tuple[str, ...]


def python_access(link_dir: Path) -> PythonAccess:
    """Build ``link_dir`` (one symbolic link per importable analysis entry) and the matching filesystem rules.

    The entries are the analysis distributions and their declared
    dependencies (``wse.sandbox._analysis_entries``, the allow-list of WSE's
    own sandbox), found in this interpreter's ``site-packages``.
    """
    from wse.sandbox import _analysis_entries

    base, environment = Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve()
    executable = Path(sys.executable).absolute()
    runtime = [base]
    if environment != base:  # a virtual environment: its interpreter link and configuration only
        runtime += [executable.parent.resolve(), environment / "pyvenv.cfg"]
    sites = sorted({Path(p).resolve() for p in site.getsitepackages([sys.base_prefix, sys.prefix])
                    if os.path.isdir(p)})
    directories, files = _analysis_entries(set(sites))
    entries = sorted({*directories, *files})
    link_dir.mkdir(parents=True, exist_ok=True)
    for existing in link_dir.iterdir():
        if existing.is_symlink() or existing.is_file():
            existing.unlink()
    linked = []
    for entry in entries:
        if entry.parent in sites and entry.suffix != ".pth" and not (link_dir / entry.name).exists():
            (link_dir / entry.name).symlink_to(entry)
            linked.append(entry.name)
    return PythonAccess(bin_dir=str(executable.parent), pythonpath=str(link_dir),
                        readable=tuple(str(p) for p in dict.fromkeys([*runtime, *entries, link_dir.resolve()])),
                        denied=tuple(str(p) for p in sites), linked=tuple(sorted(linked)))


def skill_text() -> str:
    """The WSE skill shipped with the package (``wse/skills/wse/SKILL.md``)."""
    return resources.files("wse").joinpath("skills", SKILL_NAME, "SKILL.md").read_text(encoding="utf-8")


def filesystem_rules(sandbox_root: Path, skill_dir: Path, python: PythonAccess) -> dict[str, str]:
    """The ``[permissions.wse.filesystem]`` table: path (or Codex special path) -> ``read``, ``write``, ``deny``.

    A grant covers a whole subtree and the most specific rule wins, so the
    order of the table does not matter.
    """
    rules: dict[str, str] = {":minimal": "read", "/tmp": "deny"}
    rules[str(sandbox_root)] = "read"
    for name in ("work", "outputs", "tmp"):
        rules[str(sandbox_root / name)] = "write"
    rules[str(skill_dir)] = "read"
    for path in python.readable:
        rules[path] = "read"
    for path in python.denied:
        rules[path] = "deny"
    return rules


def shell_environment(sandbox_root: Path, python: PythonAccess) -> dict[str, str]:
    """The complete environment of the agent's commands (``shell_environment_policy.set``; nothing is inherited)."""
    work = sandbox_root / "work"
    return {
        "PATH": f"{python.bin_dir}:{SYSTEM_PATH}",
        "HOME": str(work),
        "TMPDIR": str(sandbox_root / "tmp") + "/",
        "LANG": "en_US.UTF-8",
        "PYTHONPATH": python.pythonpath,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "MPLCONFIGDIR": str(work / ".matplotlib"),
        "XDG_CACHE_HOME": str(work / ".cache"),
    }


def codex_config(*, model: str, connection: RunConnection, sandbox_root: Path, skill_dir: Path,
                 python: PythonAccess, provider_config: Mapping[str, Any] | None, startup_timeout_seconds: float,
                 tool_timeout_seconds: float, model_settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The ``config.toml`` of a run (see the module docstring); every key is one Codex 0.167 accepts strictly."""
    command = list(connection.serve_command)
    config: dict[str, Any] = {
        "model": model,
        "approval_policy": "never",
        # A login shell runs /etc/profile, whose path_helper puts /usr/bin (and its python3) before the environment's
        # interpreter; commands run as `bash -c` with exactly the PATH below.
        "allow_login_shell": False,
        "default_permissions": PROFILE_NAME,
        "check_for_update_on_startup": False,
        "web_search": "disabled",
        "project_doc_max_bytes": 0,  # no AGENTS.md or other project documents from the directories around the run
        "project_root_markers": [],  # the run's sandbox is the project root, never an enclosing checkout
    }
    for key, value in (model_settings or {}).items():
        if key not in MODEL_SETTING_KEYS:
            raise WSEError(ErrorCode.INVALID_ARGUMENT, f"codex_cli@1 does not support the model setting {key!r}; "
                           f"supported: {sorted(MODEL_SETTING_KEYS)}")
        config[MODEL_SETTING_KEYS[key]] = value
    for key, value in (provider_config or {}).items():
        config[key] = value
    config.update({
        "history": {"persistence": "none"},
        "analytics": {"enabled": False},
        "features": {**{name: False for name in DISABLED_FEATURES},
                     "code_mode": {"direct_only_tool_namespaces": [MCP_NAMESPACE],
                                   "excluded_tool_namespaces": list(EXCLUDED_TOOL_NAMESPACES)}},
        "skills": {"bundled": {"enabled": False}},
        "shell_environment_policy": {"inherit": "none", "set": shell_environment(sandbox_root, python)},
        "permissions": {PROFILE_NAME: {"filesystem": filesystem_rules(sandbox_root, skill_dir, python),
                                       "network": {"enabled": False}}},
        "mcp_servers": {MCP_SERVER_NAME: {
            "command": command[0],
            "args": command[1:],
            "cwd": connection.cwd,
            "startup_timeout_sec": max(1, math.ceil(startup_timeout_seconds)),
            "tool_timeout_sec": max(1, math.ceil(tool_timeout_seconds)),
            "required": True,
            "startup_readiness": "catalog",
            "default_tools_approval_mode": "approve",
            "enabled_tools": list(connection.allowed_tools),
            "env": dict(connection.serve_env),
        }},
    })
    return config


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def parse_mcp_result(item: Mapping[str, Any]) -> tuple[str, bool, str | None, Any]:
    """``(text, ok, error_code, parsed)`` of a Codex ``mcp_tool_call`` item.

    The text is the result's text content (or its structured content as
    JSON); a WSE domain error arrives as ``Error executing tool <name>:``
    followed by ``{"error": {...}}``, whose code is returned. A call that
    Codex itself ended (a timeout or a transport error) has an ``error``.
    """
    result = item.get("result") if isinstance(item.get("result"), Mapping) else {}
    texts = [str(part.get("text", "")) for part in result.get("content") or [] if isinstance(part, Mapping)
             and part.get("type", "text") == "text"]
    text = "".join(texts)
    if not text and result.get("structured_content") is not None:
        text = json.dumps(result["structured_content"], sort_keys=True)
    error = item.get("error")
    if error:
        message = str(error.get("message") if isinstance(error, Mapping) else error)
        code = "TOOL_TIMEOUT" if re.search(r"time[d]? ?out", message, re.IGNORECASE) else "TOOL_ERROR"
        return text or message, False, code, {"message": message}
    if text.startswith(ERROR_PREFIX) or item.get("status") == "failed":
        body = text.split(": ", 1)[1] if text.startswith(ERROR_PREFIX) and ": " in text else text
        try:
            parsed = json.loads(body)["error"]
        except (ValueError, KeyError, TypeError):
            return text, False, "TOOL_ERROR", None
        return text, False, parsed.get("code") if isinstance(parsed, Mapping) else "TOOL_ERROR", parsed
    try:
        return text, True, None, json.loads(text)
    except ValueError:
        return text, True, None, None


def _usage_fields(usage: Mapping[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    fields = {name: int(usage.get(name) or 0) for name in ("input_tokens", "cached_input_tokens", "output_tokens",
                                                            "reasoning_output_tokens")}
    fields["total_tokens"] = int(usage.get("total_tokens") or fields["input_tokens"] + fields["output_tokens"])
    return fields


def _first_line(text: str | None) -> str:
    return (text or "").strip().splitlines()[0].strip() if (text or "").strip() else ""


class _Recorder:
    """Emits normalized events to the runner's sink with sequence numbers and elapsed time."""

    def __init__(self, sink: EventSink) -> None:
        self.sink = sink
        self.started = time.monotonic()
        self.seq = 0

    def emit(self, kind: str, **fields: Any) -> None:
        from wse.database import utc_now

        self.seq += 1
        event = AgentEvent(seq=self.seq, at=utc_now(), elapsed_seconds=round(time.monotonic() - self.started, 6),
                           kind=kind, **fields)
        try:
            self.sink(event)
        except Exception:  # noqa: BLE001 - a failing sink must not break the agent run
            logger.exception("event sink failed on %s", kind)


@dataclass
class _Totals:
    """Counts and usage over all segments of a run."""

    tool_calls: dict[str, int] = field(default_factory=lambda: {"wse": 0, "shell": 0, "apply_patch": 0,
                                                                "other": 0})
    shell_seconds: float = 0.0
    responses: dict[str, dict[str, int]] = field(default_factory=dict)  # response ID -> usage, from the rollout
    turn_usage: dict[str, int] | None = None  # the latest cumulative usage of turn.completed
    reasoning_items: int = 0
    last_message: str | None = None

    @property
    def requests(self) -> int:
        return len(self.responses)


class _Translator:
    """Turns one segment's Codex JSONL events into normalized events and segment facts."""

    def __init__(self, recorder: _Recorder, totals: _Totals, segment: int) -> None:
        self.recorder = recorder
        self.totals = totals
        self.segment = segment
        self.session_id: str | None = None
        self.ending: str | None = None  # turn_completed or turn_failed
        self.errors: list[str] = []
        self.final_message: str | None = None
        self.called: set[str] = set()
        self.started_at: dict[str, float] = {}

    def _call_id(self, item: Mapping[str, Any]) -> str:
        return f"s{self.segment}:{item.get('id')}"

    def handle(self, event: Mapping[str, Any]) -> None:
        kind = event.get("type")
        if kind == "thread.started":
            self.session_id = str(event.get("thread_id")) if event.get("thread_id") else None
        elif kind == "turn.completed":
            self.ending = "turn_completed"
            self.totals.turn_usage = _usage_fields(event.get("usage") if isinstance(event.get("usage"), Mapping)
                                                   else None)
        elif kind == "turn.failed":
            self.ending = "turn_failed"
            error = event.get("error")
            self.errors.append(str(error.get("message") if isinstance(error, Mapping) else error)[:2000])
        elif kind == "error":
            self.errors.append(str(event.get("message"))[:2000])
        elif kind in ("item.started", "item.updated", "item.completed") and isinstance(event.get("item"), Mapping):
            self._item(kind, event["item"])

    def _tool_call(self, item: Mapping[str, Any], tool: str, arguments: Any) -> None:
        call_id = self._call_id(item)
        if call_id in self.called:
            return
        self.called.add(call_id)
        self.started_at[call_id] = time.monotonic()
        self.recorder.emit("tool_call", tool=tool, call_id=call_id, arguments=arguments,
                           details={"segment": self.segment, "item_type": item.get("type")})

    def _tool_result(self, item: Mapping[str, Any], tool: str, text: str, ok: bool, code: str | None,
                     summary: str | None, details: Mapping[str, Any] | None = None) -> None:
        encoded = text.encode("utf-8")
        self.recorder.emit("tool_result", tool=tool, call_id=self._call_id(item), ok=ok, error_code=code,
                           summary=summary, output=text[:OUTPUT_EXCERPT_CHARS], output_bytes=len(encoded),
                           output_hash=content_hash(encoded),
                           details={"segment": self.segment, "status": item.get("status"), **(details or {})})

    def _item(self, kind: str, item: Mapping[str, Any]) -> None:
        item_type = item.get("type")
        done = kind == "item.completed"
        if item_type == "mcp_tool_call":
            server, name = item.get("server"), str(item.get("tool"))
            tool = name if server == MCP_SERVER_NAME else f"{server}:{name}"
            self._tool_call(item, tool, item.get("arguments"))
            if done:
                self.totals.tool_calls["wse" if server == MCP_SERVER_NAME else "other"] += 1
                text, ok, code, parsed = parse_mcp_result(item)
                summary = None
                if isinstance(parsed, Mapping):
                    summary = parsed.get("summary") if ok else parsed.get("message")
                self._tool_result(item, tool, text, ok, code, None if summary is None else str(summary)[:500])
        elif item_type == "command_execution":
            self._tool_call(item, "shell", {"command": item.get("command")})
            if done:
                self.totals.tool_calls["shell"] += 1
                started = self.started_at.get(self._call_id(item))
                seconds = None if started is None else round(time.monotonic() - started, 3)
                if seconds is not None:
                    self.totals.shell_seconds += seconds
                status, exit_code = item.get("status"), item.get("exit_code")
                ok = status == "completed" and exit_code == 0
                code = None if ok else ("COMMAND_DECLINED" if status == "declined" else
                                        f"EXIT_{exit_code}" if isinstance(exit_code, int) else "COMMAND_FAILED")
                self._tool_result(item, "shell", str(item.get("aggregated_output") or ""), ok, code, None,
                                  {"exit_code": exit_code, "wall_seconds": seconds})
        elif item_type == "file_change":
            changes = item.get("changes") or []
            if done:
                self._tool_call(item, "apply_patch", {"changes": changes})
                self.totals.tool_calls["apply_patch"] += 1
                ok = item.get("status") == "completed"
                self._tool_result(item, "apply_patch", json.dumps(changes, sort_keys=True), ok,
                                  None if ok else "PATCH_FAILED", None)
        elif item_type == "agent_message":
            if done:
                text = str(item.get("text") or "")
                self.final_message = self.totals.last_message = text
                self.recorder.emit("message", output=text[:MESSAGE_EXCERPT_CHARS],
                                   output_bytes=len(text.encode("utf-8")),
                                   details={"role": "assistant", "segment": self.segment})
        elif item_type == "reasoning":
            if done:
                self.totals.reasoning_items += 1  # counted; private reasoning is never stored
        elif item_type == "error":
            if done:
                message = str(item.get("message") or "")
                self.errors.append(message[:2000])
                self.recorder.emit("message", output=message[:MESSAGE_EXCERPT_CHARS],
                                   details={"role": "codex", "error": True, "segment": self.segment})
        elif item_type in ("web_search", "collab_tool_call", "mcp_resource_call"):
            name = f"codex:{item_type}"
            self._tool_call(item, name, {k: v for k, v in item.items() if k not in ("id", "type", "status")})
            if done:
                self.totals.tool_calls["other"] += 1
                ok = item.get("status") not in ("failed", "declined")
                self._tool_result(item, name, json.dumps(item, sort_keys=True, default=str), ok,
                                  None if ok else "TOOL_ERROR", None)


class _RolloutTail:
    """Reads Codex's session record (``sessions/**/rollout-*-<session>.jsonl``) for per-request usage."""

    def __init__(self, home: Path, recorder: _Recorder, totals: _Totals) -> None:
        self.home = home
        self.recorder = recorder
        self.totals = totals
        self.path: Path | None = None
        self.offset = 0
        self.partial = b""
        self.pending_calls: bool | None = None  # whether the response being recorded called a tool
        self.last_called_tool: bool | None = None  # the same, for the latest recorded response

    def poll(self, session_id: str | None, segment: int) -> int:
        """Emit a ``model_response`` for each new usage record; return how many were new.

        ``last_called_tool`` tells whether the latest response asked for a tool
        (so Codex will make another request), from the response items that the
        record lists before that response's usage; ``None`` when unknown.
        """
        if session_id is None:
            return 0
        if self.path is None:
            matches = sorted((self.home / "sessions").glob(f"*/*/*/rollout-*-{session_id}.jsonl"))
            if not matches:
                return 0
            self.path = matches[-1]
        try:
            with self.path.open("rb") as handle:
                handle.seek(self.offset)
                data = handle.read()
        except OSError:
            return 0
        self.offset += len(data)
        lines = (self.partial + data).split(b"\n")
        self.partial = lines.pop()
        new = 0
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, Mapping):
                continue
            payload = record.get("payload") if isinstance(record.get("payload"), Mapping) else {}
            if record.get("type") == "response_item":
                if payload.get("type") in ("function_call", "custom_tool_call", "local_shell_call"):
                    self.pending_calls = True
                elif payload.get("type") == "message" and payload.get("role") == "assistant":
                    self.pending_calls = bool(self.pending_calls)
                continue
            if record.get("type") != "token_usage_record":
                continue
            response_id = str(payload.get("response_id") or f"response-{len(self.totals.responses) + 1}")
            if response_id in self.totals.responses:
                continue
            usage = _usage_fields(payload.get("usage") if isinstance(payload.get("usage"), Mapping) else None)
            self.totals.responses[response_id] = usage
            self.last_called_tool, self.pending_calls = self.pending_calls, None
            new += 1
            self.recorder.emit("model_response", usage={"requests": 1, **usage},
                               details={"response_id": response_id, "turn_id": payload.get("turn_id"),
                                        "segment": segment, "source": "codex session record",
                                        "reasoning_content": "omitted: private reasoning is never stored"})
        return new


# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------


def _process_table() -> list[tuple[int, int, str]]:
    try:
        output = subprocess.run(["/bin/ps", "-A", "-o", "pid=,ppid=,command="], capture_output=True, text=True,
                                timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in output.splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            rows.append((int(parts[0]), int(parts[1]), parts[2] if len(parts) > 2 else ""))
    return rows


def descendants(pid: int) -> dict[int, str]:
    """The live descendants of ``pid`` and their command lines (from ``ps``)."""
    children: dict[int, list[tuple[int, str]]] = {}
    for child, parent, command in _process_table():
        children.setdefault(parent, []).append((child, command))
    found: dict[int, str] = {}
    stack = [pid]
    while stack:
        for child, command in children.get(stack.pop(), []):
            if child not in found:
                found[child] = command
                stack.append(child)
    return found


def _alive_as(pid: int, command: str) -> bool:
    """Whether ``pid`` still runs ``command`` (a reused process ID runs something else)."""
    for row_pid, _, row_command in _process_table():
        if row_pid == pid:
            return row_command == command
    return False


def reap_orphans(seen: Mapping[int, str], grace_seconds: float) -> list[dict[str, Any]]:
    """Wait for the processes in ``seen`` to exit; terminate, then kill, those still running the same command."""
    deadline = time.monotonic() + grace_seconds
    alive = {pid: command for pid, command in seen.items() if _alive_as(pid, command)}
    while alive and time.monotonic() < deadline:
        time.sleep(0.1)
        alive = {pid: command for pid, command in alive.items() if _alive_as(pid, command)}
    stopped = []
    for signum in (signal.SIGTERM, signal.SIGKILL):
        for pid, command in list(alive.items()):
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                alive.pop(pid)
                continue
            stopped.append({"pid": pid, "command": command[:200], "signal": signal.Signals(signum).name})
        end = time.monotonic() + 2.0
        while alive and time.monotonic() < end:
            time.sleep(0.05)
            alive = {pid: command for pid, command in alive.items() if _alive_as(pid, command)}
        if not alive:
            break
    return stopped


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


@dataclass
class _Paths:
    root: Path
    home: Path
    skill: Path
    pythonpath: Path
    sandbox: Path


@dataclass
class _Session:
    connection: RunConnection
    config: AgentConfig
    paths: _Paths | None = None
    binary: CodexBinary | None = None
    env: dict[str, str] = field(default_factory=dict)
    prompt: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    connect_error: str | None = None
    cancel: tuple[str, str] | None = None
    loop: asyncio.AbstractEventLoop | None = None
    wake: asyncio.Event | None = None
    proxy: Any = None
    token: str | None = None
    worker: Any = None


@dataclass
class _Segment:
    index: int
    kind: str
    reason: str
    log: str
    stderr: str
    session_id: str | None = None
    exit_code: int | None = None
    ending: str | None = None
    errors: list[str] = field(default_factory=list)
    final_message: str | None = None
    signals: list[str] = field(default_factory=list)
    reaped: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    requests: int = 0

    def record(self) -> dict[str, Any]:
        return {"index": self.index, "kind": self.kind, "reason": self.reason, "session_id": self.session_id,
                "exit_code": self.exit_code, "ending": self.ending, "errors": self.errors[:5],
                "signals": self.signals, "reaped_processes": self.reaped, "elapsed_seconds": self.elapsed_seconds,
                "model_requests": self.requests, "log": self.log, "stderr": self.stderr}


class _TestDoubleModel:
    """Marks a test-double adapter for the runner's records: the strategy that chooses every tool call."""

    provider_name = TEST_DOUBLE_PROVIDER

    def __init__(self, strategy: Any) -> None:
        self.strategy = strategy


class CodexCLIAdapter(AgentAdapter):
    """The Codex CLI adapter (see the module docstring).

    ``codex_path`` names the binary (default: ``WSE_CODEX_PATH``, then the
    search of ``find_codex``); ``codex_command`` replaces it with a command
    prefix that is run unverified (test doubles only). ``provider_config``
    and ``proxy_token`` are the model provider fragment and its per-run
    token; without them the adapter starts ``wse.model_proxy.ModelProxy`` on
    the operator's ``OPENAI_BASE_URL`` and ``OPENAI_API_KEY`` and stops it in
    ``close``. ``max_resumes`` bounds the resumed segments; the grace periods
    bound the stop sequence (SIGINT, SIGTERM, SIGKILL) and the wait for
    descendant processes. ``log_dir`` defaults to the directory of the run's
    server log (``--log-file`` of the serve command), else
    ``<workspace>/logs``. ``extra_env`` adds variables to Codex's own
    environment (not its commands'); names that look like credentials are
    refused. ``tool_timeout_seconds`` overrides how long one WSE tool call
    may take (default: the run's wall-clock limit, at least the runner's
    session timeout).
    """

    runs_own_code = True  # the model runs its own commands in Codex's shell: every run is unverified

    adapter_id = ADAPTER_ID
    version = ADAPTER_VERSION

    def __init__(self, *, codex_path: str | os.PathLike[str] | None = None,
                 codex_command: Sequence[str] | None = None, provider_config: Mapping[str, Any] | None = None,
                 proxy_token: str | None = None, max_resumes: int = DEFAULT_MAX_RESUMES,
                 interrupt_grace_seconds: float = DEFAULT_INTERRUPT_GRACE_SECONDS,
                 terminate_grace_seconds: float = DEFAULT_TERMINATE_GRACE_SECONDS,
                 orphan_grace_seconds: float = DEFAULT_ORPHAN_GRACE_SECONDS, log_dir: str | os.PathLike[str] | None = None,
                 extra_env: Mapping[str, str] | None = None, verify_signature: bool | None = None,
                 tool_timeout_seconds: float | None = None, model: Any = None) -> None:
        if max_resumes < 0:
            raise WSEError(ErrorCode.INVALID_ARGUMENT, "max_resumes must be zero or more")
        from wse.runner import SECRET_NAME

        for name in extra_env or {}:
            if SECRET_NAME.search(name):
                raise WSEError(ErrorCode.POLICY_DENIED, f"extra_env variable {name} looks like a credential; Codex "
                               f"receives only the per-run {PROXY_TOKEN_ENV}")
        self.codex_path = codex_path
        self.codex_command = tuple(codex_command) if codex_command is not None else None
        self.provider_config = dict(provider_config) if provider_config is not None else None
        self.proxy_token = proxy_token
        self.max_resumes = int(max_resumes)
        self.interrupt_grace_seconds = float(interrupt_grace_seconds)
        self.terminate_grace_seconds = float(terminate_grace_seconds)
        self.orphan_grace_seconds = float(orphan_grace_seconds)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.extra_env = dict(extra_env or {})
        self.verify_signature = verify_signature
        self.tool_timeout_seconds = tool_timeout_seconds
        self.model = model  # a _TestDoubleModel for test doubles (the runner records its strategy)
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    @classmethod
    def test_double(cls, strategy: Any, **options: Any) -> CodexCLIAdapter:
        """An adapter that runs the offline Codex stand-in (``wse.agent_adapters.codex_test_double``) for ``strategy``.

        The model name ``wse-test-double:<strategy>@<version>`` tells the stand-in
        which registered strategy chooses the calls; it is a toolchain check,
        not an agent.
        """
        options.setdefault("codex_command", (sys.executable, "-m", TEST_DOUBLE_MODULE))
        options.setdefault("provider_config", {})
        return cls(model=_TestDoubleModel(strategy), **options)

    # -- lifecycle ---------------------------------------------------------------------

    def _binary(self) -> CodexBinary:
        if self.codex_command is not None:
            return CodexBinary(command=self.codex_command, source="codex_command",
                               test_double=isinstance(self.model, _TestDoubleModel))
        return find_codex(self.codex_path, verify=self.verify_signature)

    def _log_dir(self, connection: RunConnection) -> Path:
        """The directory of the run's Codex home and logs, as an absolute path.

        Codex starts in the sandbox directory, where a relative ``CODEX_HOME``
        or analysis ``PYTHONPATH`` would not resolve.
        """
        if self.log_dir is not None:
            return Path(self.log_dir).expanduser().resolve()
        command = list(connection.serve_command)
        if "--log-file" in command and command.index("--log-file") + 1 < len(command):
            return Path(command[command.index("--log-file") + 1]).expanduser().resolve().parent
        return (Path(connection.workspace) / "logs").resolve()

    def _provider(self, session: _Session) -> tuple[dict[str, Any], str | None]:
        """The model provider fragment and token: the configured ones, else a model proxy started for this run."""
        if self.provider_config is not None:
            return dict(self.provider_config), self.proxy_token
        try:
            from wse.model_proxy import ModelProxy, codex_provider_config  # type: ignore[import-not-found]
        except ImportError as exc:
            raise WSEError(ErrorCode.UNSUPPORTED_CAPABILITY, "codex_cli@1 needs a model provider: pass "
                           "provider_config and proxy_token, or install a WSE with wse.model_proxy") from exc
        upstream, key = os.environ.get("OPENAI_BASE_URL"), os.environ.get("OPENAI_API_KEY")
        if not upstream or not key:
            raise WSEError(ErrorCode.UNSUPPORTED_CAPABILITY, "codex_cli@1 starts a model proxy for the operator's "
                           "endpoint: set OPENAI_BASE_URL and OPENAI_API_KEY (they stay in this process)")
        assert session.paths is not None
        proxy = ModelProxy(upstream, key, log_path=session.paths.root / "model-proxy.jsonl")
        proxy.start()
        session.proxy = proxy
        return dict(codex_provider_config(proxy)), proxy.token

    def _environment(self, session: _Session) -> dict[str, str]:
        """Codex's own environment, from scratch: no inherited credentials, a home of its own, the proxy token."""
        assert session.paths is not None
        user_home = session.paths.root / "user-home"  # keeps the operator's ~/.agents skills and shell files out
        user_home.mkdir(parents=True, exist_ok=True)
        env = {"PATH": SYSTEM_PATH, "HOME": str(user_home), "CODEX_HOME": str(session.paths.home),
               "SHELL": "/bin/bash", "LANG": os.environ.get("LANG", "en_US.UTF-8")}
        if os.environ.get("TMPDIR"):
            env["TMPDIR"] = os.environ["TMPDIR"]
        if session.token:
            env[PROXY_TOKEN_ENV] = session.token
        env.update(self.extra_env)
        return env

    async def start(self, connection: RunConnection, agent_config: AgentConfig) -> None:
        session = _Session(connection=connection, config=agent_config)
        with self._lock:
            self._sessions[connection.experiment_id] = session
        try:
            session.binary = await asyncio.to_thread(self._binary)
            root = self._log_dir(connection) / f"codex-{connection.run_id}"
            sandbox = Path(connection.workspace).resolve() / "sandboxes" / connection.run_id
            session.paths = _Paths(root=root, home=root / "home", skill=root / "home" / "skills" / SKILL_NAME,
                                   pythonpath=root / "pythonpath", sandbox=sandbox)
            if session.paths.home.exists():
                shutil.rmtree(session.paths.home)
            for directory in (sandbox / "inputs", sandbox / "work", sandbox / "outputs", sandbox / "tmp",
                              session.paths.skill):
                directory.mkdir(parents=True, exist_ok=True)
            (session.paths.skill / "SKILL.md").write_text(skill_text(), encoding="utf-8")
            python = await asyncio.to_thread(python_access, session.paths.pythonpath)
            provider, session.token = self._provider(session)
            if "--no-worker" not in connection.serve_command:
                session.worker = await asyncio.to_thread(self._start_worker, connection)
                connection = connection.model_copy(update={"serve_command": [*connection.serve_command,
                                                                             "--no-worker"]})
            tool_timeout = (self.tool_timeout_seconds if self.tool_timeout_seconds is not None else
                            max(agent_config.client_session_timeout_seconds, agent_config.wall_clock_seconds))
            config = codex_config(model=agent_config.model, connection=connection, sandbox_root=sandbox,
                                  skill_dir=session.paths.skill, python=python, provider_config=provider,
                                  startup_timeout_seconds=agent_config.connect_timeout_seconds,
                                  tool_timeout_seconds=tool_timeout, model_settings=agent_config.model_settings)
            (session.paths.home / "config.toml").write_text(toml_dumps(config), encoding="utf-8")
            session.settings = {"worker": "adapter" if session.worker is not None else "runner",
                                "tool_timeout_sec": config["mcp_servers"][MCP_SERVER_NAME]["tool_timeout_sec"],
                                "startup_timeout_sec": config["mcp_servers"][MCP_SERVER_NAME]["startup_timeout_sec"],
                                "python": {"bin_dir": python.bin_dir, "pythonpath": python.pythonpath,
                                           "linked": list(python.linked), "denied": list(python.denied)},
                                "filesystem": config["permissions"][PROFILE_NAME]["filesystem"],
                                "provider": provider.get("model_provider")}
            session.env = self._environment(session)
            session.prompt = self.prompt(connection, session.paths)
            (root / "prompt.txt").write_text(session.prompt, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - every setup failure is reported by run()
            session.connect_error = describe_exception(exc)
            await self._stop_proxy(session)
            await self._stop_worker(session)

    @staticmethod
    def _start_worker(connection: RunConnection) -> Any:
        """The workspace's evaluation worker, with the server's environment and setup hooks (module docstring)."""
        from wse.worker import WorkerSupervisor

        return WorkerSupervisor(connection.workspace, extra_args=worker_arguments(connection.serve_command),
                                env=dict(connection.serve_env)).start()

    @staticmethod
    def prompt(connection: RunConnection, paths: _Paths) -> str:
        """The first message: the runner's instructions, how the run works in Codex, and the initial message."""
        codex = "\n".join([
            "How this run works in Codex:",
            f"- Run {connection.run_id}, experiment {connection.experiment_id}.",
            f"- Before your first tool call, read the skill `{SKILL_NAME}`: {paths.skill / 'SKILL.md'}",
            f"- Your working directory is this run's sandbox, {paths.sandbox}: inputs/ is read only; work/, outputs/, "
            "and tmp/ are writable. Your commands have no network; `python` has NumPy, PyTorch (CPU), and "
            "safetensors.",
            "- The WSE tools are the only way to change, evaluate, or select checkpoints; your own code's "
            "computation is not metered by WSE.",
            f"- To stop, reply with a final message whose first line starts with `{STOP_MARKER}:` and gives the "
            "reason. A reply without it while budget remains means you will be asked to continue.",
        ])
        return "\n\n".join([connection.instructions, codex, connection.initial_message])

    @staticmethod
    def resume_prompt(connection: RunConnection, failure: str | None = None) -> str:
        first = (f"The previous turn ended with an error ({failure[:300]}). " if failure else
                 "WSE has not ended this run: no stop condition is met yet. ")
        return (first + f"Continue experiment {connection.experiment_id} from where you stopped. If no further idea "
                f"is worth the remaining budget, reply with a final message whose first line starts with "
                f"`{STOP_MARKER}:` and gives the reason.")

    async def close(self, experiment_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(experiment_id, None)
        if session is not None:
            await self._stop_proxy(session)
            await self._stop_worker(session)

    @staticmethod
    async def _stop_worker(session: _Session) -> None:
        worker, session.worker = session.worker, None
        if worker is not None:
            try:
                await asyncio.to_thread(worker.stop, 5.0)
            except Exception:  # noqa: BLE001 - stopping the worker must not mask the run's outcome
                logger.exception("stopping the evaluation worker failed")

    @staticmethod
    async def _stop_proxy(session: _Session) -> None:
        proxy, session.proxy = session.proxy, None
        if proxy is not None:
            try:
                await asyncio.to_thread(proxy.stop)
            except Exception:  # noqa: BLE001 - stopping the proxy must not mask the run's outcome
                logger.exception("stopping the model proxy failed")

    def cancel(self, experiment_id: str, reason: str = "cancelled") -> None:
        with self._lock:
            session = self._sessions.get(experiment_id)
        if session is None:
            return
        if session.cancel is None:
            session.cancel = ("stopped" if reason not in ("cancelled", "timeout") else reason, reason)
        loop, wake = session.loop, session.wake
        if loop is None or wake is None:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            wake.set()
        else:
            loop.call_soon_threadsafe(wake.set)

    # -- the run -------------------------------------------------------------------------

    def _metadata(self, session: _Session, totals: _Totals | None, segments: list[_Segment]) -> dict[str, Any]:
        config, binary = session.config, session.binary
        counts = dict(totals.tool_calls) if totals is not None else {}
        return {
            "cli": "codex",
            "codex_version": binary.version if binary else None,
            "codex_path": binary.path if binary else None,
            "codex_source": binary.source if binary else None,
            "codex_team_identifier": binary.team_identifier if binary else None,
            "test_double": bool(binary and binary.test_double),
            "transport": "stdio",
            "permission_profile": PROFILE_NAME,
            "approval_policy": "never",
            "max_turns": config.max_turns,
            "wall_clock_seconds": config.wall_clock_seconds,
            "max_resumes": self.max_resumes,
            "resumes": max(0, len(segments) - 1),
            "segments": [segment.record() for segment in segments],
            "allowed_tools": list(session.connection.allowed_tools),
            "tool_call_counts": counts,
            "shell_seconds": round(totals.shell_seconds, 3) if totals is not None else 0.0,
            "reasoning_items": totals.reasoning_items if totals is not None else 0,
            "unmetered": "Codex ran the agent's own commands in its sandbox; WSE meters only its own tool calls, so "
                         "the computation of shell commands (counted in tool_call_counts.shell, with their wall-clock "
                         "time in shell_seconds) is not metered",
            "execution_channels": ["wse_mcp_stdio", "codex_shell_sandboxed"],
            "codex_home": str(session.paths.home) if session.paths else None,
            "prompt_sha256": content_hash(session.prompt.encode("utf-8")) if session.prompt else None,
            "model_settings": dict(config.model_settings),
            "benchmark": config.benchmark,
            "tracing_disabled": config.tracing_disabled,
            **session.settings,
        }

    def _usage(self, session: _Session, totals: _Totals | None) -> AgentUsage:
        test_double = isinstance(self.model, _TestDoubleModel)
        provider = TEST_DOUBLE_PROVIDER if test_double else session.settings.get("provider") or "codex"
        unavailable = {"cost_usd": "Codex reports token counts, not cost; price the tokens from the provider's price "
                                   "list for the recorded model",
                       "model_version": "Codex's events do not identify the served model snapshot"}
        if totals is None or (not totals.responses and totals.turn_usage is None):
            reason = "no model request was recorded"
            unavailable.update({name: reason for name in ("requests", "input_tokens", "output_tokens",
                                                          "total_tokens")})
            return AgentUsage(provider=provider, model=session.config.model, unavailable=unavailable)
        if totals.responses:
            sums = {name: sum(usage[name] for usage in totals.responses.values())
                    for name in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens",
                                 "total_tokens")}
            requests: int | None = totals.requests
        else:  # no session record: the cumulative usage of the last completed turn
            sums = dict(totals.turn_usage or {})
            requests = None
            unavailable["requests"] = "Codex's session record was not found; turn usage does not count requests"
        return AgentUsage(requests=requests, input_tokens=sums["input_tokens"], output_tokens=sums["output_tokens"],
                          total_tokens=sums["total_tokens"], cached_input_tokens=sums["cached_input_tokens"],
                          reasoning_output_tokens=sums["reasoning_output_tokens"], provider=provider,
                          model=session.config.model, unavailable=unavailable)

    async def run(self, experiment_id: str, agent_config: AgentConfig, event_sink: EventSink) -> AgentRunResult:
        with self._lock:
            session = self._sessions.get(experiment_id)
        if session is None:
            raise WSEError(ErrorCode.VERSION_CONFLICT, f"start() was not called for experiment {experiment_id}")
        if session.connect_error is not None or session.paths is None or session.binary is None:
            return AgentRunResult(status="connection_failed", reason=session.connect_error or "Codex is not set up",
                                  usage=self._usage(session, None), provider_metadata=self._metadata(session, None,
                                                                                                   []))
        session.loop = asyncio.get_running_loop()
        session.wake = asyncio.Event()
        if session.cancel is not None:
            session.wake.set()
        recorder = _Recorder(event_sink)
        totals = _Totals()
        recorder.emit("run_started", details={
            "model": agent_config.model, "max_turns": agent_config.max_turns,
            "wall_clock_seconds": agent_config.wall_clock_seconds, "max_resumes": self.max_resumes,
            "allowed_tools": list(session.connection.allowed_tools), "codex_version": session.binary.version,
            "sandbox": str(session.paths.sandbox), "tool_timeout_sec": session.settings.get("tool_timeout_sec")})
        deadline = recorder.started + agent_config.wall_clock_seconds
        segments: list[_Segment] = []
        session_id: str | None = None
        prompt, kind, why = session.prompt, "initial", "start"
        status: str | None = None
        reason = ""
        try:
            while True:
                if session.cancel is not None:
                    status, reason = session.cancel
                    break
                if time.monotonic() >= deadline:
                    status, reason = "timeout", (f"the wall-clock limit of {agent_config.wall_clock_seconds:g} s was "
                                                 "reached")
                    break
                segment = await self._segment(session, recorder, totals, len(segments) + 1, kind, why, prompt,
                                              session_id, deadline)
                segments.append(segment)
                session_id = segment.session_id or session_id
                if session.cancel is not None:
                    status, reason = session.cancel
                    break
                resumes_left = self.max_resumes - (len(segments) - 1)
                if totals.requests >= agent_config.max_turns and segment.ending != "turn_completed":
                    resumes_left = 0
                if totals.requests >= agent_config.max_turns and segment.ending == "turn_completed" and not \
                        _first_line(segment.final_message).startswith(STOP_MARKER):
                    status, reason = "max_turns", (f"the turn limit of {agent_config.max_turns} model requests was "
                                                   "reached")
                    break
                if segment.ending == "turn_completed":
                    line = _first_line(segment.final_message)
                    if line.startswith(STOP_MARKER):
                        status, reason = "completed", f"the agent stopped: {line[:500]}"
                        break
                    if resumes_left > 0 and session_id is not None:
                        prompt, kind, why = (self.resume_prompt(session.connection), "resume",
                                             "the agent ended its turn without a stop condition or WSE-STOP")
                        recorder.emit("message", output=prompt, details={"role": "user", "segment": len(segments) + 1,
                                                                         "resume": why})
                        continue
                    status, reason = "completed", (f"the agent ended its turn; {len(segments) - 1} of "
                                                   f"{self.max_resumes} resumes used")
                    break
                failure = "; ".join(segment.errors) or (f"codex exited with code {segment.exit_code} before its "
                                                        "turn completed")
                if MCP_STARTUP_FAILURE.search(failure) and totals.requests == 0 and segment.index == 1:
                    status, reason = "connection_failed", f"the WSE MCP server did not start in Codex: {failure}"
                    break
                if (resumes_left > 0 and session_id is not None and not NON_RECOVERABLE.search(failure)
                        and time.monotonic() < deadline):
                    prompt, kind, why = (self.resume_prompt(session.connection, failure), "resume",
                                         f"recoverable failure: {failure[:300]}")
                    recorder.emit("message", output=prompt, details={"role": "user", "segment": len(segments) + 1,
                                                                     "resume": why})
                    continue
                status, reason = "error", f"Codex failed: {failure}"
                break
        except Exception as exc:  # noqa: BLE001 - every adapter failure becomes a termination record
            logger.exception("the Codex run failed")
            status, reason = "error", describe_exception(exc)
        assert status is not None
        if session.token:
            leaks = await asyncio.to_thread(_files_containing, session.paths.root, session.token)
            if leaks:
                logger.error("the model proxy token appears in %d Codex files", len(leaks))
                session.settings["token_found_in"] = leaks
        recorder.emit("run_finished", details={"status": status, "reason": reason, "segments": len(segments)})
        return AgentRunResult(status=status, reason=reason, final_message=totals.last_message,
                              conversation_ref=session_id, turns=totals.requests,
                              tool_calls=sum(totals.tool_calls.values()),
                              elapsed_seconds=round(time.monotonic() - recorder.started, 6),
                              usage=self._usage(session, totals),
                              provider_metadata=self._metadata(session, totals, segments))

    def _command(self, session: _Session, prompt: str, session_id: str | None) -> list[str]:
        assert session.binary is not None and session.paths is not None
        common = ["--json", "--skip-git-repo-check", "--strict-config", "-m", session.config.model]
        if session_id is None:
            return [*session.binary.command, "exec", *common, "-C", str(session.paths.sandbox), prompt]
        return [*session.binary.command, "exec", "resume", session_id, *common, prompt]

    async def _segment(self, session: _Session, recorder: _Recorder, totals: _Totals, index: int, kind: str,
                       why: str, prompt: str, session_id: str | None, deadline: float) -> _Segment:
        """Run one ``codex exec`` (or ``exec resume``) to its end, or stop it, and record what happened."""
        assert session.paths is not None and session.wake is not None
        paths = session.paths
        segment = _Segment(index=index, kind=kind, reason=why, log=str(paths.root / f"segment-{index}.jsonl"),
                           stderr=str(paths.root / f"segment-{index}.stderr.log"), session_id=session_id)
        translator = _Translator(recorder, totals, index)
        translator.session_id = session_id
        rollout = _RolloutTail(paths.home, recorder, totals)
        started = time.monotonic()
        stderr = open(segment.stderr, "wb")  # noqa: SIM115 - closed below, after the process ends
        try:
            process = await asyncio.create_subprocess_exec(
                *self._command(session, prompt, session_id), cwd=str(paths.sandbox), env=session.env,
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=stderr,
                start_new_session=True, limit=STREAM_LIMIT_BYTES)
        except BaseException:
            stderr.close()
            raise
        seen: dict[int, str] = {}
        limit_hit = False
        requests_before = totals.requests

        async def read() -> None:
            assert process.stdout is not None
            with open(segment.log, "ab") as log:
                while True:
                    try:
                        line = await process.stdout.readline()
                    except ValueError:  # a line beyond the stream limit: keep its record, skip its content
                        line = b'{"type":"wse.line_too_long"}\n'
                    if not line:
                        break
                    log.write(line)
                    log.flush()
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, Mapping):
                        translator.handle(event)

        reader = asyncio.ensure_future(read())
        stopper: asyncio.Task | None = None
        next_snapshot = 0.0
        exited_at: float | None = None
        try:
            while not reader.done():
                now = time.monotonic()
                if process.returncode is not None:  # a descendant that kept the stream open must not hold the run
                    exited_at = exited_at or now
                    if now - exited_at > STREAM_DRAIN_SECONDS:
                        reader.cancel()
                        break
                if now >= next_snapshot:
                    seen.update(await asyncio.to_thread(descendants, process.pid))
                    next_snapshot = now + SNAPSHOT_SECONDS
                rollout.poll(translator.session_id, index)
                # The last permitted response may still answer and end the turn; one that asks for a tool, or any
                # request beyond the limit, ends the run at the turn limit.
                if not limit_hit and translator.ending is None and (
                        totals.requests > session.config.max_turns
                        or (totals.requests == session.config.max_turns and rollout.last_called_tool)):
                    limit_hit = True
                    if session.cancel is None:
                        session.cancel = ("max_turns", f"the turn limit of {session.config.max_turns} model requests "
                                                       "was reached")
                if stopper is None and (session.cancel is not None or now >= deadline or limit_hit):
                    if session.cancel is None:
                        session.cancel = ("timeout", f"the wall-clock limit of {session.config.wall_clock_seconds:g} "
                                                     "s was reached")
                    seen.update(await asyncio.to_thread(descendants, process.pid))
                    stopper = asyncio.ensure_future(self._stop(process, segment))
                waits = [reader] + ([] if stopper is not None else [asyncio.ensure_future(session.wake.wait())])
                timeout = max(0.0, min(ROLLOUT_POLL_SECONDS, deadline - time.monotonic())) if stopper is None \
                    else ROLLOUT_POLL_SECONDS
                done, pending = await asyncio.wait(waits, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                for task in waits[1:]:
                    if not task.done():
                        task.cancel()
            try:
                await reader
            except asyncio.CancelledError:
                if not reader.cancelled():
                    raise
            segment.exit_code = await process.wait()
            if stopper is not None:
                await stopper
        except asyncio.CancelledError:
            await asyncio.shield(self._stop(process, segment))
            raise
        finally:
            if process.returncode is None:
                await self._stop(process, segment)
            stderr.close()
            rollout.poll(translator.session_id, index)
            segment.reaped = await asyncio.to_thread(reap_orphans, seen, self.orphan_grace_seconds)
        segment.session_id = translator.session_id
        segment.ending = translator.ending
        segment.errors = translator.errors or ([] if segment.ending == "turn_completed" or segment.signals
                                               else _stderr_errors(Path(segment.stderr)))
        segment.final_message = translator.final_message
        segment.elapsed_seconds = round(time.monotonic() - started, 3)
        segment.requests = totals.requests - requests_before
        return segment

    async def _stop(self, process: asyncio.subprocess.Process, segment: _Segment) -> None:
        """SIGINT to Codex's process group, then SIGTERM and SIGKILL after their grace periods."""
        for signum, grace in ((signal.SIGINT, self.interrupt_grace_seconds),
                              (signal.SIGTERM, self.terminate_grace_seconds), (signal.SIGKILL, 10.0)):
            if process.returncode is not None:
                return
            try:
                os.killpg(process.pid, signum)
            except (ProcessLookupError, PermissionError):
                pass
            segment.signals.append(signal.Signals(signum).name)
            try:
                await asyncio.wait_for(process.wait(), timeout=grace)
                return
            except TimeoutError:
                continue


def _stderr_errors(path: Path, tail_bytes: int = 8192) -> list[str]:
    """The error lines at the end of a segment's standard error (Codex reports a failed start only there)."""
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - tail_bytes))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return []
    errors = [line.strip() for line in lines if re.search(r"\berror\b", line, re.IGNORECASE)]
    return [line[:2000] for line in errors[-3:]]


def worker_arguments(serve_command: Sequence[str]) -> list[str]:
    """The worker options that a ``wse serve`` command would pass its worker: ``--setup`` hooks and ``--worker-args``."""
    import shlex

    command, arguments = list(serve_command), []
    for index, item in enumerate(command[:-1]):
        if item == "--setup":
            arguments += ["--setup", command[index + 1]]
        elif item == "--worker-args":
            arguments += shlex.split(command[index + 1])
    return arguments


def _files_containing(root: Path, secret: str, max_bytes: int = 256 << 20) -> list[str]:
    """Files under ``root`` (not following links) whose bytes contain ``secret``; their paths only."""
    needle = secret.encode("utf-8")
    found = []
    for directory, _, names in os.walk(root, followlinks=False):
        for name in names:
            path = Path(directory) / name
            try:
                if path.is_symlink() or path.stat().st_size > max_bytes:
                    continue
                if needle in path.read_bytes():
                    found.append(str(path))
            except OSError:
                continue
    return found
