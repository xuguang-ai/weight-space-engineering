"""An offline stand-in for the Codex CLI, driven by a registered WSE strategy: a toolchain check, not an agent.

``python -m wse.agent_adapters.codex_test_double exec ...`` accepts the
arguments that ``CodexCLIAdapter`` passes to ``codex exec`` and
``codex exec resume <session_id>`` and answers the way Codex does, so the
adapter's whole path runs without a model or the network:

* it reads ``$CODEX_HOME/config.toml`` that the adapter wrote and starts the
  configured stdio MCP server (``mcp_servers.wse``: the run's ``wse serve``,
  with its command, arguments, working directory, and environment) and
  talks to it with a small JSON-RPC client of its own (the legacy MCP
  handshake, as Codex 0.167 does; the MCP SDK stays in ``wse.mcp_server``
  only), and refuses a tool outside ``enabled_tools``;
* the model name ``wse-test-double:<strategy>@<version>`` names a registered
  stepwise strategy (resolved with the built-in registry and the server's
  ``--setup`` hooks), which chooses every tool call from the tool results so
  far: the first call reads the experiment, and
  a ``stop`` action becomes the final message ``WSE-STOP: <reason>``;
* it prints Codex-shaped JSONL events (``thread.started``, ``turn.started``,
  ``item.started``/``item.completed`` of ``mcp_tool_call`` items,
  ``agent_message``, ``turn.completed`` with cumulative usage) and appends
  one ``token_usage_record`` per call to a session record
  ``$CODEX_HOME/sessions/YYYY/MM/DD/rollout-<time>-<session>.jsonl``;
  usage is synthetic (about four characters per token);
* a session keeps its observations in ``$CODEX_HOME/test-double/<session>.json``,
  so ``exec resume`` continues where the session stopped, and each segment
  appends what the model would be shown (the prompt and the offered tools
  with their schemas) to ``<session>.requests.jsonl`` next to it.

A script in place of a strategy (``WSE_CODEX_TEST_DOUBLE`` with ``"script"``,
for tests of the adapter; any model name) plays fixed steps, one per model
request: ``["call", tool, arguments]``, ``["say", text]`` (a final message),
``["sleep", seconds]`` and ``["interrupt", seconds]`` (SIGINT to the runner,
this process's parent, as from the operator's Ctrl-C), each followed by the
final message ``done``. After its end the script repeats its last step.

``WSE_CODEX_TEST_DOUBLE`` (JSON) adds behaviors for tests of the adapter:
``end_turn_after`` (end the first segment after that many calls without
stopping), ``fail_first`` (end the first segment with ``turn.failed``),
``hang`` (after the first call, wait for a signal: ``"sleep"``;
``"ignore-sigint"``, which ignores SIGINT; or ``"ignore-signals"``, which also
ignores SIGTERM), ``request_seconds`` (the latency of each model
request), ``grandchild`` (start a long-running process in a session of its
own, to test that no process outlives the run), and ``no_rollout`` (keep no
session record).
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import tomllib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PREFIX = "wse-test-double:"
STOP_MARKER = "WSE-STOP"


def emit(event: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(event, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def parse(argv: list[str]) -> dict[str, Any]:
    """The subset of ``codex exec`` arguments the adapter passes."""
    if not argv or argv[0] != "exec":
        raise SystemExit("usage: codex_test_double exec [resume SESSION_ID] [--json] [-C DIR] -m MODEL PROMPT")
    rest, options = argv[1:], {"session_id": None, "cd": None, "model": None, "prompt": None, "flags": []}
    if rest[:1] == ["resume"]:
        options["session_id"] = rest[1]
        rest = rest[2:]
    while rest:
        item = rest.pop(0)
        if item in ("-C", "--cd"):
            options["cd"] = rest.pop(0)
        elif item in ("-m", "--model"):
            options["model"] = rest.pop(0)
        elif item.startswith("-"):
            options["flags"].append(item)
        else:
            options["prompt"] = item
    return options


class Session:
    """The persistent state of one stand-in session: its observations and its usage."""

    def __init__(self, home: Path, session_id: str | None) -> None:
        self.home = home
        self.id = session_id or str(uuid.uuid4())
        self.path = home / "test-double" / f"{self.id}.json"
        state = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.observations: list[dict[str, Any]] = state.get("observations", [])
        self.segments: int = int(state.get("segments", 0)) + 1
        self.calls: int = int(state.get("calls", 0))
        self.step: int = int(state.get("step", 0))
        self.usage: dict[str, int] = state.get("usage") or {"input_tokens": 0, "cached_input_tokens": 0,
                                                             "cache_write_input_tokens": 0, "output_tokens": 0,
                                                             "reasoning_output_tokens": 0}
        self.rollout = self._rollout_path(state.get("rollout"))

    def _rollout_path(self, existing: str | None) -> Path:
        if existing:
            return Path(existing)
        now = datetime.now(UTC)
        return (self.home / "sessions" / now.strftime("%Y/%m/%d") /
                f"rollout-{now.strftime('%Y-%m-%dT%H-%M-%S')}-{self.id}.jsonl")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"observations": self.observations, "segments": self.segments,
                                         "calls": self.calls, "step": self.step, "usage": self.usage,
                                         "rollout": str(self.rollout)}))

    def charge(self, prompt_chars: int, produced_chars: int, record: bool, item: dict[str, Any] | None = None) -> None:
        """Count one model request; with ``record``, append its response item and usage to the session record."""
        self.calls += 1
        usage = {"input_tokens": prompt_chars // 4 + 1, "cached_input_tokens": 0, "cache_write_input_tokens": 0,
                 "output_tokens": produced_chars // 4 + 1, "reasoning_output_tokens": 0}
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        for name in self.usage:
            self.usage[name] += usage.get(name, 0)
        if record:
            self.rollout.parent.mkdir(parents=True, exist_ok=True)
            with self.rollout.open("a", encoding="utf-8") as handle:
                if item is not None:
                    handle.write(json.dumps({"timestamp": datetime.now(UTC).isoformat(), "type": "response_item",
                                             "payload": item}) + "\n")
                handle.write(json.dumps({"timestamp": datetime.now(UTC).isoformat(), "type": "token_usage_record",
                                         "payload": {"thread_id": self.id, "response_id": f"td_{self.id}_{self.calls}",
                                                     "usage": usage}}) + "\n")


def strategy_for(model: str, server: dict[str, Any]) -> Any:
    """The registered strategy that the model name names, with the server's setup hooks applied."""
    from wse.registry import builtin_registry
    from wse.worker import _load_setup

    for entry in str((server.get("env") or {}).get("PYTHONPATH", "")).split(os.pathsep):
        if entry and entry not in sys.path:
            sys.path.insert(0, entry)
    registry = builtin_registry()
    args = list(server.get("args") or [])
    for index, arg in enumerate(args[:-1]):
        if arg == "--setup":
            _load_setup(args[index + 1])(registry)
    ref = registry.resolve_reference("strategy", model[len(PREFIX):])
    return registry.resolve("strategy", ref.extension_id, ref.version)()


def observe(tool: str, arguments: dict[str, Any], text: str, is_error: bool) -> dict[str, Any]:
    """An ``Observation`` (as JSON) of one tool result, as a strategy receives it from the runner."""
    if is_error:
        body = text.split(": ", 1)[1] if ": " in text else text
        try:
            error = json.loads(body)["error"]
        except (ValueError, KeyError, TypeError):
            error = {"code": "TOOL_ERROR", "message": text}
        return {"tool": tool, "arguments": arguments, "ok": False, "error": error}
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = {}
    return {"tool": tool, "arguments": arguments, "ok": True, "result": parsed.get("result"),
            "_usage": (parsed.get("usage") or {}).get("cumulative")}


def next_action(strategy: Any, observations: list[dict[str, Any]]) -> Any:
    from wse.schemas import BudgetEntry, ObjectiveSpec
    from wse.strategies.base import Action, Observation

    if not observations:
        return Action.call("wse_get_experiment", "Read the task and budget.")
    budget: dict[str, BudgetEntry] = {}
    objective = None
    for observation in observations:
        for name, entry in (observation.get("_usage") or {}).items():
            budget[name] = BudgetEntry(**entry)
        if objective is None and observation["tool"] == "wse_get_experiment" and observation["ok"]:
            task = (observation.get("result") or {}).get("task") or {}
            objective = ObjectiveSpec(**task["objective"]) if task.get("objective") else None
    history = [Observation(**{k: v for k, v in observation.items() if not k.startswith("_")})
               for observation in observations]
    return strategy.next_action(objective, history, budget)


class StdioClient:
    """A minimal MCP client over stdio (JSON-RPC 2.0, one message per line), as a host other than the MCP SDK speaks it.

    It performs the legacy ``initialize`` handshake (protocol 2025-06-18, as
    Codex 0.167 does), lists tools, and calls them. The server runs in a
    session of its own, as Codex starts it, and gets exactly the configured
    environment.
    """

    PROTOCOL = "2025-06-18"

    def __init__(self, server: dict[str, Any]) -> None:
        self.server = server
        self.process: asyncio.subprocess.Process | None = None
        self.next_id = 0

    async def start(self) -> list[dict[str, Any]]:
        self.process = await asyncio.create_subprocess_exec(
            self.server["command"], *list(self.server.get("args") or []), cwd=self.server.get("cwd"),
            env=dict(self.server.get("env") or {}), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=sys.stderr, start_new_session=True, limit=64 << 20)
        await self.request("initialize", {"protocolVersion": self.PROTOCOL, "capabilities": {},
                                          "clientInfo": {"name": "wse-codex-test-double", "version": "1"}})
        await self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        tools, cursor = [], None
        while True:
            listed = await self.request("tools/list", {"cursor": cursor} if cursor else {})
            tools += [tool for tool in listed.get("tools", []) if isinstance(tool, dict)]
            cursor = listed.get("nextCursor")
            if not cursor:
                return tools

    async def send(self, message: dict[str, Any]) -> None:
        assert self.process is not None and self.process.stdin is not None
        self.process.stdin.write((json.dumps(message) + "\n").encode())
        await self.process.stdin.drain()

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert self.process is not None and self.process.stdout is not None
        self.next_id += 1
        ident = self.next_id
        await self.send({"jsonrpc": "2.0", "id": ident, "method": method, "params": params})
        while True:
            line = await self.process.stdout.readline()
            if not line:
                raise ConnectionError(f"the MCP server closed its output during {method}")
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if message.get("id") == ident and "method" not in message:
                if "error" in message:
                    raise ConnectionError(f"{method} failed: {message['error']}")
                return message.get("result") or {}
            if "method" in message and "id" in message:  # a request from the server: this host supports none
                await self.send({"jsonrpc": "2.0", "id": message["id"],
                                 "error": {"code": -32601, "message": "method not found"}})

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool, Any]:
        """``(text, is_error, structured_content)`` of one tool call."""
        result = await self.request("tools/call", {"name": name, "arguments": arguments})
        text = "".join(str(part.get("text", "")) for part in result.get("content") or [] if isinstance(part, dict))
        return text, bool(result.get("isError")), result.get("structuredContent")

    async def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except TimeoutError:
            process.kill()
            await process.wait()


async def drive(options: dict[str, Any], config: dict[str, Any], session: Session, behavior: dict[str, Any]) -> int:
    server = config["mcp_servers"]["wse"]
    enabled = set(server.get("enabled_tools") or [])
    record = not behavior.get("no_rollout")
    first_segment = session.segments == 1
    client = StdioClient(server)
    try:
        try:
            listed = await client.start()
        except Exception as exc:  # noqa: BLE001 - reported the way Codex reports a failed MCP start
            print(f"Error: thread/start failed: required MCP servers failed to initialize: wse: {exc!r}",
                  file=sys.stderr)
            return 1
        strategy = None
        if behavior.get("script") is None:  # the model, like Codex's, matters only once the session exists
            if not str(options["model"] or "").startswith(PREFIX):
                print(f"Error: the Codex test double needs a model {PREFIX}<strategy>@<version> or a script, got "
                      f"{options['model']!r}", file=sys.stderr)
                return 2
            strategy = strategy_for(options["model"], server)
        return await converse(client, listed, enabled, strategy, options, session, behavior, first_segment, record)
    finally:
        await client.close()


def script_step(script: list[list[Any]], session: Session) -> list[Any]:
    """The next step of a script; after its end, its last step again."""
    step = script[min(session.step, len(script) - 1)]
    session.step += 1
    return list(step)


async def converse(client: Any, tools: list[dict[str, Any]], enabled: set[str], strategy: Any,
                   options: dict[str, Any], session: Session, behavior: dict[str, Any], first_segment: bool,
                   record: bool) -> int:
    """One turn: the calls of the strategy or the script until it stops, a behavior ends the turn, or a signal."""
    from wse.strategies.base import Action

    emit({"type": "thread.started", "thread_id": session.id})
    emit({"type": "turn.started"})
    listed = {tool.get("name") for tool in tools}
    offered = [tool for tool in tools if not enabled or tool.get("name") in enabled]
    with (session.home / "test-double" / f"{session.id}.requests.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"segment": session.segments, "model": options["model"],
                                 "prompt": options.get("prompt"), "tools": offered}) + "\n")
    script = behavior.get("script")
    missing = sorted(enabled - listed)
    if missing:
        emit({"type": "error", "message": f"MCP server wse does not offer the enabled tools {missing}"})
        return 1
    if behavior.get("grandchild") and first_segment:
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"], start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    prompt_chars = len(options.get("prompt") or "")
    calls = 0
    while True:
        if first_segment and behavior.get("fail_first") and calls >= 1:
            session.save()
            emit({"type": "turn.failed", "error": {"message": "stream disconnected before completion: "
                                                              "test-double transient failure"}})
            return 1
        if first_segment and behavior.get("end_turn_after") and calls >= int(behavior["end_turn_after"]):
            text = "Pausing here (test double)."
            session.charge(prompt_chars, len(text), record, {"type": "message", "role": "assistant"})
            emit({"type": "item.completed", "item": {"id": f"item_{calls}", "type": "agent_message", "text": text}})
            break
        if behavior.get("hang") and calls >= 1:
            if behavior["hang"] in ("ignore-sigint", "ignore-signals"):
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            if behavior["hang"] == "ignore-signals":
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
            await asyncio.sleep(3600)
        await asyncio.sleep(float(behavior.get("request_seconds", 0)))  # the latency of a model request
        if script is not None:
            step = script_step(script, session)
            if step[0] in ("sleep", "interrupt"):  # one model request that takes this long, then says "done"
                session.charge(prompt_chars, 4, record, {"type": "message", "role": "assistant"})
                if step[0] == "interrupt":  # the operator's Ctrl-C reaches the runner, this process's parent
                    os.kill(os.getppid(), signal.SIGINT)
                await asyncio.sleep(float(step[1]))
                emit({"type": "item.completed", "item": {"id": f"item_{calls}", "type": "agent_message",
                                                         "text": "done"}})
                break
            if step[0] == "say":
                text = str(step[1])
                session.charge(prompt_chars, len(text), record, {"type": "message", "role": "assistant"})
                emit({"type": "item.completed", "item": {"id": f"item_{calls}", "type": "agent_message",
                                                         "text": text}})
                break
            action = Action(kind="call", tool=str(step[1]), arguments=dict(step[2]))
        else:
            action = next_action(strategy, session.observations)
        if action.kind == "stop":
            text = f"{STOP_MARKER}: {action.reason}"
            session.charge(prompt_chars, len(text), record, {"type": "message", "role": "assistant"})
            emit({"type": "item.completed", "item": {"id": f"item_{calls}", "type": "agent_message", "text": text}})
            break
        item = {"id": f"item_{calls}", "type": "mcp_tool_call", "server": "wse", "tool": action.tool,
                "arguments": action.arguments, "result": None, "error": None, "status": "in_progress"}
        session.charge(prompt_chars + len(json.dumps(session.observations)), len(json.dumps(action.arguments)),
                       record, {"type": "function_call", "name": action.tool, "namespace": "mcp__wse"})
        prompt_chars = 0
        emit({"type": "item.started", "item": item})
        if enabled and action.tool not in enabled:
            text, is_error = f"Error executing tool {action.tool}: unknown tool", True
            result = {"content": [{"type": "text", "text": text}], "structured_content": None}
        else:
            text, is_error, structured = await client.call_tool(action.tool, action.arguments)
            result = {"content": [{"type": "text", "text": text}],
                      "structured_content": None if is_error else structured}
        emit({"type": "item.completed", "item": {**item, "result": result,
                                                 "status": "failed" if is_error else "completed"}})
        session.observations.append(observe(action.tool, action.arguments, text, is_error))
        session.save()
        calls += 1
    session.save()
    emit({"type": "turn.completed", "usage": dict(session.usage)})
    return 0


def main(argv: list[str] | None = None) -> int:
    options = parse(list(sys.argv[1:] if argv is None else argv))
    home = Path(os.environ["CODEX_HOME"])
    config = tomllib.loads((home / "config.toml").read_text(encoding="utf-8"))
    options["model"] = options["model"] or config.get("model")
    behavior = json.loads(os.environ.get("WSE_CODEX_TEST_DOUBLE") or "{}")
    if options["cd"]:
        os.chdir(options["cd"])
    session = Session(home, options["session_id"])
    session.save()
    try:
        return asyncio.run(drive(options, config, session, behavior))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
