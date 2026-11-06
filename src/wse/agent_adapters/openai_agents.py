"""The M1 reference agent adapter: OpenAI Agents SDK for Python over local stdio MCP.

``OpenAIAgentsAdapter`` runs the SDK's own ``Agent``/``Runner`` loop with one
MCP server, ``agents.mcp.MCPServerStdio`` running ``wse serve`` bound to the
run grant, and no other tool: the agent's only execution channel is the
allow-listed WSE tools (a client-side static tool filter). The SDK handles
the conversation and model calls; every weight operation, job, budget check,
selection, and evaluation stays in WSE.

* **Limits.** ``Runner.run`` always receives ``max_turns`` explicitly (the
  configured 100 by default; the SDK's own default is 10), and the loop runs
  under a wall-clock limit (900 s by default), independently of WSE's
  computation budget.
* **Tracing.** The SDK traces by default and exports tool inputs and outputs
  to ``api.openai.com``; benchmark configurations disable it
  (``RunConfig(tracing_disabled=True)``, no sensitive trace data), and the
  setting is reported in ``provider_metadata`` for exports.
* **Events and usage.** Run hooks normalize model responses (per request:
  token usage, the latency of the SDK's model call including the client's
  retries, and the provider's response and request IDs; reasoning items are
  counted, their content is never stored), tool
  calls, and tool results (ok or the domain error code, a bounded excerpt,
  size and hash) into ``AgentEvent`` records for the runner's event sink.
  Cost and the served model snapshot are not reported by the SDK, so those
  usage fields are ``None`` with a reason. The usage provider is the class
  of the endpoint the SDK calls (``provider_for_base_url`` of
  ``OPENAI_BASE_URL``: ``openai``, ``azure_openai``, or
  ``openai_compatible``); the endpoint URL itself is never recorded.
* **Model client.** A run that names its model gets its own ``AsyncOpenAI``
  client, configured from ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` like the
  SDK's default client, and closes it when the run ends. The SDK's default
  client shares one process-wide HTTP connection pool, whose connections
  belong to the event loop that opened them, and each trial runs in its own
  event loop.
* **Cancellation.** ``cancel`` cancels the SDK run task (from any thread); the
  runner then settles WSE jobs and freezes while the server stays connected,
  and finally calls ``close``.

The SDK is an optional extra (``agents-openai``); it is imported lazily, so
``import wse`` and this module work without it, and using the adapter without
it fails with an actionable ``UNSUPPORTED_CAPABILITY`` error.

``strategy_model`` builds an SDK test double: an implementation of the SDK's
``Model`` interface whose tool calls are chosen by a registered stepwise
``Strategy`` from the tool outputs it receives. It makes no network call and
lets the real SDK runner drive a real ``wse serve`` deterministically, for
tests and offline demonstrations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from urllib.parse import urlsplit

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

__all__ = ["ERROR_PREFIX", "OpenAIAgentsAdapter", "parse_tool_output", "provider_for_base_url", "strategy_model"]

logger = logging.getLogger("wse.agents.openai")

ERROR_PREFIX = "Error executing tool "
OUTPUT_EXCERPT_CHARS = 2000
MESSAGE_EXCERPT_CHARS = 2000
TEST_DOUBLE_PROVIDER = "wse-test-double"
AZURE_OPENAI_HOST_SUFFIXES = (".openai.azure.com", ".services.ai.azure.com", ".cognitiveservices.azure.com")


def provider_for_base_url(base_url: str | None) -> str:
    """The class of the endpoint that the SDK's OpenAI client calls, given ``OPENAI_BASE_URL``.

    ``openai`` for the OpenAI API (the client's default when the variable is
    unset, or a regional ``*.api.openai.com`` host), ``azure_openai`` for an
    Azure OpenAI or Azure AI Foundry resource, and ``openai_compatible`` for
    any other value. Only this class is recorded: the URL can name a private
    resource, so it never enters controller records or exports.
    """
    if base_url is None:
        return "openai"
    host = (urlsplit(base_url.strip()).hostname or "").lower()
    if host == "api.openai.com" or host.endswith(".api.openai.com"):
        return "openai"
    if host.endswith(AZURE_OPENAI_HOST_SUFFIXES):
        return "azure_openai"
    return "openai_compatible"


def _require_sdk() -> Any:
    try:
        import agents
    except ImportError as exc:
        raise WSEError(
            ErrorCode.UNSUPPORTED_CAPABILITY,
            "the OpenAI Agents SDK adapter needs the optional 'agents-openai' extra (package openai-agents); "
            "install the project with that extra",
            details={"extra": "agents-openai"},
        ) from exc
    return agents


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _utc_now() -> str:
    from wse.database import utc_now

    return utc_now()


def parse_tool_output(result: Any) -> tuple[str, bool, str | None, Any]:
    """Return ``(text, ok, error_code, parsed_json)`` of an MCP tool output as the SDK reports it."""
    if isinstance(result, dict) and "text" in result:
        text = str(result["text"])
    elif isinstance(result, list):
        text = "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in result)
    else:
        text = str(result)
    if text.startswith(ERROR_PREFIX):
        body = text.split(": ", 1)[1] if ": " in text else ""
        try:
            error = json.loads(body)["error"]
        except (ValueError, KeyError, TypeError):
            return text, False, "TOOL_ERROR", None
        return text, False, error.get("code"), error
    try:
        return text, True, None, json.loads(text)
    except ValueError:
        return text, True, None, None


@dataclass
class _Session:
    connection: RunConnection
    server: Any = None
    connect_error: str | None = None
    tools_seen: list[str] = field(default_factory=list)
    protocol_version: str | None = None
    task: asyncio.Task | None = None
    loop: asyncio.AbstractEventLoop | None = None
    cancel: tuple[str, str] | None = None


class _Recorder:
    """Normalizes SDK hook callbacks into ``AgentEvent`` records and running usage totals."""

    def __init__(self, sink: EventSink) -> None:
        self.sink = sink
        self.started = time.monotonic()
        self.seq = 0
        self.turns = 0
        self.tool_calls = 0
        self.tokens = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self.last_message: str | None = None
        self.request_started: float | None = None

    def emit(self, kind: str, **fields: Any) -> None:
        self.seq += 1
        event = AgentEvent(seq=self.seq, at=_utc_now(), elapsed_seconds=round(time.monotonic() - self.started, 6),
                           kind=kind, **fields)
        try:
            self.sink(event)
        except Exception:  # noqa: BLE001 - a failing sink must not break the SDK loop
            logger.exception("event sink failed on %s", kind)

    def model_request(self) -> None:
        self.request_started = time.monotonic()

    def model_response(self, response: Any) -> None:
        # The SDK's whole model call: the client's retries are inside it.
        latency = None if self.request_started is None else round(time.monotonic() - self.request_started, 6)
        self.request_started = None
        self.turns += 1
        usage = getattr(response, "usage", None)
        counted = {name: int(getattr(usage, name, 0) or 0) for name in self.tokens}
        for name, value in counted.items():
            self.tokens[name] += value
        kinds: dict[str, int] = {}
        texts: list[str] = []
        for item in getattr(response, "output", []) or []:
            kind = getattr(item, "type", type(item).__name__)
            kinds[kind] = kinds.get(kind, 0) + 1
            if kind == "message":
                for part in getattr(item, "content", []) or []:
                    if getattr(part, "type", None) == "output_text":
                        texts.append(part.text)
        details: dict[str, Any] = {"output_items": kinds, "latency_seconds": latency,
                                   "response_id": getattr(response, "response_id", None),
                                   "request_id": getattr(response, "request_id", None)}
        if kinds.get("reasoning"):
            details["reasoning_content"] = "omitted: private reasoning is never stored"
        self.emit("model_response", usage=counted, details=details)
        if texts:
            self.last_message = "\n".join(texts)
            self.emit("message", output=self.last_message[:MESSAGE_EXCERPT_CHARS],
                      output_bytes=len(self.last_message.encode("utf-8")))

    def tool_call(self, context: Any, tool: Any) -> None:
        self.tool_calls += 1
        raw = getattr(context, "tool_arguments", None)
        try:
            arguments = json.loads(raw) if isinstance(raw, str) and raw else raw
        except ValueError:
            arguments = raw
        self.emit("tool_call", tool=getattr(tool, "name", None), call_id=getattr(context, "tool_call_id", None),
                  arguments=arguments)

    def tool_result(self, context: Any, tool: Any, result: Any) -> None:
        text, ok, code, parsed = parse_tool_output(result)
        summary = parsed.get("summary") if ok and isinstance(parsed, dict) else None
        if not ok and isinstance(parsed, dict):
            summary = parsed.get("message")
        encoded = text.encode("utf-8")
        self.emit("tool_result", tool=getattr(tool, "name", None), call_id=getattr(context, "tool_call_id", None),
                  ok=ok, error_code=code, summary=summary, output=text[:OUTPUT_EXCERPT_CHARS],
                  output_bytes=len(encoded), output_hash=content_hash(encoded))


class OpenAIAgentsAdapter(AgentAdapter):
    """OpenAI Agents SDK adapter (see the module docstring).

    ``model`` is the operator-chosen model name, or an SDK ``Model`` instance
    (a test double); when omitted, the ``AgentConfig.model`` name is used.
    Constructing the adapter without the optional SDK fails with
    ``UNSUPPORTED_CAPABILITY``.
    """

    adapter_id = "openai_agents"
    version = "1"

    def __init__(self, model: Any = None) -> None:
        _require_sdk()  # fail before the runner records a controller for the run grant
        self.model = model
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------------------

    async def start(self, connection: RunConnection, agent_config: AgentConfig) -> None:
        agents = _require_sdk()
        from agents.mcp import MCPServerStdio
        from agents.mcp.util import create_static_tool_filter

        if agent_config.tracing_disabled:
            agents.set_tracing_disabled(True)
        session = _Session(connection=connection)
        with self._lock:
            self._sessions[connection.experiment_id] = session
        command = list(connection.serve_command)
        session.server = MCPServerStdio(
            {"command": command[0], "args": command[1:], "env": dict(connection.serve_env), "cwd": connection.cwd},
            name="wse",
            cache_tools_list=True,
            client_session_timeout_seconds=agent_config.client_session_timeout_seconds,
            tool_filter=create_static_tool_filter(allowed_tool_names=list(connection.allowed_tools)),
        )
        try:
            async with asyncio.timeout(agent_config.connect_timeout_seconds):
                await session.server.connect()
                session.tools_seen = sorted(tool.name for tool in await session.server.list_tools())
            session.protocol_version = getattr(session.server.session, "protocol_version", None)
            missing = sorted(set(connection.allowed_tools) - set(session.tools_seen))
            if missing:
                raise WSEError(ErrorCode.NOT_FOUND, f"the WSE server does not offer the allowed tools {missing}")
        except Exception as exc:  # noqa: BLE001 - every connection failure is reported by run()
            session.connect_error = describe_exception(exc)
            await self._cleanup(session)

    async def close(self, experiment_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(experiment_id, None)
        if session is not None:
            await self._cleanup(session)

    @staticmethod
    async def _cleanup(session: _Session) -> None:
        server, session.server = session.server, None
        if server is not None:
            try:
                await server.cleanup()
            except Exception:  # noqa: BLE001 - cleanup after a failure must not mask it
                logger.exception("MCP server cleanup failed")

    def cancel(self, experiment_id: str, reason: str = "cancelled") -> None:
        with self._lock:
            session = self._sessions.get(experiment_id)
        if session is None:
            return
        if session.cancel is None:
            session.cancel = ("stopped" if reason not in ("cancelled", "timeout") else reason, reason)
        task, loop = session.task, session.loop
        if task is None or loop is None or task.done():
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            task.cancel()
        else:
            loop.call_soon_threadsafe(task.cancel)

    # -- the SDK loop --------------------------------------------------------------------

    def _metadata(self, session: _Session, config: AgentConfig) -> dict[str, Any]:
        return {
            "sdk": "openai-agents",
            "sdk_version": _version("openai-agents"),
            "openai_version": _version("openai"),
            "mcp_version": _version("mcp"),
            "mcp_protocol_version": session.protocol_version,
            "transport": "stdio",
            "tracing_disabled": config.tracing_disabled,
            "trace_include_sensitive_data": not config.tracing_disabled and not config.benchmark,
            "max_turns": config.max_turns,
            "wall_clock_seconds": config.wall_clock_seconds,
            "allowed_tools": list(session.connection.allowed_tools),
            "tools_seen": list(session.tools_seen),
            "use_structured_content": False,
            "model_settings": dict(config.model_settings),
            "execution_channels": ["wse_mcp_stdio"],
            "benchmark": config.benchmark,
        }

    def _usage(self, recorder: _Recorder | None, config: AgentConfig) -> AgentUsage:
        test_double = self.model is not None and not isinstance(self.model, str)
        provider = (getattr(self.model, "provider_name", type(self.model).__name__) if test_double
                    else provider_for_base_url(os.environ.get("OPENAI_BASE_URL")))
        unavailable = {
            "cost_usd": "OpenAI Agents SDK reports token counts, not cost; price the tokens from the provider's "
                        "price list for the recorded model",
            "model_version": "the SDK's model responses do not identify the served model snapshot",
        }
        if recorder is None:
            reason = "no model request was made (the MCP connection failed before the SDK loop started)"
            unavailable.update({name: reason for name in ("requests", "input_tokens", "output_tokens",
                                                          "total_tokens")})
            return AgentUsage(provider=provider, model=config.model, unavailable=unavailable)
        return AgentUsage(provider=provider, model=config.model, unavailable=unavailable, **recorder.tokens)

    async def run(self, experiment_id: str, agent_config: AgentConfig, event_sink: EventSink) -> AgentRunResult:
        with self._lock:
            session = self._sessions.get(experiment_id)
        if session is None:
            raise WSEError(ErrorCode.VERSION_CONFLICT, f"start() was not called for experiment {experiment_id}")
        if session.connect_error is not None or session.server is None:
            return AgentRunResult(status="connection_failed",
                                  reason=session.connect_error or "the MCP server is not connected",
                                  usage=self._usage(None, agent_config),
                                  provider_metadata=self._metadata(session, agent_config))
        _require_sdk()
        from agents import Agent, ModelSettings, MultiProvider, RunConfig, Runner, RunHooks
        from agents.exceptions import MaxTurnsExceeded
        from openai import AsyncOpenAI

        recorder = _Recorder(event_sink)

        class Hooks(RunHooks):
            async def on_llm_start(self, context, agent, system_prompt, input_items):  # noqa: ANN001
                recorder.model_request()

            async def on_llm_end(self, context, agent, response):  # noqa: ANN001
                recorder.model_response(response)

            async def on_tool_start(self, context, agent, tool):  # noqa: ANN001
                recorder.tool_call(context, tool)

            async def on_tool_end(self, context, agent, tool, result):  # noqa: ANN001
                recorder.tool_result(context, tool, result)

        agent = Agent(name="wse_controller", instructions=session.connection.instructions,
                      model=self.model if self.model is not None else agent_config.model,
                      model_settings=ModelSettings(**agent_config.model_settings), tools=[],
                      mcp_servers=[session.server])
        metadata = self._metadata(session, agent_config)

        async def sdk_run() -> Any:
            # A model name gets a client of its own, closed in this loop: the SDK's default client pools
            # connections process-wide, and a pooled connection dies with an earlier trial's event loop.
            client = AsyncOpenAI() if isinstance(agent.model, str) else None
            try:
                provider = {} if client is None else {"model_provider": MultiProvider(openai_client=client)}
                run_config = RunConfig(tracing_disabled=agent_config.tracing_disabled,
                                       trace_include_sensitive_data=not agent_config.tracing_disabled
                                       and not agent_config.benchmark,
                                       workflow_name="wse_experiment", **provider)
                return await Runner.run(agent, session.connection.initial_message, max_turns=agent_config.max_turns,
                                        run_config=run_config, hooks=Hooks())
            finally:
                if client is not None:
                    try:
                        await client.close()
                    except Exception:  # noqa: BLE001 - closing must not mask the run's outcome
                        logger.exception("closing the model client failed")

        recorder.emit("run_started", details={"model": agent_config.model, "max_turns": agent_config.max_turns,
                                              "wall_clock_seconds": agent_config.wall_clock_seconds,
                                              "tracing_disabled": agent_config.tracing_disabled,
                                              "allowed_tools": list(session.connection.allowed_tools)})
        session.loop = asyncio.get_running_loop()
        session.task = asyncio.ensure_future(sdk_run())
        if session.cancel is not None:  # cancelled before the loop started
            session.task.cancel()
        final_message, conversation_ref = None, None
        try:
            done, _ = await asyncio.wait({session.task}, timeout=agent_config.wall_clock_seconds)
            if not done:
                if session.cancel is None:
                    session.cancel = ("timeout", f"the wall-clock limit of {agent_config.wall_clock_seconds:g} s "
                                                 "was reached")
                session.task.cancel()
            try:
                result = await session.task
            except asyncio.CancelledError:
                if not session.task.cancelled():
                    raise
                status, reason = session.cancel or ("cancelled", "the SDK run was cancelled")
            except MaxTurnsExceeded as exc:
                status, reason = "max_turns", f"the SDK turn limit of {agent_config.max_turns} was reached: {exc}"
            except Exception as exc:  # noqa: BLE001 - every SDK failure becomes a termination record
                status, reason = "error", describe_exception(exc)
            else:
                status, reason = "completed", "the agent stopped voluntarily"
                final_message = None if result.final_output is None else str(result.final_output)
                conversation_ref = getattr(result, "last_response_id", None)
        except asyncio.CancelledError:
            session.task.cancel()
            raise
        finally:
            session.task = None
        recorder.emit("run_finished", details={"status": status, "reason": reason})
        return AgentRunResult(status=status, reason=reason, final_message=final_message,
                              conversation_ref=conversation_ref, turns=recorder.turns,
                              tool_calls=recorder.tool_calls,
                              elapsed_seconds=round(time.monotonic() - recorder.started, 6),
                              usage=self._usage(recorder, agent_config), provider_metadata=metadata)


# ---------------------------------------------------------------------------
# SDK test double driven by a stepwise strategy
# ---------------------------------------------------------------------------


def strategy_model(strategy: Any, *, experiment_id: str) -> Any:
    """An SDK ``Model`` whose tool calls are chosen by ``strategy`` (no network; see the module docstring).

    Its first call reads the experiment (``wse_get_experiment``); every later
    call turns the latest tool output into an ``Observation`` and asks the
    strategy for the next action. A ``stop`` action becomes the final
    message. Token usage is synthetic (about four characters per token) and
    the provider is reported as ``wse-test-double``.
    """
    _require_sdk()
    from agents.items import ModelResponse
    from agents.models.interface import Model
    from agents.usage import Usage
    from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

    from wse.schemas import BudgetEntry
    from wse.strategies.base import Action, Observation

    def last_output(items: Any, call_id: str) -> Any:
        for item in reversed(items if isinstance(items, list) else []):
            data = item if isinstance(item, dict) else getattr(item, "model_dump", lambda: {})()
            if data.get("type") == "function_call_output" and data.get("call_id") == call_id:
                output = data.get("output")
                if isinstance(output, list):
                    return [part if isinstance(part, dict) else {"text": str(part)} for part in output]
                return output
        return ""

    class StrategyModel(Model):
        provider_name = TEST_DOUBLE_PROVIDER

        def __init__(self) -> None:
            self.observations: list[Observation] = []
            self.budget: dict[str, BudgetEntry] = {}
            self.objective: Any = None
            self.pending: tuple[str, Action] | None = None
            self.calls = 0
            self.requests: list[dict[str, Any]] = []

        def _observe(self, call_id: str, action: Action, output: Any) -> None:
            text, ok, _, parsed = parse_tool_output(output)
            if ok and isinstance(parsed, dict):
                observation = Observation(tool=action.tool, arguments=action.arguments, ok=True,
                                          result=parsed.get("result"))
                usage = parsed.get("usage") or {}
                for name, entry in (usage.get("cumulative") or {}).items():
                    self.budget[name] = BudgetEntry(**entry)
                if self.objective is None and action.tool == "wse_get_experiment":
                    task = (parsed.get("result") or {}).get("task") or {}
                    self.objective = task.get("objective")
            else:
                error = parsed if isinstance(parsed, dict) else {"code": "TOOL_ERROR", "message": text}
                observation = Observation(tool=action.tool, arguments=action.arguments, ok=False, error=error)
            self.observations.append(observation)

        async def get_response(self, system_instructions, input, model_settings, tools, output_schema,  # noqa: A002
                               handoffs, tracing, *, previous_response_id=None, conversation_id=None, prompt=None):
            self.calls += 1
            self.requests.append({"system_instructions": system_instructions,
                                  "tools": [{"name": t.name, "description": getattr(t, "description", None),
                                             "parameters": getattr(t, "params_json_schema", None)} for t in tools]})
            if self.pending is None:
                action = Action.call("wse_get_experiment", experiment_id=experiment_id)
            else:
                call_id, previous = self.pending
                self._observe(call_id, previous, last_output(input, call_id))
                from wse.schemas import ObjectiveSpec

                objective = ObjectiveSpec(**self.objective) if isinstance(self.objective, dict) else None
                action = strategy.next_action(objective, list(self.observations), dict(self.budget))
            prompt_chars = len(system_instructions or "") + len(json.dumps(input, default=str))
            if action.kind == "stop":
                self.pending = None
                text = f"Stopping: {action.reason}"
                output = [ResponseOutputMessage(id=f"msg_{self.calls}", type="message", role="assistant",
                                                status="completed",
                                                content=[ResponseOutputText(type="output_text", text=text,
                                                                            annotations=[])])]
                produced = len(text)
            else:
                call_id = f"call_{self.calls}"
                arguments = json.dumps(action.arguments, sort_keys=True)
                self.pending = (call_id, action)
                output = [ResponseFunctionToolCall(id=f"fc_{self.calls}", call_id=call_id, name=action.tool,
                                                   arguments=arguments, type="function_call", status="completed")]
                produced = len(arguments)
            input_tokens, output_tokens = prompt_chars // 4 + 1, produced // 4 + 1
            return ModelResponse(output=output, usage=Usage(requests=1, input_tokens=input_tokens,
                                                            output_tokens=output_tokens,
                                                            total_tokens=input_tokens + output_tokens),
                                 response_id=None)

        async def stream_response(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise NotImplementedError("the strategy test double does not stream")
            yield  # pragma: no cover

    return StrategyModel()
