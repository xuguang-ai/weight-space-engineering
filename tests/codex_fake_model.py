"""A local Responses API endpoint that plays a script, for running the real Codex binary offline in tests.

Codex sends ``POST /v1/responses`` with ``stream: true``; each request gets
the next entry of the script, streamed as server-sent events. An entry is a
list of Responses output items (``function_call`` with a ``namespace``,
``custom_tool_call``, ``message``) or a callable that receives the request
body and returns that list, so a step can use an ID from an earlier tool
result. After the script, every request gets a final message. Each request
body is kept (``requests``) together with whether it carried the expected
bearer token; usage is synthetic. ``GET`` requests (model discovery) get an
empty model list. Nothing leaves the machine: the server listens on
127.0.0.1 only.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

Step = list[dict[str, Any]] | Callable[[dict[str, Any]], list[dict[str, Any]]]


def message(text: str, index: int = 0) -> dict[str, Any]:
    return {"type": "message", "role": "assistant", "id": f"msg_{index}", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def wse_call(tool: str, arguments: dict[str, Any], index: int) -> dict[str, Any]:
    """A call of a WSE tool, which Codex exposes as a direct function in the ``mcp__wse`` namespace."""
    return {"type": "function_call", "id": f"fc_{index}", "call_id": f"call_{index}", "name": tool,
            "namespace": "mcp__wse", "arguments": json.dumps(arguments), "status": "completed"}


def shell(command: str, index: int) -> dict[str, Any]:
    """A shell command, run through Codex's code-mode ``exec`` tool (the tool surface of gpt-5.6-luna)."""
    script = (f"const r = await tools.exec_command({{cmd: {json.dumps(command)}, login: false}});\n"
              "text(r.output);\n")
    return {"type": "custom_tool_call", "id": f"ctc_{index}", "call_id": f"call_{index}", "name": "exec",
            "input": script, "status": "completed"}


def tool_outputs(body: dict[str, Any]) -> list[str]:
    """The texts of the tool outputs in a request's input, oldest first."""
    found = []
    for item in body.get("input") or []:
        if item.get("type") in ("function_call_output", "custom_tool_call_output"):
            output = item.get("output")
            if isinstance(output, list):
                output = "".join(part.get("text", "") for part in output if isinstance(part, dict))
            found.append(str(output))
    return found


class FakeResponses:
    """A scripted local Responses endpoint (see the module docstring)."""

    def __init__(self, script: list[Step], token: str) -> None:
        self.script = list(script)
        self.token = token
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.server: ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        assert self.server is not None
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def provider_config(self, env_key: str) -> dict[str, Any]:
        """The Codex provider fragment for this endpoint (the shape of the model proxy's)."""
        return {"model_provider": "wse_test_endpoint",
                "model_providers": {"wse_test_endpoint": {"name": "WSE test endpoint", "base_url": self.url,
                                                          "env_key": env_key, "wire_api": "responses"}}}

    def start(self) -> FakeResponses:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def _send(self, status: int, content_type: str, payload: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:  # noqa: N802
                self._send(200, "application/json", json.dumps({"object": "list", "data": []}).encode())

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                authorized = self.headers.get("Authorization") == f"Bearer {fake.token}"
                with fake.lock:
                    index = len(fake.requests)
                    fake.requests.append({"path": self.path, "authorized": authorized, "body": body})
                    step = fake.script[index] if index < len(fake.script) else [message("Done.", index)]
                if not authorized:
                    self._send(401, "application/json", b'{"error": {"message": "unauthorized"}}')
                    return
                items = step(body) if callable(step) else step
                rid = f"resp_{index + 1}"
                events = [{"type": "response.created", "response": {"id": rid, "status": "in_progress",
                                                                    "output": []}}]
                for position, item in enumerate(items):
                    events.append({"type": "response.output_item.added", "output_index": position, "item": item})
                    events.append({"type": "response.output_item.done", "output_index": position, "item": item})
                events.append({"type": "response.completed", "response": {
                    "id": rid, "status": "completed", "output": items,
                    "usage": {"input_tokens": 1000 + index, "input_tokens_details": {"cached_tokens": 100},
                              "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 5},
                              "total_tokens": 1020 + index}}})
                payload = b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)
                self._send(200, "text/event-stream", payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
