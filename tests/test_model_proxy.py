"""The per-run model proxy (``wse.model_proxy``) against a fake upstream on 127.0.0.1.

The fake upstream is the standard library's ``http.server``, an HTTP
implementation independent of the proxy's own (uvicorn toward the client,
httpx2 toward the upstream), and it records every request it receives.
"""

from __future__ import annotations

import http.client
import json
import select
import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx2
import pytest

from wse.model_proxy import PROXY_TOKEN_ENV, ModelProxy, _Redactor, codex_provider_config
from wse.schemas import ErrorCode, WSEError

KEY = "sk-wse-test-key-0123456789abcdefghijklmnop"
LOG_FIELDS = {"time", "method", "path", "status", "latency_s", "duration_s", "request_id", "served_model", "error"}
CREATED = b'event: response.created\ndata: {"type": "response.created"}\n\n'
COMPLETED = b'event: response.completed\ndata: {"type": "response.completed"}\n\n'


class FakeUpstream:
    """An OpenAI-like API under ``/openai/v1`` that records each request.

    ``/echo`` answers with what it received; ``/status/<code>`` answers with
    that error status; ``/stream`` sends one Server-Sent Event, then holds its
    last one until ``release`` is set (``sent_last`` is set just before it goes
    out, ``peer_closed`` if the proxy closes the connection while it waits);
    ``/leak`` echoes the authorization it received, in a header and in a body
    that splits the key across two chunks; ``/slow`` answers after a second.
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.release = threading.Event()
        self.sent_last = threading.Event()
        self.peer_closed = threading.Event()
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                upstream.handle(self)

            do_POST = do_DELETE = do_GET

            def log_message(self, *args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/openai/v1"

    def handle(self, handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length") or 0)
        body = handler.rfile.read(length) if length else b""
        path, _, query = handler.path.partition("?")
        self.requests.append({"method": handler.command, "path": path, "query": query,
                              "headers": list(handler.headers.items()), "body": body})
        route = path.removeprefix("/openai/v1")
        if route == "/echo":
            _send_json(handler, 200, {"method": handler.command, "path": path, "query": query,
                                      "body": json.loads(body) if body else None},
                       {"x-request-id": "req-echo", "x-ms-served-model": "fake-model-2026-11-01"})
        elif route.startswith("/status/"):
            status = int(route.removeprefix("/status/"))
            _send_json(handler, status, {"error": {"message": f"fake upstream error {status}", "code": status}},
                       {"x-request-id": f"req-{status}"})
        elif route == "/stream":
            self._stream(handler)
        elif route == "/slow":
            time.sleep(1.0)
            _send_json(handler, 200, {"late": True})
        elif route == "/leak":
            authorization = handler.headers.get("Authorization", "")
            key = authorization.removeprefix("Bearer ")
            _start_chunked(handler, "text/plain", {"x-echo-key": authorization})
            _send_chunk(handler, f"before {key[:6]}".encode())
            time.sleep(0.05)
            _send_chunk(handler, f"{key[6:]} after".encode())
            _send_chunk(handler, b"")
        else:
            _send_json(handler, 404, {"error": {"message": "no such route"}})

    def _stream(self, handler: BaseHTTPRequestHandler) -> None:
        _start_chunked(handler, "text/event-stream", {"x-request-id": "req-stream"})
        _send_chunk(handler, CREATED)
        deadline = time.monotonic() + 10
        while not self.release.wait(0.02):
            if _peer_closed(handler.connection):
                self.peer_closed.set()
                handler.close_connection = True
                return
            if time.monotonic() > deadline:
                break
        self.sent_last.set()
        try:
            _send_chunk(handler, COMPLETED)
            _send_chunk(handler, b"")
        except OSError:
            handler.close_connection = True


def _send_json(handler: BaseHTTPRequestHandler, status: int, payload: object,
               headers: dict[str, str] | None = None) -> None:
    data = json.dumps(payload).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    for name, value in (headers or {}).items():
        handler.send_header(name, value)
    handler.end_headers()
    handler.wfile.write(data)


def _start_chunked(handler: BaseHTTPRequestHandler, content_type: str, headers: dict[str, str]) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Transfer-Encoding", "chunked")
    for name, value in headers.items():
        handler.send_header(name, value)
    handler.end_headers()


def _send_chunk(handler: BaseHTTPRequestHandler, data: bytes) -> None:
    handler.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
    handler.wfile.flush()


def _peer_closed(connection: socket.socket) -> bool:
    readable, _, _ = select.select([connection], [], [], 0)
    if not readable:
        return False
    try:
        return connection.recv(1, socket.MSG_PEEK) == b""
    except OSError:
        return True


@pytest.fixture
def upstream() -> Iterator[FakeUpstream]:
    fake = FakeUpstream()
    fake.thread.start()
    yield fake
    fake.release.set()
    fake.server.shutdown()
    fake.server.server_close()


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "proxy" / "requests.jsonl"


@pytest.fixture
def proxy(upstream: FakeUpstream, log_path: Path) -> Iterator[ModelProxy]:
    with ModelProxy(upstream.base_url, KEY, log_path=log_path) as running:
        yield running


def _client() -> httpx2.Client:
    return httpx2.Client(trust_env=False, timeout=10)


def _auth(proxy: ModelProxy) -> dict[str, str]:
    return {"Authorization": f"Bearer {proxy.token}"}


def _log_entries(log_path: Path) -> list[dict[str, Any]]:
    return sorted((json.loads(line) for line in log_path.read_text().splitlines()), key=lambda entry: entry["time"])


def _header_lists(headers: list[tuple[str, str]]) -> dict[str, list[str]]:
    collected: dict[str, list[str]] = {}
    for name, value in headers:
        collected.setdefault(name.lower(), []).append(value)
    return collected


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_requests_without_the_token_get_401_and_never_reach_the_upstream(proxy, upstream, log_path):
    attempts = [
        {},
        {"Authorization": "Bearer not-the-token"},
        {"Authorization": f"Basic {proxy.token}"},
        {"Authorization": f"Bearer {KEY}"},
        {"api-key": proxy.token},
        [("Authorization", f"Bearer {proxy.token}"), ("Authorization", "Bearer other")],
    ]
    with _client() as client:
        for headers in attempts:
            response = client.post(f"{proxy.url}/responses", json={"model": "m"}, headers=headers)
            assert response.status_code == 401
            assert response.json()["error"]["code"] == "invalid_proxy_token"
            assert response.headers["www-authenticate"] == "Bearer"
    proxy.stop()
    assert upstream.requests == []
    assert [entry["status"] for entry in _log_entries(log_path)] == [401] * len(attempts)


def test_the_upstream_gets_the_key_and_the_client_never_sees_it(proxy, upstream):
    with _client() as client:
        response = client.post(f"{proxy.url}/echo?api-version=preview", json={"input": "hello"},
                               headers={**_auth(proxy), "api-key": "client-supplied", "x-custom": "kept",
                                        "Connection": "keep-alive, x-hop", "x-hop": "dropped"})
    assert response.status_code == 200
    assert response.json() == {"method": "POST", "path": "/openai/v1/echo", "query": "api-version=preview",
                               "body": {"input": "hello"}}
    (received,) = upstream.requests
    headers = _header_lists(received["headers"])
    assert headers["authorization"] == [f"Bearer {KEY}"]
    assert headers["x-custom"] == ["kept"]
    assert headers["accept-encoding"] == ["identity"]
    assert "api-key" not in headers and "x-hop" not in headers
    assert proxy.token not in repr(received)
    assert KEY not in response.text and KEY not in repr(response.headers.raw)
    assert response.headers["x-request-id"] == "req-echo"


def test_server_sent_events_reach_the_client_as_the_upstream_sends_them(proxy, upstream):
    with _client() as client, client.stream("POST", f"{proxy.url}/stream", json={"stream": True},
                                            headers=_auth(proxy)) as response:
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream"
        lines = response.iter_lines()
        first = next(line for line in lines if line.startswith("data:"))
        # The upstream holds its last event until released, so the first one came through unbuffered.
        assert not upstream.sent_last.is_set()
        upstream.release.set()
        rest = [line for line in lines if line.startswith("data:")]
    assert first == 'data: {"type": "response.created"}'
    assert rest == ['data: {"type": "response.completed"}']


@pytest.mark.parametrize("status", [400, 401, 404, 429, 500, 503])
def test_upstream_errors_pass_through(proxy, upstream, status):
    with _client() as client:
        response = client.post(f"{proxy.url}/status/{status}", json={}, headers=_auth(proxy))
    assert response.status_code == status
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"error": {"message": f"fake upstream error {status}", "code": status}}
    assert response.headers["x-request-id"] == f"req-{status}"


def test_an_unreachable_upstream_gives_502(tmp_path):
    log_path = tmp_path / "requests.jsonl"
    with ModelProxy(f"http://127.0.0.1:{_closed_port()}/openai/v1", KEY, log_path=log_path) as proxy, \
            _client() as client:
        response = client.post(f"{proxy.url}/responses", json={}, headers=_auth(proxy))
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_unreachable"
    assert KEY not in response.text
    (entry,) = _log_entries(log_path)
    assert (entry["status"], entry["error"]) == (502, "upstream_unreachable")


def test_an_upstream_that_goes_silent_gives_504_or_a_cut_off_stream(upstream, tmp_path, monkeypatch):
    monkeypatch.setattr("wse.model_proxy.UPSTREAM_TIMEOUT", httpx2.Timeout(5.0, read=0.3))
    log_path = tmp_path / "requests.jsonl"
    with ModelProxy(upstream.base_url, KEY, log_path=log_path) as proxy, _client() as client:
        late = client.post(f"{proxy.url}/slow", json={}, headers=_auth(proxy))
        assert late.status_code == 504
        assert late.json()["error"]["code"] == "upstream_timeout"
        with client.stream("POST", f"{proxy.url}/stream", json={"stream": True},
                           headers=_auth(proxy)) as response:
            lines = response.iter_lines()
            assert next(lines) == "event: response.created"
            with pytest.raises(httpx2.RemoteProtocolError):
                list(lines)
    assert not upstream.sent_last.is_set()
    assert [(entry["status"], entry["error"]) for entry in _log_entries(log_path)] == [
        (504, "upstream_timeout"), (200, "upstream_stream_error")]


def test_the_log_has_one_line_per_request_and_no_secrets(proxy, upstream, log_path):
    upstream.release.set()
    with _client() as client:
        assert client.post(f"{proxy.url}/echo", json={"input": "private prompt"}).status_code == 401
        assert client.post(f"{proxy.url}/echo?api-version=preview", json={"input": "private prompt"},
                           headers=_auth(proxy)).status_code == 200
        assert client.post(f"{proxy.url}/status/429", json={}, headers=_auth(proxy)).status_code == 429
        assert client.post(f"{proxy.url}/stream", json={"stream": True}, headers=_auth(proxy)).status_code == 200
    proxy.stop()
    entries = _log_entries(log_path)
    assert [(entry["method"], entry["path"], entry["status"], entry["error"]) for entry in entries] == [
        ("POST", "/v1/echo", 401, "invalid_proxy_token"),
        ("POST", "/v1/echo", 200, None),
        ("POST", "/v1/status/429", 429, None),
        ("POST", "/v1/stream", 200, None),
    ]
    assert [entry["request_id"] for entry in entries] == [None, "req-echo", "req-429", "req-stream"]
    assert [entry["served_model"] for entry in entries] == [None, "fake-model-2026-11-01", None, None]
    for entry in entries:
        assert set(entry) == LOG_FIELDS
        assert entry["time"].endswith("Z")
        assert 0 <= entry["latency_s"] <= entry["duration_s"]
    text = log_path.read_text()
    for secret in (KEY, proxy.token, "private prompt", "api-version", "openai/v1"):
        assert secret not in text


def test_a_key_the_upstream_echoes_reaches_the_client_redacted(proxy, upstream):
    with _client() as client:
        response = client.get(f"{proxy.url}/leak", headers=_auth(proxy))
    assert response.status_code == 200
    assert response.headers["x-echo-key"] == "Bearer [REDACTED]"
    assert response.text == "before [REDACTED] after"


def test_redaction_holds_across_any_chunk_boundary():
    secret = KEY.encode()
    payload = b"a" + secret + b"b" + secret[:5] + b"c" + secret + secret
    expected = payload.replace(secret, b"[REDACTED]")
    for cut in range(len(payload) + 1):
        redactor = _Redactor(secret)
        assert redactor.feed(payload[:cut]) + redactor.feed(payload[cut:]) + redactor.flush() == expected
    redactor = _Redactor(secret)
    assert b"".join(redactor.feed(payload[i:i + 1]) for i in range(len(payload))) + redactor.flush() == expected


def test_only_paths_below_v1_reach_the_upstream(proxy, upstream):
    root = proxy.url.removesuffix("/v1")
    with _client() as client:
        for url in (f"{root}/v2/echo", f"{root}/v1x/echo", f"{root}/", f"{proxy.url}/%2e%2e/admin"):
            assert client.post(url, json={}, headers=_auth(proxy)).status_code == 404, url
    host, port = httpx2.URL(proxy.url).host, httpx2.URL(proxy.url).port
    connection = http.client.HTTPConnection(host, port, timeout=5)
    try:  # http.client sends the path as written, dot segments included
        connection.request("POST", "/v1/../admin", body=b"{}", headers=_auth(proxy))
        assert connection.getresponse().status == 404
    finally:
        connection.close()
    assert upstream.requests == []


def test_a_client_that_leaves_closes_the_upstream_request(proxy, upstream, log_path):
    with _client() as client:
        with client.stream("POST", f"{proxy.url}/stream", json={"stream": True},
                           headers=_auth(proxy)) as response:
            assert next(response.iter_lines()) == "event: response.created"
        # Closing the stream unread closes the client's connection.
        assert upstream.peer_closed.wait(5)
    proxy.stop()
    (entry,) = _log_entries(log_path)
    assert (entry["status"], entry["error"]) == (200, "client_disconnected")


def test_stop_closes_the_port_promptly(upstream):
    proxy = ModelProxy(upstream.base_url, KEY).start()
    port = httpx2.URL(proxy.url).port
    with _client() as client:  # keeps an idle connection to the proxy open while it stops
        assert client.post(f"{proxy.url}/echo", json={}, headers=_auth(proxy)).status_code == 200
        started = time.monotonic()
        proxy.stop()
        assert time.monotonic() - started < 2.0
    assert not proxy.running
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("127.0.0.1", port), timeout=2).close()
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
    proxy.stop()  # stopping again does nothing


def test_stop_cuts_off_an_open_stream(proxy, upstream, log_path):
    with _client() as client, client.stream("POST", f"{proxy.url}/stream", json={"stream": True},
                                            headers=_auth(proxy)) as response:
        lines = response.iter_lines()
        assert next(lines) == "event: response.created"
        started = time.monotonic()
        proxy.stop()
        assert time.monotonic() - started < 2.0
        # Cut off, not ended: the client cannot take the partial stream for a complete one.
        with pytest.raises(httpx2.RemoteProtocolError):
            list(lines)
    assert upstream.peer_closed.wait(5)
    assert not upstream.sent_last.is_set()
    (entry,) = _log_entries(log_path)
    assert (entry["status"], entry["error"]) == (200, "proxy_stopped")


@pytest.mark.parametrize("base_url, api_key", [
    ("", KEY),
    ("   ", KEY),
    (None, KEY),
    ("https://example.invalid/openai/v1", ""),
    ("https://example.invalid/openai/v1", "  \n"),
    ("https://example.invalid/openai/v1", None),
])
def test_refuses_to_start_without_an_upstream_url_or_a_key(base_url, api_key):
    with pytest.raises(WSEError) as caught:
        ModelProxy(base_url, api_key)
    assert caught.value.code == ErrorCode.INVALID_ARGUMENT


@pytest.mark.parametrize("base_url", [
    "example.invalid/openai/v1",
    "ftp://example.invalid/openai/v1",
    "http://example.invalid/openai/v1",
    "https://user:secret@example.invalid/openai/v1",
    "https://example.invalid/openai/v1?api-version=preview",
    "https://example.invalid/openai/v1#fragment",
])
def test_refuses_upstream_urls_that_could_expose_the_key(base_url):
    with pytest.raises(WSEError) as caught:
        ModelProxy(base_url, KEY)
    assert caught.value.code == ErrorCode.INVALID_ARGUMENT


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "localhost", ""])
def test_binds_only_to_a_loopback_address(host):
    with pytest.raises(WSEError) as caught:
        ModelProxy("https://example.invalid/openai/v1", KEY, host=host)
    assert caught.value.code == ErrorCode.INVALID_ARGUMENT


def test_a_malformed_key_is_refused_without_repeating_it():
    with pytest.raises(WSEError) as caught:
        ModelProxy("https://example.invalid/openai/v1", "sk-wse-bad\nkey")
    assert "sk-wse-bad" not in f"{caught.value} {caught.value.details}"


def test_each_proxy_has_its_own_token_and_starts_once(upstream):
    first, second = ModelProxy(upstream.base_url, KEY), ModelProxy(upstream.base_url, KEY)
    assert first.token != second.token and len(first.token) >= 40
    with pytest.raises(RuntimeError):
        first.url  # noqa: B018 - no address before start()
    with first:
        assert first.running
        assert first.url.startswith("http://127.0.0.1:") and first.url.endswith("/v1")
        with pytest.raises(RuntimeError):
            first.start()
    assert not first.running
    with pytest.raises(RuntimeError):
        first.start()
    assert KEY not in repr(first) and KEY not in repr(vars(first))


def test_codex_provider_config_points_codex_at_the_proxy(proxy):
    assert PROXY_TOKEN_ENV == "WSE_MODEL_PROXY_TOKEN"
    assert codex_provider_config(proxy) == {
        "model_providers": {"wse_proxy": {"name": "WSE model proxy", "base_url": proxy.url,
                                          "env_key": "WSE_MODEL_PROXY_TOKEN", "wire_api": "responses"}},
        "model_provider": "wse_proxy",
    }
    assert codex_provider_config(proxy, provider_id="wse-run-7")["model_provider"] == "wse-run-7"
    with pytest.raises(WSEError):
        codex_provider_config(proxy, provider_id="wse.proxy")
