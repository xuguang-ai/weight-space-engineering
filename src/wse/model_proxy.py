"""A per-run model proxy on 127.0.0.1 that adds the operator's API key to OpenAI-compatible calls.

An agent runner that starts its model client in another process, such as the
Codex CLI (``codex exec``), would otherwise have to give that process the
operator's API key, and every command the agent runs would inherit it.
``ModelProxy`` keeps the key in the WSE runner process. For one run it serves
an OpenAI-compatible API at ``http://127.0.0.1:<port>/v1`` (``url``) and
forwards each request to the upstream base URL (``OPENAI_BASE_URL``, for
example an Azure OpenAI ``.../openai/v1`` endpoint) with
``Authorization: Bearer <key>``. The agent's process receives only ``token``,
a random per-run token (``secrets.token_urlsafe(32)``) that it sends as its
own bearer token. The token works only against this proxy and only while the
run lasts: the proxy listens on loopback alone and closes its port when the
run ends. ``codex_provider_config`` returns the Codex ``config.toml`` entries
for a model provider that points at the proxy and reads the token from
``WSE_MODEL_PROXY_TOKEN`` (``PROXY_TOKEN_ENV``).

* **Access.** A request needs exactly one ``Authorization: Bearer <token>``
  header, compared in constant time; without it the answer is 401 and the
  upstream is never contacted. A path outside ``/v1``, or one with dot
  segments, gets 404, also without reaching the upstream.
* **Forwarding.** Method, the path below ``/v1`` (joined onto the upstream
  base URL), query, body, and end-to-end headers go upstream. Hop-by-hop
  headers and every client credential (``Authorization``, ``api-key``,
  ``x-api-key``, ``Ocp-Apim-Subscription-Key``) are dropped; the proxy sets
  its own authorization and asks for an uncompressed response. Redirects
  are returned to the client, not followed. The upstream connection honors
  the environment's proxy settings (``HTTPS_PROXY``, ``NO_PROXY``), as the
  OpenAI SDK does, except toward a loopback upstream.
* **Responses.** The upstream's status, headers, and body come back as they
  arrive, chunk by chunk, so the Server-Sent Events of a ``stream: true``
  Responses API call reach the client as they are produced. If the upstream
  echoes the key, in a header or the body, the client gets ``[REDACTED]``
  instead; a compressed body the proxy cannot decode (and so cannot check)
  gets 502. An unreachable upstream gives 502, a timeout before the
  response 504. A stream that breaks midway is cut off rather than ended
  normally, so the client sees that it is incomplete, and when the client
  goes away the upstream request is closed.
* **Log.** With ``log_path``, each request appends one JSON line: ``time``,
  ``method``, ``path``, ``status``, ``latency_s`` (until the response
  headers), ``duration_s`` (until the response ended), ``request_id`` (the
  upstream's ``x-request-id``, else ``apim-request-id``), ``served_model``
  (``x-ms-served-model`` or ``openai-model``, when the upstream names it),
  and ``error``, the proxy's code when it ended the request itself. The key,
  the token, headers, query strings, bodies, and the upstream URL (which can
  name a private resource) are never written.
* **Lifecycle.** ``start`` binds the port and serves from a background
  thread with its own event loop (uvicorn over h11 toward the client, an
  httpx2 client toward the upstream), so synchronous code such as the
  runner can use it; it is also a context manager. ``stop`` cuts off open
  streams, closes the port, and returns in about 0.2 s (uvicorn checks for
  shutdown every 0.1 s). A proxy starts once, so each run has its own token.
* **Fails closed.** The constructor raises without an upstream URL or a
  non-empty key, for a bind address that is not a loopback IP address, and
  for an upstream URL that carries credentials, a query, or a fragment, or
  that uses plain ``http`` to a host other than loopback (the key would
  cross the network unencrypted).

**Limits.** The token is a bearer credential for the model. Any process
that can read it while the run lasts, such as a command the agent runs with
the token in its environment, can call the upstream model through the
proxy, although it never learns the key. The proxy does not restrict
models, endpoints below ``/v1``, or spending, and it reads no request or
response content except to redact the key; budgets stay with the runner and
the provider. Redaction covers the key as sent, not a transformed copy such
as a masked or re-encoded one. The proxy speaks HTTP/1.1 without WebSocket
upgrades.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import ipaddress
import json
import logging
import re
import secrets
import socket
import threading
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import unquote, urlsplit

import httpx2
import uvicorn

from wse.database import utc_now
from wse.schemas import ErrorCode, WSEError

__all__ = ["PROXY_TOKEN_ENV", "ModelProxy", "codex_provider_config"]

logger = logging.getLogger("wse.model_proxy")

PROXY_TOKEN_ENV = "WSE_MODEL_PROXY_TOKEN"
PROXY_PATH = "/v1"
REDACTED = b"[REDACTED]"
HOP_BY_HOP_HEADERS = frozenset({b"connection", b"keep-alive", b"proxy-authenticate", b"proxy-authorization",
                                b"proxy-connection", b"te", b"trailer", b"transfer-encoding", b"upgrade"})
CLIENT_CREDENTIAL_HEADERS = frozenset({b"authorization", b"api-key", b"x-api-key", b"ocp-apim-subscription-key"})
# Set again for the upstream request: host and length by the HTTP client, the encoding by the proxy. The proxy
# reads the whole body before forwarding it, so a client's 100-continue expectation is already met.
RESET_REQUEST_HEADERS = frozenset({b"host", b"content-length", b"accept-encoding", b"expect"})
# The body goes out decoded, possibly redacted, in chunks of the proxy's own; Alt-Svc describes the upstream's
# transports, not the proxy's.
DROPPED_RESPONSE_HEADERS = frozenset({b"content-length", b"content-encoding", b"alt-svc"})
DECODED_ENCODINGS = frozenset({"identity", "gzip", "deflate"})
REQUEST_ID_HEADERS = ("x-request-id", "apim-request-id")
SERVED_MODEL_HEADERS = ("x-ms-served-model", "openai-model")
# Read covers the longest silence in a streamed response, such as a reasoning model thinking. Clients give up
# sooner (Codex has its own stream idle timeout), and a client that leaves closes the upstream request.
UPSTREAM_TIMEOUT = httpx2.Timeout(connect=10.0, read=600.0, write=120.0, pool=30.0)
LISTEN_BACKLOG = 128
START_TIMEOUT_SECONDS = 10.0
STOP_TIMEOUT_SECONDS = 5.0
GRACEFUL_SHUTDOWN_SECONDS = 2
# Longer than clients commonly keep idle connections (reqwest, Codex's HTTP library, keeps them 90 s by default),
# so the client closes an idle connection first and never sends a request on one the proxy is closing.
KEEP_ALIVE_SECONDS = 120
LOGGED_PATH_CHARS = 300
PROVIDER_ID = re.compile(r"[A-Za-z0-9_-]+")

ASGIMessage = dict[str, Any]
Receive = Callable[[], Awaitable[ASGIMessage]]
Send = Callable[[ASGIMessage], Awaitable[None]]
RawHeaders = Sequence[tuple[bytes, bytes]]


class _Credential:
    """The upstream key, kept out of reprs (and so out of tracebacks that print locals)."""

    __slots__ = ("authorization", "secret")

    def __init__(self, key: str) -> None:
        self.secret = key.encode("ascii")
        self.authorization = b"Bearer " + self.secret

    def __repr__(self) -> str:
        return "<redacted credential>"


@dataclass
class _Exchange:
    """What the log records about one request, and how far its response got."""

    time: str
    method: str
    path: str
    started: float
    status: int | None = None
    latency_s: float | None = None
    request_id: str | None = None
    served_model: str | None = None
    error: str | None = None
    response_started: bool = False
    response_complete: bool = False

    def elapsed(self) -> float:
        return round(time.monotonic() - self.started, 4)

    def log_entry(self) -> dict[str, Any]:
        return {"time": self.time, "method": self.method, "path": self.path, "status": self.status,
                "latency_s": self.latency_s, "duration_s": self.elapsed(), "request_id": self.request_id,
                "served_model": self.served_model, "error": self.error}


class _Redactor:
    """Replaces a secret in a byte stream, including an occurrence split across chunks.

    Each call returns everything but the longest end of the data seen so far
    that begins the secret; that end is held back until the next chunk shows
    whether the secret continues there.
    """

    def __init__(self, secret: bytes) -> None:
        self._secret = secret
        self._held = b""

    def feed(self, chunk: bytes) -> bytes:
        data = (self._held + chunk).replace(self._secret, REDACTED)
        cut = len(data) - _secret_prefix_at_end(data, self._secret)
        self._held = data[cut:]
        return data[:cut]

    def flush(self) -> bytes:
        held, self._held = self._held, b""
        return held


def _secret_prefix_at_end(data: bytes, secret: bytes) -> int:
    """The length of the longest end of ``data`` that is a proper prefix of ``secret``."""
    for size in range(min(len(secret) - 1, len(data)), 0, -1):
        if data.endswith(secret[:size]):
            return size
    return 0


class _Server(uvicorn.Server):
    """A uvicorn server that reports when it is listening, so ``start`` can wait for it."""

    def __init__(self, config: uvicorn.Config, on_started: Callable[[], None]) -> None:
        super().__init__(config)
        self._on_started = on_started

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        self._on_started()


class ModelProxy:
    """A loopback reverse proxy for one agent run that adds the upstream API key to every request.

    ``upstream_base_url`` is the API base the key belongs to (``OPENAI_BASE_URL``),
    and requests to ``url`` + ``/<path>`` go to ``upstream_base_url`` + ``/<path>``.
    Clients authenticate with ``token``. ``log_path`` receives one JSON line per
    request. ``host`` must be a loopback IP address; ``port`` 0 picks a free port.
    """

    def __init__(self, upstream_base_url: str, api_key: str, *, log_path: Path | None = None,
                 host: str = "127.0.0.1", port: int = 0) -> None:
        self._upstream_base, self._upstream_origin, self._upstream_loopback = _checked_upstream(upstream_base_url)
        self._credential = _Credential(_checked_key(api_key))
        self._host, self._family = _checked_host(host)
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise WSEError(ErrorCode.INVALID_ARGUMENT, "the model proxy's port must be an integer from 0 to 65535")
        self._port = port
        self._log_path = Path(log_path) if log_path is not None else None
        self._token = secrets.token_urlsafe(32)
        self._token_bytes = self._token.encode("ascii")
        self._lock = threading.Lock()
        self._state = "new"
        self._stop_requested = False
        self._ready = threading.Event()
        self._failure: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._log_file: TextIO | None = None
        # Owned by the proxy's event loop once it runs.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopping: asyncio.Event | None = None
        self._server: _Server | None = None
        self._client: httpx2.AsyncClient | None = None

    @property
    def token(self) -> str:
        """The per-run bearer token that clients of this proxy send (``Authorization: Bearer <token>``)."""
        return self._token

    @property
    def url(self) -> str:
        """The proxy's API base URL, such as ``http://127.0.0.1:54321/v1``; known once ``start`` has bound the port."""
        if self._state == "new":
            raise RuntimeError("the model proxy has no address before start()")
        host = f"[{self._host}]" if self._family == socket.AF_INET6 else self._host
        return f"http://{host}:{self._port}{PROXY_PATH}"

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive() and not self._stop_requested

    def __repr__(self) -> str:
        address = "unbound" if self._state == "new" else self.url
        return f"ModelProxy({address}, running={self.running})"

    def __enter__(self) -> ModelProxy:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def start(self) -> ModelProxy:
        """Bind the port, open the log, and serve from a background thread; returns once the proxy listens."""
        with self._lock:
            if self._state != "new":
                raise RuntimeError("a ModelProxy starts only once; create a new one, with a new token, for each run")
            self._state = "running"
        sock = None
        try:
            if self._log_path is not None:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                self._log_file = self._log_path.open("a", encoding="utf-8")
            sock = socket.socket(self._family, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self._host, self._port))
            sock.listen(LISTEN_BACKLOG)
            self._port = sock.getsockname()[1]
        except BaseException:
            if sock is not None:
                sock.close()
            if self._log_file is not None:
                self._log_file.close()
            self._state = "stopped"
            raise
        self._thread = threading.Thread(target=self._run, args=(sock,), name=f"wse-model-proxy-{self._port}",
                                        daemon=True)
        self._thread.start()
        if not self._ready.wait(START_TIMEOUT_SECONDS) or self._failure is not None:
            failure = self._failure
            self.stop()
            raise RuntimeError("the model proxy did not start") from failure
        return self

    def stop(self) -> None:
        """Cut off open responses, close the port, and wait for the proxy's thread (idempotent)."""
        with self._lock:
            state, self._state = self._state, "stopped"
            if state == "new":
                return
            self._stop_requested = True
        loop = self._loop
        if loop is not None:
            with contextlib.suppress(RuntimeError):  # the loop has already closed
                loop.call_soon_threadsafe(self._begin_shutdown)
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(STOP_TIMEOUT_SECONDS)
            if thread.is_alive():
                logger.warning("the model proxy did not stop within %.0f s", STOP_TIMEOUT_SECONDS)

    # The proxy's thread and event loop.

    def _run(self, sock: socket.socket) -> None:
        try:
            asyncio.run(self._serve(sock))
        except BaseException as exc:  # noqa: BLE001 - reported to start() and logged; the thread ends here
            self._failure = exc
            logger.error("the model proxy stopped with an error: %s", type(exc).__name__)
        finally:
            sock.close()
            self._ready.set()
            if self._log_file is not None:
                self._log_file.close()

    async def _serve(self, sock: socket.socket) -> None:
        self._stopping = asyncio.Event()
        self._loop = asyncio.get_running_loop()
        async with httpx2.AsyncClient(timeout=UPSTREAM_TIMEOUT, follow_redirects=False,
                                      trust_env=not self._upstream_loopback) as client:
            self._client = client
            # No logging configuration of uvicorn's own: its startup and access lines are below warning, and
            # the proxy writes its own request log.
            config = uvicorn.Config(self._app, http="h11", ws="none", lifespan="off", interface="asgi3",
                                    log_config=None, log_level="warning", access_log=False, proxy_headers=False,
                                    server_header=False, date_header=False, timeout_keep_alive=KEEP_ALIVE_SECONDS,
                                    timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS)
            self._server = _Server(config, self._ready.set)
            if self._stop_requested or self._stopping.is_set():
                self._server.should_exit = True
            await self._server.serve(sockets=[sock])

    def _begin_shutdown(self) -> None:
        if self._stopping is not None:
            self._stopping.set()
        if self._server is not None:
            self._server.should_exit = True

    # One request.

    async def _app(self, scope: dict[str, Any], receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # lifespan events are off and WebSockets are not served
            return
        exchange = _Exchange(time=utc_now(), method=scope["method"], path=scope["path"][:LOGGED_PATH_CHARS],
                             started=time.monotonic())
        try:
            await self._handle(scope, receive, send, exchange)
        finally:
            self._log(exchange)

    async def _handle(self, scope: dict[str, Any], receive: Receive, send: Send, exchange: _Exchange) -> None:
        headers: RawHeaders = scope["headers"]
        if not self._authorized(headers):
            await _reply(send, exchange, 401, "invalid_proxy_token",
                         "this proxy needs its per-run token as 'Authorization: Bearer <token>'",
                         [(b"www-authenticate", b"Bearer")])
            return
        url = self._upstream_url(scope)
        if url is None:
            await _reply(send, exchange, 404, "not_found",
                         f"the proxy forwards paths below {PROXY_PATH}/ only, without dot segments")
            return
        body = await _read_body(receive)
        if body is None:
            exchange.status, exchange.error = 499, "client_disconnected"
            return
        assert self._client is not None and self._stopping is not None
        request = self._client.build_request(scope["method"], url, headers=self._upstream_headers(headers),
                                             content=body)
        forward = asyncio.create_task(self._forward(request, send, exchange))
        left = asyncio.create_task(_client_left(receive, exchange))
        stopping = asyncio.create_task(self._stopping.wait())
        tasks = (forward, left, stopping)
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if forward in done or (left in done and not left.result()):
                await forward  # finished, or closing the upstream response after the last chunk went out
                return
            forward.cancel()
            await asyncio.gather(forward, return_exceptions=True)
            if left in done:
                exchange.error = "client_disconnected"
                if exchange.status is None:
                    exchange.status = 499
            else:
                # A started response is left incomplete: uvicorn then closes the connection, so the client
                # cannot take a cut-off body for a whole one.
                exchange.error = "proxy_stopped"
                if not exchange.response_started:
                    await _reply(send, exchange, 503, "proxy_stopped", "the model proxy is shutting down")
        finally:
            pending = [task for task in tasks if not task.done()]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    async def _forward(self, request: httpx2.Request, send: Send, exchange: _Exchange) -> None:
        assert self._client is not None
        try:
            response = await self._client.send(request, stream=True)
        except httpx2.TimeoutException as exc:
            await _reply(send, exchange, 504, "upstream_timeout",
                         f"the upstream did not respond in time ({type(exc).__name__})")
            return
        except httpx2.HTTPError as exc:
            await _reply(send, exchange, 502, "upstream_unreachable",
                         f"the upstream request failed ({type(exc).__name__})")
            return
        try:
            exchange.latency_s = exchange.elapsed()
            exchange.request_id = _first_header(response.headers, REQUEST_ID_HEADERS)
            exchange.served_model = _first_header(response.headers, SERVED_MODEL_HEADERS)
            encodings = {coding.strip().lower()
                         for coding in response.headers.get_list("content-encoding", split_commas=True)}
            if not encodings <= DECODED_ENCODINGS | {""}:
                await _reply(send, exchange, 502, "unsupported_encoding",
                             "the upstream response has a content encoding the proxy cannot decode")
                return
            exchange.status, exchange.response_started = response.status_code, True
            await send({"type": "http.response.start", "status": response.status_code,
                        "headers": self._response_headers(response.headers.raw)})
            redactor = _Redactor(self._credential.secret)
            try:
                async with contextlib.aclosing(response.aiter_bytes()) as chunks:
                    async for chunk in chunks:
                        data = redactor.feed(chunk)
                        if data:
                            await send({"type": "http.response.body", "body": data, "more_body": True})
            except httpx2.HTTPError as exc:
                # Ending the body normally would pass a cut-off response off as complete; returning without the
                # final message makes uvicorn close the connection instead.
                exchange.error = "upstream_stream_error"
                logger.warning("model proxy: the upstream response broke off (%s)", type(exc).__name__)
                return
            await send({"type": "http.response.body", "body": redactor.flush(), "more_body": False})
            exchange.response_complete = True
        finally:
            await response.aclose()

    # Requests and responses.

    def _authorized(self, headers: RawHeaders) -> bool:
        values = [value for name, value in headers if name.lower() == b"authorization"]
        if len(values) != 1:
            return False
        scheme, _, credentials = values[0].strip().partition(b" ")
        return scheme.lower() == b"bearer" and hmac.compare_digest(credentials.strip(), self._token_bytes)

    def _upstream_url(self, scope: Mapping[str, Any]) -> httpx2.URL | None:
        """The upstream URL for a request, or ``None`` when the proxy does not forward its path."""
        try:
            raw_path = scope["raw_path"].decode("ascii")
            query = scope["query_string"].decode("ascii")
        except (KeyError, AttributeError, UnicodeDecodeError):
            return None
        if raw_path == PROXY_PATH:
            suffix = ""
        elif raw_path.startswith(PROXY_PATH + "/"):
            suffix = raw_path[len(PROXY_PATH):]
        else:
            return None
        if any(segment in {".", ".."} for segment in unquote(suffix).split("/")):
            return None
        try:
            url = httpx2.URL(self._upstream_base + suffix + (f"?{query}" if query else ""))
        except httpx2.InvalidURL:
            return None
        return url if (url.scheme, url.host, url.port) == self._upstream_origin else None

    def _upstream_headers(self, headers: RawHeaders) -> list[tuple[bytes, bytes]]:
        dropped = HOP_BY_HOP_HEADERS | CLIENT_CREDENTIAL_HEADERS | RESET_REQUEST_HEADERS | _connection_tokens(headers)
        forwarded = [(name.lower(), value) for name, value in headers if name.lower() not in dropped]
        return [*forwarded, (b"authorization", self._credential.authorization), (b"accept-encoding", b"identity")]

    def _response_headers(self, headers: RawHeaders) -> list[tuple[bytes, bytes]]:
        dropped = HOP_BY_HOP_HEADERS | DROPPED_RESPONSE_HEADERS | _connection_tokens(headers)
        secret = self._credential.secret
        return [(name.lower(), value.replace(secret, REDACTED)) for name, value in headers
                if name.lower() not in dropped]

    def _log(self, exchange: _Exchange) -> None:
        if self._log_file is None:
            return
        try:
            self._log_file.write(json.dumps(exchange.log_entry()) + "\n")
            self._log_file.flush()
        except (OSError, ValueError) as exc:  # ValueError: the file is already closed
            logger.warning("model proxy: cannot write its request log (%s)", type(exc).__name__)


def codex_provider_config(proxy: ModelProxy, *, provider_id: str = "wse_proxy") -> dict[str, Any]:
    """The Codex ``config.toml`` entries that send an agent's model calls through ``proxy``.

    A custom model provider (``model_providers.<provider_id>``) with the proxy's
    URL as ``base_url``, the Responses API wire protocol, and the bearer token
    read from ``PROXY_TOKEN_ENV``, selected as ``model_provider``. The Codex
    process needs ``WSE_MODEL_PROXY_TOKEN=<proxy.token>`` in its environment
    and no API key. ``provider_id`` is a TOML bare key (letters, digits, ``_``,
    ``-``) and must not name one of Codex's built-in providers, which a
    configuration cannot override.
    """
    if not isinstance(provider_id, str) or not PROVIDER_ID.fullmatch(provider_id):
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "a Codex model provider ID has letters, digits, '_', and '-' only",
                       details={"provider_id": str(provider_id)})
    return {
        "model_providers": {
            provider_id: {"name": "WSE model proxy", "base_url": proxy.url, "env_key": PROXY_TOKEN_ENV,
                          "wire_api": "responses"},
        },
        "model_provider": provider_id,
    }


# Validation and small helpers.


def _checked_upstream(base_url: object) -> tuple[str, tuple[str, str, int | None], bool]:
    """The upstream base URL without a trailing slash, its origin as httpx2 parses it, and whether it is loopback."""
    if not isinstance(base_url, str) or not base_url.strip():
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "the model proxy needs an upstream base URL (OPENAI_BASE_URL)")
    try:
        parts = urlsplit(base_url.strip())
        has_credentials = parts.username is not None or parts.password is not None
        _ = parts.port  # raises for a malformed port
    except ValueError as exc:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "the model proxy's upstream base URL is malformed") from exc
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "the model proxy's upstream base URL must be http(s)://<host>/...")
    if has_credentials or parts.query or parts.fragment:
        raise WSEError(ErrorCode.INVALID_ARGUMENT,
                       "the model proxy's upstream base URL must not carry credentials, a query, or a fragment")
    loopback = _is_loopback_host(parts.hostname)
    if parts.scheme == "http" and not loopback:
        raise WSEError(ErrorCode.INVALID_ARGUMENT,
                       "the model proxy sends the key upstream only over https (plain http only to loopback)")
    base = f"{parts.scheme}://{parts.netloc}{parts.path.rstrip('/')}"
    try:
        url = httpx2.URL(base)
    except httpx2.InvalidURL as exc:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "the model proxy's upstream base URL is malformed") from exc
    return base, (url.scheme, url.host, url.port), loopback


def _checked_key(api_key: object) -> str:
    key = api_key.strip() if isinstance(api_key, str) else ""
    if not key:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "the model proxy needs a non-empty API key (OPENAI_API_KEY)")
    if not all(33 <= ord(character) <= 126 for character in key):
        raise WSEError(ErrorCode.INVALID_ARGUMENT,
                       "the API key has characters that cannot appear in an HTTP header "
                       "(spaces, control characters, or non-ASCII)")
    return key


def _checked_host(host: object) -> tuple[str, socket.AddressFamily]:
    try:
        address = ipaddress.ip_address(host) if isinstance(host, str) else None
    except ValueError:
        address = None
    if address is None or not address.is_loopback:
        raise WSEError(ErrorCode.INVALID_ARGUMENT,
                       "the model proxy binds only to a loopback IP address, such as 127.0.0.1")
    return str(address), socket.AF_INET6 if address.version == 6 else socket.AF_INET


def _is_loopback_host(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _connection_tokens(headers: RawHeaders) -> frozenset[bytes]:
    """The header names that a ``Connection`` header marks as hop-by-hop."""
    return frozenset(token.strip().lower() for name, value in headers if name.lower() == b"connection"
                     for token in value.split(b",") if token.strip())


def _first_header(headers: httpx2.Headers, names: Iterable[str]) -> str | None:
    for name in names:
        value = headers.get(name)
        if value:
            return value[:200]
    return None


async def _read_body(receive: Receive) -> bytes | None:
    """The whole request body, or ``None`` if the client went away first."""
    chunks = []
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return None
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            return b"".join(chunks)


async def _client_left(receive: Receive, exchange: _Exchange) -> bool:
    """Wait for the connection's end: True if the client left before the response was complete."""
    while (await receive())["type"] != "http.disconnect":
        pass
    return not exchange.response_complete


async def _reply(send: Send, exchange: _Exchange, status: int, code: str, message: str,
                 headers: Sequence[tuple[bytes, bytes]] = ()) -> None:
    """Answer from the proxy itself, with an OpenAI-style error body."""
    body = json.dumps({"error": {"message": message, "type": "wse_model_proxy_error", "code": code}}).encode()
    exchange.status, exchange.error, exchange.response_started = status, code, True
    if exchange.latency_s is None:
        exchange.latency_s = exchange.elapsed()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                            *headers]})
    await send({"type": "http.response.body", "body": body, "more_body": False})
    exchange.response_complete = True
