"""Run agent-written Python in a fresh, allow-listed, network-less process (M3 plan Section 2.1).

Each call writes the code to ``calls/<n>/code.py`` in the run's sandbox
directory and runs it with the environment's own Python under macOS
``sandbox-exec``. The profile denies everything, imports Apple's
``system.sb`` (the system libraries, devices, and services every process
needs), and then allows only:

* reading and mapping the Python runtime (``sys.base_prefix`` and
  ``sys.prefix``) and executing its interpreter;
* reading ``inputs/`` and ``calls/``, and the sandbox root itself, which
  Python needs to resolve its working directory;
* reading and writing ``work/``, ``outputs/``, and ``tmp/``;
* the metadata of the directories above those paths, which path
  resolution needs, and nothing else of them.

Everything else is denied: the rest of the workspace (evaluator storage
among it), the repository, the user's files, other executables, new
processes, and the network. The profile is written next to the sandbox
directory, not inside it. The environment is an allow-list
(``ENVIRONMENT_NAMES``), so no credential reaches the process.

A framework build of Python (python.org's macOS installer) starts its real
interpreter from a launcher with ``posix_spawn``; the sandbox runs that
interpreter directly and names the environment's Python in
``__PYVENV_LAUNCHER__``, as the launcher would, so no new process is
needed. A bootstrap sets the CPU-time and file-size limits before the
agent's code runs, and the server enforces the wall-clock limit by killing
the process group. Standard output and error are each returned within a
byte limit and stored in full up to ``STORED_OUTPUT_BYTES``.

``outputs/`` is emptied before each call, so a call's outputs are the files
it wrote there; ``work/`` keeps files between calls. Only regular files with
a single link count as outputs. The server never follows a symbolic link or
a hard link that the sandboxed code made: either could point at a file the
sandbox itself cannot read. Each call keeps a copy of its output files of at
most ``KEPT_OUTPUT_BYTES`` in ``calls/<n>/outputs/``, for the record.

The isolation is offered only on macOS with ``/usr/bin/sandbox-exec``, and
only after a self-check confirms that an allowed read works and a read
outside the allow-list fails (``available``). Elsewhere it fails closed.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO

from wse.schemas import ErrorCode, WSEError

SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
STORED_OUTPUT_BYTES = 1 << 20
TAIL_BYTES = 64 * 1024
FILE_SIZE_LIMIT = 1 << 30
KEPT_OUTPUT_BYTES = 1 << 20
ENVIRONMENT_NAMES = ("HOME", "LANG", "MPLCONFIGDIR", "PATH", "PYTHONIOENCODING", "TMPDIR", "XDG_CACHE_HOME",
                     "__PYVENV_LAUNCHER__")
BOOTSTRAP = """\
import resource
import runpy
import sys

cpu_seconds, file_bytes, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
sys.argv = [path]
runpy.run_path(path, run_name="__main__")
"""

__all__ = ["ENVIRONMENT_NAMES", "KEPT_OUTPUT_BYTES", "OutputFile", "SandboxLayout", "SandboxResult", "available",
           "profile_text", "read_output", "run_python"]


@dataclass(frozen=True)
class SandboxLayout:
    """A run's sandbox directory: ``inputs`` (read only), ``work``, ``outputs``, ``tmp``, and ``calls``."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    @property
    def inputs(self) -> Path:
        return self.root / "inputs"

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def tmp(self) -> Path:
        return self.root / "tmp"

    @property
    def calls(self) -> Path:
        return self.root / "calls"

    @property
    def profile_path(self) -> Path:
        """The profile, next to the sandbox directory, where the sandboxed code cannot read it."""
        return self.root.parent / f"{self.root.name}.sb"

    def create(self) -> SandboxLayout:
        for directory in (self.inputs, self.work, self.outputs, self.tmp, self.calls):
            directory.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class OutputFile:
    """A regular file in ``outputs/``: its path relative to ``outputs/``, its size, and its SHA-256."""

    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class SandboxResult:
    """One call: how it ended, what it cost, its bounded output, and the files it left in ``outputs/``.

    ``ending`` is ``exited`` (see ``exit_code``), ``timed_out`` (the
    wall-clock limit), ``cpu_limit``, ``file_size_limit``, or ``signal``
    (see ``signal``). ``stdout`` and ``stderr`` are bounded for the agent;
    ``calls/<call>/`` stores them in full up to ``STORED_OUTPUT_BYTES``.
    """

    call: int
    ending: str
    exit_code: int | None
    signal: str | None
    wall_seconds: float
    cpu_seconds: float
    max_rss_bytes: int
    stdout: str
    stderr: str
    stdout_bytes: int
    stderr_bytes: int
    stdout_truncated: bool
    stderr_truncated: bool
    outputs: tuple[OutputFile, ...]
    refused_outputs: tuple[str, ...]
    code_sha256: str


@dataclass(frozen=True)
class _Runtime:
    executable: Path
    launcher: Path | None
    roots: tuple[Path, ...]


def _runtime() -> _Runtime:
    """The interpreter to run, the launcher path a framework build expects, and the runtime's directories."""
    environment_python = Path(sys.executable).absolute()
    roots = tuple(dict.fromkeys(Path(p).resolve() for p in (sys.base_prefix, sys.prefix)))
    app = Path(sys.base_prefix).resolve() / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"
    if app.exists():
        return _Runtime(app, environment_python, roots)
    return _Runtime(environment_python, None, roots)


def _quoted(path: Path | str) -> str:
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _ancestors(paths: list[Path]) -> list[Path]:
    found: dict[str, Path] = {}
    for path in paths:
        for parent in path.parents:
            found[str(parent)] = parent
    return [found[key] for key in sorted(found)]


def profile_text(layout: SandboxLayout) -> str:
    """The ``sandbox-exec`` profile of ``layout`` (module docstring)."""
    runtime = _runtime()
    executables = {runtime.executable, Path(os.path.realpath(runtime.executable))}
    if runtime.launcher is None:
        executables |= {Path(sys.executable).absolute(), Path(os.path.realpath(sys.executable))}
    allowed = [*runtime.roots, layout.root]

    def group(kind: str, paths: list[Path] | set[Path]) -> str:
        return " ".join(f"({kind} {_quoted(p)})" for p in sorted(paths, key=str))

    lines = [
        "(version 1)",
        "(deny default)",
        '(import "system.sb")',
        f"(allow process-exec {group('literal', executables)})",
        f"(allow file-read* file-map-executable {group('subpath', list(runtime.roots))})",
        f"(allow file-read* (literal {_quoted(layout.root)}) {group('subpath', [layout.inputs, layout.calls])})",
        f"(allow file-read* file-write* {group('subpath', [layout.work, layout.outputs, layout.tmp])})",
        f"(allow file-read-metadata {group('literal', _ancestors(allowed))})",
        "(allow signal (target self))",
    ]
    return "\n".join(lines) + "\n"


def _environment(layout: SandboxLayout, runtime: _Runtime) -> dict[str, str]:
    environment = {
        "HOME": str(layout.work),
        "LANG": "en_US.UTF-8",
        "MPLCONFIGDIR": str(layout.work / ".matplotlib"),
        "PATH": "/usr/bin:/bin",
        "PYTHONIOENCODING": "utf-8",
        "TMPDIR": str(layout.tmp) + "/",
        "XDG_CACHE_HOME": str(layout.work / ".cache"),
    }
    if runtime.launcher is not None:
        environment["__PYVENV_LAUNCHER__"] = str(runtime.launcher)
    return environment


class _Capture:
    """Read a stream to its end, keeping its head and tail and counting every byte."""

    def __init__(self, stream: IO[bytes]) -> None:
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0
        self._thread = threading.Thread(target=self._read, args=(stream,), daemon=True)
        self._thread.start()

    def _read(self, stream: IO[bytes]) -> None:
        try:
            while chunk := stream.read(65536):
                self.total += len(chunk)
                room = STORED_OUTPUT_BYTES - len(self.head)
                if room > 0:
                    self.head += chunk[:room]
                    chunk = chunk[room:]
                if chunk:
                    self.tail += chunk
                    del self.tail[:-TAIL_BYTES]
        except (OSError, ValueError):
            pass

    def join(self, timeout: float) -> None:
        self._thread.join(timeout)

    def stored(self) -> bytes:
        if self.total <= len(self.head):
            return bytes(self.head)
        return bytes(self.head) + f"\n[... {self.total - len(self.head) - len(self.tail)} bytes not stored ...]\n".encode() \
            + bytes(self.tail)

    def bounded(self, limit: int) -> tuple[str, bool]:
        """At most ``limit`` UTF-8 bytes for the agent: all of it, or its start and its end around a marker."""
        if self.total <= limit:
            return bytes(self.head).decode("utf-8", errors="replace"), False
        marker = f"\n[... {self.total} bytes in all; the middle is not shown ...]\n"
        budget = max(0, limit - len(marker.encode()))
        first = bytes(self.head[: budget // 2]).decode("utf-8", errors="ignore")
        end = bytes(self.head) + bytes(self.tail) if self.total > len(self.head) else bytes(self.head)
        last_bytes = budget - len(first.encode())
        last = end[len(end) - last_bytes:].decode("utf-8", errors="ignore") if last_bytes > 0 else ""
        return first + marker + last, True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def list_outputs(layout: SandboxLayout) -> tuple[tuple[OutputFile, ...], tuple[str, ...]]:
    """The regular, singly linked files of ``outputs/`` with their hashes, and every other entry, refused."""
    files: list[OutputFile] = []
    refused: list[str] = []
    for directory, subdirectories, names in os.walk(layout.outputs, followlinks=False):
        base = Path(directory)
        for name in sorted(subdirectories):
            if (base / name).is_symlink():
                refused.append(str((base / name).relative_to(layout.outputs)))
        for name in sorted(names):
            path = base / name
            relative = str(path.relative_to(layout.outputs))
            info = os.lstat(path)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                refused.append(relative)
                continue
            files.append(OutputFile(relative, info.st_size, _sha256_file(path)))
    return tuple(sorted(files, key=lambda f: f.path)), tuple(sorted(refused))


def read_output(layout: SandboxLayout, output: OutputFile, limit: int) -> bytes | None:
    """The bytes of ``output`` if it is still a regular, singly linked file of at most ``limit`` bytes with its hash.

    The file is opened without following a link, so a link the sandboxed
    code made is never read through.
    """
    try:
        descriptor = os.open(layout.outputs / output.path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > limit:
            return None
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            data = handle.read(limit + 1)
    finally:
        os.close(descriptor)
    if len(data) > limit or "sha256:" + hashlib.sha256(data).hexdigest() != output.sha256:
        return None
    return data


def _empty(directory: Path) -> None:
    """Remove everything inside ``directory`` without following links."""
    with os.scandir(directory) as entries:
        for entry in list(entries):
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path)
            else:
                os.unlink(entry.path)


def _keep(layout: SandboxLayout, directory: Path, outputs: tuple[OutputFile, ...]) -> None:
    """Copy each output file of at most ``KEPT_OUTPUT_BYTES`` into ``directory/outputs``."""
    for output in outputs:
        data = read_output(layout, output, KEPT_OUTPUT_BYTES)
        if data is None:
            continue
        target = directory / "outputs" / output.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _new_call(layout: SandboxLayout) -> tuple[int, Path]:
    numbers = [int(p.name) for p in layout.calls.iterdir() if p.is_dir() and p.name.isdigit()]
    number = max(numbers, default=0) + 1
    while True:
        directory = layout.calls / str(number)
        try:
            directory.mkdir()
            return number, directory
        except FileExistsError:
            number += 1


def _ending(status: int, timed_out: bool) -> tuple[str, int | None, str | None]:
    if timed_out:
        return "timed_out", None, "SIGKILL"
    if os.WIFEXITED(status):
        return "exited", os.WEXITSTATUS(status), None
    number = os.WTERMSIG(status)
    name = signal.Signals(number).name if number in signal.Signals._value2member_map_ else str(number)
    if number == signal.SIGXCPU:
        return "cpu_limit", None, name
    if number == signal.SIGXFSZ:
        return "file_size_limit", None, name
    return "signal", None, name


@functools.cache
def available() -> tuple[bool, str]:
    """Whether the sandbox can run here, and why: macOS, ``sandbox-exec``, and a passing self-check."""
    if sys.platform != "darwin":
        return False, f"the sandbox needs macOS's sandbox-exec; this platform is {sys.platform}"
    if not (SANDBOX_EXEC.is_file() and os.access(SANDBOX_EXEC, os.X_OK)):
        return False, f"the sandbox needs {SANDBOX_EXEC}, which is missing"
    with tempfile.TemporaryDirectory(prefix="wse-sandbox-check-") as temporary:
        layout = SandboxLayout(Path(temporary) / "sandboxes" / "check").create()
        (layout.inputs / "allowed.txt").write_text("allowed")
        planted = Path(temporary) / "planted.txt"
        planted.write_text("planted")
        code = ("try:\n"
                f"    open({str(planted)!r}).read(1)\n"
                "    print('leaked')\n"
                "except OSError:\n"
                "    print(open('inputs/allowed.txt').read())\n")
        try:
            result = _run(layout, code, wall_seconds=60.0, cpu_seconds=60, max_output_bytes=1024)
        except OSError as exc:
            return False, f"the sandbox self-check could not start: {exc}"
    if (result.ending, result.exit_code, result.stdout.strip()) != ("exited", 0, "allowed"):
        return False, (f"the sandbox self-check failed: ended {result.ending}, exit code {result.exit_code}, "
                       f"output {result.stdout.strip()[:80]!r}, errors {result.stderr.strip()[-200:]!r}")
    return True, "sandbox-exec with an allow-list, checked"


def run_python(layout: SandboxLayout, code: str, *, wall_seconds: float, cpu_seconds: int,
               max_output_bytes: int) -> SandboxResult:
    """Run ``code`` once in the sandbox of ``layout`` (module docstring); raise if the sandbox is unavailable."""
    ok, reason = available()
    if not ok:
        raise WSEError(ErrorCode.UNSUPPORTED_CAPABILITY, f"code execution is not available here: {reason}",
                       details={"reason": reason})
    return _run(layout, code, wall_seconds=wall_seconds, cpu_seconds=cpu_seconds, max_output_bytes=max_output_bytes)


def _run(layout: SandboxLayout, code: str, *, wall_seconds: float, cpu_seconds: int,
         max_output_bytes: int) -> SandboxResult:
    if wall_seconds <= 0 or cpu_seconds <= 0 or max_output_bytes <= 0:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "the sandbox limits must be positive")
    layout.create()
    _empty(layout.outputs)
    runtime = _runtime()
    layout.profile_path.write_text(profile_text(layout), encoding="utf-8")
    bootstrap = layout.calls / "bootstrap.py"
    if not bootstrap.exists() or bootstrap.read_text(encoding="utf-8") != BOOTSTRAP:
        bootstrap.write_text(BOOTSTRAP, encoding="utf-8")
    number, directory = _new_call(layout)
    code_path = directory / "code.py"
    code_path.write_text(code, encoding="utf-8")
    command = [str(SANDBOX_EXEC), "-f", str(layout.profile_path), str(runtime.executable), "-I", "-B",
               str(bootstrap), str(int(cpu_seconds)), str(FILE_SIZE_LIMIT), str(code_path)]
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=layout.root, env=_environment(layout, runtime), stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    assert process.stdout is not None and process.stderr is not None
    out, err = _Capture(process.stdout), _Capture(process.stderr)
    timed_out = False
    deadline = started + wall_seconds
    while True:
        pid, status, usage = os.wait4(process.pid, os.WNOHANG)
        if pid:
            break
        if time.monotonic() >= deadline:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            pid, status, usage = os.wait4(process.pid, 0)
            break
        time.sleep(0.02)
    process.returncode = os.waitstatus_to_exitcode(status)
    wall = time.monotonic() - started
    for capture in (out, err):
        capture.join(5.0)
    for stream in (process.stdout, process.stderr):
        stream.close()
    ending, exit_code, signal_name = _ending(status, timed_out)
    stdout, stdout_cut = out.bounded(max_output_bytes)
    stderr, stderr_cut = err.bounded(max_output_bytes)
    (directory / "stdout.txt").write_bytes(out.stored())
    (directory / "stderr.txt").write_bytes(err.stored())
    outputs, refused = list_outputs(layout)
    _keep(layout, directory, outputs)
    result = SandboxResult(
        call=number, ending=ending, exit_code=exit_code, signal=signal_name, wall_seconds=round(wall, 3),
        cpu_seconds=round(usage.ru_utime + usage.ru_stime, 3), max_rss_bytes=int(usage.ru_maxrss),
        stdout=stdout, stderr=stderr, stdout_bytes=out.total, stderr_bytes=err.total, stdout_truncated=stdout_cut,
        stderr_truncated=stderr_cut, outputs=outputs, refused_outputs=refused,
        code_sha256="sha256:" + hashlib.sha256(code.encode("utf-8")).hexdigest())
    summary = {key: value for key, value in asdict(result).items() if key not in ("stdout", "stderr")}
    (directory / "result.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
