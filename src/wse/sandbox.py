"""Run agent-written Python in a persistent, allow-listed, network-less process (M3 plan Section 2.1).

A run's sandbox holds one **kernel**: a Python process started on the first
call and kept between calls, so variables, imports, and loaded tensors
persist, as in a notebook. Each call writes its code to
``calls/<n>/code.py`` and the kernel executes it in the same namespace. The
kernel runs the environment's own Python under macOS ``sandbox-exec``. The
profile denies everything, imports Apple's ``system.sb`` (the system
libraries, devices, and services every process needs), and then allows only:

* reading and mapping the Python runtime and executing its interpreter:
  the base installation (``sys.base_prefix``) and the environment
  (``sys.prefix``) without their ``site-packages`` directories, and, in
  those, only the files of the analysis libraries
  (``ANALYSIS_DISTRIBUTIONS``: NumPy, PyTorch, safetensors) and of their
  declared dependencies, as each distribution's record lists them. A
  ``site-packages`` directory itself may be listed, not read, so WSE's own
  distribution (whose metadata holds its README) and every other installed
  package stay unreadable;
* reading ``inputs/`` and ``calls/``, and the sandbox root itself, which
  Python needs to resolve its working directory;
* reading and writing what is inside ``work/``, ``outputs/``, and ``tmp/``,
  but not those directories themselves: a rule after the allow (Seatbelt
  applies the last matching rule) denies removing, renaming, replacing, or
  changing the entries ``inputs``, ``work``, ``outputs``, and ``tmp``, so the
  code cannot put a link where WSE expects one of them;
* the metadata of the directories above those paths, which path
  resolution needs, and nothing else of them.

Everything else is denied: the rest of the workspace (evaluator storage
and the database among it), the repository, the user's files, other
executables, new processes, and the network. The profile is written next to
the sandbox directory, not inside it. The environment is an allow-list
(``ENVIRONMENT_NAMES``), so no credential reaches the process.

**The host side.** WSE never resolves a path inside the sandbox directory
that agent code can change, and never follows a link there. It opens the
root without following a link, and ``inputs/``, ``work/``, ``outputs/``,
``tmp/``, and ``calls/`` relative to it the same way; an entry that is not a
real directory (a link, a file, a FIFO that the agent's own commands left)
is removed itself, never what it points to, a new directory is made in its
place, and the result of the call says so (``repairs``). WSE checks before
each call and again after it, before it lists ``outputs/``.

A framework build of Python (python.org's macOS installer) starts its real
interpreter from a launcher with ``posix_spawn``; the sandbox runs that
interpreter directly and names the environment's Python in
``__PYVENV_LAUNCHER__``, as the launcher would, so no new process is
needed.

**Limits.** A call that outlives its wall-clock limit is interrupted
(``SIGINT``, a ``KeyboardInterrupt`` in the code) and keeps the kernel and
its variables; a call that does not stop within ``INTERRUPT_GRACE_SECONDS``
of the interrupt is killed with its process group, and the next call starts
a new kernel. While a call runs, WSE reads the kernel's physical memory
footprint every ``MEMORY_POLL_SECONDS`` and kills it above its limit. The
kernel's CPU time over its whole life is limited (``RLIMIT_CPU``), as is the
size of any file it writes (``FILE_SIZE_LIMIT``). A crash, an ``os._exit``,
or a limit ends the kernel; ``sys.exit`` only ends the call. Each result
says whether the kernel and its variables survived (``state_kept``) and
whether the call started a new one (``restarted``). Code may start threads,
which keep running between calls: the limits bound the kernel, not each
thread.

**Output.** The kernel's standard output and error are read continuously;
each call takes what it wrote up to a marker the kernel prints when the
call ends, so output that a thread prints later belongs to the next call.
Each stream is returned to the agent within a byte limit (``shorten``: its
start and end around a marker) and stored, up to its first
``STORED_OUTPUT_BYTES`` and last ``TAIL_BYTES``, in ``calls/<n>/``.

``outputs/`` is emptied before each call, so a call's outputs are the files
it wrote there; ``work/`` keeps files between calls. WSE empties it by
renaming the directory aside (relative to the root's descriptor), making a
new one, and deleting the old tree by descriptor without following a link.
Only regular files with a single link count as outputs. WSE lists
``outputs/`` from descriptors only: each entry is opened relative to its
directory's descriptor without following a link and without waiting on a
FIFO, and is listed only if that descriptor is a regular file with a single
link (a hard link could name a file the sandbox itself cannot read); links,
FIFOs, sockets, hard-linked files, and directories deeper than
``MAX_OUTPUT_DEPTH`` are refused, never followed. Each file is hashed, kept,
and handed on from the one descriptor it was opened as: a call keeps a copy
of its output files of at most ``KEPT_OUTPUT_BYTES`` in
``calls/<n>/outputs/``, for the record, and ``run_python``'s ``use_output``
reads the first ones from their descriptors.

The isolation is offered only on macOS with ``/usr/bin/sandbox-exec``, and
only after a self-check confirms that an allowed read works and a read
outside the allow-list fails (``available``). Elsewhere it fails closed.
"""

from __future__ import annotations

import atexit
import bisect
import contextlib
import ctypes
import errno
import functools
import hashlib
import importlib.metadata
import json
import os
import queue
import re
import secrets
import shutil
import signal
import site
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import IO, Any

from wse.schemas import ErrorCode, WSEError

SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
STORED_OUTPUT_BYTES = 1 << 20  # the start of a call's stream stored in calls/<n>/
TAIL_BYTES = 512 * 1024  # and its end: at least half of the largest output limit a policy may set
FILE_SIZE_LIMIT = 4 << 30
DEFAULT_MEMORY_LIMIT = 16 << 30
INTERRUPT_GRACE_SECONDS = 5.0
MEMORY_POLL_SECONDS = 0.25
OUTPUT_SETTLE_SECONDS = 2.0  # how long the end-of-call markers may trail the kernel's reply
# The libraries agent code may import; their declared dependencies come with them (``_analysis_entries``).
ANALYSIS_DISTRIBUTIONS = ("numpy", "torch", "safetensors")
KEPT_OUTPUT_BYTES = 1 << 20
MAX_OUTPUT_DEPTH = 32  # levels of directories in outputs/ that are listed; a directory deeper is refused
SANDBOX_DIRECTORIES = ("inputs", "work", "outputs", "tmp", "calls")  # each a real directory, checked by descriptor
PROTECTED_DIRECTORIES = ("inputs", "work", "outputs", "tmp")  # entries the kernel may not remove, rename, or replace
_REMOVED_PREFIX = ".wse-removed-"  # an emptied outputs/, renamed aside in the root until it is deleted
# Opening an entry agent code may have made: never through a link, never waiting on a FIFO.
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
ENVIRONMENT_NAMES =("HOME", "LANG", "MPLCONFIGDIR", "PATH", "PYTHONIOENCODING", "TMPDIR", "XDG_CACHE_HOME",
                     "__PYVENV_LAUNCHER__")
# The kernel: limits, then one call per request line, in one namespace. Requests arrive on their own descriptor and
# replies leave on another, so the code's standard input is empty and its standard output is its own.
KERNEL = """\
import json
import os
import resource
import signal
import sys
import time
import traceback

cpu_seconds, file_bytes, request_fd, reply_fd = (int(value) for value in sys.argv[1:5])
resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
requests = os.fdopen(request_fd, "r", encoding="utf-8")
replies = os.fdopen(reply_fd, "w", encoding="utf-8")
sys.stdin = open(os.devnull, encoding="utf-8")
sys.argv = ["wse-kernel"]
running = [False]


def interrupt(signum, frame):
    if running[0]:
        raise KeyboardInterrupt


signal.signal(signal.SIGINT, interrupt)


def cpu():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def serve():
    namespace = {"__name__": "__main__", "__builtins__": __builtins__}
    for line in requests:
        request = json.loads(line)
        status, error = "completed", None
        started_cpu, started = cpu(), time.monotonic()
        try:
            with open(request["path"], encoding="utf-8") as handle:
                code = compile(handle.read(), request["path"], "exec")
            running[0] = True
            try:
                exec(code, namespace)
            finally:
                running[0] = False
        except KeyboardInterrupt as exc:
            status, error = "interrupted", "KeyboardInterrupt"
            traceback.print_exception(type(exc), exc, exc.__traceback__.tb_next)
        except SystemExit as exc:
            status, error = "error", "SystemExit: " + repr(exc.code)
        except BaseException as exc:
            status, error = "error", (type(exc).__name__ + ": " + str(exc))[:500]
            traceback.print_exception(type(exc), exc, exc.__traceback__.tb_next)
        for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
            try:
                stream.flush()
            except Exception:
                pass
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        marker = request["marker"].encode()
        os.write(1, marker)
        os.write(2, marker)
        replies.write(json.dumps({"marker": request["marker"], "status": status, "error": error,
                                  "cpu_seconds": round(cpu() - started_cpu, 3),
                                  "wall_seconds": round(time.monotonic() - started, 3)}) + "\\n")
        replies.flush()


serve()
"""

__all__ = ["ANALYSIS_DISTRIBUTIONS", "DEFAULT_MEMORY_LIMIT", "ENVIRONMENT_NAMES", "KEPT_OUTPUT_BYTES", "Kernel",
           "MAX_OUTPUT_DEPTH", "NotRegularFile", "OutputFile", "SandboxLayout", "SandboxResult", "available",
           "close_kernels", "footprint", "kernel_for", "list_outputs", "open_beneath", "profile_text", "read_output",
           "run_python", "shorten"]


class NotRegularFile(OSError):
    """An entry of the sandbox directory that is not a regular file: a FIFO, a socket, a directory, or a device."""


def _kind(mode: int) -> str:
    """What a file of ``mode`` is, as a phrase."""
    if stat.S_ISLNK(mode):
        return "a symbolic link"
    if stat.S_ISREG(mode):
        return "a regular file"
    if stat.S_ISDIR(mode):
        return "a directory"
    if stat.S_ISFIFO(mode):
        return "a FIFO"
    if stat.S_ISSOCK(mode):
        return "a socket"
    return "a device or other special file"


def _open_root(root: Path) -> int:
    """Open a sandbox's root, which only WSE writes, as a directory; its last level is never followed as a link."""
    root.mkdir(parents=True, exist_ok=True)
    return os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)


def _ensure_directory(parent: int, name: str) -> tuple[int, str | None]:
    """Open ``name`` in the directory ``parent`` as a real directory, making it when missing; return it and what it replaced.

    The entry is opened without following a link. An entry of any other kind
    (a link, a file, a FIFO) is removed itself, never what it points to, and a
    new, empty directory is made in its place; the second value then says
    what the entry was.
    """
    replaced = None
    for _ in range(8):  # an entry that agent code keeps changing is retried, then reported as an error
        try:
            return os.open(name, _DIRECTORY_FLAGS, dir_fd=parent), replaced
        except FileNotFoundError:
            pass
        except OSError as exc:
            if exc.errno not in (errno.ELOOP, errno.ENOTDIR):
                raise
            try:
                info = os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                info = None
            if info is not None and not stat.S_ISDIR(info.st_mode):
                os.unlink(name, dir_fd=parent)  # unlink removes the entry itself; it never follows a link
                replaced = _kind(info.st_mode)
        with contextlib.suppress(FileExistsError):
            os.mkdir(name, dir_fd=parent)
    raise OSError(errno.EAGAIN, f"{name} kept changing while WSE made it a directory")


def _repair_note(name: str, kind: str, when: str) -> str:
    return (f"{name}/ was {kind}, not a directory, {when}: WSE removed that entry itself (not what it pointed to) "
            f"and made a new, empty {name}/")


def _open_layout(root: Path, when: str) -> tuple[int, list[str]]:
    """Open the sandbox root and make each of ``SANDBOX_DIRECTORIES`` a real directory; return the root and the repairs."""
    descriptor = _open_root(root)
    repairs = []
    try:
        for name in SANDBOX_DIRECTORIES:
            child, replaced = _ensure_directory(descriptor, name)
            os.close(child)
            if replaced is not None:
                repairs.append(_repair_note(name, replaced, when))
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, repairs


# Repairs made outside a call (``SandboxLayout.create``), which the next call in this process reports.
_PENDING_REPAIRS: dict[Path, list[str]] = {}
_PENDING_LOCK = threading.Lock()


def _take_repairs(root: Path) -> list[str]:
    with _PENDING_LOCK:
        return _PENDING_REPAIRS.pop(root, [])


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
        """Make the sandbox's directories, each a real directory, opened by descriptor without following a link.

        An entry that is not a directory is removed itself, never what it
        points to, and replaced by a new directory; the next call's result
        says so (``SandboxResult.repairs``).
        """
        descriptor, repairs = _open_layout(self.root, "found outside a call")
        os.close(descriptor)
        if repairs:
            with _PENDING_LOCK:
                _PENDING_REPAIRS.setdefault(self.root, []).extend(repairs)
        return self


@dataclass(frozen=True)
class OutputFile:
    """A regular file in ``outputs/``: its path relative to ``outputs/``, its size, and its SHA-256."""

    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class SandboxResult:
    """One call: how it ended, what it cost, its bounded output, the files it left in ``outputs/``, and its kernel.

    ``ending`` is ``completed``; ``error`` (the code raised; ``error`` names
    the exception and ``stderr`` holds its traceback); ``timed_out`` (the
    wall-clock limit; ``state_kept`` says whether the interrupt sufficed);
    ``memory_limit``; ``cpu_limit`` (the kernel's lifetime CPU time);
    ``file_size_limit``; or ``crashed`` (the kernel died: see ``exit_code``
    and ``signal``). ``stdout`` and ``stderr`` are bounded for the agent;
    ``calls/<call>/`` stores them (``STORED_OUTPUT_BYTES`` and ``TAIL_BYTES``).
    ``session`` counts the kernels this sandbox started, and ``session_call``
    the calls of this kernel, this one included. ``repairs`` says which of
    the sandbox's directories was not a real directory before or after the
    call and was replaced (module docstring).
    """

    call: int
    ending: str
    exit_code: int | None
    signal: str | None
    error: str | None
    wall_seconds: float
    cpu_seconds: float | None
    peak_memory_bytes: int | None
    stdout: str
    stderr: str
    stdout_bytes: int
    stderr_bytes: int
    stdout_truncated: bool
    stderr_truncated: bool
    outputs: tuple[OutputFile, ...]
    refused_outputs: tuple[str, ...]
    code_sha256: str
    session: int
    session_call: int
    restarted: bool
    state_kept: bool
    repairs: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Runtime:
    executable: Path
    launcher: Path | None
    directories: tuple[Path, ...]  # readable and mappable subtrees
    files: tuple[Path, ...]  # readable and mappable single paths: files, links, and directories that may be listed


def _covering(root: Path, excluded: set[Path]) -> tuple[list[Path], list[Path]]:
    """Subtrees and single paths that together cover ``root`` except the ``excluded`` directories in it."""
    if root in excluded:
        return [], []
    if not any(path.is_relative_to(root) for path in excluded):
        return [root], []
    directories: list[Path] = []
    files = [root]
    with os.scandir(root) as entries:
        for entry in sorted(entries, key=lambda e: e.name):
            path = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                more_directories, more_files = _covering(path, excluded)
                directories += more_directories
                files += more_files
            elif path not in excluded:
                files.append(path)
    return directories, files


def _requirements(distribution: importlib.metadata.Distribution) -> list[str]:
    """The names a distribution requires outside its extras."""
    names = []
    for text in distribution.requires or ():
        requirement, _, marker = text.partition(";")
        match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
        if match and "extra" not in marker:
            names.append(match.group(1))
    return names


def _analysis_entries(sites: set[Path]) -> tuple[list[Path], list[Path]]:
    """The top-level entries, in ``sites``, of the analysis distributions and their declared dependencies."""
    directories: list[Path] = []
    files: list[Path] = []
    pending, seen = list(ANALYSIS_DISTRIBUTIONS), set()
    while pending:
        name = pending.pop()
        key = re.sub(r"[-_.]+", "-", name).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        pending += _requirements(distribution)
        for record in distribution.files or ():
            if not record.parts or record.parts[0] == "..":
                continue
            # A shared __pycache__ holds other modules' bytecode too, so only the distribution's own files in it.
            top = record if record.parts[0] == "__pycache__" else Path(record.parts[0])
            path = Path(distribution.locate_file(top))
            if path.parent.resolve() not in sites and path.parent.parent.resolve() not in sites:
                continue
            if not os.path.lexists(path):
                continue
            path = path.parent.resolve() / path.name
            if path.is_dir() and not path.is_symlink():
                directories.append(path)
            else:
                files.append(path)
    return sorted(set(directories)), sorted(set(files))


@functools.cache
def _runtime() -> _Runtime:
    """The interpreter to run, the launcher path a framework build expects, and what of the runtime it may read."""
    environment_python = Path(sys.executable).absolute()
    sites = {Path(p).resolve() for p in site.getsitepackages([sys.base_prefix, sys.prefix]) if os.path.isdir(p)}
    directories, files = _analysis_entries(sites)
    files += sorted(sites)
    for prefix in dict.fromkeys(Path(p).resolve() for p in (sys.base_prefix, sys.prefix)):
        more_directories, more_files = _covering(prefix, sites)
        directories += more_directories
        files += more_files
    app = Path(sys.base_prefix).resolve() / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"
    launcher = environment_python if app.exists() else None
    return _Runtime(app if app.exists() else environment_python, launcher, tuple(dict.fromkeys(directories)),
                    tuple(dict.fromkeys(files)))


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
    allowed = [*runtime.directories, *runtime.files, layout.root]

    def group(kind: str, paths: list[Path] | set[Path]) -> str:
        return " ".join(f"({kind} {_quoted(p)})" for p in sorted(paths, key=str))

    lines = [
        "(version 1)",
        "(deny default)",
        '(import "system.sb")',
        f"(allow process-exec {group('literal', executables)})",
        f"(allow file-read* file-map-executable {group('subpath', list(runtime.directories))} "
        f"{group('literal', list(runtime.files))})",
        f"(allow file-read* (literal {_quoted(layout.root)}) {group('subpath', [layout.inputs, layout.calls])})",
        f"(allow file-read* file-write* {group('subpath', [layout.work, layout.outputs, layout.tmp])})",
        # After the allow it narrows, because Seatbelt applies the last matching rule: the directory entries
        # themselves cannot be removed, renamed, replaced, or changed, only what is inside them.
        f"(deny file-write* {group('literal', [layout.root / name for name in PROTECTED_DIRECTORIES])})",
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


class _Stream:
    """One output stream of a kernel, read continuously; each call takes what was written up to its marker."""

    def __init__(self, stream: IO[bytes]) -> None:
        self._condition = threading.Condition()
        self._marker: bytes | None = None
        self._found = False
        self._closed = False
        self._head = bytearray()
        self._tail = bytearray()
        self._total = 0
        self._recent = b""
        self._spill = bytearray()  # written outside a call: it opens the next call's output
        self._thread = threading.Thread(target=self._read, args=(stream.fileno(),), daemon=True)
        self._thread.start()

    def _read(self, descriptor: int) -> None:
        try:
            while chunk := os.read(descriptor, 65536):
                with self._condition:
                    self._feed(chunk)
                    self._condition.notify_all()
        except OSError:
            pass
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _append(self, data: bytes) -> None:
        self._total += len(data)
        room = STORED_OUTPUT_BYTES - len(self._head)
        if room > 0:
            self._head += data[:room]
            data = data[room:]
        if data:
            self._tail += data
            del self._tail[:-TAIL_BYTES]

    def _drop_last(self, count: int) -> None:
        self._total -= count
        taken = min(count, len(self._tail))
        if taken:
            del self._tail[len(self._tail) - taken:]
        if count > taken:
            del self._head[len(self._head) - (count - taken):]

    def _feed(self, data: bytes) -> None:
        if self._marker is None or self._found:
            self._spill += data
            del self._spill[:-TAIL_BYTES]
            return
        window = self._recent + data
        index = window.find(self._marker)
        if index < 0:
            self._append(data)
            self._recent = window[-(len(self._marker) - 1):]
            return
        start = index - len(self._recent)  # where the marker starts in ``data``; before it when negative
        if start >= 0:
            self._append(data[:start])
        else:
            self._drop_last(-start)
        self._spill += window[index + len(self._marker):]
        self._found = True

    def begin(self, marker: bytes) -> None:
        """Start a call whose output ends at ``marker``; output written between calls opens it."""
        with self._condition:
            self._marker, self._found = marker, False
            self._head, self._tail, self._total, self._recent = bytearray(), bytearray(), 0, b""
            spill, self._spill = bytes(self._spill), bytearray()
            if spill:
                self._feed(spill)

    def finish(self, timeout: float) -> tuple[bytes, int]:
        """The call's stored bytes and its length, once its marker arrived, the stream closed, or ``timeout``."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while not (self._found or self._closed):
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._condition.wait(left)
            if self._total <= len(self._head):
                stored = bytes(self._head)
            else:
                gap = self._total - len(self._head) - len(self._tail)
                middle = f"\n[... {gap} bytes not stored ...]\n".encode() if gap > 0 else b""
                stored = bytes(self._head) + middle + bytes(self._tail)
            total = self._total
            self._marker = None
            return stored, total

    def join(self, timeout: float) -> None:
        self._thread.join(timeout)


def shorten(stored: bytes, total: int, limit: int) -> tuple[str, bool]:
    """A stream of ``total`` bytes, stored as ``stored``, in at most ``limit`` UTF-8 bytes, and whether it was cut.

    All of it when it fits; otherwise its start and its end around a marker
    that gives the stream's length. Stored bytes of a long stream carry their
    own marker where bytes were not kept.
    """
    if total <= limit and len(stored) == total:
        return stored.decode("utf-8", errors="replace"), False
    marker = f"\n[... {total} bytes in all; the middle is not shown ...]\n"
    budget = max(0, limit - len(marker.encode()))
    first = stored[: budget // 2].decode("utf-8", errors="ignore")
    last_bytes = budget - len(first.encode())
    last = stored[len(stored) - last_bytes:].decode("utf-8", errors="ignore") if last_bytes > 0 else ""
    return first + marker + last, True


class _RusageInfo(ctypes.Structure):
    """``struct rusage_info_v0`` of macOS's ``<sys/resource.h>``, up to the fields WSE reads."""

    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16), ("ri_user_time", ctypes.c_uint64),
                ("ri_system_time", ctypes.c_uint64), ("ri_pkg_idle_wkups", ctypes.c_uint64),
                ("ri_interrupt_wkups", ctypes.c_uint64), ("ri_pageins", ctypes.c_uint64),
                ("ri_wired_size", ctypes.c_uint64), ("ri_resident_size", ctypes.c_uint64),
                ("ri_phys_footprint", ctypes.c_uint64), ("ri_proc_start_abstime", ctypes.c_uint64),
                ("ri_proc_exit_abstime", ctypes.c_uint64)]


@functools.cache
def _proc_pid_rusage() -> Any:
    function = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True).proc_pid_rusage
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    function.restype = ctypes.c_int
    return function


def footprint(pid: int) -> int | None:
    """The physical memory footprint of process ``pid`` in bytes (macOS), or ``None`` when it cannot be read."""
    try:
        info = _RusageInfo()
        if _proc_pid_rusage()(pid, 0, ctypes.byref(info)) != 0:
            return None
        return int(info.ri_phys_footprint)
    except (OSError, AttributeError):
        return None


def open_beneath(directory: int, parts: Sequence[str]) -> tuple[int, os.stat_result]:
    """Open the regular file ``parts`` below the directory ``directory`` for reading; return it and its status.

    Each level is opened relative to the descriptor of the one above it,
    without following a link and without waiting on a FIFO; the file's
    descriptor is returned in blocking mode. ``NotRegularFile`` (an
    ``OSError``) when the entry is not a regular file, ``ValueError`` for a
    part that cannot name an entry below ``directory`` (empty, ``.``, ``..``,
    or holding a NUL byte).
    """
    if not parts or any(part in ("", ".", "..") or "/" in part for part in parts):
        raise ValueError(f"{'/'.join(parts)!r} does not name an entry below the directory")
    current = directory
    try:
        for part in parts[:-1]:
            child = os.open(part, _DIRECTORY_FLAGS, dir_fd=current)
            if current != directory:
                os.close(current)
            current = child
        descriptor = os.open(parts[-1], _FILE_FLAGS, dir_fd=current)
    finally:
        if current != directory:
            os.close(current)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise NotRegularFile(errno.EINVAL, f"{'/'.join(parts)} is {_kind(info.st_mode)}, not a regular file")
        os.set_blocking(descriptor, True)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, info


def read_descriptor(descriptor: int, limit: int) -> tuple[str, int, bytes | None]:
    """The SHA-256 and length of the file open at ``descriptor``, read from its start, and its bytes if at most ``limit``.

    The descriptor is left at the file's start.
    """
    digest = hashlib.sha256()
    kept: bytearray | None = bytearray()
    total = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while block := os.read(descriptor, 1 << 20):
        digest.update(block)
        total += len(block)
        if kept is not None:
            kept += block
            if len(kept) > limit:
                kept = None
    os.lseek(descriptor, 0, os.SEEK_SET)
    return "sha256:" + digest.hexdigest(), total, None if kept is None else bytes(kept)


def _write_copy(target: Path, data: bytes) -> None:
    """Write a kept copy below ``calls/<n>/outputs/``, which only WSE writes; a path too long to copy is skipped."""
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    except OSError as exc:
        if exc.errno != errno.ENAMETOOLONG:
            raise


@dataclass
class _Listing:
    """``outputs/`` as listed by descriptor; ``held`` keeps the descriptors of its first files, sorted by path."""

    outputs: list[OutputFile] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    held: list[tuple[str, int]] = field(default_factory=list)

    def close(self) -> None:
        held, self.held = self.held, []
        for _, descriptor in held:
            os.close(descriptor)


def _hold(listing: _Listing, path: str, descriptor: int, hold: int) -> bool:
    """Keep ``descriptor`` if ``path`` is among the first ``hold`` paths so far; whether it was kept."""
    if hold <= 0:
        return False
    bisect.insort(listing.held, (path, descriptor))
    if len(listing.held) <= hold:
        return True
    dropped_path, dropped = listing.held.pop()
    if dropped_path == path:
        return False
    os.close(dropped)
    return True


def _walk(directory: int, prefix: str, depth: int, listing: _Listing, copies: Path | None, hold: int) -> None:
    with os.scandir(directory) as entries:
        names = sorted(entry.name for entry in entries)
    for name in names:
        relative = prefix + name
        try:
            descriptor = os.open(name, _FILE_FLAGS, dir_fd=directory)
        except FileNotFoundError:
            continue  # removed since the directory was listed
        except OSError:  # a link (never followed), a socket, or anything else that cannot be opened as it is
            listing.refused.append(relative)
            continue
        held = False
        try:
            info = os.fstat(descriptor)
            if stat.S_ISDIR(info.st_mode):
                if depth + 1 < MAX_OUTPUT_DEPTH:
                    _walk(descriptor, relative + "/", depth + 1, listing, copies, hold)
                else:
                    listing.refused.append(relative)
            elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                listing.refused.append(relative)  # a FIFO, a device, or a file with another link
            else:
                os.set_blocking(descriptor, True)
                sha256, size, data = read_descriptor(descriptor, KEPT_OUTPUT_BYTES if copies is not None else -1)
                listing.outputs.append(OutputFile(relative, size, sha256))
                if copies is not None and data is not None:
                    _write_copy(copies / relative, data)
                held = _hold(listing, relative, descriptor, hold)
        finally:
            if not held:
                os.close(descriptor)


def _list_descriptor(outputs: int, *, copies: Path | None = None, hold: int = 0) -> _Listing:
    """List the directory open at ``outputs`` by descriptor (module docstring); keep copies in ``copies``.

    The listing holds the descriptors of its first ``hold`` files; the caller
    closes them (``_Listing.close``).
    """
    listing = _Listing()
    try:
        _walk(outputs, "", 0, listing, copies, hold)
    except BaseException:
        listing.close()
        raise
    listing.outputs.sort(key=lambda output: output.path)
    listing.refused.sort()
    return listing


def list_outputs(layout: SandboxLayout) -> tuple[tuple[OutputFile, ...], tuple[str, ...]]:
    """The regular, singly linked files of ``outputs/`` with their hashes, and every other entry, refused.

    The walk opens every level by descriptor without following a link; an
    ``outputs`` that is not a real directory lists nothing.
    """
    root = _open_root(layout.root)
    try:
        try:
            outputs = os.open("outputs", _DIRECTORY_FLAGS, dir_fd=root)
        except OSError:
            return (), ()
        try:
            listing = _list_descriptor(outputs)
        finally:
            os.close(outputs)
    finally:
        os.close(root)
    return tuple(listing.outputs), tuple(listing.refused)


def read_output(layout: SandboxLayout, output: OutputFile, limit: int) -> bytes | None:
    """The bytes of ``output`` if it is still a regular, singly linked file of at most ``limit`` bytes with its hash.

    The file is opened level by level from the sandbox root's descriptor
    without following a link (``open_beneath``), so a link the sandboxed
    code made is never read through.
    """
    try:
        root = _open_root(layout.root)
    except OSError:
        return None
    try:
        descriptor, info = open_beneath(root, ("outputs", *PurePosixPath(output.path).parts))
    except (OSError, ValueError):
        return None
    finally:
        os.close(root)
    try:
        if info.st_nlink != 1 or info.st_size > limit:
            return None
        sha256, _, data = read_descriptor(descriptor, limit)
    finally:
        os.close(descriptor)
    return data if data is not None and sha256 == output.sha256 else None


def _remove_tree(parent: int, name: str) -> None:
    """Delete the directory ``name`` of ``parent`` and all it holds by descriptor, never following a link (best effort)."""
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(name, dir_fd=parent)  # no longer a directory: the entry itself goes
        return
    try:
        with os.scandir(descriptor) as entries:
            children = [(entry.name, entry.is_dir(follow_symlinks=False)) for entry in entries]
        for child, is_directory in children:
            if is_directory:
                _remove_tree(descriptor, child)
            else:
                with contextlib.suppress(OSError):
                    os.unlink(child, dir_fd=descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
    with contextlib.suppress(OSError):
        os.rmdir(name, dir_fd=parent)


def _remove_entry(parent: int, name: str) -> int | None:
    """Remove the entry ``name`` of ``parent`` without following a link; return its mode (``None``: it was gone).

    A directory goes with everything in it: through ``shutil.rmtree`` relative
    to ``parent`` where it resists symbolic link attacks on this platform,
    and otherwise by descriptor (``_remove_tree``). Deleting is best effort:
    what cannot be deleted stays under its removed name and is retried when
    ``outputs/`` is next emptied.
    """
    try:
        info = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISDIR(info.st_mode):
        with contextlib.suppress(FileNotFoundError):
            os.unlink(name, dir_fd=parent)
    elif shutil.rmtree.avoids_symlink_attacks:
        shutil.rmtree(name, ignore_errors=True, dir_fd=parent)
    else:
        _remove_tree(parent, name)
    return info.st_mode


def _empty_outputs(root: int) -> str | None:
    """Give the sandbox a new, empty ``outputs/`` without following a link; return what it was if not a directory.

    The old directory is renamed aside relative to the root's descriptor,
    where the sandboxed code can neither read nor write it, and then deleted.
    """
    with os.scandir(root) as entries:
        stale = [entry.name for entry in entries if entry.name.startswith(_REMOVED_PREFIX)]
    for name in stale:
        _remove_entry(root, name)
    aside: str | None = _REMOVED_PREFIX + secrets.token_hex(8)
    try:
        os.rename("outputs", aside, src_dir_fd=root, dst_dir_fd=root)
    except FileNotFoundError:
        aside = None
    descriptor, _ = _ensure_directory(root, "outputs")
    os.close(descriptor)
    mode = _remove_entry(root, aside) if aside is not None else None
    return None if mode is None or stat.S_ISDIR(mode) else _kind(mode)


def _keep(layout: SandboxLayout, directory: Path, outputs: tuple[OutputFile, ...]) -> None:
    """Copy each listed output file of at most ``KEPT_OUTPUT_BYTES`` into ``directory/outputs``.

    Each file is read from a descriptor opened without following a link at
    any level (``read_output``) and copied only if it still has its listed
    hash. ``run_python`` keeps its copies while it lists, from the
    descriptor each file was hashed from.
    """
    for output in outputs:
        data = read_output(layout, output, KEPT_OUTPUT_BYTES)
        if data is not None:
            _write_copy(directory / "outputs" / output.path, data)


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


def _death(returncode: int | None) -> tuple[str, int | None, str | None]:
    """The ending of a kernel that died with ``returncode`` (negative: the signal that ended it)."""
    if returncode is None:
        return "crashed", None, None
    if returncode >= 0:
        return "crashed", returncode, None
    number = -returncode
    name = signal.Signals(number).name if number in signal.Signals._value2member_map_ else str(number)
    if number == signal.SIGXCPU:
        return "cpu_limit", None, name
    if number == signal.SIGXFSZ:
        return "file_size_limit", None, name
    return "crashed", None, name


class Kernel:
    """The persistent sandboxed Python process of one sandbox (module docstring); safe to share between threads.

    ``cpu_seconds`` limits the CPU time of each kernel over its life, and
    ``max_memory_bytes`` its physical footprint while a call runs.
    """

    def __init__(self, layout: SandboxLayout, *, cpu_seconds: int, max_memory_bytes: int | None = None) -> None:
        if cpu_seconds <= 0 or (max_memory_bytes is not None and max_memory_bytes <= 0):
            raise WSEError(ErrorCode.INVALID_ARGUMENT, "the sandbox limits must be positive")
        self.layout = layout
        self.cpu_seconds = int(cpu_seconds)
        self.max_memory_bytes = int(max_memory_bytes or DEFAULT_MEMORY_LIMIT)
        self.session = 0
        self.session_call = 0
        self._process: subprocess.Popen[bytes] | None = None
        self._requests: IO[str] | None = None
        self._replies: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._streams: tuple[_Stream, _Stream] | None = None
        self._lock = threading.Lock()

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    def _start(self) -> None:
        layout = self.layout.create()
        runtime = _runtime()
        layout.profile_path.write_text(profile_text(layout), encoding="utf-8")
        script = layout.calls / "kernel.py"
        if not script.exists() or script.read_text(encoding="utf-8") != KERNEL:
            script.write_text(KERNEL, encoding="utf-8")
        request_read, request_write = os.pipe()
        reply_read, reply_write = os.pipe()
        command = [str(SANDBOX_EXEC), "-f", str(layout.profile_path), str(runtime.executable), "-I", "-B",
                   str(script), str(self.cpu_seconds), str(FILE_SIZE_LIMIT), str(request_read), str(reply_write)]
        try:
            process = subprocess.Popen(command, cwd=layout.root, env=_environment(layout, runtime),
                                       stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       pass_fds=(request_read, reply_write), start_new_session=True)
        except BaseException:
            for descriptor in (request_read, request_write, reply_read, reply_write):
                os.close(descriptor)
            raise
        os.close(request_read)
        os.close(reply_write)
        assert process.stdout is not None and process.stderr is not None
        replies: queue.Queue[dict[str, Any] | None] = queue.Queue()

        def read_replies() -> None:
            with os.fdopen(reply_read, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        replies.put(json.loads(line))
                    except ValueError:
                        continue
            replies.put(None)

        threading.Thread(target=read_replies, daemon=True).start()
        self._process, self._replies = process, replies
        self._requests = os.fdopen(request_write, "w", encoding="utf-8")
        self._streams = (_Stream(process.stdout), _Stream(process.stderr))
        self.session += 1
        self.session_call = 0

    def _kill(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - SIGKILL does not fail
            pass

    def _release(self) -> None:
        """Close the pipes of a kernel that has ended."""
        if self._requests is not None:
            try:
                self._requests.close()
            except OSError:
                pass
        if self._streams is not None:
            for stream in self._streams:
                stream.join(1.0)
        if self._process is not None:
            for pipe in (self._process.stdout, self._process.stderr):
                if pipe is not None:
                    pipe.close()
        self._process, self._requests, self._streams = None, None, None

    def close(self) -> None:
        """End the kernel; its variables are lost."""
        with self._lock:
            self._kill()
            self._release()

    def run(self, code: str, call: int, directory: Path, *, wall_seconds: float, max_output_bytes: int,
            restart: bool = False) -> tuple[SandboxResult, bytes, bytes]:
        """Run ``code`` as call ``call`` (its record is ``directory``); return the result and the stored streams."""
        if wall_seconds <= 0 or max_output_bytes <= 0:
            raise WSEError(ErrorCode.INVALID_ARGUMENT, "the sandbox limits must be positive")
        with self._lock:
            restarted = False
            if restart and self._process is not None:
                self._kill()
            if not self.alive:
                restarted = self.session > 0
                self._release()
                self._start()
            assert self._process is not None and self._requests is not None and self._streams is not None
            process, streams = self._process, self._streams
            self.session_call += 1
            code_path = directory / "code.py"
            code_path.write_text(code, encoding="utf-8")
            marker = f"\n<<wse end of call {call} {secrets.token_hex(12)}>>\n"
            for stream in streams:
                stream.begin(marker.encode())
            peak = footprint(process.pid)
            started = time.monotonic()
            reply: dict[str, Any] | None = None
            ending, killed = None, None
            try:
                self._requests.write(json.dumps({"path": str(code_path), "marker": marker}) + "\n")
                self._requests.flush()
            except OSError:
                killed = "crashed"
            interrupted = None
            deadline = started + wall_seconds
            while killed is None:
                now = time.monotonic()
                limit = deadline if interrupted is None else interrupted + INTERRUPT_GRACE_SECONDS
                try:
                    item = self._replies.get(timeout=max(0.0, min(MEMORY_POLL_SECONDS, limit - now)))
                except queue.Empty:
                    item = {}
                if item is None:  # the reply pipe closed: the kernel ended
                    killed = "crashed"
                    break
                if item:
                    if item.get("marker") == marker:
                        reply = item
                        break
                    continue  # a line that answers no request of this call
                used = footprint(process.pid)
                if used is not None:
                    peak = max(peak or 0, used)
                    if used > self.max_memory_bytes:
                        self._kill()
                        killed = "memory_limit"
                        break
                now = time.monotonic()
                if interrupted is None and now >= deadline:
                    try:
                        os.kill(process.pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass
                    interrupted = now
                elif interrupted is not None and now >= interrupted + INTERRUPT_GRACE_SECONDS:
                    self._kill()
                    killed = "timed_out"
            wall = time.monotonic() - started
            exit_code = signal_name = None
            if reply is not None:
                if interrupted is not None and reply.get("status") == "interrupted":
                    ending = "timed_out"
                else:
                    ending = {"completed": "completed", "error": "error", "interrupted": "error"}.get(
                        str(reply.get("status")), "error")
            elif killed == "crashed":
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:  # pragma: no cover - a kernel that closed its reply pipe and lives
                    self._kill()
                ending, exit_code, signal_name = _death(process.returncode)
            else:
                ending, signal_name = killed, "SIGKILL"
            kept = reply is not None and self.alive
            settle = OUTPUT_SETTLE_SECONDS if kept else 0.5
            (stdout, stdout_total), (stderr, stderr_total) = (stream.finish(settle) for stream in streams)
            if not kept:
                self._kill()
                self._release()
            result = SandboxResult(
                call=call, ending=ending, exit_code=exit_code, signal=signal_name,
                error=None if reply is None else reply.get("error"), wall_seconds=round(wall, 3),
                cpu_seconds=None if reply is None else reply.get("cpu_seconds"), peak_memory_bytes=peak,
                stdout="", stderr="", stdout_bytes=stdout_total, stderr_bytes=stderr_total,
                stdout_truncated=False, stderr_truncated=False, outputs=(), refused_outputs=(),
                code_sha256="sha256:" + hashlib.sha256(code.encode("utf-8")).hexdigest(),
                session=self.session, session_call=self.session_call, restarted=restarted, state_kept=kept)
            shown_stdout, stdout_cut = shorten(stdout, stdout_total, max_output_bytes)
            shown_stderr, stderr_cut = shorten(stderr, stderr_total, max_output_bytes)
            from dataclasses import replace

            return replace(result, stdout=shown_stdout, stderr=shown_stderr, stdout_truncated=stdout_cut,
                           stderr_truncated=stderr_cut), stdout, stderr


_KERNELS: dict[Path, Kernel] = {}
_KERNELS_LOCK = threading.Lock()


def kernel_for(layout: SandboxLayout, *, cpu_seconds: int, max_memory_bytes: int | None = None) -> Kernel:
    """The kernel of ``layout`` in this process, created on first use with these limits."""
    with _KERNELS_LOCK:
        kernel = _KERNELS.get(layout.root)
        if kernel is None:
            kernel = _KERNELS[layout.root] = Kernel(layout, cpu_seconds=cpu_seconds,
                                                    max_memory_bytes=max_memory_bytes)
        return kernel


def close_kernels() -> None:
    """End every kernel this process started (at exit, and between tests)."""
    with _KERNELS_LOCK:
        kernels = list(_KERNELS.values())
        _KERNELS.clear()
    for kernel in kernels:
        kernel.close()


atexit.register(close_kernels)


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
        kernel = Kernel(layout, cpu_seconds=60)
        try:
            number, directory = _new_call(layout)
            result, _, _ = kernel.run(code, number, directory, wall_seconds=60.0, max_output_bytes=1024)
        except OSError as exc:
            return False, f"the sandbox self-check could not start: {exc}"
        finally:
            kernel.close()
    if (result.ending, result.stdout.strip()) != ("completed", "allowed"):
        return False, (f"the sandbox self-check failed: ended {result.ending}, output {result.stdout.strip()[:80]!r}, "
                       f"errors {result.stderr.strip()[-200:]!r}")
    return True, "sandbox-exec with an allow-list, checked"


def run_python(layout: SandboxLayout, code: str, *, wall_seconds: float, cpu_seconds: int, max_output_bytes: int,
               max_memory_bytes: int | None = None, restart: bool = False,
               use_output: Callable[[int, OutputFile, int], None] | None = None,
               use_first: int = 0) -> SandboxResult:
    """Run ``code`` as the next call of ``layout``'s kernel (module docstring); raise if the sandbox is unavailable.

    ``restart`` ends the kernel first, so the code runs in a new one.
    ``cpu_seconds`` and ``max_memory_bytes`` apply when this call starts the
    sandbox's first kernel. ``use_output(call, output, descriptor)`` is called
    for each of the first ``use_first`` output files, in path order, with the
    descriptor that file was listed and hashed from (at its start, in
    blocking mode); WSE closes it afterwards.
    """
    ok, reason = available()
    if not ok:
        raise WSEError(ErrorCode.UNSUPPORTED_CAPABILITY, f"code execution is not available here: {reason}",
                       details={"reason": reason})
    root, found = _open_layout(layout.root, "before this call")
    try:
        repairs = _take_repairs(layout.root) + found
        kernel = kernel_for(layout, cpu_seconds=cpu_seconds, max_memory_bytes=max_memory_bytes)
        number, directory = _new_call(layout)
        replaced = _empty_outputs(root)
        if replaced is not None:
            repairs.append(_repair_note("outputs", replaced, "before this call"))
        result, stdout, stderr = kernel.run(code, number, directory, wall_seconds=wall_seconds,
                                            max_output_bytes=max_output_bytes, restart=restart)
        (directory / "stdout.txt").write_bytes(stdout)
        (directory / "stderr.txt").write_bytes(stderr)
        outputs = -1
        try:
            for name in SANDBOX_DIRECTORIES:  # the code (or a command that ran meanwhile) may have replaced one
                descriptor, replaced = _ensure_directory(root, name)
                if replaced is not None:
                    repairs.append(_repair_note(name, replaced, "after this call")
                                   + ("; the call lists no outputs" if name == "outputs" else ""))
                if name == "outputs":
                    outputs = descriptor
                else:
                    os.close(descriptor)
            listing = _list_descriptor(outputs, copies=directory / "outputs",
                                       hold=use_first if use_output is not None else 0)
        finally:
            if outputs >= 0:
                os.close(outputs)
    finally:
        os.close(root)
    try:
        from dataclasses import replace

        result = replace(result, outputs=tuple(listing.outputs), refused_outputs=tuple(listing.refused),
                         repairs=tuple(repairs))
        summary = {key: value for key, value in asdict(result).items() if key not in ("stdout", "stderr")}
        (directory / "result.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8")
        if use_output is not None:
            listed = {output.path: output for output in listing.outputs}
            for path, descriptor in listing.held:
                use_output(number, listed[path], descriptor)
    finally:
        listing.close()
    return result
