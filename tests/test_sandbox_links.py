"""The host never resolves the sandbox directory by name or follows a link that agent code made (``wse.sandbox``).

WSE opens a run's sandbox directory, and its ``inputs/``, ``work/``,
``outputs/``, ``tmp/``, and ``calls/`` within it, by descriptor without
following a link; it replaces an entry that is not a real directory (the
entry itself, never what it points to) and says so in the call's result. It
lists ``outputs/`` from descriptors only, copies each kept file from the
descriptor it hashed, and empties ``outputs/`` by renaming it aside. The
kernel's profile denies removing, renaming, or replacing those directories
themselves. Programs run in the sandbox print only whether an operation was
allowed, never the content of a file.
"""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path, PurePosixPath

import pytest

from wse import sandbox
from wse.sandbox import Kernel, SandboxLayout, _new_call, run_python
from wse.sandbox_runs import _open_beneath

AVAILABLE, REASON = sandbox.available()
needs_sandbox = pytest.mark.skipif(not AVAILABLE, reason=REASON)
LIMITS = {"wall_seconds": 120.0, "cpu_seconds": 120, "max_output_bytes": 16384}
SECRET = "evaluator-dataset-stand-in"


@pytest.fixture
def layout(tmp_path: Path) -> SandboxLayout:
    return SandboxLayout(tmp_path / "workspace" / "sandboxes" / "run_links").create()


@pytest.fixture
def victim(layout: SandboxLayout) -> Path:
    """A directory of the workspace outside the sandbox, standing for the evaluator's datasets."""
    directory = layout.root.parent.parent / "evaluator"
    (directory / "datasets").mkdir(parents=True)
    (directory / "datasets" / "final.json").write_text(SECRET)
    return directory


def _intact(victim: Path) -> bool:
    final = victim / "datasets" / "final.json"
    return final.is_file() and final.read_text() == SECRET


def _without_directory_denial(real):
    """``profile_text`` without its rule that protects the sandbox's directories: the control of the denial tests."""
    def profile(layout: SandboxLayout) -> str:
        lines = real(layout).splitlines()
        return "\n".join(line for line in lines if not line.startswith("(deny file-write*")) + "\n"
    return profile


DIRECTORY_OPERATIONS = """\
import os
def attempt(name, action):
    try:
        action()
        print(name, 'allowed')
    except OSError:
        print(name, 'denied')
attempt('write in outputs', lambda: open('outputs/x.txt', 'w').write('x'))
attempt('rmdir tmp', lambda: os.rmdir('tmp'))
if not os.path.isdir('tmp'):
    os.mkdir('tmp')
attempt('rename outputs', lambda: os.rename('outputs', 'work/moved'))
if os.path.isdir('work/moved'):
    os.rename('work/moved', 'outputs')
os.mkdir('work/empty')
attempt('replace tmp', lambda: os.rename('work/empty', 'tmp'))
"""
SYMLINK_AT_TMP = ("import os\ntry:\n    os.symlink('work', 'tmp')\n    print('symlink at tmp allowed')\n"
                  "except OSError:\n    print('symlink at tmp denied')\n")


@needs_sandbox
@pytest.mark.parametrize("control", [False, True], ids=["profile", "control-without-the-rule"])
def test_the_kernel_cannot_remove_rename_or_replace_its_directories(layout: SandboxLayout,
                                                                     monkeypatch: pytest.MonkeyPatch,
                                                                     control: bool) -> None:
    # The control runs the same code under the profile without the denial, so it proves each operation would
    # otherwise succeed (Seatbelt applies the last matching rule: the denial must follow the allow it narrows).
    if control:
        monkeypatch.setattr(sandbox, "profile_text", _without_directory_denial(sandbox.profile_text))
    kernel = Kernel(layout, cpu_seconds=60)
    try:
        number, directory = _new_call(layout)
        first, _, _ = kernel.run(DIRECTORY_OPERATIONS, number, directory, wall_seconds=60.0, max_output_bytes=4096)
        if layout.tmp.is_dir() and not layout.tmp.is_symlink():
            shutil.rmtree(layout.tmp)  # the host takes tmp/ away while the kernel lives, so creating it can be tried
        number, directory = _new_call(layout)
        second, _, _ = kernel.run(SYMLINK_AT_TMP, number, directory, wall_seconds=60.0, max_output_bytes=4096)
    finally:
        kernel.close()
    assert (first.ending, second.ending) == ("completed", "completed"), first.stderr + second.stderr
    outcome = "allowed" if control else "denied"
    assert (first.stdout + second.stdout).splitlines() == [
        "write in outputs allowed", f"rmdir tmp {outcome}", f"rename outputs {outcome}", f"replace tmp {outcome}",
        f"symlink at tmp {outcome}"]


@needs_sandbox
def test_a_link_made_at_outputs_between_calls_is_removed_not_followed(layout: SandboxLayout, victim: Path) -> None:
    run_python(layout, "pass\n", **LIMITS)
    layout.outputs.rmdir()
    os.symlink("../../evaluator", layout.outputs)  # what the agent's own commands could do between two calls
    result = run_python(layout, "import os\nprint(sorted(os.listdir('outputs')))\n"
                                "open('outputs/new.txt', 'w').write('n')\n", **LIMITS)
    assert _intact(victim), "emptying outputs/ followed the link into the workspace"
    assert result.ending == "completed", result.stderr
    assert layout.outputs.is_dir() and not layout.outputs.is_symlink()
    assert result.stdout.strip() == "[]" and [o.path for o in result.outputs] == ["new.txt"]
    [note] = result.repairs
    assert note.startswith("outputs/ was a symbolic link") and "before this call" in note


@needs_sandbox
def test_a_link_a_call_leaves_at_outputs_is_neither_listed_copied_nor_emptied(layout: SandboxLayout, victim: Path,
                                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    # The kernel's profile now forbids replacing outputs/, so a profile without that rule stands for code that
    # still can (the agent's own commands, say): the host's own handling must hold without the kernel's rule.
    monkeypatch.setattr(sandbox, "profile_text", _without_directory_denial(sandbox.profile_text))
    first = run_python(layout, "import os\nos.rmdir('outputs')\nos.symlink('../../evaluator', 'outputs')\n", **LIMITS)
    kept = layout.calls / str(first.call) / "outputs"
    assert first.outputs == () and not (kept.exists() and any(kept.rglob("*"))), "the host listed through the link"
    second = run_python(layout, "import glob\nprint(len(glob.glob('calls/*/outputs/**/*.json', recursive=True)))\n",
                        **LIMITS)
    assert _intact(victim), "the host's emptying of outputs/ deleted what the link pointed to"
    assert first.ending == "completed" and second.stdout.strip() == "0", first.stderr + second.stderr
    [note] = first.repairs
    assert note.startswith("outputs/ was a symbolic link") and "after this call" in note
    assert layout.outputs.is_dir() and not layout.outputs.is_symlink() and not second.repairs


@needs_sandbox
def test_a_link_to_a_directory_in_outputs_is_refused_and_nothing_below_it_is_listed_or_copied(
        layout: SandboxLayout, victim: Path) -> None:
    code = (f"import os\nos.symlink({str(victim / 'datasets')!r}, 'outputs/d')\n"
            "os.makedirs('outputs/real/deeper')\nopen('outputs/real/deeper/kept.txt', 'w').write('k')\n")
    result = run_python(layout, code, **LIMITS)
    assert result.ending == "completed", result.stderr
    assert [o.path for o in result.outputs] == ["real/deeper/kept.txt"] and result.refused_outputs == ("d",)
    kept = layout.calls / str(result.call) / "outputs"
    assert sorted(p.relative_to(kept).as_posix() for p in kept.rglob("*") if p.is_file()) == ["real/deeper/kept.txt"]
    assert _intact(victim)


class _Listed:
    """What ``os.scandir`` returned, read in full: an iterator and a context manager like the real one."""

    def __init__(self, entries: list[os.DirEntry]) -> None:
        self._entries = iter(entries)

    def __iter__(self) -> _Listed:
        return self

    def __next__(self) -> os.DirEntry:
        return next(self._entries)

    def __enter__(self) -> _Listed:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def close(self) -> None:
        pass


def test_a_subdirectory_swapped_for_a_link_between_listing_and_opening_is_not_followed(
        layout: SandboxLayout, victim: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # outputs/d is a real directory when outputs/ is listed and a link to the victim when the host first opens or
    # lists d: the moment the reviewer's timed swap hit, made deterministic.
    (layout.outputs / "d").mkdir()
    (layout.outputs / "d" / "placeholder.txt").write_text("p")
    real_open, real_scandir = os.open, os.scandir
    swapped: list[str] = []

    def swap(how: str) -> None:
        if not swapped:
            swapped.append(how)
            shutil.rmtree(layout.outputs / "d")
            os.symlink(victim / "datasets", layout.outputs / "d")

    def opened(path, flags, mode=0o777, *, dir_fd=None):
        if (path == "d" and dir_fd is not None) or (not isinstance(path, int) and Path(path) == layout.outputs / "d"):
            swap("open")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def scanned(path="."):
        if not isinstance(path, int) and Path(path) == layout.outputs / "d":
            swap("scandir")
        with real_scandir(path) as entries:
            return _Listed(list(entries))

    monkeypatch.setattr(os, "open", opened)
    monkeypatch.setattr(os, "scandir", scanned)
    outputs, refused = sandbox.list_outputs(layout)
    directory = layout.calls / "1"
    directory.mkdir()
    sandbox._keep(layout, directory, outputs)
    monkeypatch.undo()
    assert swapped, "the hook never ran, so the test proves nothing"
    assert "d/final.json" not in [o.path for o in outputs], "the walk followed the swapped link"
    assert not (directory / "outputs" / "d").exists()
    assert refused == ("d",) and _intact(victim)


@needs_sandbox
def test_emptying_outputs_without_a_safe_rmtree_follows_no_link(layout: SandboxLayout, victim: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil.rmtree, "avoids_symlink_attacks", False)
    code = (f"import os\nos.makedirs('outputs/a/b')\nos.symlink({str(victim)!r}, 'outputs/a/b/link')\n"
            f"os.symlink({str(victim / 'datasets' / 'final.json')!r}, 'outputs/f')\n"
            "open('outputs/a/b/x.txt', 'w').write('x')\n")
    first = run_python(layout, code, **LIMITS)
    second = run_python(layout, "import os\nprint(sorted(os.listdir('outputs')))\n", **LIMITS)
    assert (first.ending, second.stdout.strip()) == ("completed", "[]"), first.stderr + second.stderr
    assert _intact(victim)
    assert not [name for name in os.listdir(layout.root) if name.startswith(".wse-")]


def test_a_fifo_in_outputs_is_refused_without_blocking_the_host(layout: SandboxLayout) -> None:
    os.mkfifo(layout.outputs / "pipe.safetensors")
    (layout.outputs / "plain.txt").write_text("p")
    found: dict[str, object] = {}
    thread = threading.Thread(target=lambda: found.update(listing=sandbox.list_outputs(layout)), daemon=True)
    thread.start()
    thread.join(10)
    assert not thread.is_alive(), "listing outputs/ blocked on a FIFO"
    outputs, refused = found["listing"]
    assert [o.path for o in outputs] == ["plain.txt"] and refused == ("pipe.safetensors",)


def test_opening_a_fifo_the_agent_named_does_not_block_and_is_refused(layout: SandboxLayout) -> None:
    os.mkfifo(layout.work / "cand.safetensors")
    found: dict[str, object] = {}

    def attempt() -> None:
        try:
            found["descriptor"] = _open_beneath(layout.root, PurePosixPath("work/cand.safetensors"))
        except OSError as exc:
            found["error"] = exc

    thread = threading.Thread(target=attempt, daemon=True)
    thread.start()
    thread.join(10)
    blocked = thread.is_alive()
    if blocked:  # release the blocked open so the test can end
        with open(layout.work / "cand.safetensors", "wb"):
            pass
        thread.join(10)
    if "descriptor" in found:
        os.close(found["descriptor"])
    assert not blocked, "the host's open of a FIFO blocked"
    assert isinstance(found.get("error"), OSError) and "not a regular file" in str(found["error"])


def test_a_layout_replaces_a_linked_directory_without_touching_its_target(layout: SandboxLayout,
                                                                         victim: Path) -> None:
    layout.work.rmdir()
    os.symlink(victim, layout.work)
    layout.tmp.rmdir()
    (layout.root / "tmp").write_text("a file where tmp/ belongs")
    layout.create()
    assert layout.work.is_dir() and not layout.work.is_symlink() and layout.tmp.is_dir()
    assert not any(layout.work.iterdir()) and _intact(victim)
