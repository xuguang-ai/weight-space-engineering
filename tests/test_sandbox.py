"""The sandbox process: one allow-listed, network-less Python process per call (M3 plan Section 2.1, Task 1).

The programs run inside the sandbox print only whether an operation was
allowed, never the content of a file, so a failure of the isolation cannot
put private content into the test output.
"""

from __future__ import annotations

import hashlib
import socket
from pathlib import Path

import pytest

from wse import sandbox
from wse.sandbox import SandboxLayout, run_python
from wse.schemas import ErrorCode, WSEError

REPO_FILE = Path(__file__).resolve().parents[1] / "pyproject.toml"
AVAILABLE, REASON = sandbox.available()
needs_sandbox = pytest.mark.skipif(not AVAILABLE, reason=REASON)
LIMITS = {"wall_seconds": 120.0, "cpu_seconds": 120, "max_output_bytes": 16384}


@pytest.fixture
def layout(tmp_path: Path) -> SandboxLayout:
    workspace = tmp_path / "workspace"
    layout = SandboxLayout(workspace / "sandboxes" / "run_test")
    layout.create()
    (workspace / "metadata.sqlite").write_text("final examples")
    (tmp_path / "planted.txt").write_text("planted")
    (layout.inputs / "a.txt").write_text("input")
    return layout


def _status_lines(targets: dict[str, str], operation: str) -> str:
    """Code that tries ``operation`` on each target and prints only ``name: allowed`` or ``name: denied``."""
    lines = ["import os"]
    for name, path in targets.items():
        lines += ["try:", f"    {operation.format(path=repr(path))}", f"    print({name!r} + ': allowed')",
                  "except OSError:", f"    print({name!r} + ': denied')"]
    return "\n".join(lines) + "\n"


@needs_sandbox
def test_sandbox_runs_numpy_and_torch_on_the_cpu(layout: SandboxLayout) -> None:
    code = "import numpy, torch\nprint(float(torch.ones(8, 8).matmul(torch.ones(8, 8)).sum()), bool(numpy.__version__))\n"
    result = run_python(layout, code, **LIMITS)
    assert (result.ending, result.exit_code) == ("exited", 0), result.stderr
    assert result.stdout.startswith("512.0 True")


@needs_sandbox
def test_sandbox_reads_its_inputs_and_writes_only_its_work_and_outputs(layout: SandboxLayout) -> None:
    code = (
        "import numpy as np\n"
        "print(open('inputs/a.txt').read())\n"
        "np.save('outputs/v.npy', np.arange(4.0))\n"
        "open('work/state.txt', 'w').write('s')\n"
        + _status_lines({"write inputs": "inputs/x.txt", "write root": "x.txt"}, "open({path}, 'w').write('x')")
    )
    result = run_python(layout, code, **LIMITS)
    assert (result.ending, result.exit_code) == ("exited", 0), result.stderr
    assert result.stdout.splitlines() == ["input", "write inputs: denied", "write root: denied"]
    assert (layout.work / "state.txt").read_text() == "s"
    assert [(f.path, f.size) for f in result.outputs] == [("v.npy", (layout.outputs / "v.npy").stat().st_size)]
    assert result.outputs[0].sha256 == "sha256:" + hashlib.sha256((layout.outputs / "v.npy").read_bytes()).hexdigest()


@needs_sandbox
def test_sandbox_cannot_read_the_workspace_the_repository_or_a_planted_file(layout: SandboxLayout,
                                                                            tmp_path: Path) -> None:
    workspace = layout.root.parent.parent
    targets = {"workspace file": str(workspace / "metadata.sqlite"), "planted file": str(tmp_path / "planted.txt"),
               "repository file": str(REPO_FILE), "profile": str(layout.profile_path)}
    code = (_status_lines(targets, "open({path}).read(1)")
            + _status_lines({"workspace listing": str(workspace), "home listing": str(Path.home())},
                            "os.listdir({path})"))
    result = run_python(layout, code, **LIMITS)
    assert (result.ending, result.exit_code) == ("exited", 0), result.stderr
    assert result.stdout.splitlines() == [f"{name}: denied" for name in
                                          [*targets, "workspace listing", "home listing"]]


@needs_sandbox
def test_sandbox_cannot_open_a_network_connection(layout: SandboxLayout) -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        code = ("import socket\n"
                "try:\n"
                f"    socket.create_connection(('127.0.0.1', {port}), timeout=5).close()\n"
                "    print('connected')\n"
                "except OSError:\n"
                "    print('denied')\n")
        result = run_python(layout, code, **LIMITS)
    assert result.stdout.strip() == "denied", result.stderr


@needs_sandbox
def test_sandbox_environment_carries_no_credentials(layout: SandboxLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    result = run_python(layout, "import os\nprint(' '.join(sorted(os.environ)))\n", **LIMITS)
    names = set(result.stdout.split())
    assert not {"OPENAI_API_KEY", "OPENAI_BASE_URL"} & names
    assert names <= set(sandbox.ENVIRONMENT_NAMES) | {"__CF_USER_TEXT_ENCODING"}


@needs_sandbox
def test_wall_clock_limit_kills_the_process_and_records_it(layout: SandboxLayout) -> None:
    result = run_python(layout, "import time\ntime.sleep(60)\n", wall_seconds=2.0, cpu_seconds=60,
                        max_output_bytes=1024)
    assert result.ending == "timed_out" and result.exit_code is None
    assert 2.0 <= result.wall_seconds < 15.0


@needs_sandbox
def test_cpu_limit_stops_a_busy_loop(layout: SandboxLayout) -> None:
    result = run_python(layout, "while True:\n    pass\n", wall_seconds=60.0, cpu_seconds=1, max_output_bytes=1024)
    assert result.ending == "cpu_limit"
    assert result.wall_seconds < 30.0


@needs_sandbox
def test_output_is_returned_within_its_limit_and_stored_in_full(layout: SandboxLayout) -> None:
    code = "import sys\nprint('x' * 100_000)\nprint('e' * 50_000, file=sys.stderr)\n"
    result = run_python(layout, code, wall_seconds=60.0, cpu_seconds=60, max_output_bytes=1000)
    assert result.stdout_truncated and result.stderr_truncated
    assert len(result.stdout.encode()) <= 1000 and len(result.stderr.encode()) <= 1000
    assert (result.stdout_bytes, result.stderr_bytes) == (100_001, 50_001)
    call = layout.calls / str(result.call)
    assert (call / "stdout.txt").stat().st_size == 100_001
    assert (call / "stderr.txt").stat().st_size == 50_001


@needs_sandbox
def test_links_in_outputs_are_refused_not_followed(layout: SandboxLayout, tmp_path: Path) -> None:
    code = (f"import os\nos.symlink({str(tmp_path / 'planted.txt')!r}, 'outputs/link.txt')\n"
            "open('outputs/plain.txt', 'w').write('p')\n")
    result = run_python(layout, code, **LIMITS)
    assert (result.ending, result.exit_code) == ("exited", 0), result.stderr
    assert [f.path for f in result.outputs] == ["plain.txt"]
    assert result.refused_outputs == ("link.txt",)


@needs_sandbox
def test_the_sandbox_cannot_start_another_process(layout: SandboxLayout) -> None:
    code = ("import subprocess\n"
            "try:\n"
            "    subprocess.run(['/bin/echo', 'hi'], check=True)\n"
            "    print('started')\n"
            "except OSError:\n"
            "    print('denied')\n")
    result = run_python(layout, code, **LIMITS)
    assert result.stdout.strip() == "denied", result.stderr


@needs_sandbox
def test_each_call_is_a_fresh_process_and_files_persist(layout: SandboxLayout) -> None:
    first = run_python(layout, "x = 1\nopen('work/s.txt', 'w').write('kept')\n", **LIMITS)
    second = run_python(layout, "print(open('work/s.txt').read(), 'x' in globals())\n", **LIMITS)
    assert (first.call, second.call) == (1, 2)
    assert second.stdout.strip() == "kept False"
    assert (layout.calls / "1" / "code.py").read_text() == "x = 1\nopen('work/s.txt', 'w').write('kept')\n"
    assert first.code_sha256 == "sha256:" + hashlib.sha256((layout.calls / "1" / "code.py").read_bytes()).hexdigest()


def test_the_sandbox_fails_closed_without_sandbox_exec(layout: SandboxLayout, tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "SANDBOX_EXEC", tmp_path / "missing-sandbox-exec")
    sandbox.available.cache_clear()
    try:
        ok, reason = sandbox.available()
        assert not ok and "sandbox-exec" in reason
        with pytest.raises(WSEError) as caught:
            run_python(layout, "print('never')\n", **LIMITS)
        assert caught.value.code == ErrorCode.UNSUPPORTED_CAPABILITY
    finally:
        sandbox.available.cache_clear()
