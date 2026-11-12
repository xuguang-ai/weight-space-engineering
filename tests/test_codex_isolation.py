"""The real Codex binary under the adapter's configuration: isolation, environments, events, and long tool calls.

These tests run only on macOS with a Codex CLI that ``find_codex`` verifies
(OpenAI's code signature). The model is a local scripted endpoint
(``codex_fake_model.FakeResponses``): Codex, its sandbox, its MCP client,
``wse serve``, and the adapter are real, and nothing leaves the machine.
They are toolchain checks, not agent runs.

* ``codex sandbox -P wse`` with the configuration the adapter writes shows
  what the agent's commands can read and write, and that they have no
  network.
* A full run through the reference runner shows, from a command that Codex
  itself runs, that the same paths are unreadable and the analysis
  libraries are not; the environment that the agent's commands and the WSE
  MCP server receive (no credential, no proxy token); how Codex's events are
  normalized; and that no process outlives the run.
* A blocking ``wse_get_job`` wait completes as one tool call under the
  adapter's tool timeout and ends as ``TOOL_TIMEOUT`` under a shorter one.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import site
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from wse.agent_adapters.base import AgentConfig
from wse.agent_adapters.codex_cli import (
    CODEX_TEAM_IDENTIFIER,
    PROXY_TOKEN_ENV,
    CodexCLIAdapter,
    PythonAccess,
    find_codex,
    shell_environment,
)
from wse.api import Engine
from wse.policy import SMALL_MODEL_POLICY_ID, ExperimentPolicy
from wse.runner import ReferenceRunner, allowed_tools
from wse.schemas import WSEError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codex_fake_model import FakeResponses, message, shell, tool_outputs, wse_call  # noqa: E402

TESTS_DIR = Path(__file__).resolve().parent
REPO = TESTS_DIR.parent
FAST_WORKER = ["--poll-seconds", "0.02"]
FINAL_SEED = 5_550_199_223_310_017
TOKEN = "wse-proxy-token-for-tests-1c9e44d07a3b5f26"
MODEL = "gpt-5.6-luna"


def _binary():
    if sys.platform != "darwin":
        return None
    try:
        return find_codex()
    except WSEError:
        return None


BINARY = _binary()
pytestmark = pytest.mark.skipif(BINARY is None, reason="needs macOS and a Codex CLI signed by OpenAI")


@pytest.fixture
def plugins():
    import wse_test_plugins

    return wse_test_plugins


@pytest.fixture
def prepared(engine, signed_sum, tmp_path, plugins):
    plugins.setup(engine.registry)
    ids = signed_sum.setup_workspace(engine, tmp_path / "bundles", final_seed=FINAL_SEED, with_reference=True)
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / ".env").write_text("OPENAI_API_KEY=planted-not-a-real-key\n")
    return {**ids, "planted_env": tmp_path / "repo" / ".env"}


def _home_file() -> Path | None:
    """A file in the user's home directory to try to open (never read)."""
    for name in (".zshrc", ".bash_profile", ".profile", ".gitconfig", ".zprofile"):
        if (Path.home() / name).is_file():
            return Path.home() / name
    return None


def forbidden(workspace: Path, codex_root: Path, planted_env: Path) -> dict[str, tuple[str, Path]]:
    """What the agent's commands must not reach: label -> (how to try: cat, ls, or open, path).

    The repository checkout (its README and data), WSE's installed metadata,
    the workspace's database and evaluator storage (final suites among
    them), keys, Codex homes, the model proxy's files, and the user's home.
    """
    probes: dict[str, tuple[str, Path]] = {
        "repository README": ("cat", REPO / "README.md"),
        "repository source": ("cat", REPO / "src" / "wse" / "api.py"),
        "repository checkout": ("ls", REPO),
        "workspace database": ("cat", workspace / "metadata.sqlite"),
        "workspace": ("ls", workspace),
        "evaluator storage": ("ls", workspace / "evaluator"),
        "planted .env": ("cat", planted_env),
        "this run's Codex config": ("cat", codex_root / "home" / "config.toml"),
        "this run's Codex sessions": ("ls", codex_root / "home"),
    }
    if (REPO / "examples" / "data" / "math_test.jsonl").exists():
        probes["repository final test problems"] = ("cat", REPO / "examples" / "data" / "math_test.jsonl")
    for index, path in enumerate(sorted((workspace / "evaluator" / "datasets").glob("*.json"))):
        probes[f"evaluator dataset {index}"] = ("cat", path)
    for directory in {Path(p).resolve() for p in site.getsitepackages([sys.base_prefix, sys.prefix])}:
        probes[f"site-packages {directory.parent.parent.parent.name}"] = ("ls", directory)
        for metadata in sorted(directory.glob("wse-*.dist-info/METADATA")):
            probes[f"installed wse metadata {metadata.parent.name}"] = ("cat", metadata)
    for path in sorted(codex_root.glob("model-proxy*")):
        probes[f"model proxy file {path.name}"] = ("cat", path)
    for label, path in (("~/.codex", Path.home() / ".codex"), ("~/.cache/huggingface",
                                                                Path.home() / ".cache" / "huggingface")):
        if path.exists():
            probes[label] = ("ls", path)
    for label, path in (("~/.codex/auth.json", Path.home() / ".codex" / "auth.json"), ("a file in ~/", _home_file())):
        if path is not None and path.exists():
            probes[label] = ("open", path)  # opened, never read
    return probes


def probe_script(probes: dict[str, tuple[str, Path]]) -> str:
    """A shell script printing ``PROBE <n> OK`` or ``PROBE <n> DENIED`` per probe, in order."""
    lines = []
    for index, (how, path) in enumerate(probes.values()):
        quoted = shlex.quote(str(path))
        attempt = {"cat": f"cat {quoted} > /dev/null", "ls": f"ls {quoted} > /dev/null",
                   "open": f"( exec 3< {quoted} )"}[how]
        lines.append(f"{attempt} 2>/dev/null && echo 'PROBE {index} OK' || echo 'PROBE {index} DENIED'")
    return "\n".join(lines)


def probe_results(probes: dict[str, tuple[str, Path]], output: str) -> dict[str, str]:
    found = dict(line.split()[1:3] for line in output.splitlines() if line.startswith("PROBE "))
    return {label: found.get(str(index), "missing") for index, label in enumerate(probes)}


def test_the_codex_binary_is_verified_before_use():
    assert BINARY.team_identifier == CODEX_TEAM_IDENTIFIER and BINARY.version.startswith("codex-cli ")
    assert BINARY.source in ("PATH", "vscode-extension", "WSE_CODEX_PATH")


# -- what the agent's commands can reach (codex sandbox) --------------------------------------------------------


@pytest.fixture
def configured(engine, workspace, prepared):
    """A run grant, its experiment and connection, and the Codex home the adapter writes for it (not launched)."""
    run_id = engine.grant_run(prepared["task_id"], SMALL_MODEL_POLICY_ID, prepared["task_checkpoint_id"]).run_id
    with Engine(workspace, run_id=run_id) as bound:
        eid = bound.create_experiment(run_id, prepared["task_id"], SMALL_MODEL_POLICY_ID, "isolation").experiment_id
        policy, task = bound.experiment_policy(eid), bound.get_task(prepared["task_id"])
    runner = ReferenceRunner(workspace, setup=["wse_test_plugins:setup"], serve_env={"PYTHONPATH": str(TESTS_DIR)})
    connection = runner.connection(run_id, eid, task, policy, "unverified")
    adapter = CodexCLIAdapter(codex_path=BINARY.path, provider_config={}, proxy_token=TOKEN)
    asyncio.run(adapter.start(connection, AgentConfig(adapter="codex_cli@1", model=MODEL)))
    session = adapter._sessions[eid]
    assert session.connect_error is None, session.connect_error
    sandbox = session.paths.sandbox
    (sandbox / "inputs" / "weights").mkdir(parents=True, exist_ok=True)
    import torch
    from safetensors.torch import save_file

    save_file({"layers.0.weight": torch.ones(2, 3)}, str(sandbox / "inputs" / "weights" / "base.safetensors"))
    yield {"workspace": workspace, "run_id": run_id, "session": session, "sandbox": sandbox,
           "planted_env": prepared["planted_env"]}
    asyncio.run(adapter.close(eid))  # stops the evaluation worker that start() began


def _sandboxed(configured: dict[str, Any], command: str) -> subprocess.CompletedProcess[str]:
    """``command`` under ``codex sandbox -P wse`` in the run's sandbox, with the commands' own environment."""
    session = configured["session"]
    python = session.settings["python"]
    access = PythonAccess(bin_dir=python["bin_dir"], pythonpath=python["pythonpath"], readable=(), denied=(),
                          linked=())
    env = {**shell_environment(configured["sandbox"], access), "CODEX_HOME": str(session.paths.home)}
    return subprocess.run([BINARY.path, "sandbox", "-P", "wse", "-C", str(configured["sandbox"]), "--", "/bin/sh",
                           "-c", command], capture_output=True, text=True, env=env, cwd=configured["sandbox"],
                          timeout=300, stdin=subprocess.DEVNULL)


def _allowed(configured: dict[str, Any], command: str) -> bool:
    result = _sandboxed(configured, f"{command} && echo PROBE-OK || echo PROBE-DENIED")
    assert "PROBE-OK" in result.stdout or "PROBE-DENIED" in result.stdout, (result.stdout, result.stderr)
    return "PROBE-OK" in result.stdout


def test_the_native_profile_lets_commands_read_inputs_and_write_only_their_directories(configured):
    session = configured["session"]
    load = ("python -c \"from safetensors.torch import load_file; "
            "print(sorted(load_file('inputs/weights/base.safetensors')))\"")
    result = _sandboxed(configured, load)
    assert result.returncode == 0 and "layers.0.weight" in result.stdout, result.stderr
    libraries = _sandboxed(configured, "python -c 'import numpy, torch, safetensors.torch; "
                                       "print(\"libraries\", numpy.__version__, float(torch.ones(4).sum()))'")
    assert libraries.returncode == 0 and "libraries" in libraries.stdout and "4.0" in libraries.stdout
    numpy_init = Path(__import__("numpy").__file__).resolve()
    assert _allowed(configured, f"cat {numpy_init} > /dev/null")
    assert _allowed(configured, f"cat {session.paths.skill / 'SKILL.md'} > /dev/null")
    for name in ("work", "outputs", "tmp"):
        assert _allowed(configured, f"echo x > {name}/probe.txt"), name
    assert not _allowed(configured, "echo x > inputs/probe.txt")
    assert not _allowed(configured, "echo x > probe.txt")  # the sandbox directory itself
    assert not _allowed(configured, f"echo x > {configured['workspace']}/probe.txt")
    assert not _allowed(configured, "echo x > /tmp/wse-codex-probe.txt")


def test_the_native_profile_keeps_wse_storage_keys_and_the_network_away_from_commands(configured):
    workspace, session = configured["workspace"], configured["session"]
    probes = forbidden(workspace, session.paths.root, configured["planted_env"])
    assert any(label.startswith("evaluator dataset") for label in probes)  # the suites' data, final ones included
    assert all(path.exists() for _, path in probes.values())
    script = "cat inputs/weights/base.safetensors > /dev/null && echo CONTROL-OK\n" + probe_script(probes)
    result = _sandboxed(configured, script)
    assert "CONTROL-OK" in result.stdout, result.stderr
    assert probe_results(probes, result.stdout) == {label: "DENIED" for label in probes}
    # The environment's other packages, WSE's own distribution among them, cannot be imported.
    assert not _allowed(configured, "python -c 'import wse'")
    assert not _allowed(configured, "python -c 'import importlib.metadata as m; m.version(\"wse\")'")
    # No network: not the internet, and not a port listening on this machine.
    assert not _allowed(configured, "curl -s -m 5 -o /dev/null https://example.com")
    assert not _allowed(configured, "python -c 'import socket; socket.create_connection((\"1.1.1.1\", 53), 5)'")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert not _allowed(configured, "python -c 'import socket; "
                                        f"socket.create_connection((\"127.0.0.1\", {port}), 5)'")


# -- a full run with the real binary ----------------------------------------------------------------------------


def _segment_items(result: dict[str, Any], item_type: str) -> list[dict[str, Any]]:
    items = []
    for segment in result["provider_metadata"]["segments"]:
        for line in Path(segment["log"]).read_text().splitlines():
            event = json.loads(line)
            if event.get("type") == "item.completed" and event["item"].get("type") == item_type:
                items.append(event["item"])
    return items


def _processes_naming(text: str) -> list[str]:
    table = subprocess.run(["/bin/ps", "-A", "-o", "pid=,command="], capture_output=True, text=True).stdout
    return [line for line in table.splitlines() if text in line and "/bin/ps" not in line]


def _events(engine, eid) -> list[dict[str, Any]]:
    rows = engine.store.db.conn.execute("SELECT kind, payload FROM controller_events WHERE experiment_id = ? "
                                        "ORDER BY seq", (eid,)).fetchall()
    return [{"kind": row["kind"], **json.loads(row["payload"])} for row in rows]


def test_a_real_codex_run_isolates_its_commands_and_keeps_credentials_out(engine, workspace, prepared, tmp_path,
                                                                           monkeypatch):
    # Upstream credentials in the operator's environment must reach neither Codex's commands nor the WSE server.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-planted-operator-key-0000")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://planted.invalid/openai/v1")
    record, token_file = tmp_path / "mcp-env.json", tmp_path / "token.txt"
    token_file.write_text(TOKEN)
    original = ReferenceRunner.connection

    def probed(self, *args, **kwargs):
        connection = original(self, *args, **kwargs)
        command = [sys.executable, str(TESTS_DIR / "codex_env_probe.py"), str(record), str(token_file), "--",
                   *connection.serve_command]
        return connection.model_copy(update={"serve_command": command})

    monkeypatch.setattr(ReferenceRunner, "connection", probed)
    run_id = engine.grant_run(prepared["task_id"], SMALL_MODEL_POLICY_ID, prepared["task_checkpoint_id"]).run_id
    codex_root = workspace / "logs" / f"codex-{run_id}"
    probes = forbidden(workspace, codex_root, prepared["planted_env"])
    numpy_init = Path(__import__("numpy").__file__).resolve()
    command = "\n".join([
        "env; echo ENV-END",
        f"head -3 {codex_root / 'home' / 'skills' / 'wse' / 'SKILL.md'}",
        "python -c 'import numpy, torch, safetensors.torch; print(\"numpy\", numpy.__version__)'",
        "echo INTERPRETERS $(command -v python) $(command -v python3)",
        "echo x > work/control.txt && cat work/control.txt > /dev/null && echo CONTROL-RUN-DIR-OK",
        f"cat {numpy_init} > /dev/null && echo CONTROL-NUMPY-OK",
        probe_script(probes),
    ])
    fake = FakeResponses([[shell(command, 1)], [wse_call("wse_get_experiment", {}, 2)],
                          [message("WSE-STOP: toolchain check finished.", 3)]], token=TOKEN).start()
    try:
        def factory(config, eid):
            return CodexCLIAdapter(codex_path=BINARY.path, provider_config=fake.provider_config(PROXY_TOKEN_ENV),
                                   proxy_token=TOKEN, interrupt_grace_seconds=5.0)

        runner = ReferenceRunner(workspace, setup=["wse_test_plugins:setup"], worker_args=FAST_WORKER,
                                 serve_env={"PYTHONPATH": str(TESTS_DIR)}, adapter_factory=factory,
                                 settle_timeout_seconds=60)
        eid = runner.run_experiment(run_id, prepared["task_id"], SMALL_MODEL_POLICY_ID,
                                    {"kind": "agent", "adapter": "codex_cli@1", "model": MODEL,
                                     "wall_clock_seconds": 300})
    finally:
        fake.stop()
    result = runner.last_result["result"]
    assert result["status"] == "completed" and result["reason"].startswith("the agent stopped: WSE-STOP:"), result
    assert engine.get_experiment(eid).state == "FROZEN"
    # Every model request carried the per-run token, and Codex offered exactly the allow-listed WSE tools.
    assert len(fake.requests) == 3 and all(request["authorized"] for request in fake.requests)
    namespaces = {tool["name"]: [t["name"] for t in tool.get("tools", [])]
                  for item in fake.requests[0]["body"]["input"] if item.get("type") == "additional_tools"
                  for tool in item["tools"]}
    assert sorted(namespaces["mcp__wse"]) == sorted(allowed_tools(engine.get_policy(SMALL_MODEL_POLICY_ID)))
    # A command Codex ran could read the run directory and the analysis libraries, and nothing it must not.
    (item,) = _segment_items(result, "command_execution")
    output = item["aggregated_output"]
    assert "CONTROL-RUN-DIR-OK" in output and "CONTROL-NUMPY-OK" in output and "numpy " in output
    assert "name: wse" in output  # the skill is readable
    # Codex runs commands in a non-login shell, so both names find the environment's interpreter first.
    (interpreters,) = [line.split()[1:] for line in output.splitlines() if line.startswith("INTERPRETERS ")]
    bin_dir = result["provider_metadata"]["python"]["bin_dir"]
    assert interpreters == [f"{bin_dir}/python", f"{bin_dir}/python3"] and "-lc" not in item["command"]
    assert probe_results(probes, output) == {label: "DENIED" for label in probes}
    # The command's environment is the allow-list: no token (by name or value) and no upstream credential.
    environment = dict(line.split("=", 1) for line in output.split("ENV-END")[0].splitlines() if "=" in line)
    assert TOKEN not in output and PROXY_TOKEN_ENV not in environment
    assert "OPENAI_API_KEY" not in environment and "OPENAI_BASE_URL" not in environment and "planted" not in output
    python = result["provider_metadata"]["python"]
    assert environment["PATH"] == f"{python['bin_dir']}:/usr/bin:/bin:/usr/sbin:/sbin"
    assert environment["HOME"].endswith(f"sandboxes/{run_id}/work") and environment["PYTHONPATH"] == python["pythonpath"]
    # The WSE server's environment: what the runner allow-lists, plus Codex's defaults; never the token.
    server = json.loads(record.read_text())
    assert server["token_values"] == [] and server["token_substrings"] == []
    assert not {PROXY_TOKEN_ENV, "OPENAI_API_KEY", "OPENAI_BASE_URL"} & set(server["names"])
    # Normalized events from Codex's real stream, and usage from its session record.
    events = _events(engine, eid)
    calls = [(e["tool"], e["call_id"]) for e in events if e["kind"] == "tool_call"]
    results = [(e["tool"], e["call_id"], e["ok"]) for e in events if e["kind"] == "tool_result"]
    assert [tool for tool, _ in calls] == ["shell", "wse_get_experiment"]
    assert results == [(tool, call_id, True) for tool, call_id in calls]
    responses = [e for e in events if e["kind"] == "model_response"]
    usage = result["usage"]
    assert usage["requests"] == len(responses) == 3 == result["turns"]
    assert usage["input_tokens"] == 1000 + 1001 + 1002 and usage["cached_input_tokens"] == 300
    assert usage["output_tokens"] == 60 and usage["reasoning_output_tokens"] == 15
    # No file Codex or the adapter wrote holds the token, and no process of the run is left.
    leaks = [str(p) for p in [*codex_root.rglob("*"), *(workspace / "logs").glob("serve-*.log")]
             if p.is_file() and not p.is_symlink() and TOKEN.encode() in p.read_bytes()]
    assert leaks == [] and "token_found_in" not in result["provider_metadata"]
    assert _processes_naming(run_id) == []


# -- one blocking call instead of many polls --------------------------------------------------------------------


def _long_wait_setup(engine, prepared, signed_sum) -> str:
    policy = json.loads(json.dumps(signed_sum.load_policy()))
    policy.update({"policy_id": "small_model_long_wait_v1", "max_job_wait_seconds": 60.0})
    engine.register_policy(ExperimentPolicy.model_validate(policy))
    engine.register_suite({"suite_id": "signed_sum_dev_slow", "task_id": prepared["task_id"], "split": "development",
                           "evaluator": "slow_batches@1", "sample_counts": [64], "seeds": [0],
                           "parameters": {"sleep_seconds": 6, "cooperative": True},
                           "data": signed_sum.suite_data(32, signed_sum.DEV_SEED)})
    return policy["policy_id"]


def _wait_run(engine, workspace, prepared, policy_id: str, **adapter_options: Any) -> tuple[dict[str, Any], list]:
    run_id = engine.grant_run(prepared["task_id"], policy_id, prepared["task_checkpoint_id"]).run_id
    evaluation = {"checkpoint_id": prepared["task_checkpoint_id"], "suite_id": "signed_sum_dev_slow",
                  "eval_config": {"sample_count": 64, "seed": 0}, "idempotency_key": "slow"}

    def wait(body: dict[str, Any]) -> list[dict[str, Any]]:
        text = tool_outputs(body)[-1]
        job_id = json.loads(text[text.index("{"):])["result"]["job_id"]
        return [wse_call("wse_get_job", {"job_id": job_id, "wait_seconds": 55}, 2)]

    fake = FakeResponses([[wse_call("wse_submit_evaluation", evaluation, 1)], wait,
                          [message("WSE-STOP: waited once.", 3)]], token=TOKEN).start()
    try:
        runner = ReferenceRunner(workspace, setup=["wse_test_plugins:setup"], worker_args=FAST_WORKER,
                                 serve_env={"PYTHONPATH": str(TESTS_DIR)}, settle_timeout_seconds=60,
                                 adapter_factory=lambda config, eid: CodexCLIAdapter(
                                     codex_path=BINARY.path, provider_config=fake.provider_config(PROXY_TOKEN_ENV),
                                     proxy_token=TOKEN, interrupt_grace_seconds=5.0, **adapter_options))
        eid = runner.run_experiment(run_id, prepared["task_id"], policy_id,
                                    {"kind": "agent", "adapter": "codex_cli@1", "model": MODEL,
                                     "wall_clock_seconds": 300})
    finally:
        fake.stop()
    return runner.last_result["result"], [e for e in _events(engine, eid) if e["kind"] == "tool_result"]


def test_a_long_job_wait_is_one_tool_call_under_the_adapters_tool_timeout(engine, workspace, prepared, signed_sum):
    policy_id = _long_wait_setup(engine, prepared, signed_sum)
    result, results = _wait_run(engine, workspace, prepared, policy_id)
    assert result["provider_metadata"]["tool_timeout_sec"] >= 300
    waits = [event for event in results if event["tool"] == "wse_get_job"]
    assert len(waits) == 1 and waits[0]["ok"] and '"state":"succeeded"' in waits[0]["output"], waits


def test_codex_ends_a_tool_call_at_the_configured_tool_timeout(engine, workspace, prepared, signed_sum):
    policy_id = _long_wait_setup(engine, prepared, signed_sum)
    result, results = _wait_run(engine, workspace, prepared, policy_id, tool_timeout_seconds=2)
    assert result["provider_metadata"]["tool_timeout_sec"] == 2
    (wait,) = [event for event in results if event["tool"] == "wse_get_job"]
    assert not wait["ok"] and wait["error_code"] == "TOOL_TIMEOUT", wait


# -- no instructions or configuration from outside the run ----------------------------------------------------


def test_codex_loads_no_instructions_or_configuration_from_around_the_run(signed_sum, tmp_path, plugins, monkeypatch):
    """A run directory inside a git checkout, and a home full of Codex files: none of it reaches the model."""
    checkout, home = tmp_path / "checkout", tmp_path / "home"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    canaries = {
        checkout / "AGENTS.md": "CANARY-CHECKOUT-AGENTS",
        checkout / "AGENTS.override.md": "CANARY-CHECKOUT-OVERRIDE",
        checkout / ".codex" / "config.toml": 'developer_instructions = "CANARY-CHECKOUT-CONFIG"\n',
        checkout / ".agents" / "skills" / "canary" / "SKILL.md": ("---\nname: canary\ndescription: "
                                                                   "CANARY-CHECKOUT-SKILL\n---\nCANARY-BODY\n"),
        checkout / "runs" / "AGENTS.md": "CANARY-PARENT-AGENTS",
        home / "AGENTS.md": "CANARY-HOME-AGENTS",
        home / ".codex" / "AGENTS.md": "CANARY-USER-CODEX-AGENTS",
        home / ".codex" / "config.toml": 'developer_instructions = "CANARY-USER-CODEX-CONFIG"\n',
        home / ".agents" / "skills" / "canary" / "SKILL.md": ("---\nname: canary-home\ndescription: "
                                                               "CANARY-HOME-SKILL\n---\nCANARY-BODY\n"),
    }
    for path, text in canaries.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    monkeypatch.setenv("HOME", str(home))
    workspace = checkout / "runs" / "workspace"
    with Engine(workspace) as engine:
        plugins.setup(engine.registry)
        ids = signed_sum.setup_workspace(engine, tmp_path / "bundles", final_seed=FINAL_SEED)
        run_id = engine.grant_run(ids["task_id"], SMALL_MODEL_POLICY_ID, ids["task_checkpoint_id"]).run_id
        fake = FakeResponses([[message("WSE-STOP: nothing to do.", 1)]], token=TOKEN).start()
        try:
            runner = ReferenceRunner(workspace, setup=["wse_test_plugins:setup"], worker_args=FAST_WORKER,
                                     serve_env={"PYTHONPATH": str(TESTS_DIR)}, settle_timeout_seconds=60,
                                     adapter_factory=lambda config, eid: CodexCLIAdapter(
                                         codex_path=BINARY.path, provider_config=fake.provider_config(PROXY_TOKEN_ENV),
                                         proxy_token=TOKEN))
            eid = runner.run_experiment(run_id, ids["task_id"], SMALL_MODEL_POLICY_ID,
                                        {"kind": "agent", "adapter": "codex_cli@1", "model": MODEL,
                                         "wall_clock_seconds": 300})
        finally:
            fake.stop()
        assert engine.get_experiment(eid).state == "FROZEN"
    result = runner.last_result["result"]
    assert result["status"] == "completed", result
    (request,) = fake.requests
    seen = json.dumps(request["body"])
    assert [text for text in ("CANARY-CHECKOUT-AGENTS", "CANARY-CHECKOUT-OVERRIDE", "CANARY-CHECKOUT-CONFIG",
                              "CANARY-CHECKOUT-SKILL", "CANARY-PARENT-AGENTS", "CANARY-HOME-AGENTS",
                              "CANARY-USER-CODEX-AGENTS", "CANARY-USER-CODEX-CONFIG", "CANARY-HOME-SKILL",
                              "CANARY-BODY") if text in seen] == []
    # The only skill is WSE's, and the workspace root Codex reports is the run's sandbox, not the checkout.
    sandbox = (workspace / "sandboxes" / run_id).resolve()
    assert "- wse: " in seen and "- canary" not in seen
    assert f"<root>{sandbox}</root>" in seen and f"<root>{checkout.resolve()}</root>" not in seen


def test_codex_offers_no_sub_agents(engine, workspace, prepared):
    """gpt-5.6-luna's code mode lists sub-agent tools; the adapter's configuration removes them."""
    run_id = engine.grant_run(prepared["task_id"], SMALL_MODEL_POLICY_ID, prepared["task_checkpoint_id"]).run_id
    script = ("try { const r = await tools.multi_agent_v1__spawn_agent({message: 'Say hello.'}); "
              "text('SPAWNED ' + JSON.stringify(r)); } catch (e) { text('NO-SPAWN ' + String(e)); }")
    probe = {"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "exec", "input": script,
             "status": "completed"}
    fake = FakeResponses([[probe], [message("WSE-STOP: probed.", 2)]], token=TOKEN).start()
    try:
        runner = ReferenceRunner(workspace, setup=["wse_test_plugins:setup"], worker_args=FAST_WORKER,
                                 serve_env={"PYTHONPATH": str(TESTS_DIR)}, settle_timeout_seconds=60,
                                 adapter_factory=lambda config, eid: CodexCLIAdapter(
                                     codex_path=BINARY.path, provider_config=fake.provider_config(PROXY_TOKEN_ENV),
                                     proxy_token=TOKEN))
        runner.run_experiment(run_id, prepared["task_id"], SMALL_MODEL_POLICY_ID,
                              {"kind": "agent", "adapter": "codex_cli@1", "model": MODEL, "wall_clock_seconds": 300})
    finally:
        fake.stop()
    assert runner.last_result["result"]["status"] == "completed"
    (output,) = tool_outputs(fake.requests[1]["body"])
    assert "NO-SPAWN" in output and "SPAWNED" not in output
    assert len(fake.requests) == 2  # no sub-agent made a model request of its own
