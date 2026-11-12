# Weight-Space Engineering M3: Code-Based Analysis in a Sandbox, and LLM Agents on Qwen3.5-0.8B

> **For agentic workers:** Use the `subagent-driven-development` skill during implementation, delivering and reviewing one task at a time. Track execution with the checkboxes below. Steps marked **(operator)** are the user's own actions or live runs that cost money. This document names p50c's layers and the article's edits, so no agent under trial may read it: the sandbox's allow-list (Section 2.1) keeps the repository out of reach, as the reference runner does for the MCP tools.

**Goal:** Let an agent explore with its own Python code while WSE keeps model evaluation, edits, records, and the final test, and find out whether that combination helps an LLM agent repair Qwen3.5-0.8B's reasoning loops.

**Architecture:** A run whose policy grants the sandbox gets one more MCP tool, `wse_run_python`. Each call runs the agent's code in a fresh Python process under macOS `sandbox-exec`, with an allow-list: the process may read the Python runtime and the run's sandbox inputs (permitted weights of the base checkpoint, results of the run's probes, and outputs of its development passes), may write only the run's sandbox outputs and scratch files, and has no network. Models are evaluated only by WSE jobs. Edits remain registered operators, whose array parameters may name sandbox outputs; WSE validates those values and stores exact patches. Such runs are labeled `unverified`, and WSE reports what it metered as WSE-metered calls, never as the run's total computation.

**Tech stack:** M2's stack, plus macOS's `/usr/bin/sandbox-exec` (an Apple-signed platform binary, present on the operator's macOS 26.7.1). No new Python dependency.

**Date and status:** Created 2026-11-11, after release 0.1.2. The same evening the user decided:

1. **Open analysis first.** The agent may write Python that reads permitted weights and activations, for custom statistics, decompositions, and plots. Model evaluation stays in WSE jobs, and edits stay validated and saved as exact patches, reusing the existing operators. A general submission of candidates computed elsewhere can come later.
2. **Honest labels, strict final test.** Budgets that are not strictly comparable and `constraint_status: unverified` are acceptable at first, but final-test isolation is never relaxed. What WSE metered is recorded as WSE-metered calls and never presented as the sandbox's total computation: a general Python sandbox that holds full weights cannot be shown, by rules alone, not to run forward passes itself. The first version ships this honestly labeled unverified mode; strictly verifiable execution is not required yet.
3. **Pipeline first, comparison second.** One sandbox run on Qwen first confirms that analysis, submission, evaluation, long-job waiting, and recording all work. Then two arms, with and without the sandbox, try to repair the loops without p50c's values.

The [M2 plan](./agentic-weight-space-engineering-m2-plan.md) listed "a sandboxed research-Python mode" under M3 or later; this plan brings it forward. Its Q3 (Tasks 13 to 15, the agent trials on Qwen) has not run; Stage S3 below absorbs it.

## Global Constraints

- Every M1 and M2 global constraint still holds, with one change: M2's "Declarative, metered tools only" now binds every run whose policy does not grant the sandbox, which includes every verified benchmark run.
- **The final test stays isolated, by the operating system.** The sandbox's file access is an allow-list enforced by `sandbox-exec`, never a deny-list or a rule the agent is asked to follow. It never reaches evaluator storage, the rest of the workspace, the repository, model snapshots, the Hugging Face cache, published edits, or the network.
- **Honest labels.** Any run whose policy grants the sandbox is `unverified`, with the reason recorded, whatever its controller. Reports and exports call WSE's usage "WSE-metered calls" and the sandbox's usage "sandbox wall-clock time (not metered computation)".
- **Fail closed.** Where `sandbox-exec` is missing or its self-check fails, the tool is not offered and the runner refuses to start a sandbox run.
- **No credentials in the sandbox.** Its environment is an allow-list; nothing from the operator's environment, such as `OPENAI_API_KEY`, passes through.
- **Existing policies do not change.** New policy fields are optional and omitted while unset, so `small_model_v1`, `qwen_loop_v1`, and their hashes stay byte-identical, as M2's fields did for M1.
- **The suite keeps passing.** All 822 tests pass at the end of every task.

## 1. Stages

| Stage | Purpose | Exit condition |
| --- | --- | --- |
| **S1: An unverified analysis sandbox** | The `wse_run_python` tool, its isolation, inputs, records, and labels; sandbox outputs as edit parameters; long waits for jobs | The isolation tests pass. Offline, a scripted strategy repairs the signed-sum model through MCP with an edit computed in the sandbox, its run is recorded as `unverified`, and its export replays to identical weights. |
| **S2: Pipeline check on Qwen** | One live agent run in the sandbox mode on the Qwen loop task, without p50c's values | The run exercises and records all five functions: analysis in the sandbox, an edit whose parameters come from a sandbox output, a generation pass waited for in a few long waits, selection and freeze, and export. A repair is not required. |
| **S3: Two-arm comparison** | Agents with and without the sandbox, at the same evaluation budget, on the Qwen loop task | Every trial of both arms and the controls are reported against a preregistered criterion, as in M1. |

## 2. Design

### 2.1 The sandbox process

- **One process per call.** `wse_run_python` writes the agent's code to `<sandbox>/calls/<n>/code.py` and starts `sandbox-exec -f <profile> <python> -I -B <code.py>` with the sandbox directory as its working directory. Files persist between calls in `work/`; the process does not.
- **Profile.** `(deny default)`, then allow:
  - reading the Python runtime (`sys.base_prefix` and the environment's `sys.prefix`) and the system libraries Python, NumPy, and PyTorch load at import;
  - reading `<sandbox>/inputs/`, and reading and writing `<sandbox>/work/` and `<sandbox>/outputs/`;
  - starting threads, but no other executables.

  Everything else is denied, including network access, other files, and GPU devices; the sandbox computes on the CPU. Task 1 determines the minimal set of system allowances and lists each one in `docs/experiments.md`.
- **Limits.** Wall-clock time per call and in total, and CPU time (`RLIMIT_CPU`), from the policy. The server kills a process at its wall-clock limit and records the ending. Peak memory is recorded; macOS does not enforce an address-space limit, which Section 6 lists as a risk.
- **Output.** Standard output and error are each returned to the agent within the policy's byte limit, and stored in full, up to a cap, as artifacts.
- **Where the directory lives.** `<workspace>/sandboxes/<run_id>/`. Being inside the workspace grants nothing: the profile allows only that subpath.

### 2.2 What the sandbox can read

- **Weights:** `inputs/weights/base.safetensors`, the base checkpoint's tensors that the policy permits. By default these are all tensors that the family descriptor does not protect; for Qwen3.5-0.8B that leaves out the embeddings, the output head, and the final norm. Candidates are not exported in S1: they differ from the base by the agent's own patches.
- **Activations:** `inputs/probes/<job_id>.npz`, with the vectors of each finished probe of the run (per text and site, the group means, and the contrasts), next to the probe's JSON summary.
- **Development outputs:** `inputs/passes/<report_id>.jsonl`, with each output of the run's finished development passes: problem index, item ID, grade, finished flag, generated tokens, and text. No final-suite data is ever exported.
- **Manifest:** `inputs/manifest.json` lists every input with its size and SHA-256. Inputs are exported when their job settles; the base weights are exported when the run starts.

### 2.3 Sandbox results in edits

- An array-valued operator parameter, such as `project_out`'s `direction` and `columns`, `add_low_rank`'s `u` and `v`, or `set_slice`'s `values`, may be given as `wse://sandbox/outputs/<file>.npy`.
- WSE reads that file from the run's own sandbox directory, checks its dtype, shape, finiteness, and size against the policy, stores it as an artifact by SHA-256, and freezes the values into the patch. This extends the existing vector references (`resolve_vector_reference`, which already resolves probe results such as `wse://analyses/<sha256>#/sites/0/contrast/direction`).
- Patches therefore stay exact and self-contained: export and replay need no sandbox file, and the element limit (`max_patch_elements`, which counts inline values) applies as before.

### 2.4 Evaluation and long waits

- Generation passes and probes stay WSE jobs, metered as in M2.
- `wse_get_job`'s `wait_seconds` is capped by a new policy limit, `max_job_wait_seconds`. Unset, it is 10 seconds, as now. The run-bound server publishes the tool's schema with the policy's cap, and the OpenAI Agents adapter sets `client_session_timeout_seconds` above it.
- With a cap of 600 seconds, a 37-minute pass needs about 4 waits instead of about 220. This applies to both arms of S3, because the tools-only arm needs it as much as the sandbox arm.

### 2.5 Policy, budget, and labels

- **`ExperimentPolicy.sandbox`**, optional: `calls`, `seconds_per_call`, `total_seconds`, `max_output_bytes`, `readable_tensors` (patterns or roles; by default every unprotected tensor), and `export_pass_texts`.
- **Ledger counters** `sandbox_calls` and `sandbox_seconds` (wall-clock). They are reported apart from WSE-metered calls, under the names of the Global Constraints.
- **Status.** A run under a policy that grants the sandbox is recorded with `constraint_status: unverified` and `constraint_reason: "sandbox code execution: WSE meters only its own calls"`.
- **Policy `qwen_loop_sandbox_v1`:** `qwen_loop_v1`'s limits, plus a sandbox of 100 calls, at most 300 seconds per call and 2 hours in total, and 16 KiB of output per stream, and `max_job_wait_seconds` of 600. Task 9's run sheet confirms or lowers these values before the live run.

### 2.6 Records and export

- **Trace.** Each call adds a tool call carrying the code, and a tool result carrying the exit status, wall-clock and CPU time, peak memory, bounded output, and every output file's name, size, and SHA-256. An audit event records the same.
- **Export.** It adds `sandbox/calls/<n>/` with the code, the full output, and output files up to a size cap, all hashed in the export manifest. Exact replay is unchanged.

### 2.7 Agent instructions

`render_instructions` adds a section when the policy grants the sandbox. It lists the tool, the input layout, the limits, how to name a sandbox output in an edit, and that WSE meters only its own calls. It never names p50c, its layers, the article, or its logs.

### 2.8 Not in this milestone

- A general submission of candidate weights computed elsewhere (a later milestone, as decided).
- Verifiable sandbox execution, metered or proven free of forward passes.
- Persistent kernels, GPU access in the sandbox, WSE's Python API inside it, and network access.
- Isolation on Linux (for example `bubblewrap`): until then the tool fails closed outside macOS.

## 3. Implementation Tasks (S1)

### Task 1: The sandbox process and its profile

**Files:** create `src/wse/sandbox.py` and `tests/test_sandbox.py`; modify `docs/experiments.md`.

- [ ] Write failing tests:
  - `test_sandbox_runs_numpy_and_torch_on_the_cpu`;
  - `test_sandbox_reads_its_inputs_and_writes_only_its_work_and_outputs`;
  - `test_sandbox_cannot_read_the_workspace_the_repository_or_a_planted_home_file`;
  - `test_sandbox_cannot_open_a_network_connection`;
  - `test_sandbox_environment_carries_no_credentials`;
  - `test_wall_clock_limit_kills_the_process_and_records_it`;
  - `test_output_is_returned_within_its_limit_and_stored_in_full`;
  - `test_tool_fails_closed_without_sandbox_exec`.
- [ ] Run the tests; they must fail.
- [ ] Implement the profile, the process runner, the limits, and the self-check; record every system allowance that imports need.
- [ ] Run the new tests and the full suite. Review and commit.

### Task 2: Policy, ledger, and labels

**Files:** modify `src/wse/policy.py`, `src/wse/budget.py`, `src/wse/schemas.py`, `src/wse/database.py` (a migration adding the two counters), `src/wse/experiments.py`, `src/wse/runner.py`, and `src/wse/audit.py`; create `tests/test_sandbox_policy.py`.

- [ ] Write failing tests:
  - `test_existing_policies_and_their_hashes_are_unchanged`;
  - `test_a_sandbox_policy_marks_every_run_unverified_with_its_reason`;
  - `test_sandbox_usage_is_charged_to_its_own_counters`;
  - `test_reports_separate_wse_metered_calls_from_sandbox_time`;
  - `test_qwen_loop_sandbox_v1_limits`.
- [ ] Run the tests; they must fail. Implement; run the new tests and the full suite. Review and commit.

### Task 3: Sandbox inputs

**Files:** create `src/wse/sandbox_inputs.py` and `tests/test_sandbox_inputs.py`; modify `src/wse/jobs.py` and `src/wse/runner.py`.

- [ ] Write failing tests:
  - `test_permitted_base_tensors_are_exported_and_protected_ones_are_not`;
  - `test_probe_vectors_appear_in_the_inputs_when_the_job_settles`;
  - `test_development_outputs_are_exported_and_final_data_never_is`;
  - `test_the_manifest_hashes_every_input`.
- [ ] Run the tests; they must fail. Implement; run the new tests and the full suite. Review and commit.

### Task 4: The tool `wse_run_python`

**Files:** modify `src/wse/mcp_server.py`, `src/wse/experiments.py`, `src/wse/runner.py`, and `docs/protocol.md`; create `tests/test_sandbox_tool.py`.

- [ ] Write failing tests:
  - `test_run_python_returns_bounded_output_and_its_output_files`;
  - `test_run_python_is_refused_without_a_sandbox_grant`;
  - `test_run_python_is_recorded_in_the_trace_and_the_audit_log`;
  - `test_mcp_and_python_agree_on_run_python`, following M1's parity test;
  - `test_exhausted_sandbox_calls_stop_the_run`.
- [ ] Run the tests; they must fail. Implement; run the new tests and the full suite. Review and commit.

### Task 5: Sandbox outputs as edit parameters

**Files:** modify `src/wse/analysis.py`, `src/wse/patches.py`, and `src/wse/experiments.py`; create `tests/test_sandbox_references.py`.

- [ ] Write failing tests:
  - `test_a_project_out_direction_from_a_sandbox_file_is_frozen_exactly`;
  - `test_set_slice_values_from_a_sandbox_file_respect_the_element_limit`;
  - `test_non_finite_mis_shaped_or_foreign_files_are_refused`;
  - `test_replaying_the_export_needs_no_sandbox_file`.
- [ ] Run the tests; they must fail. Implement; run the new tests and the full suite. Review and commit.

### Task 6: Long waits for jobs

**Files:** modify `src/wse/jobs.py`, `src/wse/mcp_server.py`, `src/wse/agent_adapters/openai_agents.py`, and `src/wse/runner.py`; create `tests/test_long_waits.py`.

- [ ] Write failing tests:
  - `test_get_job_waits_up_to_the_policy_cap`;
  - `test_policies_without_a_cap_keep_ten_seconds`;
  - `test_the_published_schema_states_the_policy_cap`;
  - `test_the_adapter_session_timeout_exceeds_the_cap`.
- [ ] Run the tests; they must fail. Implement; run the new tests and the full suite. Review and commit.

### Task 7: Instructions, export, documentation, and leak checks

**Files:** modify `src/wse/runner.py` (`render_instructions`), `src/wse/audit.py`, `docs/experiments.md`, `docs/protocol.md`, and `docs/architecture.md`; create `tests/test_sandbox_records.py`.

- [ ] Write failing tests:
  - `test_instructions_describe_the_sandbox_and_name_no_oracle`;
  - `test_the_export_holds_every_sandbox_call_and_still_replays`;
  - `test_qwen_sandbox_inputs_hold_no_published_edit_or_final_item`.
- [ ] Run the tests; they must fail. Implement; document the mode, its labels, and its limits; run the new tests and the full suite. Review and commit.

### Task 8: Offline end-to-end check

**Files:** create `src/wse/strategies/sandbox_search.py` (`sandbox_search@1`) and `tests/test_sandbox_e2e.py`; modify `src/wse/registry.py`.

- [ ] Write a failing test, `test_a_scripted_sandbox_strategy_repairs_signed_sum_through_mcp`. Through a real `wse serve` and the SDK test double, the strategy reads `layers.1.weight` from the sandbox inputs, computes a corrected row with NumPy, writes it to `outputs/`, previews and applies a `set_slice` that names that file, evaluates, and selects. The final accuracy is 1.0, the run is `unverified` with its reason, and the export replays to identical weights.
- [ ] Run it; it must fail. Implement; run it and the full suite. Review and commit. **S1 exit:** the conditions of Section 1 hold.

## 4. Stage S2: Pipeline Check on Qwen

### Task 9: One live sandbox run

**Files:** create `examples/qwen_loops/run_agent.py` (operator-only) and `dev/m3-pipeline-check.md`.

- [ ] Write the run sheet into the report before the run: the task (the existing Qwen loop task, whose practice suite is 32 GSM8K training problems at 8,192 tokens), the policy limits, the agent (`gpt-5.6-luna`, `max_turns`, and a wall-clock limit of several hours), and the expected tokens and cost.
- [ ] Confirm that the agent can reach no p50c value: the sandbox inputs hold the base checkpoint only, and the allow-list excludes the repository, the operator's model storage, and the article's logs.
- [ ] **(operator)** Run it. **It is outward-facing:** it sends task instructions and tool results to the model provider and costs money. Before starting, post the expected tokens and cost in chat, check that no other luna job is running, and load the key with `--env-file .env`.
- [ ] Verify from the export: each of the five functions in Section 1's S2 row was exercised and recorded; the long waits took a few calls per pass; the final test ran only after the freeze; the export replays; a scan finds no key, endpoint, final-suite item, or oracle value in any output.
- [ ] Write `dev/m3-pipeline-check.md` with the run's timeline, costs (WSE-metered calls, sandbox time, agent tokens), what the agent computed in the sandbox, every candidate, and every failure. Commit. **S2 exit.**

## 5. Stage S3: Two-Arm Comparison

### Task 10: Preregistration, controls, and trials

Planned in detail after S2, from its measured time and cost. It absorbs M2's Tasks 13 to 15:

- **Arms.** A uses the tools only (`qwen_loop_v1` with long waits); B adds the sandbox (`qwen_loop_sandbox_v1`). The task, the evaluation budget, and the agent configuration are the same, and neither arm has access to p50c's values.
- **Preregistration** before any trial: the criterion (final accuracy above the base with a paired interval that excludes zero, and no increase in the limit-hit rate), the number of trials per arm, the final suite, and the analysis plan.
- **Controls** on the same final suite: the base, the replayed p50c as an operator-only reference, and random projections matched to p50c's changed-element count and norm.
- **Trials.** Final evaluations run only after every trial of both arms has frozen. The report, `dev/m3-qwen-trials.md`, states each arm's result against the criterion, including failures, and keeps WSE-metered calls apart from sandbox time.

## 6. Risks and Open Questions

- **`sandbox-exec` is deprecated by Apple.** It works on macOS 26.7.1; a later macOS may remove or change it. The self-check detects that, and the tool fails closed.
- **Import-time allowances.** PyTorch may need system services or reads beyond the obvious paths. Task 1 lists every allowance it adds, and none may grant reads of user directories.
- **Memory.** The exported Qwen weights take about 1 GB, and macOS enforces no address-space limit, so a careless analysis can exhaust memory; the server records peak memory and enforces the wall-clock limit.
- **Time and cost.** Each generation pass takes about 37 minutes on the operator's Mac, so S2 takes hours and S3 days of machine time, plus agent tokens that grow with code and outputs.
- **Unverified by design.** An agent with the weights in a sandbox can run the model itself. S2 and S3 report what the sandbox was used for from the recorded code, but they cannot exclude such computation.
- **Estimated effort:** S1 about two to three working days; S2 about one day including the run.
