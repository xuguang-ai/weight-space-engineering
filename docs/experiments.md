# Running experiments and interpreting results

This document describes how an operator runs managed weight-space
engineering trials with the reference runner, what each trial records and
exports, and how to read the results. It covers stepwise strategies, the
OpenAI Agents SDK reference agent, exports and replay, and the offline demo
whose reports appear below. Tool contracts are in [protocol.md](protocol.md);
extension contracts in [extensions.md](extensions.md).

## Workflow

1. **Prepare the workspace (operator).** Import the task model and register
   the task, policy, development and final data, and suites. For the
   signed-sum task, `examples/signed_sum/build_fixture.py OUT --workspace
   WORKSPACE` does this; the final set and its generation seed live only in
   the workspace's evaluator storage.
2. **Grant one run per trial** (`wse grant-run TASK POLICY BASE_CHECKPOINT`).
   A grant permits one experiment and one controller run.
3. **Run the controller** (`wse run --run-id RUN ...`, or
   `wse.runner.run_experiment(run_id, task_id, policy_id, agent_config)`).
   The runner persists the experiment, runs one controller, settles every
   outstanding job, and freezes the selected checkpoint. It never closes the
   experiment.
4. **Final-evaluate after every trial has frozen** (`wse final-evaluate
   EXPERIMENT`): the frozen candidate and the base-checkpoint control on the
   same final examples, then the experiment closes.
5. **Export, replay, and report** (`wse export`, `wse replay`, `wse trials`).

`examples/signed_sum/run_trials.py` performs steps 1 to 5 for a batch of
trials, running the final evaluations only after all of them have frozen.

## Controllers

### Stepwise strategies

`--controller scripted` (or `{"kind": "strategy", "strategy":
"scripted_search@1", "parameters": {...}}`) runs a registered `Strategy`.
The runner gives it the experiment description (task, objective, budget,
operations, suites) as its first observation and executes each proposed
action through an Engine bound to the run grant: the same operations,
policy, budget, and audit records as the MCP tools. For these runs the
runner owns the evaluation worker. Limits: `max_steps` (500) and
`wall_clock_seconds` (900).

`scripted_search@1` is deterministic and task-agnostic: it reads the
manifest, orders single-row `scale_slice` edits of every weight matrix from
the output end (factors default to `[-1.0]`), and in `bounded_eval` mode
evaluates the base once, then previews, applies, evaluates, and compares one
candidate at a time, selecting each improvement. In `weights_only` mode it
commits and selects its first candidate and stops. It reads neither the
fixture builder nor any reference model or oracle.

### OpenAI Agents SDK reference agent

`--controller openai-agents --model MODEL` (or `{"kind": "agent", "model":
MODEL}`) runs `OpenAIAgentsAdapter` from the optional `agents-openai` extra.
The operator chooses the model name; credentials come from the SDK's normal
environment configuration (for example `OPENAI_API_KEY` in the operator's
shell) and are never read, recorded, or passed to the WSE server.

| Setting | Value |
| --- | --- |
| SDK loop | the SDK's own `Agent` and `Runner.run`; no second tool loop around it |
| Tools | only the allow-listed WSE MCP tools (static tool filter); evaluation tools only when the policy permits evaluation; no hosted, shell, file, or code tools |
| MCP connection | `MCPServerStdio` running `wse serve --run-id RUN` on the same workspace, started in the workspace directory with an allow-listed environment (`PATH`, `HOME`, locale, `TMPDIR`, `PYTHONPATH`, `VIRTUAL_ENV`; names that look like credentials are refused); `client_session_timeout_seconds` 30 |
| Scheduler | the `wse serve` process owns the run's job scheduler (one worker); the runner starts none |
| Turn limit | `max_turns=100`, always passed explicitly (the SDK default is 10) |
| Wall clock | 900 s per trial, independent of WSE's 300 s computation budget |
| Tracing | `RunConfig(tracing_disabled=True)`, no sensitive trace data, and `OPENAI_AGENTS_DISABLE_TRACING=1` for the server; a benchmark configuration cannot enable tracing |
| Instructions | generated from `task.json` and the policy (`wse.runner.render_instructions`), never from repository documentation |

The agent cannot reach the source code, this documentation, the plan, the
fixture builder, or the final data: its only channel is the run-bound WSE
tools, which see only the grant's base checkpoint and the run's own
candidates, patches, jobs, and reports.

`--model wse-test-double:STRATEGY` replaces the language model with an SDK
`Model` test double whose tool calls come from a registered strategy. The
real SDK runner then drives a real `wse serve` without any network access;
the demo below and the adapter tests use it. Its token counts are synthetic
(about four characters per token) and its provider is reported as
`wse-test-double`.

### Stop conditions

After every operation the runner reads authoritative WSE state and stops the
controller (cancelling the SDK run) when:

- an operation fails with `BUDGET_EXCEEDED`, no computation time remains, or
  (`bounded_eval`) no candidate commits or evaluation batches remain; a
  stepwise action that the remaining budget cannot cover is not executed;
- (`bounded_eval`) a full development evaluation of the **selected**
  checkpoint meets the task's target;
- (`bounded_eval`) 3 consecutive evaluated candidates show no improvement
  over the best development accuracy before them;
- 3 consecutive operations are denied by the policy;
- (`weights_only`) the single permitted candidate is committed and selected;
  this mode has no development-score threshold.

The controller may also stop voluntarily (a final message, or a strategy's
`stop`). After any ending, including the turn limit, the wall-clock limit,
cancellation, an SDK error, a failed connection, or a failure inside the
runner, the runner cancels every queued or running job, waits for the worker
to settle it (recovering jobs whose worker was lost), and freezes the
selected checkpoint, while the MCP server stays connected; only then does it
close the connection.

The operator can stop a running trial with Ctrl-C. The first SIGINT stops a
stepwise strategy before its next action, or cancels the SDK loop; the
runner then settles and freezes as above, records the trial as `cancelled`
with the stop condition `operator_interrupt`, and only then raises
`KeyboardInterrupt` to its caller. A second Ctrl-C aborts at once and can
leave the experiment open. The `wse serve` process runs in its own process
group, so a terminal's Ctrl-C reaches only the runner; its evaluation worker
stops after its current job if `wse serve` dies without cleanup (a crash or
SIGKILL), so no orphaned worker claims a later trial's jobs.

## Records

Each experiment records its controller (`controller_runs`): kind, exact ID
and version (`scripted_search@1`, `openai_agents@1`), configuration and its
hash, `constraint_status`, the tracing setting, and the termination record:
status (`completed`, `stopped`, `max_turns`, `timeout`, `cancelled`,
`connection_failed`, `error`), reason, stop condition, final message, turns,
tool calls, elapsed time, usage (for an agent: token counts, the model name,
and the class of the endpoint the SDK called, `openai` when `OPENAI_BASE_URL`
is unset, `azure_openai` for an Azure resource, otherwise
`openai_compatible`; the URL itself is never recorded), provider metadata (SDK, MCP, and protocol
versions, limits, allowed and observed tools), the settlement and freeze, the
environment (Python, platform, CPU, package versions), and, for an agent, the
path of the `wse serve` log (`serve_log`; the reason of a `connection_failed`
or `error` ending names the underlying exception and points to that log).
The normalized trace (`controller_events`) holds tool calls with their
arguments, tool results with their error codes, sizes, hashes, and bounded
excerpts, model responses with per-request usage, messages, and stop
requests. Private
reasoning is never requested or stored: reasoning items are counted, not
kept, and hypotheses are one or two sentences of observable evidence.

Usage fields that the host cannot report are `null` with a reason, never
zero: OpenAI Agents SDK reports token counts but not cost or the served model
snapshot; a stepwise strategy calls no language model, so its token fields
are "not applicable".

`constraint_status` is `verified` when the controller's only execution
channel is the run-bound WSE interface: the scripted strategy (actions
executed by the runner) and the reference agent in its benchmark
configuration (no tool besides the allow-listed WSE tools). Any other client
of `wse serve` is `unverified` unless its operator attests otherwise.

## Exports and replay

```sh
wse --workspace WORKSPACE export EXPERIMENT --output EXPORT_DIR
wse --workspace CLEAN_WORKSPACE replay EXPORT_DIR
```

An export (operator only) contains `export_manifest.json` (hashes of the
base checkpoint, every patch, candidate, and report; the selected
checkpoint; the final summary; controller, tracing, `constraint_status`,
costs, environment, and the content hash of every other file),
`experiment.json` (state and run grant), `task.json`, `policy.json`,
`checkpoints/` (manifests), `patches/` (each preview with its frozen spec and
exact materialized patch), `reports/` (development and final reports),
`final_report.json`, `audit_events.json`, `controller.json`,
`tool_trace.json`, and `base_model/` (the base bundle, so replay needs
nothing else). Final-set inputs, labels, and the generation seed are
excluded; `--include-final-data` adds them for an operator who needs them.
Costs include the budget ledger, candidate count, evaluated examples,
controller usage and elapsed time, and the selected candidate's changed
elements and fraction of parameters.

`wse replay` verifies every file hash, imports the base bundle, and rebuilds
each candidate from its parent by exact patch replay; every replayed weights
hash must equal the recorded one (byte-identical weights). It does not
re-execute recipes, and it does not reproduce metrics: evaluation metrics
across platforms or library versions agree only within statistical
tolerances.

## Interpreting results

- **Engineering acceptance versus agent acceptance.** The test suite
  verifies the toolchain (including an oracle repair in engineering tests
  only). Agent trials are reported separately: a trial succeeds when its
  frozen candidate reaches at least 99% final accuracy, and the M1 target is
  at least 4 of 5 trials. Repairing a synthetic fault is not evidence of
  general capability improvement.
- **Intervals.** Final reports give the accuracy of the candidate and of the
  base-checkpoint control on the same examples, their paired difference with
  a cluster-bootstrap interval (clusters are the generated `x, -x` pairs),
  and an exact Clopper-Pearson 95% interval on fully correct pairs: 512/512
  gives a lower bound of 99.28%, 511/512 gives 98.92%.
- **Per trial**, `wse trials EXPERIMENT... --format markdown` reports the
  controller, status and stop condition, candidates, evaluated examples,
  elapsed time, tokens, the changed fraction of parameters, the final
  accuracy with its control, and every failure cause.

## Offline demo and representative reports

These commands reproduce the reports below without a model or network
(each takes a few seconds on CPU; IDs differ per run):

```sh
uv run --no-sync python examples/signed_sum/run_trials.py DEMO --label scripted
uv run --no-sync python examples/signed_sum/run_trials.py DEMO --label control --policy small_model_weights_only_v1
uv run --no-sync python examples/signed_sum/run_trials.py DEMO --label sdk-double \
    --controller openai-agents --model wse-test-double:scripted_search@1
uv run --no-sync python examples/signed_sum/run_trials.py DEMO --label sdk-double-control \
    --controller openai-agents --model wse-test-double:scripted_search@1 --policy small_model_weights_only_v1
```

Representative output (2026-11-04, macOS 26.7.1 arm64, CPU, Python 3.14.8,
torch 2.15.0, openai-agents 0.24.0, mcp 2.3.0):

| Report | Controller | Policy | Status | Candidates | Evaluated examples | Tokens (in/out) | Changed fraction | Final accuracy (control) | CP 95% lower |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scripted (success) | scripted_search@1 | small_model_v1 | stopped: target_met | 2 | 192 | n/a (no language model) | 0.1818 | 1.0000 (0.5000) | 0.9928 |
| control (failure) | scripted_search@1 | small_model_weights_only_v1 | stopped: candidate_committed | 1 | 0 | n/a (no language model) | 0.1818 | 0.0000 (0.5000) | 0.0000 |
| sdk-double (success) | openai_agents@1, wse-test-double:scripted_search@1 | small_model_v1 | stopped: target_met | 2 | 192 | 131042/948 (synthetic) | 0.1818 | 1.0000 (0.5000) | 0.9928 |
| sdk-double-control (failure) | openai_agents@1, wse-test-double:scripted_search@1 | small_model_weights_only_v1 | stopped: candidate_committed | 1 | 0 | 19920/336 (synthetic) | 0.1818 | 0.0000 (0.5000) | 0.0000 |

Every export replayed to byte-identical weights. The failure reports state
their cause, for example: "final accuracy 0 is below 0.99; the controller
ended stopped: candidate_committed: the single permitted candidate is
committed and selected". The weights-only control commits one candidate
without development feedback, so its outcome depends only on the search
order; the scripted and test-double runs exercise the toolchain and say
nothing about an agent's ability.

## Live agent trials

The M1 agent acceptance run uses 5 trials with isolated run grants and fresh
contexts and a fixed, operator-chosen model:

```sh
export OPENAI_API_KEY=...   # in the operator's own shell; never written to the workspace or exports
uv run --no-sync python examples/signed_sum/run_trials.py LIVE --label live \
    --controller openai-agents --model MODEL_NAME --trials 5
```

The key can instead live in a `.env` file at the repository root, which git
ignores, together with `OPENAI_BASE_URL` for an endpoint other than OpenAI's
(for an Azure resource, `https://RESOURCE.services.ai.azure.com/openai/v1`).
`uv run --no-sync --env-file .env python examples/signed_sum/run_trials.py ...`
loads both for that one command. uv reads no `.env` without the flag, and
neither variable reaches the WSE server, whose environment is an allow-list.

Each trial runs to its freeze before the next starts; the final evaluations
run only after all 5 have frozen; every trial is exported, replayed, and
listed with its failure cause in `LIVE/reports/live.md`. A trial that raises
is listed with its error and counts as unsuccessful, and the remaining
trials still run. Ctrl-C stops the running trial (settled and frozen first),
starts no further trial, and still final-evaluates and reports the frozen
ones. The script exits with 0, with 1 if any trial or step failed, or with
130 after an interrupt; each server's log is `LIVE/logs/serve-RUN_ID.log`. The exact SDK and
model settings are in each export's `controller.json`. Running these trials
sends the task instructions and tool results to the model provider.
