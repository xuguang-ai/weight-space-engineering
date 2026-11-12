# Running experiments and interpreting results

This document describes how an operator runs managed weight-space
engineering trials with the reference runner, what each trial records and
exports, and how to read the results. It covers tasks, agent instructions,
and budgets, stepwise strategies, the OpenAI Agents SDK reference agent and
its model configuration, exports and replay, and the offline demo whose
reports appear below. Tool contracts are in [protocol.md](protocol.md);
extension contracts in [extensions.md](extensions.md).

## Workflow

1. **Prepare the workspace (operator).** Import the task model and register
   the task, policy, development and final data, and suites. For the
   signed-sum task, steps 1 to 4 of
   [examples/signed_sum/agent_api.py](../examples/signed_sum/agent_api.py),
   which the README walks through, make these calls;
   `examples/signed_sum/build_fixture.py OUT --workspace WORKSPACE` makes
   the same ones for the trial harness. The final set lives only in the
   workspace's evaluator storage. [Your own task](#your-own-task) describes
   each part.
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

## Tasks, instructions, and budgets

### What the agent is told

The reference agent's instructions are generated from the registered task
and policy by `wse.runner.render_instructions`, and its first message is
always `Begin work on experiment EXPERIMENT_ID.` Beyond these two texts, the
agent sees only the descriptions of its WSE tools and the tool results. The
task states the behavior to reach and how it is measured, never what is
wrong with the model, how the model was built, or how to change it.

For the signed-sum task, the README's live run of 2026-11-11 received this
text, rendered here from its export's `task.json` and `policy.json` with the
run's experiment ID and tools. An earlier run that day, under the built-in
`small_model_v1` with the same values, received the same text apart from
its experiment ID and the policy's name:

```text
You control one managed weight-space engineering experiment through the WSE tools listed below. They are your only tools: you cannot run code, read files, or use the network.

Experiment ID: exp_ab39f014f54b5d65a6811e0a5d742b4b (the tools use it when you omit experiment_id)
Task signed_sum_v1: Signed-sum classification
The registered model reads two real inputs x = (x0, x1) and returns logits for two classes; its prediction is the class with the larger logit. It should predict class 1 when x0 + x1 > 0 and class 0 otherwise. Using only the permitted inspection and editing operations, produce a candidate checkpoint that follows this rule, and select it.
Model input: x = (x0, x1), two float32 values
Model output: logits for class 0 and class 1; the predicted class is the argmax
Objective: Predict class 1 exactly when x0 + x1 > 0, measured as accuracy on the development suite.
Acceptance: accuracy at least 1.
The target is met only by a successful evaluation of development suite signed_sum_dev_v1 covering all 64 examples with accuracy at least 1.

Policy signed_sum_policy_v1 (mode bounded_eval); budget for the whole experiment:
- candidate commits: 20
- patch previews: 50
- tensor analyses: 50
- tensor reads: 100
- development evaluation batches: 10, at most 64 examples each and 640 examples in total
- computation time: 300 s in total, at most 30 s per operation
Consumed allowance is never refunded; every result reports usage and the remaining budget.
Analyses: norm, stats, top_k. Edit operations: scale_slice, set_slice. A read returns at most 2048 elements; a patch has at most 32 operations.

How to work:
1. Call wse_get_experiment with the experiment ID for the task, budget, permitted operations, development suites, and checkpoints.
2. Gather evidence with wse_inspect_model, wse_read_tensor, and wse_analyze on the experiment's checkpoints.
3. Propose an edit with wse_preview_patch (base_checkpoint_id, a hypothesis of one or two sentences stating the observable evidence, and the operations), then create a candidate with wse_apply_patch by patch_id.
4. Evaluate checkpoints with wse_submit_evaluation on a development suite with a preregistered eval_config (sample_count, seed), wait with wse_get_job (wait_seconds up to 10), read a whole report with wse_get_report, and compare two reports with wse_compare.
5. Select a checkpoint with wse_select_checkpoint, passing the currently selected ID as expected_selected_id; selecting an earlier checkpoint rolls back.
6. When you are done, reply with a short final message and no tool call. The selected checkpoint is then frozen for a separate final evaluation that you cannot see.

The run also ends automatically once a full development evaluation of the selected checkpoint meets the target, after 3 consecutive evaluated candidates without improvement, or when the budget cannot cover the next operation (BUDGET_EXCEEDED).
Every mutating call needs a new idempotency_key; repeating a key with the same input returns the original result.
Keep tool arguments and messages to brief hypotheses and observable evidence; do not write out private reasoning.
Available tools: wse_list_models, wse_get_experiment, wse_inspect_model, wse_read_tensor, wse_analyze, wse_preview_patch, wse_apply_patch, wse_submit_evaluation, wse_get_job, wse_get_report, wse_cancel_job, wse_compare, wse_select_checkpoint.
```

A stepwise strategy receives the same task, objective, budget, operations,
and suites as its first observation ([Stepwise strategies](#stepwise-strategies)).

### Your own task

An experiment is registered in four parts. The README's agent example writes
each one out, in steps 1 to 4 of
[examples/signed_sum/agent_api.py](../examples/signed_sum/agent_api.py):

1. **The target checkpoint.** `Engine.import_model` stores tensors given in
   memory with a registered adapter (`mlp@1` for the signed-sum
   classifier). `Engine.import_hf_snapshot`, or `WSE.from_pretrained`,
   imports a Hugging Face snapshot at a fixed revision
   ([models.md](models.md)).
2. **The task,** a `TaskDefinition` registered with `Engine.register_task`.
   Its `description` is the problem statement that the agent reads (up to
   4,096 characters). `input_description` and `output_description` describe
   the model's interface, and `objective` states what is measured and the
   gates that count as success: for signed sum, accuracy of at least 1.0 on
   the development suite named by `development_suite_id`, over all of its
   `development_examples`.
3. **The evaluation suites.** `Engine.register_dataset` stores examples in
   evaluator storage, and `Engine.register_suite` names the evaluator
   (`classification@1` for signed sum, `generation@1` for a language model),
   the split, and the preregistered sample counts and seeds. The controller
   evaluates only `development` suites; only the final evaluator reads the
   `final` suite, after the selected checkpoint is frozen.
4. **The policy,** registered with `Engine.register_policy`: the mode, the
   permitted edits and analyses, and the budget ([Budgets](#budgets)). A
   registered policy is immutable: registering an ID again with other
   content fails with `VERSION_CONFLICT`, so a policy with other values
   needs a new `policy_id`. Every workspace already holds the built-in
   policies, `small_model_v1` among them.

A run is then granted on the task, the policy, and the base checkpoint, as
in [One experiment in one call](#one-experiment-in-one-call).
`examples/qwen_loops/loop_setup.py` registers a generation task on
Qwen3.5-0.8B the same way: its objective is to lower the development
limit-hit rate without lowering accuracy (the gate `{"metric": "accuracy",
"min_delta": 0.0}`), and its suite uses `generation@1` with the profile
`qwen35think@1`, the grader `gsm8k_numeric@1`, and at most 8,192 new tokens.
`examples/signed_sum/build_fixture.py` registers the same signed-sum task,
from [task.json](../examples/signed_sum/task.json) and
[policy.json](../examples/signed_sum/policy.json), for the trial harness and
the tests.

### Budgets

A policy's budget covers the whole experiment. Consumed allowance is never
refunded, every tool result reports usage and the remaining budget, and an
operation the remaining budget cannot cover fails with `BUDGET_EXCEEDED`.
The two policies used so far:

| Budget item | `small_model_v1` (signed sum) | `qwen_loop_v1` (Qwen3.5-0.8B) |
| --- | --- | --- |
| Development evaluation batches, one model forward pass each | 10, at most 64 examples each and 640 in total | none |
| Generation passes | none | 10, at most 32 problems each and 8,192 new tokens per answer |
| Activation-probe calls | none | 25, at most 64 texts of up to 1,024 tokens each |
| Tensor analyses | 50 | 50 |
| Tensor reads, at most 2,048 values each | 100 | 100 |
| Patch previews | 50 | 50 |
| Candidate commits | 20 | 20 |
| Computation time | 300 s, at most 30 s per operation | 6 h, at most 2 h per operation |
| Analyses | `norm`, `stats`, `top_k` | `direction_projection`, `norm`, `stats`, `svd_top_k`, `tensor_norms`, `top_k` |
| Edit operations | `scale_slice`, `set_slice` | `add_low_rank`, `fill_slice`, `project_out`, `scale_slice`, `set_slice` |

`small_model_v1` is defined in
[examples/signed_sum/policy.json](../examples/signed_sum/policy.json) and
`qwen_loop_v1` by `wse.policy.qwen_loop_policy()`. `qwen_loop_sandbox_v1`
(`wse.policy.qwen_loop_sandbox_policy()`) is `qwen_loop_v1` with code
execution: 100 sandbox calls of at most 300 s of wall-clock time each and
7,200 s in all, and job waits of up to 600 s
([below](#code-execution-in-a-sandbox-unverified)). The README's agent
example registers `small_model_v1`'s values as `signed_sum_policy_v1`. The budgets in the
article that WSE reproduces count larger passes: there, one Qwen pass could
generate answers to up to 512 practice problems or read up to 1,024 texts
([summary](../dev/weight-space-programming-summary.md)), so its 50 forward
passes for p50c do not translate one-to-one into these limits.

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
The operator chooses the model name, which must name a model with tool calls
at the provider; every recorded trial used `gpt-5.6-luna`. Credentials come
from the SDK's normal environment configuration (`OPENAI_API_KEY`, and
`OPENAI_BASE_URL` for an endpoint other than OpenAI's; see
[Agent model configuration](#agent-model-configuration)) and are never
read, recorded, or passed to the WSE server.

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

### Agent model configuration

The reference agent needs a model that supports tool calls, named with
`--model` (or `model=`), and the provider's key and endpoint in the
environment of the process that runs it. The SDK reads them from
`OPENAI_API_KEY` and `OPENAI_BASE_URL`. Models other than `gpt-5.6-luna`
have not been tested; the recorded trials reached it through an Azure
OpenAI deployment.

**The `.env` file.** Copy the repository's template,
[.env.example](../.env.example), to `.env` at the repository root, which git
ignores, and fill in your values:

```sh
cp .env.example .env
```

```text
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
```

Then load the file for one command with uv's `--env-file` flag (uv reads no
`.env` file without it):

```sh
uv run --extra agents-openai --env-file .env \
  python examples/signed_sum/agent_api.py \
  demo/live \
  --model YOUR_AGENT_MODEL
```

**OpenAI.** Keep the template's endpoint, which is OpenAI's own and the
SDK's default, and name an OpenAI model with `--model`.

**Azure OpenAI or another compatible endpoint.** Set `OPENAI_BASE_URL` to
the endpoint. For an Azure OpenAI resource, that is the resource's endpoint,
with the resource's name in place of `YOUR-RESOURCE`, and `--model` is the
deployment's name. The recorded trials used this route, through the
Responses API:

```text
OPENAI_API_KEY=your-azure-key
OPENAI_BASE_URL=https://YOUR-RESOURCE.services.ai.azure.com/openai/v1
```

Do not leave `OPENAI_BASE_URL` empty; to use the SDK's default, delete the
line.

**Without a file.** To keep the key out of files, enter it at a hidden
prompt instead and leave out `--env-file .env` (for another endpoint, also
export `OPENAI_BASE_URL`). Run the line on its own, because in a pasted
block of several lines `read` can take the next line as the key:

```sh
printf 'OpenAI API key: '; read -rs OPENAI_API_KEY; echo; export OPENAI_API_KEY
```

**What stays private, and what is sent.** Neither variable reaches the WSE
server, whose environment is an allow-list, and neither is written to the
workspace or an export: the controller record names only the kind of
endpoint (`openai`, `azure_openai`, or `openai_compatible`). The model
provider receives the task instructions and the tool results, and bills the
calls to your account. The README's signed-sum run of 2026-11-11 made 8
requests, with 68,866 input and 1,428 output tokens.

### Stop conditions

After every operation the runner reads authoritative WSE state and stops the
controller (cancelling the SDK run) when:

- an operation fails with `BUDGET_EXCEEDED`, no computation time remains, or
  (`bounded_eval`) no candidate commits or evaluation batches remain; a
  stepwise action that the remaining budget cannot cover is not executed;
- (`bounded_eval`) a full development evaluation of the **selected**
  checkpoint meets the task's target;
- (`bounded_eval`) 3 consecutive evaluated candidates show no improvement
  over the best checkpoint before them: a higher development accuracy or,
  on a generation suite, a lower limit-hit rate at no lower accuracy;
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

## Code execution in a sandbox (unverified)

A policy with a `sandbox` section grants one more tool, `wse_run_python`,
which runs the agent's own Python (NumPy, PyTorch on the CPU, safetensors)
in a fresh process per call. Analysis is open; evaluation and edits are not:
generation passes and probes stay WSE jobs, and every edit is still a
validated exact patch of a registered operator.

**What the code sees.** The process's working directory is the run's
sandbox, `<workspace>/sandboxes/<run_id>/`:

- `inputs/`, read only. `weights/base.safetensors` holds the base
  checkpoint's tensors in their stored dtype, except tensors tied as aliases
  and those the model family protects (for Qwen3.5-0.8B: the embeddings, the
  output head, and the final norm); a policy's `sandbox.readable_tensors`
  patterns narrow it further. `passes/<report_id>.jsonl` holds every output
  of the run's development generation passes (problem, item, sample, grade,
  answer, finished, limit hit, generated tokens, and text);
  `probes/<job_id>.npz` and `.json` each finished probe's vectors
  (`site<i>_mean`, `site<i>_group_mean_a` and `_b`, `site<i>_contrast`) and
  its document; `manifest.json` every input's size, SHA-256, and source.
  WSE brings them up to date before each call. Final suites are never
  exported, and candidates are not: they differ from the base only by the
  run's own patches.
- `work/`, read and write, kept between calls; `outputs/`, read and write,
  emptied before each call; `tmp/`, the process's `TMPDIR`.

**Isolation.** On macOS the process runs under `sandbox-exec` with a
profile that denies everything, imports Apple's `system.sb` (the system
libraries and services every process needs), and then allows only reading
the Python runtime and executing its interpreter, the sandbox directories
above, and the metadata of their parent directories. The rest of the
workspace (evaluator storage among it), the repository, the user's files,
other executables, new processes, and the network are denied, and the
environment is an allow-list, so no credential reaches the process. A call
is killed at its wall-clock limit (the policy's `seconds_per_call`, or the
time left); CPU time is limited to that many seconds per core, and files to
1 GiB each. Peak memory is recorded but not limited: macOS enforces no
address-space limit. Before its first use WSE checks that an allowed read
works and a read outside the allow-list fails; elsewhere, and on other
platforms, the tool fails closed with `UNSUPPORTED_CAPABILITY`.

**From code to an edit.** Each finite numeric array of one or two
dimensions, of at most 65,536 values, that the code saves to `outputs/` with
`numpy.save` is stored as a run-scoped result and returned with a pointer,
`<result_uri>#/values` (a row: `#/values/<row>`). The parameters that take
stored vectors accept it: `project_out`'s `direction`, each column of
`add_low_rank`'s `u` and `v`, and `direction_projection`'s `direction`.
`set_slice` takes its values inline, from what the code printed; a row of
any length is also one exact edit as `add_low_rank` on that row with
`u = [[1.0]]` and `v` a pointer to the new row minus the old one. A patch
records each pointer with the length and hash of the vector it named, and
its exact after-values, so replay needs no sandbox file.

**What is metered, and the label.** Each call is charged one
`sandbox_calls` and its wall-clock seconds to `sandbox_seconds`; nothing is
held while the code runs. What the code computes is not metered, and
`compute_seconds` is not charged: WSE counts its own calls, not the work
inside them. Every run of such a policy is therefore recorded as
`constraint_status: unverified` with `constraint_reason: "sandbox code
execution: WSE meters only its own calls"`, and summaries and reports list
the WSE-metered calls apart from "sandbox wall-clock time (not metered
computation)". The isolation of the final test is unchanged: the final
suite is evaluated only after the freeze, by the worker, from storage the
sandbox cannot read.

**Records.** Each call stores `calls/<n>/code.py`, `stdout.txt` and
`stderr.txt` (in full up to 1 MiB each), `result.json` (the ending, times,
peak memory, and output files with their hashes), and copies of its output
files of at most 1 MiB in `calls/<n>/outputs/`. It is recorded as a
`sandbox.ran` audit event and is idempotent by its key.

## Records

Each experiment records its controller (`controller_runs`): kind, exact ID
and version (`scripted_search@1`, `openai_agents@1`), configuration and its
hash, `constraint_status`, the tracing setting, and the termination record:
status (`completed`, `stopped`, `max_turns`, `timeout`, `cancelled`,
`connection_failed`, `error`), reason, stop condition, final message, turns,
tool calls, elapsed time, usage (for an agent: token counts, the model name,
and the class of the endpoint the SDK called, `openai` for OpenAI's endpoint
or when `OPENAI_BASE_URL` is unset, `azure_openai` for an Azure resource,
otherwise `openai_compatible`; the URL itself is never recorded), provider metadata (SDK, MCP, and protocol
versions, limits, allowed and observed tools), the settlement and freeze, the
environment (Python, platform, CPU, package versions), and, for an agent, the
path of the `wse serve` log (`serve_log`; the reason of a `connection_failed`
or `error` ending names the underlying exception and points to that log).
The normalized trace (`controller_events`) holds tool calls with their
arguments, tool results with their error codes, sizes, hashes, and bounded
excerpts, model responses with per-request usage, latency (the SDK's whole
model call, the client's retries included), and the provider's response and
request IDs, messages, and stop requests. Private
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
of `wse serve` is `unverified` unless its operator attests otherwise, and a
run under a policy that grants the sandbox is always `unverified`, with its
`constraint_reason` ([above](#code-execution-in-a-sandbox-unverified)).

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
elements and fraction of parameters. A run whose policy grants the sandbox
adds `sandbox/`: each call's code, stored output, result summary, and kept
output files (`sandbox/calls/<n>/`); `sandbox/inputs_manifest.json` (what the
code could read, with hashes and sources, not the inputs themselves); and
each stored sandbox array byte for byte (`sandbox/arrays/<sha256>.json`, so
its file hashes to the digest in the URI an edit names). The manifest's
`sandbox` field indexes them, and `costs.sandbox` gives the calls and their
wall-clock time under the label above.

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
(each takes a few seconds on CPU; IDs differ per run). The last two run the
test double through the agent SDK, which the `agents-openai` extra
installs:

```sh
uv run python examples/signed_sum/run_trials.py DEMO --label scripted
uv run python examples/signed_sum/run_trials.py DEMO --label control --policy small_model_weights_only_v1
uv run --extra agents-openai python examples/signed_sum/run_trials.py DEMO --label sdk-double \
    --controller openai-agents --model wse-test-double:scripted_search@1
uv run --extra agents-openai python examples/signed_sum/run_trials.py DEMO --label sdk-double-control \
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

### One experiment in one call

`wse.agent_runs.run_agent_experiment` grants a run, runs the controller, and
freezes the experiment. With `final_evaluation=True`, as the example below
passes, it also runs the final evaluation, exports the closed experiment,
and replays the export. `examples/signed_sum/agent_api.py` sets up a new
signed-sum workspace in the directory it is given and runs one experiment
this way; the README walks through its six steps. Offline, with a scripted
controller in place of a model, it takes a few seconds. The IDs differ per
run:

```sh
uv run python examples/signed_sum/agent_api.py demo/offline
```

```text
scripted_search@1 on model ckpt_eb88256683d4b9a797c577954a537086, task signed_sum_v1, policy signed_sum_policy_v1
scripted_search@1 ended stopped: target_met: a full development evaluation of the selected checkpoint meets the target
17 tool calls in 2 s
edit: scale_slice@1 on layers.1.weight[1:2, 0:4], factor -1.0
layers.1.weight: [[-1.0, 1.0, -1.0, 1.0], [-1.0, 1.0, -1.0, 1.0]] -> [[-1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 1.0, -1.0]]
patch: demo/offline/workspace/agent_runs/run_7b3bc43f0068b7e75aac39fcb523d7a0/export/patches/patch_98ebc508141555498571c25a7cdf68e9.json
final test, 1,024 examples the agent never saw: 50% correct before the edit, 100% after
the export replays to byte-identical weights: True
```

For a live LLM agent, add `--model YOUR_AGENT_MODEL`, a model with tool
calls, with the key configured as in
[Agent model configuration](#agent-model-configuration). For a series of
trials, leave the final evaluation off (the default) and run it only after
every trial has frozen, so that no trial can adapt to an earlier trial's
final result.

### Serving a run to an MCP client

An operator grants a run, a permission for one experiment under one policy,
and serves it over MCP; the agent connects to that server and works only
through its tools. On the workspace that `agent_api.py` set up in
`demo/offline`, from the repository root (the base checkpoint is the model
the workspace imported; its candidates come from patches):

```bash
W=demo/offline/workspace
BASE=$(uv run python -c "
from wse.api import Engine
with Engine('$W') as engine:
    print(next(m.checkpoint_id for m in engine.list_models() if m.provenance.kind == 'import'))
")

RUN=$(uv run wse --workspace "$W" \
  grant-run signed_sum_v1 signed_sum_policy_v1 "$BASE" \
  | uv run python -c "import json, sys; print(json.load(sys.stdin)['run_id'])")

# An MCP server over stdio, bound to that run
uv run wse --workspace "$W" \
  serve --run-id "$RUN"
```

From Python, `run_agent_experiment` does it in one call: it grants a run,
runs the controller, and freezes the experiment when the controller stops.
With `model=...`, the controller is the OpenAI Agents SDK agent, connected
to the run's MCP server (run it with `uv run --extra agents-openai`):

```python
from wse.agent_runs import run_agent_experiment
from wse.api import Engine

workspace = "demo/offline/workspace"  # set up by examples/signed_sum/agent_api.py
with Engine(workspace) as engine:
    base = next(m.checkpoint_id for m in engine.list_models() if m.provenance.kind == "import")

run = run_agent_experiment(
    workspace,
    task="signed_sum_v1",
    base=base,
    policy="signed_sum_policy_v1",
    model="wse-test-double:scripted_search@1",
)
print(run.state, run.summary["status"])  # FROZEN stopped: the final evaluation is off by default
```

## Live agent trials

The M1 agent acceptance run uses 5 trials with isolated run grants and fresh
contexts and a fixed, operator-chosen model (`gpt-5.6-luna` in the recorded
trials). With the key in `.env` as in
[Agent model configuration](#agent-model-configuration):

```sh
uv run --extra agents-openai --env-file .env \
  python examples/signed_sum/run_trials.py \
  LIVE \
  --label live \
  --controller openai-agents \
  --model YOUR_AGENT_MODEL \
  --trials 5
```

Each trial runs to its freeze before the next starts; the final evaluations
run only after all 5 have frozen; every trial is exported, replayed, and
listed with its failure cause in `LIVE/reports/live.md`. A trial that raises
is listed with its error and counts as unsuccessful, and the remaining
trials still run. Ctrl-C stops the running trial (settled and frozen first),
starts no further trial, and still final-evaluates and reports the frozen
ones. The script exits with 0, with 1 if any trial or step failed, or with
130 after an interrupt; each server's log is `LIVE/logs/serve-RUN_ID.log`. The exact SDK and
model settings are in each export's `controller.json`. Running these trials
sends the task instructions and tool results to the model provider. The M1
acceptance trials and their results are in [m1-agent-trials.md](../dev/m1-agent-trials.md).
