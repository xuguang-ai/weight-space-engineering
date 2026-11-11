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
   signed-sum task, `examples/signed_sum/build_fixture.py OUT --workspace
   WORKSPACE` does this; the final set and its generation seed live only in
   the workspace's evaluator storage. [Your own task](#your-own-task) shows
   the calls.
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

For the signed-sum task under `small_model_v1`, the live run of 2026-11-11
received this text, rendered here from its export's `task.json` and
`policy.json` with the run's experiment ID:

```text
You control one managed weight-space engineering experiment through the WSE tools listed below. They are your only tools: you cannot run code, read files, or use the network.

Experiment ID: exp_cba05a78e444f8cc8be78c21af7743ad (the tools use it when you omit experiment_id)
Task signed_sum_v1: Signed-sum classification
The registered model reads two real inputs x = (x0, x1) and returns logits for two classes; its prediction is the class with the larger logit. It should predict class 1 when x0 + x1 > 0 and class 0 otherwise. Using only the permitted inspection and editing operations, produce a candidate checkpoint that follows this rule, and select it.
Model input: x = (x0, x1), two float32 values
Model output: logits for class 0 and class 1; the predicted class is the argmax
Objective: Predict class 1 exactly when x0 + x1 > 0, measured as accuracy on the development suite.
Acceptance: accuracy at least 1.
The target is met only by a successful evaluation of development suite signed_sum_dev_v1 covering all 64 examples with accuracy at least 1.

Policy small_model_v1 (mode bounded_eval); budget for the whole experiment:
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

A task is a `TaskDefinition`, written as `task.json` and registered with
`Engine.register_task`. Its `description` is the problem statement that the
agent reads (up to 4,096 characters). `input_description` and
`output_description` describe the model's interface, and `objective` states
what is measured and the gates that count as success: for signed sum,
accuracy of at least 1.0 on the development suite named by
`development_suite_id`, over all of its `development_examples`. See
[examples/signed_sum/task.json](../examples/signed_sum/task.json).

The rest of an experiment is registered with the task: the target
checkpoint, a policy (`policy.json`: the budget and the permitted
operations), and the development and final evaluation suites. The
signed-sum example's `setup_workspace`, in
[examples/signed_sum/build_fixture.py](../examples/signed_sum/build_fixture.py),
is a complete template (shown here without its docstring and type hints,
and with comments added):

```python
def setup_workspace(engine, out_dir, *, final_seed=None, with_reference=False):
    paths = build(Path(out_dir), with_reference=with_reference)

    # The target checkpoint, the task statement, and the policy
    task_model = engine.import_bundle(paths["task"])
    engine.register_task(load_task())
    policy = engine.register_policy(load_policy())

    # Development and final data, generated at setup time
    development = engine.register_dataset(
        suite_data(DEV_PAIRS, DEV_SEED),
        generation={"generator": GENERATOR, "pairs": DEV_PAIRS, "seed": DEV_SEED, "margin": MARGIN},
    )
    seed = secrets.randbits(63) if final_seed is None else int(final_seed)
    final = engine.register_dataset(
        suite_data(FINAL_PAIRS, seed),
        generation={"generator": GENERATOR, "pairs": FINAL_PAIRS, "seed": seed, "margin": MARGIN},
    )

    # The suites: development for the controller, final for the final evaluator only
    engine.register_suite(development_suite(development))
    engine.register_suite({
        "suite_id": FINAL_SUITE_ID,
        "task_id": TASK_ID,
        "split": "final",
        "evaluator": EVALUATOR,
        "sample_counts": [2 * FINAL_PAIRS],
        "seeds": [0],
        "dataset": final.model_dump(mode="json"),
    })

    ids = {
        "task_id": TASK_ID,
        "policy_id": policy.policy_id,
        "task_checkpoint_id": task_model.checkpoint_id,
        "development_suite_id": DEV_SUITE_ID,
        "final_suite_id": FINAL_SUITE_ID,
    }
    if with_reference:
        ids["reference_checkpoint_id"] = engine.import_bundle(paths["reference"]).checkpoint_id
    return ids
```

`load_task()` and `load_policy()` read the example's `task.json` and
`policy.json`. Only the final evaluator reads the final suite's data, after
the selected checkpoint is frozen. A run is then granted on the task, the
policy, and the base checkpoint, as in
[One experiment in one call](#one-experiment-in-one-call).
`examples/qwen_loops/loop_setup.py` registers a generation task on
Qwen3.5-0.8B the same way.

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
`qwen_loop_v1` by `wse.policy.qwen_loop_policy()`. The budgets in the
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
`--model` (or `model=`), and the provider's key in the environment of the
process that runs it. Models other than `gpt-5.6-luna` have not been
tested; the recorded trials reached it through an Azure OpenAI deployment.

**OpenAI.** The SDK reads the key from `OPENAI_API_KEY`. Put it in a `.env`
file at the repository root, which git ignores:

```text
OPENAI_API_KEY=your-api-key
```

Then load the file for one command with uv's `--env-file` flag (uv reads no
`.env` file without it):

```sh
uv run --no-sync --env-file .env \
  python examples/signed_sum/agent_api.py \
  demo/live \
  --model YOUR_AGENT_MODEL
```

To keep the key out of files, enter it at a hidden prompt instead and leave
out `--env-file .env`. Run the line on its own, because in a pasted block of
several lines `read` can take the next line as the key:

```sh
printf 'OpenAI API key: '; read -rs OPENAI_API_KEY; echo; export OPENAI_API_KEY
```

**Azure OpenAI or another compatible endpoint.** Also set `OPENAI_BASE_URL`;
the SDK calls OpenAI's own endpoint only when it is unset. For an Azure
OpenAI resource, the `.env` file holds the Azure key and the resource's
endpoint, with the resource's name in place of `RESOURCE`:

```text
OPENAI_API_KEY=your-azure-key
OPENAI_BASE_URL=https://RESOURCE.services.ai.azure.com/openai/v1
```

`--model` is then the deployment's name. The recorded trials used this
route, through the Responses API.

**What stays private, and what is sent.** Neither variable reaches the WSE
server, whose environment is an allow-list, and neither is written to the
workspace or an export: the controller record names only the kind of
endpoint (`openai`, `azure_openai`, or `openai_compatible`). The model
provider receives the task instructions and the tool results, and bills the
calls to your account. The signed-sum run of 2026-11-11 made 8 requests,
with 68,681 input and 1,402 output tokens.

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

### One experiment in one call

`wse.agent_runs.run_agent_experiment` grants a run, runs the controller, and
freezes the experiment. With `final_evaluation=True`, as the example below
passes, it also runs the final evaluation, exports the closed experiment,
and replays the export. `examples/signed_sum/agent_api.py` builds the
signed-sum workspace on its first use and runs one experiment this way.
Offline, with a scripted controller in place of a model, it takes a few
seconds. The IDs differ per run, and the script prints the export path in
full, shortened here to `OUT/...`:

```sh
uv run --no-sync python examples/signed_sum/agent_api.py OUT
```

```text
experiment exp_a0373ebd434304fad7bacc2ef3280b99 (run run_ad9c7452e48359213d4dbef2da5e6a2c): CLOSED
controller scripted_search@1 ended stopped: target_met: a full development evaluation of the selected checkpoint meets the target
success: True; final: {"candidate_accuracy": 1.0, "clopper_pearson_lower": 0.9928210499737405, "clopper_pearson_upper": 1.0, "clusters": 512, "control_accuracy": 0.5, "difference": 0.5, "fully_correct_clusters": 512}
export: OUT/workspace/agent_runs/run_ad9c7452e48359213d4dbef2da5e6a2c/export; replay byte-identical: True
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
through its tools. On the workspace that `agent_api.py` prepared in
`demo/offline`, from the repository root:

```bash
A=demo/offline
BASE=$(uv run --no-sync python -c \
  "import json; print(json.load(open('$A/ids.json'))['task_checkpoint_id'])")

RUN=$(uv run --no-sync wse --workspace "$A/workspace" \
  grant-run signed_sum_v1 small_model_v1 "$BASE" \
  | uv run --no-sync python -c "import json, sys; print(json.load(sys.stdin)['run_id'])")

# An MCP server over stdio, bound to that run
uv run --no-sync wse --workspace "$A/workspace" \
  serve --run-id "$RUN"
```

From Python, `run_agent_experiment` does it in one call: it grants a run,
runs the controller, and freezes the experiment when the controller stops.
With `model=...`, the controller is the OpenAI Agents SDK agent, connected
to the run's MCP server:

```python
import json
from pathlib import Path

from wse.agent_runs import run_agent_experiment

A = Path("demo/offline")  # prepared by examples/signed_sum/agent_api.py
ids = json.loads((A / "ids.json").read_text())

run = run_agent_experiment(
    A / "workspace",
    task=ids["task_id"],
    base=ids["task_checkpoint_id"],
    policy=ids["policy_id"],
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
uv run --no-sync --env-file .env \
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
