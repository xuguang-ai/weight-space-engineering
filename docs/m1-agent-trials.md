# M1 agent trials

This report records the live trials of the OpenAI Agents SDK reference agent
that the [plan](../agentic-weight-space-engineering-plan.md) requires for M1.
Agent acceptance asks for 5 independent trials with a fixed configuration,
with a target of at least 4 of 5 trials whose selected candidate reaches at
least 99% final accuracy. The report describes the injected fault and its
fix. Like the plan and `examples/signed_sum/build_fixture.py`, it must stay
out of reach of any agent under trial; the reference runner gives the agent
only the WSE tools.

**Result.** The acceptance set, `main2`, met the target.

- **`main2`:** all 5 trials found and committed the fix. Final accuracy was
  1.0 (512 of 512 final pairs; 95% Clopper–Pearson lower bound 0.9928),
  against 0.5 for the unedited control.
- **`main`:** the earlier set reached 3 of 5. Its two failures came from a
  harness defect, described below and fixed before `main2`.

Both sets are reported in full. These trials show that the toolchain lets an
agent diagnose and repair a planted fault in a 22-parameter model. One small
fixture says little about agent capability in general.

## Setup

- **Task (`signed_sum_v1`).** Classify x = (x0, x1) as class 1 when
  x0 + x1 > 0. The model is an FP32 2 → 4 (ReLU) → 2 MLP with 22
  parameters.
  - Its first layer computes the four positive and negative input directions.
  - The reference second-layer weight is `[[-1, 1, -1, 1], [1, -1, 1, -1]]`.
  - The injected fault multiplies row 1 of `layers.1.weight` by −1. Both rows
    then match, the logits tie, and accuracy is 0.5.
- **Data.**
  - The development suite has 32 pairs (64 examples). A full development
    evaluation costs 64 examples, and the controller's `target_met` stop needs
    one with accuracy 1.0.
  - The final suite has 512 independently generated pairs (1,024 examples,
    margin 0.1). Only the final evaluator can read its seed.
  - Final evaluations run after every trial of a set has frozen. Each one
    includes the unedited base checkpoint as a fixed control.
- **Policy:** `small_model_v1`; budgets and tools are in
  [experiments.md](experiments.md).
- **Controller.** `openai_agents@1` with openai-agents 0.24.0, openai 3.46.0
  and mcp 2.3.0 (protocol 2026-07-28), on Python 3.14.8 and macOS 26.7.1
  (arm64).
  - The model is `gpt-5.6-luna`, served by an Azure OpenAI deployment through
    the Responses API; the recorded endpoint class is `azure_openai`.
  - It runs with default model settings and at most 100 turns and 900 s per
    trial. SDK tracing is disabled, and the agent can use the 12 `wse_*`
    tools the policy allows.
- **Fixed configuration.** Every recorded controller configuration field is
  identical across all trials of the three sets except `cwd`, the workspace
  path. The configuration hash includes `cwd`, so it differs between sets but
  is constant within each set.
- **Isolation.** Each trial has its own run grant and a fresh context. It
  runs to its freeze before the next trial starts.
- **Command:** `examples/signed_sum/run_trials.py OUT --label LABEL
  --controller openai-agents --model gpt-5.6-luna --trials N`, as described
  in [experiments.md](experiments.md#live-agent-trials).

## Sets

| Set | Run (UTC) | Code | Trials meeting the target | Tokens in / out | Cost bound |
| --- | --- | --- | --- | --- | --- |
| `pilot` | 2026-11-06 04:33–04:34 | 69d1d58 | 1 / 1 | 95,985 / 1,850 | $0.021 |
| `main` | 2026-11-06 14:54–14:55 | 6c15661 | 3 / 5 | 262,079 / 5,149 | $0.059 |
| `main2` (acceptance) | 2026-11-06 20:59–21:07 | 7ea498b | 5 / 5 | 406,911 / 8,352 | $0.091 |

The cost bounds assume prices of $0.20 per million input tokens and $1.20
per million output tokens. Cached input tokens are not recorded, so each
bound assumes none were cached. The pilot ran before 6c15661, so its usage
record names the endpoint class `openai`; the endpoint was the same.

## Acceptance set (`main2`)

| Trial | Experiment | Turns | Tool calls | Tokens in / out | Elapsed (s) | Dev examples | Changed parameters | Final accuracy (control) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `exp_ea5c0faf…` | 12 | 16 | 104,128 / 1,858 | 33.1 | 128 | 4 / 22 | 1.0 (0.5) |
| 2 | `exp_df7f71ff…` | 9 | 12 | 69,534 / 1,523 | 21.3 | 64 | 4 / 22 | 1.0 (0.5) |
| 3 | `exp_88cd4c6b…` | 9 | 12 | 70,061 / 1,586 | 21.1 | 64 | 4 / 22 | 1.0 (0.5) |
| 4 | `exp_aac37529…` | 9 | 12 | 69,489 / 1,514 | 364.4 | 64 | 4 / 22 | 1.0 (0.5) |
| 5 | `exp_6e447989…` | 11 | 14 | 93,699 / 1,871 | 24.4 | 128 | 4 / 22 | 1.0 (0.5) |

Every trial stopped on `target_met` with one candidate. Each scored 512 of
512 final pairs (Clopper–Pearson 95% lower bound 0.9928) and replayed to
byte-identical weights.

All five committed the same patch,
`set_slice layers.1.weight[1:2, 0:4] = [[1, -1, 1, -1]]`, which restores
the reference row. Each hypothesis named the cause. Trial 2's, for example:
"The first layer exposes positive and negative parts of each input. Both
output rows currently equal [-1,1,-1,1], so the logits tie; setting class-1
to the opposite row makes logit1-logit0 proportional to x0+x1."

The final evaluations ran after the last trial ended. The final reports are
stamped 21:07:10 and the last controller ended at 21:07:09.

## Earlier set (`main`) and the harness defect

| Trial | Experiment | Status | Turns | Tool calls | Tokens in / out | Elapsed (s) | Dev examples | Changed parameters | Final accuracy (control) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `exp_086cbcdc…` | stopped: `target_met` | 9 | 12 | 70,188 / 1,565 | 20.1 | 64 | 4 / 22 | 1.0 (0.5) |
| 2 | `exp_0740705c…` | error | 0 | 0 | 0 / 0 | 0.01 | 0 | 0 / 22 | 0.5 (0.5) |
| 3 | `exp_6c0f30b0…` | stopped: `target_met` | 11 | 14 | 96,566 / 1,875 | 24.0 | 128 | 4 / 22 | 1.0 (0.5) |
| 4 | `exp_42fc3875…` | error | 0 | 0 | 0 / 0 | 0.01 | 0 | 0 / 22 | 0.5 (0.5) |
| 5 | `exp_f16bba0d…` | stopped: `target_met` | 11 | 14 | 95,325 / 1,709 | 25.6 | 128 | 4 / 22 | 1.0 (0.5) |

The three successful trials each committed one candidate, with the same
patch as in `main2`. Trials 2 and 4 committed none: they ended with
`RuntimeError: Event loop is closed` before any model response.
Each complete trace has two events: `run_started`, then `run_finished` with
status `error`. Both trials were still settled, frozen, final-evaluated on
the unedited base checkpoint, exported, and replayed.

The cause was in WSE, not the agent.

- openai-agents 0.24.0 keeps one HTTP connection pool per process for every
  default OpenAI client (`shared_http_client()` in
  `agents/models/openai_provider.py`).
- A pooled connection belongs to the event loop that opened it, and
  `run_trials.py` runs each trial in its own event loop.
- Trial 2 therefore reused a keep-alive connection from trial 1's closed loop
  and failed on its first request.
- The failure dropped that connection, so trial 3 opened a new one and
  succeeded. Failures alternated from then on.

Commit 7ea498b gives each model-name run its own `AsyncOpenAI` client. The
client is configured from `OPENAI_API_KEY` and `OPENAI_BASE_URL` like the
SDK's default and is closed in the same loop.
`test_consecutive_model_name_trials_in_one_process_each_reach_the_endpoint`
runs three trials in one process against a loopback Responses stub. Before
the fix it gave completed, error, completed; after it, three completions.

## Observations

None of these changed an outcome.

- **Nested values.** Every successful live trial (9 of 9 across the three
  sets) first sent `set_slice` values as a flat list for a two-axis
  selection. Each received `INVALID_PATCH` with the expected shape and sent
  nested lists on its next turn. The operator description already states the
  rule.
- **Experiment IDs.** In `main2`, two trials mistyped the 32-character
  experiment ID in `wse_read_tensor` calls and got `NOT_FOUND`, then used the
  correct ID:
  - trial 1 dropped 4 characters, in two calls;
  - trial 5 dropped 3 characters, in one call.
- **One slow response.** In `main2` trial 4, the ninth model response took
  348 s, while the other eight took about 1.2–3.4 s.
  - That request used 11,843 input and 117 output tokens, in line with the
    trial's other requests, so the time went to the provider or the network,
    not to generation.
  - The trial recorded no per-request timing. From commit 8f6be85, each
    `model_response` event records the latency of the SDK's model call
    (including the client's retries) and the provider's response and request
    IDs.
  - Azure OpenAI returns the request ID in `x-request-id`.

## Checks

- **Replay.** The exports of all three sets replayed to byte-identical
  weights.
- **Leak scans.** Scans of each set's output directory found:
  - no API key and no endpoint host;
  - neither the final-suite seed nor sampled final-suite values outside the
    workspace.

  The scans covered 40 files for `pilot`, 125 for `main`, and 141 for
  `main2`, with 160 sampled values for `main` and `main2`.
- **Tests.** 365 engineering tests pass at 7ea498b and 366 at 8f6be85.

Raw outputs stay with the operator, outside this repository. They comprise
the exports with tool traces and patches, the reports, server logs, replays
and workspaces.

## Open items

- **Agent-facing changes for later configurations.** Each of these changes
  what the agent sees, so it starts a new configuration:
  - accept a flat list for a selection whose other axes have length 1, or
    name the expected nesting in the error;
  - let a run-bound server default the experiment ID.
- **HTTP-level diagnostics.** Attempts, status codes, and Azure's region and
  spillover headers are not captured, because the SDK does not expose
  response headers.
- **Cached input tokens.** They are not recorded, so cost figures are upper
  bounds.
- **Misleading log hint.** An error's reason says "see the server log" even
  when the runner raised the error, as in `main`; the server logs there were
  clean.
