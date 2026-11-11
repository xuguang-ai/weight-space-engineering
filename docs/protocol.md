# WSE MCP protocol

This document describes the MCP interface of the weight-space engineering
framework (`wse serve`, implemented in `src/wse/mcp_server.py`): its tools,
resources, result and error formats, and two example sessions. The M1
examples use a hypothetical 8-input MLP with the task `example_task_v1` and
the development suite `example_dev_v1`; the language-model examples use
Llama-3.2-1B-Instruct with a generic arithmetic task `example_math_v1`, its
generation suite `example_math_dev_v1`, and an operator-registered policy
`example_lm_v1` that grants generation passes and probes. No example
describes a model or task used in an agent trial. IDs are illustrative; real
IDs are opaque strings returned by the server.

MCP is one front end to the Python library. Every tool and resource calls the
same `Engine` operation as the Python API and the CLI, so policy decisions,
budgets, idempotency, and stored artifacts are identical across the three
interfaces. Handlers validate arguments, call one service operation, and map
its result or error; they contain no editing, analysis, budget, or evaluation
logic.

## Starting a server

The operator prepares the workspace (imports models and registers tasks,
policies, and suites), issues a run grant, and starts one stdio server for
that grant:

```sh
wse --workspace WORKSPACE grant-run example_task_v1 small_model_v1 ckpt_example_base
wse --workspace WORKSPACE serve --run-id run_example
```

`wse serve` options:

| Option | Meaning |
| --- | --- |
| `--run-id` | The persisted run grant the server is bound to (required). An unknown grant fails before serving, with the error JSON on stderr and exit code 2. |
| `--setup module:function` | Operator setup hook called with the registry of the server and of its worker (repeatable), for registered extensions. |
| `--no-worker` | Do not start the evaluation worker. By default the server process owns the run's job scheduler and keeps one worker process running until it exits. |
| `--worker-args "..."` | Extra worker options, for example `"--poll-seconds 0.02"`. |
| `--constraint-status verified` | Attest that the host has no other execution channel (Section 6.1 of the plan). The default is `unverified`. |
| `--log-file PATH` | Append the server's and its worker's log lines to this file instead of stderr (the reference runner uses it). |

The transport is stdio: stdout carries only the protocol and every log line
goes to stderr. When the client closes the connection, the server stops its
worker and exits. Experiment state lives in the workspace, never in the
connection, so a new server for the same grant continues where the previous
one stopped.

### Run binding

The server serves through an Engine bound to its run grant. Every tool and
resource can see only the grant's base checkpoint, the run's single
experiment, and that experiment's candidates, patches, jobs, and reports. Any
other ID, including checkpoints and reports of other runs and the operator's
own reference models, fails with the same `NOT_FOUND` error as an ID that
does not exist. `wse_create_experiment` accepts only the grant's `run_id`,
`task_id`, and `policy_id`, and the run has at most one experiment: retries,
even with a new idempotency key, return it unchanged and never reset its
budget. Operator actions (imports, grants, freezing, final evaluation,
exports) are not exposed through MCP. A run-bound Engine in Python follows
the same rules: it cannot open checkpoints or read weights outside the
metered experiment operations, and it cannot read the exact materialized
patch. Final reports are withheld until the
experiment is closed.

### Protocol versions

The server uses the official MCP Python SDK 2.3.0 (`MCPServer`) and speaks MCP
2026-07-28, whose requests are self-contained: no experiment or session state
depends on an `initialize` handshake. The SDK serves older clients through
its legacy handshake. Observed on 2026-11-04 (macOS, Python 3.14.8):

| Client | Negotiated version | Notes |
| --- | --- | --- |
| `mcp.Client` 2.3.0, `mode="auto"` | 2026-07-28 | Used by the protocol tests. |
| `mcp.Client` 2.3.0, `mode="legacy"` | 2025-11-25 | Served by the SDK's handshake; same tools and results. |
| OpenAI Agents SDK 0.24.0 `agents.mcp.MCPServerStdio` (with mcp 2.3.0) | 2026-07-28 | Tool names arrive verbatim; with the default `use_structured_content=False` the model sees the text content, which carries the same JSON as `structuredContent`. Tracing disabled, no model call, no API key. |

## Tools

Tool names use only letters, digits, and `_` (at most 64 characters),
because OpenAI Agents SDK forwards MCP tool names verbatim as model function
names. Every tool declares an input and an output JSON Schema. Mutations take
an `idempotency_key`: the same key and input return the original result
without creating another version or charging the budget again; the same key
with different input fails with `IDEMPOTENCY_CONFLICT`.

Every tool that names the experiment takes `experiment_id` as an optional
argument: a run-bound server has exactly one experiment, which an omitted ID
means. An ID that is given is used as given, so a mistyped one still fails
with `NOT_FOUND`; before `wse_create_experiment` has run, an omitted ID fails
with `NOT_FOUND` and says to create the experiment first.

| Tool | Inputs | Result |
| --- | --- | --- |
| `wse_list_models` | `cursor` (optional) | One page (50) of model summaries and `next_cursor`; only checkpoints of this run. |
| `wse_create_experiment` | `run_id, task_id, policy_id, idempotency_key` | The experiment description (below). |
| `wse_get_experiment` | `experiment_id` (optional) | The experiment description: state, task definition and `ObjectiveSpec`, selected checkpoint, budget, permitted operations and limits, development suites, checkpoints, `constraint_status`. |
| `wse_inspect_model` | `experiment_id` (optional), `checkpoint_id, cursor` (optional) | The manifest and whole-tensor statistics by layer, from the policy's registered `stats` analysis. The manifest is free; the statistics are charged as compute seconds (not analysis calls), need an OPEN experiment, and stop at the per-operation time limit. A model whose tensors do not fit one result is paged (below). |
| `wse_read_tensor` | `experiment_id` (optional), `checkpoint_id, tensor, selection` | A bounded slice (at most 2,048 elements); charges one `tensor_read_calls`. |
| `wse_analyze` | `experiment_id` (optional), `checkpoint_id, kind, tensor, selection, k` (optional), `parameters` (optional) | A permitted analysis (`stats`, `norm`, `top_k` in M1; `tensor_norms`, `svd_top_k`, and `direction_projection` where the policy adds them), charged under its operator's usage contract. |
| `wse_probe` | `experiment_id` (optional), `checkpoint_id, texts, sites, idempotency_key, groups` (optional), `top` (optional) | The queued probe job, returned immediately (below). |
| `wse_preview_patch` | `experiment_id` (optional), `patch_spec, idempotency_key` | An immutable preview with `patch_id`; the selected checkpoint does not change. Each new preview charges one `patch_previews`, because its deltas reveal the base values of the edited region. |
| `wse_apply_patch` | `experiment_id` (optional), `patch_id, idempotency_key` | The candidate's manifest; charges one `candidate_commits`. |
| `wse_submit_evaluation` | `experiment_id` (optional), `checkpoint_id, suite_id, eval_config, idempotency_key` | The queued job, returned immediately; development suites only. A generation suite's evaluation is a generation pass (below). |
| `wse_get_job` | `experiment_id` (optional), `job_id, wait_seconds` (0-10) | The job; `wait_seconds` holds the call until the job is terminal or the wait expires. A succeeded evaluation carries its report summary (metrics, sample count, errors, `target_met`, report URI) inline and its `report_id`; a succeeded probe carries its summary and `result_uri`. |
| `wse_get_report` | `experiment_id` (optional), `report_id` | A report of the experiment, within the inline-result limit (below); not charged. Final reports only after the experiment is closed. |
| `wse_cancel_job` | `experiment_id` (optional), `job_id, idempotency_key` | The job; a queued job is cancelled and its reservation released, a running job stops at its next check, and work already performed stays charged. |
| `wse_compare` | `experiment_id` (optional), `base_report_id, candidate_report_id` | Paired difference with cluster-bootstrap and exact intervals; two generation reports are paired on accuracy, limit-hit rate, and mean generated tokens. Mismatched reports fail with `INCOMPATIBLE_REPORTS`. |
| `wse_select_checkpoint` | `experiment_id` (optional), `checkpoint_id, expected_selected_id, idempotency_key` | The experiment after a compare-and-swap of the selection; selecting an ancestor is a rollback. |

`selection` is a list of `{"start", "stop", "step"}` objects, one per axis;
only nonnegative, nonempty, in-bounds contiguous slices with step 1 are
accepted, and tensor names must match exactly. `eval_config` is
`{"sample_count", "seed"}` with values preregistered by the suite; the server
fixes the device and batch size.

### Results

A successful call returns `structuredContent` that validates against the
tool's `outputSchema`, and one text content block holding the canonical
compact JSON of exactly the same object (OpenAI Agents SDK shows the model
the text by default). Every result has the same envelope:

```json
{
  "summary": "patch patch_example changes 8 of 82 parameters",
  "uris": ["wse://patches/patch_example"],
  "usage": {
    "call": {"compute_seconds": 0.0021, "artifact_bytes": 1532, "...": "..."},
    "cumulative": {"candidate_commits": {"limit": 20, "consumed": 0, "reserved": 0, "remaining": 20}, "...": "..."}
  },
  "result": {"patch_id": "patch_example", "...": "..."}
}
```

`usage.call` is the usage the domain result records (reads, analyses,
previews, settled jobs) and `null` otherwise; the call's charges are then
visible in `usage.cumulative`, the experiment's ledger. Results that can
fill the inline-result limit list only the counters they charge in
`cumulative`: reads (`tensor_read_calls`, `compute_seconds`), analyses
(their counter and `compute_seconds`), probe jobs (`probe_calls`,
`teacher_forced_tokens`, `compute_seconds`), previews (`patch_previews`,
`compute_seconds`), applied candidates (`candidate_commits`,
`compute_seconds`), and inspections (`compute_seconds`). `wse_get_report`,
which charges nothing, lists none, and `wse_get_experiment` lists every
counter. `wse_list_models` has no experiment and returns `usage: null`.

### Response size

The canonical JSON of a read or analysis result is limited to 32 KiB less a
1 KiB allowance for the envelope and JSON-RPC framing (31 KiB under the
default policy), so the complete response, which carries the result twice,
stays within 64 KiB. Oversized requests fail with `INVALID_ARGUMENT` and
`details.largest_permitted_selection`. Measured through the protocol for a
64x64 FP32 tensor: 2,048 values of 14 characters each give a 31,060-byte
result in a 63,505-byte JSON-RPC response; values of about 1e15 (19-20
characters each) limit the read to 1,536 elements, a 31,386-byte result in a
64,155-byte response.

The envelope (summary, URIs, `usage`, and the framing) stays within that
allowance in both copies, with every number at its widest, because results
that can fill the limit list only the counters they charge (above): about
1.5 to 1.8 KB, where every counter of a language-model policy's ledger
would take about 5 KB.

Language models add results whose JSON is mostly text: output tails, tensor
names and axes, and report keys. The text mirror escapes every quote,
backslash, and control character, so such a result takes more than twice
its JSON's size. The bounded results of the next section therefore measure
both copies (`wse.policy.response_size`) against twice the inline-result
limit (62 KiB under the default policies). M1 results fit whole and keep
their M1 forms.

### Language-model tools

These tools and extensions serve open-weight causal language models (M2
plan Section 5.6). A policy grants generation passes and probes through the
budget counters `generation_passes`, `generated_tokens`, `probe_calls`, and
`teacher_forced_tokens`, with the per-operation limits
`problems_per_generation_pass`, `new_tokens_per_output`,
`texts_per_probe_call`, and `tokens_per_probe_text`; under a policy that
grants none of them, the tools refuse with `POLICY_DENIED`.

**Generation passes.** `wse_submit_evaluation` on a generation suite (its
evaluator is `generation@1`) queues a generation pass: it reserves one
`generation_passes` and problems x samples x `max_new_tokens`
`generated_tokens`, and the worker settles the tokens it generated. A pass
holds at most `problems_per_generation_pass` problems, each output at most
`new_tokens_per_output` tokens; `eval_config` must use a preregistered
`sample_count` and seed of the suite (`POLICY_DENIED` otherwise, before
anything is reserved). Final suites are refused like any final suite.

**`wse_get_report(experiment_id, report_id)`.** The report of a finished
evaluation (`report_id` from `wse_get_job`): identities, metrics, intervals,
per-example `scores` and `cluster_ids`, and, for a generation pass,
`problems[i].samples[j]` with `correct`, `answer`, `generated_tokens`,
`finished`, `limit_hit`, `output_sha256`, `tail` (the end of the output's
text, at most 256 bytes of JSON), and `output_uri` (the whole output, a
run-scoped artifact that `wse_probe` takes). Not charged; read-only. The
result stays within the inline-result limit:

- the report itself when it fits;
- otherwise, for a generation report, the report with every tail shortened
  to the same bound, each still the end of its output (the stored report and
  the resource keep the tails as generated);
- otherwise a `ReportOverview`: every field except the per-example results,
  with `omitted` naming the fields left out (`problems`, `scores`,
  `cluster_ids`), `report_bytes` the whole report's JSON size, `limit_bytes`
  the policy's `max_result_json_bytes`, and `resource_uri` the paged resource
  `wse://reports/{report_id}` that holds the whole report.

A host that cannot read resources (OpenAI Agents SDK exposes them only to
application code) sees per-output results by evaluating fewer problems, a
smaller preregistered `sample_count`, whose report fits, and pairs reports
with `wse_compare`. A report of another experiment fails with `NOT_FOUND`, and
so does a final report until the experiment is closed.

**`wse_probe(experiment_id, checkpoint_id, texts, sites, idempotency_key,
groups=None, top=None)`.** Queues a teacher-forced probe of a checkpoint of
the experiment and returns its job; wait for it with `wse_get_job`.

- `texts` (1 to `texts_per_probe_call`): `{"output_uri": ..., "last": n}`, an
  output of this run's generation passes (`last`, optional, keeps its last
  `n` tokens); `{"text": "..."}`, 1 to 8,192 characters encoded as plain
  text; or `{"suite_id": ..., "item": i}`, an item of a development suite of
  the run's task, as a pass renders its prompt. Each text is cut to its last
  `min(tokens_per_probe_text, 16384)` tokens (16,384 is `MAX_PROBE_TOKENS`,
  the longest forward of one text), and at most the model's context. A
  string must encode to at least 2 tokens, since its window starts after
  its first token.
- `sites` (1 to 128 distinct sites): `"residual:<i>"`, the residual stream
  entering decoder layer `i` (`residual:<L>` is the last layer's output);
  `"<module path>:input"` or `":output"`, where a module path is a tensor
  name without its last part (`model.layers.7.self_attn.o_proj`); or a role
  site of the model's family descriptor, `{"role": "attn.o", "layer": 7,
  "point": "output"}`. A site name has at most 512 characters, and a layer
  (a residual boundary or a role site's layer) at most 9 digits.
- `groups` (optional): `{"a": [...], "b": [...]}`, disjoint text indices with
  at least three texts together; each site then gets the contrast
  `mean(a) - mean(b)`, its norm, and standardized differences.
- `top` (optional, 1-64, default 8): top standardized dimensions per site in
  the summary.

It reserves one `probe_calls`, texts x `min(tokens_per_probe_text, 16384)`
`teacher_forced_tokens` (the most its forwards can read), and the
per-operation compute time; the worker settles the forward lengths it read.
The finished job's `result` is the
inline summary (per text its tokens, window, dropped tokens, and mean
log-probability; per site its features, norms, and top dimensions) with
`result_uri`, the run-scoped artifact of every vector; `wse_get_job` lists
that URI in `uris`. Its vectors are named `<result_uri>#/sites/<i>/mean`,
`#/sites/<i>/group_means/a` (or `/b`), and `#/sites/<i>/contrast/direction`,
with `i` the site's position in `sites`.

Errors of the call, before anything is reserved:

- `POLICY_DENIED`: a policy without probes, or an experiment not in
  `bounded_eval` mode; more texts than `texts_per_probe_call`; a policy
  whose `tokens_per_probe_text` is below 2; an item of a final suite.
- `NOT_FOUND`: a checkpoint the experiment does not own; an output URI this
  experiment did not produce; a suite that does not exist or belongs to
  another task, and an item beyond its `example_count`; a module path that
  names no module (`details.candidates`), and a residual boundary beyond
  the model's layers; a role site on a model without a family descriptor,
  or in a layer without that role (`details.layers`).
- `INVALID_ARGUMENT`: malformed texts, sites (a site name longer than 512
  characters or a layer of more than 9 digits included), groups, or `top`;
  an unknown role, or a layer given or missing against the role; a site
  listed twice; a string that encodes to fewer than 2 tokens; a stored
  result of the run that is not a generation output; an item of a suite
  that is not a generation suite; a summary that could not fit the
  inline-result limit (`details.max_sites`: probe fewer sites or texts).
- `UNSUPPORTED_CAPABILITY`: a model whose adapter cannot be probed.
- `VERSION_CONFLICT`: an experiment that is not `OPEN`;
  `BUDGET_EXCEEDED`: too few probe calls, teacher-forced tokens, or compute
  seconds left for the reservation; `IDEMPOTENCY_CONFLICT`: a key used for
  another request.

Failures in the worker end the job instead (`wse_get_job` shows its
`state` and `error`), with the work already done charged: a site whose
capture is not one vector per position (for example a module that returns
a tuple, receives its input by keyword, runs more than once, or never runs)
or residual sites on a model whose decoder layers cannot be found
(`failed`); an output whose prompt the probed checkpoint renders
differently (`failed`); the per-operation time limit (`failed`,
`details.reason` `"timeout"`); a cancellation (`cancelled`); and a lost
worker (`WORKER_FAILED`).

**Analyses and stored vectors.** `wse_analyze` passes `parameters`, the
operator's own parameters, to the analysis: `{"axis": "rows"}` for
`tensor_norms`, `{"k": 4}` for `svd_top_k`, and `{"direction": [...] or
"<result_uri>#<JSON pointer>", "axis": "columns"}` for `direction_projection`.
Vector results are stored at `result_uri`, with a summary and the top
entries inline. A stored vector, of an analysis or a probe of this
experiment, is named `<result_uri>#<JSON pointer>` (for example
`#/right_vectors/0` or `#/sites/1/contrast/direction`) wherever an operator
takes a direction or factor: in `wse_analyze` parameters and in
`wse_preview_patch` operations (`project_out`'s `direction`, and each column
of `add_low_rank`'s `u` and `v`). A URI of another experiment fails like one
that does not exist.

**Edits.** `wse_preview_patch` takes every edit the policy permits:
`set_slice` (a flat list of values is accepted for a selection with at most
one axis longer than 1, and is recorded in the selection's nesting),
`scale_slice`, `fill_slice` (`value`), `project_out` (`direction`,
`columns`, `alpha`), and `add_low_rank` (`u`, `v`, `alpha`). A run sees an
operator's vector parameters as `{length, sha256}` (and the reference) in
the patch resource. A number is accepted wherever a float32 is expected when
it rounds to a finite float32, so `3.4028235e+38` is.

**Comparisons.** `wse_compare` of two generation reports returns a
`GenerationComparison`: `base` and `candidate` metrics (`accuracy`,
`limit_hit_rate`, `mean_generated_tokens`) and `differences`, each the
candidate-minus-base estimate with a paired cluster-bootstrap interval over
problems, plus `problems`, `samples`, the experiment, both report IDs, and
both checkpoint IDs. Reports of different suites, data, configurations,
sample counts, or prompts fail with `INCOMPATIBLE_REPORTS`. Other reports
give M1's `ComparisonReport`.

**Large models.** A result that would exceed the inline-result limit takes a
bounded form. Inspection pages depend on the model and the policy only; a
preview's form also depends on its patch (texts, operations, and diff
values), and a candidate's view on the tensors its patch changed; none
depends on measured times:

- `wse_preview_patch`: `tensors` lists the diffs of the changed tensors only,
  and `unchanged_tensors` counts the others (absent when every tensor is
  listed). The aggregates cover the whole model. `hypothesis` and
  `expected_behavior` take at most 2,000 characters each. A request whose
  bounded preview, or the bounded view of the candidate it would create,
  could exceed the limit is refused with `INVALID_PATCH` before anything is
  materialized or charged (`details`: `limit_bytes`, `preview_bound_bytes`,
  `candidate_bound_bytes`). The bound comes from the request and the
  manifest alone: every tensor an operation names counts as changed and
  every number takes its widest form. Only many operations on tensors with
  long names together with long texts full of characters that JSON escapes
  (quotes, backslashes, control characters) reach it: on a 321-tensor
  layout, 32 operations with 2,000-character plain texts fit, and texts of
  2,000 quotes do not. Propose fewer operations per patch or shorten the
  texts.
- `wse_inspect_model`: the inspection is paged. `manifest.tensors` and
  `tensor_statistics` hold one page's tensors, `tensor_page` (`start`,
  `stop`, `tensor_count`, `page`, `page_count`, `next_cursor`) says which,
  and `cursor` set to `next_cursor` returns the next page; only that page's
  statistics are computed and charged. Any other cursor fails with
  `INVALID_ARGUMENT`.
- `wse_apply_patch`: the candidate's manifest lists the tensors whose
  content differs from the parent checkpoint, and `unchanged_tensors` counts
  the others.
- `wse_get_report`: as above.

The whole documents stay available as resources (`wse://models/{id}/manifest`,
`wse://patches/{id}`, `wse://reports/{id}`).

Measured on 2026-11-10 through the protocol (`mcp.Client` against
`wse serve`, PyTorch MPS) on Qwen3.5-0.8B, an open-weight model with 321
tensors, under a policy with the default 32 KiB inline limit, when every
result listed all counters in `cumulative` (its responses are now about 1.5
to 3 KB smaller):

| Result | Whole form (JSON) | Bounded form (JSON) | Largest response |
| --- | --- | --- | --- |
| Inspection | 200,275 bytes | 9 pages of at most 28,415 bytes | 62,984 bytes |
| Preview of one `project_out` | 127,947 bytes | 1,958 bytes (1 tensor listed, 320 counted) | 7,245 bytes |
| Candidate manifest | 97,725 bytes | 4,826 bytes (1 tensor listed, 320 counted) | 13,006 bytes |
| Report of a 2-problem pass at 256 tokens | 5,100 bytes (whole) | | 11,655 bytes |

On the test fixtures, a 32-problem pass whose answers reach the graders'
200-character limit and whose tails are backslashes and newlines stores a
31,727-byte report, whose two copies take 66,950 bytes; the tool shortens
its tails evenly to a 64,323-byte response. A probe summary filled to its
allowance (64 texts at up to 128 sites) gives at most a 64,439-byte
`wse_get_job` response.

### Errors

A domain failure is an MCP tool error (`isError: true`) whose text is
`Error executing tool <name>: ` followed by the canonical JSON of the domain
error:

```json
{"error": {"code": "VERSION_CONFLICT",
           "message": "the selected checkpoint is ckpt_example_b, not 'ckpt_example_a'; read the experiment and retry with the current selected ID",
           "retryable": false,
           "details": {"expected_selected_id": "ckpt_example_a", "selected_checkpoint_id": "ckpt_example_b"}}}
```

Codes: `NOT_FOUND`, `POLICY_DENIED`, `INVALID_PATCH`, `BUDGET_EXCEEDED`,
`VERSION_CONFLICT`, `IDEMPOTENCY_CONFLICT`, `INCOMPATIBLE_REPORTS`,
`WORKER_FAILED`, `INVALID_ARGUMENT`, `INVALID_MODEL`, `ARTIFACT_CORRUPT`, and
`UNSUPPORTED_CAPABILITY`. Arguments that fail a tool's input schema, and
unknown tool names, are reported by the SDK as tool errors without the JSON
body. Protocol failures remain JSON-RPC errors.

## Resources

| URI template | Content |
| --- | --- |
| `wse://models/{checkpoint_id}/manifest{?page}` | A checkpoint's manifest. |
| `wse://experiments/{experiment_id}/policy{?page}` | The experiment's immutable policy. |
| `wse://reports/{report_id}{?page}` | An evaluation report, whole; final reports only after the experiment is closed. A `ReportOverview` from `wse_get_report` names it as `resource_uri`. |
| `wse://patches/{patch_id}{?page}` | `{"preview", "spec", "steps", "result_weights_hash"}`: the preview, the frozen spec, and per-step summaries (tensor, selection, before/after hashes, changed elements). The materialized after-values are omitted because they reproduce base weights; the exact patch is an operator view (`wse patch`). |
| `wse://analyses/{digest}{?page}` | A large analysis result referenced by an analysis `result_uri`; readable only by the run whose experiment produced it. |

Resource reads use the same bound Engine, and so the same authorization, as
tools. A failed read is a JSON-RPC error whose message is the domain error
JSON: `-32602` (invalid params) for `NOT_FOUND`, `-32603` otherwise.

Reads are bounded to 32 KiB of JSON. A document that fits is returned as
`{"uri", "page": 0, "page_count": 1, "next_uri": null, "content": {...}}`. A
larger document is returned in pages `{"uri", "page", "page_count",
"next_uri", "chunk"}`: follow `next_uri` (`?page=N`) and concatenate the
chunks to obtain the document's JSON. Tools carry inline everything a
controller needs, because OpenAI Agents SDK exposes resources only to
application code; resources are an optional view. The one exception is the
per-example part of a report too large for one result, whose tool view is a
`ReportOverview` (see `wse_get_report`).

## Example session

Calls are shown as tool name and arguments; results are abbreviated.

1. `wse_create_experiment {"run_id": "run_example", "task_id": "example_task_v1", "policy_id": "small_model_v1", "idempotency_key": "create-1"}`
   returns the experiment description with `experiment.experiment_id = "exp_example"`,
   `experiment.selected_checkpoint_id = "ckpt_example_base"`, the task and its objective,
   `operations` (`stats`, `norm`, `top_k`; `set_slice`, `scale_slice`) and `suites` (`example_dev_v1`).
2. `wse_inspect_model {"experiment_id": "exp_example", "checkpoint_id": "ckpt_example_base"}`
   returns the manifest (tensors `layers.0.weight` `[16, 8]`, ...) and per-tensor statistics.
3. `wse_read_tensor {"experiment_id": "exp_example", "checkpoint_id": "ckpt_example_base", "tensor": "layers.0.weight", "selection": [{"start": 3, "stop": 4, "step": 1}, {"start": 0, "stop": 8, "step": 1}]}`
   returns 8 values and charges one tensor read.
4. `wse_preview_patch` with the patch spec

   ```json
   {
     "base_checkpoint_id": "ckpt_example_base",
     "hypothesis": "Hidden unit 3 dominates the output for inputs outside the training range.",
     "expected_behavior": "Halve that unit's input weights while keeping development accuracy unchanged.",
     "development_suite_id": "example_dev_v1",
     "operations": [
       {
         "op": "scale_slice",
         "tensor": "layers.0.weight",
         "selection": [{"start": 3, "stop": 4, "step": 1}, {"start": 0, "stop": 8, "step": 1}],
         "factor": 0.5
       }
     ]
   }
   ```

   and `"idempotency_key": "preview-1"` returns `patch_id = "patch_example"`, the per-tensor diff,
   and the changed-element count. The server resolves the base content hash itself; do not send it.
5. `wse_apply_patch {"experiment_id": "exp_example", "patch_id": "patch_example", "idempotency_key": "apply-1"}`
   returns the candidate manifest `ckpt_example_candidate`.
6. `wse_submit_evaluation {"experiment_id": "exp_example", "checkpoint_id": "ckpt_example_candidate", "suite_id": "example_dev_v1", "eval_config": {"sample_count": 64, "seed": 0}, "idempotency_key": "eval-1"}`
   returns a queued job; `wse_get_job {"experiment_id": "exp_example", "job_id": "job_example", "wait_seconds": 10}`
   returns it once succeeded, with its metrics and `report_id` inline.
7. `wse_compare` pairs the base and candidate reports; `wse_select_checkpoint {"experiment_id": "exp_example", "checkpoint_id": "ckpt_example_candidate", "expected_selected_id": "ckpt_example_base", "idempotency_key": "select-1"}`
   selects the candidate, and selecting `ckpt_example_base` again (with the candidate as the expected ID) rolls back.

Freezing and the final evaluation are operator actions (`wse freeze`,
`wse final-evaluate`) performed after the controller stops.

## Example session: a language model

The operator has imported Llama-3.2-1B-Instruct (`ckpt_example_llama`; 16
decoder layers, hidden size 2,048, 32 attention heads of 64 dimensions, so
`self_attn.o_proj` weights `[2048, 2048]`),
registered the task `example_math_v1` with the generation suite
`example_math_dev_v1` (16 arithmetic word problems, the model's default
profile `llama_3_2_1b_instruct_default@1`, the grader `gsm8k_numeric@1`, 1
sample per problem, 1,024 new tokens, preregistered `sample_count` 16 and 8,
seed 0), registered the policy `example_lm_v1`, and granted `run_example_lm`.
`experiment_id` is omitted after the first call.

1. `wse_create_experiment {"run_id": "run_example_lm", "task_id": "example_math_v1", "policy_id": "example_lm_v1", "idempotency_key": "create-1"}`
   returns the description; its budget lists `generation_passes`,
   `generated_tokens`, `probe_calls`, and `teacher_forced_tokens`.
2. `wse_inspect_model {"checkpoint_id": "ckpt_example_llama"}` returns the
   first page of the manifest's tensors and their statistics, with a
   `tensor_page.next_cursor` such as `"tensors:48"`; `wse_inspect_model
   {"checkpoint_id": "ckpt_example_llama", "cursor": "tensors:48"}` returns
   the next page.
3. `wse_submit_evaluation {"checkpoint_id": "ckpt_example_llama", "suite_id": "example_math_dev_v1", "eval_config": {"sample_count": 16, "seed": 0}, "idempotency_key": "pass-1"}`
   queues a generation pass (reserved: 1 pass and 16,384 generated tokens);
   `wse_get_job {"job_id": "job_example_pass", "wait_seconds": 10}` returns it
   once succeeded, with `report_id`.
4. `wse_get_report {"report_id": "report_example_pass"}` returns the report;
   suppose the answers of samples 2, 5, and 11 are wrong (`correct` false)
   and the others right.
5. `wse_probe` with

   ```json
   {
     "checkpoint_id": "ckpt_example_llama",
     "texts": [{"output_uri": "wse://analyses/<sample 2>"}, {"output_uri": "wse://analyses/<sample 5>"},
               {"output_uri": "wse://analyses/<sample 11>"}, {"output_uri": "wse://analyses/<sample 0>"},
               {"output_uri": "wse://analyses/<sample 1>"}, {"output_uri": "wse://analyses/<sample 3>"}],
     "sites": ["residual:8", {"role": "attn.o", "layer": 7, "point": "output"}],
     "groups": {"a": [0, 1, 2], "b": [3, 4, 5]},
     "idempotency_key": "probe-1"
   }
   ```

   queues a probe; `wse_get_job` returns its summary with `result_uri`
   `wse://analyses/<probe>`. The contrast at the output of layer 7's
   attention projection `self_attn.o_proj` has 2,048 values, one per output
   row: `wse://analyses/<probe>#/sites/1/contrast/direction`.
6. `wse_analyze {"checkpoint_id": "ckpt_example_llama", "kind": "direction_projection", "tensor": "model.layers.7.self_attn.o_proj.weight", "selection": [{"start": 0, "stop": 2048, "step": 1}, {"start": 0, "stop": 2048, "step": 1}], "parameters": {"direction": "wse://analyses/<probe>#/sites/1/contrast/direction", "axis": "columns"}}`
   returns the columns most aligned with the contrast in `result.top`.
7. `wse_preview_patch` with the patch spec

   ```json
   {
     "base_checkpoint_id": "ckpt_example_llama",
     "hypothesis": "Outputs with wrong answers differ from right ones along this direction at layer 7's attention output.",
     "operations": [
       {
         "op": "project_out",
         "tensor": "model.layers.7.self_attn.o_proj.weight",
         "selection": [{"start": 0, "stop": 2048, "step": 1}, {"start": 0, "stop": 2048, "step": 1}],
         "direction": "wse://analyses/<probe>#/sites/1/contrast/direction",
         "columns": [118, 942, 1305, 1911]
       }
     ]
   }
   ```

   and `"idempotency_key": "preview-1"` returns a preview whose `tensors`
   lists the one changed tensor and whose `unchanged_tensors` counts the
   rest; `wse_apply_patch` returns the candidate's manifest in the same
   bounded form.
8. A pass on the candidate and `wse_compare {"base_report_id": "report_example_pass", "candidate_report_id": "report_example_candidate"}`
   give the paired differences in accuracy, limit-hit rate, and mean
   generated tokens.
