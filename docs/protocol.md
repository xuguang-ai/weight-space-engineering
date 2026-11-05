# WSE MCP protocol

This document describes the MCP interface of the weight-space engineering
framework (`wse serve`, implemented in `src/wse/mcp_server.py`): its tools,
resources, result and error formats, and an example session. Every example
uses a hypothetical 8-input MLP with the task `example_task_v1` and the
development suite `example_dev_v1`; no example describes a model or task used
in an agent trial. IDs are illustrative; real IDs are opaque strings returned
by the server.

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

| Tool | Inputs | Result |
| --- | --- | --- |
| `wse_list_models` | `cursor` (optional) | One page (50) of model summaries and `next_cursor`; only checkpoints of this run. |
| `wse_create_experiment` | `run_id, task_id, policy_id, idempotency_key` | The experiment description (below). |
| `wse_get_experiment` | `experiment_id` | The experiment description: state, task definition and `ObjectiveSpec`, selected checkpoint, budget, permitted operations and limits, development suites, checkpoints, `constraint_status`. |
| `wse_inspect_model` | `experiment_id, checkpoint_id` | The manifest and whole-tensor statistics by layer, from the policy's registered `stats` analysis. The manifest is free; the statistics are charged as compute seconds (not analysis calls), need an OPEN experiment, and stop at the per-operation time limit. |
| `wse_read_tensor` | `experiment_id, checkpoint_id, tensor, selection` | A bounded slice (at most 2,048 elements); charges one `tensor_read_calls`. |
| `wse_analyze` | `experiment_id, checkpoint_id, kind, tensor, selection, k` (optional) | A permitted analysis (`stats`, `norm`, `top_k` in M1), charged under its operator's usage contract. |
| `wse_preview_patch` | `experiment_id, patch_spec, idempotency_key` | An immutable preview with `patch_id`; the selected checkpoint does not change. Each new preview charges one `patch_previews`, because its deltas reveal the base values of the edited region. |
| `wse_apply_patch` | `experiment_id, patch_id, idempotency_key` | The candidate's manifest; charges one `candidate_commits`. |
| `wse_submit_evaluation` | `experiment_id, checkpoint_id, suite_id, eval_config, idempotency_key` | The queued job, returned immediately; development suites only. |
| `wse_get_job` | `experiment_id, job_id, wait_seconds` (0-10) | The job; `wait_seconds` holds the call until the job is terminal or the wait expires. A succeeded job carries its report summary (metrics, sample count, errors, `target_met`, report URI) inline. |
| `wse_cancel_job` | `experiment_id, job_id, idempotency_key` | The job; a queued job is cancelled and its reservation released, a running job stops at its next check, and work already performed stays charged. |
| `wse_compare` | `experiment_id, base_report_id, candidate_report_id` | Paired difference with cluster-bootstrap and exact intervals; mismatched reports fail with `INCOMPATIBLE_REPORTS`. |
| `wse_select_checkpoint` | `experiment_id, checkpoint_id, expected_selected_id, idempotency_key` | The experiment after a compare-and-swap of the selection; selecting an ancestor is a rollback. |

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
visible in `usage.cumulative`, the experiment's ledger. Read and analysis
results list only the counters they charge in `cumulative`.
`wse_list_models` has no experiment and returns `usage: null`.

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
| `wse://reports/{report_id}{?page}` | An evaluation report; final reports only after the experiment is closed. |
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
application code; resources are an optional view.

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
