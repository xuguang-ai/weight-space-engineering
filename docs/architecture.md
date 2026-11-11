# Architecture

How WSE's parts fit together, for readers who want to extend WSE or contribute to it. To use WSE, start with the [README](../README.md); [experiments.md](experiments.md) covers managed experiments in detail.

## Layers

| Layer | What it does | Entry points |
| --- | --- | --- |
| Core | Works in memory on a model handle: loads, analyzes, probes, edits, evaluates, and replays, without a database, service, or MCP. Its results are labeled `unmanaged`. | `wse.load_model`, `wse.apply_operator`, `wse.evaluate_candidate`, `wse.replay_patch` |
| Engine | Adds a workspace with SQLite metadata and content-addressed files: imports, stored checkpoints, and managed experiments with run grants, policies, budgets, audit records, final evaluation, export, and replay. | `wse.Engine` |
| Worker | A separate process that claims, runs, meters, and settles jobs (evaluations, generation passes, and probes), one at a time. It runs registered evaluators over registered adapters only, never code an agent wrote. | `wse worker`, or the job scheduler that `wse serve` and the runner start |
| MCP server | A thin front end to the Engine for one run grant: each tool validates its arguments against the published schema and calls one Engine operation. | `wse serve --run-id RUN` |
| Runner | Grants a run, runs one controller (a registered strategy, or an agent connected to `wse serve`), settles every outstanding job, and freezes the selected checkpoint. | `wse run`, `wse.agent_runs.run_agent_experiment` |
| `WSE` | The high-level Python entry point for open-weight language models, over the core and the Engine. | `wse.WSE` |
| CLI | Imports, run grants, serving, runs, final evaluation, export, replay, and reports, for operators. | `wse` |

No MCP handler contains an editing, analysis, budget, or evaluation algorithm. The Python API, the CLI, and MCP therefore share the same operations, policy decisions, and usage accounting.

## A managed experiment

1. **Setup (operator).** Import the target checkpoint and register the task, the policy, and the development and final suites. The final suite's data lives only in the evaluator's storage.
2. **Run grant.** A grant permits one experiment under one policy on one base checkpoint.
3. **Controller.** The controller works only through the run-bound interface. Every operation is checked against the policy, charged to the budget before it runs, and recorded in the audit log, refusals included.
4. **Freeze.** When the controller stops, the runner cancels or settles every outstanding job and freezes the selected checkpoint.
5. **Final evaluation.** The frozen candidate and the unedited base are evaluated on the same final examples, which the controller never saw; then the experiment closes.
6. **Export and replay.** An export holds the base model, every patch, report, and audit event, and the controller's trace; replay rebuilds every candidate in a clean workspace and checks every hash.

[experiments.md](experiments.md#workflow) gives the commands for each step.

## Registry and versioned contracts

Extensions are registered by kind, name, and version, with `registry.register(kind, extension_id, version, implementation)`; a duplicate registration is rejected. The kinds are model adapters, family descriptors (per-family name maps, roles, and protected tensors), edit operators, evaluators, graders, sampling profiles, and strategies. Agent adapters are named and versioned too, such as the reference agent's `openai_agents@1`.

Every record names the exact version it used, such as `scale_slice@1` or `qwen35think@1`, and a recipe is re-executed only with the exact versions it names, never with the latest. Registering an extension does not permit it: a policy maps public names to exact versions, and the server refuses everything else. A setup hook (`--setup module:function` on `wse serve`, `wse worker`, and `wse run`) registers extensions in each process. [extensions.md](extensions.md) specifies each contract.

## Exact patches and storage

A patch records, for every step, the operator and its version, the selection, the after-values in the stored dtype, and the tensor hashes before and after. Replay writes those values without running the operator and verifies every hash, so it reproduces the recorded weights byte for byte or fails. Re-executing a recipe, which runs the operators again, is a separate path that yields a new candidate needing fresh evaluation ([exact replay and recipes](extensions.md#exact-patch-replay-and-recipe-re-execution)).

Edit arithmetic is platform-independent: exact products, correctly rounded sums, and round-to-nearest-even to BF16, F16, or F32. So far this has been checked on macOS only.

Tensors are stored as content-addressed blobs, one per tensor, each named by the SHA-256 of its bytes in the stored dtype. A checkpoint maps tensor names to blobs, dtypes, and shapes, so a candidate stores only the tensors that differ from its parent, and identical tensors share one blob. Files are written, synced, and published by an atomic rename, and become visible only when the SQLite transaction that references them commits. WSE's weights hash covers each tensor's name, dtype, and shape as well as its bytes.

## Models

Causal language models that `transformers` implements natively, as unquantized safetensors, need no model-specific code when the checkpoint's tensor names match the module (Level 0). Family descriptors, which are registered data, map other names and add semantic roles and protected tensors (Level 1). The importer reads only allow-listed files: it never unpickles or runs repository code, and it refuses `trust_remote_code` architectures. [models.md](models.md) covers support levels, importing, and roles.

## What carries over

Each part of an experiment can change while the rest stays in place:

| To change | You write | WSE keeps | Shown so far |
| --- | --- | --- | --- |
| The agent | A model name for the reference agent, a registered strategy, or your own MCP client | The tools, policy checks, budgets, audit records, final evaluation, and export | A scripted strategy, the SDK agent driven by a test double, and `gpt-5.6-luna` ran the same signed-sum task |
| The edit method | An operator registered by name and version, then permitted by a policy | The patch format, hashes, exact replay, preview and commit tools, and evaluation | `project_out@1` ran from Python and through MCP on Qwen3.5-0.8B |
| The target model | Nothing for a causal model that `transformers` implements natively and whose tensor names match; a family descriptor adds roles and protections | Import checks, analyses, probes, generation evaluation, and patches | Imports of Qwen3.5-0.8B, Llama-3.2-1B-Instruct, Qwen3-0.6B, and gemma-3-1b-it verified |
| The task | A task statement and objective, evaluation suites, and a policy | The runner, MCP server, budget ledger, paired comparison, and records | Signed-sum classification, and finishing math answers on Qwen3.5-0.8B |

No developer outside the project has added a method or a task yet, so the cost of an extension has not been measured.

## Where the code is

| Module | Role |
| --- | --- |
| `api.py` | Direct library functions and the managed `Engine` |
| `highlevel.py` | `WSE`, one object per model for the four steps of an experiment |
| `experiments.py`, `budget.py`, `policy.py` | The experiment state machine, atomic budget reservations, and policies |
| `jobs.py`, `worker.py` | The job queue and the worker that runs jobs |
| `mcp_server.py` | The MCP tools, resources, and protocol mapping |
| `runner.py`, `agent_runs.py` | The reference runner and the one-call experiment |
| `registry.py` | Explicit, versioned extension registration |
| `patches.py`, `patch_blocks.py`, `external_patches.py` | Materialized patches and exact replay, binary patch files, and imports of published edits |
| `store.py`, `database.py`, `tensors.py` | Content-addressed storage, the SQLite schema, and tensor hashing and selections |
| `audit.py` | Audit events, controller records, and exports |
| `analysis.py`, `probes.py` | Bounded tensor reads and analyses, and activation probes |
| `hf_snapshot.py` | Allow-listed reading of Hugging Face snapshots |
| `adapters/`, `engines/` | Model adapters and family descriptors, and the PyTorch inference engine for imported causal LMs |
| `operators/`, `evaluators/`, `strategies/` | Built-in edit operators and analyses, evaluators and graders, and stepwise controllers |
| `agent_adapters/` | The vendor-neutral agent adapter contract and the OpenAI Agents SDK adapter |

All paths are under [src/wse/](../src/wse).
