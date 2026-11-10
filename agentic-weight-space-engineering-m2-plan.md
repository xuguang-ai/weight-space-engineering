# Weight-Space Engineering M2: Open-Weight Language Models, First Target Qwen3.5-0.8B

> **For agentic workers:** Use the `subagent-driven-development` skill during implementation, delivering and reviewing one task at a time. Track execution with the checkboxes below. Steps marked **(operator)** are the user's own actions: installs, downloads, access requests, and live runs that cost money. Implementation started on 2026-11-09, and the checkboxes record its progress. This document, like the M1 plan, names the oracle edits of the Qwen trial, so the reference runner must keep it out of every agent's reach.

**Goal:** Let users and agents inspect, edit, evaluate, and replay changes to the weights of open-weight causal language models through the same managed contracts as M1. Qwen3.5-0.8B is the first target. A widely used model from another vendor, Llama-3.2-1B-Instruct, must work through the same generic path without new code.

**Architecture:** A generic Hugging Face causal-LM adapter, `hf_causal_lm@1`, imports any architecture that the pinned `transformers` implements natively. It derives tensor axes from module types and runs forward passes and generation through the `transformers` implementation inside the WSE worker. Small registered family descriptors add only what generic code cannot infer: checkpoint name mappings, semantic roles such as attention heads and MLP projections, and default protections. Storage, patches, evaluators, budgets, and MCP tools stay model-agnostic.

**Tech Stack:** M1's stack (Python 3.14, MCP Python SDK 2.3.0, PyTorch 2.15, NumPy, Pydantic, safetensors, SQLite, pytest, uv), plus an optional `models-hf` extra with `transformers` and `math-verify`. Execution starts on the operator's Mac (Apple M4 Pro, 48 GB, 20-core GPU) with PyTorch MPS and a CPU fallback.

**Date and status:** Created 2026-11-09, after M1 closed. On 2026-11-09 the user set the direction:

- practice on Qwen3.5-0.8B as soon as possible, without limiting the framework to Qwen;
- first replay the article's published edits, then build agent tools, then run agent trials;
- evaluate locally with PyTorch MPS first;
- give agents only declarative, metered WSE tools;
- use Llama-3.2-1B-Instruct as the second model;
- defer the Claude Agent SDK adapter and the two additional small-model fault classes until after the first Qwen agent trials.

This plan replaces the M2 row of the [M1 plan](./agentic-weight-space-engineering-plan.md)'s Section 10 and brings its "registered pretrained model" deliverable forward from M3. The reference case is experiment 8 of [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), whose [logs](https://github.com/RikiyaT/lm_understand-logs) publish bit-exact edits of Qwen3.5-0.8B. The [article summary](./weight-space-programming-summary.md) reports its results.

## Global Constraints

- Every M1 global constraint still holds. In particular:
  - core modules import neither an agent SDK nor `transformers`;
  - every managed computation is governed by policy and budgets;
  - development feedback stays isolated from final evaluation;
  - toolchain correctness is verified separately from agent capability.
- **Generic before specific.** A capability that generic code can provide for every supported model belongs in the generic adapter, tools, or evaluators. A family descriptor holds only what generic code cannot infer. Adding a model family adds a registered descriptor and changes no core dispatch, storage, MCP handler, or evaluator.
- **Supported models in M2.** M2 supports causal language models that meet three conditions:
  - the pinned `transformers` implements the architecture natively;
  - the weights are safetensors in BF16, F16, or F32;
  - the model fits the host's memory.

  Excluded in M2:
  - architectures that need `trust_remote_code`;
  - quantized formats such as GGUF, GPTQ, AWQ, and MXFP4;
  - encoder–decoder models;
  - per-expert semantics of mixture-of-experts models, whose tensors stay readable and editable through generic tools.
- **Safe import.** The importer reads only allow-listed files: `config.json`, `generation_config.json`, `*.safetensors` and its index, tokenizer files, and the chat template. It never unpickles a file and never executes repository code, and it rejects a `config.json` that contains `auto_map`.
- **Declarative, metered tools only.** Agents get no Python, shell, gradients, or training. Every forward pass, generated token, probe, and analysis is charged to the run's budget.
- **Weights, datasets, and credentials stay outside the repository and its GitHub remote.** Model snapshots and datasets live in operator storage, by default `~/Desktop/wse-models/`. An access token for a gated model is entered at a silent prompt for the download command only. WSE never reads or stores it.
- **The M1 MLP path keeps working.** All 366 M1 tests pass at the end of every task.

## 1. Sequencing and Roadmap Change

| Stage | Purpose | Exit condition |
| --- | --- | --- |
| **Q1: Open-weight models load, replay, and evaluate** | Generic import, per-tensor storage, the PyTorch engine, BF16 patches, exact replay of the published Qwen edits, generation evaluation, and measured local throughput | All three published Qwen3.5-0.8B edits replay to their published SHA-256 hashes. A local base-versus-p50c comparison moves in the article's direction. Llama-3.2-1B-Instruct imports, passes logits parity, round-trips a patch, and completes an evaluation with no new code. |
| **Q2: Agent-usable tools** | Family descriptors; direction, low-rank, and fill edits; large-tensor analyses; activation probes; generation passes; budgets; policy; and MCP tools | The known repair, expressed through the tools with a direction recovered from the published diff, lowers the development limit-hit rate. A scripted discovery attempt that uses only the tools is reported. Qwen3-0.6B and Gemma-3-1b-it pass the generic smoke checks with no new code. |
| **Q3: Agent trials on Qwen3.5-0.8B** | A preregistered protocol, controls, and five agent trials | All five trials and the controls are reported, as in M1. |

How this maps to the M1 roadmap:

- **M2 deliverables kept on this path:** a second model family and task (Qwen3.5 and Llama 3.2), Transformer diagnostics (probes and analyses), low-rank and directional edits, and independent replay of an experiment (the article's edits).
- **Brought forward from M3:** a registered pretrained model, and CPU editing with accelerator inference.
- **Deferred until after Q3:** the Claude Agent SDK adapter, with its condition "Claude connects without engine changes", and validation of two additional fault classes on small models.
- **Still M3 or later:** GPU workers and vLLM or MLX inference backends, unless the Task 7 decision gate brings one forward; sharded storage beyond per-tensor blobs; workflows beyond loop repair; and a sandboxed research-Python mode.

## 2. Model Support Levels

### 2.1 Level 0: the generic adapter `hf_causal_lm@1`

Capabilities: `weights.read`, `weights.slice_edit`, `weights.direction_edit`, `forward.logits`, `generate.text`, and `probe.hidden_states`.

- **Import.** The importer takes a local snapshot directory with its source repository and revision, records the SHA-256 of every file it reads, and selects the language-model tensors. At Level 0 the checkpoint names must equal the parameter names of the `transformers` module.
- **Identity.** A model's identity comprises:
  - `weights_hash`, over per-tensor blob IDs in canonical name order;
  - `config_hash`;
  - `tokenizer_hash`, over every tokenizer file present that `transformers` reads: `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`, `tokenizer.model`, `vocab.json`, and `merges.txt`;
  - `chat_template_hash`;
  - `generation_config_hash`;
  - the source repository and revision.

  The M1 plan's Section 3 required this for the LLM phase.
- **Axes come from module types, never from tensor names.** The adapter instantiates the module on the meta device and maps each parameter:
  - an `nn.Linear` weight is `[out_features, in_features]`, and its bias is `[out_features]`;
  - an `nn.Embedding` weight is `[num_embeddings, embedding_dim]`;
  - an `nn.Conv1d` weight is `[out_channels, in_channels/groups, kernel]`;
  - a one-dimensional parameter of a normalization module is `[features]`;
  - any other parameter has axes `unknown`. It stays readable, and it accepts only shape-checked elementwise edits.
- **Alias groups.** Parameters that are the same object in the instantiated module, such as tied input and output embeddings, form one alias group, stored as one blob. The default policies protect alias groups from edits. Whole-group edit semantics remain deferred (M1 plan Section 5.2).
- **Forward and generation.** The engine builds the module with `from_config` on the meta device, loads WSE tensors with `load_state_dict(..., assign=True)`, and runs on `mps`, falling back to `cpu`. In managed experiments only the worker runs it.
- **Stored dtypes are kept.** Every tensor runs in its stored dtype. This differs from `from_pretrained(dtype="auto")`:
  - That call casts Qwen3.5-0.8B's 36 F32 tensors (`A_log` and `linear_attn.norm.weight` in its 18 linear-attention layers) to BF16, which moves logits by up to 0.53.
  - With those tensors kept in F32, `from_pretrained` matches WSE bitwise on CPU and MPS (Task 4), so parity checks use that reference.
  - Results can therefore differ slightly from tools that cast these tensors.
- **Hidden states.** The adapter reads the residual stream at layer boundaries through `output_hidden_states=True`.

### 2.2 Level 1: family descriptors

A family descriptor is a registered extension of the new kind `family`, selected at import by `config.json`'s `model_type`:

```python
class PrefixRule(BaseModel):
    checkpoint_prefix: str            # "model.language_model."
    module_prefix: str | None         # "model."; None drops matching tensors

class RoleRule(BaseModel):
    pattern: str                      # regex over canonical tensor names, with a named group "layer"
    role: str                         # "mlp.down", "attn.o", ...
    heads: HeadGrouping | None = None # num_heads and head_dim from config keys, and the axis that indexes heads

class FamilyDescriptor(BaseModel):
    family_id: str                    # "qwen3_5", "llama"
    version: str
    model_types: list[str]            # accepted config.json model_type values
    text_config_key: str | None       # "text_config" for multimodal wrappers
    causal_lm_class: str | None       # e.g. "Qwen3_5ForCausalLM" when the checkpoint's class is multimodal
    name_map: list[PrefixRule]
    roles: list[RoleRule] = []
    protected_roles: list[str] = []   # protected by the family's default policies
```

- **Roles.** `embed`, `lm_head`, `final_norm`, `attn.q`, `attn.k`, `attn.v`, `attn.o`, `linear_attn.in`, `linear_attn.out`, `mlp.gate`, `mlp.up`, `mlp.down`, `norm.input`, and `norm.post_attention`. Head grouping takes `num_heads` and `head_dim` from config keys and names the axis that indexes heads, for example the input axis of `attn.o`.
- **Built-in descriptors.**
  - `qwen3_5@1`: its name map comes in Q1 and its roles in Q2.
  - `llama@1`: roles only, in Q2. Llama checkpoints load at Level 0 without it.
  - The Q2 smoke models need no descriptor.

### 2.3 What a user does to add a model

| Situation | Work |
| --- | --- |
| `transformers` implements the architecture natively, and checkpoint names match its modules | None: import the snapshot (Level 0). |
| Names need a mapping, or structure-aware tools are wanted | Register a family descriptor: data only, with no core changes. |
| `transformers` does not implement the architecture | Register a new model adapter (code), as M1's extension contract allows. |

[docs/models.md](./docs/models.md) documents these levels for users (Task 12).

## 3. Contracts Added or Changed

### 3.1 Per-tensor content-addressed storage

- **Blob.** A tensor blob holds one tensor's raw bytes in its stored dtype, little-endian and row-major. Its ID is the SHA-256 of those bytes, the convention of the article's `apply_edit.py`, so published hashes compare directly:

  ```python
  def blob_id(t: torch.Tensor) -> str:
      raw = t.detach().contiguous().cpu().view(torch.uint8).numpy()  # native little-endian on arm64 and x86-64
      return hashlib.sha256(raw.tobytes()).hexdigest()
  ```

- **Manifest.** A checkpoint manifest maps canonical tensor names to blob IDs, shapes, dtypes, axes, roles, and alias groups.
- **Candidates.** A candidate stores only the blobs that differ from its parent. A p50c candidate adds 2 × 1,024 × 3,584 × 2 bytes ≈ 14.7 MB instead of a full copy of the roughly 1.5 GB text model.
- **Migration 8.** M1 ends at schema version 7. Migration 8 moves M1 checkpoints to blob manifests in place, and M1's identities (`weights_hash` and manifest fields) must not change.
- **Export.** Export writes a safetensors file under the canonical names.

### 3.2 Stored dtypes and edits

- Stored dtypes are BF16, F16, and F32.
- Edits compute in FP32 and write back with round-to-nearest-even. Changed elements are counted after write-back, as in M1.
- Re-execution of a low-rank recipe may accumulate in FP64, for parity with the article's p10c rebuild.
- In JSON, a stored value appears as the shortest decimal that round-trips its float32 widening. In binary artifacts it appears as the bit pattern of its stored dtype.

### 3.3 Binary patch artifacts

- **Blocks.** For each tensor, a materialized patch step holds a sparse block (flat row-major `indices` as uint64 and `values` as stored-dtype bit patterns) or a dense block for a contiguous slice, together with the base and result blob IDs.
- **Views and replay.** Run-facing JSON views summarize each block and omit values, as M1 does for after-values. Replay reads the binary artifact.
- **The M1 limit of 65,536 elements per patch.** It still applies to values that an agent supplies inline. It does not apply to operator results: p50c touches 131,072 elements.

### 3.4 External patch import

`wse import-patch --format lm-understand-v1 EDIT_DIR --base CHECKPOINT_ID` reads `manifest.json` and `edit.npz`, verifies each `base_sha256`, rebuilds each tensor exactly as the article's `apply_edit.py` does, verifies each `edited_sha256`, and records an exact WSE patch. The patch's provenance names the source repository, run, and file hashes.

```python
def rebuild(base_bits: np.ndarray, entry: dict, z: np.lib.npyio.NpzFile, i: int) -> np.ndarray:
    """base_bits: uint16 bit patterns of a BF16 tensor; returns the edited bit patterns."""
    if entry["form"] == "lowrank":
        w = bf16_to_f32(base_bits).astype(np.float64)
        a, b = z[f"t{i}_a"], z[f"t{i}_b"]           # a: (rows, rank); b: (rank, cols)
        for k in range(a.shape[1]):
            w += np.outer(a[:, k], b[k])
        e = f32_to_bf16_bits_rne(w.astype(np.float32)).reshape(-1)
    else:
        e = base_bits.copy().reshape(-1)
    e[z[f"t{i}_idx"]] = z[f"t{i}_val"]              # exact overwrite with stored bit patterns
    return e.reshape(base_bits.shape)
```

The published SHA-256 decides exactness. If the low-rank path's rounding ever differs from the article's, the hash check fails and the patch is refused.

## 4. Inference and Evaluation

### 4.1 Engine

- `wse.engines.hf_torch` builds the module and runs forward passes, teacher-forced reads, and batched generation with a KV cache.
- Device selection is explicit: `mps` when available, otherwise `cpu`. The worker records the device, dtype, and library versions in every report.
- Batched generation derives `position_ids` from the attention mask (the cumulative sum minus one), at prefill and at every decode step, as `transformers`' `generate` does. A left-padded row must compute the same logits as the same prompt run alone. Without this, a padded row is shifted by its padding: on Llama-3.2-1B-Instruct, the first-step logits of a row with 16 pads moved by 0.156.
- Bitwise reproducibility holds within one device and library version, not across them (M1 plan Section 7.3).

### 4.2 Generation profiles

A profile is a registered, versioned sampling configuration. Each imported model also receives a default profile frozen from its `generation_config.json` at import. The profile is named after the repository: the name is lowercased, every character other than a letter or digit becomes `_`, `m_` is prefixed if the result would start with a digit, and `_default@1` is appended. For example, Llama-3.2-1B-Instruct gets `llama_3_2_1b_instruct_default@1`, which satisfies M1's extension-ID pattern `^[a-z][a-z0-9_]{0,63}$`.

`qwen35think@1` reproduces the article's `qwen35think` profile (`exp8_math/code/profiles.py`):

- thinking enabled through the chat template;
- the user message is the problem followed by "Please reason step by step, and put your final answer within \boxed{}.";
- temperature 0.6, top_p 0.95, top_k 20, and presence penalty 1.5;
- at most 32,768 new tokens;
- seed 0.

`transformers` has no presence penalty, so WSE adds a logits processor with vLLM's semantics. It subtracts the penalty once from each token that already occurs in the generated output, ignoring prompt tokens, and runs before temperature, top-k, and top-p:

```python
class PresencePenalty(LogitsProcessor):
    def __init__(self, penalty: float, prompt_len: int):
        self.penalty, self.prompt_len = penalty, prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated = input_ids[:, self.prompt_len:]           # left-padded batches share one prompt length
        present = torch.zeros_like(scores, dtype=torch.bool)
        present.scatter_(1, generated, True)
        return scores - self.penalty * present.to(scores.dtype)
```

Each (problem, sample) pair gets its own seed, derived from the profile seed, the problem ID, and the sample index.

### 4.3 Suites and graders

- **Final suite.** MATH-500: `HuggingFaceH4/MATH-500` at revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`, file `test.jsonl`, 500 problems. It is held in evaluator storage, which no run-bound engine can reach.
- **Practice (development) suites.** GSM8K train, from `train.jsonl` in `openai/grade-school-math`. MATH train items join them when the operator supplies a JSON Lines copy. Registration rejects any item whose normalized problem text appears in MATH-500.
- **Graders.**
  - `math_boxed@1` grades the last `\boxed{}` answer with math-verify, as the article does.
  - `gsm8k_numeric@1` compares the boxed number with the reference after `####`.
- **Metrics.**
  - `accuracy`: the mean over samples;
  - `limit_hit_rate`: the share of outputs that reach the token limit;
  - `mean_generated_tokens`.

  Clusters are problems. Reports give paired differences with cluster-bootstrap intervals and an exact interval for each proportion.

### 4.4 Local protocol and the decision gate

- **The article's protocol** runs MATH-500 with 4 samples per problem, up to 32,768 tokens each. Its independent re-run of the edited model generated 43.4 million tokens in about 51 minutes on an H200 with vLLM. The base model reaches the limit on about 93% of answers, so it generates even more.
- **Task 7 measures local throughput** and estimates the cost of three protocols: the 25-problem comparison, a 100-problem single-sample final suite, and the full protocol.
- **The user then decides** whether Q3's final evaluation runs locally on a subset or on a GPU worker. A GPU worker would become its own task, inserted before Task 13.

## 5. Agent-Facing Tools (Q2)

### 5.1 Edit operators

Each new edit operator requires a two-dimensional tensor with known axes. Policies reject edits to alias-group members and protected roles.

- **`fill_slice@1`:** sets a slice to one finite scalar, for example to ablate an attention head, as p25c did.
- **`project_out@1`:** removes an output direction from selected input-axis indices `C`, as in the M1 plan's Section 5.2: `W'[:, C] = W[:, C] − α d (dᵀ W[:, C]) / (dᵀ d)`.
  - `d` lies on the output axis. It is given inline (at most 2,048 values) or as an analysis or probe artifact URI.
  - `α` must be finite, and `‖d‖` must exceed the policy threshold.
  - p50c has this form, applied to 64 columns of `mlp.down` in two layers.
- **`add_low_rank@1`:** `W' = W + α U Vᵀ`, with `U: [out, r]`, `V: [in, r]`, and `r ≤ 16`. Factors are given inline or by artifact URI.

### 5.2 Analyses

- **`tensor_norms@1`:** L2 norms per row or per column.
- **`svd_top_k@1`:** the top k singular values and vectors of a tensor or slice, with k ≤ 16, in FP32. Above a size threshold it uses a randomized SVD.
- **`direction_projection@1`:** projections or cosines of a direction against the rows or columns of a tensor.

Vector results are stored as run-scoped artifacts (M1's `wse://analyses/{digest}`). The response carries a summary, the inline top entries, and the URI. Other operations accept that URI as a direction or factor.

### 5.3 Probes

`wse_probe` runs a teacher-forced forward pass over up to N texts. The texts can be:

- registered practice items;
- outputs of the run's own generation passes, by reference;
- texts the agent supplies, within length limits.

It returns summaries and vector artifacts:

- the mean residual stream at chosen layers (Level 0);
- mean activations at role sites such as the input of `mlp.down` (Level 1);
- the contrast direction between two text groups;
- per-text log-probability statistics.

### 5.4 Generation passes

Generation suites are evaluated as development evaluations through M1's job system. Each report carries per-problem correctness, length, and limit hits, plus a bounded excerpt of each output's final tokens, so that the agent can see repetition. Full outputs remain artifacts, readable by URI within the response limits.

### 5.5 Budgets and policy

- **New ledger counters:** `generated_tokens`, `teacher_forced_tokens`, `probe_calls`, and `generation_passes`.
- **Policy `qwen_loop_v1`, initial limits.** Task 13 may lower them from Task 7's measurements before any trial.
  - 10 generation passes, each of at most 32 practice problems at most 8,192 new tokens;
  - 25 probe calls, each of at most 64 texts of at most 1,024 tokens;
  - 50 analyses, 100 slice reads, 50 previews, and 20 candidate commits;
  - 6 hours of computation per run, and 2 hours per operation.
- **Protections:** alias groups (embeddings and output), the final norm, and edits that scale a whole tensor by a constant. The article's `save_edit` refused the same.

### 5.6 MCP tools

- New tool: `wse_probe`.
- `wse_analyze` and `wse_preview_patch` gain the new kinds and operators, and `wse_submit_evaluation` accepts generation suites.
- Run-bound isolation, idempotency, and response limits stay as in M1.

## 6. Agent Trials (Q3)

- **Task.** The agent-visible `task.json` asks the agent to reduce Qwen3.5-0.8B's failures to finish math answers without lowering accuracy, with objective metrics on the development suites.
- **Isolation.** The agent never sees MATH-500 items, the article, its logs, the published edits, or this plan. The runner isolates them as in M1, and a leak test checks every agent-visible surface.
- **Controls, evaluated on the same final suite:**
  - the base checkpoint;
  - the replayed p50c edit, as an operator-only reference;
  - five random projections matched to p50c's changed-element count and L2 norm (M1 plan Section 7.3).
- **Agent.** The OpenAI Agents SDK adapter with `gpt-5.6-luna`, in one fixed configuration. Turn and time limits are preregistered in Task 13.
- **Trials.** Five trials. Final evaluations run only after every trial has frozen. The report follows M1's form in `docs/m2-qwen-trials.md`.

## 7. Planned Files

```text
pyproject.toml, uv.lock                # models-hf extra
src/wse/
  hf_snapshot.py                       # allow-listed snapshot reading, file hashes, safetensors headers
  adapters/hf_causal_lm.py             # Level 0 adapter: manifest, axes, alias groups, capabilities
  adapters/families.py                 # FamilyDescriptor; built-in qwen3_5@1 and llama@1
  engines/__init__.py
  engines/hf_torch.py                  # module construction, device, forward, reads, generation
  patch_blocks.py                      # sparse and dense binary patch blocks
  external_patches.py                  # lm-understand-v1 importer
  operators/directions.py              # fill_slice, project_out, add_low_rank
  operators/linalg.py                  # tensor_norms, svd_top_k, direction_projection
  probes.py                            # worker-side probes and their results
  evaluators/sampling.py               # profiles and the presence-penalty processor
  evaluators/generation.py             # generation evaluator and metrics
  evaluators/graders.py                # math_boxed, gsm8k_numeric
  evaluators/text_suites.py            # JSON Lines suites, hashes, overlap checks
  strategies/direction_search.py       # direction_search@1, the scripted discovery strategy (Task 12)
  (modified) store.py, database.py, tensors.py, patches.py, operators/tensor.py, registry.py,
             schemas.py, policy.py, budget.py, jobs.py, worker.py, mcp_server.py, cli.py, runner.py
tests/
  hf_fixtures.py                       # tiny random checkpoints built offline with transformers configs
  test_blob_store.py  test_hf_import.py  test_hf_engine.py  test_patch_blocks.py
  test_external_patches.py  test_generation_eval.py  test_families.py  test_direction_ops.py
  test_linalg_analyses.py  test_probes.py  test_llm_policy_mcp.py
examples/qwen_loops/
  task.json  policy.json  profiles.json  # agent-visible task and fixed policy (Q3)
  check_expressiveness.py              # operator-only scripted MCP client for the Q2 expressiveness check
docs/
  models.md                            # model support levels and adding a model
  m2-q1-report.md                      # Q1 measurements and the decision gate
  m2-q2-report.md                      # Q2 expressiveness, scripted discovery, and smoke matrix
  m2-qwen-trials-prereg.md             # Q3 preregistration
  m2-qwen-trials.md                    # Q3 report
```

Operator storage (outside the repository), by default under `~/Desktop/wse-models/`:

- model snapshots;
- the article's logs (sparse checkout);
- datasets;
- WSE workspaces for language models.

## 8. Implementation Tasks

Tasks run in order 1→15. Q1 is Tasks 1–7, Q2 is Tasks 8–12, and Q3 is Tasks 13–15. Each task ends with the full suite (`uv run --no-sync pytest -q`) and a commit. Tests that need the `models-hf` extra skip, with a stated reason, when it is absent, as M1's actual-host test does. Every new test runs offline on tiny random checkpoints. Steps that use real models are marked **(real model)** and record their results in the task's report.

### Task 1: The `models-hf` extra and operator setup

**Files:** `pyproject.toml`, `uv.lock`.

- [x] Add `models-hf = ["transformers>=5.22.2", "math-verify>=0.9"]` to the optional dependencies. `transformers` 5.22.2 was the latest release on PyPI on 2026-11-09, and its source tree contains `models/qwen3_5` with pure-PyTorch fallbacks for the Gated DeltaNet layers. The lock pins the exact versions.
- [x] **(operator)** In the user's own terminal (Claude Code's auto mode refuses installs from PyPI), run `uv lock` and then `uv sync --extra agents-openai --extra models-hf`.
- [x] **(operator)** On Hugging Face, request access to `meta-llama/Llama-3.2-1B-Instruct`; approval is manual. It is a `LlamaForCausalLM` with 1,235,814,400 BF16 parameters under the Llama 3.2 Community License.
- [x] **(operator)** From the repository root, download the snapshots, fetching only allow-listed files. Llama repositories also hold an `original/` folder with a pickled checkpoint. The `hf` command comes with `transformers` and is installed in the project environment:

  ```bash
  M=~/Desktop/wse-models
  # huggingface_hub 2.x takes one pattern per --include; extra patterns after one flag become file names.
  INC=(--include "*.json" --include "*.safetensors" --include "*.jinja" --include "tokenizer*" --include "vocab*" --include "merges*")
  uv run --no-sync hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 "${INC[@]}" --local-dir "$M/Qwen3.5-0.8B"
  ( printf 'HF token: '; read -rs T; echo; HF_TOKEN="$T" uv run --no-sync hf download meta-llama/Llama-3.2-1B-Instruct "${INC[@]}" --local-dir "$M/Llama-3.2-1B-Instruct" )
  ```

  Record the Llama revision that the download resolved. It is listed under the snapshot's `.cache/huggingface/download/` metadata.
- [x] **(operator)** Fetch the article's edits and code, and the datasets:

  ```bash
  git clone --filter=blob:none --sparse https://github.com/RikiyaT/lm_understand-logs "$M/lm_understand-logs"
  git -C "$M/lm_understand-logs" sparse-checkout set exp8_math/code exp8_math/runs/p10c/edit exp8_math/runs/p25c/edit exp8_math/runs/p50c/edit
  uv run --no-sync hf download HuggingFaceH4/MATH-500 test.jsonl --repo-type dataset --revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be --local-dir "$M/datasets/MATH-500"
  curl -L -o "$M/datasets/gsm8k-train.jsonl" https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl
  curl -L -o "$M/datasets/gsm8k-test.jsonl" https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
  ```

- [x] Verify the environment:
  - `uv run --no-sync python -c "from transformers import Qwen3_5ForCausalLM, LlamaForCausalLM; import math_verify"` succeeds;
  - the full suite still passes 366 tests;
  - the installed versions are recorded for `docs/m2-q1-report.md`.
- [x] Commit `pyproject.toml` and `uv.lock`.

### Task 2: Per-tensor content-addressed storage and stored dtypes

**Files:** modify `src/wse/store.py`, `src/wse/database.py` (migration 8), and `src/wse/tensors.py`; create `tests/test_blob_store.py`.

- [x] Write failing tests:
  - `test_blob_id_is_sha256_of_little_endian_row_major_bytes`: a 2×3 BF16 tensor of known values (1.0, −2.0, 0.5, −0.0, a subnormal, 3.0). The expected bit patterns (`0x3F80`, `0xC000`, `0x3F00`, `0x8000`, …) are packed with `struct.pack("<6H", ...)` and hashed in the test.
  - `test_a_candidate_differing_in_one_tensor_adds_one_blob`.
  - `test_bf16_f16_f32_round_trip_bitwise_including_signed_zero_and_subnormals`.
  - `test_migration_8_keeps_m1_identities`: build an M1 workspace at schema version 7, migrate it, and check that `weights_hash`, manifests, tensors, and selected checkpoints are unchanged.
  - `test_a_tampered_blob_marks_every_checkpoint_that_uses_it_corrupt`.
- [x] Run `uv run --no-sync pytest tests/test_blob_store.py -q`. The tests must fail because the blob store is missing.
- [x] Implement:
  - blobs under `artifacts/blobs/<sha256>`, published by atomic rename as in M1's Section 5.3;
  - manifests that reference blobs;
  - reference counting for recovery;
  - migration 8;
  - dtype-aware loading of BF16 and F16 through `torch.frombuffer` on the raw bytes.
- [x] Run the new tests and the full suite; all must pass.
- [x] Review and commit.

### Task 3: Generic import, identity, and the Qwen3.5 name map

**Files:** create `src/wse/hf_snapshot.py`, `src/wse/adapters/hf_causal_lm.py`, `src/wse/adapters/families.py`, and `tests/hf_fixtures.py`; modify `src/wse/registry.py` (kind `family` and built-ins), `src/wse/schemas.py`, and `src/wse/cli.py` (`wse import-hf SNAPSHOT --repo ID --revision SHA`); create `tests/test_hf_import.py`.

- [x] Write `tests/hf_fixtures.py`. It builds tiny random checkpoints offline with `save_pretrained(safe_serialization=True)`:
  - a Llama-style model with tied embeddings and one without them (`LlamaConfig`: 2 layers, hidden 16, 4 heads, 2 KV heads);
  - a Qwen3 model (`Qwen3Config`);
  - a Gemma 3 text model (`Gemma3TextConfig`);
  - a Qwen3.5 multimodal wrapper (`Qwen3_5Config` with a 2-layer text config, one linear-attention and one full-attention layer, and a 1-block vision config) that carries `model.language_model.*`, `model.visual.*`, and `mtp.*` tensors.

  The helper also adds decoy files: `training_args.bin`, `original/consolidated.00.pth`, and `onnx/model.onnx`.
- [x] Write failing tests in `tests/test_hf_import.py`:
  - `test_only_allow_listed_files_are_read`: the importer's read log contains no decoy;
  - `test_a_snapshot_with_only_pickled_weights_is_rejected`;
  - `test_auto_map_in_config_is_rejected`;
  - `test_an_architecture_transformers_does_not_implement_is_rejected`;
  - `test_axes_follow_module_types`: q_proj is `[out, in]`, the embedding is `[vocab, hidden]`, norm weights are `[features]`;
  - `test_tied_embeddings_form_one_alias_group_and_one_blob`;
  - `test_identity_changes_with_tokenizer_chat_template_and_generation_config`;
  - `test_qwen3_5_descriptor_maps_the_language_model_and_drops_vision_and_mtp`: canonical names stay the checkpoint names, the module names come from the map, and no `model.visual.*` or `mtp.*` tensor is stored;
  - `test_unmapped_names_fail_with_the_list_of_names`;
  - `test_reimporting_a_snapshot_creates_no_new_checkpoint_or_blob`;
  - `test_llama_style_checkpoints_import_without_a_descriptor`.
- [x] Run the tests; they must fail.
- [x] Implement:
  - snapshot reading through safetensors headers and memory-mapped tensors;
  - the meta-device module walk for axes and alias groups;
  - the descriptor registry and `qwen3_5@1`'s name map. If `Qwen3_5ForCausalLM` expects `model.layers.*`, the rule `model.language_model.` → `model.` applies; the fixture test pins whichever prefix the class actually uses. Its drop rules cover `model.visual.` and `mtp.`;
  - the identity hashes;
  - `wse import-hf`.
- [x] Run the new tests and the full suite.
- [x] **(real model)** Import both snapshots and record, for each: tensor count, parameter count, dtypes, alias groups, and identity hashes. Qwen3.5-0.8B's index lists 488 tensors (320 `model.language_model.*`, 153 `model.visual.*`, and 15 `mtp.*`), and its config ties input and output embeddings.
- [x] Review and commit.

### Task 4: Engine, logits parity, and greedy generation

**Files:** create `src/wse/engines/__init__.py` and `src/wse/engines/hf_torch.py`; modify `src/wse/adapters/hf_causal_lm.py`; create `tests/test_hf_engine.py`.

- [x] Write failing tests:
  - `test_wse_built_module_matches_from_pretrained_logits_bitwise_on_cpu`, for every fixture;
  - `test_mps_logits_match_cpu_within_tolerance`, skipped without MPS, with `atol` and `rtol` stated per dtype;
  - `test_greedy_generation_matches_transformers_generate`;
  - `test_core_modules_import_no_transformers`, extending M1's import-boundary tests.
- [x] Run the tests; they must fail.
- [x] Implement module construction on the meta device, assignment of WSE tensors, device selection, forward passes, and batched greedy and sampled generation with a KV cache.
- [x] Run the new tests and the full suite.
- [x] **(real model)** For three fixed prompts, compare each model's logits with `from_pretrained` on the same snapshot, on CPU and on MPS. Record the maximum absolute difference and any operator that needed `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- [x] Review and commit.

### Task 5: BF16 edits, binary patches, and exact replay of the published edits

**Files:** modify `src/wse/patches.py` and `src/wse/operators/tensor.py`; create `src/wse/patch_blocks.py` and `src/wse/external_patches.py`; modify `src/wse/cli.py` (`wse import-patch`); create `tests/test_patch_blocks.py` and `tests/test_external_patches.py`.

- [x] Write failing tests:
  - `test_bf16_scale_computes_in_fp32_and_rounds_to_nearest_even`, with hand-computed bit patterns, including a tie that rounds to even;
  - `test_changed_elements_are_counted_after_write_back`;
  - `test_sparse_and_dense_blocks_round_trip_and_bind_base_and_result_ids`;
  - `test_replay_of_a_binary_patch_is_hash_exact_and_runs_no_operator`;
  - `test_lm_understand_v1_sparse_and_lowrank_rebuild`: a synthetic `manifest.json` and `edit.npz`, written by the test, in both forms;
  - `test_a_wrong_base_hash_is_refused_before_any_write`;
  - `test_a_wrong_edited_hash_leaves_no_patch_and_no_candidate`.
- [x] Run the tests; they must fail.
- [x] Implement:
  - the dtype-aware operators;
  - the helpers `bf16_to_f32` and `f32_to_bf16_bits_rne` in `wse.tensors`, with round-to-nearest-even on the float32 bit pattern;
  - the patch blocks;
  - `rebuild()` from Section 3.4;
  - `wse import-patch`.
- [x] Run the new tests and the full suite.
- [x] **(real model)** Replay p10c, p25c, and p50c on the imported Qwen3.5-0.8B. All edited hashes must equal their manifests. For p50c these are:

  | Tensor | `base_sha256` | `edited_sha256` | Changed entries |
  | --- | --- | --- | --- |
  | `model.language_model.layers.15.mlp.down_proj.weight` | `ef60d2c687bab19464dca9c8bfd7777e46ad226df0c9e25643b0bab04b5a1cca` | `82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53` | 65,260 |
  | `model.language_model.layers.19.mlp.down_proj.weight` | `638eedeb701b94aa861f4438e615c07ec6c60ee703db1bfee2de577bb6498c3e` | `8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2` | 64,928 |

  The p50c candidate adds exactly two blobs.
- [x] Review and commit.

### Task 6: Generation evaluation: profiles, presence penalty, graders, and suites

**Files:** create `src/wse/evaluators/sampling.py`, `src/wse/evaluators/generation.py`, `src/wse/evaluators/graders.py`, `src/wse/evaluators/text_suites.py`, and `tests/test_generation_eval.py`; modify `src/wse/registry.py` and `src/wse/schemas.py`.

- [x] Write failing tests:
  - `test_presence_penalty_subtracts_once_per_present_output_token_and_ignores_the_prompt`;
  - `test_sampling_order_is_penalty_temperature_top_k_top_p`;
  - `test_each_problem_and_sample_has_its_own_reproducible_seed`;
  - `test_math_boxed_grades_the_last_boxed_answer`, with cases taken from math-verify's documented examples;
  - `test_gsm8k_numeric_reads_the_reference_after_hashes`;
  - `test_limit_hits_and_lengths_are_reported_per_problem`;
  - `test_suites_record_hashes_and_reject_overlap_with_the_final_suite`;
  - `test_generation_reports_bind_model_profile_suite_and_grader_identities`;
  - `test_default_profiles_are_frozen_from_generation_config`.
- [x] Run the tests; they must fail.
- [x] Implement:
  - the profiles, `qwen35think@1` from Section 4.2 among them, and the processor from Section 4.2;
  - chat-template rendering with per-profile template arguments;
  - batched generation with per-row seeds;
  - the graders, metrics, and suite registration.
- [x] Before freezing `PresencePenalty`, compare its semantics with vLLM's sampler source and record the comparison in a code comment.
- [x] **(real model)** Render one MATH-500 prompt with Qwen3.5-0.8B's template and `qwen35think@1`, and confirm that thinking is enabled.
- [x] Run the new tests and the full suite.
- [x] Review and commit.

### Task 7: Q1 runs, report, and the decision gate

**Files:** create `docs/m2-q1-report.md`.

- [x] **(real model)** Measure generation throughput on MPS and on CPU for both models at batch sizes 1, 8, and 32. For Qwen3.5-0.8B, also measure at contexts of up to 4,096 and up to 32,768 tokens. Record tokens per second and peak memory.
- [x] **(real model)** Compare base and p50c on Qwen3.5-0.8B:
  - 25 MATH-500 problems, chosen by a recorded seed;
  - 1 sample per problem, with `qwen35think@1`;
  - report accuracy, limit-hit rate, and mean generated tokens, with intervals.

  The expected direction comes from the article: the base reached the limit on 93.0% of answers and the edit on 33.1%, with MATH-500 accuracy of 7.9% and 48.1%. The size of the change is not expected to match, because the sample is small and the engine differs.
- [x] **(real model)** Run Llama-3.2-1B-Instruct through Level 0 with no new code:
  - import it and check logits parity (Tasks 3–4);
  - preview, apply, and replay a `scale_slice` on one MLP projection, byte-identical;
  - evaluate 25 GSM8K test problems with `llama_3_2_1b_instruct_default@1`.

  Then confirm that `src/wse` has no Llama-specific code: `grep -rn -i llama src/wse` finds nothing at this point, because the `llama@1` descriptor arrives only in Task 8.
- [x] Write `docs/m2-q1-report.md` with:
  - versions, device, and throughput;
  - the replay hashes;
  - both comparisons;
  - cost estimates for the three protocols of Section 4.4.
- [ ] **Decision gate (user):** choose local subset or GPU worker for Q3's final evaluation. If a GPU worker is chosen, insert its task before Task 13.
- [ ] Review and commit. **Q1 exit:** the Section 1 conditions for Q1 hold. Reviewed and committed; the exit waits for the plan author's reading of the Llama patch round trip and the strictly scored GSM8K evaluation (`docs/m2-q1-report.md`, Section 9).

### Task 8: Family descriptors with roles, heads, and protections

**Files:** modify `src/wse/adapters/families.py` and `src/wse/policy.py`; create `tests/test_families.py`.

- [ ] Write failing tests:
  - `test_every_role_rule_resolves_to_tensors_with_expected_shapes` on the Qwen3.5 and Llama fixtures. For example, `mlp.down` is `[hidden, intermediate]`, and `attn.o`'s head axis has `num_heads × head_dim` entries.
  - `test_head_groups_partition_the_head_axis`;
  - `test_protected_roles_reject_edits_with_an_actionable_policy_error`;
  - `test_a_separately_supplied_descriptor_registers_without_core_changes`, in the spirit of M1's extension tests;
  - `test_descriptors_are_versioned_and_recorded_in_manifests`.
- [ ] Run the tests; they must fail.
- [ ] Implement the role and head rules of `qwen3_5@1` and `llama@1` from the fixtures' module names, with `final_norm`, `embed`, and `lm_head` protected.
- [ ] Run the new tests and the full suite.
- [ ] Review and commit.

### Task 9: Edit operators `fill_slice`, `project_out`, and `add_low_rank`

**Files:** create `src/wse/operators/directions.py` and `tests/test_direction_ops.py`; modify `src/wse/registry.py` and `src/wse/schemas.py`.

- [ ] Write failing tests:
  - known-value tests for each operator in FP32 and BF16;
  - `test_project_out_leaves_columns_outside_c_byte_identical`;
  - `test_project_out_removes_d_from_the_selected_columns_in_fp32`: after the edit, `dᵀ W'[:, C]` is about zero when α = 1;
  - `test_directions_by_artifact_uri_must_belong_to_the_run`;
  - `test_low_rank_rank_and_shape_limits`;
  - `test_alias_groups_and_protected_roles_are_refused`.
- [ ] Run the tests; they must fail.
- [ ] Implement the three operators through M1's operator contract. Their materialized results use Task 5's binary blocks.
- [ ] Run the new tests and the full suite.
- [ ] **(real model)** Separate recipe re-execution from exact replay:
  1. Recover p50c's direction `d` and scale `α` from the published diff, as the top singular vector of `ΔW[:, C]` in each layer.
  2. Re-execute `project_out` on the base.
  3. Report how many entries differ from the exact p50c tensors, and by how many units in the last place.

  Exact replay (Task 5) stays the hash-exact path. The recovered `d` is an operator-only artifact.
- [ ] Review and commit.

### Task 10: Analyses for large tensors

**Files:** create `src/wse/operators/linalg.py` and `tests/test_linalg_analyses.py`; modify `src/wse/registry.py`, `src/wse/policy.py`, and `src/wse/schemas.py`.

- [ ] Write failing tests:
  - `test_tensor_norms_on_a_known_matrix`;
  - `test_svd_top_k_matches_numpy_on_a_small_matrix_up_to_sign`;
  - `test_randomized_svd_is_used_above_the_threshold_and_is_reproducible_with_its_seed`;
  - `test_direction_projection_on_known_vectors`;
  - `test_vector_results_return_a_summary_and_a_run_scoped_uri_within_31_kib`;
  - `test_analyses_never_call_forward`, as in M1;
  - `test_analysis_usage_is_metered`.
- [ ] Run the tests; they must fail.
- [ ] Implement the three analyses through M1's operator contract.
- [ ] Run the new tests and the full suite.
- [ ] Review and commit.

### Task 11: Probes, generation passes, budgets, policy, and MCP tools

**Files:** create `src/wse/probes.py`, `tests/test_probes.py`, and `tests/test_llm_policy_mcp.py`; modify `src/wse/budget.py`, `src/wse/policy.py`, `src/wse/jobs.py`, `src/wse/worker.py`, `src/wse/mcp_server.py`, `src/wse/schemas.py`, `src/wse/database.py` (migration 10), and `docs/protocol.md`.

- [ ] Write failing tests:
  - `test_probe_hidden_states_match_a_direct_forward_on_the_fixture`;
  - `test_probe_role_sites_capture_the_input_of_mlp_down`;
  - `test_contrast_direction_is_the_difference_of_group_means`;
  - `test_probe_and_generation_usage_is_reserved_and_settled_in_tokens`;
  - `test_generation_pass_reports_tail_excerpts_within_the_response_limit`;
  - `test_qwen_loop_v1_limits_and_protections`;
  - `test_final_suite_is_unreachable_through_probe_and_generation_tools`;
  - `test_mcp_and_python_agree_on_new_tools_errors_and_usage`, following M1's parity test;
  - `test_a_flat_list_is_accepted_when_the_other_axes_have_length_one`;
  - `test_a_run_bound_server_defaults_the_experiment_id`.

  The last two are agent-facing items carried over from M1.
- [ ] Run the tests; they must fail.
- [ ] Implement:
  - probes in the worker;
  - generation passes as development evaluations;
  - the four ledger counters, with migration 10 backfilling existing experiments (migration 9, from Task 5, added the `imported_patches` table);
  - policy `qwen_loop_v1` from Section 5.5;
  - `wse_probe`, and the extended `wse_analyze`, `wse_preview_patch`, and `wse_submit_evaluation`;
  - a worker-side cache that keeps loaded models across jobs, keyed by checkpoint ID.
- [ ] Update `docs/protocol.md`. Its examples use Llama-3.2-1B-Instruct and generic tasks, never the Qwen loop task.
- [ ] Run the new tests and the full suite.
- [ ] Review and commit.

### Task 12: Q2 exit: expressiveness, scripted discovery, and the smoke matrix

**Files:** create `src/wse/strategies/direction_search.py` (`direction_search@1`), `examples/qwen_loops/check_expressiveness.py` (operator-only), `docs/models.md`, and `docs/m2-q2-report.md`; modify `src/wse/registry.py`.

- [ ] **(real model)** Check expressiveness: a scripted client drives the managed tools through MCP and applies `project_out` with Task 9's recovered direction, as an operator-only artifact, to the same 64 columns of layers 15 and 19. On a fixed practice subset, the development limit-hit rate must fall relative to the base.
- [ ] **(real model)** Run scripted discovery, which has no oracle and uses only agent-available tools. The strategy:
  1. runs a generation pass on practice problems;
  2. probes the outputs that reached the limit against those that finished, at the input of `mlp.down` in every layer;
  3. projects the contrast direction out of the most aligned columns of the layer with the largest separation;
  4. evaluates and selects.

  Report the outcome whether or not it improves on the base.
- [ ] **(real model)** Run the smoke matrix through Level 0 with no new code. For `Qwen/Qwen3-0.6B` (ungated) and `google/gemma-3-1b-it` (gated, so request access first):
  - import, with the download allow-list of Task 1;
  - check logits parity;
  - replay a patch;
  - run the generic analyses;
  - probe hidden states;
  - run one short generation.
- [ ] Write `docs/models.md`: the support levels, the boundaries of Global Constraints, how to import a snapshot, how to write a family descriptor, and the models verified so far.
- [ ] Write `docs/m2-q2-report.md` with the expressiveness check, the scripted discovery outcome, and the smoke matrix.
- [ ] Run the full suite.
- [ ] Review and commit. **Q2 exit:** the Section 1 conditions for Q2 hold.

### Task 13: Q3 preregistration and leak checks

**Files:** create `examples/qwen_loops/task.json`, `examples/qwen_loops/policy.json`, `examples/qwen_loops/profiles.json`, and `docs/m2-qwen-trials-prereg.md`; modify `src/wse/runner.py` and `src/wse/agent_adapters/openai_agents.py`; extend `tests/test_runner.py` and `tests/test_agent_adapter.py`.

- [ ] Write the preregistration and commit it before any trial. It covers:
  - the task text and objective metrics;
  - the policy limits, adjusted from Task 7's measurements;
  - the development suites;
  - the final suite: the subset size or the full set, as the decision gate chose;
  - profiles and graders;
  - the controls of Section 6;
  - the agent configuration: model, `max_turns`, wall-clock limit, tracing disabled;
  - five trials;
  - the primary success criterion: final accuracy above the base, with a paired interval that excludes zero, and no increase in the limit-hit rate;
  - the analysis plan.
- [ ] Write failing tests:
  - `test_qwen_loop_surfaces_reveal_no_oracle`: the instructions, tool names, descriptions, schemas, and outputs of a dry run contain no MATH-500 item, no p50c layer indices or column lists, no recovered direction values, and no reference to the article or its logs;
  - `test_cached_input_tokens_are_recorded`, a carried-over M1 item;
  - `test_runner_raised_errors_do_not_point_to_the_server_log`, a carried-over M1 item.
- [ ] Run the tests; they must fail.
- [ ] Implement until they pass, then run the full suite.
- [ ] Review and commit.

### Task 14: Controls

**Files:** append to `docs/m2-qwen-trials.md`.

- [ ] **(real model)** Generate five random projections matched to p50c's changed-element count and L2 norm, with the seeds recorded in the preregistration. Evaluate them, the base, and p50c on the final suite with the preregistered profile. These are operator runs; no agent is involved.
- [ ] Record each control's report hashes and costs.
- [ ] Commit.

### Task 15: Five agent trials and the report

**Files:** create `docs/m2-qwen-trials.md`.

- [ ] **(operator)** Run the five live trials with the fixed configuration. **They are outward-facing:** they send task instructions and tool results to the model provider, and they cost money. Before starting, post the expected tokens and cost in chat and check that no other luna-heavy job is running.
- [ ] Run final evaluations only after all five trials have frozen. Each pairs its candidate with the base control, as in M1.
- [ ] Verify as M1 did:
  - replay every export;
  - scan every output for the key, the endpoint, final-suite items, and oracle values;
  - re-derive every reported number from the raw exports.
- [ ] Write `docs/m2-qwen-trials.md`. For every trial it records:
  - candidates, development and final metrics, generated and teacher-forced tokens, probe calls, and changed parameters;
  - agent tokens and cost;
  - elapsed time and failure causes.

  The report states the result against the preregistered criterion, including failures.
- [ ] Commit. **Q3 exit:** the Section 1 condition for Q3 holds.

## 9. Carried Over From M1

| M1 open item | M2 decision |
| --- | --- |
| Accept a flat list for a selection whose other axes have length 1, or name the expected nesting | Task 11, as part of the new agent-facing configuration |
| Let a run-bound server default the experiment ID | Task 11 |
| Record cached input tokens | Task 13 |
| The "see the server log" hint misleads for runner-side errors | Task 13 |
| HTTP-level diagnostics (attempts, status codes, Azure region and spillover headers) | Deferred; it needs a custom HTTP client |

## 10. Deferred and Out of Scope

- The Claude Agent SDK adapter and two additional fault classes on small models come after Q3, as the user decided on 2026-11-09.
- GPU workers, and vLLM or MLX inference backends, unless the Task 7 decision gate brings one forward. `mlx-lm` implements Qwen3.5 (`mlx_lm/models/qwen3_5.py`), which makes it a candidate for a later local backend.
- Edits to alias groups (tied embeddings) and to the vision tower.
- A sandboxed research-Python mode for agents.
- IFEval, GPQA, and the article's other benchmarks.
- Models outside the boundaries of Global Constraints.
