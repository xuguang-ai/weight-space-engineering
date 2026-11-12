# Models

WSE works on causal language models that the pinned `transformers` (5.22.2) implements natively, from safetensors weights in BF16, F16, or F32 that fit the host's memory. Excluded:

- architectures that need `trust_remote_code`;
- quantized formats (GGUF, GPTQ, AWQ, MXFP4, ...);
- encoder–decoder models.

Mixture-of-experts tensors stay readable and editable through the generic tools, without per-expert semantics. The small MLP models of milestone M1 use their own adapter, `mlp@1`.

## The Model API

A `Model` is one version of a model in a workspace. Its identity never changes: `checkpoint_id`, `weights_hash`, and `source` (the repository and the commit its revision resolved to). An edit makes a new `Model`, and the original stays as it was.

```python
from wse import Model, compare_models
```

| Step | Call | What it gives |
| --- | --- | --- |
| Load | `Model.from_pretrained(source, revision=..., workspace=...)` | Imports a Hugging Face snapshot once and opens it. A repository ID is downloaded at `revision` (default `main`), resolved to its commit; a local directory also needs `repo=` and `revision=`, the 40-character commit it was downloaded from ([Importing a model](#importing-a-model)). |
| | `Model.open(checkpoint_id, workspace=...)` | A stored checkpoint: an import or a saved edit. |
| Inspect | `model.inspect()` | Identity, architecture, parameters, tensors grouped by layer, protected tensors, shared parameters, and how many tensors of the snapshot the import left out. |
| | `model.tensors("layers.14.self_attn.*")` | The name, shape, dtype, and protection of each matching tensor (a glob, or else a substring). |
| | `model.tensor("layers.14.self_attn.o_proj")` | A copy of one tensor (short names allowed): changing it changes nothing in the model. |
| Diagnose | `model.generate(prompt, max_new_tokens=...)` | One answer: `text`, `generated_tokens`, and `finish_reason`, which is `stop`, or `length` for an answer cut off at the limit. |
| | `model.evaluate(data, evaluator="math", metrics=[...], max_new_tokens=...)` | Accuracy, limit-hit rate, and mean generated tokens, with 95% intervals. `samples()` lists every answer with its grade and text. |
| Investigate | `model.probe(evaluation, sites=[...], last=...)` | Mean activations at named sites over each answer's last tokens. `samples` lines the rows up with the answers (problem, grade, limit hit, text), and `contrast(site, a, b)` compares two groups of answers. |
| Intervene | `model.edit(target, operator, **parameters)` | An `Edit`: one registered operator, applied in memory as an exact patch. |
| | `model.replace_weights({name: tensor})` | An `Edit` that replaces whole tensors with values computed any other way, after checking their names, shapes, stored dtypes, finiteness, and the protected tensors. |
| | `model.apply_patch(path, format=...)` | A stored `Edit` imported from a published edit (`lm-understand-v1`, every hash checked) or from a directory that `Edit.export` wrote (`wse-patch-v1`). |
| Evaluate | `compare_models(base=..., candidate=..., data=..., evaluator=..., metrics=..., max_new_tokens=..., out_dir=...)` | Both models on the same problems, prompts, sampling settings, and seeds, with paired differences and 95% intervals. `out_dir` records every problem and resumes an interrupted comparison. |
| Deliver | `model.export(path)` | A Hugging Face model directory ([Exporting a model](#exporting-a-model)). |
| | `edit.export(path)` | The exact patch and its provenance. |

**Targets and operators.** A target is a tensor name with an optional NumPy-style index: `"layers.14.self_attn.o_proj[:, 896:1024]"` selects every row and columns 896 to 1,023 (slices take step 1 only). A short name resolves to the one tensor whose name ends with it, and a module's name means its weight. The operators are `scale_slice` (`factor`), `set_slice` (`values`), `fill_slice` (`value`), `project_out` (`direction`, `columns`, `alpha`), and `add_low_rank` (`u`, `v`, `alpha`); a name without a version means version 1 ([extensions.md](extensions.md) gives each contract). Tensors that a family descriptor protects, and shared parameters, refuse every edit ([Roles, heads, and protections](#roles-heads-and-protections)).

**Edits.** An `Edit` keeps the new model in memory, so `edit.model` can generate, evaluate, and probe before anything is stored. `edit.table()` lists each changed tensor with its changed entries, relative L2 change, and hashes. `edit.verify()` replays the patch on the original without running the operator and checks every hash. `edit.save()` stores the edited model as a checkpoint with its patch, and returns the edit with `patch_id` and `edit.model.checkpoint_id` filled in.

**Direct use.** A `Model` belongs to no experiment and no budget. Imports and saved edits are recorded in the workspace's audit log; an edit kept in memory is not. Agents work through `Experiment` instead ([experiment-api.md](experiment-api.md)).

**The limit-hit rate** is the share of answers cut off at the token limit. It is not a repetition rate: an answer can repeat itself and still finish, or reach the limit while still making progress.

### The quickstart

[examples/model_quickstart.py](../examples/model_quickstart.py) runs each step on Qwen3-0.6B: it loads the model at commit `c1899de289a04d12100db370d81485cdf75e47ca`, inspects one attention layer, generates one answer, zeroes the 128 input columns of attention head 7 in layer 14's output projection with `fill_slice`, verifies and saves the edit, compares the edited model with the original on 8 of the 16 problems in [examples/data/math_dev.jsonl](../examples/data/math_dev.jsonl) at 1,024 tokens, and exports the patch and the edited model. The [README](../README.md#use-the-python-api) walks through it. Run it in a directory that holds `math_dev.jsonl`, with WSE installed (`pip install "wse[models-hf]"`):

```bash
curl -fLO https://raw.githubusercontent.com/xuguang-ai/weight-space-engineering/main/examples/data/math_dev.jsonl
curl -fLO https://raw.githubusercontent.com/xuguang-ai/weight-space-engineering/main/examples/model_quickstart.py
python model_quickstart.py
```

On an M4 Pro with MPS, an earlier version of the script, which took the model from a local copy of the same commit, ran in 140 to 149 seconds in each of three runs. The edit and the comparison printed:

```text
1 tensor changed, 131,072 entries; weights 92865a3c8750 -> 092773d60909
  model.layers.14.self_attn.o_proj.weight: 131,072 of 2,097,152 entries, relative L2 change 0.2756, 70e83e3c4328 -> ad2d2c42779e
the patch replays to the edited weights: True
```

```text
8 problems, 8 samples per model
metric                        base   candidate   difference [95% interval]
accuracy                     0.125       0.125   +0.000 [+0.000, +0.000]
limit_hit_rate               0.750       0.625   -0.125 [-0.375, +0.000]
mean_generated_tokens        915.6       866.5   -49.1 [-157.4, +24.5]
```

Each model answered the 8 problems in one batch of 46 to 49 seconds. Both answered 1 of the 8 correctly; 6 of the original's answers and 5 of the edited model's reached the limit, and that interval includes zero. The ablation shows how the API works; it is not evidence that the head matters. The three runs, each in a new directory, printed the same hashes and the same table.

## Support levels

| Level | What it is | When it is needed |
| --- | --- | --- |
| **0: the generic adapter `hf_causal_lm@1`** | Import, identity, exact patches, analyses, probes, and generation for any supported architecture, with no model-specific code | Always the base: `transformers` implements the architecture, and the checkpoint names equal its module's parameter names |
| **1: a family descriptor** | Registered data, selected by `config.json`'s `model_type`: a name map, semantic roles, head groupings, and protected roles | The checkpoint names need mapping (multimodal wrappers), or structure-aware tools (roles, heads, protections) are wanted |
| **A new adapter** | Code, under the extension contract in [extensions.md](extensions.md) | `transformers` does not implement the architecture |

At Level 0 the adapter instantiates the module on the meta device and takes every tensor's axes from its module type, never from its name:

- an `nn.Linear` weight is `[out_features, in_features]`;
- an `nn.Embedding` weight is `[num_embeddings, embedding_dim]`;
- a normalization weight is `[features]`;
- anything else is `unknown`.

Tied input and output embeddings are one object in the module, so they form an *alias group*: stored once, counted once, and never edited. Every tensor runs in its stored dtype. Qwen3.5-0.8B, for example, keeps 36 F32 tensors that `from_pretrained(dtype="auto")` would cast to BF16.

## Importing a model

The importer reads only allow-listed files at the snapshot root:

- `config.json` and `generation_config.json`;
- `*.safetensors` and its index;
- the tokenizer files and the chat template.

It never unpickles a file or runs repository code, and it refuses a `config.json` that declares `auto_map`. The manifest records the SHA-256 of every file read. A model's identity is its weights hash (over per-tensor blob IDs), configuration hash, tokenizer, chat-template, and generation-configuration hashes, and the source repository and revision. Importing the same snapshot again returns the stored checkpoint.

From Python:

```python
from wse import Model

# A Hugging Face repository: the revision is resolved to its commit, and only the allow-listed files are downloaded.
model = Model.from_pretrained("Qwen/Qwen3-0.6B")
print(model.source.revision)  # the commit "main" resolved to

# A local snapshot names its source for the provenance record.
model = Model.from_pretrained("~/models/Qwen3-0.6B", repo="Qwen/Qwen3-0.6B",
                              revision="c1899de289a04d12100db370d81485cdf75e47ca")
```

A snapshot in the Hugging Face cache (`models--ORG--NAME/snapshots/SHA`) names its repository and commit by its path. From the command line:

```bash
uv run --extra models-hf wse import-hf ~/models/Qwen3-0.6B --repo Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca
```

Gated repositories (Llama, Gemma) need access granted on the Hub and the user's own login (`hf auth login`). WSE never reads or stores the token.

## Exporting a model

`model.export(path)` writes a Hugging Face model directory into `path`, which must be new or empty:

- `config.json`, and the tokenizer, chat-template, and generation files as imported;
- `model.safetensors`, with every tensor under its checkpoint name (each name of a shared parameter as its own copy);
- `wse_provenance.json`: the checkpoint, its weights hash, the snapshot it came from, and the parent checkpoint and patch it derives from.

`Model.from_pretrained(path)` imports the directory again with the same weights hash. `edit.export(path)` writes the edit instead: `patch.wsepatch`, the exact patch with its after-values, and `patch.json`, its provenance (the original's weights hash and source, each step's operator, tensor, and selection, and the result's weights hash). `model.apply_patch(path, format="wse-patch-v1")` replays it onto the same original, in any workspace.

Checked on 2026-11-12 with the [quickstart's](#the-quickstart) exports of Qwen3-0.6B:

- the exported model, imported into a new workspace, has the edited model's weights hash, `sha256:092773d6…`;
- `transformers` loads it as `Qwen3ForCausalLM`, and all 311 of its tensors equal WSE's;
- the exported patch, applied to the original imported into another new workspace, gives the same weights hash, and `verify()` holds.

**Qwen3.5 exports its language model only.** Qwen3.5-0.8B's snapshot is an image-text model. WSE imports its language model, 321 tensors, and leaves out 168 tensors of its vision encoder and its multi-token prediction (`inspect().dropped_tensors`). An export therefore holds the language model only, with the original `config.json`. Checked on 2026-11-12 with an export of the unedited model:

- `Model.from_pretrained` imports it again with the same weights hash;
- `transformers`' `AutoModelForCausalLM` loads it as the text model `Qwen3_5ForCausalLM`, with every language-model weight (with `dtype="auto"` it loads the 36 tensors stored in F32 as BF16, as it does for the original snapshot);
- loaded as the image-text model, with `AutoModelForImageTextToText`, the export lacks the 153 tensors of the vision encoder, which `transformers` initializes at random, with a warning. Use the export as a text model.

## Roles, heads, and protections

A family descriptor names the tensors that play each role:

- per layer: `attn.q`, `attn.k`, `attn.v`, `attn.o`, `linear_attn.in`, `linear_attn.out`, `mlp.gate`, `mlp.up`, `mlp.down`, `norm.input`, `norm.post_attention`;
- once per model: `embed`, `lm_head`, `final_norm`.

A head grouping says how a role's tensor splits into attention heads.

```python
down = model.role("mlp.down", 15)    # a TensorRole
down.tensor                          # 'model.language_model.layers.15.mlp.down_proj.weight' (Qwen3.5)
down.site + ":input"                 # a probe site: the MLP activations entering down_proj
head = model.role("attn.o", 15)
model.edit((head, head.selection(head=7)), "fill_slice", value=0.0)   # ablate one attention head
```

The built-in descriptors protect `embed`, `lm_head`, and `final_norm`. Every edit operator refuses those tensors under every policy, with `POLICY_DENIED`; the error lists the editable roles. Exact replay and the operator's import of a published patch write recorded values and check no protection, so that published edits reproduce exactly.

A released descriptor is frozen: its version is recorded in each model's configuration, and so in the model's identity. Changing its roles or protections therefore needs a new version. A test pins the content of every built-in descriptor.

Built-in descriptors:

- **`qwen3_5@1`**: maps Qwen3.5's multimodal checkpoint (`model.language_model.*`) onto the text-only `Qwen3_5ForCausalLM` and drops the vision tower and the multi-token-prediction tensors. Its roles cover the full-attention layers, the Gated DeltaNet linear-attention layers (`linear_attn.in`, `linear_attn.out`), and the MLPs. The full-attention `q_proj` holds each head's query and output gate side by side.
- **`llama@1`**: roles only. Llama checkpoints name the module's parameters.

## Writing a family descriptor

A descriptor is data. Adding a family registers one and changes no core code. This one gives Qwen3 models roles:

```python
from wse.adapters.families import FamilyDescriptor, HeadGrouping, RoleRule

LAYER = r"model\.layers\.(?P<layer>\d+)\."
QWEN3_ROLES = FamilyDescriptor(
    family_id="qwen3_roles", version="1", model_types=["qwen3"],
    roles=[
        RoleRule(pattern=r"model\.embed_tokens\.weight", role="embed"),
        RoleRule(pattern=r"lm_head\.weight", role="lm_head"),
        RoleRule(pattern=r"model\.norm\.weight", role="final_norm"),
        RoleRule(pattern=LAYER + r"self_attn\.o_proj\.weight", role="attn.o",
                 heads=HeadGrouping(num_heads_key="num_attention_heads", head_dim_key="head_dim", axis=1)),
        RoleRule(pattern=LAYER + r"mlp\.down_proj\.weight", role="mlp.down"),
    ],
    protected_roles=["embed", "lm_head", "final_norm"],
)


def setup(registry):
    registry.register("family", "qwen3_roles", "1", QWEN3_ROLES)
```

Rules:

- a pattern must match whole canonical tensor names;
- a per-layer role has a named group `layer`, and a model-wide role has none;
- at most one rule may match a tensor, and at most one tensor may hold a role in a layer;
- every protected role needs a rule.

The import checks the rules against the module before it reads any tensor data. A descriptor that does not fit is refused with `INVALID_MODEL`.

A registered descriptor is selected automatically when it is the only one that accepts the configuration's `model_type`. Otherwise name it on import: `import_hf_snapshot(..., family="qwen3_roles@1")`, or `wse import-hf ... --family qwen3_roles@1`. Processes pick it up through setup hooks (`--setup my_extensions:setup`, see [extensions.md](extensions.md)).

## Verified models

Every check below ran on an Apple M4 Pro with PyTorch 2.15.0 and `transformers` 5.22.2.

"Logits" means bitwise-equal logits against `from_pretrained` with the stored dtypes, on CPU and on MPS (maximum absolute difference 0.0).

| Model | Revision | Level | Checks |
| --- | --- | --- | --- |
| `Qwen/Qwen3.5-0.8B` | `2fc06364715b` | 1 (`qwen3_5@1`) | Import (321 tensors, 752,393,024 parameters); logits; exact replay of the three published edits; generation evaluation; probes; managed experiments ([m2-q1-report.md](../dev/m2-q1-report.md)) |
| `meta-llama/Llama-3.2-1B-Instruct` | `9213176726f5` | 1 (`llama@1`) | Import (1,235,814,400 parameters); logits; a patch round trip; GSM8K generation (checked at Level 0 before `llama@1` existed; [m2-q1-report.md](../dev/m2-q1-report.md)) |
| `Qwen/Qwen3-0.6B` | `c1899de289a0` | 0 | Import (311 tensors, 596,049,920 parameters, tied embeddings); logits; a patch round trip (edit, replay, store, reopen); the analyses `stats`, `tensor_norms`, `svd_top_k`, and `direction_projection`; probes equal to `output_hidden_states` at the layer boundaries; generation; the [quickstart](#the-quickstart), with exports imported again and loaded by `transformers` ([Exporting a model](#exporting-a-model)) |
| `google/gemma-3-1b-it` | `dcc83ea841ab` | 0 | The same checks as Qwen3-0.6B (341 tensors, 999,885,952 parameters, tied embeddings) |

The Qwen3 and Gemma checks needed no model-specific code (M2 plan, Q2 exit).

## Known limits

- **Qwen3.5 on the CPU.** `transformers` runs Gated DeltaNet through reference PyTorch code there, where a decoding step takes about 5–10 s, about 170 times slower than on MPS. Use `mps` on Apple silicon.
- **Probes.** A probe reads at most 16,384 positions per text in direct use. Under the agent policy `qwen_loop_v1`, a probe text has at most 1,024 tokens.
- **Export.** An export of a whole managed experiment (`wse export`) on an imported language model holds its base checkpoint and every binary patch, and `wse replay` rebuilds every candidate byte for byte; the test suite checks this on a tiny Llama model. Patches of open-weight models replay exactly, and imports are hash-checked. A Qwen3.5 model exports its language model only ([Exporting a model](#exporting-a-model)).
- **Reproducibility.** Bitwise reproducibility holds within one device and library version, not across them. Edits are computed on the CPU in a platform-independent way, so patches are exact everywhere.
