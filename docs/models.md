# Models

WSE works on causal language models that the pinned `transformers` (5.22.2) implements natively, from safetensors weights in BF16, F16, or F32 that fit the host's memory. Excluded:

- architectures that need `trust_remote_code`;
- quantized formats (GGUF, GPTQ, AWQ, MXFP4, ...);
- encoder–decoder models.

Mixture-of-experts tensors stay readable and editable through the generic tools, without per-expert semantics. The small MLP models of milestone M1 use their own adapter, `mlp@1`.

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
from wse import WSE

# A Hugging Face repository: the revision is resolved to its commit, and only the allow-listed files are downloaded.
model = WSE.from_pretrained("Qwen/Qwen3-0.6B")

# A local snapshot names its source for the provenance record.
model = WSE.from_pretrained("~/models/Qwen3-0.6B", repo="Qwen/Qwen3-0.6B",
                            revision="c1899de289a04d12100db370d81485cdf75e47ca")
```

A snapshot in the Hugging Face cache (`models--ORG--NAME/snapshots/SHA`) names its repository and commit by its path. From the command line:

```bash
uv run --no-sync wse import-hf ~/models/Qwen3-0.6B --repo Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca
```

Gated repositories (Llama, Gemma) need access granted on the Hub and the user's own login (`hf auth login`). WSE never reads or stores the token.

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
| `Qwen/Qwen3-0.6B` | `c1899de289a0` | 0 | Import (311 tensors, 596,049,920 parameters, tied embeddings); logits; a patch round trip (edit, replay, store, reopen); the analyses `stats`, `tensor_norms`, `svd_top_k`, and `direction_projection`; probes equal to `output_hidden_states` at the layer boundaries; generation |
| `google/gemma-3-1b-it` | `dcc83ea841ab` | 0 | The same checks as Qwen3-0.6B (341 tensors, 999,885,952 parameters, tied embeddings) |

The Qwen3 and Gemma checks needed no model-specific code (M2 plan, Q2 exit).

## Known limits

- **Qwen3.5 on the CPU.** `transformers` runs Gated DeltaNet through reference PyTorch code there, where a decoding step takes about 5–10 s, about 170 times slower than on MPS. Use `mps` on Apple silicon.
- **Probes.** A probe reads at most 16,384 positions per text in direct use. Under the agent policy `qwen_loop_v1`, a probe text has at most 1,024 tokens.
- **Export.** Exporting and replaying a whole managed experiment (`wse export`, `wse replay`) carries M1's small models only. Patches of open-weight models replay exactly, and imports are hash-checked.
- **Reproducibility.** Bitwise reproducibility holds within one device and library version, not across them. Edits are computed on the CPU in a platform-independent way, so patches are exact everywhere.
