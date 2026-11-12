# Weight-Space Engineering (WSE)

[![PyPI](https://img.shields.io/pypi/v/wse)](https://pypi.org/project/wse/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/wse/)
[![License](https://img.shields.io/pypi/l/wse)](https://github.com/xuguang-ai/weight-space-engineering/blob/main/LICENSE)

**Give AI agents tools to inspect, edit, and test model weights.**

WSE is a Python framework and MCP harness for experiments that edit a model's weights directly. You state a goal and how it is measured. An agent investigates the model and proposes weight edits; WSE evaluates candidates on the same problems, selects one by your rule, freezes it, and only then tests it on problems the agent never saw. Every edit is an exact patch that replays to identical weights, so each result can be checked and reproduced.

[Install](https://github.com/xuguang-ai/weight-space-engineering#install) · [Quick start](https://github.com/xuguang-ai/weight-space-engineering#quick-start) · [Python API](https://github.com/xuguang-ai/weight-space-engineering#use-the-python-api) · [Agents](https://github.com/xuguang-ai/weight-space-engineering#how-the-agent-works) · [Results](https://github.com/xuguang-ai/weight-space-engineering#results) · [Documentation](https://github.com/xuguang-ai/weight-space-engineering#documentation)

## Install

```bash
pip install "wse[models-hf]"
```

WSE needs Python 3.11 or later; `models-hf` adds the Hugging Face model support. An agent experiment, on macOS for now, also needs OpenAI's [Codex CLI](https://github.com/openai/codex), release 0.167 or later, which runs the agent:

```bash
npm install -g @openai/codex
```

## Quick start

An agent edits Qwen3-0.6B's weights so that it finishes more of its math answers within 1,024 tokens. WSE evaluates every candidate on the same 16 development problems, keeps the one that cuts off the fewest answers without losing accuracy, freezes that choice, and only then tests it against the original on 16 problems the agent never saw.

**1. Download the problems**: 16 GSM8K training problems to develop on and 16 GSM8K test problems, one JSON object per line ([format](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/experiment-api.md#data)).

```bash
curl -fLO "https://raw.githubusercontent.com/xuguang-ai/weight-space-engineering/main/examples/data/math_{dev,test}.jsonl"
```

**2. Set your key.** WSE reads your model provider's key and endpoint from the environment. Codex reaches the provider through a local proxy that WSE runs and that adds the key, so neither Codex nor the agent's code holds it.

```bash
export OPENAI_API_KEY=your-api-key
export OPENAI_BASE_URL=https://api.openai.com/v1
```

For Azure OpenAI, set `OPENAI_BASE_URL` to your resource's endpoint, `https://YOUR-RESOURCE.services.ai.azure.com/openai/v1`, and use your deployment's name in step 3.

**3. Run it.** Save this code as `fix_loops.py` next to the problems, put your model's name, or your deployment's, in place of `YOUR_AGENT_MODEL` (the recorded runs used `gpt-5.6-luna`), and run `python fix_loops.py`:

```python
from wse import Experiment

experiment = Experiment(
    model="Qwen/Qwen3-0.6B",
    goal="Reduce repetitive reasoning so the model finishes more math answers.",
    dev="math_dev.jsonl",
    evaluator="math",
    minimize="limit_hit_rate",
    preserve=["accuracy"],
    max_new_tokens=1024,
    workspace="runs/fix-loops",
)
result = experiment.run(
    agent="codex:YOUR_AGENT_MODEL",
    sandbox="python",
    max_evaluations=10,
)
print(result.summary())

report = result.test("math_test.jsonl")  # only the frozen choice and the original, on new problems
print(report.compare())
result.export("artifacts/fix-loops")
```

**4. Read the result.** `result.summary()` reports what the agent tried, what WSE kept, and why; `report.compare()` reports the test. The run takes about 15 minutes on an M4 Pro. In the recorded run of this configuration on WSE's final code (2026-11-12, `gpt-5.6-luna`), the agent scaled half of the channels of a late MLP layer's down-projection, and WSE kept the candidate that scales layer 27's by 0.8:

| 16 problems in each set | Original | WSE's choice |
| --- | --- | --- |
| Development problems answered correctly | 5 | 7 |
| Development answers cut off at 1,024 tokens | 10 | 8 |
| Test problems answered correctly | 4 | 3 |
| Test answers cut off at 1,024 tokens | 11 | 12 |

The choice improved the problems it was chosen on, not the new ones; the other recorded run ended the same way ([Results](https://github.com/xuguang-ai/weight-space-engineering#an-agent-edits-a-small-language-model)).

- **`goal` guides; `minimize` and `preserve` decide.** A candidate qualifies when its development limit-hit rate is below the original's and its development accuracy is at least the original's; WSE keeps the qualifying candidate with the lowest limit-hit rate, and when none qualifies, `result.model` is the original. These are point estimates, not significance tests.
- **The evaluation is fixed.** Every model is evaluated on the same problems, prompts, sampling settings, seeds, and token limit, and the agent cannot change the grading. `max_evaluations` counts the evaluations of whole models on the development problems, the original's included.
- **The limit-hit rate** is the share of answers cut off at the token limit. It is not a repetition rate.

[How the agent works](https://github.com/xuguang-ai/weight-space-engineering#how-the-agent-works) explains what Codex can and cannot reach; the [Experiment API](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/experiment-api.md) covers every argument, where results go, test isolation, and what WSE does and does not verify.

## Use the Python API

A `Model` is one version of a model in a workspace: load it, inspect it, generate, edit its weights, and compare the edit with the original. The blocks below are one script, [examples/model_quickstart.py](https://github.com/xuguang-ai/weight-space-engineering/blob/main/examples/model_quickstart.py), run in the directory that holds `math_dev.jsonl` from step 1 above. On an M4 Pro it takes about two and a half minutes after a download of 1.5 GB, and it writes 2.7 GB to `runs/model-quickstart`, 1.5 GB of it the exported model; delete that directory to run it again. Checkpoint and patch IDs differ per run.

### Load and inspect

`from_pretrained` resolves the revision to a commit, records it, and imports the weights into a workspace; without `revision`, it takes the latest commit. `tensors` lists names, shapes, and dtypes; `tensor` returns a copy.

```python
from wse import Model, compare_models

# 1. Load: the snapshot at a resolved commit, imported into a workspace.
original = Model.from_pretrained(
    "Qwen/Qwen3-0.6B",
    revision="c1899de289a04d12100db370d81485cdf75e47ca",
    workspace="runs/model-quickstart/workspace",
)
print(original.inspect())

# 2. Inspect: the tensors of one attention layer, and a copy of one weight (changing the copy changes nothing).
for entry in original.tensors("layers.14.self_attn.*"):
    print(entry)
o_proj = original.tensor("layers.14.self_attn.o_proj")
print(f"norm of head 7's input columns in layer 14's o_proj: {o_proj[:, 896:1024].float().norm():.2f}")
```

```text
model.layers.14.self_attn.k_norm.weight [128] bfloat16
model.layers.14.self_attn.k_proj.weight [1024, 1024] bfloat16
model.layers.14.self_attn.o_proj.weight [1024, 2048] bfloat16
model.layers.14.self_attn.q_norm.weight [128] bfloat16
model.layers.14.self_attn.q_proj.weight [2048, 1024] bfloat16
model.layers.14.self_attn.v_proj.weight [1024, 1024] bfloat16
norm of head 7's input columns in layer 14's o_proj: 8.75
```

<details>
<summary>What <code>inspect()</code> printed before these lines</summary>

```text
model Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca, checkpoint ckpt_b67e9a97d644396f3cd4c1f151e48338, weights 92865a3c8750
adapter hf_causal_lm@1, family none (Level 0), model type qwen3, 28 layers
596,049,920 parameters in 311 tensors (311 bfloat16)
tensors:
  lm_head.weight  x1  [151936, 1024] bfloat16  155,582,464 parameters
  model.embed_tokens.weight  x1  [151936, 1024] bfloat16  155,582,464 parameters
  model.layers.{i}.input_layernorm.weight  x28  [1024] bfloat16  28,672 parameters
  model.layers.{i}.mlp.down_proj.weight  x28  [1024, 3072] bfloat16  88,080,384 parameters
  model.layers.{i}.mlp.gate_proj.weight  x28  [3072, 1024] bfloat16  88,080,384 parameters
  model.layers.{i}.mlp.up_proj.weight  x28  [3072, 1024] bfloat16  88,080,384 parameters
  model.layers.{i}.post_attention_layernorm.weight  x28  [1024] bfloat16  28,672 parameters
  model.layers.{i}.self_attn.k_norm.weight  x28  [128] bfloat16  3,584 parameters
  model.layers.{i}.self_attn.k_proj.weight  x28  [1024, 1024] bfloat16  29,360,128 parameters
  model.layers.{i}.self_attn.o_proj.weight  x28  [1024, 2048] bfloat16  58,720,256 parameters
  model.layers.{i}.self_attn.q_norm.weight  x28  [128] bfloat16  3,584 parameters
  model.layers.{i}.self_attn.q_proj.weight  x28  [2048, 1024] bfloat16  58,720,256 parameters
  model.layers.{i}.self_attn.v_proj.weight  x28  [1024, 1024] bfloat16  29,360,128 parameters
  model.norm.weight  x1  [1024] bfloat16  1,024 parameters
protected (every edit is refused): none
shared parameters: lm_head.weight = model.embed_tokens.weight
```

</details>

### Generate

```python
# 3. Generate: one answer, how many tokens it took, and why it ended.
answer = original.generate("What is 17 * 24?", max_new_tokens=2048)
print(f"{answer.generated_tokens} tokens, finish reason {answer.finish_reason}")
print(answer.text.split("</think>")[-1].strip())  # the answer after the model's thinking
```

```text
1079 tokens, finish reason stop
```

The model thinks before it answers: the answer ends with `\boxed{408}` at token 1,079. With a limit of 1,024 tokens, the same answer is cut off before its box, with the finish reason `length`; that is what the limit-hit rate counts.

<details>
<summary>The answer after the model's thinking</summary>

```text
To solve the multiplication problem $17 \times 24$, we can break it down into simpler components for clarity and accuracy.

---

### Step-by-Step Breakdown:

1. **Break 17 into 20 - 3**:
   $$
   17 \times 24 = (20 - 3) \times 24
   $$

2. **Apply the distributive property**:
   $$
   (20 - 3) \times 24 = 20 \times 24 - 3 \times 24
   $$

3. **Calculate each part**:
   - $20 \times 24 = 480$
   - $3 \times 24 = 72$

4. **Subtract**:
   $$
   480 - 72 = 408
   $$

---

### Final Answer:

$$
\boxed{408}
$$
```

</details>

### Edit

`edit` applies a registered operator in memory and returns an exact patch; the original never changes. Here `fill_slice` zeroes the 128 input columns of attention head 7 in layer 14's output projection, an ablation of one head. `verify` replays the patch without running the operator and checks every hash; `save` stores the edited model in the workspace.

```python
# 4. Edit: zero the 128 input columns of attention head 7 in layer 14's output projection.
edit = original.edit(
    "layers.14.self_attn.o_proj[:, 896:1024]",
    "fill_slice",
    value=0.0,
)
print(edit.table())
print(f"the patch replays to the edited weights: {edit.verify()}")
edited = edit.save().model
print(f"saved: patch {edit.patch_id}, checkpoint {edited.checkpoint_id}")
```

```text
1 tensor changed, 131,072 entries; weights 92865a3c8750 -> 092773d60909
  model.layers.14.self_attn.o_proj.weight: 131,072 of 2,097,152 entries, relative L2 change 0.2756, 70e83e3c4328 -> ad2d2c42779e
the patch replays to the edited weights: True
saved: patch patch_bb8fab2e1dbaf10f02bd660b545d19c1, checkpoint ckpt_0da05bb923c740440677d84f26d003b5
```

The registered operators scale, set, or fill a slice, remove a direction's projection, or add a low-rank term; `replace_weights` submits whole tensors computed any other way, after checking their names, shapes, dtypes, and values ([the Model API](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/models.md#the-model-api)).

### Compare

`compare_models` runs both models on the same problems, prompts, sampling settings, and seeds, and reports paired differences with 95% intervals. `out_dir` records every problem, so an interrupted comparison resumes where it stopped.

```python
# 5. Compare: the same problems, prompts, sampling settings, and seeds for both models.
comparison = compare_models(
    base=original,
    candidate=edited,
    data="math_dev.jsonl",
    evaluator="math",
    metrics=["accuracy", "limit_hit_rate", "mean_generated_tokens"],
    max_new_tokens=1024,
    problems=8,
    out_dir="runs/model-quickstart/comparison",
)
print(comparison.table())
```

```text
8 problems, 8 samples per model
metric                        base   candidate   difference [95% interval]
accuracy                     0.125       0.125   +0.000 [+0.000, +0.000]
limit_hit_rate               0.750       0.625   -0.125 [-0.375, +0.000]
mean_generated_tokens        915.6       866.5   -49.1 [-157.4, +24.5]
```

On 8 of the 16 problems in [examples/data/math_dev.jsonl](https://github.com/xuguang-ai/weight-space-engineering/blob/main/examples/data/math_dev.jsonl), both models answered 1 correctly. At 1,024 tokens, 6 of the original's answers and 5 of the edited model's were cut off, and that interval includes zero: the ablation shows how the API works, not that this head matters. Each model answered the 8 problems in one batch of 46 to 49 seconds, over three runs that printed the same table.

### Export

```python
# 6. Export: the exact patch with its provenance, and the edited model as a Hugging Face directory.
print(f"patch: {edit.export('runs/model-quickstart/patch')}")
print(f"model: {edited.export('runs/model-quickstart/edited-model')}")
```

```text
patch: runs/model-quickstart/patch
model: runs/model-quickstart/edited-model
```

The model directory holds `config.json`, the tokenizer, `model.safetensors`, and `wse_provenance.json`. Imported again with `from_pretrained`, it has the edited model's weights hash, and `transformers` loads it as `Qwen3ForCausalLM` with all 311 tensors equal; the patch replays onto the original in another workspace ([exporting](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/models.md#exporting-a-model)).

The same calls run the full-size case on Qwen3.5-0.8B, which reconstructs a published edit and measures it on 32 problems ([Results](https://github.com/xuguang-ai/weight-space-engineering#results)).

## How the agent works

`Experiment` hands one model to an agent under a fixed evaluation. The agent is the [Codex CLI](https://github.com/openai/codex). WSE starts it in the experiment's working directory and connects it to the run's WSE tools over MCP; the agent can also write and run Python there, under Codex's sandbox, which WSE configures so that the agent can read the model's weights but not the test data, WSE's database, or your keys. WSE records each candidate the agent submits as an exact patch and evaluates the ones the agent asks it to; when the agent stops, WSE selects among the evaluated candidates by the rule and freezes the choice before any test. Because the agent's own code can compute anything from the weights it reads, WSE meters only its own calls and labels these runs `unverified`.

WSE runs the Codex binary that `WSE_CODEX_PATH` names, else the `codex` on your `PATH` (for an npm install, the native binary that its launcher starts), else the newest one bundled with OpenAI's VS Code extension. On macOS it first checks that the binary's code signature names OpenAI, and it skips releases older than 0.167 ([details](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/experiment-api.md#install)).

[Experiment API](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/experiment-api.md): the selection rule, test isolation, what is verified, and current limits · [Other MCP clients](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/experiments.md#serving-a-run-to-an-mcp-client) · [Policies and budgets](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/experiments.md#budgets)

## Results

Every number below comes from a recorded run ([results](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/results.md)); each says which controller made the edit.

### Reconstructing a published edit on Qwen3.5-0.8B

Asked 32 GSM8K training problems with room for 8,192 new tokens, Qwen3.5-0.8B answered 2 correctly and ran 30 to the limit; 24 of those 30 ended by repeating the same lines. [examples/p50c_four_steps.py](https://github.com/xuguang-ai/weight-space-engineering/blob/main/examples/p50c_four_steps.py) diagnoses this, probes two MLP layers, edits them with WSE's `project_out` operator, and compares the edited model with the original, all from Python:

| 32 GSM8K problems | Original | Edited | Paired difference [95% interval] |
| --- | --- | --- | --- |
| Accuracy | 2/32 (0.062) | 10/32 (0.312) | +0.250 [+0.125, +0.406] |
| Answers cut off at the limit | 30/32 (0.938) | 21/32 (0.656) | −0.281 [−0.438, −0.125] |
| Mean generated tokens | 8,063.5 | 7,082.0 | −981.5 [−1,597.1, −434.7] |

**The edit is reconstructed, not discovered.** Its layers, its 64 columns in each, and its direction come from p50c, the edit that GPT-6 Astra made in [Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang, and Khattab, 2026); WSE's operator re-executes it, and 91% of the edited entries are byte-identical to p50c's. None of the 21 answers still cut off ends in a loop: they keep re-checking the problem in new words until the limit, so the limit-hit rate fell less than the loops did. The script took 1 hour 54 minutes on an M4 Pro ([walkthrough, downloads, and every step's output](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/results.md#3-reconstructing-p50c-in-four-steps)).

### The published edit on 25 MATH-500 problems

Applied as published, p50c was compared with the original on 25 MATH-500 problems with the article's sampling (`qwen35think@1`, up to 32,768 new tokens), one sample per problem:

| 25 MATH-500 problems | Original | p50c | Paired difference [95% interval] |
| --- | --- | --- | --- |
| Accuracy | 3/25 (12%) | 13/25 (52%) | +0.40 [+0.16, +0.60] |
| Answers cut off at the limit | 23/25 (92%) | 9/25 (36%) | −0.56 [−0.76, −0.32] |
| Mean generated tokens | 30,824 | 21,554 | −9,270 [−13,854, −4,753] |

Eleven problems were solved only with p50c, and one only with the original. The article reports 7.9% → 48.1% accuracy and 93.0% → 33.1% of answers cut off, with 4 samples on all 500 problems; twenty-five problems confirm the direction of that result, not its size ([reproduce it](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/results.md#reproduce-the-comparison), about six hours).

### A live agent on a small network

An LLM agent, `gpt-5.6-luna` through the OpenAI Agents SDK runner that WSE used before Codex, repaired a 22-parameter classifier with a planted fault in 5 of 5 acceptance trials, working only through WSE's MCP tools: final accuracy 100% on 1,024 examples it never saw, against 50% before. A single run on 2026-11-11 made 8 model requests and 11 tool calls in 20 seconds, with 68,866 input and 1,428 output tokens, and changed 4 of the 22 parameters with one `set_slice` ([trial report](https://github.com/xuguang-ai/weight-space-engineering/blob/main/dev/m1-agent-trials.md), [the example](https://github.com/xuguang-ai/weight-space-engineering/blob/main/examples/signed_sum/agent_api.py)). One small fixture says little about agent capability in general.

### An agent edits a small language model

Through `Experiment`, Codex with `gpt-5.6-luna` worked on Qwen3-0.6B's answers to 16 GSM8K training problems at 1,024 new tokens, to cut fewer answers off at the limit without losing accuracy ([examples/experiment_quickstart.py](https://github.com/xuguang-ai/weight-space-engineering/blob/main/examples/experiment_quickstart.py)). In 13 minutes it probed the residual stream, computed 8 candidates in Python, each scaling down 32 to 128 entries of the final RMSNorm weight (one also scaled layer 27's input norm), evaluated 7, and stopped itself. WSE's rule chose the one that halves 64 entries: on the development problems 5 → 7 correct and 10 → 7 cut off. On 16 GSM8K test problems, none of which appears in anything the agent was sent or wrote, it did not help: 4 → 3 correct and 11 → 12 cut off (accuracy difference −0.062 [−0.188, +0.000]). A second run on this version's code chose another edit, through WSE's `scale_slice` operator: half of layer 27's MLP down-projection channels scaled by 0.8, 5 → 7 correct and 10 → 8 cut off on the development problems, and again 4 → 3 correct and 11 → 12 cut off on the test problems. The runs show the whole path working with a live agent, not an edit that generalizes ([the runs](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/results.md#6-a-live-llm-agent-edits-a-language-model-through-experiment)).

### What has been demonstrated

| Experiment | Controller | Recorded result |
| --- | --- | --- |
| Repair a planted fault in a 22-parameter network | Live LLM agent over MCP | 5/5 acceptance trials repaired it, from 50% to 100% final accuracy |
| Apply published p50c to Qwen3.5-0.8B | Command line, same runner as Python | 3/25 → 13/25 correct on MATH-500, with a 32,768-token limit |
| Reconstruct p50c's projection edit with WSE operators | Direct Python | 2/32 → 10/32 correct on GSM8K practice problems; answers cut off at the limit fell from 30/32 to 21/32 |
| Apply that reconstruction through MCP | Scripted controller | Answers cut off at the limit fell from 30/32 to 19/32 on another practice set |
| An agent edits a language model (`Experiment` with Codex) | Live LLM agent | Two runs on Qwen3-0.6B: 5/16 → 7/16 correct on the development problems each choice was made on, 4/16 → 3/16 on test problems |

The article's three Qwen3.5-0.8B edits import to their published hashes, and WSE's own `fill_slice@1` operator reproduces one of them, p25c, hash-exactly. The reconstruction takes its direction and columns from p50c's published change; a scripted search that used its form without its values found no repair. Two live agent runs have edited a language model, and in both the gain on the development problems did not carry over to the test problems.

## Current support

| Area | Status |
| --- | --- |
| Agent interface | `Experiment` with the Codex CLI (two live runs, on Qwen3-0.6B); the MCP server for other MCP clients |
| Language-model weights | Unquantized safetensors for causal models implemented natively by `transformers` |
| Verified model imports | Qwen3.5-0.8B, Llama-3.2-1B-Instruct, Qwen3-0.6B, and gemma-3-1b-it |
| Generation and activation probes | Apple MPS and CPU, tested on macOS with Apple silicon |
| Agent sandbox | macOS |
| Tested versions | Python 3.14, PyTorch 2.15, and `transformers` 5.22.2; WSE requires Python 3.11 or later |
| Patch replay | Exact replay of stored edits, verified by hashes |
| Model export | Hugging Face model directories; Qwen3.5 exports its language model only, without its vision encoder |
| Next work | Live agent runs on Qwen3.5-0.8B with `Experiment`, larger development and test sets, and CUDA support ([handover](https://github.com/xuguang-ai/weight-space-engineering/blob/main/dev/handover-cuda.md)) |

Local language-model evaluation is slow: a 32-problem generation pass at 8,192 tokens took about 37–40 minutes on an M4 Pro.

## Documentation

- [Experiment API](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/experiment-api.md): run an agent with a goal and a selection rule; data, results, test isolation, and limits
- [Models](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/models.md): the `Model` API, model support, importing and exporting weights, and tensor roles
- [Managed experiments](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/experiments.md): tasks, policies and budgets, controllers, the code sandbox, serving a run over MCP, records, exports, and replay
- [MCP protocol](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/protocol.md): the tools and their contracts
- [Extensions](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/extensions.md): the registry; add operators, adapters, evaluators, and strategies
- [Architecture](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/architecture.md): how the core, the Engine, the worker, the MCP server, and patch storage fit together
- [Results](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/results.md): every measurement and how to reproduce it
- [Development](https://github.com/xuguang-ai/weight-space-engineering/tree/main/dev): plans, milestone reports, and handovers

## Contributing

Clone the repository and run the offline test suite, which uses tiny models built from configuration classes. With [uv](https://docs.astral.sh/uv/), `uv run --all-extras` installs WSE, every optional extra, and the test tools the first time:

```bash
git clone https://github.com/xuguang-ai/weight-space-engineering.git
cd weight-space-engineering
uv run --all-extras pytest -q
```

Bug reports should include the command, model revision, platform, and relevant error output. [Open an issue](https://github.com/xuguang-ai/weight-space-engineering/issues)

## License

WSE is released under the [Apache License 2.0](https://github.com/xuguang-ai/weight-space-engineering/blob/main/LICENSE).
