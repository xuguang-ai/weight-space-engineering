# Weight-Space Engineering (WSE)

[![PyPI](https://img.shields.io/pypi/v/wse)](https://pypi.org/project/wse/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/wse/)
[![License](https://img.shields.io/pypi/l/wse)](https://github.com/xuguang-ai/weight-space-engineering/blob/main/LICENSE)

**Give AI agents tools to inspect, edit, and test model weights.**

WSE is a Python framework and MCP harness for experiments that edit a model's weights directly. You state a goal and how it is measured. An agent investigates the model and proposes weight edits; WSE evaluates candidates on the same problems, selects one by your rule, freezes it, and only then tests it on problems the agent never saw. Every edit is an exact patch that replays to identical weights, so each result can be checked and reproduced.

[Experiment](https://github.com/xuguang-ai/weight-space-engineering#an-experiment-in-a-few-lines) · [Install](https://github.com/xuguang-ai/weight-space-engineering#install) · [Python API](https://github.com/xuguang-ai/weight-space-engineering#use-the-python-api) · [Agents](https://github.com/xuguang-ai/weight-space-engineering#use-an-agent) · [Results](https://github.com/xuguang-ai/weight-space-engineering#results) · [Documentation](https://github.com/xuguang-ai/weight-space-engineering#documentation)

> **Current release: alpha.** A live LLM agent has repaired a small neural network through WSE's MCP tools. Language-model edits have run from Python and, with scripted controllers, through MCP. Through the `Experiment` API below, a live LLM agent has edited a language model once, Qwen3-0.6B: its choice improved the development problems and did not improve the test problems. Development and testing have used macOS on Apple silicon.

## An experiment in a few lines

An agent edits Qwen3.5-0.8B's weights so that it finishes more of its math answers within 8,192 tokens; WSE evaluates candidates on fixed development problems, keeps the one that cuts off the fewest answers without losing accuracy, and only then tests it against the original on problems registered after the choice is frozen.

```python
from wse import Experiment

experiment = Experiment(
    model="Qwen/Qwen3.5-0.8B",
    goal="Reduce repetitive reasoning so the model finishes more math answers.",
    dev="examples/data/math_dev.jsonl",
    evaluator="math",
    minimize="limit_hit_rate",
    preserve=["accuracy"],
    max_new_tokens=8192,
    workspace="runs/fix-loops",
)
result = experiment.run(
    agent="codex:YOUR_AGENT_MODEL",
    sandbox="python",
    max_evaluations=10,
)
print(result.summary())  # behavior change, candidates, where they changed, why the choice; audit details after

selected = result.model  # the frozen choice, a Model; the original when no candidate qualifies
report = result.test("examples/data/math_test.jsonl")  # only the frozen choice and the original, on new problems
print(report.compare())
result.export("artifacts/fix-loops")
```

> **This Qwen3.5-0.8B example has not been run.** The same API has run once with a live agent, on Qwen3-0.6B ([Results](https://github.com/xuguang-ai/weight-space-engineering#an-agent-edits-a-small-language-model)).

<details>
<summary>What the same calls printed on Qwen3-0.6B (examples/experiment_quickstart.py, 2026-11-12)</summary>

`result.summary()`, as this version prints it for that run (two of the seven candidates shown):

```text
Experiment exp_0a0f8280e629ba70fdab1083806fcdee (CLOSED): Reduce repetitive reasoning so the model finishes more math answers.
Rule: WSE keeps the candidate with the lowest limit_hit_rate on the full development set among those that improve on the baseline's limit_hit_rate, with accuracy no worse than the baseline's; when none does, it keeps the original model.

Behavior on the development set (point estimates):
  metric                    original  selected
  accuracy                     0.312     0.438
  limit_hit_rate               0.625     0.438
  mean_generated_tokens        893.9     844.2
Choice: ckpt_9a07584b2251d762ac22fbce5c5c1918 was kept: limit_hit_rate 0.625 -> 0.4375, accuracy 0.3125 -> 0.4375; 5 of 7 evaluated candidates qualified

Candidates (7 evaluated):
  - ckpt_9a07584b2251d762ac22fbce5c5c1918: accuracy 0.4375, limit_hit_rate 0.4375, mean_generated_tokens 844.2; 64 elements in model.norm.weight; selected
    hypothesis: The final residual probe shows the largest finished-vs-limit contrast at residual:28. Halving only the 64 strongest corresponding model.norm dimensions should reduce late repetitive continuation while preserving most output features.
  - ckpt_c12a1e8686479788a317b0f08699539d: accuracy 0.3125, limit_hit_rate 0.5625, mean_generated_tokens 904.1; 64 elements in model.norm.weight; qualified; another candidate scored better
    hypothesis: A stronger version of the successful late-norm intervention tests whether more suppression of the same 64 high-contrast dimensions further lowers looping.
  ... (5 more candidates)
Candidates without a full development evaluation (1): ckpt_68d97766330258baf71c35413fa6ec55

The agent's final message: WSE-STOP: Best evaluated candidate is `ckpt_9a07584b2251d762ac22fbce5c5c1918`, improving accuracy to 0.4375 and reducing limit-hit rate to 0.4375 from 0.625.

Audit:
  controller codex_cli@1 model gpt-5.6-luna, ended completed (the agent stopped: WSE-STOP: Best evaluated candidate is `ckpt_9a07584b2251d762ac22fbce5c5c1918`, improving accuracy to 0.4375 and reducing limit-hit rate to 0.4375 from 0.625.)
  evaluations 8 of 10 (the baseline included)
  constraint status unverified (the agent runs its own commands (Codex's shell), whose computation WSE does not meter: WSE meters only its own calls; sandbox code execution: WSE meters only its own calls)
  WSE-metered computation 699.1 s; agent tokens 1,583,478 (input 1,577,323, output 6,155)
  sandbox calls 4, 1.0 s of sandbox wall-clock time (not metered computation)
```

`report.compare()` on the 16 test problems:

```text
test data examples/data/math_test.jsonl: 16 problems; selected ckpt_9a07584b2251d762ac22fbce5c5c1918, original ckpt_60a7ac2ff87136505570931729913354
metric                    original  selected   difference [95% interval]
accuracy                     0.250     0.188   -0.062 [-0.188, +0.000] paired
limit_hit_rate               0.688     0.750   +0.062 original [0.438, 0.875]; selected [0.562, 0.938]
mean_generated_tokens        922.3     945.9   +23.6 original [823.2, 1001.6]; selected [863.6, 1009.9]
```

</details>

- **`goal` guides; `minimize` and `preserve` decide.** A candidate qualifies when its development limit-hit rate is below the original's and its development accuracy is at least the original's; WSE keeps the qualifying candidate with the lowest limit-hit rate, and when none qualifies, `result.model` is the original. These are point estimates, not significance tests.
- **The evaluation is fixed.** Every model is evaluated on the same problems, prompts, sampling settings, seeds, and token limit, and the agent cannot change the grading. `max_evaluations` counts the evaluations of whole models on the development problems, the original's included.
- **The limit-hit rate** is the share of answers cut off at the token limit. It is not a repetition rate.

[Use an agent](https://github.com/xuguang-ai/weight-space-engineering#use-an-agent) gives the setup; the [Experiment API](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/experiment-api.md) covers the data format, where results go, test isolation, and what WSE does and does not verify.

## Install

The examples run from a source checkout. You need Python 3.11 or later and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/xuguang-ai/weight-space-engineering.git
cd weight-space-engineering
uv sync
```

Each command names the optional extra it needs, and `uv run --extra models-hf` installs the Hugging Face model support the first time a command uses it. An agent also needs the Codex CLI ([below](https://github.com/xuguang-ai/weight-space-engineering#use-an-agent)). The release on PyPI, 0.1.2, predates the `Model` and `Experiment` APIs.

## Use the Python API

A `Model` is one version of a model in a workspace: load it, inspect it, generate, edit its weights, and compare the edit with the original. [examples/model_quickstart.py](https://github.com/xuguang-ai/weight-space-engineering/blob/main/examples/model_quickstart.py) runs each step on Qwen3-0.6B and takes about two and a half minutes on an M4 Pro, after a download of 1.5 GB; its output directory takes another 2.7 GB, 1.5 GB of it the exported model:

```bash
uv run --extra models-hf python examples/model_quickstart.py \
  runs/model-quickstart \
  --revision c1899de289a04d12100db370d81485cdf75e47ca
```

`--revision` fixes the model's commit. The output below came from a run of this script on a local copy of that commit, on an M4 Pro with MPS; checkpoint and patch IDs differ per run. **Run the complete script; the blocks below are a walkthrough of it.** Its start, the imports and arguments, is folded, so the visible code alone does not run.

<details>
<summary>The start of the script: imports and arguments</summary>

```python
import argparse
from pathlib import Path

from wse import Model, compare_models

parser = argparse.ArgumentParser(description="WSE's Model API: load, inspect, generate, edit, compare, and export.")
parser.add_argument("out", type=Path, help="a new directory for the workspace, the comparison, and the exports")
parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="a Hugging Face repository ID or a local snapshot")
parser.add_argument("--repo", help="a local snapshot's source repository")
parser.add_argument("--revision", help="the revision to download (default main), or a local snapshot's commit")
parser.add_argument("--data", type=Path, default=Path("examples/data/math_dev.jsonl"), help="math problems")
parser.add_argument("--problems", type=int, default=8, help="how many of the problems the comparison uses")
parser.add_argument("--max-new-tokens", type=int, default=1024, help="the token limit of each answer")
args = parser.parse_args()
if args.out.exists():
    parser.error(f"{args.out} exists; each run needs a new directory")
```

</details>

### Load and inspect

`from_pretrained` resolves the revision to a commit, records it, and imports the weights into a workspace. `tensors` lists names, shapes, and dtypes; `tensor` returns a copy.

```python
# 1. Load: the snapshot at a resolved commit, imported into a workspace.
original = Model.from_pretrained(
    args.model,
    repo=args.repo,
    revision=args.revision,
    workspace=args.out / "workspace",
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
model Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca, checkpoint ckpt_79ed598420508d401177a910eb8fcabf, weights 92865a3c8750
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
1 tensors changed, 131,072 entries; weights 92865a3c8750 -> 092773d60909
  model.layers.14.self_attn.o_proj.weight: 131,072 of 2,097,152 entries, relative L2 change 0.2756, 70e83e3c4328 -> ad2d2c42779e
the patch replays to the edited weights: True
saved: patch patch_6dd10a15732045e23deda035973d0a14, checkpoint ckpt_62960adbd9f8f4b026eb47be8fb6edd8
```

The registered operators scale, set, or fill a slice, remove a direction's projection, or add a low-rank term; `replace_weights` submits whole tensors computed any other way, after checking their names, shapes, dtypes, and values ([the Model API](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/models.md#the-model-api)).

### Compare

`compare_models` runs both models on the same problems, prompts, sampling settings, and seeds, and reports paired differences with 95% intervals. `out_dir` records every problem, so an interrupted comparison resumes where it stopped.

```python
# 5. Compare: the same problems, prompts, sampling settings, and seeds for both models.
comparison = compare_models(
    base=original,
    candidate=edited,
    data=args.data,
    evaluator="math",
    metrics=["accuracy", "limit_hit_rate", "mean_generated_tokens"],
    max_new_tokens=args.max_new_tokens,
    problems=args.problems,
    out_dir=args.out / "comparison",
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
print(f"patch: {edit.export(args.out / 'patch')}")
print(f"model: {edited.export(args.out / 'edited-model')}")
```

```text
patch: runs/model-quickstart/patch
model: runs/model-quickstart/edited-model
```

The model directory holds `config.json`, the tokenizer, `model.safetensors`, and `wse_provenance.json`. Imported again with `from_pretrained`, it has the edited model's weights hash, and `transformers` loads it as `Qwen3ForCausalLM` with all 311 tensors equal; the patch replays onto the original in another workspace ([exporting](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/models.md#exporting-a-model)).

The same calls run the full-size case on Qwen3.5-0.8B, which reconstructs a published edit and measures it on 32 problems ([Results](https://github.com/xuguang-ai/weight-space-engineering#results)).

## Use an agent

`Experiment` hands one model to an agent under a fixed evaluation. The agent is the [Codex CLI](https://github.com/openai/codex). WSE starts it in the experiment's working directory and connects it to the run's WSE tools over MCP; the agent can also write and run Python there, under Codex's sandbox, which WSE configures so that the agent can read the model's weights but not the test data, WSE's database, or your keys. WSE records each candidate the agent submits as an exact patch and evaluates the ones the agent asks it to; when the agent stops, WSE selects among the evaluated candidates by the rule and freezes the choice before any test. Because the agent's own code can compute anything from the weights it reads, WSE meters only its own calls and labels these runs `unverified`.

**1. Install the Codex CLI.** WSE runs the binary that `WSE_CODEX_PATH` names, else the `codex` on your `PATH`, else the newest one bundled with OpenAI's VS Code extension; on macOS it first checks that the binary's code signature names OpenAI. Check that `codex --version` runs; WSE has been tested with codex-cli 0.167.0-alpha.4.1.

**2. Add your key.** Copy the template, [.env.example](https://github.com/xuguang-ai/weight-space-engineering/blob/main/.env.example), and fill in your key and endpoint; git ignores `.env`:

```bash
cp .env.example .env
```

```text
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
```

For Azure OpenAI, set `OPENAI_BASE_URL` to your resource's endpoint, `https://YOUR-RESOURCE.services.ai.azure.com/openai/v1`, and put your deployment's name after `codex:`. Codex reaches the provider through a local proxy that WSE runs and that adds the key, so neither Codex nor the agent's code holds it ([credentials](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/experiment-api.md#agent-credentials)).

**3. Run an experiment.** The small script below has run once with a live agent ([Results](https://github.com/xuguang-ai/weight-space-engineering#an-agent-edits-a-small-language-model)); the Qwen3.5-0.8B one has not been run. With your model's name, or your deployment's name, in place of `YOUR_AGENT_MODEL`, the small one runs Qwen3-0.6B on the 16 development problems at 1,024 tokens, with up to 10 evaluations (about a minute and a half each, by the model quickstart's timing), and then tests on 16 GSM8K test problems:

```bash
uv run --extra models-hf --env-file .env \
  python examples/experiment_quickstart.py \
  --agent codex:YOUR_AGENT_MODEL
```

The full-size one gives the agent Qwen3.5-0.8B's reasoning loops: 32 GSM8K training problems on which the original answered 2 correctly and ran 30 to the limit, at 8,192 tokens with the article's sampling profile, `qwen35think@1`. One evaluation took 37 to 40 minutes in the recorded runs, so 10 take 6.2 to 6.6 hours; `caffeinate -is` keeps the Mac awake:

```bash
caffeinate -is uv run --extra models-hf --env-file .env \
  python examples/experiment_qwen_loops.py \
  --agent codex:YOUR_AGENT_MODEL
```

`--extra models-hf` installs the model support on first use, and `--env-file .env` loads the key for this command only. Both scripts take `--model`, `--dev`, `--test`, `--workspace`, and `--export` ([examples](https://github.com/xuguang-ai/weight-space-engineering/blob/main/examples/experiment_quickstart.py)).

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

Through `Experiment`, Codex with `gpt-5.6-luna` worked on Qwen3-0.6B's answers to 16 GSM8K training problems at 1,024 new tokens, to cut fewer answers off at the limit without losing accuracy ([examples/experiment_quickstart.py](https://github.com/xuguang-ai/weight-space-engineering/blob/main/examples/experiment_quickstart.py)). In 13 minutes it probed the residual stream, computed 8 candidates in Python, each scaling down 32 to 128 entries of the final RMSNorm weight, evaluated 7, and stopped itself. WSE's rule chose the one that halves 64 entries: on the development problems 5 → 7 correct and 10 → 7 cut off. On 16 GSM8K test problems registered after the freeze it did not help: 4 → 3 correct and 11 → 12 cut off (accuracy difference −0.062 [−0.188, +0.000]). The run shows the whole path working with a live agent, not an edit that generalizes ([the run](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/results.md#6-a-live-llm-agent-edits-a-language-model-through-experiment)).

### What has been demonstrated

| Experiment | Controller | Recorded result |
| --- | --- | --- |
| Repair a planted fault in a 22-parameter network | Live LLM agent over MCP | 5/5 acceptance trials repaired it, from 50% to 100% final accuracy |
| Apply published p50c to Qwen3.5-0.8B | Command line, same runner as Python | 3/25 → 13/25 correct on MATH-500, with a 32,768-token limit |
| Reconstruct p50c's projection edit with WSE operators | Direct Python | 2/32 → 10/32 correct on GSM8K practice problems; answers cut off at the limit fell from 30/32 to 21/32 |
| Apply that reconstruction through MCP | Scripted controller | Answers cut off at the limit fell from 30/32 to 19/32 on another practice set |
| An agent edits a language model (`Experiment` with Codex) | Live LLM agent | One run on Qwen3-0.6B: 5/16 → 7/16 correct on the development problems it was chosen on, 4/16 → 3/16 on test problems |

The article's three Qwen3.5-0.8B edits import to their published hashes, and WSE's own `fill_slice@1` operator reproduces one of them, p25c, hash-exactly. The reconstruction takes its direction and columns from p50c's published change; a scripted search that used its form without its values found no repair. One live agent run has edited a language model, and its gain on the development problems did not carry over to the test problems.

## Current support

| Area | Status |
| --- | --- |
| Agent interface | `Experiment` with the Codex CLI (one live run, on Qwen3-0.6B); the MCP server for other MCP clients |
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

From the repository checkout, install every extra and run the offline test suite. It uses tiny models built from configuration classes:

```bash
uv sync --all-extras
uv run --no-sync pytest -q
```

Keep `--all-extras` when syncing: a plain `uv sync` removes the optional extras. Bug reports should include the command, model revision, platform, and relevant error output. [Open an issue](https://github.com/xuguang-ai/weight-space-engineering/issues)

## License

WSE is released under the [Apache License 2.0](https://github.com/xuguang-ai/weight-space-engineering/blob/main/LICENSE).
