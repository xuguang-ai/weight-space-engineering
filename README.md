# Weight-Space Engineering (WSE)

[![PyPI](https://img.shields.io/pypi/v/wse)](https://pypi.org/project/wse/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/wse/)
[![License](https://img.shields.io/pypi/l/wse)](https://github.com/xuguang-ai/weight-space-engineering/blob/main/LICENSE)

**Give AI agents tools to inspect, edit, and test model weights.**

WSE is an MCP harness and Python toolkit for direct weight-editing experiments. Give an agent a task, a target model, and an experiment budget. It can inspect tensors, investigate failures, try weight edits, and compare candidates through WSE's tools. WSE records the changes and checks that saved patches replay to identical weights.

Each experiment gives you a selected model checkpoint, evaluation reports, exact weight patches, and a record of the controller's actions.

The Python API and the MCP tools share the same registered edit operators, evaluators, and patch format. A new edit method, target model, task, or agent therefore reuses the rest of the experiment infrastructure: budgets, policy checks, paired evaluation, audit records, and exact replay ([what carries over](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/architecture.md#what-carries-over)). This README first edits a language model from Python, then hands the same operations to an agent through MCP.

[Install](https://github.com/xuguang-ai/weight-space-engineering#install) · [Edit with Python](https://github.com/xuguang-ai/weight-space-engineering#edit-a-language-model-with-python) · [Run an agent](https://github.com/xuguang-ai/weight-space-engineering#run-an-agent-through-mcp) · [Results](https://github.com/xuguang-ai/weight-space-engineering#results) · [Documentation](https://github.com/xuguang-ai/weight-space-engineering#documentation)

> **Current release: alpha.** A live LLM agent has repaired a small neural network through MCP. Language-model editing and evaluation have run from Python and with scripted controllers; no LLM agent has edited a language model yet. Development and testing have used macOS with Apple silicon.

## Install

The examples run from a source checkout. You need Python 3.11 or later and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/xuguang-ai/weight-space-engineering.git
cd weight-space-engineering
uv sync
```

Each example names the optional extra it needs, and `uv run --extra NAME` installs that extra the first time a command uses it: `models-hf` for Hugging Face language models, `agents-openai` for the reference agent. Outside a checkout, `pip install "wse[models-hf]"` installs the package from PyPI.

## Edit a language model with Python

**Qwen keeps looping in its reasoning. Inspect the model, edit its weights, and measure whether it finishes more answers.**

Asked 32 GSM8K practice problems with room for 8,192 new tokens, Qwen3.5-0.8B finishes 2 answers. The other 30 run into the limit, and 24 of those end by repeating the same lines. The four steps below measure this, look at the weights and activations of two MLP layers, edit those weights, and compare the edited model with the original on the same problems: 10 answers correct instead of 2, and 21 cut off instead of 30.

The edit is recovered from p50c, the edit that GPT-6 Astra made in [Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang, and Khattab, 2026): its layers, columns, and direction come from p50c's published change, so these steps reconstruct that repair rather than discover it.

The script needs about 3 GB of disk and a Mac with Apple silicon: WSE generates text on Apple's MPS, or on the CPU, which is about 170 times slower. On an M4 Pro it took 1 hour 54 minutes, almost all of it in three generation passes of about 37 minutes each.

<details>
<summary>Download the published edit and the practice problems</summary>

The model downloads when `from_pretrained` runs. These commands fetch p50c's directory from the article's logs and GSM8K's training set, both at fixed revisions. Run the clone step once for this directory.

```bash
M=~/wse-data
mkdir -p "$M"

git clone --filter=blob:none --no-checkout \
  https://github.com/RikiyaT/lm_understand-logs \
  "$M/lm_understand-logs"
git -C "$M/lm_understand-logs" sparse-checkout set exp8_math/runs/p50c/edit
git -C "$M/lm_understand-logs" checkout 4d09ba6d6a7e891dd8063974dfe81b3d78e3494f

curl -L --create-dirs -o "$M/gsm8k/train.jsonl" \
  https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl
```

</details>

The four steps are one script, [examples/p50c_four_steps.py](https://github.com/xuguang-ai/weight-space-engineering/blob/main/examples/p50c_four_steps.py):

```bash
uv run --extra models-hf python examples/p50c_four_steps.py ~/wse-data
```

It loads the model at a fixed revision into a local workspace:

```python
import sys
from pathlib import Path

import numpy as np

from wse import WSE

data = Path(sys.argv[1] if len(sys.argv) > 1 else "~/wse-data").expanduser()
gsm8k = data / "gsm8k/train.jsonl"
base = WSE.from_pretrained(
    "Qwen/Qwen3.5-0.8B",
    revision="2fc06364715b967f1860aea9cf38778875588b17",
    workspace=data / "workspace",
)
```

### 1. Diagnose

Generate one answer to each problem with the article's sampling (`qwen35think@1`), then count the answers cut off at the token limit and, among them, those that end in a loop.

```python
# 1. Diagnose: how many answers are cut off at the token limit, and how many of those end in a loop?
run = base.evaluate(
    gsm8k,
    format="gsm8k",
    problems=32,
    profile="qwen35think@1",
    max_new_tokens=8192,
)
print(run.table())
```

<details>
<summary>How ends_in_loop detects a loop: at least half of the lines in an answer's last 2,000 characters repeat an earlier line there</summary>

```python
def ends_in_loop(text: str, window: int = 2000) -> bool:
    """True when at least half of the lines in the last ``window`` characters repeat an earlier line there."""
    lines = [line.strip() for line in text[-window:].split("\n")[1:-1]]
    lines = [line for line in lines if line]
    return bool(lines) and 2 * (len(lines) - len(set(lines))) >= len(lines)
```

</details>

```python
cut_off, finished = run.limit_hit(), run.finished()
loops = [k for k in cut_off if ends_in_loop(run.outputs[k].text)]
print(f"{len(cut_off)} answers cut off at the limit, {len(finished)} finished")
print(f"{len(loops)} of the {len(cut_off)} cut-off answers end in a loop; the end of the first:")
if loops:
    print("\n".join(run.outputs[loops[0]].text.splitlines()[-9:-1]))
```

```text
32 problems, 32 samples
accuracy                     0.062 [0.000, 0.156]
limit_hit_rate               0.938 [0.844, 1.000]
mean_generated_tokens       8063.5 [7843.9, 8192.0]
30 answers cut off at the limit, 2 finished
24 of the 30 cut-off answers end in a loop; the end of the first:
```

<details>
<summary>The end of the first cut-off answer</summary>

```text
        *   Wait, is it possible the 400 files were the *number of files* he had *downloaded* in the *second* round?
        *   Yes.
        *   Wait, is it possible the 400 files were the *number of files* he had *downloaded* in the *first* round?
        *   No, he downloaded 800 files.
        *   Wait, is it possible the 400 files were the *number of files* he had *downloaded* in the *second* round?
        *   Yes.
        *   Wait, is it possible the 400 files were the *number of files* he had *downloaded* in the *first* round?
        *   No, he downloaded 800 files.
```

</details>

The model answers 2 of 32 problems correctly, and 30 answers reach the limit. The first of them, to problem `gsm8k_27`, asks itself the same two questions, about which download round 400 files belong to, from line 250 of its 369 lines until the limit.

### 2. Probe

Import p50c with `apply_patch`, which checks every tensor against its published SHA-256, and read where it changes the weights. `recover` takes the columns that p50c changes in one layer and the direction it acts on there, the top singular vector of its change. The probe then reads the activations at the output of those layers over the last 1,024 tokens of each answer, and contrasts the cut-off answers with the finished ones.

```python
# 2. Probe: where does p50c change the weights, and do cut-off and finished answers differ along its direction?
p50c = base.apply_patch(
    data / "lm_understand-logs/exp8_math/runs/p50c/edit",
    repo="RikiyaT/lm_understand-logs",
    revision="4d09ba6d6a7e891dd8063974dfe81b3d78e3494f",
)
print(p50c.table())
```

<details>
<summary>How recover reads p50c's columns and direction in one layer</summary>

```python
def recover(layer: int) -> tuple[str, np.ndarray, np.ndarray]:
    """p50c in one layer: the tensor, the direction it acts on (its change's top singular vector), its columns."""
    name = base.name(f"layers.{layer}.mlp.down_proj")
    delta = p50c.model.tensor(name).double().numpy() - base.tensor(name).double().numpy()
    columns = np.flatnonzero(delta.any(axis=0))
    d = np.linalg.svd(delta[:, columns], full_matrices=False)[0][:, 0]
    d *= np.sign(d[np.argmax(np.abs(d))])  # svd_top_k's sign convention: the largest-magnitude entry is positive
    return name, d, columns
```

</details>

```python
sites = {layer: f"layers.{layer}.mlp.down_proj:output" for layer in (15, 19)}
if cut_off and finished:  # a contrast needs both groups
    probe = base.probe(run, sites=list(sites.values()), last=1024)
    for layer, site in sites.items():
        _, d, columns = recover(layer)
        contrast = probe.contrast(site, cut_off, finished)
        cosine = contrast.direction @ d / contrast.norm
        print(f"layer {layer}: p50c changes {len(columns)} columns; "
              f"cosine of the cut-off minus finished contrast with its direction {cosine:+.2f}")
else:
    print("the probe needs at least one cut-off and one finished answer")
```

```text
layer 15: p50c changes 64 columns; cosine of the cut-off minus finished contrast with its direction +0.37
layer 19: p50c changes 64 columns; cosine of the cut-off minus finished contrast with its direction -0.45
```

<details>
<summary>p50c's change per tensor, printed before these lines</summary>

```text
2 tensors changed, 130,188 entries; weights cb5a704b1fd2 -> b8dccf8ca463
  model.language_model.layers.15.mlp.down_proj.weight: 65,260 of 3,670,016 entries, relative L2 change 0.1064, db89c2e60dc5 -> c8df99feb499
  model.language_model.layers.19.mlp.down_proj.weight: 64,928 of 3,670,016 entries, relative L2 change 0.08106, 2c8c10a14a67 -> d0ae5310479d
```

</details>

p50c changes 130,188 weights of `mlp.down_proj` in layers 15 and 19, within 64 columns of each. In both layers the contrast leans toward p50c's direction (a singular vector's sign is arbitrary, so the size is what counts). With only 2 finished answers, this points to where to look; it does not establish a mechanism. The [results](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/results.md#step-2-probe) compare it with arbitrary splits of the answers.

### 3. Edit

`project_out` with alpha 6 subtracts six times each column's projection onto the direction, in p50c's 64 columns of each layer, as p50c does; alpha 1 would remove the component. `save()` stores each edit in the workspace with its exact patch.

```python
# 3. Edit: in both layers, subtract six times each of p50c's columns' projection onto its direction.
model = base
for layer in sites:
    name, d, columns = recover(layer)
    edit = model.edit(
        name,
        "project_out",
        direction=d,
        columns=columns,
        alpha=6.0,
    )
    model = edit.save().model
print(f"{sum(diff.changed_elements for diff in model.diff_from(base)):,} weights changed")
```

```text
130,134 weights changed
```

p50c itself changes 130,188. WSE's operator rounds differently from the article's code, so 91% of the edited entries are byte-identical to p50c's and the rest differ by one or a few BF16 steps ([comparison](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/results.md#step-3-edit)).

### 4. Evaluate

`compare` runs the base and the edited model on the same 32 problems, prompts, and seeds, and reports paired differences with 95% intervals. `out_dir` records every finished problem, so an interrupted comparison resumes where it stopped.

```python
# 4. Evaluate: the base and the edited model on the same problems, prompts, and seeds.
comparison = base.compare(
    model,
    gsm8k,
    format="gsm8k",
    problems=32,
    profile="qwen35think@1",
    max_new_tokens=8192,
    out_dir=data / "runs/four-steps",
)
print(comparison.table())
```

```text
32 problems, 32 samples per model
metric                        base   candidate   difference [95% interval]
accuracy                     0.062       0.312   +0.250 [+0.125, +0.406]
limit_hit_rate               0.938       0.656   -0.281 [-0.438, -0.125]
mean_generated_tokens       8063.5      7082.0   -981.5 [-1597.1, -434.7]
```

Accuracy rose from 2 to 10 of 32 and answers cut off fell from 30 to 21; all three intervals exclude zero. Eight problems were right only with the edit, none only with the base. None of the 21 answers still cut off ends in a loop by step 1's measure: they keep re-checking the problem in new words until the limit ([step-by-step results](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/results.md#3-reconstructing-p50c-in-four-steps)).

To build other edits, WSE's registered operators scale, set, or fill slices, remove a direction's projection, or add a low-rank term ([operators](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/extensions.md)).

## Run an agent through MCP

An agent edits weights through the same operations. The runner starts a `wse serve` process bound to one run and connects the agent to it as an MCP server. The agent's only tools are WSE's, so it cannot run code, read files, or use the network. A policy decides which tools and edits it may use and sets one budget for the whole experiment; the server refuses everything else and records each refusal. When the agent stops, the runner freezes the checkpoint it selected, and only then does the final test run, on examples the agent never saw.

The example below repairs a classifier with 22 parameters, small enough for an agent to finish in under a minute. No LLM agent has run the Qwen experiment yet: a generation pass there takes about 37 minutes, and an agent waits for it by calling `wse_get_job`, which returns within 10 seconds. One pass therefore costs about 220 tool calls, which by our estimate fill a 128,000-token context before the pass ends. A scripted controller has run the Qwen reconstruction through the same MCP server ([results](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/results.md#4-through-the-mcp-tools-with-scripted-controllers)).

The six steps are one script, [examples/signed_sum/agent_api.py](https://github.com/xuguang-ai/weight-space-engineering/blob/main/examples/signed_sum/agent_api.py), and each is a piece you replace for your own experiment. The script takes a new directory and, for a live agent, `--model`, and opens a workspace in that directory with `Engine`.

<details>
<summary>The start of the script: imports, arguments, and the workspace</summary>

```python
import argparse
import os
import secrets
from pathlib import Path

import numpy as np
import torch

from wse.agent_runs import run_agent_experiment
from wse.api import Engine

parser = argparse.ArgumentParser(description="One agent experiment on a faulty 22-parameter classifier.")
parser.add_argument("out", type=Path, help="a new directory for the workspace and the run's records")
parser.add_argument("--model", help="the agent's model; without it, the scripted controller scripted_search@1")
args = parser.parse_args()
if args.out.exists():
    parser.error(f"{args.out} exists; each run needs a new directory")
workspace = args.out / "workspace"
engine = Engine(workspace)
```

</details>

### 1. Load the target model

The target is a 2 → 4 → 2 classifier with ReLU and 22 parameters, given as tensors with `mlp@1`, WSE's adapter for small multilayer perceptrons. Its two output rows have identical weights, so it always predicts class 0 and is right on half of the examples. The agent is not told this. `import_model` stores the weights and returns the base checkpoint, whose ID the run names in step 5.

```python
# 1. Load the target model: a 2 -> 4 (ReLU) -> 2 classifier with 22 parameters.
#    Its two output rows have identical weights, so it always predicts class 0.
base = engine.import_model(
    {
        "adapter": "mlp@1",
        "config": {"layer_sizes": [2, 4, 2], "activation": "relu", "bias": True},
        "provenance": {"source": "signed_sum model v1"},
        "license": "generated fixture; no third-party weights",
        "input_description": "x = (x0, x1)",
        "output_description": "logits for class 0 and class 1; the predicted class is the argmax",
    },
    {
        "layers.0.weight": torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
        "layers.0.bias": torch.zeros(4),
        "layers.1.weight": torch.tensor([[-1.0, 1.0, -1.0, 1.0], [-1.0, 1.0, -1.0, 1.0]]),
        "layers.1.bias": torch.zeros(2),
    },
)
```

A language model comes from a Hugging Face snapshot at a fixed revision instead: `WSE.from_pretrained`, as in the Python example, imports it, and its `checkpoint_id` is the base checkpoint ([models](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/models.md)).

### 2. Define the task

The task is the problem statement that the agent reads, and the objective that decides success. Here the agent reads:

> The registered model reads two real inputs x = (x0, x1) and returns logits for two classes; its prediction is the class with the larger logit. It should predict class 1 when x0 + x1 > 0 and class 0 otherwise. Using only the permitted inspection and editing operations, produce a candidate checkpoint that follows this rule, and select it.

Success is accuracy of 1.0 on all 64 development examples. The task states the behavior to reach, never how the model was built; WSE writes the agent's [instructions](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/experiments.md#what-the-agent-is-told) from it and the policy.

<details>
<summary>The task definition</summary>

```python
# 2. Define the task: the statement the agent reads, and the objective that counts as success.
task_definition = {
    "task_id": "signed_sum_v1",
    "title": "Signed-sum classification",
    "description": (
        "The registered model reads two real inputs x = (x0, x1) and returns logits for two classes; "
        "its prediction is the class with the larger logit. It should predict class 1 when x0 + x1 > 0 "
        "and class 0 otherwise. Using only the permitted inspection and editing operations, produce a "
        "candidate checkpoint that follows this rule, and select it."
    ),
    "input_description": "x = (x0, x1), two float32 values",
    "output_description": "logits for class 0 and class 1; the predicted class is the argmax",
    "objective": {
        "task_id": "signed_sum_v1",
        "description": "Predict class 1 exactly when x0 + x1 > 0, measured as accuracy on the development suite.",
        "gates": [{"metric": "accuracy", "min_value": 1.0}],
    },
    "target_accuracy": 1.0,
    "development_suite_id": "signed_sum_dev_v1",
    "development_examples": 64,
}
```

</details>

```python
task = engine.register_task(task_definition)
```

For the Qwen loops, the objective is to lower the rate of answers cut off at the limit without lowering accuracy: a gate `{"metric": "accuracy", "min_delta": 0.0}` against the base ([the Qwen task](https://github.com/xuguang-ai/weight-space-engineering/blob/main/examples/qwen_loops/loop_setup.py)).

### 3. Connect the evaluation

Two suites of labeled examples: 64 development examples on which the agent may evaluate candidates, and 1,024 final examples that only the final evaluator reads, after the agent's choice is frozen. Each example is a point x with its label, followed by −x; the final examples are drawn with a random seed when the workspace is set up, which is why each run needs a new directory. `classification@1` runs the model on every input, takes the class with the larger logit, and reports accuracy, with intervals over the x, −x pairs.

<details>
<summary>The function that generates the labeled examples</summary>

```python
# 3. Connect the evaluation: labeled examples, scored by accuracy.
def examples(pairs: int, seed: int) -> dict:
    """``pairs`` points x with |x0 + x1| >= 0.1, each followed by -x; the label is 1 when x0 + x1 > 0."""
    rng = np.random.default_rng(seed)
    points = []
    while len(points) < pairs:
        x = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        if abs(float(x[0]) + float(x[1])) >= 0.1:
            points.append(x)
    inputs = np.empty((2 * pairs, 2), dtype=np.float32)
    inputs[0::2] = np.stack(points)
    inputs[1::2] = -inputs[0::2]
    return {
        "inputs": inputs.tolist(),
        "labels": (inputs.sum(axis=1) > 0).astype(np.int64).tolist(),
        "clusters": np.repeat(np.arange(pairs), 2).tolist(),
        "cluster_unit": "pair",
    }
```

</details>

```python
development = engine.register_dataset(examples(32, seed=20261104))
final = engine.register_dataset(examples(512, seed=secrets.randbits(63)))
for suite_id, split, dataset in [
    ("signed_sum_dev_v1", "development", development),
    ("signed_sum_final_v1", "final", final),
]:
    engine.register_suite({
        "suite_id": suite_id,
        "task_id": task.task_id,
        "split": split,
        "evaluator": "classification@1",
        "sample_counts": [dataset.example_count],
        "seeds": [0],
        "dataset": dataset.model_dump(mode="json"),
    })
```

A language model is evaluated by generation instead: a suite with the evaluator `generation@1`, a sampling profile, a grader, and a token limit reports accuracy and the rate of answers cut off at the limit ([the Qwen practice suite](https://github.com/xuguang-ai/weight-space-engineering/blob/main/examples/qwen_loops/loop_setup.py)).

### 4. Configure the policy

The policy is everything the agent may do: the edits `scale_slice` and `set_slice`, three analyses, and one budget for the whole experiment, with 10 development evaluations of up to 64 examples, 20 candidate commits, 50 patch previews, 50 analyses, 100 tensor reads, and 300 seconds of computation. Spent allowance is never refunded, and every tool result reports what remains. These are the values of the built-in `small_model_v1`, which the recorded trials used; a registered policy cannot change, so a policy with other values needs a new ID.

<details>
<summary>The policy definition</summary>

```python
# 4. Configure the policy: the operations the agent may use, and the budget for the whole experiment.
policy_definition = {
    "policy_id": "signed_sum_policy_v1",
    "mode": "bounded_eval",
    "operations": {
        "edits": {"scale_slice": "scale_slice@1", "set_slice": "set_slice@1"},
        "analyses": {"stats": "tensor_stats@1", "norm": "tensor_norm@1", "top_k": "tensor_top_k@1"},
    },
    "budget": {
        "evaluation_batches": 10,
        "examples_per_batch": 64,
        "evaluated_examples": 640,
        "model_forward_calls": 10,
        "candidate_commits": 20,
        "patch_previews": 50,
        "analysis_calls": 50,
        "tensor_read_calls": 100,
        "compute_seconds": 300.0,
        "seconds_per_operation": 30.0,
    },
}
```

</details>

```python
policy = engine.register_policy(policy_definition)
engine.close()  # the setup is registered; the runner opens the workspace itself
```

For the Qwen loops, `qwen_loop_v1` grants generation passes instead of evaluation batches: 10 passes of up to 32 problems and 8,192 new tokens per answer, 25 activation probes, and 6 hours of computation ([budgets](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/experiments.md#budgets)). No policy limits the agent's own model tokens; the runner caps its turns (100) and its time (900 seconds). Tensors that a model family protects, such as Qwen3.5's embeddings and output head, cannot be edited under any policy.

### 5. Start the agent

`run_agent_experiment` takes the registered IDs. It grants a run, a permission for one experiment with this task, policy, and base checkpoint; starts `wse serve` for it; and connects the agent as its MCP client. When the run ends, it freezes the selected checkpoint, runs the final test, exports the experiment, and replays the export in a new workspace.

```python
# 5. Start the agent: the runner grants a run on these IDs, connects the agent to WSE through MCP,
#    and freezes the checkpoint it selects; then come the final test, an export, and its replay.
print(f"{args.model or 'scripted_search@1'} on model {base.checkpoint_id}, task {task.task_id}, "
      f"policy {policy.policy_id}")
run = run_agent_experiment(
    workspace,
    task=task.task_id,
    base=base.checkpoint_id,
    policy=policy.policy_id,
    model=args.model,
    strategy=None if args.model else "scripted_search@1",
    final_evaluation=True,
)
```

The agent's model needs your key and endpoint. Copy the template, [.env.example](https://github.com/xuguang-ai/weight-space-engineering/blob/main/.env.example), and fill in your values; git ignores `.env`:

```bash
cp .env.example .env
```

```text
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
```

For Azure OpenAI, set `OPENAI_BASE_URL` to your resource's endpoint, `https://YOUR-RESOURCE.services.ai.azure.com/openai/v1`, and pass your deployment's name as the model. Neither value reaches the WSE server or the records ([configuration](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/experiments.md#agent-model-configuration)).

Then run the script, with your model or deployment name in place of `YOUR_AGENT_MODEL`:

```bash
uv run --extra agents-openai --env-file .env \
  python examples/signed_sum/agent_api.py \
  demo/live \
  --model YOUR_AGENT_MODEL
```

Recorded runs used `gpt-5.6-luna` on Azure OpenAI. `--extra agents-openai` installs the OpenAI Agents SDK on first use, and `--env-file .env` loads the key for this command only.

<details>
<summary>Check the workflow offline, without a key</summary>

Without `--model`, the scripted controller `scripted_search@1` takes the agent's place. It needs no key or network and takes a few seconds:

```bash
uv run python examples/signed_sum/agent_api.py demo/offline
```

With `--extra agents-openai` and `--model wse-test-double:scripted_search@1`, the same scripted calls go through the agent SDK's loop and a real `wse serve`, still offline.

</details>

### 6. Read the result

The script then prints how the run ended and what it cost, the edit, the final test, and where the patch is. A run with `gpt-5.6-luna`:

| Result | This run |
| --- | --- |
| How it ended | Stopped when a full development evaluation of the selected checkpoint met the target |
| Cost | 8 model requests and 11 tool calls in 20 seconds, with 68,866 input and 1,428 output tokens |
| The edit | `set_slice` on `layers.1.weight[1:2, 0:4]`: class 1's row went from [−1, 1, −1, 1] to [1, −1, 1, −1], 4 of the 22 parameters |
| Final test | 50% correct before the edit and 100% after, on 1,024 examples the agent never saw |
| Patch | In the run's export, `demo/live/workspace/agent_runs/<run_id>/export/patches/`, with the exact values and hashes; the export replays to byte-identical weights |

<details>
<summary>The code that reads the result, and the run's full output</summary>

```python
# 6. Read the result: how the run ended, the edit, the final test, and the patch.
summary = run.summary
print(f"{summary['controller']} ended {summary['status']}: {summary['reason']}")
print(f"{summary['tool_calls']} tool calls in {summary['elapsed_seconds']:.0f} s")
if summary["tokens"]["input_tokens"] is not None:
    print(f"{summary['turns']} model requests, {summary['tokens']['input_tokens']:,} input and "
          f"{summary['tokens']['output_tokens']:,} output tokens")
if run.state != "CLOSED":
    raise SystemExit(f"the experiment ended {run.state}: {summary['failure_cause']}")
engine = Engine(workspace)
selected = engine.get_experiment(run.experiment_id).selected_checkpoint_id
patch_id = next(c.patch_id for c in engine.list_experiment_checkpoints(run.experiment_id)
                if c.checkpoint_id == selected)
if patch_id is None:
    print("edit: none; the agent kept the unedited model")
else:
    patch = engine.find_patch(patch_id)
    for operation in patch.spec.operations:
        where = ", ".join(f"{axis.start}:{axis.stop}" for axis in operation.selection)
        values = ", ".join(f"{name} {value}" for name, value in operation.parameters.items())
        print(f"edit: {operation.operator} on {operation.tensor}[{where}], {values}")
    for diff in patch.preview.tensors:
        if diff.changed_elements:
            before = engine.open_checkpoint(base.checkpoint_id).tensor(diff.tensor).tolist()
            after = engine.open_checkpoint(selected).tensor(diff.tensor).tolist()
            print(f"{diff.tensor}: {before} -> {after}")
    print(f"patch: {os.path.relpath(run.export_dir / 'patches' / f'{patch_id}.json')}")
comparison = engine.get_final_report(run.experiment_id).comparison
print(f"final test, {comparison.sample_count:,} examples the agent never saw: "
      f"{comparison.base_value:.0%} correct before the edit, {comparison.candidate_value:.0%} after")
print(f"the export replays to byte-identical weights: {run.replay_identical}")
engine.close()
```

```text
gpt-5.6-luna on model ckpt_23cd0df78da1fa59a5dab0aed2a054f8, task signed_sum_v1, policy signed_sum_policy_v1
openai_agents@1 ended stopped: target_met: a full development evaluation of the selected checkpoint meets the target
11 tool calls in 20 s
8 model requests, 68,866 input and 1,428 output tokens
edit: set_slice@1 on layers.1.weight[1:2, 0:4], values [[1.0, -1.0, 1.0, -1.0]]
layers.1.weight: [[-1.0, 1.0, -1.0, 1.0], [-1.0, 1.0, -1.0, 1.0]] -> [[-1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 1.0, -1.0]]
patch: demo/live/workspace/agent_runs/run_cec5c4dbeda7b5f5784788a717217d6d/export/patches/patch_17b4bd6d3b16e96a6c1d7e276b498acf.json
final test, 1,024 examples the agent never saw: 50% correct before the edit, 100% after
the export replays to byte-identical weights: True
```

</details>

Before editing, the agent read the four tensors and stated its hypothesis: the two output rows are identical, and setting class 1's weights to the negation of class 0's gives the logits (−s, s), where s = x0 + x1. Its first evaluation met the target; of its budget it used 1 of 10 evaluations, 1 of 20 commits, and 4 of 100 tensor reads. In the recorded five-trial acceptance set, the agent repaired the model in 5 of 5 trials ([trial report](https://github.com/xuguang-ai/weight-space-engineering/blob/main/dev/m1-agent-trials.md)); individual runs differ. The export also holds the reports, the audit log, and the tool trace.

[MCP tool contracts](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/protocol.md) · [Connecting other MCP clients](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/experiments.md#serving-a-run-to-an-mcp-client) · [Running trials](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/experiments.md#live-agent-trials)

## Results

### Published p50c edit: 12% → 52% on 25 MATH-500 problems

Applied as published, p50c was compared with the base on 25 MATH-500 problems with the article's sampling (`qwen35think@1`, up to 32,768 new tokens), one sample per problem:

| 25 MATH-500 problems | Base | p50c | Paired difference [95% interval] |
| --- | --- | --- | --- |
| Accuracy | 3/25 (12%) | 13/25 (52%) | +0.40 [+0.16, +0.60] |
| Answers cut off at the limit | 23/25 (92%) | 9/25 (36%) | −0.56 [−0.76, −0.32] |
| Mean generated tokens | 30,824 | 21,554 | −9,270 [−13,854, −4,753] |

Eleven problems were solved only with p50c, and one only with the base. The article reports 7.9% → 48.1% accuracy and 93.0% → 33.1% of answers cut off, with 4 samples on all 500 problems; twenty-five problems confirm the direction of that result, not its size.

<details>
<summary>Reproduce it (about six hours on an M4 Pro)</summary>

With p50c's directory from the four steps' downloads, fetch MATH-500 at a fixed revision. The `hf` command comes with the `models-hf` extra.

```bash
uv run --extra models-hf hf download HuggingFaceH4/MATH-500 test.jsonl \
  --repo-type dataset \
  --revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be \
  --local-dir ~/wse-data/MATH-500
```

Then save the following as `compare_p50c.py` and run it with `uv run --extra models-hf python compare_p50c.py`:

```python
from pathlib import Path

from wse import WSE

data = Path("~/wse-data").expanduser()

base = WSE.from_pretrained(
    "Qwen/Qwen3.5-0.8B",
    revision="2fc06364715b967f1860aea9cf38778875588b17",
    workspace=data / "workspace",
)

edited = base.apply_patch(
    data / "lm_understand-logs/exp8_math/runs/p50c/edit",
    repo="RikiyaT/lm_understand-logs",
    revision="4d09ba6d6a7e891dd8063974dfe81b3d78e3494f",
)

print(edited.table())
assert edited.verify()

comparison = base.compare(
    edited,
    data / "MATH-500/test.jsonl",
    problems=25,
    profile="qwen35think@1",
    out_dir=data / "runs/p50c",
)
print(comparison.table())
```

`apply_patch` refuses the edit unless every tensor matches its published SHA-256, and `verify()` replays the patch and checks every hash. The recorded run used `wse compare-generation`, which runs the same comparison. For a quick check, pass `problems=2` and `max_new_tokens=1024` and leave out `out_dir`: this takes about 80 seconds on an M4 Pro, but the short limit cuts off both models' answers.

</details>

### What has been demonstrated

| Experiment | Controller | Recorded result |
| --- | --- | --- |
| Repair a planted fault in a 22-parameter network | Live LLM agent over MCP | 5/5 acceptance trials repaired it, from 50% to 100% final accuracy |
| Apply published p50c to Qwen3.5-0.8B | Command line, same runner as Python | 3/25 → 13/25 correct on MATH-500, with a 32,768-token limit |
| Reconstruct p50c's projection edit with WSE operators | Direct Python | 2/32 → 10/32 correct on GSM8K practice problems; answers cut off at the limit fell from 30/32 to 21/32 |
| Apply that reconstruction through MCP | Scripted controller | Answers cut off at the limit fell from 30/32 to 19/32 on another practice set |

The article's three Qwen3.5-0.8B edits import to their published hashes, and WSE's own `fill_slice@1` operator reproduces one of them, p25c, hash-exactly. The reconstruction takes its direction and columns from p50c's published change; a scripted search that used its form without its values found no repair. No LLM agent has edited a language model yet: that is the next experiment.

[Results, paired intervals, and reproduction details](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/results.md) · [Live agent trial report](https://github.com/xuguang-ai/weight-space-engineering/blob/main/dev/m1-agent-trials.md)

## Current support

| Area | Status |
| --- | --- |
| Agent interface | MCP server and OpenAI Agents SDK reference controller |
| Language-model weights | Unquantized safetensors for causal models implemented natively by `transformers` |
| Verified model imports | Qwen3.5-0.8B, Llama-3.2-1B-Instruct, Qwen3-0.6B, and gemma-3-1b-it |
| Generation and activation probes | Apple MPS and CPU, tested on macOS with Apple silicon |
| Tested versions | Python 3.14, PyTorch 2.15, and `transformers` 5.22.2; WSE requires Python 3.11 or later |
| Patch replay | Exact replay of stored edits, verified by hashes |
| Whole-experiment export and replay | Small-model tasks |
| Next work | CUDA support ([handover](https://github.com/xuguang-ai/weight-space-engineering/blob/main/dev/handover-cuda.md)), long-job waiting for agents, and live agent trials on Qwen |

Local language-model evaluation is slow: a 32-problem generation pass at 8,192 tokens took about 37–40 minutes on an M4 Pro.

## Documentation

- [Experiments](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/experiments.md): configure the agent model, write tasks and budgets, run controllers, and read exports
- [MCP protocol](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/protocol.md): the tools and their contracts
- [Models](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/models.md): model support, importing weights, and tensor roles
- [Extensions](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/extensions.md): add operators, adapters, evaluators, and strategies
- [Architecture](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/architecture.md): how the core, the Engine, the MCP server, the registry, and patch storage fit together
- [Results](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/results.md): every measurement and how to reproduce it
- [Development](https://github.com/xuguang-ai/weight-space-engineering/tree/main/dev): plans, milestone reports, and handovers

## Contributing

From the repository checkout, install the extras and run the offline test suite. It uses tiny models built from configuration classes and takes about four minutes:

```bash
uv sync --extra models-hf --extra agents-openai
uv run --no-sync pytest -q
```

Keep the `--extra` flags when syncing: a plain `uv sync` removes the optional extras. Bug reports should include the command, model revision, platform, and relevant error output. [Open an issue](https://github.com/xuguang-ai/weight-space-engineering/issues)

## License

WSE is released under the [Apache License 2.0](https://github.com/xuguang-ai/weight-space-engineering/blob/main/LICENSE).
