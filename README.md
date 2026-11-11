# Weight-Space Engineering (WSE)

[![PyPI](https://img.shields.io/pypi/v/wse)](https://pypi.org/project/wse/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/wse/)
[![License](https://img.shields.io/pypi/l/wse)](https://github.com/xuguang-ai/weight-space-engineering/blob/main/LICENSE)

**Give AI agents tools to inspect, edit, and test model weights.**

WSE is an MCP harness and Python toolkit for direct weight-editing experiments. Give an agent a task, a target model, and an experiment budget. It can inspect tensors, investigate failures, try weight edits, and compare candidates through WSE's tools. WSE records the changes and checks that saved patches replay to identical weights.

Each experiment gives you a selected model checkpoint, evaluation reports, exact weight patches, and a record of the controller's actions.

The Python API and the MCP tools share the same registered edit operators, evaluators, and patch format. A new edit method, target model, task, or agent therefore reuses the rest of the experiment infrastructure: budgets, policy checks, paired evaluation, audit records, and exact replay ([what carries over](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/architecture.md#what-carries-over)).

[Run an agent](https://github.com/xuguang-ai/weight-space-engineering#run-your-first-agent-experiment) · [Use Python](https://github.com/xuguang-ai/weight-space-engineering#edit-a-language-model-with-python) · [MCP and harness](https://github.com/xuguang-ai/weight-space-engineering#what-the-agent-and-harness-do) · [Results](https://github.com/xuguang-ai/weight-space-engineering#what-has-been-demonstrated) · [Documentation](https://github.com/xuguang-ai/weight-space-engineering#documentation)

> **Current release: alpha.** A live LLM agent has repaired a small neural network through MCP. Language-model editing and evaluation have run from Python and with scripted controllers; no LLM agent has edited a language model yet. Development and testing have used macOS with Apple silicon.

## Run your first agent experiment

Start with a small model that classifies whether two numbers sum to a positive value. The example prepares a faulty 22-parameter network, development examples, a separate final test, and an editing policy for you.

The agent model is the LLM you choose to investigate the fault. The target model is the small local network it will inspect and edit: it never predicts class 1, so it is right on only half of the examples. The agent is not told this. It receives the behavior to reach, from [examples/signed_sum/task.json](https://github.com/xuguang-ai/weight-space-engineering/blob/main/examples/signed_sum/task.json):

> The registered model reads two real inputs x = (x0, x1) and returns logits for two classes; its prediction is the class with the larger logit. It should predict class 1 when x0 + x1 > 0 and class 0 otherwise. Using only the permitted inspection and editing operations, produce a candidate checkpoint that follows this rule, and select it.

From this statement, the target, the budget, and the permitted operations, WSE writes the agent's [instructions](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/experiments.md#what-the-agent-is-told). To pose your own problem, write it as a task's `description` and `objective` ([your own task](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/experiments.md#your-own-task)).

### 1. Install

The examples run from a source checkout. You need Python 3.11 or later and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/xuguang-ai/weight-space-engineering.git
cd weight-space-engineering
uv sync --extra agents-openai
```

### 2. Add your API key

Create a `.env` file in the repository root (git ignores it):

```text
OPENAI_API_KEY=your-api-key
```

### 3. Start the agent

Replace `YOUR_AGENT_MODEL` with your model or deployment name:

```bash
uv run --no-sync --env-file .env \
  python examples/signed_sum/agent_api.py \
  demo/live \
  --model YOUR_AGENT_MODEL
```

WSE prepares the small target model, connects the agent through MCP, evaluates its edits, and saves the results under `demo/live`.

Recorded trials used `gpt-5.6-luna` on Azure OpenAI. For Azure or another compatible endpoint, see the [configuration guide](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/experiments.md#agent-model-configuration).

### 4. Read the result

The script prints the controller's stop reason, whether the repair succeeded, the final comparison, the export path, and whether replay produced identical weights. A recorded run printed (the script prints the export's full path):

```text
experiment exp_cba05a78e444f8cc8be78c21af7743ad (run run_12426d61f52cb0a292122aeff29fbe42): CLOSED
controller openai_agents@1 ended stopped: target_met: a full development evaluation of the selected checkpoint meets the target
success: True; final: {"candidate_accuracy": 1.0, "clopper_pearson_lower": 0.9928210499737405, "clopper_pearson_upper": 1.0, "clusters": 512, "control_accuracy": 0.5, "difference": 0.5, "fully_correct_clusters": 512}
export: demo/live/workspace/agent_runs/run_12426d61f52cb0a292122aeff29fbe42/export; replay byte-identical: True
```

| Result | What it tells you |
| --- | --- |
| `ended stopped: target_met` | Why the run ended: here, a full development evaluation of the selected checkpoint met the target |
| `success` | Whether the selected checkpoint reached 99% accuracy on the final test, which the agent never sees |
| `control_accuracy` / `candidate_accuracy` | How the original and the edited model scored on the same 1,024 final examples |
| `replay byte-identical` | Whether the exported patches recreated the recorded weights exactly |
| `export` | Where to inspect the patches, reports, audit events, and tool trace |

In this run the agent read four tensors, saw that the two output rows had identical weights, and negated one of them: a single edit of 4 of the 22 parameters, in 11 tool calls and 8 model requests (about 30 seconds and 70,000 tokens).

Each run is saved under `demo/live/workspace/agent_runs/<run_id>/`. In the recorded five-trial acceptance set, the agent repaired the model in 5 of 5 trials, with final accuracy rising from 50% to 100%. Individual agent runs may differ. [Trial report](https://github.com/xuguang-ai/weight-space-engineering/blob/main/dev/m1-agent-trials.md)

<details>
<summary>Check the workflow offline, without a key</summary>

This runs the same experiment with a scripted controller in place of the LLM, on CPU and without network access. It takes about 20 seconds the first time and a few seconds afterwards:

```bash
uv run --no-sync python examples/signed_sum/agent_api.py demo/offline
```

The scripted strategy, `scripted_search@1`, calls WSE directly. Adding `--model wse-test-double:scripted_search@1` sends the same strategy through the agent SDK's loop and the MCP server, still offline. [Example source](https://github.com/xuguang-ai/weight-space-engineering/blob/main/examples/signed_sum/agent_api.py)

</details>

## What the agent and harness do

| You provide | The agent chooses | WSE manages |
| --- | --- | --- |
| A registered task and target checkpoint | Which tensors and reports to inspect | A run bound to that task and checkpoint |
| Development data and permitted operations | Which supported edits to propose and test | Tensor protections, policy checks, and computation budgets |
| A final evaluation suite | Which candidate to select | Checkpoint freeze, separate final evaluation, audit records, and patches |

The runner starts a run-bound `wse serve` process and connects the agent to it as an MCP server. The agent's only tools are WSE's: it cannot run code, read files, or use the network. The policy decides which tools and edits the agent may use and sets the budget for the whole experiment. In this example, `small_model_v1` grants 13 of the 15 tools and the edits `scale_slice` and `set_slice`, and allows 10 development evaluations (each one forward pass over up to 64 examples), 20 candidate commits, 50 patch previews, 100 tensor reads, and 300 seconds of computation. Spent allowance is never refunded, and every tool result reports what remains; the run above used 1 evaluation, 1 commit, and 4 tensor reads.

The server refuses any request the policy does not permit and records the refusal. Tensors that a model family protects, such as Qwen3.5's embeddings and output head, cannot be edited under any policy. The final test stays with the evaluator and runs only after the selected checkpoint is frozen. [Budgets and instructions](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/experiments.md#tasks-instructions-and-budgets) · [MCP tool contracts](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/protocol.md) · [Connecting other MCP clients](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/experiments.md#serving-a-run-to-an-mcp-client)

## Edit a language model with Python

Use the Python API to apply a known edit and compare its effect. This example applies p50c, a published intervention for Qwen3.5-0.8B from [Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang, and Khattab, 2026). In that article, an LLM, GPT-6 Astra, traced the model's looping in math answers to its weights and fixed it by editing two MLP projections; p50c is the edit it made with a budget of 50 forward passes. WSE imports the published values, verifies their hashes, and evaluates the edited model against the base. It does not discover the edit.

This needs about 3 GB of disk and is meant for a Mac with Apple silicon: WSE generates text on Apple's MPS or on the CPU, which is about 170 times slower. From the repository checkout, add support for Hugging Face models, keeping the agent extra:

```bash
uv sync --extra agents-openai --extra models-hf
```

Outside a checkout, `pip install "wse[models-hf]"` installs the same package from PyPI.

<details>
<summary>Prepare the published edit and MATH-500 data</summary>

These commands download the edit's directory and the dataset at fixed revisions; the model downloads when `from_pretrained` runs. They use Git and the Hugging Face `hf` command, which comes with the `models-hf` extra. Run the clone step once for this directory.

```bash
M=~/wse-data
mkdir -p "$M"

git clone --filter=blob:none --no-checkout \
  https://github.com/RikiyaT/lm_understand-logs \
  "$M/lm_understand-logs"
git -C "$M/lm_understand-logs" sparse-checkout set exp8_math/runs/p50c/edit
git -C "$M/lm_understand-logs" checkout 4d09ba6d6a7e891dd8063974dfe81b3d78e3494f

uv run --no-sync hf download HuggingFaceH4/MATH-500 test.jsonl \
  --repo-type dataset \
  --revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be \
  --local-dir "$M/MATH-500"
```

</details>

Then save the following as `compare_p50c.py` and run it with `uv run --no-sync python compare_p50c.py`. Downloading and importing the model takes about a minute on an M4 Pro; the 25-problem comparison takes about six hours.

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

`apply_patch` refuses the edit unless every tensor matches its published SHA-256, and `verify()` replays the patch and checks every hash. `edited.table()` prints the change with WSE's own hashes, which also cover each tensor's name, dtype, and shape:

```text
2 tensors changed, 130,188 entries; weights cb5a704b1fd2 -> b8dccf8ca463
  model.language_model.layers.15.mlp.down_proj.weight: 65,260 of 3,670,016 entries, relative L2 change 0.1064, db89c2e60dc5 -> c8df99feb499
  model.language_model.layers.19.mlp.down_proj.weight: 64,928 of 3,670,016 entries, relative L2 change 0.08106, 2c8c10a14a67 -> d0ae5310479d
```

The comparison runs both models on the same problems, prompts, and seeds with the article's sampling (`qwen35think@1`, up to 32,768 new tokens), and reports accuracy, answers cut off at the token limit, and generated tokens, with paired intervals. `out_dir` records each finished problem, so an interrupted run resumes where it stopped. For a quick check, pass `problems=2` and `max_new_tokens=1024` and leave out `out_dir`: this takes about 80 seconds on an M4 Pro, but the short limit cuts off both models' answers.

### Published p50c edit: 12% → 52% on 25 MATH-500 problems

The full comparison, with one sample per problem, recorded with `wse compare-generation`, which runs the same comparison:

| 25 MATH-500 problems | Base | p50c | Paired difference [95% interval] |
| --- | --- | --- | --- |
| Accuracy | 3/25 (12%) | 13/25 (52%) | +0.40 [+0.16, +0.60] |
| Answers cut off at the limit | 23/25 (92%) | 9/25 (36%) | −0.56 [−0.76, −0.32] |
| Mean generated tokens | 30,824 | 21,554 | −9,270 [−13,854, −4,753] |

Eleven problems were solved only with p50c, and one only with the base. The article reports 7.9% → 48.1% accuracy and 93.0% → 33.1% of answers cut off, with 4 samples on all 500 problems; twenty-five problems confirm the direction of that result, not its size.

To construct edits yourself, use WSE's registered operators to scale, set, or fill slices, remove a direction's projection, or add a low-rank term. A four-step walkthrough rebuilds p50c's projection edit with `project_out`: it diagnoses the cut-off answers, probes an MLP projection, edits, and evaluates. [Walkthrough and output](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/results.md#3-reconstructing-p50c-in-four-steps) · [Complete script](https://github.com/xuguang-ai/weight-space-engineering/blob/main/examples/p50c_four_steps.py) · [Operator API](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/extensions.md)

## What has been demonstrated

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

Local language-model evaluation is slow: a 32-problem generation pass at 8,192 tokens took about 37–40 minutes on an M4 Pro. Agent job polling currently returns within 10 seconds, so one such pass takes about 220 polls, which makes long generation jobs expensive in agent turns and context.

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

WSE is released under the [Apache License 2.0](https://github.com/xuguang-ai/weight-space-engineering/blob/main/LICENSE). Copyright 2026 Xuguang Ai.
