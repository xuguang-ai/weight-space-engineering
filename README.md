# Weight-Space Engineering (WSE)

**Agent-first · Developer-friendly · Experiment-driven.**

WSE turns direct edits of neural network weights into reproducible experiments: diagnose a behavior, look for it inside the model, change the weights, and measure the effect against the unedited model. Every edit is an exact patch with before-and-after hashes, which replays to byte-identical weights or fails.

- **Agent-first.** An MCP server gives an LLM agent 15 tools for a whole weight experiment, inside a managed experiment: a budget, a policy of permitted operations, a final evaluation the agent never sees, an audit log of every change and every refused request, and a trace of every tool call. So far a live LLM agent has used the tools on a 22-parameter model, and scripted controllers on Qwen3.5-0.8B.
- **Developer-friendly.** The same steps take a few lines of Python, on causal language models that `transformers` implements natively, with no model-specific code. The `wse` command line covers imports, comparisons, and managed experiments.
- **Experiment-driven.** Comparisons run the edited and the unedited model on the same problems, prompts, and seeds, with paired intervals. Runs are recorded and resume after an interruption, and a published edit is imported only if every tensor matches its published hash.

The reference case is Qwen3.5-0.8B and the edit p50c from [Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang, and Khattab, 2026). In that article, an LLM, GPT-6 Astra, traced the model's looping in math answers to its weights and fixed it by editing two MLP projections. p50c is the edit it made with a budget of 50 forward passes, named after its run in the article's published logs. WSE imports that edit exactly, measures it, and rebuilds it with its own operators; it does not rediscover it.

## Quick Start

Install WSE with the extra for open-weight language models. It needs Python 3.11 or later and has been tested with Python 3.14, PyTorch 2.15, and `transformers` 5.22.2 on macOS with Apple silicon:

```bash
pip install "wse[models-hf]"
```

### Reproduce a published weight edit

Download the edit and the problems; `from_pretrained` downloads the model:

```bash
M=~/wse-data
mkdir -p "$M"
git clone --filter=blob:none --no-checkout https://github.com/RikiyaT/lm_understand-logs "$M/lm_understand-logs"
git -C "$M/lm_understand-logs" sparse-checkout set exp8_math/runs/p50c/edit
git -C "$M/lm_understand-logs" checkout 4d09ba6d6a7e891dd8063974dfe81b3d78e3494f
hf download HuggingFaceH4/MATH-500 test.jsonl --repo-type dataset \
  --revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be --local-dir "$M/MATH-500"
```

Import p50c as an exact patch, verify it, and compare the edited model with the base:

```python
from pathlib import Path

from wse import WSE

M = Path("~/wse-data").expanduser()

base = WSE.from_pretrained("Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17",
                           workspace=M / "workspace")   # imported once; about 1.5 GB
p50c = base.apply_patch(M / "lm_understand-logs/exp8_math/runs/p50c/edit",
                        repo="RikiyaT/lm_understand-logs", revision="4d09ba6d6a7e891dd8063974dfe81b3d78e3494f")
print(p50c.table())     # what changed, with before and after hashes
assert p50c.verify()    # replays the exact patch on the base and checks every hash

# qwen35think@1 is the article's sampling: thinking on, a boxed answer, up to 32,768 new tokens
comparison = base.compare(p50c, M / "MATH-500/test.jsonl", problems=2, profile="qwen35think@1",
                          max_new_tokens=1024)
print(comparison.table())
```

The patch table, with WSE's own hashes, which also cover each tensor's name, dtype, and shape:

```text
2 tensors changed, 130,188 entries; weights cb5a704b1fd2 -> b8dccf8ca463
  model.language_model.layers.15.mlp.down_proj.weight: 65,260 of 3,670,016 entries, relative L2 change 0.1064, db89c2e60dc5 -> c8df99feb499
  model.language_model.layers.19.mlp.down_proj.weight: 64,928 of 3,670,016 entries, relative L2 change 0.08106, 2c8c10a14a67 -> d0ae5310479d
```

`apply_patch` refuses the edit unless every tensor matches the SHA-256 published with it in the article's logs. On an M4 Pro the download and import take about a minute. The two-problem comparison takes about 80 seconds and only checks that generation works: at 1,024 new tokens, the limit cuts off both models on both problems. The effect shows at the profile's full limit of 32,768 tokens, in the results below.

### Run an agent experiment through MCP

From a clone of this repository, one command runs a whole managed experiment on the small model of WSE's first milestone, a 22-parameter MLP with a planted fault. It grants a run, starts `wse serve` as the MCP server, runs the OpenAI Agents SDK agent against it, freezes the experiment, runs the hidden final evaluation, and exports and replays the result:

```bash
git clone https://github.com/xuguang-ai/weight-space-engineering && cd weight-space-engineering
uv sync --extra agents-openai
uv run --no-sync python examples/signed_sum/agent_api.py ~/wse-data/agent --model wse-test-double:scripted_search@1
```

`wse-test-double:scripted_search@1` stands in for the language model with a scripted strategy, so this runs offline, without an API key, in about 15 seconds; everything else is the real harness. It printed (IDs differ per run; the export path is shortened):

```text
experiment exp_16bf675040e94703766b41bc8ff30f9f (run run_3652b71bbea0711c4e73c5a69be29d63): CLOSED
controller openai_agents@1 ended stopped: target_met: a full development evaluation of the selected checkpoint meets the target
success: True; final: {"candidate_accuracy": 1.0, "clopper_pearson_lower": 0.9928210499737405, "clopper_pearson_upper": 1.0, "clusters": 512, "control_accuracy": 0.5, "difference": 0.5, "fully_correct_clusters": 512}
export: ~/wse-data/agent/workspace/agent_runs/run_3652b71bbea0711c4e73c5a69be29d63/export; replay byte-identical: True
```

For a live agent, pass `--model MODEL_NAME`, a model that the OpenAI Agents SDK can reach with the credentials in your environment (for example `OPENAI_API_KEY`). In WSE's first milestone, `gpt-5.6-luna` repaired this fault in 5 of 5 trials. This task shows the harness at work, not what agents can do on language models: on Qwen3.5-0.8B, only scripted controllers have run so far.

## Results

> **MATH-500 accuracy: 12% → 52%.** The published p50c edit against the unedited Qwen3.5-0.8B, on 25 problems with a paired evaluation. WSE reproduces and evaluates this intervention; it does not discover it.

Passing `problems=25` and `out_dir=...` (which records every problem and resumes an interrupted run), and leaving out `max_new_tokens`, runs the comparison of the first Quick Start example on 25 MATH-500 problems, one sample each, at the full 32,768-token limit. The recorded run used `wse compare-generation`, which calls the same runner with the same defaults, and took about six hours on an M4 Pro:

| 25 MATH-500 problems | Base | p50c | Paired difference [95% interval] |
| --- | --- | --- | --- |
| Accuracy | 3/25 (12%) | 13/25 (52%) | +0.40 [+0.16, +0.60] |
| Answers cut off at the limit | 23/25 (92%) | 9/25 (36%) | −0.56 [−0.76, −0.32] |
| Mean generated tokens | 30,824 | 21,554 | −9,270 [−13,854, −4,753] |

Eleven problems were solved only with p50c, and one only with the base. The article reports 7.9% → 48.1% accuracy and 93.0% → 33.1% of answers cut off, with 4 samples on all 500 problems. Twenty-five problems confirm the direction of that result, not its size.

Also measured:

- **Published edits.** The article's three Qwen3.5-0.8B edits, p10c, p25c, and p50c, import to their published hashes. WSE's own `fill_slice@1` operator reproduces p25c hash-exactly.
- **A reconstruction.** WSE's `project_out` operator re-executed p50c's form from its published change (next section). On 32 GSM8K practice problems, answers cut off at the limit fell from 30 to 21, and accuracy rose from 2 to 10. Two full runs recorded the same 64 answers, token for token.
- **Through MCP.** Scripted controllers drove the MCP tools on Qwen3.5-0.8B. Applied through the tools on another 32 problems, the same reconstruction reduced the answers cut off at the limit from 30 to 19. A scripted search that read none of p50c's values, but borrowed its form, found no repair.
- **A live agent.** An LLM agent repaired a fault planted in a 22-parameter model in 5 of 5 trials.

Setups, every number, and caveats are in [docs/results.md](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/results.md).

## How WSE Works

An experiment follows four steps, each with its own tools:

| Step | Question | WSE |
| --- | --- | --- |
| **Behavioral diagnosis** | How does the model fail? | Generation evaluation: accuracy, the rate of answers cut off at the token limit, and generated tokens, with intervals |
| **Mechanism investigation** | Where inside the model? | Weight analyses (norms, singular vectors, projections) and activation probes (mean activations over the last tokens of each output, at the residual stream or any module, and contrasts between groups of outputs) |
| **Weight intervention** | What change fixes it? | Exact patches from registered operators (scale, set, fill, project out a direction, add a low-rank term), or published edits imported with hash checks |
| **Evaluation** | Did it help, and at what cost? | A paired comparison with the base on the same problems, resumable and recorded |

### Reconstructing a Published Intervention: Diagnosis → Probe → Edit → Evaluate

Astra found p50c itself, by analyzing the weights in a Python kernel and spending its 50 forward passes on practice problems. This example does not rediscover it. In each of layers 15 and 19, p50c subtracts six times the projection of 64 columns of `mlp.down_proj` onto one direction of that layer. The example reads that direction and those columns from p50c's published change, and WSE's own `project_out` operator re-executes the edit.

The practice problems come from GSM8K's training set, so the MATH-500 problems stay unseen:

```bash
curl -L --create-dirs -o ~/wse-data/gsm8k/train.jsonl \
  https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl
```

The four steps, on the model and the edit of the first Quick Start example:

```python
GSM8K = M / "gsm8k/train.jsonl"
SITE = "layers.15.mlp.down_proj:output"

# 1. Diagnose: which answers run into the token limit?
run = base.evaluate(GSM8K, format="gsm8k", problems=32, profile="qwen35think@1", max_new_tokens=8192)

# 2. Probe: how do cut-off and finished answers differ at the output of an MLP projection?
contrast = base.probe(run, sites=[SITE], last=1024).contrast(SITE, run.limit_hit(), run.finished())

# 3. Edit: project a direction out of columns of that projection, in layers 15 and 19.
model = base
for layer in (15, 19):
    name, direction, columns = recover(layer)  # p50c's direction and columns, read from its published change
    model = model.edit(name, "project_out", direction=direction, columns=columns, alpha=6.0).save().model

# 4. Evaluate: a paired, recorded comparison on the same problems, prompts, and seeds.
comparison = base.compare(model, GSM8K, format="gsm8k", problems=32, profile="qwen35think@1",
                          max_new_tokens=8192, out_dir=M / "runs/four-steps")
print(comparison.table())
```

[examples/p50c_four_steps.py](https://github.com/xuguang-ai/weight-space-engineering/blob/main/examples/p50c_four_steps.py) is the complete script: it defines `recover`, probes both layers, and compares each contrast with p50c's direction. On an M4 Pro it took 1 h 54 min, almost all of it in three generation passes of about 37 minutes each; a generation pass generates one answer to each problem, and `compare` makes two. It printed (timing lines omitted):

```text
32 problems, 32 samples
accuracy                     0.062 [0.000, 0.156]
limit_hit_rate               0.938 [0.844, 1.000]
mean_generated_tokens       8063.5 [7843.9, 8192.0]
30 answers cut off at the limit, 2 finished
layer 15: cosine of the contrast with p50c's direction +0.37
layer 19: cosine of the contrast with p50c's direction -0.45
130,134 weights changed
32 problems, 32 samples per model
metric                        base   candidate   difference [95% interval]
accuracy                     0.062       0.312   +0.250 [+0.125, +0.406]
limit_hit_rate               0.938       0.656   -0.281 [-0.438, -0.125]
mean_generated_tokens       8063.5      7082.0   -981.5 [-1597.1, -434.7]
```

Answers cut off at the limit fell from 30 to 21 of 32, and accuracy rose from 2 to 10; all three intervals exclude zero. Two full runs recorded the same answers, token for token. The probe's contrast leans toward p50c's direction, but with only 2 finished answers it points to where to look; it does not establish a mechanism. The re-executed edit is close to p50c's weights but not identical, because the two round differently. [docs/results.md](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/results.md) has the full analysis.

## Agent Harness & MCP

### Connect an agent

An operator grants a run, a permission for one experiment under one policy, and serves it over MCP; the agent connects to that server and works only through its tools. On the workspace that the Quick Start's agent experiment created:

```bash
A=~/wse-data/agent
BASE=$(uv run --no-sync python -c "import json, sys; print(json.load(open(sys.argv[1]))['task_checkpoint_id'])" "$A/ids.json")
RUN=$(uv run --no-sync wse --workspace "$A/workspace" grant-run signed_sum_v1 small_model_v1 "$BASE" \
      | uv run --no-sync python -c "import json, sys; print(json.load(sys.stdin)['run_id'])")
uv run --no-sync wse --workspace "$A/workspace" serve --run-id "$RUN"   # an MCP server over stdio, bound to that run
```

From Python, `run_agent_experiment` does it in one call: it grants a run, runs the controller, and freezes the experiment when the controller stops. With `model=...`, the controller is the OpenAI Agents SDK agent, connected to the run's MCP server:

```python
import json
from pathlib import Path

from wse.agent_runs import run_agent_experiment

A = Path("~/wse-data/agent").expanduser()   # written by the Quick Start's agent experiment
ids = json.loads((A / "ids.json").read_text())
run = run_agent_experiment(A / "workspace", task=ids["task_id"], base=ids["task_checkpoint_id"],
                           policy=ids["policy_id"], model="wse-test-double:scripted_search@1")
print(run.state, run.summary["status"])   # FROZEN stopped: the final evaluation is off by default
```

[examples/signed_sum/agent_api.py](https://github.com/xuguang-ai/weight-space-engineering/blob/main/examples/signed_sum/agent_api.py) is the complete script, and [docs/experiments.md](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/experiments.md#one-experiment-in-one-call) explains the final evaluation, exports, and series of trials.

### What the agent gets

The 15 tools cover the four steps: generation passes and reports, inspection, weight analyses and probes, patch previews and commits, paired comparisons, and checkpoint selection. [docs/protocol.md](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/protocol.md) specifies each one.

A policy sets the permitted operations and the budget. `qwen_loop_v1`, for example, allows up to 10 generation passes of at most 32 problems at 8,192 tokens and 25 probes, within 6 hours of computation. Tied embeddings are never editable, and under every policy the edit operators refuse the tensors that a family descriptor protects: for Qwen3.5, the embeddings, the output head, and the final norm. When the controller stops, the experiment freezes, and only then does the operator run the final evaluation on a suite the controller never saw.

Evidence so far:

- **A live agent, on a small model.** An OpenAI Agents SDK agent repaired a fault planted in a 22-parameter MLP in 5 of 5 trials, and every trial's export replayed to byte-identical weights.
- **Qwen3.5-0.8B through MCP, with scripted controllers.** The tools expressed the p50c reconstruction (answers cut off: 30 → 19 of 32). A scripted search that borrowed p50c's form but none of its values found no repair. No LLM agent has run these steps on a language model yet.

## Architecture & Documentation

How the parts fit:

- **The core** works in memory on a model handle: it loads, analyzes, probes, edits, evaluates, and replays.
- **`Engine`** adds a workspace (SQLite metadata and content-addressed files) with imports, stored checkpoints, and managed experiments.
- **The MCP server** exposes one managed experiment to one agent.
- **`WSE`**, the Python entry point, wraps the core and the engine for direct use. Below it sits the low-level API: `load_model`, `apply_operator`, `replay_patch`, and `Engine`.
- **Extensions** are registered by name and version: model adapters, family descriptors (per-family name maps, roles, and protections), edit operators, evaluators, graders, sampling profiles, and strategies.

Design choices:

- **Exact patches.** A patch records every step's operator version, selection, after-values in the stored dtype, and the tensor hashes before and after. Replay writes those values without running the operator and verifies every hash. Edit arithmetic is platform-independent: exact products, correctly rounded sums, and round-to-nearest-even to BF16, F16, or F32.
- **Models.** Causal language models that `transformers` implements natively, as unquantized safetensors, need no model-specific code when the checkpoint's tensor names match the module (Level 0). Family descriptors, which are registered data, map other names and add semantic roles and protected tensors (Level 1). Verified so far: Qwen3.5-0.8B with its descriptor, and Llama-3.2-1B-Instruct, Qwen3-0.6B, and gemma-3-1b-it at Level 0 (Llama has since gained a descriptor, `llama@1`).
- **Evaluation.** Registered sampling profiles: `qwen35think@1` reproduces the article's sampling (thinking on, a boxed answer, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, up to 32,768 tokens). Graders score MATH and GSM8K answers. Comparisons are paired, with cluster-bootstrap and exact intervals.
- **Safety.** The importer reads only allow-listed files: it never unpickles or runs repository code, and it refuses `trust_remote_code` architectures. Agents get no Python, shell, gradients, or training, only declarative, metered tools.

Names may be short (`layers.15.mlp.down_proj` for `model.language_model.layers.15.mlp.down_proj.weight`), indices are NumPy-style (`[:, :64]`), and roles name tensors across families (`base.role("mlp.down", 15)`). Every patch still records full names and explicit ranges.

Documents:

- [docs/results.md](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/results.md): every measurement so far, with setups and caveats.
- [docs/models.md](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/models.md): supported models, support levels, importing, roles and protections, and verified models.
- [docs/extensions.md](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/extensions.md): the extension contracts, the edit operators and analyses, vector references, and exact replay versus recipe re-execution.
- [docs/experiments.md](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/experiments.md): running managed experiments, and what each trial records and exports.
- [docs/protocol.md](https://github.com/xuguang-ai/weight-space-engineering/blob/main/docs/protocol.md): the MCP tools, resources, errors, and limits.
- [dev/](https://github.com/xuguang-ai/weight-space-engineering/tree/main/dev): how WSE is being built: the plans, the milestone reports (M1's agent trials, M2's Q1 and Q2), and handovers.

To work on WSE itself, clone the repository and use [uv](https://docs.astral.sh/uv/). The tests run offline, on tiny models built from configuration classes, in about four minutes:

```bash
uv sync --extra models-hf --extra agents-openai
uv run --no-sync pytest -q
```

Run commands with `uv run --no-sync`: `uv sync` without the `--extra` flags removes the extras.

## Limitations & Roadmap

| Milestone | Status |
| --- | --- |
| **M1:** contracts, managed experiments, MCP, and an LLM agent repairing a small model | Done |
| **M2:** open-weight language models: import, exact replay of published edits, generation evaluation, and agent-usable tools for probes, edits, and budgets | Built and measured ([M2 Q1](https://github.com/xuguang-ai/weight-space-engineering/blob/main/dev/m2-q1-report.md) and [Q2](https://github.com/xuguang-ai/weight-space-engineering/blob/main/dev/m2-q2-report.md) reports) |
| **CUDA:** generation and probes on NVIDIA GPUs | Planned; [handover](https://github.com/xuguang-ai/weight-space-engineering/blob/main/dev/handover-cuda.md) written |
| **Agent trials on Qwen3.5-0.8B:** preregistered trials of LLM agents | Next |

Known limits:

- Generation and probes run on Apple's MPS or the CPU only, and WSE has been tested only on macOS with Apple silicon. On the CPU, a decoding step of Qwen3.5 takes 5–10 s, about 170 times slower than on MPS.
- Local generation is slow: 37–40 minutes per model for 32 problems at 8,192 tokens on an M4 Pro.
- LLM agents cannot yet wait on long jobs: `wse_get_job` returns within 10 seconds, so one 37-minute generation pass takes about 220 polls, enough to outgrow a model's context window.
- Whole managed experiments export and replay only for the small models of M1; patches of open-weight models replay exactly.
- No LLM agent has yet run the four steps on a language model, and the evidence so far comes from samples of 25 and 32 problems and one synthetic task.

## License

WSE is released under the [Apache License 2.0](https://github.com/xuguang-ai/weight-space-engineering/blob/main/LICENSE). Copyright 2026 Xuguang Ai.
