# Weight-Space Engineering (WSE)

**A reproducible framework for programmatic and agent-driven neural network weight editing.**

Neural network weights can be inspected and modified directly, without gradient-based training. Turning such interventions into reproducible, auditable experiments takes more than tensor operations. WSE is a common framework for the whole experiment:

- inspect a model's weights and internal activations;
- apply a targeted edit;
- evaluate how the model's behavior changes;
- record exactly what changed.

Every edit produces an exact patch with before-and-after tensor hashes, which replays to byte-identical weights or fails. People use WSE from Python or the `wse` command line. Agents work through an MCP server, inside managed experiments with explicit budgets, a policy of permitted operations, and an audit log of every action. The goal is to make weight-space interventions programmable, reproducible, and independently verifiable.

## 1. Overview

PyTorch can already change a weight. WSE turns the change into an experiment: a named intervention on an identified model, evaluated against the unedited model on the same problems, prompts, and seeds, and replayable by anyone who has the base checkpoint. It is experimental infrastructure for weight-space interventions, not a tensor-editing library.

An experiment follows four steps, each with its own tools:

| Step | Question | WSE |
| --- | --- | --- |
| **Behavioral diagnosis** | How does the model fail? | Generation evaluation: accuracy, the rate of answers cut off at the token limit, and generated tokens, with intervals |
| **Mechanism investigation** | Where inside the model? | Weight analyses (norms, singular vectors, projections) and activation probes (window means at the residual stream or any module, and contrasts between groups of outputs) |
| **Weight intervention** | What change fixes it? | Exact patches from registered operators (scale, set, fill, project out a direction, add a low-rank term), or published edits imported with hash checks |
| **Evaluation** | Did it help, and at what cost? | A paired comparison with the base on the same problems, resumable and recorded |

How the parts fit:

- **The core** works in memory on a model handle: it loads, analyzes, probes, edits, evaluates, and replays.
- **`Engine`** adds a workspace (SQLite metadata and content-addressed files) with imports, stored checkpoints, and managed experiments.
- **The MCP server** exposes one managed experiment to one agent.
- **`WSE`**, the Python entry point, wraps the core and the engine for direct use.
- **Extensions** are registered by name and version: model adapters, family descriptors (per-family name maps, roles, and protections), edit operators, evaluators, graders, sampling profiles, and strategies.

WSE's first open-weight target is Qwen3.5-0.8B, and the reference case is experiment 8 of [Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang, and Khattab, 2026). That experiment published bit-exact edits aimed at the model's looping in its math answers. By the article's numbers, the strongest, p50c, cut the share of MATH-500 answers that reach the 32,768-token limit from 93.0% to 33.1%. p50c changes 130,188 weights in two MLP projections.

## 2. Key Features

- **Exact patches.** A patch records every step's operator version, selection, after-values in the stored dtype, and the tensor hashes before and after. Replay writes those values without running the operator and verifies every hash. Edit arithmetic is platform-independent by construction: exact products, correctly rounded sums, and round-to-nearest-even to BF16, F16, or F32.
- **Published edits.** `import-patch` rebuilds an edit from its published files and refuses it unless every tensor matches its published SHA-256. The article's three Qwen3.5-0.8B edits (p10c, p25c, p50c) import to their published hashes. WSE's own `fill_slice@1` operator also reproduces p25c hash-exactly.
- **Evaluation.** Generation suites run with registered sampling profiles. `qwen35think@1` reproduces the article's sampling: thinking on, a boxed answer, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, up to 32,768 tokens. Graders score MATH and GSM8K answers. Comparisons are paired, with cluster-bootstrap and exact intervals. The recorded runner resumes after an interruption.
- **Investigation.** Analyses (`tensor_norms@1`, `svd_top_k@1`, `direction_projection@1`) and activation probes. In managed experiments, their vectors can be passed by reference to the edit operators, so an agent turns a direction found by a probe or an analysis into an edit without copying numbers.
- **Agent experiments.** A run grant fixes the task (its development and final evaluation suites), the policy, and the base checkpoint, and allows one experiment. Budgets are counted in calls, examples, tokens, and compute seconds. Every action, including refused ones, is an audit event. The final evaluation is hidden from the agent. Closed experiments on the small MLP models of WSE's first milestone, M1, are exported and replayed as a whole; patches of open-weight models replay exactly. Controllers are LLM agents (an OpenAI Agents SDK adapter) or registered stepwise strategies.
- **Models.** Causal language models that `transformers` implements natively, as unquantized safetensors (BF16, F16, or F32), with no model-specific code. When the checkpoint's tensor names match the module, nothing else is needed (Level 0). Family descriptors, which are registered data, map other names (Qwen3.5's multimodal checkpoint) and add semantic roles, attention-head groupings, and protected tensors (Level 1). Verified so far: Qwen3.5-0.8B and Llama-3.2-1B-Instruct with descriptors, and Qwen3-0.6B and gemma-3-1b-it at Level 0 ([docs/models.md](docs/models.md)).
- **Safety.** The importer reads only allow-listed files: it never unpickles or runs repository code, and it refuses `trust_remote_code` architectures. Agents get no Python, shell, gradients, or training, only declarative, metered tools.

## 3. Installation

WSE uses [uv](https://docs.astral.sh/uv/) and Python 3.11 or later. It has been tested with Python 3.14, PyTorch 2.15, and `transformers` 5.22.2 on macOS with Apple silicon. Generation and probes run on Apple's MPS or on the CPU; there is no CUDA support yet, and Qwen3.5 on the CPU takes 5–10 s per generated token, so Sections 4 and 5 need an Apple-silicon Mac. From the repository root:

```bash
uv sync --extra models-hf --extra agents-openai
```

- **`models-hf`** adds `transformers` (which brings `huggingface_hub`) and `math-verify` for open-weight language models.
- **`agents-openai`** adds the OpenAI Agents SDK for the reference agent.
- **The core** alone needs neither extra; it is enough for M1's small models.

Run commands with `uv run --no-sync ...`, so that uv uses the environment as `uv sync` built it; running `uv sync` again without the `--extra` flags removes the extras. Install from this repository: the `wse` package on PyPI only holds the name for the first release and contains no code yet.

## 4. Quick Start: Reproduce a Published Weight Edit

This example loads Qwen3.5-0.8B, applies the article's p50c edit as an exact patch, checks it, and compares the two models on MATH-500. First download the edit and the problems. The model is downloaded by `from_pretrained`.

```bash
M=~/wse-data
git clone --filter=blob:none --no-checkout https://github.com/RikiyaT/lm_understand-logs "$M/lm_understand-logs"
git -C "$M/lm_understand-logs" sparse-checkout set exp8_math/runs/p50c/edit
git -C "$M/lm_understand-logs" checkout 4d09ba6d6a7e891dd8063974dfe81b3d78e3494f
uv run --no-sync hf download HuggingFaceH4/MATH-500 test.jsonl --repo-type dataset \
  --revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be --local-dir "$M/MATH-500"
```

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

comparison = base.compare(p50c, M / "MATH-500/test.jsonl", problems=2, profile="qwen35think@1",
                          max_new_tokens=1024)
print(comparison.table())
```

On an M4 Pro, the download and import took about a minute and the comparison 78 seconds. It prints:

```text
2 tensors changed, 130,188 entries; weights cb5a704b1fd2 -> b8dccf8ca463
  model.language_model.layers.15.mlp.down_proj.weight: 65,260 of 3,670,016 entries, relative L2 change 0.1064, db89c2e60dc5 -> c8df99feb499
  model.language_model.layers.19.mlp.down_proj.weight: 64,928 of 3,670,016 entries, relative L2 change 0.08106, 2c8c10a14a67 -> d0ae5310479d
2 problems, 2 samples per model
metric                        base   candidate   difference [95% interval]
accuracy                     0.000       0.000   +0.000 [+0.000, +0.000]
limit_hit_rate               1.000       1.000   +0.000 [+0.000, +0.000]
mean_generated_tokens       1024.0      1024.0   +0.0 [+0.0, +0.0]
```

`from_pretrained` resolves the revision to its commit and downloads only the allow-listed files into the Hugging Face cache. `apply_patch` checked every tensor against the SHA-256 that the article published. The table shows WSE's own hashes, which also cover each tensor's name, dtype, and shape; `b8dccf8ca463…` is the edited model's weights hash.

At 1,024 new tokens, both models hit the limit on both problems, so this run shows only that the pipeline works. The difference appears at the profile's full limit of 32,768 tokens. With `problems=25`, no `max_new_tokens`, and `out_dir=...` (which records every problem and resumes an interrupted run), this is the comparison in [docs/m2-q1-report.md](docs/m2-q1-report.md). It took about six hours on an M4 Pro:

| 25 MATH-500 problems, 1 sample each | Base | p50c | Paired difference [95% interval] |
| --- | --- | --- | --- |
| Accuracy | 3/25 (12%) | 13/25 (52%) | +0.40 [+0.16, +0.60] |
| Answers cut off at the limit | 23/25 (92%) | 9/25 (36%) | −0.56 [−0.76, −0.32] |
| Mean generated tokens | 30,824 | 21,554 | −9,270 [−13,854, −4,753] |

Eleven problems were solved only with p50c, and one only with the base. Twenty-five problems confirm the direction of the article's result, not its size: the article reports 7.9% → 48.1% accuracy and 93.0% → 33.1% of answers cut off, with 4 samples on all 500 problems.

## 5. Programmatic Weight Editing

The four steps in Python, on the model and the edit loaded above. Practice problems come from GSM8K's training set, so that the MATH-500 problems stay unseen:

```bash
curl -L --create-dirs -o ~/wse-data/gsm8k/train.jsonl \
  https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl
```

p50c acts on one direction in 64 columns of `mlp.down` in layers 15 and 19: it subtracts six times each column's projection onto that direction, which turns that component into −5 times itself (alpha 1 would remove it). `recover` reads the columns and the direction from p50c's published change: the columns it changes, and the top singular vector of its change to them, signed as `svd_top_k@1` signs its vectors.

```python
import numpy as np

GSM8K = M / "gsm8k/train.jsonl"
SITES = {layer: f"layers.{layer}.mlp.down_proj:output" for layer in (15, 19)}


def recover(layer):
    """p50c in one layer: the tensor, the direction it acts on (its change's top singular vector), its columns."""
    name = base.name(f"layers.{layer}.mlp.down_proj")
    delta = p50c.model.tensor(name).double().numpy() - base.tensor(name).double().numpy()
    columns = np.flatnonzero(delta.any(axis=0))
    d = np.linalg.svd(delta[:, columns], full_matrices=False)[0][:, 0]
    d *= np.sign(d[np.argmax(np.abs(d))])  # the sign convention of svd_top_k: the largest entry is positive
    return name, d, columns


# 1. Behavioral diagnosis: which answers never finish?
run = base.evaluate(GSM8K, format="gsm8k", problems=32, profile="qwen35think@1", max_new_tokens=8192)
print(run.table())
looping, finished = run.limit_hit(), run.finished()
print(f"{len(looping)} answers cut off at the limit, {len(finished)} finished")

# 2. Mechanism investigation: do cut-off and finished answers differ along p50c's direction?
if looping and finished:  # a contrast needs both groups
    probe = base.probe(run, sites=list(SITES.values()), last=1024)
    for layer, site in SITES.items():
        _, d, _ = recover(layer)
        contrast = probe.contrast(site, looping, finished)
        cosine = contrast.direction @ d / contrast.norm
        print(f"layer {layer}: cosine of the contrast with p50c's direction {cosine:+.2f}")
else:
    print("step 2 needs at least one cut-off and one finished answer")

# 3. Weight intervention: p50c's form, project_out of that direction from its columns with alpha 6.
model = base
for layer in SITES:
    name, d, columns = recover(layer)
    model = model.edit(name, "project_out", direction=d, columns=columns, alpha=6.0).save().model
print(f"{sum(diff.changed_elements for diff in model.diff_from(base)):,} weights changed")

# 4. Evaluation: a paired, recorded comparison on the same problems, prompts, and seeds.
comparison = base.compare(model, GSM8K, format="gsm8k", problems=32, profile="qwen35think@1",
                          max_new_tokens=8192, out_dir=M / "runs/four-steps")
print(comparison.table())
```

The script [examples/p50c_four_steps.py](examples/p50c_four_steps.py) runs these steps and times each one. On an M4 Pro it took 1 h 54 min, almost all of it in three generation passes of about 37 minutes each. A generation pass generates one answer to each problem: the diagnosis makes one, and `compare` makes two, because it generates the base's answers again. It printed (timing lines omitted):

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

Two full runs in separate processes, the second from a fresh download of the model and the data, recorded the same weights and the same 64 answers in the comparison, token for token ([Q2 report, Section 6.2](docs/m2-q2-report.md#62-reproducibility-of-a-generation-pass)).

Step by step:

1. **Diagnosis.** 30 of the 32 answers ran into the 8,192-token limit, and 2 finished.
2. **Investigation.** The contrast between the 30 cut-off and the 2 finished answers has cosines of +0.37 (layer 15) and −0.45 (layer 19) with p50c's direction. A singular vector's sign is arbitrary, so the size is what counts, and it needs the right baseline. These answers vary along p50c's direction anyway: it carries 7% and 14% of their variance in the two layers, against 0.1% for a random direction. So even arbitrary groups give cosines of 0.24 and 0.36, the medians over all 496 ways to set 2 of the 32 answers apart. The actual split ranks 4th and 7th of those 496. Along the direction, the 2 finished answers have the smallest components in magnitude of all 32 answers in layer 15, and two of the three smallest in layer 19. With 2 finished answers, and windows that end at an answer in one group and inside a loop in the other, this points to where to look; it does not establish a mechanism ([Q2 report, Section 4.3](docs/m2-q2-report.md#43-why-it-missed)).
3. **Intervention.** The two `project_out` edits change 130,134 weights, and p50c changes 130,188. About 91% of the edited entries equal p50c's byte for byte. The rest differ by at most 0.00049, most of them by one BF16 spacing: none of five rounding paths tried reproduces p50c's arithmetic exactly ([Q2 report, Section 6.1](docs/m2-q2-report.md#61-recipe-re-execution-versus-exact-replay-task-9)).
4. **Evaluation.** Answers cut off at the limit fell from 30 to 21 of 32, accuracy rose from 2 to 10 of 32, and answers became about 980 tokens shorter on average. All three intervals exclude zero.

This reproduces a known repair. The direction and the columns are read from p50c's published change, so the script shows the workflow, not a discovery. Through the MCP tools, a scripted search that started from behavior, with p50c's form but none of its values, did not find a repair (Section 6).

Names may be short (`layers.15.mlp.down_proj` for `model.language_model.layers.15.mlp.down_proj.weight`), indices are NumPy-style (`[:, :64]`), and roles name tensors across families (`base.role("mlp.down", 15)`). Every patch still records full names and explicit ranges. Below `WSE` sits the low-level API: `load_model`, `apply_operator`, and `replay_patch` ([docs/extensions.md](docs/extensions.md)), and `Engine` for workspaces and managed experiments ([docs/experiments.md](docs/experiments.md)).

## 6. Agent-Driven Experiments

A managed experiment separates two roles:

- **The operator**, the person who runs the experiment, imports the base model, registers the task (development and final evaluation suites) and the policy, and grants a run.
- **The controller** works on that run's single experiment within the policy's budget. It is an LLM agent through the MCP server, or a registered stepwise strategy.

The controller inspects, analyzes, probes, previews and applies patches, runs development evaluations, and selects a checkpoint. Every action, including a refused one, is an audit event. When the controller stops, the experiment is frozen. Only then does the operator run the final evaluation on a suite the controller never saw. For M1's small models, the closed experiment is then exported and its patches replayed.

The MCP server (`wse serve --run-id RUN`) serves one run grant and offers 15 tools:

| Purpose | Tools |
| --- | --- |
| The experiment | `wse_create_experiment`, `wse_get_experiment`, `wse_list_models` |
| Diagnosis | `wse_submit_evaluation` (also generation passes, which generate an answer to every problem of a suite), `wse_get_job`, `wse_cancel_job`, `wse_get_report` |
| Investigation | `wse_inspect_model`, `wse_read_tensor`, `wse_analyze`, `wse_probe` |
| Intervention | `wse_preview_patch`, `wse_apply_patch` |
| Evaluation | `wse_compare`, `wse_select_checkpoint` |

The tools follow these rules:

- every mutation takes an idempotency key;
- results stay within WSE's response size limits ([docs/protocol.md](docs/protocol.md)), with paging for large models;
- an argument naming another run's data fails exactly as a missing one does.

Policies set the permitted operations and the budget. `small_model_v1` serves M1's MLP task. `qwen_loop_v1` serves language models, for example: 10 generation passes of at most 32 problems at 8,192 tokens, 25 probe calls, 20 candidate commits, and 6 hours of computation per run. Tied embeddings are never editable. With a family descriptor (built in for Qwen3.5 and Llama), the embeddings, the output head, and the final norm are protected as well.

One experiment in one call, on M1's task: `wse.agent_runs.run_agent_experiment` grants a run, runs the controller, and freezes the experiment. With `final_evaluation=True`, as the example passes, it also runs the final evaluation, exports the closed experiment, and replays the export. [examples/signed_sum/agent_api.py](examples/signed_sum/agent_api.py) builds M1's workspace on its first use and runs one experiment. Offline, with a scripted controller in place of a model, it takes a few seconds (the IDs differ per run, and the script prints the export path in full, shortened here to `OUT/...`):

```bash
uv run --no-sync python examples/signed_sum/agent_api.py OUT
```

```text
experiment exp_a0373ebd434304fad7bacc2ef3280b99 (run run_ad9c7452e48359213d4dbef2da5e6a2c): CLOSED
controller scripted_search@1 ended stopped: target_met: a full development evaluation of the selected checkpoint meets the target
success: True; final: {"candidate_accuracy": 1.0, "clopper_pearson_lower": 0.9928210499737405, "clopper_pearson_upper": 1.0, "clusters": 512, "control_accuracy": 0.5, "difference": 0.5, "fully_correct_clusters": 512}
export: OUT/workspace/agent_runs/run_ad9c7452e48359213d4dbef2da5e6a2c/export; replay byte-identical: True
```

For a live LLM agent, add `--model MODEL_NAME`, with the model provider's credentials in the environment. For a series of trials, leave the final evaluation off (the default) and run it only after every trial has frozen, so that no trial can adapt to an earlier trial's final result.

**Evidence so far:**

- **M1, a live agent.** An OpenAI Agents SDK agent (`gpt-5.6-luna`) repaired a fault planted in a 22-parameter MLP in 5 of 5 trials of the acceptance set (an earlier set reached 3 of 5 because of a harness defect, since fixed). It used 9–12 turns and 12–16 tool calls per trial. Every trial reached a final accuracy of 1.0 on 512 pairs (Clopper–Pearson lower bound 0.9928), against 0.5 for the unedited model, and its export replayed to byte-identical weights ([docs/m1-agent-trials.md](docs/m1-agent-trials.md)). Repairing one synthetic fault says little about agent capability in general.
- **M2, through MCP on Qwen3.5-0.8B, with scripted controllers** (no LLM agent yet). Both runs used the same 32 GSM8K practice problems, a different draw from Section 5's ([docs/m2-q2-report.md](docs/m2-q2-report.md)).
  - **A known repair.** The tools expressed p50c's form with a direction recovered from the published diff, and answers cut off at the limit fell from 30/32 to 19/32 (−0.34 [−0.50, −0.19]).
  - **Discovery.** A scripted search read no published weights, directions, columns, or evaluation items, but took p50c's form: `project_out` at the output of `mlp.down`, 64 columns, and alpha 1 or 6. It chose layer 9 and found no repair: its better candidate's intervals included zero. Its layer score ranked p50c's layers only 10th and 17th of 24, although the contrast there is aligned with p50c's direction (cosines +0.35 and −0.44, to be read against arbitrary splits of the answers, as in Section 5).

## 7. Documentation & Roadmap

| Milestone | Status |
| --- | --- |
| **M1:** contracts, managed experiments, MCP, an agent repairing a small model | Done |
| **M2 Q1:** open-weight models load, replay published edits, and evaluate | Runs and report done; the exit waits for the author's reading of the Llama condition ([docs/m2-q1-report.md](docs/m2-q1-report.md), Section 9) |
| **M2 Q2:** agent-usable tools for language models (roles, edits, analyses, probes, generation passes, budgets, MCP) | Done ([docs/m2-q2-report.md](docs/m2-q2-report.md)) |
| **M2 Q3:** preregistered agent trials on Qwen3.5-0.8B | Paused until after the open-source release |

Documents:

- [docs/models.md](docs/models.md): supported models, support levels, importing, roles and protections, family descriptors, and verified models.
- [docs/extensions.md](docs/extensions.md): the extension contracts, the edit operators and analyses, vector references, direct Python use (with a small in-memory example), and exact replay versus recipe re-execution.
- [docs/protocol.md](docs/protocol.md): the MCP tools, resources, errors, and limits.
- [docs/experiments.md](docs/experiments.md): running managed experiments, and what each trial records and exports.
- Reports: [docs/m1-agent-trials.md](docs/m1-agent-trials.md), [docs/m2-q1-report.md](docs/m2-q1-report.md), and [docs/m2-q2-report.md](docs/m2-q2-report.md).
- Plans: [M1 and the roadmap](agentic-weight-space-engineering-plan.md) and [M2](agentic-weight-space-engineering-m2-plan.md).

Tests run offline, on tiny models built from configuration classes, in about four minutes:

```bash
uv run --no-sync pytest -q
```

Known limits:

- runs on Apple's MPS or the CPU only (no CUDA yet), and has been tested only on macOS with Apple silicon;
- local generation is slow: 37–40 minutes for 32 problems at 8,192 tokens per model on an M4 Pro;
- an agent waits on such a job by polling `wse_get_job` for at most 10 seconds at a time, and hundreds of polls outgrow a model's context window, so long jobs need another way to wait before LLM agents run them;
- whole managed experiments export and replay only for M1's small models, while patches of open-weight models replay exactly;
- the evidence so far comes from small samples and one synthetic M1 task.
