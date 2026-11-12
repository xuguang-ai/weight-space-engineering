# The Experiment API

`Experiment` gives one model to an agent, with a goal and a fixed way to measure it. The agent investigates the model and proposes weight edits. WSE evaluates candidates on the same development problems, selects by a rule stated before the run, and freezes the choice. Only then does it test the frozen choice against the original model, on test problems registered at that moment.

> **Status: not yet run with an agent.** `Experiment` and its Codex agent runner are new in this version, and no agent has completed a run with them yet. The `Model` API they build on has been run ([models.md](models.md#the-model-api)).

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
print(result.summary())

selected = result.model
report = result.test("examples/data/math_test.jsonl")
print(report.compare())
result.export("artifacts/fix-loops")
```

| Argument | What it sets |
| --- | --- |
| `model` | The original model: a Hugging Face repository ID or a local snapshot, imported as `Model.from_pretrained` does (`repo=` and `revision=` name a snapshot's source and fix the commit), or a `Model`, whose checkpoint and workspace the experiment uses; then leave out `workspace=`. |
| `goal` | The agent's research brief, in your words. It guides the agent; it decides nothing. |
| `dev` | The development problems. Every evaluation runs a model on all of them. |
| `evaluator` | How problems are prompted and answers graded: `"math"` ([Data](#data)). |
| `minimize` | The selection metric, which WSE computes from its own development evaluations (or `maximize` for a metric where higher is better). |
| `preserve` | Metrics a candidate must not lose against the original ([The selection rule](#the-selection-rule)). |
| `max_new_tokens` | The token limit of every answer in every evaluation. |
| `workspace` | The directory where WSE stores the model, the candidates, the evaluations, and the records. |
| `agent` | `codex:` followed by your model's name, or your Azure OpenAI deployment's name. |
| `sandbox` | `"python"`: the agent may write and run Python in its working directory ([What the agent can reach](#what-the-agent-can-reach)). |
| `max_evaluations` | How many development evaluations the run may make, the original's included. |

`Experiment` also takes `profile` (the sampling profile; by default the model's own, frozen from its `generation_config.json`), `problems` and `subset_seed` (a subset of the development file), `samples` (answers per problem, default 1), and `seed` (the evaluation seed, default 0). `run` also takes `max_turns` and `wall_clock_hours`, which bound the agent (defaults 400 turns and 6 hours), `device`, the device of WSE's evaluation worker (MPS when available), and `out_dir`, the directory of the run's logs.

`result.summary()` reports the change in behavior, the candidates, where each changed the weights, and why the choice was made, followed by the audit details. `result.model` is the frozen choice, a `Model`; it is the original when no candidate qualified. `result.test(path)` evaluates only the frozen choice and the original, and `report.compare()` prints each metric for both models with their difference: for accuracy, the paired 95% interval of the difference; for the other metrics, each model's own 95% interval.

## Install

The examples run from a source checkout, with Python 3.11 or later and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/xuguang-ai/weight-space-engineering.git
cd weight-space-engineering
uv sync
```

Language models need the `models-hf` extra, which `uv run --extra models-hf` installs the first time a command uses it.

The agent is the [Codex CLI](https://github.com/openai/codex), which you install separately. WSE runs the `codex` on your `PATH`, or the executable that the environment variable `WSE_CODEX_PATH` names. Check that `codex --version` runs.

## Agent credentials

The agent's model runs at your provider. Copy the template, [.env.example](../.env.example), to `.env`, which git ignores, and fill in your key and endpoint:

```bash
cp .env.example .env
```

```text
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
```

For Azure OpenAI, set `OPENAI_BASE_URL` to your resource's endpoint, `https://YOUR-RESOURCE.services.ai.azure.com/openai/v1`, and put your deployment's name after `codex:`. `uv run --env-file .env` loads the file for one command only.

The key stays outside the agent: Codex reaches your provider through a proxy on `127.0.0.1` that WSE runs for the experiment and that adds the key to each request (`wse.model_proxy`). The agent's process holds only a random token for that proxy, valid while the run lasts; neither Codex nor the code the agent writes holds the key, and the proxy's log records no key, token, or request body.

## Data

A JSON Lines file, one problem per line, with the problem's text and its final answer. The first two lines of [examples/data/math_dev.jsonl](../examples/data/math_dev.jsonl):

```json
{"problem": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "answer": "72"}
{"problem": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "answer": "10"}
```

The `math` evaluator also accepts:

- `solution` in place of `answer`, a worked solution whose last `\boxed{}` holds the answer (the MATH dataset's format);
- an `id` (or `unique_id`) for each problem; without one, WSE numbers the problems in file order;
- GSM8K's own rows, `question` and an `answer` that ends in `#### <number>`; the grader then compares that number.

**How answers are graded.** Each problem, followed by the instruction "Please reason step by step, and put your final answer within \boxed{}.", is sent through the model's chat template. The grader (`math_boxed@1`) reads the last `\boxed{}` after the model's thinking block and compares it with the reference using [math-verify](https://github.com/huggingface/Math-Verify), so `\boxed{1,080}` matches `1080`. An answer without a box is wrong. An answer cut off at the limit is graded like any other: if its unfinished thinking already holds a box, the last one counts.

**The sample files** in [examples/data](../examples/data) come from GSM8K (MIT License): 16 development problems from its training set, 16 test problems from its test set, and the 32 training problems of the recorded Qwen3.5-0.8B diagnosis in [results.md](results.md#3-reconstructing-p50c-in-four-steps). Their [README](../examples/data/README.md) gives the source, the license, and each file's rows.

## Run

Two example scripts write out every argument. Neither has been run with an agent yet.

**A small model, in minutes per evaluation.** [examples/experiment_quickstart.py](../examples/experiment_quickstart.py) runs Qwen3-0.6B on the 16 development problems at 1,024 tokens, with up to 10 evaluations, and tests on the 16 test problems:

```bash
uv run --extra models-hf --env-file .env \
  python examples/experiment_quickstart.py \
  --agent codex:YOUR_AGENT_MODEL
```

One evaluation should take about a minute and a half on an M4 Pro: in the model quickstart, Qwen3-0.6B answered a batch of 8 of these problems at 1,024 tokens in 46 to 49 seconds on MPS ([models.md](models.md#the-quickstart)).

**Qwen3.5-0.8B's reasoning loops.** [examples/experiment_qwen_loops.py](../examples/experiment_qwen_loops.py) runs the full-size case: the 32 development problems of the recorded diagnosis, on which the original answered 2 correctly and ran 30 to the limit, at 8,192 tokens with the article's sampling profile, `qwen35think@1`:

```bash
caffeinate -is uv run --extra models-hf --env-file .env \
  python examples/experiment_qwen_loops.py \
  --agent codex:YOUR_AGENT_MODEL
```

One evaluation of the 32 problems took 37 to 40 minutes on an M4 Pro in the recorded runs, so 10 evaluations take 6.2 to 6.6 hours. The agent's time limit includes its waits for evaluations, so the script raises it from `run`'s default of 6 hours to 10 (`--wall-clock-hours`). `caffeinate -is` keeps the Mac awake.

Both scripts take `--model`, `--repo`, and `--revision` for another model or a local snapshot, `--dev` and `--test` for other problems, and `--workspace` and `--export` for where results go.

## What happens during a run

1. **Baseline.** WSE evaluates the original model on the development problems. This evaluation counts toward `max_evaluations`.
2. **Agent.** WSE starts Codex (`codex exec`) in the experiment's working directory, with WSE's skill file (`SKILL.md`), which gives it the goal, the run, and its budget, and with the run's WSE tools connected over MCP. The agent inspects the weights, probes activations, writes and runs its own Python, and submits candidates in two ways: as edits by WSE's registered operators, or as tensors its code computed and saved to a `.safetensors` file under `outputs/` or `work/` of its working directory, which `wse_submit_candidate(..., path="outputs/candidate.safetensors")` turns into a candidate after checking every tensor's name, shape, dtype, values, and protection. It asks WSE to evaluate a candidate with `wse_evaluate`, which returns when the evaluation has finished, so a long evaluation costs one tool call instead of repeated polling. WSE records the Codex session, which `codex exec resume` continues.
3. **Selection and freeze.** The agent does not choose: when it stops, or its budget or time runs out, WSE applies the selection rule to its own development evaluations and freezes the choice.
4. **Test, on request.** `result.test(path)` evaluates the frozen choice and the original on the test problems.

**The budget.** Besides `max_evaluations`, a run's policy (`wse.experiment.experiment_policy`) allows 25 activation-probe calls of at most 64 texts of up to 1,024 tokens, 50 tensor analyses, 100 tensor reads, 50 patch previews, 20 candidate commits, and 24 hours of WSE-metered computation, at most 4 hours per operation; with `sandbox="python"`, WSE's own Python tool also allows 400 calls of at most 10 minutes each and 8 hours in all. Every tool result reports what remains.

## Where results go

- **The workspace** (`workspace=`) holds the original model, every candidate as a checkpoint with its exact patch, every development evaluation with each answer's text and grade, the agent's record, and the audit log of every operation. The run's logs go to `<workspace>/experiments/<run_id>/`, or to `out_dir`.
- **`result.summary()`** prints the goal and the rule; the original's and the choice's development metrics; the choice and why; every evaluated candidate with its metrics, the tensors it changed, whether the rule kept it and why, and the agent's hypothesis; the agent's final message; and then the audit: how the agent's run ended, the evaluations used, the `unverified` label and its reason, WSE-metered computation, and the agent's tokens.
- **`result.model`** is the frozen choice, a `Model`; `result.model.export(path)` writes it as a Hugging Face model directory ([models.md](models.md#exporting-a-model)).
- **`result.test(path)`** returns the test report; `report.compare()` prints the comparison.
- **`result.export(path)`** writes into a new directory `record/`, the experiment's export (every candidate's exact patch, the reports, the audit events, the agent's trace and code records, and the original model), which `wse replay` rebuilds byte for byte in another workspace; `model/`, the selected model as a Hugging Face directory; and `summary.txt`, the summary.

## The selection rule

The development evaluation is fixed before the agent starts: the same problems, prompts, sampling settings, seeds, and token limit for every model. The agent cannot change how answers are graded.

`max_evaluations` counts evaluations of one model on all the development problems, and it includes the original's evaluation, which comes first.

When the run ends, WSE takes the first full development evaluation of each checkpoint. A candidate qualifies when, on the development problems,

- its value of the `minimize` metric is below the original's, and
- its value of every `preserve` metric is no worse than the original's.

Of the qualifying candidates, the one with the lowest value of the `minimize` metric is the choice. Ties go to the better `preserve` metrics, in the order listed, then to the candidate that changed fewer weights, then to the earlier one. When none qualifies, the original is kept, and `result.model` is the original.

The values are point estimates. `preserve=["accuracy"]` means that the candidate answered at least as many development problems correctly as the original; it is not a test of statistical significance, and it guarantees nothing on other problems or tasks. The development values of the choice are also optimistic, because it was chosen for them: the test is the estimate to report.

## Test isolation

- **Registered after the freeze.** The test problems enter the workspace only when `result.test(path)` is called, after the choice is frozen. During the run, the agent's file access does not include the test data ([below](#what-the-agent-can-reach)).
- **No overlap.** A test file that shares a problem with any development problems registered in the workspace is refused, and so is later development data that shares a problem with a registered test file. Problems are compared after Unicode NFKC normalization, lowercasing, and collapsing whitespace.
- **Two models only.** The test evaluates the frozen choice and the original, as the control, on the same problems, prompts, sampling settings, and seeds. For accuracy it reports the paired 95% interval of the difference; for the limit-hit rate and the mean generated tokens, each model's own 95% interval.
- **No second search.** A test cannot restart the search, add a candidate, or change the choice.
- **One test set per run.** Calling `result.test` again with the same file returns the stored report instead of evaluating again; other test data for the same run is refused, and needs a new run.

## What the agent can reach

The agent works in a Codex sandbox that WSE configures for the run:

| | Access |
| --- | --- |
| The experiment's working directory | Read and write |
| The model's files and the Python analysis libraries | Read only |
| The test data, WSE's database, and your keys | None |
| Your model provider | Through the local proxy only |

It reaches WSE only through the run's MCP tools, which act on this experiment's model and candidates.

## What is verified, and what is not

**WSE verifies:**

- every candidate is an exact patch of its parent: replay rebuilds its weights byte for byte, checked by hashes;
- every evaluation runs WSE's evaluator with the fixed settings, and WSE counts its evaluations against `max_evaluations`;
- the choice follows from WSE's own evaluations by the stated rule;
- the test runs after the freeze, on data registered then.

**WSE does not verify:**

- **What the agent's code computes.** The agent can read the model's weights and run its own computations on them, forward passes included. WSE meters its own calls, not that work, so every run with `sandbox="python"` is labeled `unverified`, and its records say why.
- **That an improvement generalizes.** The test checks one more set of problems of the same kind; a few dozen problems give wide intervals, and other tasks are not tested.
- **The agent's explanations.** What the agent says about why an edit works is its own claim; WSE checks the weights and the metrics, not the reasoning.

## Current limits

- **Not yet run with an agent** (above).
- **macOS only.** The agent's sandbox has been configured and tested on macOS only.
- **Apple MPS or the CPU.** Generation runs on MPS or the CPU; CUDA support is planned ([handover](../dev/handover-cuda.md)). On the CPU, a decoding step of Qwen3.5 takes about 5 to 10 seconds, about 170 times slower than on MPS ([models.md](models.md#known-limits)).
- **Slow evaluations.** On an M4 Pro, 32 problems at 8,192 tokens took 37 to 40 minutes for Qwen3.5-0.8B.
- **One built-in evaluator,** `math`.
- **Qwen3.5 exports its language model only.** WSE imports Qwen3.5-0.8B's language model and leaves out its vision encoder and its multi-token-prediction tensors, so an exported Qwen3.5 model has no vision encoder ([models.md](models.md#exporting-a-model)).
- **The limit-hit rate is not a repetition rate.** It is the share of answers cut off at the token limit. An answer can repeat itself and still finish, or reach the limit while still making progress. In the recorded Qwen3.5-0.8B runs, 24 of the base model's 30 cut-off answers ended in a loop, while none of the 21 answers that the edited model still ran to the limit did: they kept re-checking the problem in new words ([results](results.md#step-4-evaluate)). Read the answers to see which kind a change removed.
- **Small problem sets.** With 16 problems, one answer moves accuracy by 6.25 points.
