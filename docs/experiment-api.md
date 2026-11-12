# The Experiment API

`Experiment` gives one model to an agent, with a goal and a fixed way to measure it. The agent investigates the model and proposes weight edits. WSE evaluates candidates on the same development problems, selects by a rule stated before the run, and freezes the choice. Only then does it test the frozen choice against the original model, on test problems registered at that moment.

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

This is the README's example, [examples/experiment_quickstart.py](../examples/experiment_quickstart.py). With Codex and `gpt-5.6-luna`, this configuration ran three times on 2026-11-12 ([results](results.md#6-a-live-llm-agent-edits-a-language-model-through-experiment)): each run's choice improved the 16 development problems and did not improve the 16 test problems.

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

`Experiment` also takes `profile` (the sampling profile; by default the model's own, frozen from its `generation_config.json`), `problems` and `subset_seed` (a subset of the development file), `samples` (answers per problem, default 1), and `seed` (the evaluation seed, default 0). `run` also takes `max_turns` and `wall_clock_hours`, which bound the agent (defaults 400 turns and 12 hours, its waits for evaluations included), `device`, the device of WSE's evaluation worker (MPS when available), and `out_dir`, the directory of the run's logs.

`result.summary()` reports the change in behavior, the candidates, where each changed the weights, and why the choice was made, followed by the audit details. `result.model` is the frozen choice, a `Model`; it is the original when no candidate qualified. `result.test(path)` evaluates only the frozen choice and the original, and `report.compare()` prints each metric for both models with their difference: for accuracy, the paired 95% interval of the difference; for the other metrics, each model's own 95% interval.

## Install

```bash
pip install "wse[models-hf]"
npm install -g @openai/codex
```

WSE needs Python 3.11 or later; language models need the `models-hf` extra. The agent is OpenAI's [Codex CLI](https://github.com/openai/codex), release 0.167 or later. WSE runs the binary that the environment variable `WSE_CODEX_PATH` names, else the `codex` on your `PATH`, else the newest one bundled with OpenAI's VS Code extension. An npm install puts a Node launcher script on `PATH`; WSE runs the native binary that the launcher would start, from the same install. On macOS, WSE first checks that the binary's code signature verifies and names OpenAI (Team ID `2DC432GLL2`); it then reads `codex --version` and skips a release older than 0.167.0. A binary that `WSE_CODEX_PATH` names and that fails a check is an error. WSE has been tested with codex-cli 0.167.0-alpha.4.1 (OpenAI's VS Code extension) and 0.167.2 (npm).

## Agent credentials

The agent's model runs at your provider. WSE reads two variables from its own environment: `OPENAI_API_KEY`, your API key, and `OPENAI_BASE_URL`, the provider's endpoint, the base URL of its API (`https://api.openai.com/v1` for OpenAI's own). WSE assumes no endpoint, so the key goes only where you say:

```bash
export OPENAI_API_KEY=your-api-key
export OPENAI_BASE_URL=https://api.openai.com/v1
```

For Azure OpenAI, set `OPENAI_BASE_URL` to your resource's endpoint, `https://YOUR-RESOURCE.services.ai.azure.com/openai/v1`, and put your deployment's name after `codex:`. In a source checkout, you can instead copy [.env.example](../.env.example) to `.env`, which git ignores, fill it in, and load it for one command with `uv run --env-file .env`.

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

**The README's example.** Download the problems into a directory, save the code above there, put your model's name in place of `YOUR_AGENT_MODEL`, and run it:

```bash
curl -fLO "https://raw.githubusercontent.com/xuguang-ai/weight-space-engineering/main/examples/data/math_{dev,test}.jsonl"
python fix_loops.py
```

It runs Qwen3-0.6B on the 16 development problems at 1,024 tokens, with up to 10 evaluations, and tests on the 16 test problems. One evaluation takes about a minute and a half on an M4 Pro, and a whole run took 15 to 20 minutes there, the model's download included.

**Qwen3.5-0.8B's reasoning loops, not yet run with an agent.** [examples/experiment_qwen_loops.py](../examples/experiment_qwen_loops.py) sets up the full-size case: the 32 development problems of the recorded diagnosis, on which the original answered 2 correctly and ran 30 to the limit, at 8,192 tokens with the article's sampling profile, `qwen35think@1`. From a source checkout:

```bash
caffeinate -is uv run --extra models-hf --env-file .env \
  python examples/experiment_qwen_loops.py \
  --agent codex:YOUR_AGENT_MODEL
```

One evaluation of the 32 problems took 37 to 40 minutes on an M4 Pro in the recorded runs, so 10 evaluations take 6.2 to 6.6 hours. The agent's time limit includes its waits for evaluations; `run`'s default, 12 hours, leaves room for ten. `caffeinate -is` keeps the Mac awake. The script takes `--model`, `--repo`, and `--revision` for another model or a local snapshot, `--dev` and `--test` for other problems, and `--workspace` and `--export` for where results go.

## What happens during a run

0. **Checks.** Before it loads the model or registers anything, `run` checks that the agent can start (the Codex binary and its signature, `OPENAI_BASE_URL` and `OPENAI_API_KEY`) and, with `sandbox="python"`, that macOS's `sandbox-exec` works, so a missing prerequisite costs no evaluation.
1. **Baseline.** WSE evaluates the original model on the development problems. This evaluation counts toward `max_evaluations`. When another experiment in the workspace already evaluated the same model on the same development problems, with the same evaluator, settings, sample count, seed, batch size, and device, and without errors, WSE links that report instead of evaluating again; the link records which report it repeats and still counts as one evaluation.
2. **Agent.** WSE starts Codex (`codex exec --json`) in the run's working directory, `<workspace>/sandboxes/<run_id>`, with a Codex home of the run's own. Its `config.toml` sends the model's requests through WSE's local proxy, sets the permission profile of the agent's commands ([below](#what-the-agent-can-reach)), and connects the run's WSE tools over MCP (`wse serve`, bound to this run and limited to the tools its policy allows). WSE's skill, `skills/wse/SKILL.md` in that home, describes the tools, analysis in Python, the two ways to submit a candidate, evaluation, the selection rule, and how to stop; the prompt gives the task, the rule, the baseline, the budget, and the run's IDs. The agent inspects the weights, probes activations, runs its own commands and Python, and submits candidates in two ways: as edits by WSE's registered operators, or as tensors its code computed and saved to a `.safetensors` file under `outputs/` or `work/`, which `wse_submit_candidate(..., path="outputs/candidate.safetensors")` turns into a candidate after checking every tensor's name, shape, dtype, values, and protection. It asks WSE to evaluate a candidate with `wse_evaluate`, which returns when the evaluation has finished, so a long evaluation costs one tool call instead of repeated polling: Codex lets one WSE call run as long as the policy's longest operation.
   The agent stops by replying with a final message whose first line starts with `WSE-STOP:`. When Codex ends its turn without that line while budget remains, or after an error that a new turn can fix, WSE continues the same session with `codex exec resume`, at most twice. The run also ends when a stop condition is met (no evaluation left, for example), at its time limit (`wall_clock_hours`, 12 by default), or at its turn limit (`max_turns`, 400 model requests by default); WSE then stops Codex with SIGINT, then SIGTERM and SIGKILL, and ends the processes it can find that the agent's commands started, also those that left Codex's process tree: processes of yours started during the run that carry the run's marker in their environment, work in its sandbox directory, or hold a file there open for writing. A process that does none of these, for example one that clears its environment and leaves the directory, can outlive the run; it keeps Codex's sandbox (no network, writes only to `work/`, `outputs/`, and `tmp/`). WSE keeps each segment's event log, Codex's session record, and the proxy's request log, and records the agent's tokens (input, cached input, output, and reasoning) per model request.
3. **Selection and freeze.** The agent does not choose: when it stops, when no development evaluation remains (a candidate made after the last one cannot be chosen), or when its budget or time runs out, WSE applies the selection rule to its own development evaluations and freezes the choice. If anything fails after the experiment exists, WSE still freezes it, and the error names the experiment, which `ExperimentResult.open(experiment_id, workspace=...)` reopens.
4. **Test, on request.** `result.test(path)` evaluates the frozen choice and the original on the test problems, on the run's `device` (or `result.test(path, device=...)`).

**The budget.** Besides `max_evaluations`, a run's policy (`wse.experiment.experiment_policy`) allows 25 activation-probe calls of at most 64 texts of up to 1,024 tokens, 50 tensor analyses, 100 tensor reads, 50 patch previews, 20 candidate commits, and 24 hours of WSE-metered computation, at most 4 hours per operation; with `sandbox="python"`, WSE's own Python tool also allows 400 calls of at most 10 minutes each and 8 hours in all. Tensor files submitted as candidates are checked against the preview and commit budget before WSE copies them, and their total per experiment is capped at 32 GiB. Every tool result reports what remains.

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

- **Registered at the test.** `result.test(path)` registers the test problems, after the choice is frozen. The workspace stores them outside the agent's sandbox ([below](#what-the-agent-can-reach)). When an earlier run of the same workspace already registered the same file, it was on disk during this run, and `report.compare()` says so in its last line (`report.previously_registered` lists those registrations); use a new workspace, or a test set that no earlier run registered, to rule that out.
- **No overlap.** A test file that shares a problem with any development problems registered in the workspace is refused and not stored, and so is later development data that shares a problem with a registered test file. Problems are compared after Unicode NFKC normalization, lowercasing, and collapsing whitespace.
- **Only after the agent ran.** A run whose agent never started (`result.agent_started` is false; `result.ending` gives the reason) is not tested: its test would compare the original with itself and use up the test set.
- **Two models only.** The test evaluates the frozen choice and the original, as the control, on the same problems, prompts, sampling settings, and seeds. For accuracy it reports the paired 95% interval of the difference; for the limit-hit rate and the mean generated tokens, each model's own 95% interval.
- **No second search.** A test cannot restart the search, add a candidate, or change the choice.
- **One test set per run.** Calling `result.test` again with the same file returns the stored report instead of evaluating again; other test data for the same run is refused, and needs a new run.

## What the agent can reach

The agent's commands run under Codex's own sandbox, configured by a permission profile that WSE writes into the run's Codex config (`[permissions.wse]`). The profile lists what commands may read and write; everything else is denied:

| | Access |
| --- | --- |
| `inputs/` in the run's working directory: the base checkpoint's weights (`weights/base.safetensors`), and the reports and generated outputs of finished development evaluations, refreshed as evaluations finish (with `sandbox="python"`) | Read only |
| `work/`, `outputs/`, and `tmp/` in the working directory | Read and write |
| The Python interpreter, and NumPy, PyTorch, and safetensors with the packages they depend on | Read only; `python` imports nothing else, WSE included |
| The repository, the workspace's database and stored data (test data included), other runs' directories, your home directory, `/tmp`, and Codex's own configuration and sessions | None |
| The network | None for the agent's commands; Codex reaches your model provider only through WSE's local proxy |

The agent reaches WSE only through the run's MCP tools, which act on this experiment's model and candidates; the MCP server runs outside the agent's sandbox with an environment of its own. Commands get an environment built from scratch, without keys or the proxy's token, in a non-login shell. Codex loads no `AGENTS.md`, configuration, or skills from the directories around the run or from your `~/.codex`.

These rules were measured with codex-cli 0.167.0-alpha.4.1 on macOS, with the run's profile under `codex sandbox` and with commands that Codex ran in a run: attempts to read the repository's README, source, and test problems, WSE's installed package metadata, the workspace's database and evaluator data, a `.env` file, Codex's config and sessions, `~/.codex`, `~/.cache/huggingface`, `~/.ssh`, the home directory, other runs' directories, `/tmp` and `/private/tmp`, and to reach the internet or a port on this machine, were all denied, while reading `inputs/` and importing the analysis libraries worked ([tests/test_codex_isolation.py](../tests/test_codex_isolation.py)). With `sandbox=None`, `inputs/` stays empty: the agent still has a shell, but no model data to run it on.

## What is verified, and what is not

**WSE verifies:**

- every candidate is an exact patch of its parent: replay rebuilds its weights byte for byte, checked by hashes;
- every evaluation runs WSE's evaluator with the fixed settings, and WSE counts its evaluations against `max_evaluations`;
- the choice follows from WSE's own evaluations by the stated rule;
- the test runs after the freeze, on data registered then.

**WSE does not verify:**

- **What the agent's code computes.** The agent runs its own commands in Codex's shell and, with `sandbox="python"`, can read the model's weights and compute anything from them, forward passes included. WSE meters its own calls, not that work, so every Codex run is labeled `unverified`, and its records say why. The budget bounds WSE's operations (evaluations, probes, previews, commits); it does not bound the agent's own computation. WSE sets no memory, disk, or CPU limit on the agent's shell commands: the run's time and turn limits bound how long the agent works, and a process that outlives the run (above) is bounded by neither. `wse_run_python` has limits of its own: wall-clock time and output per call, CPU time over its kernel's life, memory (checked only while a call runs), and 4 GiB per file.
- **That an improvement generalizes.** The test checks one more set of problems of the same kind; a few dozen problems give wide intervals, and other tasks are not tested.
- **The agent's explanations.** What the agent says about why an edit works is its own claim; WSE checks the weights and the metrics, not the reasoning.

## Current limits

- **Three live runs** (above), on a small model with 16 development and 16 test problems: too few to show that an edit generalizes.
- **macOS only.** The agent's sandbox has been configured and tested on macOS only.
- **Two Codex releases tested,** 0.167.0-alpha.4.1 and 0.167.2. WSE writes its Codex config with `--strict-config`, using keys of codex-cli 0.167 (permission profiles among them); another version may reject a key or treat it differently, and WSE skips releases older than 0.167.0. On macOS, WSE runs a Codex binary only after its code signature names OpenAI (Team ID `2DC432GLL2`). It runs the binary that `WSE_CODEX_PATH` names, else the `codex` on your `PATH`, else the newest one bundled with OpenAI's VS Code extension.
- **The turn limit can be passed by one request.** WSE counts model requests from Codex's session record, so a fast loop can make one request more than `max_turns` before it is stopped.
- **Apple MPS or the CPU.** Generation runs on MPS or the CPU; CUDA support is planned ([handover](../dev/handover-cuda.md)). On the CPU, a decoding step of Qwen3.5 takes about 5 to 10 seconds, about 170 times slower than on MPS ([models.md](models.md#known-limits)).
- **Slow evaluations.** On an M4 Pro, 32 problems at 8,192 tokens took 37 to 40 minutes for Qwen3.5-0.8B.
- **One built-in evaluator,** `math`.
- **Qwen3.5 exports its language model only.** WSE imports Qwen3.5-0.8B's language model and leaves out its vision encoder and its multi-token-prediction tensors, so an exported Qwen3.5 model has no vision encoder ([models.md](models.md#exporting-a-model)).
- **The limit-hit rate is not a repetition rate.** It is the share of answers cut off at the token limit. An answer can repeat itself and still finish, or reach the limit while still making progress. In the recorded Qwen3.5-0.8B runs, 24 of the base model's 30 cut-off answers ended in a loop, while none of the 21 answers that the edited model still ran to the limit did: they kept re-checking the problem in new words ([results](results.md#step-4-evaluate)). Read the answers to see which kind a change removed.
- **Small problem sets.** With 16 problems, one answer moves accuracy by 6.25 points.
