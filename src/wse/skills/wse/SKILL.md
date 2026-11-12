---
name: wse
description: How to run a managed weight-space engineering (WSE) experiment, that is, find and test a direct edit of a model's weights that meets a task's objective. Read it before the first tool call of any run whose prompt names a WSE experiment; it covers the WSE tools, analysis in Python, the two ways to create a candidate, evaluation, how the final candidate is chosen, and when to stop.
---

# Weight-space engineering runs

You work on one experiment of one run. The prompt gives the run and experiment IDs, the task and its
objective, and the budget; this skill gives the method. Every WSE tool result reports the budget that
remains, and WSE refuses an operation the budget cannot cover.

WSE records and verifies; your own code explores. Only WSE tool calls change the experiment: they create
candidates, evaluate them, and charge the budget. Your shell commands and Python run in a sandbox and change
nothing in WSE.

## Your working directory

The shell starts in this run's sandbox directory:

- `inputs/` (read only): what WSE exported for this run.
  - `inputs/weights/base.safetensors`: the starting checkpoint's tensors that you may read, in their stored
    dtype, under their full checkpoint names.
  - `inputs/passes/<report_id>.jsonl`: one line per output of each development generation pass of this run
    (problem, item_id, sample, correct, answer, finished, limit_hit, generated_tokens, text).
  - `inputs/probes/<job_id>.npz` and `.json`: the vectors and summary of each finished probe.
  - `inputs/manifest.json`: every input with its size and SHA-256.
  New pass outputs and probe results appear in `inputs/` after their jobs finish.
- `work/`: your scripts, notes, and intermediate files; kept for the whole run.
- `outputs/`: files you hand to WSE, such as tensor files for `wse_submit_candidate`.
- `tmp/`: scratch space.

Everything else is unreadable, including WSE's database, the evaluation data, and the network. Do not try to
work around the sandbox: a denied read or write is policy, not a fault to fix.

## Tools and when to use them

- `wse_get_experiment`: call it first. It returns the task, the objective and its acceptance gates, the
  budget, the permitted analyses and edit operations, the development suites, and the checkpoints.
- `wse_inspect_model`, `wse_read_tensor`, `wse_analyze`: metered views of a checkpoint (manifest and
  statistics, bounded slices, registered analyses). Prefer your own Python on `inputs/weights` for anything
  these do not cover; it costs no WSE budget.
- `wse_probe` (when listed): activations of a checkpoint on texts, as a job.
- `wse_preview_patch`, then `wse_apply_patch`: create a candidate from registered edit operations.
- `wse_submit_candidate` (when listed): create a candidate from tensors you computed.
- `wse_evaluate` (when listed): evaluate a checkpoint on the development suite in one call that waits for the
  result.
- `wse_submit_evaluation`, `wse_get_job`, `wse_get_report`, `wse_compare`: queue an evaluation, wait for it,
  read whole reports, and pair two reports.
- `wse_select_checkpoint` (only when listed): select a checkpoint yourself.

Every mutating call needs a new `idempotency_key`; repeating a key with the same input returns the original
result without charging again, so reuse a key only to retry the same request.

## Analysis in Python

Write scripts in `work/` and run them with `python work/<name>.py`. The shell's `python` has NumPy, PyTorch
(CPU), and safetensors, and nothing can be installed.

```python
from safetensors import safe_open
from safetensors.torch import load_file

weights = load_file("inputs/weights/base.safetensors")  # dict: full tensor name -> torch.Tensor
with safe_open("inputs/weights/base.safetensors", framework="pt") as handle:  # one tensor at a time
    names = list(handle.keys())
    tensor = handle.get_tensor(names[0])
```

Use Python for statistics, decompositions, comparisons between tensors, and for reading pass texts and probe
vectors. Model evaluations go through WSE only: numbers your code computes about model behavior are evidence
for a hypothesis, not evaluations, and do not count toward the objective.

Keep each command short. For a computation that takes minutes, run it as one command that writes its result to
`work/` and wait for that command once; do not poll it in many short steps.

## Creating a candidate

Choose the smallest edit that your evidence supports, and state that evidence in the hypothesis (one or two
sentences of observable facts).

1. Registered operators: `wse_preview_patch` with `base_checkpoint_id`, `hypothesis`, and `operations` (each
   names a permitted edit and its parameters; vectors may be inline values or stored vectors named by a
   pointer that WSE returned, `<result_uri>#<JSON pointer>`), then `wse_apply_patch` with the preview's
   `patch_id`.
2. Tensors you computed: build the new tensors in Python and save them with
   `safetensors.torch.save_file({name: tensor, ...}, "outputs/<file>.safetensors")`, each under its full
   checkpoint name, in its stored dtype and full shape. Then call `wse_submit_candidate` with
   `parent_checkpoint_id`, a `hypothesis`, and `path` set to the file relative to the working directory, for
   example `outputs/candidate.safetensors` (a file under `outputs/` or `work/`). WSE checks names, shapes,
   dtypes, finite values, and protected tensors, and records the exact patch.

Each candidate costs a candidate commit (and a preview). A candidate is not selected by creating it.

## Evaluation

Evaluate a candidate on the development suite before you trust it, with the same evaluation settings as the
starting model's evaluation so the two can be compared.

- When `wse_evaluate` is listed, use it: one call waits until the evaluation finishes and returns the
  development report and its comparison with the starting model. An evaluation of a language model can take
  many minutes; the call is allowed to take that long.
- Otherwise submit with `wse_submit_evaluation` and wait with `wse_get_job`, always passing the largest
  `wait_seconds` it allows. Never poll in quick succession and never wait with shell `sleep` loops: each
  poll adds to your context and gains nothing.

Read whole reports with `wse_get_report` and compare two reports with `wse_compare` when you need per-example
detail.

## How the final candidate is chosen

When the run ends, WSE freezes one checkpoint for a separate final evaluation that you cannot see. If
`wse_select_checkpoint` is not among your tools, the task has a selection rule and WSE applies it: among the
candidates you evaluated on the full development suite, it takes the one that best meets the task's objective
(for example, the lowest value of one metric) while every other metric the task names stays at least at the
starting model's point estimate. If no candidate qualifies, the starting model is kept. So evaluate every
candidate you want considered on the full development suite. If `wse_select_checkpoint` is listed, select the
checkpoint yourself, passing the currently selected ID as `expected_selected_id`.

## When to stop

WSE ends the run by itself when the budget cannot cover the next operation, when a stop condition the prompt
names is met, or at the run's time or turn limits; you are then stopped.

Stop yourself when you have no further idea worth the remaining budget, or when the budget left cannot cover
another candidate and its evaluation. To stop, reply with a final message whose first line begins with
`WSE-STOP:` followed by the reason and the evaluated candidates you consider best. A reply without that line
while budget remains means you will be asked to continue.

Keep messages and tool arguments to brief hypotheses and observable evidence; do not write out private
reasoning.
