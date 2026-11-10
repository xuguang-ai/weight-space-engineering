# Weight-space engineering (WSE)

WSE is a Python framework for inspecting, editing, and evaluating the weights
of neural networks directly, without training. Every edit is an exact patch:
it records the values written and the SHA-256 hash of each tensor before and
after, so it replays to byte-identical weights or fails. The same operations
are available as a Python library, as an operator command line (`wse`), and
as an MCP server through which an agent runs a metered, audited experiment.

Milestone M1 (a small MLP task and five live agent trials) is closed.
Milestone M2 adds open-weight language models, with Qwen3.5-0.8B first and
Llama-3.2-1B-Instruct second. Its first stage is built: import, BF16 patches,
exact replay of published edits, and generation evaluation. Agent tools for
these models and the agent trials come next
([M2 plan](agentic-weight-space-engineering-m2-plan.md)).

## Install

WSE uses [uv](https://docs.astral.sh/uv/) and Python 3.11 or later. It has
been tested with Python 3.14 and PyTorch 2.15 on macOS (Apple silicon, CPU
and MPS). From the repository root:

```bash
uv sync --extra models-hf --extra agents-openai
```

The core needs neither extra. `models-hf` adds `transformers` and
`math-verify` for open-weight language models; `agents-openai` adds the
OpenAI Agents SDK for the reference agent. Afterwards, run commands with
`uv run --no-sync ...`: a plain `uv run` re-syncs the environment and drops
the extras.

## Quick start: edit a small model in memory

The core works in memory, with no workspace, database, or download. This
example runs in about a second on CPU:

```python
import torch

from wse import apply_operator, evaluate_candidate, load_model, replay_patch

tensors = {"layers.0.weight": torch.randn(16, 8), "layers.0.bias": torch.zeros(16),
           "layers.1.weight": torch.randn(2, 16), "layers.1.bias": torch.zeros(2)}
model = load_model(tensors, adapter="mlp", config={"layer_sizes": [8, 16, 2], "activation": "relu", "bias": True})

# Halve row 3 of the first weight matrix. The input model is unchanged.
edit = apply_operator(model, "scale_slice", {
    "tensor": "layers.0.weight",
    "selection": [{"start": 3, "stop": 4, "step": 1}, {"start": 0, "stop": 8, "step": 1}],
    "factor": 0.5,
})  # edit.candidate (a new model), edit.patch (the exact patch), edit.diff (per tensor)

# Evaluate the candidate on a small classification suite.
inputs = torch.randn(32, 8)
suite = {"suite_id": "example_dev_v1", "task_id": "example_task_v1", "split": "development",
         "evaluator": "classification@1", "sample_counts": [32], "seeds": [0],
         "data": {"inputs": inputs.tolist(), "labels": (inputs.sum(dim=1) > 0).long().tolist()}}
result = evaluate_candidate(edit.candidate, suite, {"sample_count": 32, "seed": 0})
print(result.metrics["accuracy"], result.management)  # ... unmanaged

# The exact patch replays to byte-identical weights.
assert replay_patch(model, edit.patch).weights_hash == edit.candidate.weights_hash
```

Weights are addressed by tensor name and a `selection` of one
`{start, stop, step}` range per axis. Results made outside a managed
experiment are labeled `unmanaged`.

## Open-weight language models

WSE imports a local Hugging Face snapshot. It reads only allow-listed files
and never unpickles or executes anything from it. Download Qwen3.5-0.8B
first; Task 1 of the [M2 plan](agentic-weight-space-engineering-m2-plan.md)
lists all the downloads, including Llama, the article's edits, and the
datasets:

```bash
M=~/Desktop/wse-models
# huggingface_hub 2.x takes one pattern per --include.
uv run --no-sync hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --include "*.json" --include "*.safetensors" --include "*.jinja" --include "tokenizer*" \
  --include "vocab*" --include "merges*" --local-dir "$M/Qwen3.5-0.8B"
```

### In Python: import, edit, replay, and generate

This example imports the model into a workspace, halves 64 columns of one MLP
projection, checks that the patch replays exactly, and lets both models answer
one problem. Run it with `uv run --no-sync python` from the repository root.

```python
from pathlib import Path

from wse import Engine, apply_operator, replay_patch
from wse.evaluators.generation import generate_samples, load_text_model
from wse.evaluators.graders import MathBoxedGrader
from wse.evaluators.sampling import resolve_profile, with_overrides
from wse.schemas import TextItem

M = Path("~/Desktop/wse-models").expanduser()

with Engine(M / "workspaces" / "demo") as engine:
    # Import the snapshot. Importing the same files again returns the same checkpoint.
    base_id = engine.import_hf_snapshot(M / "Qwen3.5-0.8B", repo="Qwen/Qwen3.5-0.8B",
                                        revision="2fc06364715b967f1860aea9cf38778875588b17").checkpoint_id
    base = engine.open_checkpoint(base_id)  # an in-memory model whose weights were hash-checked on load

    name = "model.language_model.layers.15.mlp.down_proj.weight"
    print(base.tensor(name).shape)  # torch.Size([1024, 3584]); tensor() returns a copy

    # Halve the first 64 columns. The base model is unchanged.
    edit = apply_operator(base, "scale_slice", {
        "tensor": name,
        "selection": [{"start": 0, "stop": 1024, "step": 1}, {"start": 0, "stop": 64, "step": 1}],
        "factor": 0.5,
    })
    print([(d.tensor, d.changed_elements) for d in edit.diff if d.changed_elements])
    assert replay_patch(base, edit.patch, blocks=edit.blocks).weights_hash == edit.candidate.weights_hash

    # Answer one problem with each model and grade the last \boxed{} answer.
    # qwen35think@1 is the article's profile; thinking is switched off here so the answer is short.
    profile = with_overrides(resolve_profile(engine.registry, "qwen35think@1", base.source),
                             {"chat_template_kwargs": {"enable_thinking": False}, "max_new_tokens": 512})
    problems = [TextItem(item_id="demo", problem="What is 17 * 23?", reference="391")]
    files = engine.model_files(base_id)  # the stored tokenizer and chat template
    for label, model in [("base", base), ("edited", edit.candidate)]:
        text_model = load_text_model(model, files, device="mps")
        (output,) = generate_samples(text_model, profile, problems, [(0, 0)], eval_seed=0)
        grade = MathBoxedGrader().grade(output.text, problems[0].reference)
        print(label, len(output.tokens), "tokens, answer", grade.answer, "correct:", grade.correct)
```

On an M4 Pro it takes about 30 seconds and prints:

```text
torch.Size([1024, 3584])
[('model.language_model.layers.15.mlp.down_proj.weight', 65536)]
base 262 tokens, answer 391 correct: True
edited 236 tokens, answer 391 correct: True
```

The workspace keeps the imported weights (about 1.5 GB for Qwen3.5-0.8B), and
the command-line example below uses the same one. Use `device="mps"` on Apple
silicon: on CPU, Qwen3.5 takes seconds per generated token, because
`transformers` runs its Gated DeltaNet layers through reference PyTorch code.

### On the command line: replay a published edit and compare

`wse import-patch` imports a published edit as an exact patch and refuses it
unless every rebuilt tensor matches its published hash. `wse
compare-generation` then evaluates the base and the edited model on the same
seeded subset of a dataset, with the same prompts and seeds, and reports
accuracy, limit-hit rate, and mean generated tokens with intervals. This
example uses the article's p50c edit and two MATH-500 problems with at most
1,024 new tokens; `jq` reads the checkpoint IDs from the printed JSON. It takes
about two minutes on an M4 Pro:

```bash
M=~/Desktop/wse-models
W=$M/workspaces/demo
BASE=$(uv run --no-sync wse --workspace "$W" import-hf "$M/Qwen3.5-0.8B" --repo Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 | jq -r .checkpoint_id)
P50C=$(uv run --no-sync wse --workspace "$W" import-patch "$M/lm_understand-logs/exp8_math/runs/p50c/edit" \
  --format lm-understand-v1 --base "$BASE" --repo RikiyaT/lm_understand-logs \
  --revision 4d09ba6d6a7e891dd8063974dfe81b3d78e3494f | jq -r .checkpoint_id)
uv run --no-sync wse --workspace "$W" compare-generation --base "$BASE" --candidate "$P50C" \
  --dataset "$M/datasets/MATH-500/test.jsonl" --dataset-format math500 --problems 2 --subset-seed 0 \
  --profile qwen35think@1 --grader math_boxed@1 --max-new-tokens 1024 --batch-size 8 --out "$M/runs/demo"
```

At 1,024 new tokens both models hit the limit on both problems, so this run
shows only that the pipeline works. The difference the article reports appears
at the profile's full limit of 32,768 tokens: with `--problems 25` and no
`--max-new-tokens`, this is the comparison in
[docs/m2-q1-report.md](docs/m2-q1-report.md), which took about six hours on an
M4 Pro. The output directory holds `run.json`, `problems.jsonl` (one line per
finished problem and model), and `summary.json`; running the same command again
resumes an interrupted run.

## Managed experiments and agents

A managed experiment runs in a workspace (`metadata.sqlite` plus
content-addressed artifacts). The operator imports a model, registers the task,
the policy, and the evaluation suites, and grants one run per trial. The
controller, either a scripted strategy or an agent, then previews, applies, and
evaluates patches through the same operations, within the policy's budget and
with every action audited. After the trial is frozen, the operator runs the
final evaluation, exports the experiment, and replays the export. This demo
runs one scripted trial on M1's toy task, offline, in a few seconds:

```bash
uv run --no-sync python examples/signed_sum/run_trials.py /tmp/wse-demo --label scripted
```

`uv run --no-sync wse --help` lists the operator commands, among them
`grant-run`, `serve` (the MCP server for one run grant), `run`, `export`,
`replay`, and `trials`. The reference agent uses the OpenAI Agents SDK;
[docs/experiments.md](docs/experiments.md) shows how to run it against a live
model.

## Documentation

- [docs/extensions.md](docs/extensions.md): the extension contracts (model
  adapters, operators, evaluators, strategies, agent adapters), direct Python
  use, and exact replay versus recipe re-execution.
- [docs/experiments.md](docs/experiments.md): running managed experiments,
  what each trial records and exports, and how to read the results.
- [docs/protocol.md](docs/protocol.md): the MCP tools, resources, and errors.
- [docs/m1-agent-trials.md](docs/m1-agent-trials.md): the M1 agent trials.
- [docs/m2-q1-report.md](docs/m2-q1-report.md): M2's first stage, with
  throughput, replay of the published edits, and the base-versus-p50c
  comparison.
- [agentic-weight-space-engineering-plan.md](agentic-weight-space-engineering-plan.md)
  (M1 and the roadmap) and
  [agentic-weight-space-engineering-m2-plan.md](agentic-weight-space-engineering-m2-plan.md):
  the specifications.
- [weight-space-engineering-vision.md](weight-space-engineering-vision.md) and
  [weight-space-programming-summary.md](weight-space-programming-summary.md):
  background.

## Tests

```bash
uv run --no-sync pytest -q
```

The suite runs offline in about two to three minutes. It builds every model
fixture from configuration classes and needs no snapshot or network.
