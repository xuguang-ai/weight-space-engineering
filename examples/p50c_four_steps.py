"""Qwen keeps looping in its reasoning: inspect the model, edit its weights, and measure whether it finishes more answers.

The README's "Edit a language model with Python" section shows this script step by step, on Qwen3.5-0.8B and
32 GSM8K training problems:

1. Diagnose: generate answers, count those cut off at the token limit, and count how many of them end in a loop
   (at least half of the lines in their last 2,000 characters repeat an earlier line there).
2. Probe: read p50c's change to the weights of ``mlp.down`` in layers 15 and 19, recover the direction it acts on,
   and compare the activations of cut-off and finished answers with that direction.
3. Edit: ``project_out`` with p50c's alpha, 6, subtracts six times each column's projection onto that direction
   from p50c's 64 columns in each layer (alpha 1 would remove the component); each edit is saved with its exact
   patch.
4. Evaluate: a paired comparison of the base and the edited model on the same problems, prompts, and seeds,
   recorded and resumable.

The layers, the columns, and each direction come from p50c, the edit published with "Sparks of Weight-Space
Programming": a direction is the top singular vector of p50c's change to its columns, signed so that its
largest-magnitude entry is positive, as ``svd_top_k@1`` signs its vectors. The script reconstructs p50c's edit
from the published change; it does not discover it. GSM8K's training set serves as practice problems, so that
MATH-500 stays unseen. The README lists the downloads into the data directory.

Usage: ``uv run --extra models-hf python examples/p50c_four_steps.py [DATA_DIR]`` (default ``~/wse-data``).
On an M4 Pro it takes about two hours, almost all of it in three generation passes of about 37 minutes each
(``compare`` generates the base's answers again).
"""

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

# 1. Diagnose: how many answers are cut off at the token limit, and how many of those end in a loop?
run = base.evaluate(
    gsm8k,
    format="gsm8k",
    problems=32,
    profile="qwen35think@1",
    max_new_tokens=8192,
)
print(run.table())


def ends_in_loop(text: str, window: int = 2000) -> bool:
    """True when at least half of the lines in the last ``window`` characters repeat an earlier line there."""
    lines = [line.strip() for line in text[-window:].split("\n")[1:-1]]
    lines = [line for line in lines if line]
    return bool(lines) and 2 * (len(lines) - len(set(lines))) >= len(lines)


cut_off, finished = run.limit_hit(), run.finished()
loops = [k for k in cut_off if ends_in_loop(run.outputs[k].text)]
print(f"{len(cut_off)} answers cut off at the limit, {len(finished)} finished")
print(f"{len(loops)} of the {len(cut_off)} cut-off answers end in a loop; the end of the first:")
if loops:
    print("\n".join(run.outputs[loops[0]].text.splitlines()[-9:-1]))

# 2. Probe: where does p50c change the weights, and do cut-off and finished answers differ along its direction?
p50c = base.apply_patch(
    data / "lm_understand-logs/exp8_math/runs/p50c/edit",
    repo="RikiyaT/lm_understand-logs",
    revision="4d09ba6d6a7e891dd8063974dfe81b3d78e3494f",
)
print(p50c.table())


def recover(layer: int) -> tuple[str, np.ndarray, np.ndarray]:
    """p50c in one layer: the tensor, the direction it acts on (its change's top singular vector), its columns."""
    name = base.name(f"layers.{layer}.mlp.down_proj")
    delta = p50c.model.tensor(name).double().numpy() - base.tensor(name).double().numpy()
    columns = np.flatnonzero(delta.any(axis=0))
    d = np.linalg.svd(delta[:, columns], full_matrices=False)[0][:, 0]
    d *= np.sign(d[np.argmax(np.abs(d))])  # svd_top_k's sign convention: the largest-magnitude entry is positive
    return name, d, columns


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
