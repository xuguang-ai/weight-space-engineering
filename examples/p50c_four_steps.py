"""The four steps of a weight-space experiment on Qwen3.5-0.8B, with the article's p50c edit (README, How WSE Works).

1. **Behavioral diagnosis.** A generation pass on 32 GSM8K training problems shows which answers are cut off
   at the token limit.
2. **Mechanism investigation.** A probe of those answers at the output of ``mlp.down`` in layers 15 and 19
   compares the cut-off answers with the finished ones along the direction that p50c acts on there.
3. **Weight intervention.** ``project_out`` with p50c's alpha, 6, subtracts six times each column's
   projection onto that direction from p50c's 64 columns in each layer, as p50c does (alpha 1 would remove
   the component).
4. **Evaluation.** A paired comparison with the base on the same problems, prompts, and seeds, recorded and
   resumable.

Each layer's direction and columns are recovered from p50c's published change: the columns it changes and
the top singular vector of its change to them, signed so that its largest-magnitude entry is positive, as
``svd_top_k@1`` signs its vectors. They come from the published edit, not from the data: the script shows the
workflow; it does not discover the edit. The data layout is the README quick start's, plus GSM8K's training set as
practice problems, so that MATH-500 stays unseen::

    curl -L --create-dirs -o ~/wse-data/gsm8k/train.jsonl \\
        https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl

Usage: ``uv run --no-sync python examples/p50c_four_steps.py [DATA_DIR]`` (default ``~/wse-data``). On an
M4 Pro it takes about two hours, almost all of it in three generation passes of about 37 minutes each
(``compare`` generates the base's answers again).
"""

import sys
import time
from pathlib import Path

import numpy as np

from wse import WSE

M = Path(sys.argv[1] if len(sys.argv) > 1 else "~/wse-data").expanduser()
GSM8K = M / "gsm8k/train.jsonl"
SITES = {layer: f"layers.{layer}.mlp.down_proj:output" for layer in (15, 19)}
START = time.perf_counter()


def elapsed(step: str) -> None:
    print(f"[{step}: {(time.perf_counter() - START) / 60:.1f} min]", flush=True)


base = WSE.from_pretrained("Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17",
                           workspace=M / "workspace")
p50c = base.apply_patch(M / "lm_understand-logs/exp8_math/runs/p50c/edit",
                        repo="RikiyaT/lm_understand-logs", revision="4d09ba6d6a7e891dd8063974dfe81b3d78e3494f")


def recover(layer: int) -> tuple[str, np.ndarray, np.ndarray]:
    """p50c in one layer: the tensor, the direction it acts on (its change's top singular vector), its columns."""
    name = base.name(f"layers.{layer}.mlp.down_proj")
    delta = p50c.model.tensor(name).double().numpy() - base.tensor(name).double().numpy()
    columns = np.flatnonzero(delta.any(axis=0))
    d = np.linalg.svd(delta[:, columns], full_matrices=False)[0][:, 0]
    d *= np.sign(d[np.argmax(np.abs(d))])  # the sign convention of svd_top_k: the largest-magnitude entry is positive
    return name, d, columns


# 1. Behavioral diagnosis: which answers never finish?
run = base.evaluate(GSM8K, format="gsm8k", problems=32, profile="qwen35think@1", max_new_tokens=8192)
print(run.table())
looping, finished = run.limit_hit(), run.finished()
print(f"{len(looping)} answers cut off at the limit, {len(finished)} finished")
elapsed("diagnosis")

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
elapsed("investigation")

# 3. Weight intervention: p50c's form, project_out of that direction from its columns with alpha 6.
model = base
for layer in SITES:
    name, d, columns = recover(layer)
    model = model.edit(name, "project_out", direction=d, columns=columns, alpha=6.0).save().model
print(f"{sum(diff.changed_elements for diff in model.diff_from(base)):,} weights changed")
elapsed("intervention")

# 4. Evaluation: a paired, recorded comparison on the same problems, prompts, and seeds.
comparison = base.compare(model, GSM8K, format="gsm8k", problems=32, profile="qwen35think@1",
                          max_new_tokens=8192, out_dir=M / "runs/four-steps")
print(comparison.table())
elapsed("evaluation")
