"""WSE's Model API in a few minutes: load a model, inspect it, edit its weights, and compare the edit with the original.

The steps, on Qwen3-0.6B by default:

1. Load: ``Model.from_pretrained`` downloads the snapshot (or reads a local one), records the commit that its
   revision resolves to, and imports the weights into a workspace.
2. Inspect: the architecture and parameters, the tensors of one attention layer, and a copy of one weight.
3. Generate: one answer, with its length and why it ended.
4. Edit: ``fill_slice`` zeroes the 128 input columns of attention head 7 in layer 14's output projection (an
   ablation of one head). The edit is an exact patch; ``verify`` replays it, and ``save`` stores the edited
   model in the workspace.
5. Compare: ``compare_models`` runs the original and the edited model on the same problems, prompts, sampling
   settings, and seeds, and reports paired differences with 95% intervals.
6. Export: the exact patch, and the edited model as a Hugging Face model directory.

The edit shows how the API works; it is not meant to improve the model. With a few problems and a short token
limit, the comparison's intervals are wide.

Usage (the models-hf extra; the first run downloads about 1.5 GB). ``--revision`` fixes the commit that the
README's output came from; without it, the model is downloaded at its latest commit:

    uv run --extra models-hf python examples/model_quickstart.py \\
      runs/model-quickstart \\
      --revision c1899de289a04d12100db370d81485cdf75e47ca

A local snapshot names the repository and the commit it was downloaded from:

    uv run --extra models-hf python examples/model_quickstart.py runs/model-quickstart \\
      --model ~/models/Qwen3-0.6B \\
      --repo Qwen/Qwen3-0.6B \\
      --revision c1899de289a04d12100db370d81485cdf75e47ca

The layer and column indices are Qwen3-0.6B's: 28 layers, and 16 attention heads of 128 dimensions each.
"""

import argparse
from pathlib import Path

from wse import Model, compare_models

parser = argparse.ArgumentParser(description="WSE's Model API: load, inspect, generate, edit, compare, and export.")
parser.add_argument("out", type=Path, help="a new directory for the workspace, the comparison, and the exports")
parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="a Hugging Face repository ID or a local snapshot")
parser.add_argument("--repo", help="a local snapshot's source repository")
parser.add_argument("--revision", help="the revision to download (default main), or a local snapshot's commit")
parser.add_argument("--data", type=Path, default=Path("examples/data/math_dev.jsonl"), help="math problems")
parser.add_argument("--problems", type=int, default=8, help="how many of the problems the comparison uses")
parser.add_argument("--max-new-tokens", type=int, default=1024, help="the token limit of each answer")
args = parser.parse_args()
if args.out.exists():
    parser.error(f"{args.out} exists; each run needs a new directory")

# 1. Load: the snapshot at a resolved commit, imported into a workspace.
original = Model.from_pretrained(
    args.model,
    repo=args.repo,
    revision=args.revision,
    workspace=args.out / "workspace",
)
print(original.inspect())

# 2. Inspect: the tensors of one attention layer, and a copy of one weight (changing the copy changes nothing).
for entry in original.tensors("layers.14.self_attn.*"):
    print(entry)
o_proj = original.tensor("layers.14.self_attn.o_proj")
print(f"norm of head 7's input columns in layer 14's o_proj: {o_proj[:, 896:1024].float().norm():.2f}")

# 3. Generate: one answer, how many tokens it took, and why it ended.
answer = original.generate("What is 17 * 24?", max_new_tokens=2048)
print(f"{answer.generated_tokens} tokens, finish reason {answer.finish_reason}")
print(answer.text.split("</think>")[-1].strip())  # the answer after the model's thinking

# 4. Edit: zero the 128 input columns of attention head 7 in layer 14's output projection.
edit = original.edit(
    "layers.14.self_attn.o_proj[:, 896:1024]",
    "fill_slice",
    value=0.0,
)
print(edit.table())
print(f"the patch replays to the edited weights: {edit.verify()}")
edited = edit.save().model
print(f"saved: patch {edit.patch_id}, checkpoint {edited.checkpoint_id}")

# 5. Compare: the same problems, prompts, sampling settings, and seeds for both models.
comparison = compare_models(
    base=original,
    candidate=edited,
    data=args.data,
    evaluator="math",
    metrics=["accuracy", "limit_hit_rate", "mean_generated_tokens"],
    max_new_tokens=args.max_new_tokens,
    problems=args.problems,
    out_dir=args.out / "comparison",
)
print(comparison.table())

# 6. Export: the exact patch with its provenance, and the edited model as a Hugging Face directory.
print(f"patch: {edit.export(args.out / 'patch')}")
print(f"model: {edited.export(args.out / 'edited-model')}")
