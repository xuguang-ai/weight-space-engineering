"""WSE's Model API in a few minutes: load a model, inspect it, edit its weights, and compare the edit with the original.

This is the README's Python API walkthrough in one file. Run it in a directory that holds math_dev.jsonl
(examples/data/ in this repository). It writes to runs/model-quickstart; delete that directory to run it again.
The first run downloads Qwen3-0.6B, about 1.5 GB.

1. Load: ``Model.from_pretrained`` downloads the snapshot, records the commit that its revision resolves to,
   and imports the weights into a workspace.
2. Inspect: the architecture and parameters, the tensors of one attention layer, and a copy of one weight.
3. Generate: one answer, with its length and why it ended.
4. Edit: ``fill_slice`` zeroes the 128 input columns of attention head 7 in layer 14's output projection (an
   ablation of one head). The edit is an exact patch; ``verify`` replays it, and ``save`` stores the edited
   model in the workspace.
5. Compare: ``compare_models`` runs the original and the edited model on the same problems, prompts, sampling
   settings, and seeds, and reports paired differences with 95% intervals.
6. Export: the exact patch, and the edited model as a Hugging Face model directory.

The edit shows how the API works; it is not meant to improve the model. With a few problems and a short token
limit, the comparison's intervals are wide. The layer and column indices are Qwen3-0.6B's: 28 layers, and 16
attention heads of 128 dimensions each.
"""

from wse import Model, compare_models

# 1. Load: the snapshot at a resolved commit, imported into a workspace.
original = Model.from_pretrained(
    "Qwen/Qwen3-0.6B",
    revision="c1899de289a04d12100db370d81485cdf75e47ca",
    workspace="runs/model-quickstart/workspace",
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
    data="math_dev.jsonl",
    evaluator="math",
    metrics=["accuracy", "limit_hit_rate", "mean_generated_tokens"],
    max_new_tokens=1024,
    problems=8,
    out_dir="runs/model-quickstart/comparison",
)
print(comparison.table())

# 6. Export: the exact patch with its provenance, and the edited model as a Hugging Face directory.
print(f"patch: {edit.export('runs/model-quickstart/patch')}")
print(f"model: {edited.export('runs/model-quickstart/edited-model')}")
