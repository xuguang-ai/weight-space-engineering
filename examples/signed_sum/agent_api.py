"""One agent experiment on a faulty 22-parameter classifier, with all of its setup written out.

The README's "Run an agent through MCP" section shows this script step by step:

1. Load the target model: a 2 -> 4 -> 2 classifier whose two output rows have identical weights.
2. Define the task: the statement the agent reads and the objective that counts as success.
3. Connect the evaluation: development examples the agent may evaluate, and a final test it never sees.
4. Configure the policy: the operations the agent may use and the budget for the whole experiment.
5. Start the agent: the runner grants a run on the registered IDs and connects the agent through MCP.
6. Read the result: how the run ended, the edit, the final test, and the patch.

With a live agent, the key and endpoint in ``.env`` (copied from ``.env.example``)::

    uv run --extra agents-openai --env-file .env \\
      python examples/signed_sum/agent_api.py \\
      demo/live \\
      --model YOUR_AGENT_MODEL

Offline, with the scripted controller ``scripted_search@1`` in place of a model::

    uv run python examples/signed_sum/agent_api.py demo/offline

Each run needs a new directory, because the final test is drawn at random when the workspace is set up.
``build_fixture.py`` prepares the same model, task, and data, with the same policy under its built-in ID
``small_model_v1``, for ``run_trials.py`` and the tests.
"""

import argparse
import os
import secrets
from pathlib import Path

import numpy as np
import torch

from wse.agent_runs import run_agent_experiment
from wse.api import Engine

parser = argparse.ArgumentParser(description="One agent experiment on a faulty 22-parameter classifier.")
parser.add_argument("out", type=Path, help="a new directory for the workspace and the run's records")
parser.add_argument("--model", help="the agent's model; without it, the scripted controller scripted_search@1")
args = parser.parse_args()
if args.out.exists():
    parser.error(f"{args.out} exists; each run needs a new directory")
workspace = args.out / "workspace"
engine = Engine(workspace)

# 1. Load the target model: a 2 -> 4 (ReLU) -> 2 classifier with 22 parameters.
#    Its two output rows have identical weights, so it always predicts class 0.
base = engine.import_model(
    {
        "adapter": "mlp@1",
        "config": {"layer_sizes": [2, 4, 2], "activation": "relu", "bias": True},
        "provenance": {"source": "signed_sum model v1"},
        "license": "generated fixture; no third-party weights",
        "input_description": "x = (x0, x1)",
        "output_description": "logits for class 0 and class 1; the predicted class is the argmax",
    },
    {
        "layers.0.weight": torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
        "layers.0.bias": torch.zeros(4),
        "layers.1.weight": torch.tensor([[-1.0, 1.0, -1.0, 1.0], [-1.0, 1.0, -1.0, 1.0]]),
        "layers.1.bias": torch.zeros(2),
    },
)

# 2. Define the task: the statement the agent reads, and the objective that counts as success.
task_definition = {
    "task_id": "signed_sum_v1",
    "title": "Signed-sum classification",
    "description": (
        "The registered model reads two real inputs x = (x0, x1) and returns logits for two classes; "
        "its prediction is the class with the larger logit. It should predict class 1 when x0 + x1 > 0 "
        "and class 0 otherwise. Using only the permitted inspection and editing operations, produce a "
        "candidate checkpoint that follows this rule, and select it."
    ),
    "input_description": "x = (x0, x1), two float32 values",
    "output_description": "logits for class 0 and class 1; the predicted class is the argmax",
    "objective": {
        "task_id": "signed_sum_v1",
        "description": "Predict class 1 exactly when x0 + x1 > 0, measured as accuracy on the development suite.",
        "gates": [{"metric": "accuracy", "min_value": 1.0}],
    },
    "target_accuracy": 1.0,
    "development_suite_id": "signed_sum_dev_v1",
    "development_examples": 64,
}
task = engine.register_task(task_definition)


# 3. Connect the evaluation: labeled examples, scored by accuracy.
def examples(pairs: int, seed: int) -> dict:
    """``pairs`` points x with |x0 + x1| >= 0.1, each followed by -x; the label is 1 when x0 + x1 > 0."""
    rng = np.random.default_rng(seed)
    points = []
    while len(points) < pairs:
        x = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        if abs(float(x[0]) + float(x[1])) >= 0.1:
            points.append(x)
    inputs = np.empty((2 * pairs, 2), dtype=np.float32)
    inputs[0::2] = np.stack(points)
    inputs[1::2] = -inputs[0::2]
    return {
        "inputs": inputs.tolist(),
        "labels": (inputs.sum(axis=1) > 0).astype(np.int64).tolist(),
        "clusters": np.repeat(np.arange(pairs), 2).tolist(),
        "cluster_unit": "pair",
    }


development = engine.register_dataset(examples(32, seed=20261104))
final = engine.register_dataset(examples(512, seed=secrets.randbits(63)))
for suite_id, split, dataset in [
    ("signed_sum_dev_v1", "development", development),
    ("signed_sum_final_v1", "final", final),
]:
    engine.register_suite({
        "suite_id": suite_id,
        "task_id": task.task_id,
        "split": split,
        "evaluator": "classification@1",
        "sample_counts": [dataset.example_count],
        "seeds": [0],
        "dataset": dataset.model_dump(mode="json"),
    })

# 4. Configure the policy: the operations the agent may use, and the budget for the whole experiment.
policy_definition = {
    "policy_id": "signed_sum_policy_v1",
    "mode": "bounded_eval",
    "operations": {
        "edits": {"scale_slice": "scale_slice@1", "set_slice": "set_slice@1"},
        "analyses": {"stats": "tensor_stats@1", "norm": "tensor_norm@1", "top_k": "tensor_top_k@1"},
    },
    "budget": {
        "evaluation_batches": 10,
        "examples_per_batch": 64,
        "evaluated_examples": 640,
        "model_forward_calls": 10,
        "candidate_commits": 20,
        "patch_previews": 50,
        "analysis_calls": 50,
        "tensor_read_calls": 100,
        "compute_seconds": 300.0,
        "seconds_per_operation": 30.0,
    },
}
policy = engine.register_policy(policy_definition)
engine.close()  # the setup is registered; the runner opens the workspace itself

# 5. Start the agent: the runner grants a run on these IDs, connects the agent to WSE through MCP,
#    and freezes the checkpoint it selects; then come the final test, an export, and its replay.
print(f"{args.model or 'scripted_search@1'} on model {base.checkpoint_id}, task {task.task_id}, "
      f"policy {policy.policy_id}")
run = run_agent_experiment(
    workspace,
    task=task.task_id,
    base=base.checkpoint_id,
    policy=policy.policy_id,
    model=args.model,
    strategy=None if args.model else "scripted_search@1",
    final_evaluation=True,
)

# 6. Read the result: how the run ended, the edit, the final test, and the patch.
summary = run.summary
print(f"{summary['controller']} ended {summary['status']}: {summary['reason']}")
print(f"{summary['tool_calls']} tool calls in {summary['elapsed_seconds']:.0f} s")
if summary["tokens"]["input_tokens"] is not None:
    print(f"{summary['turns']} model requests, {summary['tokens']['input_tokens']:,} input and "
          f"{summary['tokens']['output_tokens']:,} output tokens")
if run.state != "CLOSED":
    raise SystemExit(f"the experiment ended {run.state}: {summary['failure_cause']}")
engine = Engine(workspace)
selected = engine.get_experiment(run.experiment_id).selected_checkpoint_id
patch_id = next(c.patch_id for c in engine.list_experiment_checkpoints(run.experiment_id)
                if c.checkpoint_id == selected)
if patch_id is None:
    print("edit: none; the agent kept the unedited model")
else:
    patch = engine.find_patch(patch_id)
    for operation in patch.spec.operations:
        where = ", ".join(f"{axis.start}:{axis.stop}" for axis in operation.selection)
        values = ", ".join(f"{name} {value}" for name, value in operation.parameters.items())
        print(f"edit: {operation.operator} on {operation.tensor}[{where}], {values}")
    for diff in patch.preview.tensors:
        if diff.changed_elements:
            before = engine.open_checkpoint(base.checkpoint_id).tensor(diff.tensor).tolist()
            after = engine.open_checkpoint(selected).tensor(diff.tensor).tolist()
            print(f"{diff.tensor}: {before} -> {after}")
    print(f"patch: {os.path.relpath(run.export_dir / 'patches' / f'{patch_id}.json')}")
comparison = engine.get_final_report(run.experiment_id).comparison
print(f"final test, {comparison.sample_count:,} examples the agent never saw: "
      f"{comparison.base_value:.0%} correct before the edit, {comparison.candidate_value:.0%} after")
print(f"the export replays to byte-identical weights: {run.replay_identical}")
engine.close()
