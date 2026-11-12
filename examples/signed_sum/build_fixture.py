"""Build the signed-sum MLP fixture.

OPERATOR AND ENGINEERING-TEST USE ONLY. This file describes the injected fault
and the clean reference weights. It must never be visible to an agent under
trial: the reference runner isolates it, and agent instructions are generated
from task.json and the policy, never from this file.

Task: given x = (x0, x1), output class 1 when x0 + x1 > 0, otherwise class 0.
The model is an FP32 2 -> 4 (ReLU) -> 2 MLP with zero biases. The reference
first layer computes the four positive/negative input directions; the second
layer produces the two class scores. The task model multiplies row 1
(zero-indexed) of the second-layer weight by -1, which makes both rows
identical, so deterministic argmax always selects class 0 (50% accuracy on
balanced data). Restoring the row gives 100%.

Bundle provenance labels are neutral: they never describe the fault, and the
bundle's location on disk is never recorded by the importer.

``setup_workspace`` is the operator setup of a workspace for the task: it
imports the task model, registers task.json and policy.json, generates the
development set (32 pairs, 64 examples, fixed seed) and an independent final
set (512 pairs, 1,024 examples, a fresh random seed unless one is given) into
the workspace's evaluator storage, and registers their suites. The final
seed is kept only in evaluator storage, next to the final examples.

Usage:
    uv run python examples/signed_sum/build_fixture.py OUT_DIR [--with-reference]
        [--workspace WORKSPACE]
"""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

ADAPTER = "mlp@1"
CONFIG = {"layer_sizes": [2, 4, 2], "activation": "relu", "bias": True}
W1 = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
W2 = [[-1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 1.0, -1.0]]
FAULT_ROW = 1
MARGIN = 0.1

TASK_MODEL_DIR = "task_model"
REFERENCE_MODEL_DIR = "reference_model"
EXAMPLE_DIR = Path(__file__).resolve().parent
TASK_FILE = EXAMPLE_DIR / "task.json"
POLICY_FILE = EXAMPLE_DIR / "policy.json"
TASK_ID = "signed_sum_v1"
DEV_SUITE_ID = "signed_sum_dev_v1"
FINAL_SUITE_ID = "signed_sum_final_v1"
EVALUATOR = "classification@1"
DEV_PAIRS = 32
FINAL_PAIRS = 512
DEV_SEED = 20261104
GENERATOR = "signed_sum.generate_examples@1"
INPUT_DESCRIPTION = "x = (x0, x1)"
OUTPUT_DESCRIPTION = "logits for class 0 and class 1; the predicted class is the argmax"


def reference_tensors() -> dict[str, torch.Tensor]:
    """Clean weights that solve the task exactly."""
    return {
        "layers.0.weight": torch.tensor(W1, dtype=torch.float32),
        "layers.0.bias": torch.zeros(4, dtype=torch.float32),
        "layers.1.weight": torch.tensor(W2, dtype=torch.float32),
        "layers.1.bias": torch.zeros(2, dtype=torch.float32),
    }


def task_tensors() -> dict[str, torch.Tensor]:
    """The task model: the reference with the injected fault."""
    tensors = reference_tensors()
    tensors["layers.1.weight"][FAULT_ROW] *= -1.0
    return tensors


def generate_examples(n_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Balanced ``x, -x`` pairs with ``|x0 + x1| >= MARGIN``.

    Returns float32 inputs ``[2 * n_pairs, 2]``, int64 labels, and the pair ID
    of every example (the cluster unit for interval estimates).
    """
    rng = np.random.default_rng(seed)
    points: list[np.ndarray] = []
    while len(points) < n_pairs:
        x = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        if abs(float(x[0]) + float(x[1])) >= MARGIN:
            points.append(x)
    inputs = np.empty((2 * n_pairs, 2), dtype=np.float32)
    inputs[0::2] = np.stack(points)
    inputs[1::2] = -inputs[0::2]
    labels = (inputs.sum(axis=1) > 0).astype(np.int64)
    pair_ids = np.repeat(np.arange(n_pairs, dtype=np.int64), 2)
    return inputs, labels, pair_ids


def suite_data(n_pairs: int, seed: int) -> dict[str, Any]:
    """Evaluation examples as a ``SuiteData`` mapping, clustered by generated pair."""
    inputs, labels, pair_ids = generate_examples(n_pairs, seed)
    return {"inputs": inputs.tolist(), "labels": labels.tolist(), "clusters": pair_ids.tolist(),
            "cluster_unit": "pair"}


def development_suite(dataset: Any = None, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """The development suite: all 64 examples, one preregistered seed."""
    suite = {"suite_id": DEV_SUITE_ID, "task_id": TASK_ID, "split": "development", "evaluator": EVALUATOR,
             "sample_counts": [2 * DEV_PAIRS], "seeds": [0]}
    if dataset is not None:
        suite["dataset"] = dataset.model_dump(mode="json") if hasattr(dataset, "model_dump") else dataset
    if data is not None:
        suite["data"] = data
    return suite


def load_task() -> dict[str, Any]:
    return json.loads(TASK_FILE.read_text(encoding="utf-8"))


def load_policy() -> dict[str, Any]:
    return json.loads(POLICY_FILE.read_text(encoding="utf-8"))


def setup_workspace(engine: Any, out_dir: Path, *, final_seed: int | None = None,
                    with_reference: bool = False) -> dict[str, Any]:
    """Operator setup: import the task model and register the task, policy, data, and suites.

    The development and final sets are generated here, at setup time, into
    the workspace's evaluator storage; the final seed is drawn at random
    unless given and is stored only with the final examples. Returns the IDs
    the operator needs to grant runs.
    """
    paths = build(Path(out_dir), with_reference=with_reference)
    task_model = engine.import_bundle(paths["task"])
    engine.register_task(load_task())
    policy = engine.register_policy(load_policy())
    development = engine.register_dataset(
        suite_data(DEV_PAIRS, DEV_SEED),
        generation={"generator": GENERATOR, "pairs": DEV_PAIRS, "seed": DEV_SEED, "margin": MARGIN})
    seed = secrets.randbits(63) if final_seed is None else int(final_seed)
    final = engine.register_dataset(
        suite_data(FINAL_PAIRS, seed),
        generation={"generator": GENERATOR, "pairs": FINAL_PAIRS, "seed": seed, "margin": MARGIN})
    engine.register_suite(development_suite(development))
    engine.register_suite({"suite_id": FINAL_SUITE_ID, "task_id": TASK_ID, "split": "final", "evaluator": EVALUATOR,
                           "sample_counts": [2 * FINAL_PAIRS], "seeds": [0],
                           "dataset": final.model_dump(mode="json")})
    ids = {"task_id": TASK_ID, "policy_id": policy.policy_id, "task_checkpoint_id": task_model.checkpoint_id,
           "development_suite_id": DEV_SUITE_ID, "final_suite_id": FINAL_SUITE_ID}
    if with_reference:
        ids["reference_checkpoint_id"] = engine.import_bundle(paths["reference"]).checkpoint_id
    return ids


def write_bundle(directory: Path, tensors: dict[str, torch.Tensor], source: str) -> Path:
    """Write a ``wse_bundle.json`` bundle and return its directory."""
    directory.mkdir(parents=True, exist_ok=False)
    save_file(tensors, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n", encoding="utf-8")
    bundle = {
        "schema_version": "1",
        "adapter": ADAPTER,
        "weights": "model.safetensors",
        "config": "config.json",
        "provenance": {"kind": "import", "source": source},
        "license": "generated fixture; no third-party weights",
        "input_description": INPUT_DESCRIPTION,
        "output_description": OUTPUT_DESCRIPTION,
    }
    (directory / "wse_bundle.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    return directory


def build(out_dir: Path, *, with_reference: bool = False) -> dict[str, Path]:
    """Write the task-model bundle and, for engineering tests, the reference bundle."""
    out_dir = Path(out_dir)
    paths = {"task": write_bundle(out_dir / TASK_MODEL_DIR, task_tensors(), "signed_sum model v1")}
    if with_reference:
        paths["reference"] = write_bundle(
            out_dir / REFERENCE_MODEL_DIR, reference_tensors(), "signed_sum engineering reference v1"
        )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--with-reference",
        action="store_true",
        help="also write the clean reference bundle (engineering tests only)",
    )
    parser.add_argument("--workspace", type=Path, default=None,
                        help="also set up this workspace: import the task model and register the task, policy, "
                             "development and final data, and suites")
    args = parser.parse_args(argv)
    if args.workspace is None:
        for name, path in build(args.out_dir, with_reference=args.with_reference).items():
            print(f"{name}: {path}")
        return 0
    from wse.api import Engine

    with Engine(args.workspace) as engine:
        ids = setup_workspace(engine, args.out_dir, with_reference=args.with_reference)
    print(json.dumps(ids, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
