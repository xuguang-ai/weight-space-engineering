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

Usage:
    uv run --no-sync python examples/signed_sum/build_fixture.py OUT_DIR [--with-reference]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    args = parser.parse_args(argv)
    for name, path in build(args.out_dir, with_reference=args.with_reference).items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
