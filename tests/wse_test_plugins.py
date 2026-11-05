"""Operator-side test extensions for worker processes (``--setup wse_test_plugins:setup``).

``slow_batches@1`` runs one metered forward call per batch and then waits
``parameters["sleep_seconds"]`` after each batch. With ``"cooperative": true``
it waits in short steps and calls ``forward.check()``, so cancellation and
timeouts stop it between steps; otherwise it blocks, and only the worker's
hard stop can end it. ``parameters["marker"]`` names a file that the
evaluator creates once its first batch has finished, so tests can act while
the job is running without sleeping for a fixed time.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch


class SlowBatchesEvaluator:
    evaluator_id = "slow_batches"
    version = "1"

    def evaluate(self, forward, suite, config, *, batch_size):
        inputs = torch.tensor(suite.data.inputs[: config.sample_count], dtype=torch.float32)
        sleep_seconds = float(suite.parameters.get("sleep_seconds", 0.0))
        cooperative = bool(suite.parameters.get("cooperative", False))
        marker = suite.parameters.get("marker")
        for start in range(0, len(inputs), batch_size):
            forward(inputs[start : start + batch_size])
            if marker and start == 0:
                Path(marker).write_text("first batch done")
            if cooperative:
                deadline = time.monotonic() + sleep_seconds
                while time.monotonic() < deadline:
                    forward.check()
                    time.sleep(0.01)
            else:
                time.sleep(sleep_seconds)
        return {"metrics": {"batches": -(-len(inputs) // batch_size)}, "sample_count": len(inputs), "errors": []}


def setup(registry):
    registry.register("evaluator", "slow_batches", "1", SlowBatchesEvaluator())


def setup_forward_marker(registry):
    """Also make every adapter forward call append to ``$WSE_TEST_FORWARD_LOG``."""
    import os

    from wse.adapters.mlp import MLPAdapter

    setup(registry)
    original = MLPAdapter.forward
    log = Path(os.environ["WSE_TEST_FORWARD_LOG"])

    def logged(self, tensors, inputs):
        with log.open("a") as handle:
            handle.write("forward\n")
        return original(self, tensors, inputs)

    MLPAdapter.forward = logged
