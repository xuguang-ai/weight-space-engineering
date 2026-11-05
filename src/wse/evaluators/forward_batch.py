"""``forward_batch@1``: the minimal evaluation job body of Task 4.

It runs one metered forward call per batch over the suite's first
``sample_count`` examples and, when the suite has labels, reports argmax
accuracy. Task 5's classification evaluator, dataset manifests, and reports
replace or extend it.
"""

from __future__ import annotations

from typing import Any

import torch

from wse.evaluators.base import MeteredForward
from wse.schemas import EvalConfig, EvaluationSuite


class ForwardBatchEvaluator:
    evaluator_id = "forward_batch"
    version = "1"

    def evaluate(
        self, forward: MeteredForward, suite: EvaluationSuite, config: EvalConfig, *, batch_size: int
    ) -> dict[str, Any]:
        count = config.sample_count
        inputs = torch.tensor(suite.data.inputs[:count], dtype=torch.float32)
        labels = None if suite.data.labels is None else torch.tensor(suite.data.labels[:count], dtype=torch.int64)
        correct = 0
        for start in range(0, count, batch_size):
            logits = forward(inputs[start : start + batch_size])
            if labels is not None:
                correct += int((logits.argmax(dim=1) == labels[start : start + batch_size]).sum())
        metrics = {} if labels is None else {"accuracy": correct / count}
        return {"metrics": metrics, "sample_count": count, "errors": []}
