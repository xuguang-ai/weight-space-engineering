"""``classification@1``: argmax accuracy, cluster-aware intervals, and paired comparisons.

The evaluator scores the first ``sample_count`` examples of a suite in
batches of the server-fixed size through the metered forward callable, and
returns one score per example (1 when the argmax of the logits equals the
label; ties select the lowest class index, as ``torch.argmax`` does). Rows
with non-finite logits score 0 and are reported in ``errors``.

The statistics here are shared by direct and managed evaluation:

* ``accuracy_interval``: a cluster-bootstrap 95% percentile interval. The
  ``x, -x`` examples of the signed-sum fixture are correlated, so clusters are
  the generated pairs; independent examples are their own clusters.
* ``clopper_pearson``: the exact binomial 95% interval, used on the number of
  fully correct clusters. At the fixture's 50% and 100% boundaries every pair
  has the same score, so the bootstrap interval has zero width; the exact
  interval still bounds the result (512/512 pairs: lower bound 99.28%).
* ``paired_comparison``: candidate-minus-base accuracy with a paired cluster
  bootstrap (both reports resampled with the same clusters).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from wse.evaluators.base import MeteredForward
from wse.schemas import ErrorCode, EvalConfig, EvaluationSuite, IntervalEstimate, WSEError

CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0


class ClassificationEvaluator:
    """Argmax accuracy over labeled examples."""

    evaluator_id = "classification"
    version = "1"

    def evaluate(
        self, forward: MeteredForward, suite: EvaluationSuite, config: EvalConfig, *, batch_size: int
    ) -> dict[str, Any]:
        data = suite.data
        if data is None or data.labels is None:
            raise WSEError(ErrorCode.INVALID_ARGUMENT, f"suite {suite.suite_id} has no labeled examples")
        count = config.sample_count
        inputs = torch.tensor(data.inputs[:count], dtype=torch.float32)
        labels = torch.tensor(data.labels[:count], dtype=torch.int64)
        scores: list[int] = []
        nonfinite = 0
        for start in range(0, count, batch_size):
            logits = forward(inputs[start : start + batch_size])
            rows = labels[start : start + batch_size]
            if not isinstance(logits, torch.Tensor) or logits.dim() != 2 or logits.shape[0] != rows.shape[0]:
                raise WSEError(ErrorCode.WORKER_FAILED, "the model returned logits of an unexpected shape")
            finite = torch.isfinite(logits).all(dim=1)
            nonfinite += int((~finite).sum())
            correct = (logits.argmax(dim=1) == rows) & finite
            scores.extend(int(value) for value in correct.tolist())
        forward.check()
        errors = [f"{nonfinite} examples produced non-finite logits and were scored incorrect"] if nonfinite else []
        return {"metrics": {"accuracy": sum(scores) / count}, "sample_count": count, "errors": errors,
                "scores": scores, "truncation_count": 0}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _log_binomial_pmf(k: int, n: int, p: float) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def _upper_tail(x: int, n: int, p: float) -> float:
    """P(X >= x) for X ~ Binomial(n, p), 0 < p < 1."""
    return min(1.0, sum(math.exp(_log_binomial_pmf(k, n, p)) for k in range(x, n + 1)))


def _bisect(function, target: float, *, increasing: bool) -> float:
    low, high = 0.0, 1.0
    for _ in range(200):
        middle = (low + high) / 2
        value = function(middle)
        if (value < target) == increasing:
            low = middle
        else:
            high = middle
        if high - low < 1e-15:
            break
    return (low + high) / 2


def clopper_pearson(successes: int, trials: int, level: float = CONFIDENCE_LEVEL) -> tuple[float, float]:
    """The exact (Clopper-Pearson) two-sided interval for a binomial proportion."""
    if trials < 1 or not 0 <= successes <= trials:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "Clopper-Pearson needs 0 <= successes <= trials and trials >= 1")
    alpha = 1 - level
    if successes == 0:
        lower = 0.0
    elif successes == trials:
        lower = (alpha / 2) ** (1 / trials)
    else:
        lower = _bisect(lambda p: _upper_tail(successes, trials, p), alpha / 2, increasing=True)
    if successes == trials:
        upper = 1.0
    elif successes == 0:
        upper = 1 - (alpha / 2) ** (1 / trials)
    else:
        # P(X <= x) = 1 - P(X >= x + 1) decreases in p.
        upper = _bisect(lambda p: 1 - _upper_tail(successes + 1, trials, p), alpha / 2, increasing=False)
    return lower, upper


def _cluster_totals(scores: Sequence[int], cluster_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Correct examples and sizes per cluster, in order of first appearance."""
    order: dict[int, int] = {}
    for cluster in cluster_ids:
        order.setdefault(cluster, len(order))
    correct = np.zeros(len(order), dtype=np.int64)
    sizes = np.zeros(len(order), dtype=np.int64)
    for score, cluster in zip(scores, cluster_ids, strict=True):
        correct[order[cluster]] += score
        sizes[order[cluster]] += 1
    return correct, sizes


def _resample_indices(cluster_count: int, resamples: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, cluster_count, size=(resamples, cluster_count))


def _percentiles(values: np.ndarray, level: float) -> tuple[float, float]:
    alpha = 1 - level
    lower, upper = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)


def accuracy_interval(scores: Sequence[int], cluster_ids: Sequence[int], cluster_unit: str, *,
                      resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
                      level: float = CONFIDENCE_LEVEL) -> IntervalEstimate:
    """Cluster-bootstrap percentile interval of accuracy."""
    correct, sizes = _cluster_totals(scores, cluster_ids)
    index = _resample_indices(len(sizes), resamples, seed)
    lower, upper = _percentiles(correct[index].sum(axis=1) / sizes[index].sum(axis=1), level)
    estimate = float(correct.sum() / sizes.sum())
    return IntervalEstimate(method="cluster_bootstrap", level=level, estimate=estimate,
                            lower=lower, upper=upper, cluster_unit=cluster_unit,
                            resamples=resamples, seed=seed)


def fully_correct_interval(scores: Sequence[int], cluster_ids: Sequence[int], cluster_unit: str, *,
                           level: float = CONFIDENCE_LEVEL) -> IntervalEstimate:
    """Clopper-Pearson interval on the proportion of clusters with every example correct."""
    correct, sizes = _cluster_totals(scores, cluster_ids)
    successes, trials = int((correct == sizes).sum()), len(sizes)
    lower, upper = clopper_pearson(successes, trials, level)
    return IntervalEstimate(method="clopper_pearson", level=level, estimate=successes / trials, lower=lower,
                            upper=upper, cluster_unit=cluster_unit, successes=successes, trials=trials)


def paired_comparison(base_scores: Sequence[int], candidate_scores: Sequence[int], cluster_ids: Sequence[int],
                      cluster_unit: str, *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
                      level: float = CONFIDENCE_LEVEL) -> dict[str, Any]:
    """Candidate-minus-base accuracy with a paired cluster-bootstrap interval."""
    base, sizes = _cluster_totals(base_scores, cluster_ids)
    candidate, _ = _cluster_totals(candidate_scores, cluster_ids)
    index = _resample_indices(len(sizes), resamples, seed)
    differences = (candidate[index].sum(axis=1) - base[index].sum(axis=1)) / sizes[index].sum(axis=1)
    lower, upper = _percentiles(differences, level)
    difference = float((candidate.sum() - base.sum()) / sizes.sum())
    return {
        "base_value": float(base.sum() / sizes.sum()),
        "candidate_value": float(candidate.sum() / sizes.sum()),
        "difference": difference,
        "difference_interval": IntervalEstimate(
            method="cluster_bootstrap", level=level, estimate=difference, lower=lower,
            upper=upper, cluster_unit=cluster_unit, resamples=resamples, seed=seed),
        "clusters_improved": int((candidate > base).sum()),
        "clusters_worsened": int((candidate < base).sum()),
        "clusters_unchanged": int((candidate == base).sum()),
    }
