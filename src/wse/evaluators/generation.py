"""The generation evaluator ``generation@1``: batched sampling, grading, and metrics (M2 plan Sections 4.2-4.3).

A generation suite holds text problems (``TextSuiteData``, in evaluator
storage) and these parameters, preregistered with the suite:

* ``profile``: the exact generation profile (a registered one such as
  ``qwen35think@1``, or the model's own ``<repo>_default@1``);
* ``grader``: the exact grader (``math_boxed@1`` or ``gsm8k_numeric@1``);
* ``samples_per_problem`` (default 1);
* optionally ``max_new_tokens``, which overrides the profile's limit.

``evaluate_generation`` renders each problem's prompt with the model's chat
template and the profile's arguments, generates every (problem, sample) row
in left-padded batches with its own seed (``wse.evaluators.sampling``), and
grades each output. A ``sample_count`` of ``n`` evaluates the first ``n``
problems.

* **Metrics.** ``accuracy`` (the mean over samples), ``limit_hit_rate`` (the
  share of outputs that reached ``max_new_tokens`` without a stop token), and
  ``mean_generated_tokens`` (stop token included).
* **Intervals.** Clusters are problems: cluster-bootstrap intervals for the
  three metrics, an exact Clopper-Pearson interval for each proportion
  (``accuracy_exact`` and ``limit_hit_rate_exact``, over samples), and M1's
  interval on fully correct problems.
* **Identity.** The result binds the model identity (weights, configuration,
  tokenizer, chat template, generation configuration), the suite and data
  hashes, and an execution configuration whose parameters add the resolved
  profile's hash, so a changed profile, grader, or limit is a different
  configuration. ``runtime`` records the engine's device, stored dtypes, and
  library versions.
* **Per-problem results.** Every sample's seed, grade, read answer, length,
  and limit hit, plus a hash of its tokens (``ProblemResult``).

``compare_generation`` pairs two evaluations of one suite and configuration.
Managed generation jobs (metered tokens, the worker's model cache) are not
part of this module; ``GenerationEvaluator.evaluate``, the M1 metered-forward
entry point, refuses them. ``transformers`` and ``math_verify`` are imported
lazily.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from wse.evaluators.classification import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    accuracy_interval,
    clopper_pearson,
    fully_correct_interval,
    paired_comparison,
)
from wse.evaluators.sampling import (
    DEFAULT_PROFILE_SUFFIX,
    logits_processors,
    profile_sampling,
    render_prompt,
    resolve_profile,
    sample_seed,
    with_overrides,
)
from wse.schemas import (
    ClusterSummary,
    ErrorCode,
    EvalConfig,
    EvaluationSuite,
    ExecutionConfig,
    ExtensionRef,
    GenerationComparison,
    GenerationProfile,
    IntervalEstimate,
    ProblemResult,
    SampleResult,
    TextItem,
    TextSuiteData,
    Usage,
    WSEError,
)

if TYPE_CHECKING:
    from wse.api import EvaluationResult, ModelHandle
    from wse.registry import Registry

EVALUATOR_ID = "generation"
EVALUATOR_VERSION = "1"
PARAMETERS = ("profile", "grader", "samples_per_problem", "max_new_tokens")
MAX_SAMPLES_PER_PROBLEM = 64
DEFAULT_BATCH_SIZE = 8
CLUSTER_UNIT = "problem"


def _invalid(message: str, **details: Any) -> WSEError:
    return WSEError(ErrorCode.INVALID_ARGUMENT, message, details=details)


# ---------------------------------------------------------------------------
# Text models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextModel:
    """What generation needs: an engine runtime, its tokenizer, and the stop and pad tokens."""

    runtime: Any  # wse.engines.hf_torch.HFTorchModel, or a stand-in with generate() and info()
    tokenizer: Any
    stop_token_ids: list[int]
    pad_token_id: int


def _config_token_ids(config: Any) -> list[int]:
    found: list[int] = []
    if isinstance(config, Mapping):
        for value in (config.get("eos_token_id"), *(
                sub.get("eos_token_id") for sub in config.values() if isinstance(sub, Mapping))):
            values = value if isinstance(value, list) else [value]
            found.extend(v for v in values if isinstance(v, int) and not isinstance(v, bool) and v >= 0)
    return found


def model_stop_tokens(model_config: Mapping[str, Any], files: Mapping[str, bytes], tokenizer: Any) -> list[int]:
    """The model's end-of-sequence tokens: the tokenizer's, the generation configuration's, and the model configuration's.

    vLLM stops at the tokenizer's end-of-sequence token and at those of the
    generation configuration (derived from the model configuration when the
    snapshot has no ``generation_config.json``); this is their union.
    """
    import json

    stops: list[int] = []
    if getattr(tokenizer, "eos_token_id", None) is not None:
        stops.append(int(tokenizer.eos_token_id))
    if "generation_config.json" in files:
        stops.extend(_config_token_ids(json.loads(files["generation_config.json"])))
    else:
        stops.extend(_config_token_ids(model_config))
    return sorted(dict.fromkeys(stops))


def load_text_model(handle: ModelHandle, files: Mapping[str, bytes], *, device: str | None = None) -> TextModel:
    """The text model of an imported checkpoint: its runtime on ``device`` and its stored tokenizer."""
    from wse.engines.hf_torch import load_tokenizer

    tokenizer = load_tokenizer(files)
    stops = model_stop_tokens(handle.config.get("model_config", {}), files, tokenizer)
    if not stops:
        raise WSEError(ErrorCode.UNSUPPORTED_CAPABILITY, "the model defines no end-of-sequence token")
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else stops[0]
    return TextModel(runtime=handle.runtime(device), tokenizer=tokenizer, stop_token_ids=stops, pad_token_id=int(pad))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleOutput:
    """One generated row: its problem index, sample index, seed, tokens (stop token included), and text."""

    index: int
    sample: int
    item_id: str
    seed: int
    tokens: list[int]
    finished: bool
    text: str


def _encode(text_model: TextModel, profile: GenerationProfile, prompt: str) -> list[int]:
    # A chat template writes its own special tokens (for example a BOS), so they are not added twice.
    ids = text_model.tokenizer(prompt, add_special_tokens=not profile.chat_template)["input_ids"]
    if not ids:
        raise _invalid("a prompt rendered to no tokens")
    return [int(token) for token in ids]


def generate_samples(text_model: TextModel, profile: GenerationProfile, items: Sequence[TextItem],
                     rows: Sequence[tuple[int, int]], *, eval_seed: int, batch_size: int = DEFAULT_BATCH_SIZE,
                     check: Callable[[], None] | None = None) -> list[SampleOutput]:
    """Generate the ``(problem index, sample index)`` rows in batches of ``batch_size``, in the order given.

    Prompts are left-padded and masked; each row draws from its own seed, so
    its tokens do not depend on its batch or position. The presence penalty
    of a batch counts output tokens only (the padded prompt length is shared).
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise _invalid("batch_size must be a positive integer")
    sampling = profile_sampling(profile)
    prompts = {index: _encode(text_model, profile, render_prompt(text_model.tokenizer, profile, items[index].problem))
               for index in sorted({index for index, _ in rows})}
    outputs: list[SampleOutput] = []
    for start in range(0, len(rows), batch_size):
        if check is not None:
            check()
        chunk = list(rows[start : start + batch_size])
        encoded = [prompts[index] for index, _ in chunk]
        width = max(len(ids) for ids in encoded)
        input_ids = torch.full((len(chunk), width), text_model.pad_token_id, dtype=torch.int64)
        mask = torch.zeros((len(chunk), width), dtype=torch.int64)
        for row, ids in enumerate(encoded):
            input_ids[row, width - len(ids):] = torch.tensor(ids, dtype=torch.int64)
            mask[row, width - len(ids):] = 1
        seeds = [sample_seed(profile.seed, eval_seed, items[index].item_id, sample) for index, sample in chunk]
        stops = profile.stop_token_ids if profile.stop_token_ids is not None else text_model.stop_token_ids
        result = text_model.runtime.generate(
            input_ids, mask, max_new_tokens=profile.max_new_tokens, stop_token_ids=stops,
            pad_token_id=text_model.pad_token_id, sampling=sampling, seeds=seeds if sampling is not None else None,
            logits_processors=logits_processors(profile, width))
        for (index, sample), seed, tokens, finished in zip(chunk, seeds, result.tokens, result.finished, strict=True):
            body = tokens[:-1] if finished and tokens else tokens  # the text excludes the stop token
            text = text_model.tokenizer.decode(body, skip_special_tokens=True)
            outputs.append(SampleOutput(index=index, sample=sample, item_id=items[index].item_id, seed=seed,
                                        tokens=list(tokens), finished=bool(finished), text=text))
    return outputs


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _exact(successes: int, trials: int) -> IntervalEstimate:
    lower, upper = clopper_pearson(successes, trials)
    return IntervalEstimate(method="clopper_pearson", level=CONFIDENCE_LEVEL, estimate=successes / trials,
                            lower=lower, upper=upper, cluster_unit="sample", successes=successes, trials=trials)


def _cluster_means(values: Sequence[float], cluster_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    order = {cluster: position for position, cluster in enumerate(dict.fromkeys(cluster_ids))}
    sums, sizes = np.zeros(len(order)), np.zeros(len(order))
    for value, cluster in zip(values, cluster_ids, strict=True):
        sums[order[cluster]] += value
        sizes[order[cluster]] += 1
    return sums, sizes


def _bootstrap(estimates: Callable[[np.ndarray], np.ndarray], clusters: int, estimate: float) -> IntervalEstimate:
    index = np.random.default_rng(BOOTSTRAP_SEED).integers(0, clusters, size=(BOOTSTRAP_RESAMPLES, clusters))
    lower, upper = np.quantile(estimates(index), [(1 - CONFIDENCE_LEVEL) / 2, 1 - (1 - CONFIDENCE_LEVEL) / 2])
    return IntervalEstimate(method="cluster_bootstrap", level=CONFIDENCE_LEVEL, estimate=float(estimate),
                            lower=float(lower), upper=float(upper), cluster_unit=CLUSTER_UNIT,
                            resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)


def mean_interval(values: Sequence[float], cluster_ids: Sequence[int]) -> IntervalEstimate:
    """Cluster-bootstrap percentile interval of a mean (resampling whole problems)."""
    sums, sizes = _cluster_means(values, cluster_ids)
    return _bootstrap(lambda index: sums[index].sum(axis=1) / sizes[index].sum(axis=1), len(sizes),
                      sums.sum() / sizes.sum())


def paired_mean_difference(base: Sequence[float], candidate: Sequence[float],
                           cluster_ids: Sequence[int]) -> IntervalEstimate:
    """Candidate-minus-base mean with a paired cluster bootstrap (both resampled with the same problems)."""
    base_sums, sizes = _cluster_means(base, cluster_ids)
    candidate_sums, _ = _cluster_means(candidate, cluster_ids)
    return _bootstrap(lambda index: (candidate_sums[index].sum(axis=1) - base_sums[index].sum(axis=1))
                      / sizes[index].sum(axis=1), len(sizes), (candidate_sums.sum() - base_sums.sum()) / sizes.sum())


def _flatten(problems: Sequence[ProblemResult]) -> tuple[list[SampleResult], list[int]]:
    samples, clusters = [], []
    for problem in problems:
        samples.extend(problem.samples)
        clusters.extend([problem.index] * len(problem.samples))
    return samples, clusters


def generation_metrics(problems: Sequence[ProblemResult]) -> tuple[dict[str, float], dict[str, IntervalEstimate],
                                                                   ClusterSummary]:
    """Metrics, intervals, and the problem clusters of per-problem results."""
    samples, clusters = _flatten(problems)
    correct = [int(s.correct) for s in samples]
    limits = [int(s.limit_hit) for s in samples]
    lengths = [float(s.generated_tokens) for s in samples]
    n = len(samples)
    metrics = {"accuracy": sum(correct) / n, "limit_hit_rate": sum(limits) / n,
               "mean_generated_tokens": sum(lengths) / n}
    fully = fully_correct_interval(correct, clusters, CLUSTER_UNIT)
    intervals = {
        "accuracy": accuracy_interval(correct, clusters, CLUSTER_UNIT),
        "accuracy_exact": _exact(sum(correct), n),
        "limit_hit_rate": accuracy_interval(limits, clusters, CLUSTER_UNIT),
        "limit_hit_rate_exact": _exact(sum(limits), n),
        "mean_generated_tokens": mean_interval(lengths, clusters),
        "fully_correct_clusters": fully,
    }
    return metrics, intervals, ClusterSummary(unit=CLUSTER_UNIT, count=fully.trials, fully_correct=fully.successes)


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------


def suite_parameters(suite: EvaluationSuite) -> dict[str, Any]:
    """The checked generation parameters of a suite (``samples_per_problem`` defaults to 1)."""
    parameters = dict(suite.parameters)
    unknown = sorted(set(parameters) - set(PARAMETERS))
    if unknown:
        raise _invalid(f"generation suites take the parameters {list(PARAMETERS)}; unknown: {unknown}")
    for key in ("profile", "grader"):
        if not isinstance(parameters.get(key), str):
            raise _invalid(f"a generation suite names its {key} exactly, as '<id>@<version>'")
        ExtensionRef.parse(parameters[key])
    samples = parameters.setdefault("samples_per_problem", 1)
    if isinstance(samples, bool) or not isinstance(samples, int) or not 1 <= samples <= MAX_SAMPLES_PER_PROBLEM:
        raise _invalid(f"samples_per_problem must be an integer from 1 to {MAX_SAMPLES_PER_PROBLEM}")
    limit = parameters.get("max_new_tokens")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise _invalid("max_new_tokens must be a positive integer")
    return parameters


class GenerationEvaluator:
    """``generation@1``: the registered evaluator of generation suites."""

    evaluator_id = EVALUATOR_ID
    version = EVALUATOR_VERSION

    def validate_suite(self, suite: EvaluationSuite, registry: Registry) -> None:
        """Registration check: text problems, known parameters, a registered grader, and a resolvable profile.

        A model's default profile (``<repo>_default@<n>``) is resolved through
        the evaluated model, so only its reference is checked here.
        """
        if not isinstance(suite.data, TextSuiteData):
            raise _invalid(f"suite {suite.suite_id}: generation suites hold text problems (a text dataset)")
        parameters = suite_parameters(suite)
        registry.resolve_ref("grader", parameters["grader"])
        if not ExtensionRef.parse(parameters["profile"]).extension_id.endswith(DEFAULT_PROFILE_SUFFIX):
            resolve_profile(registry, parameters["profile"], None)

    def evaluate(self, forward: Any, suite: EvaluationSuite, config: EvalConfig, *, batch_size: int) -> dict[str, Any]:
        raise WSEError(ErrorCode.UNSUPPORTED_CAPABILITY,
                       "generation suites are not evaluated through metered forward jobs yet; the operator evaluates "
                       "them with Engine.evaluate_generation")


def _grade(grader: Any, output: SampleOutput, reference: str, max_new_tokens: int) -> SampleResult:
    graded = grader.grade(output.text, reference)
    digest = hashlib.sha256(np.asarray(output.tokens, dtype="<i8").tobytes()).hexdigest()
    return SampleResult(sample=output.sample, seed=output.seed, correct=bool(graded.correct), answer=graded.answer,
                        generated_tokens=len(output.tokens), finished=output.finished,
                        limit_hit=not output.finished and len(output.tokens) >= max_new_tokens,
                        output_sha256="sha256:" + digest)


def _library_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for package in ("math-verify",):
        try:
            versions[package.replace("-", "_")] = version(package)
        except PackageNotFoundError:
            pass
    return versions


def evaluate_generation(
    model: ModelHandle,
    suite: EvaluationSuite,
    config: EvalConfig | Mapping[str, Any],
    *,
    files: Mapping[str, bytes] | None = None,
    text_model: TextModel | None = None,
    registry: Registry | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    profile_overrides: Mapping[str, Any] | None = None,
    check: Callable[[], None] | None = None,
) -> EvaluationResult:
    """Evaluate ``model`` on a generation suite with its problems attached (direct use, unmanaged).

    ``files`` are the model's stored tokenizer and chat-template files
    (``Engine.model_files``); ``text_model`` replaces the runtime and tokenizer
    built from them (for stand-ins). ``profile_overrides`` adjusts the
    resolved profile for direct use; the effective profile's hash is bound into
    the configuration either way.
    """
    from wse.api import EvaluationResult, _as_eval_config
    from wse.registry import builtin_registry

    registry = registry or builtin_registry()
    eval_config = _as_eval_config(config)
    if ExtensionRef.parse(suite.evaluator) != ExtensionRef(extension_id=EVALUATOR_ID, version=EVALUATOR_VERSION):
        raise _invalid(f"suite {suite.suite_id} is evaluated by {suite.evaluator}, not {EVALUATOR_ID}@{EVALUATOR_VERSION}")
    if not isinstance(suite.data, TextSuiteData):
        raise _invalid(f"suite {suite.suite_id} has no text problems attached; resolve it from evaluator storage first")
    parameters = suite_parameters(suite)
    count = eval_config.sample_count
    if not 1 <= count <= suite.data.example_count:
        raise _invalid(f"sample_count must lie between 1 and the {suite.data.example_count} problems of the suite")
    overrides = dict(profile_overrides or {})
    if parameters.get("max_new_tokens") is not None:
        overrides.setdefault("max_new_tokens", parameters["max_new_tokens"])
    profile = with_overrides(resolve_profile(registry, parameters["profile"], model.source), overrides)
    grader = registry.resolve_ref("grader", parameters["grader"])
    if text_model is None:
        if files is None:
            raise _invalid("generation needs the model's stored tokenizer files (Engine.model_files)")
        text_model = load_text_model(model, files, device=device)
    items = suite.data.items[:count]
    rows = [(index, sample) for index in range(count) for sample in range(parameters["samples_per_problem"])]
    start = time.perf_counter()
    outputs = generate_samples(text_model, profile, items, rows, eval_seed=eval_config.seed, batch_size=batch_size,
                               check=check)
    by_problem: dict[int, list[SampleResult]] = {}
    for output in outputs:
        by_problem.setdefault(output.index, []).append(_grade(grader, output, items[output.index].reference,
                                                              profile.max_new_tokens))
    elapsed = time.perf_counter() - start
    problems = tuple(ProblemResult(index=index, item_id=items[index].item_id,
                                   samples=sorted(by_problem[index], key=lambda s: s.sample))
                     for index in range(count))
    metrics, intervals, clusters = generation_metrics(problems)
    samples, cluster_ids = _flatten(problems)
    runtime = {**text_model.runtime.info(), **_library_versions()}
    execution = ExecutionConfig(evaluator=suite.evaluator,
                                parameters={**parameters, "profile_hash": profile.profile_hash},
                                eval_config=eval_config, batch_size=batch_size, device=runtime.get("device", "cpu"))
    batches = -(-len(rows) // batch_size)
    steps = sum(max(len(o.tokens) for o in outputs[b * batch_size : (b + 1) * batch_size]) for b in range(batches))
    usage = Usage(evaluation_batches=batches, evaluated_examples=len(rows), model_forward_calls=steps,
                  compute_seconds=round(elapsed, 6), analysis_calls=0, tensor_read_calls=0, candidate_commits=0,
                  patch_previews=0, artifact_bytes=0, peak_memory_bytes=None,
                  unavailable={"peak_memory_bytes": "measured only for whole evaluation processes"})
    return EvaluationResult(
        metrics=MappingProxyType(metrics),
        sample_count=count,
        model=model.identity,
        evaluator=suite.evaluator,
        evaluator_config_hash=execution.config_hash,
        data_hash=suite.data_hash,
        usage=usage,
        split=suite.split,
        suite_id=suite.suite_id,
        execution=execution,
        clusters=clusters,
        intervals=MappingProxyType(intervals),
        scores=tuple(int(s.correct) for s in samples),
        cluster_ids=tuple(cluster_ids),
        errors=(),
        truncation_count=sum(int(s.limit_hit) for s in samples),
        elapsed_seconds=round(elapsed, 6),
        problems=problems,
        runtime=MappingProxyType(runtime),
    )


def compare_generation(base: Any, candidate: Any) -> GenerationComparison:
    """Paired differences of two generation evaluations (results or reports) of one suite and configuration."""
    mismatched = [name for name in ("suite_id", "data_hash", "evaluator_config_hash", "sample_count")
                  if getattr(base, name) != getattr(candidate, name)]
    if mismatched:
        raise WSEError(ErrorCode.INCOMPATIBLE_REPORTS,
                       f"the evaluations differ in {mismatched}; compare only evaluations of the same suite, data, "
                       "and configuration", details={"mismatched": mismatched})
    if base.problems is None or candidate.problems is None:
        raise WSEError(ErrorCode.INCOMPATIBLE_REPORTS, "both evaluations need per-problem results")
    keys = [[(p.item_id, s.sample, s.seed) for s in p.samples] for p in base.problems]
    if keys != [[(p.item_id, s.sample, s.seed) for s in p.samples] for p in candidate.problems]:
        raise WSEError(ErrorCode.INCOMPATIBLE_REPORTS, "the evaluations cover different problems, samples, or seeds",
                       details={"mismatched": ["problems"]})
    base_samples, clusters = _flatten(base.problems)
    candidate_samples, _ = _flatten(candidate.problems)
    differences = {}
    for name, field in (("accuracy", "correct"), ("limit_hit_rate", "limit_hit")):
        paired = paired_comparison([int(getattr(s, field)) for s in base_samples],
                                   [int(getattr(s, field)) for s in candidate_samples], clusters, CLUSTER_UNIT)
        differences[name] = paired["difference_interval"]
    differences["mean_generated_tokens"] = paired_mean_difference(
        [float(s.generated_tokens) for s in base_samples], [float(s.generated_tokens) for s in candidate_samples],
        clusters)
    names = ("accuracy", "limit_hit_rate", "mean_generated_tokens")
    return GenerationComparison(
        suite_id=base.suite_id, data_hash=base.data_hash, evaluator_config_hash=base.evaluator_config_hash,
        base_model=base.model, candidate_model=candidate.model, problems=len(base.problems),
        samples=len(base_samples), base={name: float(base.metrics[name]) for name in names},
        candidate={name: float(candidate.metrics[name]) for name in names}, differences=differences)


__all__ = ["GenerationEvaluator", "SampleOutput", "TextModel", "compare_generation", "evaluate_generation",
           "generate_samples", "generation_metrics", "load_text_model", "mean_interval", "model_stop_tokens",
           "paired_mean_difference", "suite_parameters"]
