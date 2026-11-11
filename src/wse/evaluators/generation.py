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
  profile's hash and ``prompts_hash`` (over every evaluated problem's prompt
  token IDs), so a changed profile, grader, limit, or rendered prompt is a
  different configuration. ``runtime`` records the engine's device, stored dtypes, and
  library versions.
* **Per-problem results.** The hash of the problem's prompt tokens, and
  every sample's seed, grade, read answer, length, and limit hit, plus a hash
  of its tokens (``ProblemResult``).

``compare_generation`` pairs two evaluations of one suite and configuration.

* **Generation passes.** In a managed experiment, a development evaluation of
  a generation suite is a generation pass, a job of M1's job system that the
  worker meters in generated tokens (``wse.worker``). Its size, known before
  anything runs, is ``generation_pass_size``; the worker calls
  ``run_generation`` with ``on_step`` (called at every decoding step, so the
  pass stops at the next step on cancellation or its time limit) and
  ``on_batch``. ``GenerationEvaluator.evaluate``, the M1 metered-forward entry
  point, refuses generation suites.
* **Tails.** The report of a pass carries each output's ``tail``: the longest
  end of its decoded text whose canonical JSON takes at most ``TAIL_BYTES``
  bytes. ``report_with_tails`` shortens every tail evenly when the report
  would otherwise exceed the inline-result limit, so that a report of 32
  single-sample problems fits it with full tails unless the answers are long
  (``tests/test_llm_policy_mcp.py`` checks the sizes).

``transformers`` and ``math_verify`` are imported lazily.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
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
    EvaluationReport,
    EvaluationSuite,
    ExecutionConfig,
    ExtensionRef,
    GenerationComparison,
    GenerationProfile,
    IntervalEstimate,
    ModelSource,
    ProblemResult,
    SampleResult,
    TextItem,
    TextSuiteData,
    Usage,
    WSEError,
    canonical_json,
)

if TYPE_CHECKING:
    from wse.api import EvaluationResult, ModelHandle
    from wse.registry import Registry

EVALUATOR_ID = "generation"
EVALUATOR_VERSION = "1"
PARAMETERS = ("profile", "grader", "samples_per_problem", "max_new_tokens")
MAX_SAMPLES_PER_PROBLEM = 64
# The batch size of generation passes and of final generation evaluations (Task 7: throughput on MPS rises up to
# about 8 rows and little beyond).
DEFAULT_BATCH_SIZE = 8
CLUSTER_UNIT = "problem"
# The largest tail excerpt of one output in a pass's report, in bytes of canonical JSON (escapes included). With
# it, a report of 32 single-sample problems whose answers are short fits the 31 KiB inline-result limit.
TAIL_BYTES = 256

# Called at every decoding step of a batch with the batch's row count and the step's number: 0 when the batch
# starts, then 1, 2, ... before each step's tokens are chosen. It may raise to stop the generation.
StepHook = Callable[[int, int], None]


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
    """One generated row: its problem index, sample index, seed, prompt hash, tokens (stop token included), and text."""

    index: int
    sample: int
    item_id: str
    seed: int
    tokens: list[int]
    finished: bool
    text: str
    prompt_sha256: str


def token_hash(tokens: Sequence[int]) -> str:
    """``sha256:`` of token IDs as little-endian int64: the identity of a prompt or an output."""
    return "sha256:" + hashlib.sha256(np.asarray(list(tokens), dtype="<i8").tobytes()).hexdigest()


def prompts_hash(prompt_hashes: Sequence[str]) -> str:
    """The hash of an evaluation's prompts: SHA-256 over the per-problem prompt hashes, in problem order."""
    return "sha256:" + hashlib.sha256("\n".join(prompt_hashes).encode("ascii")).hexdigest()


def _encode(text_model: TextModel, profile: GenerationProfile, prompt: str) -> list[int]:
    # A chat template writes its own special tokens (for example a BOS), so they are not added twice.
    ids = text_model.tokenizer(prompt, add_special_tokens=not profile.chat_template)["input_ids"]
    if not ids:
        raise _invalid("a prompt rendered to no tokens")
    return [int(token) for token in ids]


def encode_prompts(text_model: TextModel, profile: GenerationProfile, items: Sequence[TextItem],
                   indices: Iterable[int]) -> dict[int, list[int]]:
    """The prompt token IDs of the problems at ``indices``: rendered with the profile, then tokenized."""
    return {index: _encode(text_model, profile, render_prompt(text_model.tokenizer, profile, items[index].problem))
            for index in sorted(set(indices))}


class _StepCounter:
    """A pass-through logits processor that reports every decoding step of one batch to a ``StepHook``.

    The engine calls its processors once per step, before choosing the
    step's tokens; this one returns the scores unchanged, so the generated
    tokens are those of a generation without it.
    """

    def __init__(self, hook: StepHook, rows: int) -> None:
        self.hook, self.rows, self.steps = hook, rows, 0

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        self.steps += 1
        self.hook(self.rows, self.steps)
        return scores


def generate_samples(text_model: TextModel, profile: GenerationProfile, items: Sequence[TextItem],
                     rows: Sequence[tuple[int, int]], *, eval_seed: int, batch_size: int = DEFAULT_BATCH_SIZE,
                     check: Callable[[], None] | None = None, prompts: Mapping[int, list[int]] | None = None,
                     on_batch: Callable[[list[SampleOutput], float], None] | None = None,
                     on_step: StepHook | None = None) -> list[SampleOutput]:
    """Generate the ``(problem index, sample index)`` rows in batches of ``batch_size``, in the order given.

    Prompts are left-padded and masked; each row draws from its own seed, so
    its tokens do not depend on its batch or position. The presence penalty
    of a batch counts output tokens only (the padded prompt length is shared).
    ``prompts`` supplies already encoded prompts (``encode_prompts``);
    ``on_batch`` receives each finished batch's outputs and its seconds, and
    ``on_step`` every decoding step (``StepHook``), through a pass-through
    logits processor that runs after the profile's.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise _invalid("batch_size must be a positive integer")
    sampling = profile_sampling(profile)
    needed = {index for index, _ in rows}
    if prompts is None or not needed <= set(prompts):
        prompts = encode_prompts(text_model, profile, items, needed)
    prompt_hashes = {index: token_hash(prompts[index]) for index in needed}
    outputs: list[SampleOutput] = []
    for start in range(0, len(rows), batch_size):
        if check is not None:
            check()
        began = time.perf_counter()
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
        processors: list[Any] = list(logits_processors(profile, width))
        if on_step is not None:
            on_step(len(chunk), 0)
            processors.append(_StepCounter(on_step, len(chunk)))
        result = text_model.runtime.generate(
            input_ids, mask, max_new_tokens=profile.max_new_tokens, stop_token_ids=stops,
            pad_token_id=text_model.pad_token_id, sampling=sampling, seeds=seeds if sampling is not None else None,
            logits_processors=processors)
        batch_outputs = []
        for (index, sample), seed, tokens, finished in zip(chunk, seeds, result.tokens, result.finished, strict=True):
            body = tokens[:-1] if finished and tokens else tokens  # the text excludes the stop token
            text = text_model.tokenizer.decode(body, skip_special_tokens=True)
            batch_outputs.append(SampleOutput(index=index, sample=sample, item_id=items[index].item_id, seed=seed,
                                              tokens=list(tokens), finished=bool(finished), text=text,
                                              prompt_sha256=prompt_hashes[index]))
        outputs.extend(batch_outputs)
        if on_batch is not None:
            on_batch(batch_outputs, time.perf_counter() - began)
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
    """``generation@1``: the registered evaluator of generation suites.

    ``generates_text`` marks an evaluator whose suites the worker runs as
    generation passes, metered in generated tokens, instead of through M1's
    metered forward callable.
    """

    evaluator_id = EVALUATOR_ID
    version = EVALUATOR_VERSION
    generates_text = True

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
                       "generation suites are not evaluated through a metered forward callable: in an experiment they "
                       "run as generation passes (Engine.submit_evaluation), and in direct use through "
                       "evaluate_generation")


def generates_text(evaluator: Any) -> bool:
    """Whether a registered evaluator's suites run as generation passes (``GenerationEvaluator.generates_text``)."""
    return getattr(evaluator, "generates_text", False) is True


def _grade(grader: Any, output: SampleOutput, reference: str, max_new_tokens: int) -> SampleResult:
    graded = grader.grade(output.text, reference)
    return SampleResult(sample=output.sample, seed=output.seed, correct=bool(graded.correct), answer=graded.answer,
                        generated_tokens=len(output.tokens), finished=output.finished,
                        limit_hit=not output.finished and len(output.tokens) >= max_new_tokens,
                        output_sha256=token_hash(output.tokens))


def library_versions() -> dict[str, str]:
    """The versions of the grading libraries, which the results report next to the runtime's torch and transformers."""
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for package in ("math-verify",):
        try:
            versions[package.replace("-", "_")] = version(package)
        except PackageNotFoundError:
            pass
    return versions


@dataclass(frozen=True)
class GenerationPlan:
    """A generation suite resolved for one evaluation: its parameters, effective profile, grader, problems, and rows.

    ``items`` are the evaluated problems (the suite's first ``sample_count``)
    and ``rows`` every ``(problem index, sample index)`` pair, in order.
    """

    suite: EvaluationSuite
    eval_config: EvalConfig
    parameters: Mapping[str, Any]
    profile: GenerationProfile
    grader: Any
    items: tuple[TextItem, ...]
    rows: tuple[tuple[int, int], ...]

    @property
    def samples_per_problem(self) -> int:
        return int(self.parameters["samples_per_problem"])


def effective_profile(parameters: Mapping[str, Any], source: ModelSource | None, registry: Registry,
                      profile_overrides: Mapping[str, Any] | None = None) -> GenerationProfile:
    """The profile a suite's checked parameters name, resolved for a model's source, with the suite's token limit."""
    overrides = dict(profile_overrides or {})
    if parameters.get("max_new_tokens") is not None:
        overrides.setdefault("max_new_tokens", parameters["max_new_tokens"])
    return with_overrides(resolve_profile(registry, parameters["profile"], source), overrides)


@dataclass(frozen=True)
class GenerationPassSize:
    """The size of one generation pass, known before anything runs: problems, samples of each, and the token limit."""

    problems: int
    samples_per_problem: int
    max_new_tokens: int

    @property
    def outputs(self) -> int:
        return self.problems * self.samples_per_problem

    @property
    def max_generated_tokens(self) -> int:
        """The most tokens the pass can generate: every output at the limit (problems x samples x max_new_tokens)."""
        return self.outputs * self.max_new_tokens


def generation_pass_size(suite: EvaluationSuite, config: EvalConfig, *, source: ModelSource | None,
                         registry: Registry) -> GenerationPassSize:
    """The size of a pass of ``suite`` under ``config`` for a model with ``source``; reads no problem.

    Submission uses it to check a pass against the policy and reserve its
    tokens without reaching the problems, which stay in evaluator storage.
    """
    parameters = suite_parameters(suite)
    profile = effective_profile(parameters, source, registry)
    return GenerationPassSize(problems=config.sample_count, samples_per_problem=int(parameters["samples_per_problem"]),
                              max_new_tokens=profile.max_new_tokens)


def plan_generation(model: ModelHandle, suite: EvaluationSuite, config: EvalConfig | Mapping[str, Any], *,
                    registry: Registry | None = None,
                    profile_overrides: Mapping[str, Any] | None = None) -> GenerationPlan:
    """Check a generation suite with its problems attached and resolve the profile and grader for ``model``."""
    from wse.api import _as_eval_config
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
    profile = effective_profile(parameters, model.source, registry, profile_overrides)
    grader = registry.resolve_ref("grader", parameters["grader"])
    rows = tuple((index, sample) for index in range(count) for sample in range(parameters["samples_per_problem"]))
    return GenerationPlan(suite=suite, eval_config=eval_config, parameters=MappingProxyType(parameters),
                          profile=profile, grader=grader, items=tuple(suite.data.items[:count]), rows=rows)


def run_problems(plan: GenerationPlan, text_model: TextModel, indices: Iterable[int], *,
                 batch_size: int = DEFAULT_BATCH_SIZE, check: Callable[[], None] | None = None,
                 prompts: Mapping[int, list[int]] | None = None,
                 on_problem: Callable[[ProblemResult, list[SampleOutput]], None] | None = None,
                 on_batch: Callable[[list[SampleOutput], float], None] | None = None,
                 on_step: StepHook | None = None,
                 ) -> tuple[dict[int, ProblemResult], list[SampleOutput]]:
    """Generate and grade every sample of the problems at ``indices``, in plan order and batches of ``batch_size``.

    ``on_problem`` receives each problem's result and outputs as soon as its
    last sample is graded, so a caller can record finished problems while
    later batches still run. ``on_batch`` and ``on_step`` are passed to
    ``generate_samples``; ``on_batch`` runs after the batch is graded.
    """
    wanted = set(indices)
    unknown = sorted(wanted - set(range(len(plan.items))))
    if unknown:
        raise _invalid(f"problem indices {unknown[:10]} are outside the plan's {len(plan.items)} problems")
    rows = [row for row in plan.rows if row[0] in wanted]
    graded: dict[int, list[SampleResult]] = {}
    pending: dict[int, list[SampleOutput]] = {}
    problems: dict[int, ProblemResult] = {}

    def finish_batch(outputs: list[SampleOutput], seconds: float) -> None:
        for output in outputs:
            graded.setdefault(output.index, []).append(
                _grade(plan.grader, output, plan.items[output.index].reference, plan.profile.max_new_tokens))
            pending.setdefault(output.index, []).append(output)
        for index in dict.fromkeys(output.index for output in outputs):
            if len(graded[index]) == plan.samples_per_problem:
                done = sorted(pending.pop(index), key=lambda o: o.sample)
                problems[index] = ProblemResult(index=index, item_id=plan.items[index].item_id,
                                                prompt_sha256=done[0].prompt_sha256,
                                                samples=sorted(graded.pop(index), key=lambda s: s.sample))
                if on_problem is not None:
                    on_problem(problems[index], done)
        if on_batch is not None:
            on_batch(outputs, seconds)

    outputs = generate_samples(text_model, plan.profile, plan.items, rows, eval_seed=plan.eval_config.seed,
                               batch_size=batch_size, check=check, prompts=prompts, on_batch=finish_batch,
                               on_step=on_step)
    return problems, outputs


def generation_result(model: ModelHandle, plan: GenerationPlan, problems: Mapping[int, ProblemResult], *,
                      runtime: Mapping[str, Any], batch_size: int, elapsed: float, usage: Usage) -> EvaluationResult:
    """The evaluation result of a plan from the results of all its problems: metrics, intervals, and identities."""
    from wse.api import EvaluationResult

    count = len(plan.items)
    missing = [index for index in range(count) if index not in problems]
    if missing:
        raise _invalid(f"results are missing for problems {missing[:10]}")
    ordered = tuple(problems[index] for index in range(count))
    for problem in ordered:
        if problem.item_id != plan.items[problem.index].item_id or len(problem.samples) != plan.samples_per_problem:
            raise _invalid(f"the result of problem {problem.index} does not match the plan")
    metrics, intervals, clusters = generation_metrics(ordered)
    samples, cluster_ids = _flatten(ordered)
    runtime = {**runtime, **library_versions()}
    execution = ExecutionConfig(evaluator=plan.suite.evaluator,
                                parameters={**plan.parameters, "profile_hash": plan.profile.profile_hash,
                                            "prompts_hash": prompts_hash([p.prompt_sha256 for p in ordered])},
                                eval_config=plan.eval_config, batch_size=batch_size,
                                device=runtime.get("device", "cpu"))
    return EvaluationResult(
        metrics=MappingProxyType(metrics),
        sample_count=count,
        model=model.identity,
        evaluator=plan.suite.evaluator,
        evaluator_config_hash=execution.config_hash,
        data_hash=plan.suite.data_hash,
        usage=usage,
        split=plan.suite.split,
        suite_id=plan.suite.suite_id,
        execution=execution,
        clusters=clusters,
        intervals=MappingProxyType(intervals),
        scores=tuple(int(s.correct) for s in samples),
        cluster_ids=tuple(cluster_ids),
        errors=(),
        truncation_count=sum(int(s.limit_hit) for s in samples),
        elapsed_seconds=round(elapsed, 6),
        problems=ordered,
        runtime=MappingProxyType(runtime),
    )


def generation_usage(rows: int, batches: int, steps: int, seconds: float) -> Usage:
    """The usage of a generation evaluation: rows, batches, decoding steps (forward calls), and seconds."""
    return Usage(evaluation_batches=batches, evaluated_examples=rows, model_forward_calls=steps,
                 compute_seconds=round(seconds, 6), analysis_calls=0, tensor_read_calls=0, candidate_commits=0,
                 patch_previews=0, artifact_bytes=0, peak_memory_bytes=None,
                 unavailable={"peak_memory_bytes": "measured only for whole evaluation processes"})


@dataclass(frozen=True)
class GenerationRun:
    """A direct generation evaluation: its plan, its result, and every sample's output, in plan order."""

    plan: GenerationPlan
    result: EvaluationResult
    outputs: tuple[SampleOutput, ...]


def run_generation(
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
    on_problem: Callable[[ProblemResult, list[SampleOutput]], None] | None = None,
    on_batch: Callable[[list[SampleOutput], float], None] | None = None,
    on_step: StepHook | None = None,
) -> GenerationRun:
    """``evaluate_generation``, also returning the plan and every sample's output (token IDs and text).

    ``on_problem`` receives each problem's result and outputs as soon as it
    is graded, ``on_batch`` each graded batch's outputs, and ``on_step`` every
    decoding step (``run_problems``); the managed worker meters a generation
    pass through them.
    """
    plan = plan_generation(model, suite, config, registry=registry, profile_overrides=profile_overrides)
    if text_model is None:
        if files is None:
            raise _invalid("generation needs the model's stored tokenizer files (Engine.model_files)")
        text_model = load_text_model(model, files, device=device)
    start = time.perf_counter()
    problems, outputs = run_problems(plan, text_model, range(len(plan.items)), batch_size=batch_size, check=check,
                                     on_problem=on_problem, on_batch=on_batch, on_step=on_step)
    elapsed = time.perf_counter() - start
    batches = -(-len(plan.rows) // batch_size)
    steps = sum(max(len(o.tokens) for o in outputs[b * batch_size : (b + 1) * batch_size]) for b in range(batches))
    result = generation_result(model, plan, problems, runtime=text_model.runtime.info(), batch_size=batch_size,
                               elapsed=elapsed, usage=generation_usage(len(plan.rows), batches, steps, elapsed))
    return GenerationRun(plan=plan, result=result, outputs=tuple(outputs))


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
    the configuration either way. The steps are ``plan_generation``,
    ``run_problems``, and ``generation_result``, which resumable runners
    (``wse.evaluators.generation_runs``) call in parts; ``run_generation``
    also returns the outputs.
    """
    return run_generation(model, suite, config, files=files, text_model=text_model, registry=registry,
                          batch_size=batch_size, device=device, profile_overrides=profile_overrides,
                          check=check).result


# ---------------------------------------------------------------------------
# Tail excerpts of a pass's outputs
# ---------------------------------------------------------------------------


def json_string_bytes(text: str) -> int:
    """The bytes ``text`` takes inside canonical JSON: UTF-8, with JSON's escapes, without the quotes."""
    return len(json.dumps(text, ensure_ascii=False).encode("utf-8")) - 2


def tail_excerpt(text: str, max_bytes: int) -> str:
    """The longest end of ``text`` that takes at most ``max_bytes`` bytes inside canonical JSON."""
    if max_bytes <= 0:
        return ""
    if json_string_bytes(text) <= max_bytes:
        return text
    used, start = 0, len(text)
    while start > 0:
        cost = json_string_bytes(text[start - 1])
        if used + cost > max_bytes:
            break
        used, start = used + cost, start - 1
    return text[start:]


def with_tails(problems: Sequence[ProblemResult], texts: Mapping[tuple[int, int], str],
               max_bytes: int) -> list[ProblemResult]:
    """``problems`` with each sample's ``tail``: the end of its text (``texts`` by problem index and sample)."""
    return [problem.model_copy(update={"samples": [
        sample.model_copy(update={"tail": tail_excerpt(texts[(problem.index, sample.sample)], max_bytes)})
        for sample in problem.samples]}) for problem in problems]


def report_with_tails(report: EvaluationReport, texts: Mapping[tuple[int, int], str], *, limit: int,
                      max_bytes: int = TAIL_BYTES) -> EvaluationReport:
    """``report`` with every sample's tail, each at most ``max_bytes`` bytes, the report within ``limit`` bytes.

    ``texts`` maps ``(problem index, sample)`` to the output's decoded text.
    When the report with full tails would exceed ``limit`` bytes of canonical
    JSON, every tail is shortened by the same number of bytes, as far as to
    empty; a report that exceeds ``limit`` even then (very long answers or many
    samples) keeps empty tails and is read in pages.
    """
    if report.problems is None:
        raise _invalid("tails belong to the per-problem results of a development report")
    samples = sum(len(problem.samples) for problem in report.problems)
    budget = max(0, max_bytes)
    while True:
        tailed = report.model_copy(update={"problems": with_tails(report.problems, texts, budget)})
        size = len(canonical_json(tailed.model_dump(mode="json")).encode("utf-8"))
        if size <= limit or budget == 0 or samples == 0:
            return tailed
        budget = max(0, budget - max(1, -(-(size - limit) // samples)))


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
    keys = [[(p.item_id, p.prompt_sha256, s.sample, s.seed) for s in p.samples] for p in base.problems]
    if keys != [[(p.item_id, p.prompt_sha256, s.sample, s.seed) for s in p.samples] for p in candidate.problems]:
        raise WSEError(ErrorCode.INCOMPATIBLE_REPORTS,
                       "the evaluations cover different problems, prompts, samples, or seeds",
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


__all__ = ["DEFAULT_BATCH_SIZE", "GenerationEvaluator", "GenerationPassSize", "GenerationPlan", "GenerationRun",
           "SampleOutput", "StepHook", "TAIL_BYTES", "TextModel", "compare_generation", "effective_profile",
           "encode_prompts", "evaluate_generation", "generate_samples", "generates_text", "generation_metrics",
           "generation_pass_size", "generation_result", "generation_usage", "json_string_bytes", "library_versions",
           "load_text_model", "mean_interval", "model_stop_tokens", "paired_mean_difference", "plan_generation",
           "prompts_hash", "report_with_tails", "run_generation", "run_problems", "suite_parameters",
           "tail_excerpt", "token_hash", "with_tails"]
