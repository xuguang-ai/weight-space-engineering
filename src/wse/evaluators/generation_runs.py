"""Resumable paired generation runs: a base and a candidate checkpoint on a seeded problem subset (M2 plan Task 7).

An operator tool for long local comparisons such as base versus p50c on
MATH-500 with ``qwen35think@1``, where one sample can take minutes:

* **Problems.** ``count`` problems of a JSON Lines dataset (read with
  ``wse.evaluators.text_suites.read_jsonl``), chosen by ``select_subset``:
  the problems with the smallest SHA-256 of ``{"item_id", "seed"}``, kept in
  file order. The subset is an in-memory suite (``subset_suite``) whose
  problems are all evaluated; its data hash, the file's SHA-256, the seed,
  and the chosen item IDs are recorded.
* **The evaluator path.** Each checkpoint's evaluation is
  ``generation@1``'s: ``plan_generation``, ``run_problems`` (prompt
  rendering, per-row seeds, batching, grading), and ``generation_result``
  (metrics, intervals, identities), then ``compare_generation``. A problem's
  samples do not depend on the batch they ran in (per-row seeds), so the
  runner can work through the problems in chunks of ``batch_size``
  problems, alternating checkpoints, and still produce the result a single
  evaluation with the same ``batch_size`` and device would. A chunk's
  ``batch_size × samples_per_problem`` rows run as generation batches of
  ``batch_size`` rows, aligned with a single evaluation's batches.
* **Identical prompts.** Both checkpoints must render the same prompt for
  every problem (a paired comparison needs it, and ``compare_generation``
  refuses otherwise). The prompt hashes of the two sides are compared
  before anything is generated or recorded; the first differing problem is
  named.
* **Resumption.** ``run.json`` in the output directory fixes the
  configuration, including the installed versions of the libraries that
  determine the outputs (torch, transformers, and the grading libraries of
  ``library_versions``, as the results report them). A restart with any
  other configuration is refused, and a version change names each differing
  library with its recorded and current version. A run whose ``run.json``
  records no library versions is not continued: its recorded results stay
  valid as recorded, and further generation needs a new output directory.
  These refusals come before any model is loaded or any file is written.
  ``problems.jsonl`` gets one line per finished (checkpoint, problem), with
  the problem's result, the output texts, and its chunk's ``batch`` ID,
  seconds (``batch_seconds``), longest output (``batch_steps``), generation
  batches (``generation_batches``), and forward calls (``forward_calls``,
  the sum of each generation batch's longest output), flushed and synced
  when written. A restart skips recorded problems after checking that their
  prompt hashes and seeds are those the current configuration produces. A
  last line that is not valid JSON (an interrupted write) is dropped; a
  complete last record without its newline is kept and terminated before
  anything is appended. Lines are written when a chunk ends, so an
  interruption loses at most the chunk in progress; the unrecorded problems
  of a partly recorded chunk run again in a smaller chunk.
* **Summary.** When every problem of both checkpoints is recorded,
  ``summary.json`` holds each checkpoint's split, errors, metrics with
  intervals, configuration hash, identities, usage, and problems, and the
  paired comparison. Its usage counts the generation batches run and their
  forward calls, so an uninterrupted run reports the usage a single
  evaluation would. It has no ``report_id`` and no peak memory: the runner
  is an operator tool, not a managed evaluation.

The output directory belongs outside the repository. This module imports
neither ``transformers`` nor ``math_verify`` itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wse.evaluators.generation import (
    EVALUATOR_ID,
    EVALUATOR_VERSION,
    GenerationPlan,
    compare_generation,
    encode_prompts,
    generation_result,
    generation_usage,
    library_versions,
    load_text_model,
    plan_generation,
    run_problems,
    token_hash,
)
from wse.evaluators.sampling import sample_seed
from wse.schemas import (
    ErrorCode,
    EvaluationSuite,
    ProblemResult,
    TextSuiteData,
    WSEError,
    canonical_json,
)

if TYPE_CHECKING:
    from wse.api import Engine, EvaluationResult

RUN_FORMAT = "wse-generation-comparison-v1"
SUBSET_METHOD = "sha256_rank@1"
LABELS = ("base", "candidate")
RUN_FILE = "run.json"
PROBLEMS_FILE = "problems.jsonl"
SUMMARY_FILE = "summary.json"


def _invalid(message: str, **details: Any) -> WSEError:
    return WSEError(ErrorCode.INVALID_ARGUMENT, message, details=details)


def select_subset(item_ids: Sequence[str], count: int, seed: int) -> list[int]:
    """Indices of ``count`` problems chosen by ``seed``: the smallest SHA-256 ranks, returned in file order."""
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(item_ids):
        raise _invalid(f"the subset size must lie between 1 and the {len(item_ids)} problems")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise _invalid("the subset seed must be a nonnegative integer")

    def rank(index: int) -> str:
        return hashlib.sha256(canonical_json({"item_id": item_ids[index], "seed": seed}).encode("utf-8")).hexdigest()

    return sorted(sorted(range(len(item_ids)), key=rank)[:count])


def subset_suite(data: TextSuiteData, *, count: int, seed: int, parameters: Mapping[str, Any], eval_seed: int = 0,
                 suite_id: str | None = None, task_id: str = "generation_comparison",
                 split: str = "final") -> tuple[EvaluationSuite, list[int]]:
    """An in-memory generation suite of ``count`` problems of ``data`` chosen by ``seed``, and their source indices."""
    indices = select_subset([item.item_id for item in data.items], count, seed)
    subset = TextSuiteData(items=[data.items[index] for index in indices])
    suite = EvaluationSuite(
        suite_id=suite_id or f"subset_{count}_seed_{seed}", task_id=task_id, split=split,
        evaluator={"extension_id": EVALUATOR_ID, "version": EVALUATOR_VERSION}, parameters=dict(parameters),
        sample_counts=[count], seeds=[eval_seed], data=subset)
    return suite, indices


# ---------------------------------------------------------------------------
# The log
# ---------------------------------------------------------------------------


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_log(path: Path) -> list[dict[str, Any]]:
    """The recorded lines of ``problems.jsonl``, with the file left ready for the next append.

    A last line that is not valid JSON (an interrupted write) is dropped and
    truncated away. A last line that is a complete record without its
    newline is kept, and the newline is written, so that the next append
    starts a line of its own. Invalid JSON before the last record is
    ``ARTIFACT_CORRUPT``.
    """
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines, kept = raw.split(b"\n"), 0
    records = []
    for number, line in enumerate(lines):
        if not line.strip():
            kept += len(line) + 1
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            if number < len(lines) - 1 and any(rest.strip() for rest in lines[number + 1:]):
                raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"{path.name} line {number + 1} is not valid JSON") \
                    from None
            with path.open("r+b") as handle:  # an interrupted last write: drop it
                handle.truncate(kept)
            break
        kept += len(line) + 1
    else:
        if lines[-1].strip():  # a complete last record whose newline was not written
            with path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
    return records


def _append(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def _run_libraries() -> dict[str, str]:
    """The installed versions of the libraries that determine a run's outputs, under the names the results report.

    These are torch and transformers (generation, reported by the runtime)
    and the grading libraries (``library_versions``).
    """
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for package in ("torch", "transformers"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    return {**versions, **library_versions()}


def _check_configuration(out: Path, recorded: Any, configuration: Mapping[str, Any]) -> None:
    """Refuse to continue a run recorded under another configuration, other library versions, or no versions."""
    if not isinstance(recorded, dict):
        raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"{out / RUN_FILE} holds no run configuration")
    if "libraries" not in recorded:
        raise _invalid(f"{out} holds a run recorded without library versions; its recorded results stay valid as "
                       "recorded, but further generation needs a new output directory")
    differing = sorted(key for key in set(recorded) | set(configuration) if recorded.get(key) != configuration.get(key))
    if not differing:
        return
    message, details = f"{out} holds a run with another configuration (differs in {differing})", {}
    if "libraries" in differing:
        old, new = recorded["libraries"], configuration["libraries"]
        libraries = {name: {"recorded": old.get(name), "current": new.get(name)}
                     for name in sorted(set(old) | set(new)) if old.get(name) != new.get(name)}
        message += "; library versions: " + ", ".join(
            f"{name} {versions['recorded'] or 'absent'} recorded, {versions['current'] or 'absent'} now"
            for name, versions in libraries.items())
        details["libraries"] = libraries
    raise _invalid(f"{message}; use a new output directory", differing=differing, **details)


def _result_summary(result: EvaluationResult) -> dict[str, Any]:
    """A checkpoint's result in ``summary.json``.

    It holds the metrics, intervals, identities, configuration hash, split,
    errors, usage, and problems of the ``EvaluationResult``. There is no
    ``report_id`` and no peak memory: the runner is an operator tool, not a
    managed evaluation, so it stores no report and does not measure a whole
    evaluation process.
    """
    return {
        "split": result.split,
        "errors": list(result.errors),
        "metrics": dict(result.metrics),
        "intervals": {name: interval.model_dump(mode="json") for name, interval in result.intervals.items()},
        "sample_count": result.sample_count,
        "truncation_count": result.truncation_count,
        "evaluator_config_hash": result.evaluator_config_hash,
        "data_hash": result.data_hash,
        "suite_id": result.suite_id,
        "model": result.model.model_dump(mode="json"),
        "execution": result.execution.model_dump(mode="json"),
        "runtime": dict(result.runtime or {}),
        "usage": result.usage.model_dump(mode="json"),
        "problems": [problem.model_dump(mode="json") for problem in result.problems or ()],
    }


def _check_recorded(record: Mapping[str, Any], plan: GenerationPlan, prompt_hash: str) -> ProblemResult:
    problem = ProblemResult.model_validate(record["problem"])
    index = problem.index
    expected_seeds = [sample_seed(plan.profile.seed, plan.eval_config.seed, plan.items[index].item_id, sample)
                      for sample in range(plan.samples_per_problem)]
    if problem.item_id != plan.items[index].item_id or problem.prompt_sha256 != prompt_hash \
            or [s.seed for s in problem.samples] != expected_seeds:
        raise _invalid(f"the recorded result of problem {index} ({problem.item_id}) does not match this "
                       "configuration (prompt or seeds differ); use a new output directory",
                       item_id=problem.item_id)
    return problem


def run_comparison(
    engine: Engine,
    *,
    base: str,
    candidate: str,
    dataset: str | Path,
    format: str,
    count: int,
    subset_seed: int,
    profile: str,
    grader: str,
    out_dir: str | Path,
    samples_per_problem: int = 1,
    max_new_tokens: int | None = None,
    eval_seed: int = 0,
    device: str | None = None,
    batch_size: int = 1,
    id_prefix: str | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate ``base`` and ``candidate`` on a seeded subset, resumably; return (and write) the summary.

    Operator only. Problems are processed in chunks of ``batch_size``
    problems: the base's rows of a chunk, then the candidate's, each in
    generation batches of ``batch_size`` rows. ``log`` receives one progress
    line per finished problem.
    """
    from wse.evaluators.text_suites import read_jsonl

    from wse.engines.hf_torch import select_device

    out = Path(out_dir)
    device = select_device(device).type
    data, manifest = read_jsonl(dataset, format=format, id_prefix=id_prefix)
    parameters: dict[str, Any] = {"profile": profile, "grader": grader, "samples_per_problem": samples_per_problem}
    if max_new_tokens is not None:
        parameters["max_new_tokens"] = max_new_tokens
    suite, indices = subset_suite(data, count=count, seed=subset_seed, parameters=parameters, eval_seed=eval_seed)
    checkpoints = {"base": base, "candidate": candidate}
    handles = {label: engine.open_checkpoint(checkpoint) for label, checkpoint in checkpoints.items()}
    files = {label: engine.model_files(checkpoint) for label, checkpoint in checkpoints.items()}
    plans = {label: plan_generation(handles[label], suite, {"sample_count": count, "seed": eval_seed},
                                    registry=engine.registry) for label in LABELS}
    if plans["base"].profile != plans["candidate"].profile:
        raise _invalid(f"profile {profile} resolves differently for the two checkpoints")
    configuration = {
        "format": RUN_FORMAT,
        "checkpoints": checkpoints,
        "models": {label: handles[label].identity.model_dump(mode="json") for label in LABELS},
        "dataset": {"path_name": Path(dataset).name, "format": format, "id_prefix": id_prefix,
                    "manifest": manifest.model_dump(mode="json")},
        "subset": {"method": SUBSET_METHOD, "seed": subset_seed, "count": count, "source_indices": indices,
                   "item_ids": [item.item_id for item in suite.data.items], "data_hash": suite.data_hash},
        "suite_id": suite.suite_id,
        "parameters": parameters,
        "profile_hash": plans["base"].profile.profile_hash,
        "eval_seed": eval_seed,
        "batch_size": batch_size,
        "device": device,
        "libraries": _run_libraries(),
    }
    run_file, log_file = out / RUN_FILE, out / PROBLEMS_FILE
    exists = run_file.exists()
    if exists:  # checked before any model is loaded or any file is written
        _check_configuration(out, json.loads(run_file.read_text(encoding="utf-8")).get("configuration"),
                             configuration)

    text_models = {label: load_text_model(handles[label], files[label], device=device) for label in LABELS}
    prompts = {label: encode_prompts(text_models[label], plans[label].profile, plans[label].items,
                                     range(count)) for label in LABELS}
    prompt_hashes = {label: {index: token_hash(ids) for index, ids in prompts[label].items()} for label in LABELS}
    for index in range(count):
        if prompt_hashes["base"][index] != prompt_hashes["candidate"][index]:
            item_id = plans["base"].items[index].item_id
            raise _invalid(f"the two checkpoints render different prompts, first for problem {index} ({item_id}); "
                           "a paired comparison needs identical prompts (check the tokenizers and chat templates)",
                           index=index, item_id=item_id)
    if not exists:
        out.mkdir(parents=True, exist_ok=True)
        _write_json(run_file, {"configuration": configuration,
                               "created": datetime.now(UTC).isoformat(timespec="seconds")})
    done: dict[str, dict[int, ProblemResult]] = {label: {} for label in LABELS}
    batches: dict[str, dict[str, dict[str, float]]] = {label: {} for label in LABELS}
    for record in read_log(log_file):
        label = record.get("checkpoint")
        if label not in LABELS or record.get("checkpoint_id") != checkpoints[label]:
            raise _invalid(f"{log_file.name} records another checkpoint ({record.get('checkpoint_id')})")
        index = record.get("problem", {}).get("index")
        if index not in prompt_hashes[label]:
            raise _invalid(f"{log_file.name} records problem {index}, outside this run's {count} problems")
        problem = _check_recorded(record, plans[label], prompt_hashes[label][index])
        if problem.index in done[label]:
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"{log_file.name} records problem {problem.index} of the "
                           f"{label} twice")
        done[label][problem.index] = problem
        batches[label][record["batch"]] = {"seconds": record["batch_seconds"],
                                           "generation_batches": record["generation_batches"],
                                           "forward_calls": record["forward_calls"]}

    for start in range(0, count, batch_size):
        chunk = range(start, min(start + batch_size, count))
        for label in LABELS:
            todo = [index for index in chunk if index not in done[label]]
            if not todo:
                continue
            batch_id = f"{label}-{start}-{time.time_ns()}"
            pending: list[tuple[ProblemResult, list[Any]]] = []

            def record_problem(problem: ProblemResult, outputs: list[Any], *,
                               pending: list[tuple[ProblemResult, list[Any]]] = pending) -> None:
                pending.append((problem, outputs))

            began = time.perf_counter()
            _, outputs = run_problems(plans[label], text_models[label], todo, batch_size=batch_size,
                                      prompts=prompts[label], on_problem=record_problem)
            seconds = time.perf_counter() - began
            # run_problems generates the rows in order, in generation batches of batch_size rows.
            longest = [max(len(output.tokens) for output in outputs[first:first + batch_size])
                       for first in range(0, len(outputs), batch_size)]
            steps = max(longest)
            batches[label][batch_id] = {"seconds": seconds, "generation_batches": len(longest),
                                        "forward_calls": sum(longest)}
            for problem, problem_outputs in pending:
                done[label][problem.index] = problem
                _append(log_file, {
                    "checkpoint": label, "checkpoint_id": checkpoints[label], "problem": problem.model_dump(mode="json"),
                    "outputs": [{"sample": o.sample, "text": o.text, "finished": o.finished} for o in problem_outputs],
                    "batch": batch_id, "batch_seconds": round(seconds, 3), "batch_steps": steps,
                    "generation_batches": len(longest), "forward_calls": sum(longest),
                    "recorded": datetime.now(UTC).isoformat(timespec="seconds")})
                if log is not None:
                    sample = problem.samples[0]
                    log(f"{label} {problem.index + 1}/{count} {problem.item_id}: correct={sample.correct} "
                        f"tokens={sample.generated_tokens} limit_hit={sample.limit_hit} batch_seconds={seconds:.1f}")

    results = {}
    for label in LABELS:
        rows = len(plans[label].rows)
        seconds = sum(batch["seconds"] for batch in batches[label].values())
        generation_batches = int(sum(batch["generation_batches"] for batch in batches[label].values()))
        steps = int(sum(batch["forward_calls"] for batch in batches[label].values()))
        results[label] = generation_result(
            handles[label], plans[label], done[label], runtime=text_models[label].runtime.info(),
            batch_size=batch_size, elapsed=seconds,
            usage=generation_usage(rows, generation_batches, steps, seconds))
    comparison = compare_generation(results["base"], results["candidate"])
    summary = {
        "configuration": configuration,
        "results": {label: _result_summary(results[label]) for label in LABELS},
        "comparison": comparison.model_dump(mode="json"),
        "generated_tokens": {label: sum(s.generated_tokens for p in results[label].problems or () for s in p.samples)
                             for label in LABELS},
        "seconds": {label: round(results[label].elapsed_seconds, 3) for label in LABELS},
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _write_json(out / SUMMARY_FILE, summary)
    return summary


__all__ = ["LABELS", "PROBLEMS_FILE", "RUN_FILE", "SUMMARY_FILE", "read_log", "run_comparison", "select_subset",
           "subset_suite"]
