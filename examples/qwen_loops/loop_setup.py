"""Shared operator setup of the Qwen loop scripts (operator use only; M2 plan Task 12).

``prepare`` brings a workspace to the state both scripts need, idempotently,
so a script can be restarted at any time:

* the model snapshot imported from a local directory (``--snapshot``, with
  ``--repo`` and ``--revision`` for its provenance); importing an identical
  snapshot again returns the existing checkpoint and stores nothing;
* a development generation suite of ``--problems`` GSM8K-train problems chosen
  by ``--subset-seed`` (``sha256_rank@1``: the smallest SHA-256 ranks of the
  item IDs, kept in file order), with profile ``qwen35think@1``, grader
  ``gsm8k_numeric@1``, one sample per problem, and at most
  ``--max-new-tokens`` new tokens (``--profile`` names another profile, for
  example a model's default profile frozen at import); item IDs keep their
  positions in the source file (``gsm8k_train_<line index>``);
* the task whose development suite it is. Suite and task IDs are derived
  from the choices (``gsm8k_train_n32_s0_t8192``,
  ``qwen_loops_gsm8k_train_n32_s0_t8192``; another profile adds its name), so
  other choices register new, immutable ones instead of conflicting with the
  old. No final suite is
  registered: a run needs none, and the final suite (MATH-500) belongs to the
  preregistered trials.

``experiment_summary`` collects everything a report needs about one
experiment from the operator's view: the controller's record and every step,
jobs, reports (with every output's grade, length, and limit hit), the probe
sites, candidates and their patches, paired comparisons with the base, the
selection, and the budget. ``write_json`` writes atomically, so an interrupted
write never leaves a partial file.

Paths default to the operator storage of the M2 plan
(``~/Desktop/wse-models``); nothing is downloaded and nothing is written
there except into the workspace the operator names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

MODELS = Path("~/Desktop/wse-models").expanduser()
DEFAULT_SNAPSHOT = MODELS / "Qwen3.5-0.8B"
DEFAULT_REPO = "Qwen/Qwen3.5-0.8B"
DEFAULT_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
DEFAULT_DATASET = MODELS / "datasets" / "gsm8k-train.jsonl"
PROFILE = "qwen35think@1"
GRADER = "gsm8k_numeric@1"
ID_PREFIX = "gsm8k_train_"
POLICY_ID = "qwen_loop_v1"
LONG_LIST = 16  # summaries shorten lists longer than this (directions, text lists)


def add_setup_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", type=Path, required=True, help="the WSE workspace (created if missing)")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT, help="the local model snapshot directory")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="the snapshot's source repository (provenance)")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="the snapshot's source revision (provenance)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="GSM8K train.jsonl")
    parser.add_argument("--problems", type=int, default=32, help="practice problems (at most 32 under qwen_loop_v1)")
    parser.add_argument("--subset-seed", type=int, default=0, help="the seed that chooses the practice problems")
    parser.add_argument("--max-new-tokens", type=int, default=8192,
                        help="new tokens per output (at most 8,192 under qwen_loop_v1)")
    parser.add_argument("--profile", default=PROFILE, help=f"the generation profile (default {PROFILE})")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps", help="the worker's engine device")


def check_setup_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    from wse.policy import builtin_experiment_policies

    budget = builtin_experiment_policies()[POLICY_ID].budget
    if not 1 <= args.problems <= budget.limit("problems_per_generation_pass"):
        parser.error(f"--problems must lie in [1, {budget.limit('problems_per_generation_pass')}] under {POLICY_ID}")
    if not 1 <= args.max_new_tokens <= budget.limit("new_tokens_per_output"):
        parser.error(f"--max-new-tokens must lie in [1, {budget.limit('new_tokens_per_output')}] under {POLICY_ID}")
    if args.subset_seed < 0:
        parser.error("--subset-seed must be nonnegative")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def prepare(engine: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Import the model and register the practice suite and task (idempotent); return their identities."""
    from wse.evaluators.generation_runs import SUBSET_METHOD, select_subset
    from wse.evaluators.text_suites import read_jsonl
    from wse.schemas import TextSuiteData

    manifest = engine.import_hf_snapshot(args.snapshot.expanduser(), repo=args.repo, revision=args.revision)
    data, _ = read_jsonl(args.dataset.expanduser(), format="gsm8k", id_prefix=ID_PREFIX)
    indices = select_subset([item.item_id for item in data.items], args.problems, args.subset_seed)
    subset = TextSuiteData(items=[data.items[index] for index in indices])
    dataset = engine.register_dataset(subset)
    tag = "" if args.profile == PROFILE else "_" + args.profile.replace("@", "_v")
    suite_id = f"gsm8k_train_n{args.problems}_s{args.subset_seed}_t{args.max_new_tokens}{tag}"
    task_id = f"qwen_loops_{suite_id}"
    engine.register_suite({
        "suite_id": suite_id, "task_id": task_id, "split": "development", "evaluator": "generation@1",
        "parameters": {"profile": args.profile, "grader": GRADER, "samples_per_problem": 1,
                       "max_new_tokens": args.max_new_tokens},
        "sample_counts": [args.problems], "seeds": [0], "dataset": dataset.model_dump(mode="json")})
    engine.register_task({
        "task_id": task_id, "title": "Finish math answers",
        "description": "The model often keeps reasoning until it reaches the token limit, without giving an answer. "
                       "Change its weights so that more of its answers to practice problems finish before the "
                       "limit, without lowering their accuracy.",
        "input_description": "A grade-school math word problem, followed by the instruction to reason step by step "
                             "and put the final answer within \\boxed{}.",
        "output_description": "Step-by-step reasoning, then the final answer within \\boxed{}.",
        "objective": {"task_id": task_id,
                      "description": "Lower the development limit-hit rate without lowering development accuracy.",
                      "gates": [{"metric": "accuracy", "min_delta": 0.0}]},
        "development_suite_id": suite_id, "development_examples": args.problems})
    return {"base_checkpoint_id": manifest.checkpoint_id, "weights_hash": manifest.weights_hash,
            "snapshot": {"path": str(args.snapshot.expanduser()), "repo": args.repo, "revision": args.revision},
            "dataset": {"path": str(args.dataset.expanduser()), "file_sha256": _sha256(args.dataset.expanduser()),
                        "format": "gsm8k", "id_prefix": ID_PREFIX, "items_in_file": len(data.items),
                        "subset": {"method": SUBSET_METHOD, "seed": args.subset_seed, "count": args.problems,
                                   "source_indices": indices, "item_ids": [item.item_id for item in subset.items]},
                        "data_hash": dataset.data_hash},
            "suite_id": suite_id, "task_id": task_id, "policy_id": POLICY_ID,
            "profile": args.profile, "grader": GRADER, "max_new_tokens": args.max_new_tokens,
            "device": args.device}


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _trim(value: Any) -> Any:
    """``value`` with long lists shortened to their length, first entries, and SHA-256 (for readable summaries)."""
    if isinstance(value, Mapping):
        return {key: _trim(item) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) > LONG_LIST:
            digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()
            return {"length": len(value), "first": [_trim(item) for item in value[:3]], "sha256": digest}
        return [_trim(item) for item in value]
    return value


def _steps(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Controller events as steps: each call with its outcome, consecutive polls of one job collapsed."""
    steps: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for event in events:
        kind = event["kind"]
        if kind == "tool_call":
            arguments = dict(event.get("arguments") or {})
            pending = {"seq": event["seq"], "at": event["created_at"], "tool": event.get("tool"),
                       "rationale": (event.get("details") or {}).get("rationale"), "arguments": _trim(arguments)}
            continue
        if kind == "tool_result" and pending is not None:
            pending.update(ok=event.get("ok"), error_code=event.get("error_code"), error=event.get("summary"))
            last = steps[-1] if steps else None
            if pending["tool"] == "wse_get_job" and last is not None and last["tool"] == "wse_get_job" \
                    and last["arguments"].get("job_id") == pending["arguments"].get("job_id") and pending["ok"]:
                last["polls"] = last.get("polls", 1) + 1
                last["last_at"] = pending["at"]
            else:
                steps.append(pending)
            pending = None
            continue
        steps.append({"seq": event["seq"], "at": event["created_at"], "event": kind,
                      **{key: _trim(value) for key, value in event.items()
                         if key not in ("seq", "created_at", "kind")}})
    return steps


def _report_entry(report: Any) -> dict[str, Any]:
    outputs = [{"problem": problem.index, "item_id": problem.item_id, "sample": sample.sample,
                "correct": sample.correct, "answer": sample.answer, "generated_tokens": sample.generated_tokens,
                "finished": sample.finished, "limit_hit": sample.limit_hit, "output_uri": sample.output_uri,
                "tail": (sample.tail or "")[-160:]}
               for problem in report.problems or [] for sample in problem.samples]
    return {"report_id": report.report_id, "checkpoint_id": report.checkpoint_id, "suite_id": report.suite_id,
            "sample_count": report.sample_count, "seed": report.seed, "metrics": dict(report.metrics),
            "intervals": {name: interval.model_dump(mode="json") for name, interval in report.intervals.items()},
            "elapsed_seconds": report.elapsed_seconds, "runtime": report.runtime, "outputs": outputs}


def _probe_sites(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    from wse.strategies.direction_search import separation

    return [{"index": index, "site": site.get("site"), "features": site.get("features"),
             "group_mean_norms": site.get("group_mean_norms"), "contrast_norm": site.get("contrast_norm"),
             "separation": separation(site), "top": site.get("top")}
            for index, site in enumerate(result.get("sites") or [])]


def experiment_summary(engine: Any, experiment_id: str) -> dict[str, Any]:
    """Everything about one experiment that a report needs, from the operator's view (module docstring)."""
    from wse.audit import controller_events, controller_record
    from wse.evaluators.generation import compare_generation
    from wse.schemas import WSEError

    experiment = engine.get_experiment(experiment_id)
    record = controller_record(engine.store.db, experiment_id)
    jobs = []
    for job in engine.list_jobs(experiment_id):
        entry = {"job_id": job.job_id, "kind": job.kind, "state": job.state, "checkpoint_id": job.checkpoint_id,
                 "suite_id": job.suite_id, "created_at": job.created_at, "started_at": job.started_at,
                 "finished_at": job.finished_at, "report_id": job.report_id,
                 "usage": None if job.usage is None else job.usage.model_dump(mode="json", exclude_none=True),
                 "error": None if job.error is None else job.error.model_dump(mode="json")}
        if job.kind == "probe" and job.result is not None:
            entry["probe"] = {"result_uri": job.result.get("result_uri"), "groups": job.result.get("groups"),
                              "tokens_per_text": job.result.get("tokens_per_text"),
                              "teacher_forced_tokens": job.result.get("teacher_forced_tokens"),
                              "texts": job.result.get("texts"), "sites": _probe_sites(job.result)}
        elif job.result is not None:
            entry["metrics"] = job.result.get("metrics")
        jobs.append(entry)
    reports = engine.list_reports(experiment_id)
    checkpoints = engine.list_experiment_checkpoints(experiment_id)
    base_reports = [r for r in reports if r.checkpoint_id == experiment.base_checkpoint_id]
    candidates = []
    for checkpoint in checkpoints:
        if checkpoint.role != "candidate":
            continue
        document = engine.find_patch(checkpoint.patch_id)
        preview = document.preview
        entry = {"checkpoint_id": checkpoint.checkpoint_id, "patch_id": checkpoint.patch_id,
                 "hypothesis": preview.hypothesis, "changed_elements": preview.changed_elements,
                 "l2_delta": preview.l2_delta, "relative_l2_delta": preview.relative_l2_delta,
                 "max_abs_delta": preview.max_abs_delta,
                 "operations": [{"op": op.op, "operator": str(op.operator), "tensor": op.tensor,
                                 "parameters": _trim(op.parameters)} for op in document.spec.operations],
                 "reports": [], "comparisons": []}
        for report in reports:
            if report.checkpoint_id != checkpoint.checkpoint_id:
                continue
            entry["reports"].append(report.report_id)
            for base in base_reports:
                if (base.suite_id, base.sample_count, base.seed) != (report.suite_id, report.sample_count,
                                                                     report.seed):
                    continue
                pair = {"base_report_id": base.report_id, "candidate_report_id": report.report_id}
                try:
                    pair.update(compare_generation(base, report).model_dump(mode="json"))
                except WSEError as exc:
                    pair["error"] = f"{exc.code.value}: {exc.message}"
                entry["comparisons"].append(pair)
        candidates.append(entry)
    return {
        "experiment_id": experiment_id, "run_id": experiment.run_id, "task_id": experiment.task_id,
        "policy_id": experiment.policy_id, "state": experiment.state,
        "base_checkpoint_id": experiment.base_checkpoint_id,
        "selected_checkpoint_id": experiment.selected_checkpoint_id,
        "budget": {name: entry.model_dump(mode="json") for name, entry in experiment.budget.items()},
        "controller": None if record is None else {
            "controller": record["controller"], "config": _trim(record["config"]), "started_at": record["started_at"],
            "ended_at": record["ended_at"], "result": record["result"]},
        "steps": _steps(controller_events(engine.store.db, experiment_id)),
        "jobs": jobs, "reports": [_report_entry(report) for report in reports], "candidates": candidates,
    }
