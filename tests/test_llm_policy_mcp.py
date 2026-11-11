"""Ledger counters, the policy ``qwen_loop_v1``, and generation passes (M2 plan Sections 4.3, 5.4, 5.5, 6; Task 11).

* **Counters.** ``generation_passes``, ``generated_tokens``, ``probe_calls``, and
  ``teacher_forced_tokens`` are ledger counters. A policy that leaves them
  unset permits none of them (limit 0); migration 10 adds them to the ledger of
  every existing experiment, with the limits its policy sets.
* **qwen_loop_v1.** The built-in policy of the Qwen loop task: 10 generation
  passes of at most 32 practice problems and 8,192 new tokens per output, 25
  probe calls of at most 64 texts of 1,024 tokens, M1's read, analysis,
  preview, and commit allowances, 6 hours of computation (2 per operation),
  the M2 edits and analyses, and the whole-tensor scaling protection.
* **Generation passes.** A development evaluation of a generation suite is a
  job, as in M1: submission reserves one pass and problems x samples x
  ``max_new_tokens`` generated tokens; the worker generates with
  ``run_generation`` on a cached model and settles the tokens it generated.
  The report carries every sample's grade, length, limit hit, and a tail
  excerpt of its output within the inline-result limit; full outputs are
  run-scoped artifacts.
* **Isolation.** Final suites stay unreachable for runs; only the operator's
  final evaluation uses them, for generation tasks too.

Models are the tiny offline fixtures of ``tests/hf_fixtures.py``. Scripted
outputs come from a stand-in runtime installed in place of the worker's text
model. Part 2 of Task 11 adds the MCP tests to this file.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from wse.api import Engine
from wse.policy import (
    LEDGER_COUNTERS,
    SMALL_MODEL_POLICY_ID,
    WEIGHTS_ONLY_POLICY_ID,
    ExperimentPolicy,
    builtin_experiment_policies,
)
from wse.schemas import ErrorCode, WSEError, canonical_json

REPO = "example-org/Llama-3.2-1B-Instruct"
REVISION = "0123456789abcdef0123456789abcdef01234567"
PROFILE = "llama_3_2_1b_instruct_default@1"
TASK = "loops_v1"
NEW_COUNTERS = ("generation_passes", "generated_tokens", "probe_calls", "teacher_forced_tokens")
QWEN_LOOP_HASH = "sha256:cbf2ccceaca494aa0e4c96af88201a7aa28d0c77cdcb407aaa59978ee3a333f4"
M1_HASHES = {SMALL_MODEL_POLICY_ID: "sha256:d8761471bfb88e1d322da1ca06003729886daafa028963f1dcf7ef807cc218b5",
             WEIGHTS_ONLY_POLICY_ID: "sha256:88d3c1e4f73e0eb8204c30b65aceb753608114c4ed05e074302db630e665fa75"}
INLINE_RESULT_LIMIT = 31 * 1024  # max_inline_json_bytes (32 KiB) less the 1 KiB response envelope
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent


def _expect(code: ErrorCode, func, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        func(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _gsm_rows(count: int, prefix: str = "p") -> list[dict]:
    # Short problems: the chat markup, the boxed-answer instruction, and 12 new tokens fit the fixture's 128 positions.
    return [{"question": f"{prefix}{i} plus {i}", "answer": f"#### {2 * i}"} for i in range(count)]


MATH_ROWS = [{"problem": f"what is {a} times {b}?", "solution": "", "answer": str(a * b), "subject": "s",
              "level": 1, "unique_id": f"test/s/{a}{b}.json"} for a, b in ((2, 3), (4, 5), (6, 7))]


def _suite(suite_id: str, dataset, *, split: str = "development", profile: str = PROFILE,
           grader: str = "gsm8k_numeric@1", samples: int = 1, max_new_tokens: int = 8,
           sample_counts: list[int] | None = None, task_id: str = TASK) -> dict:
    return {"suite_id": suite_id, "task_id": task_id, "split": split, "evaluator": "generation@1",
            "parameters": {"profile": profile, "grader": grader, "samples_per_problem": samples,
                           "max_new_tokens": max_new_tokens},
            "sample_counts": sample_counts or [dataset.example_count], "seeds": [0],
            "dataset": dataset.model_dump(mode="json")}


@pytest.fixture
def lm(hf_snapshots, tmp_path):
    """A workspace with the Llama fixture, a 4-problem practice suite (2 samples, 12 tokens), and a final suite."""
    workspace = tmp_path / "workspace"
    with Engine(workspace) as engine:
        base = engine.import_hf_snapshot(hf_snapshots["llama_untied"], repo=REPO, revision=REVISION).checkpoint_id
        practice = engine.register_text_dataset(_jsonl(tmp_path / "practice.jsonl", _gsm_rows(4)), format="gsm8k",
                                                id_prefix="gsm8k_train_")
        engine.register_suite(_suite("practice", practice, samples=2, max_new_tokens=12, sample_counts=[2, 4]))
        final = engine.register_text_dataset(_jsonl(tmp_path / "final.jsonl", MATH_ROWS), format="math500")
        engine.register_suite(_suite("final", final, split="final", grader="math_boxed@1"))
    return SimpleNamespace(workspace=workspace, base=base, root=tmp_path, practice=practice)


def _open_run(workspace: Path, base: str, policy: ExperimentPolicy | None = None,
              policy_id: str | None = None) -> tuple[str, str]:
    """Grant a run (``qwen_loop_v1`` by default) and create its experiment through the run-bound engine."""
    from wse.policy import QWEN_LOOP_POLICY_ID

    with Engine(workspace) as engine:
        if policy is not None:
            engine.register_policy(policy)
            policy_id = policy.policy_id
        policy_id = policy_id or QWEN_LOOP_POLICY_ID
        grant = engine.grant_run(TASK, policy_id, base)
    with Engine(workspace, run_id=grant.run_id) as bound:
        experiment_id = bound.create_experiment(grant.run_id, TASK, policy_id, "create").experiment_id
    return grant.run_id, experiment_id


def _worker(workspace: Path, **kwargs):
    from wse.worker import Worker

    return Worker(workspace, device="cpu", **kwargs)


def _budget(engine: Engine, experiment_id: str) -> dict[str, tuple]:
    return {name: (entry.limit, entry.consumed, entry.reserved)
            for name, entry in engine.get_experiment(experiment_id).budget.items()}


def _generation_policy(policy_id: str, **budget) -> ExperimentPolicy:
    """``qwen_loop_v1`` with other budget limits."""
    from wse.policy import QWEN_LOOP_POLICY_ID

    loop = builtin_experiment_policies()[QWEN_LOOP_POLICY_ID]
    return loop.model_copy(update={"policy_id": policy_id,
                                   "budget": loop.budget.model_copy(update=budget)})


def _samples(report) -> list:
    return [sample for problem in report.problems for sample in problem.samples]


# ---------------------------------------------------------------------------
# A stand-in runtime with scripted outputs
# ---------------------------------------------------------------------------


class ScriptedRuntime:
    """Stands in for the engine: each row generates ``text_for(prompt)``, cut at ``max_new_tokens``.

    A text ending in ``<STOP>`` ends with the stop token. Like the engine, it
    calls the logits processors once per decoding step, before the step's
    tokens are chosen, and it sleeps ``step_seconds`` per step.
    """

    def __init__(self, tokenizer, text_for, stop: int, *, step_seconds: float = 0.0,
                 started: threading.Event | None = None) -> None:
        self.tokenizer, self.text_for, self.stop = tokenizer, text_for, stop
        self.step_seconds, self.started = step_seconds, started

    def info(self) -> dict:
        return {"device": "cpu", "dtypes": ["bfloat16"], "model_class": "Scripted"}

    def generate(self, input_ids, attention_mask=None, *, max_new_tokens, stop_token_ids=(), pad_token_id=0,
                 sampling=None, seeds=None, logits_processors=()):
        import hf_fixtures
        from wse.engines.hf_torch import GenerationResult

        rows = []
        for row in range(input_ids.shape[0]):
            prompt = self.tokenizer.decode(input_ids[row][attention_mask[row].bool()].tolist())
            text = self.text_for(prompt)
            ids = self.tokenizer.encode(text.removesuffix("<STOP>"), add_special_tokens=False)
            rows.append((ids + [self.stop] if text.endswith("<STOP>") else ids)[:max_new_tokens])
        sequences = input_ids
        for step in range(max(len(ids) for ids in rows)):
            scores = torch.zeros((len(rows), hf_fixtures.VOCAB_SIZE))
            for processor in logits_processors:
                scores = processor(sequences, scores)
            if self.started is not None:
                self.started.set()
            time.sleep(self.step_seconds)
            chosen = [[ids[step] if step < len(ids) else pad_token_id] for ids in rows]
            sequences = torch.cat([sequences, torch.tensor(chosen, dtype=torch.int64)], dim=1)
        finished = [bool(ids) and ids[-1] == self.stop for ids in rows]
        return GenerationResult(tokens=rows, finished=finished)


def _script(monkeypatch, text_for, **options) -> list[str]:
    """Give the worker's text models a scripted runtime; return the weights hashes of the models it loads."""
    from wse.evaluators import generation

    original = generation.load_text_model
    loaded: list[str] = []

    def load(handle, files, *, device=None):
        loaded.append(handle.weights_hash)
        real = original(handle, files, device="cpu")
        runtime = ScriptedRuntime(real.tokenizer, text_for, real.stop_token_ids[0], **options)
        return generation.TextModel(runtime=runtime, tokenizer=real.tokenizer, stop_token_ids=real.stop_token_ids,
                                    pad_token_id=real.pad_token_id)

    monkeypatch.setattr(generation, "load_text_model", load)
    return loaded


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------


def test_qwen_loop_v1_limits_and_protections(hf_snapshots, tmp_path):
    from wse.adapters.families import role_tensor
    from wse.policy import QWEN_LOOP_POLICY_ID

    policies = builtin_experiment_policies()
    policy = policies[QWEN_LOOP_POLICY_ID]
    assert policy.policy_hash == QWEN_LOOP_HASH
    assert ExperimentPolicy.model_validate_json(policy.canonical_json()) == policy
    assert {policy_id: policies[policy_id].policy_hash for policy_id in M1_HASHES} == M1_HASHES
    assert policy.mode == "bounded_eval"
    budget = policy.budget
    assert budget.ledger_limits() == {
        "candidate_commits": 20, "patch_previews": 50, "analysis_calls": 50, "tensor_read_calls": 100,
        "evaluation_batches": 0, "evaluated_examples": 0, "model_forward_calls": 0, "compute_seconds": 6 * 3600,
        "generation_passes": 10, "generated_tokens": 10 * 32 * 8192, "probe_calls": 25,
        "teacher_forced_tokens": 25 * 64 * 1024}
    assert (budget.problems_per_generation_pass, budget.new_tokens_per_output) == (32, 8192)
    assert (budget.texts_per_probe_call, budget.tokens_per_probe_text) == (64, 1024)
    assert (budget.seconds_per_operation, budget.examples_per_batch) == (2 * 3600, 0)
    operations = policy.operations
    assert {alias: str(ref) for alias, ref in operations.edits.items()} == {
        "scale_slice": "scale_slice@1", "set_slice": "set_slice@1", "fill_slice": "fill_slice@1",
        "project_out": "project_out@1", "add_low_rank": "add_low_rank@1"}
    assert {kind: str(ref) for kind, ref in operations.analyses.items()} == {
        "stats": "tensor_stats@1", "norm": "tensor_norm@1", "top_k": "tensor_top_k@1",
        "tensor_norms": "tensor_norms@1", "svd_top_k": "svd_top_k@1",
        "direction_projection": "direction_projection@1"}
    assert operations.refuse_whole_tensor_scaling is True
    assert operations.limits.max_analysis_elements >= 248_320 * 1_024  # Qwen3.5-0.8B's tied embedding
    assert (operations.limits.max_read_elements, operations.limits.max_inline_json_bytes) == (2048, 32 * 1024)
    assert policy.evaluation_available

    # Unset counters permit nothing: the M1 policies leave them out of their JSON and give them limit 0.
    for policy_id in M1_HASHES:
        m1 = policies[policy_id]
        dumped = m1.model_dump(mode="json")["budget"]
        assert all(getattr(m1.budget, name) is None and name not in dumped for name in NEW_COUNTERS)
        assert all(m1.budget.ledger_limits()[name] == 0 for name in NEW_COUNTERS)
    assert list(LEDGER_COUNTERS[-4:]) == list(NEW_COUNTERS)
    # A weights_only policy grants no generation or probes; generation passes and probes need their per-call limits.
    no_inference = {"evaluation_batches": 0, "examples_per_batch": 0, "evaluated_examples": 0, "model_forward_calls": 0}
    for granted in ({"generation_passes": 1, "problems_per_generation_pass": 1, "new_tokens_per_output": 1,
                     "generated_tokens": 1},
                    {"probe_calls": 1, "texts_per_probe_call": 1, "tokens_per_probe_text": 1,
                     "teacher_forced_tokens": 1}):
        with pytest.raises(ValidationError, match="weights_only"):
            ExperimentPolicy(policy_id="leaky_v1", mode="weights_only", budget={**no_inference, **granted})
    with pytest.raises(ValidationError, match="problems_per_generation_pass"):
        ExperimentPolicy(policy_id="no_caps_v1", mode="bounded_eval",
                         budget={"generation_passes": 1, "generated_tokens": 100})
    with pytest.raises(ValidationError, match="texts_per_probe_call"):
        ExperimentPolicy(policy_id="no_probe_caps_v1", mode="bounded_eval",
                         budget={"probe_calls": 1, "teacher_forced_tokens": 100})

    # In a run on the Qwen3.5 fixture: the agent sees the counters and the whole-tensor rule before any refusal.
    workspace = tmp_path / "workspace"
    with Engine(workspace) as engine:
        manifest = engine.import_hf_snapshot(hf_snapshots["qwen3_5"], repo="example-org/Qwen3.5-tiny",
                                             revision=REVISION)
    base = manifest.checkpoint_id
    run_id, eid = _open_run(workspace, base)
    down, norm, embed = (role_tensor(manifest, "mlp.down", 0), role_tensor(manifest, "final_norm"),
                         role_tensor(manifest, "embed"))
    assert manifest.alias_groups and embed.tensor in manifest.alias_groups[0]

    def spec(*operations: dict) -> dict:
        return {"base_checkpoint_id": base, "hypothesis": "Changing this projection changes the outputs.",
                "operations": [{"tensor": down.tensor, "selection": down.selection(), **op} for op in operations]}

    with Engine(workspace, run_id=run_id) as bound:
        described = bound.describe_experiment(eid)
        assert described.operations.refuse_whole_tensor_scaling is True and described.evaluation_available
        assert {name: entry.limit for name, entry in described.experiment.budget.items()} == budget.ledger_limits()
        assert sorted(described.operations.edits) == sorted(operations.edits)
        error = _expect(ErrorCode.POLICY_DENIED, bound.preview_patch, eid, spec({"op": "scale_slice", "factor": 0.5}),
                        "whole")
        assert "refuse_whole_tensor_scaling" in error.message
        rows = [{"start": 0, "stop": 8, "step": 1}, {"start": 0, "stop": down.shape[1], "step": 1}]
        assert bound.preview_patch(eid, spec({"op": "scale_slice", "factor": 0.5, "selection": rows}),
                                   "rows").changed_elements > 0
        # The M2 edits are permitted.
        for key, op in (("fill", {"op": "fill_slice", "value": 0.0, "selection": rows}),
                        ("project", {"op": "project_out", "direction": [1.0] * down.shape[0], "columns": [0, 1]}),
                        ("low_rank", {"op": "add_low_rank", "u": [[1.0] * 4], "v": [[0.5, 0.5]],
                                      "selection": [{"start": 0, "stop": 4, "step": 1},
                                                    {"start": 0, "stop": 2, "step": 1}]})):
            assert bound.preview_patch(eid, spec(op), key).changed_elements > 0
        # Protected roles and alias groups are refused under this policy as under every other.
        for tensor, role in ((norm.tensor, "final_norm"), (embed.tensor, "embed")):
            shape = next(info.shape for info in manifest.tensors if info.name == tensor)
            selection = [{"start": 0, "stop": 1, "step": 1}] + [{"start": 0, "stop": size, "step": 1}
                                                                for size in shape[1:]]
            refused = _expect(ErrorCode.POLICY_DENIED, bound.preview_patch, eid, {
                "base_checkpoint_id": base, "hypothesis": "Changing this tensor changes the outputs.",
                "operations": [{"op": "scale_slice", "tensor": tensor, "selection": selection, "factor": 0.5}]},
                f"protected-{role}")
            assert tensor in refused.message
        # Protected tensors stay readable through the new analyses.
        norms = bound.analyze_model(eid, base, "tensor_norms", embed.tensor, embed.selection(),
                                    parameters={"axis": "rows"})
        assert norms.usage.analysis_calls == 1


def test_whole_tensor_scaling_is_refused_by_its_net_effect_on_the_experiments_base(hf_snapshots, tmp_path):
    """Patches whose net effect scales a whole tensor by one constant are refused, however they are written."""
    from wse.adapters.families import role_tensor

    workspace = tmp_path / "workspace"
    with Engine(workspace) as engine:
        manifest = engine.import_hf_snapshot(hf_snapshots["qwen3_5"], repo="example-org/Qwen3.5-tiny",
                                             revision=REVISION)
        down = role_tensor(manifest, "mlp.down", 0)
        weights = engine.open_checkpoint(manifest.checkpoint_id).tensor(down.tensor)  # BF16 [16, 32]
    base, (rows, columns) = manifest.checkpoint_id, down.shape

    def part(first: int, last: int) -> list[dict[str, int]]:
        return [{"start": first, "stop": last, "step": 1}, {"start": 0, "stop": columns, "step": 1}]

    def spec(parent: str, *operations: dict) -> dict:
        return {"base_checkpoint_id": parent, "hypothesis": "Changing this projection changes the outputs.",
                "operations": [{"tensor": down.tensor, **operation} for operation in operations]}

    def halves(factor: float, other: float | None = None) -> tuple[dict, dict]:
        return ({"op": "scale_slice", "factor": factor, "selection": part(0, 8)},
                {"op": "scale_slice", "factor": factor if other is None else other, "selection": part(8, rows)})

    run_id, eid = _open_run(workspace, base)
    with Engine(workspace, run_id=run_id) as bound:
        before = _budget(bound, eid)

        def refused(parent: str, key: str, *operations: dict) -> WSEError:
            error = _expect(ErrorCode.POLICY_DENIED, bound.preview_patch, eid, spec(parent, *operations), key)
            assert "refuse_whole_tensor_scaling" in error.message and error.details["tensor"] == down.tensor
            return error

        # Complementary halves with one factor scale the whole tensor, whatever the factor (0.7 is not a power of
        # two, so the factor cannot be read back exactly from any one element).
        for factor in (0.5, 0.7, -1.0, 3.0):
            error = refused(base, f"halves-{factor}", *halves(factor))
            assert abs(error.details["factor"] - factor) <= 0.01 * abs(factor)
        # So do set_slice with c * W, which needs the values of a read, and a whole fill with 0 (c = 0).
        halved = (weights.float() * 0.5).tolist()
        refused(base, "set", {"op": "set_slice", "values": halved, "selection": part(0, rows)})
        assert refused(base, "zero", {"op": "fill_slice", "value": 0.0, "selection": part(0, rows)}).details[
            "factor"] == 0
        assert _budget(bound, eid) == before  # refused previews charge nothing
        # Partial scaling is legitimate, and so are different factors on the halves or a constant fill.
        half = bound.preview_patch(eid, spec(base, halves(0.5)[0]), "half")
        assert half.changed_elements == 8 * columns
        assert bound.preview_patch(eid, spec(base, *halves(0.5, 0.25)), "mixed").changed_elements == rows * columns
        assert bound.preview_patch(eid, spec(base, {"op": "fill_slice", "value": 0.01, "selection": part(0, rows)}),
                                   "constant").changed_elements > 0
        # Chained candidates are judged against the experiment's base: scaling the rest later is refused too.
        candidate = bound.apply_patch(eid, half.patch_id, "apply-half").checkpoint_id
        refused(candidate, "rest", halves(0.5)[1])
        assert bound.preview_patch(eid, spec(candidate, halves(0.25)[1]), "rest-other").changed_elements > 0
        # Undoing the first edit restores the base, which scales nothing (c = 1).
        restored = bound.preview_patch(eid, spec(candidate, {"op": "scale_slice", "factor": 2.0,
                                                             "selection": part(0, 8)}), "restore")
        assert restored.result_weights_hash == manifest.weights_hash
    # Without the option (an M1 policy) the same patch is not refused.
    other_run, other_eid = _open_run(workspace, base, policy_id=SMALL_MODEL_POLICY_ID)
    with Engine(workspace, run_id=other_run) as bound:
        assert bound.preview_patch(other_eid, spec(base, *halves(0.5)), "halves").changed_elements == rows * columns


# ---------------------------------------------------------------------------
# Migration 10
# ---------------------------------------------------------------------------


def _downgrade_to_9(workspace: Path) -> None:
    """Remove what migration 10 adds (the four counters' ledger rows), leaving a schema-9 workspace."""
    conn = sqlite3.connect(workspace / "metadata.sqlite")
    try:
        conn.execute(f"DELETE FROM budget_ledger WHERE counter IN ({', '.join('?' * len(NEW_COUNTERS))})",
                     NEW_COUNTERS)
        conn.execute("PRAGMA user_version = 9")
        conn.commit()
    finally:
        conn.close()


def _ledger_rows(workspace: Path) -> list[tuple]:
    with sqlite3.connect(workspace / "metadata.sqlite") as conn:
        return conn.execute("SELECT experiment_id, counter, limit_value, consumed, reserved FROM budget_ledger "
                            "ORDER BY experiment_id, counter").fetchall()


def _assert_backfilled(workspace: Path, before: list[tuple], limits: dict[str, dict[str, int]]) -> None:
    """Migration 10 added exactly the four counters of each experiment, with its policy's limits, and changed no row."""
    from wse.database import LATEST_SCHEMA_VERSION

    with Engine(workspace) as engine:
        assert engine.store.db.schema_version() == LATEST_SCHEMA_VERSION == 10
        after = _ledger_rows(workspace)
        added = sorted(set(after) - set(before))
        assert sorted(set(before) - set(after)) == []
        assert added == sorted((eid, name, limits[eid][name], 0, 0) for eid in limits for name in NEW_COUNTERS)
        for eid, expected in limits.items():
            budget = engine.get_experiment(eid).budget
            assert list(budget) == list(LEDGER_COUNTERS)
            assert {name: (budget[name].limit, budget[name].consumed, budget[name].reserved,
                           budget[name].remaining) for name in NEW_COUNTERS} == {
                name: (expected[name], 0, 0, expected[name]) for name in NEW_COUNTERS}
            # A counter with limit 0 permits nothing.
            if not expected["generated_tokens"]:
                with engine.store.db.transaction() as conn:
                    with pytest.raises(WSEError) as excinfo:
                        engine.budget.reserve(eid, "job_" + "a" * 32, {"generated_tokens": 1}, conn=conn)
                assert excinfo.value.code == ErrorCode.BUDGET_EXCEEDED
                assert excinfo.value.details["counter"] == "generated_tokens"
    # Reopening is a no-op: the migration ran once.
    with Engine(workspace):
        assert _ledger_rows(workspace) == after


def test_migration_10_backfills_existing_experiments(workspace, bundles):
    from wse.policy import QWEN_LOOP_POLICY_ID

    with Engine(workspace) as engine:
        base = engine.import_bundle(bundles["task"]).checkpoint_id
        experiments = {}
        for policy_id in (SMALL_MODEL_POLICY_ID, WEIGHTS_ONLY_POLICY_ID, QWEN_LOOP_POLICY_ID):
            grant = engine.grant_run("signed_sum_v1", policy_id, base)
            experiments[policy_id] = engine.create_experiment(grant.run_id, "signed_sum_v1", policy_id,
                                                              "create").experiment_id
        # Consumed allowance survives the migration.
        engine.read_model_tensor(experiments[SMALL_MODEL_POLICY_ID], base, "layers.1.weight",
                                 [{"start": 0, "stop": 1, "step": 1}, {"start": 0, "stop": 4, "step": 1}])
    _downgrade_to_9(workspace)
    before = _ledger_rows(workspace)
    assert not {row[1] for row in before} & set(NEW_COUNTERS)
    with sqlite3.connect(workspace / "metadata.sqlite") as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 9
    policies = builtin_experiment_policies()
    _assert_backfilled(workspace, before, {
        eid: policies[policy_id].budget.ledger_limits() for policy_id, eid in experiments.items()})
    with Engine(workspace) as engine:
        assert engine.get_experiment(experiments[SMALL_MODEL_POLICY_ID]).budget["tensor_read_calls"].consumed == 1


M9_COMMIT = "51b0558"  # a commit whose code writes schema version 9 (migration 10 came after it)
M9_WRITER = """
import json, sys
from pathlib import Path
import wse
from wse.api import Engine
from wse.policy import SMALL_MODEL_POLICY_ID, WEIGHTS_ONLY_POLICY_ID
src, bundle, workspace = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
assert Path(wse.__file__).resolve().is_relative_to(src.resolve()), wse.__file__
with Engine(workspace) as engine:
    assert engine.store.db.schema_version() == 9
    base = engine.import_bundle(bundle).checkpoint_id
    ids = {}
    for policy_id in (SMALL_MODEL_POLICY_ID, WEIGHTS_ONLY_POLICY_ID):
        grant = engine.grant_run("signed_sum_v1", policy_id, base)
        ids[policy_id] = engine.create_experiment(grant.run_id, "signed_sum_v1", policy_id, "create").experiment_id
    engine.read_model_tensor(ids[SMALL_MODEL_POLICY_ID], base, "layers.1.weight",
                             [{"start": 0, "stop": 1, "step": 1}, {"start": 0, "stop": 4, "step": 1}])
print(json.dumps(ids))
"""


def _source_at(commit: str, destination: Path) -> Path | None:
    """The ``src`` tree of ``commit`` from git history, or None when it is unavailable."""
    git = shutil.which("git")
    if git is None:
        return None
    archive = subprocess.run([git, "-C", str(REPO_ROOT), "archive", "--format=tar", commit, "src"],
                             capture_output=True, timeout=60)
    if archive.returncode != 0:
        return None
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(destination, filter="data")
    return destination / "src"


def test_migration_10_on_a_workspace_written_by_schema_9_code(workspace, bundles, tmp_path):
    """The schema-9 workspace comes from the schema-9 code (git history), not from a downgrade of new output."""
    import os

    source = _source_at(M9_COMMIT, tmp_path / "m9")
    if source is None:
        pytest.skip(f"the schema-9 code (commit {M9_COMMIT}) is not available from git history")
    written = subprocess.run([sys.executable, "-c", M9_WRITER, str(source), str(bundles["task"]), str(workspace)],
                             capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(source)},
                             cwd=tmp_path, timeout=120)
    assert written.returncode == 0, written.stderr
    experiments = json.loads(written.stdout)
    before = _ledger_rows(workspace)
    assert {row[1] for row in before} == set(LEDGER_COUNTERS) - set(NEW_COUNTERS)
    _assert_backfilled(workspace, before, {eid: dict.fromkeys(NEW_COUNTERS, 0) for eid in experiments.values()})


# ---------------------------------------------------------------------------
# Generation passes
# ---------------------------------------------------------------------------


def test_generation_usage_is_reserved_and_settled_in_tokens(lm):
    from wse.worker import Worker  # noqa: F401 - the worker module must import

    run_id, eid = _open_run(lm.workspace, lm.base)
    with Engine(lm.workspace, run_id=run_id) as bound:
        before = _budget(bound, eid)
        job = bound.submit_evaluation(eid, lm.base, "practice", {"sample_count": 4, "seed": 0}, "g1")
        # 4 problems x 2 samples x 12 new tokens, one pass, and the per-operation compute time.
        assert job.state == "queued" and job.kind == "evaluation" and job.progress.batches_total == 1
        assert job.reservation.requested == {"generation_passes": 1, "generated_tokens": 96, "compute_seconds": 7200}
        held = _budget(bound, eid)
        assert held["generated_tokens"] == (10 * 32 * 8192, 0, 96) and held["generation_passes"] == (10, 0, 1)
        assert {name: held[name] for name in held if name not in ("generated_tokens", "generation_passes",
                                                                  "compute_seconds")} == \
            {name: before[name] for name in before if name not in ("generated_tokens", "generation_passes",
                                                                   "compute_seconds")}
        # A retry with the same key returns the same job and reserves nothing more.
        assert bound.submit_evaluation(eid, lm.base, "practice", {"sample_count": 4, "seed": 0}, "g1") == job
        assert _budget(bound, eid) == held
    with _worker(lm.workspace) as worker:
        assert worker.run_once() == job.job_id
    with Engine(lm.workspace, run_id=run_id) as bound:
        done = bound.get_job(eid, job.job_id)
        assert done.state == "succeeded", done.error
        report = bound.get_report(eid, done.report_id)
        tokens = sum(sample.generated_tokens for sample in _samples(report))
        assert len(_samples(report)) == 8 and 8 <= tokens <= 96
        assert done.reservation.charged == {"generation_passes": 1, "generated_tokens": tokens,
                                            "compute_seconds": done.usage.compute_seconds}
        assert 0 < done.usage.compute_seconds < 7200
        # The job's usage is what the ledger charged, counter by counter; a pass charges no forward-batch counter.
        assert {name: getattr(done.usage, name) for name in LEDGER_COUNTERS} == \
            {name: done.reservation.charged.get(name, 0) for name in LEDGER_COUNTERS}
        assert done.usage.peak_memory_bytes > 0 and done.usage.artifact_bytes > 0
        assert report.usage == done.usage and report.metrics["mean_generated_tokens"] == tokens / 8
        after = _budget(bound, eid)
        assert after["generated_tokens"] == (10 * 32 * 8192, tokens, 0)
        assert after["generation_passes"] == (10, 1, 0)
        assert after["compute_seconds"][1] >= done.usage.compute_seconds and after["compute_seconds"][2] == 0
        for name in ("evaluation_batches", "evaluated_examples", "model_forward_calls", "probe_calls",
                     "teacher_forced_tokens"):
            assert after[name] == before[name]
        assert job.job_id in json.dumps([e["subject_id"] for e in bound.audit_log(eid)])


def test_generation_passes_are_checked_against_the_policy_before_anything_is_reserved(lm):
    with Engine(lm.workspace) as engine:
        big = engine.register_text_dataset(_jsonl(lm.root / "big.jsonl", _gsm_rows(33, "q")), format="gsm8k",
                                           id_prefix="big_")
        engine.register_suite(_suite("big", big, sample_counts=[32, 33]))
        engine.register_suite(_suite("long", lm.practice, max_new_tokens=8193, sample_counts=[2]))
    run_id, eid = _open_run(lm.workspace, lm.base)
    with Engine(lm.workspace, run_id=run_id) as bound:
        before = _budget(bound, eid)
        error = _expect(ErrorCode.POLICY_DENIED, bound.submit_evaluation, eid, lm.base, "big",
                        {"sample_count": 33, "seed": 0}, "big")
        assert error.details["problems_per_generation_pass"] == 32 and error.details["permitted_sample_counts"] == [32]
        error = _expect(ErrorCode.POLICY_DENIED, bound.submit_evaluation, eid, lm.base, "long",
                        {"sample_count": 2, "seed": 0}, "long")
        assert error.details["new_tokens_per_output"] == 8192 and error.details["max_new_tokens"] == 8193
        # Unregistered configurations are refused as in M1.
        _expect(ErrorCode.POLICY_DENIED, bound.submit_evaluation, eid, lm.base, "practice",
                {"sample_count": 3, "seed": 0}, "bad")
        assert _budget(bound, eid) == before and bound.list_jobs(eid) == []
        # 32 problems are the most a pass may evaluate.
        assert bound.submit_evaluation(eid, lm.base, "big", {"sample_count": 32, "seed": 0}, "ok").reservation \
            .requested["generated_tokens"] == 32 * 8
    # The ledger counters bind: one pass, or too few tokens, is BUDGET_EXCEEDED.
    for policy, counter in ((_generation_policy("one_pass_v1", generation_passes=1), "generation_passes"),
                            (_generation_policy("few_tokens_v1", generated_tokens=95), "generated_tokens")):
        run_id, eid = _open_run(lm.workspace, lm.base, policy)
        with Engine(lm.workspace, run_id=run_id) as bound:
            if counter == "generation_passes":
                bound.submit_evaluation(eid, lm.base, "practice", {"sample_count": 2, "seed": 0}, "first")
            error = _expect(ErrorCode.BUDGET_EXCEEDED, bound.submit_evaluation, eid, lm.base, "practice",
                            {"sample_count": 4, "seed": 0}, "second")
            assert error.details["counter"] == counter
    # Under an M1 policy, which leaves the generation counters unset, no pass is permitted at all.
    run_id, eid = _open_run(lm.workspace, lm.base, policy_id=SMALL_MODEL_POLICY_ID)
    with Engine(lm.workspace, run_id=run_id) as bound:
        error = _expect(ErrorCode.POLICY_DENIED, bound.submit_evaluation, eid, lm.base, "practice",
                        {"sample_count": 2, "seed": 0}, "m1")
        assert "generation" in error.message and bound.list_jobs(eid) == []
        assert {name: bound.get_experiment(eid).budget[name].limit for name in NEW_COUNTERS} == \
            dict.fromkeys(NEW_COUNTERS, 0)


def test_a_generation_pass_matches_a_direct_evaluation(lm):
    """The worker's pass is ``run_generation`` on the same device and batch size: same results and identities."""
    with Engine(lm.workspace) as engine:
        engine.register_suite(_suite("think", lm.practice, profile="qwen35think@1", max_new_tokens=12,
                                     sample_counts=[4]))
        direct = engine.evaluate_generation(lm.base, "think", {"sample_count": 4, "seed": 0}, device="cpu",
                                            batch_size=8)
    run_id, eid = _open_run(lm.workspace, lm.base)
    with Engine(lm.workspace, run_id=run_id) as bound:
        job = bound.submit_evaluation(eid, lm.base, "think", {"sample_count": 4, "seed": 0}, "think")
    with _worker(lm.workspace) as worker:
        worker.run_once()
    with Engine(lm.workspace, run_id=run_id) as bound:
        report = bound.get_report(eid, bound.get_job(eid, job.job_id).report_id)
    assert report.split == "development" and report.management == "managed" and report.job_id == job.job_id
    assert report.metrics == dict(direct.metrics) and report.evaluator_config_hash == direct.evaluator_config_hash
    assert report.execution == direct.execution and report.execution.batch_size == 8
    assert report.model == direct.model and report.runtime == dict(direct.runtime)
    assert report.scores == list(direct.scores) and report.cluster_ids == list(direct.cluster_ids)
    stripped = [problem.model_copy(update={"samples": [sample.model_copy(update={"tail": None, "output_uri": None})
                                                       for sample in problem.samples]})
                for problem in report.problems]
    assert stripped == list(direct.problems)


def test_generation_pass_reports_tail_excerpts_within_the_response_limit(lm, monkeypatch):
    from wse.evaluators import generation
    from wse.evaluators.generation import TAIL_BYTES, json_string_bytes, report_with_tails, token_hash
    from wse.mcp_server import RESOURCE_PAGE_BYTES, _paged
    from wse.schemas import GenerationOutput

    # 32 problems, the most a pass may hold, with answers at the graders' 200-character limit and outputs much
    # longer than an excerpt: their tails loop, so repetition is visible in the report.
    loop = "\\loop {i}\n"

    def text_for(prompt: str) -> str:
        index = int(re.search(r"r(\d+) plus", prompt).group(1))
        return "so \\boxed{" + "9" * 230 + "} " + loop.format(i=index) * 40

    with Engine(lm.workspace) as engine:
        rows = engine.register_text_dataset(_jsonl(lm.root / "rows.jsonl", _gsm_rows(32, "r")), format="gsm8k",
                                            id_prefix="gsm8k_train_99")
        engine.register_suite(_suite("full", rows, max_new_tokens=1000))
    _script(monkeypatch, text_for)
    run_id, eid = _open_run(lm.workspace, lm.base)
    with Engine(lm.workspace, run_id=run_id) as bound:
        job = bound.submit_evaluation(eid, lm.base, "full", {"sample_count": 32, "seed": 0}, "full")
    with _worker(lm.workspace) as worker:
        assert worker.run_once() == job.job_id
    with Engine(lm.workspace, run_id=run_id) as bound:
        done = bound.get_job(eid, job.job_id)
        assert done.state == "succeeded", done.error
        report = bound.find_report(done.report_id)
        assert len(report.canonical_json().encode("utf-8")) <= INLINE_RESULT_LIMIT
        assert json.loads(_paged(report.report_uri, report, 0))["page_count"] == 1  # one resource read
        samples = _samples(report)
        assert len(samples) == 32 and all(len(sample.answer) == 203 for sample in samples)
        assert all(sample.limit_hit is False and sample.finished is False for sample in samples)
        for problem in report.problems:
            sample = problem.samples[0]
            document = bound.read_analysis_result(sample.output_uri)
            output = GenerationOutput.model_validate(document)
            assert (output.experiment_id, output.job_id, output.checkpoint_id, output.suite_id) == \
                (eid, job.job_id, lm.base, "full")
            assert (output.index, output.item_id, output.sample, output.seed) == \
                (problem.index, problem.item_id, 0, sample.seed)
            assert output.text == text_for(f"r{problem.index} plus") and output.prompt_sha256 == problem.prompt_sha256
            assert token_hash(output.tokens) == sample.output_sha256 and len(output.tokens) == sample.generated_tokens
            # The tail is the end of the output, within its byte bound; here the report's limit shortened it.
            assert output.text.endswith(sample.tail) and 0 < json_string_bytes(sample.tail) <= TAIL_BYTES
            assert loop.format(i=problem.index) * 3 in sample.tail
            # The full output is readable within the response limit, paged when it is larger.
            first = json.loads(_paged(sample.output_uri, document, 0))
            pages = [first] + [json.loads(_paged(sample.output_uri, document, page))
                               for page in range(1, first["page_count"])]
            assert all(len(canonical_json(page).encode("utf-8")) <= RESOURCE_PAGE_BYTES for page in pages)
            assert (first["content"] if "content" in first else
                    json.loads("".join(page["chunk"] for page in pages))) == document
        # With answers at the 203-character maximum, the tails were shortened evenly to keep the report within the
        # limit (a 2-byte escape at a cut may leave one byte unused).
        tail_bytes = {json_string_bytes(sample.tail) for sample in samples}
        assert max(tail_bytes) < TAIL_BYTES and max(tail_bytes) - min(tail_bytes) <= 1
        # With short answers, as GSM8K's, every tail keeps its full bound.
        texts = {(problem.index, 0): text_for(f"r{problem.index} plus") for problem in report.problems}
        short = report.model_copy(update={"problems": [problem.model_copy(update={"samples": [
            sample.model_copy(update={"answer": "18"}) for sample in problem.samples]})
            for problem in report.problems]})
        fitted = report_with_tails(short, texts, limit=INLINE_RESULT_LIMIT)
        assert {json_string_bytes(sample.tail) for sample in _samples(fitted)} <= {TAIL_BYTES - 1, TAIL_BYTES}
        assert len(fitted.canonical_json().encode("utf-8")) <= INLINE_RESULT_LIMIT
        # At the limit: 32 problems, 203-character answers, and outputs whose every character JSON escapes (2 bytes
        # each); the tails shrink evenly, and the report still fits.
        escaped = {key: "\\\n" * TAIL_BYTES for key in texts}
        worst = report_with_tails(report, escaped, limit=INLINE_RESULT_LIMIT)
        assert len(worst.canonical_json().encode("utf-8")) <= INLINE_RESULT_LIMIT
        assert all(0 < json_string_bytes(sample.tail) < TAIL_BYTES for sample in _samples(worst))
        assert generation.TAIL_BYTES == TAIL_BYTES
    # Another run cannot read this run's outputs or report.
    other_run, other_eid = _open_run(lm.workspace, lm.base)
    with Engine(lm.workspace, run_id=other_run) as other:
        _expect(ErrorCode.NOT_FOUND, other.read_analysis_result, samples[0].output_uri)
        _expect(ErrorCode.NOT_FOUND, other.find_report, done.report_id)


def test_generation_passes_stop_on_cancellation_and_at_the_time_limit_and_stay_charged(lm, monkeypatch):
    started = threading.Event()
    _script(monkeypatch, lambda prompt: "loop " * 40, step_seconds=0.02, started=started)
    with Engine(lm.workspace) as engine:
        engine.register_suite(_suite("slow", lm.practice, samples=2, max_new_tokens=150, sample_counts=[4]))
    reserved = 4 * 2 * 150
    # Cancellation stops the pass at its next decoding step; the steps already run stay charged.
    run_id, eid = _open_run(lm.workspace, lm.base)
    with Engine(lm.workspace, run_id=run_id) as bound:
        job = bound.submit_evaluation(eid, lm.base, "slow", {"sample_count": 4, "seed": 0}, "slow")
        with _worker(lm.workspace) as worker:
            runner = threading.Thread(target=worker.run_once)
            runner.start()
            try:
                assert started.wait(30)
                assert bound.cancel_job(eid, job.job_id, "cancel").cancel_requested
                done = bound.get_job(eid, job.job_id, wait_seconds=10)
            finally:
                runner.join()
        assert done.state == "cancelled" and done.error.details["reason"] == "cancelled"
        assert done.report_id is None and done.usage.generation_passes == 1
        assert 0 < done.usage.generated_tokens < reserved
        assert done.reservation.charged["generated_tokens"] == done.usage.generated_tokens
        budget = _budget(bound, eid)
        assert budget["generated_tokens"][1:] == (done.usage.generated_tokens, 0)
        assert budget["generation_passes"][1:] == (1, 0)
    # The per-operation time limit stops a pass the same way.
    run_id, eid = _open_run(lm.workspace, lm.base, _generation_policy("short_passes_v1", seconds_per_operation=1.0))
    with Engine(lm.workspace, run_id=run_id) as bound:
        job = bound.submit_evaluation(eid, lm.base, "slow", {"sample_count": 4, "seed": 0}, "slow")
        assert job.reservation.requested["compute_seconds"] == 1
        with _worker(lm.workspace) as worker:
            worker.run_once()
        done = bound.get_job(eid, job.job_id)
        assert done.state == "failed" and done.error.details["reason"] == "timeout"
        assert done.usage.compute_seconds == pytest.approx(1.0) and done.usage.generation_passes == 1
        assert 0 < done.usage.generated_tokens < reserved
        assert _budget(bound, eid)["generated_tokens"][1:] == (done.usage.generated_tokens, 0)
        with Engine(lm.workspace) as engine:
            assert engine.store.db.conn.execute("SELECT COUNT(*) FROM analysis_results WHERE experiment_id = ?",
                                                (eid,)).fetchone()[0] == 0


def test_the_worker_reuses_a_loaded_model_across_jobs(lm, monkeypatch):
    from wse.evaluators import generation

    original = generation.load_text_model
    loads: list[str] = []

    def counting(handle, files, *, device=None):
        loads.append(handle.weights_hash)
        return original(handle, files, device=device)

    monkeypatch.setattr(generation, "load_text_model", counting)
    # Each queued pass holds its per-operation compute time, so a shorter limit lets nine passes queue at once.
    run_id, eid = _open_run(lm.workspace, lm.base, _generation_policy("cache_v1", seconds_per_operation=60.0))
    with Engine(lm.workspace, run_id=run_id) as bound:
        spec = {"base_checkpoint_id": lm.base, "hypothesis": "Scaling one row changes some samples.",
                "operations": [{"op": "scale_slice", "tensor": "model.layers.0.mlp.down_proj.weight", "factor": -3.0,
                                "selection": [{"start": 0, "stop": 1, "step": 1}, {"start": 0, "stop": 32,
                                                                                   "step": 1}]}]}
        candidate = bound.apply_patch(eid, bound.preview_patch(eid, spec, "p").patch_id, "a").checkpoint_id

        def submit(checkpoint: str, key: str) -> str:
            return bound.submit_evaluation(eid, checkpoint, "practice", {"sample_count": 2, "seed": 0}, key).job_id

        order = [lm.base, lm.base, candidate, lm.base]
        first = [submit(checkpoint, f"one-{i}") for i, checkpoint in enumerate(order)]
        with _worker(lm.workspace, model_cache_size=1) as worker:
            assert [worker.run_once() for _ in first] == first
            assert len(worker.models) == 1
        weights = {checkpoint: bound.inspect_model(eid, checkpoint).weights_hash for checkpoint in (lm.base, candidate)}
        # One entry: consecutive jobs on a checkpoint load it once; another checkpoint replaces it.
        assert loads == [weights[lm.base], weights[candidate], weights[lm.base]]
        loads.clear()
        second = [submit(checkpoint, f"two-{i}") for i, checkpoint in enumerate([lm.base, candidate] * 2)]
        with _worker(lm.workspace, model_cache_size=2) as worker:
            assert [worker.run_once() for _ in second] == second
            assert len(worker.models) == 2
        assert loads == [weights[lm.base], weights[candidate]]
        loads.clear()
        third = submit(lm.base, "three")
        with _worker(lm.workspace, model_cache_size=0) as worker:  # no cache: every job loads its model
            worker.run_once()
            assert len(worker.models) == 0
        assert loads == [weights[lm.base]]
        # A cached model gives the results of a freshly loaded one.
        outputs = {}
        for job_id in first + second + [third]:
            job = bound.get_job(eid, job_id)
            assert job.state == "succeeded", job.error
            report = bound.get_report(eid, job.report_id)
            outputs.setdefault(job.checkpoint_id, set()).add(
                tuple(sample.output_sha256 for sample in _samples(report)))
        assert len(outputs[lm.base]) == 1 and len(outputs[candidate]) == 1


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def test_final_suite_is_unreachable_through_generation_passes(lm):
    run_id, eid = _open_run(lm.workspace, lm.base)
    with Engine(lm.workspace, run_id=run_id) as bound:
        before = _budget(bound, eid)
        for func in (bound.get_suite, bound.jobs.load_suite):
            _expect(ErrorCode.NOT_FOUND, func, "final")
        denied = _expect(ErrorCode.POLICY_DENIED, bound.submit_evaluation, eid, lm.base, "final",
                         {"sample_count": 3, "seed": 0}, "final")
        assert "development" in denied.message
        _expect(ErrorCode.POLICY_DENIED, bound.evaluate_generation, lm.base, "final", {"sample_count": 3, "seed": 0})
        _expect(ErrorCode.POLICY_DENIED, bound.final_evaluate, eid)
        assert bound.list_jobs(eid) == [] and _budget(bound, eid) == before
        assert [suite.suite_id for suite in bound.describe_experiment(eid).suites] == ["practice"]
        job = bound.submit_evaluation(eid, lm.base, "practice", {"sample_count": 4, "seed": 0}, "practice")
    with Engine(lm.workspace) as engine:
        # Not even the operator queues a final suite as a development evaluation.
        _expect(ErrorCode.POLICY_DENIED, engine.submit_evaluation, eid, lm.base, "final",
                {"sample_count": 3, "seed": 0}, "operator-final")
    with _worker(lm.workspace) as worker:
        assert worker.run_once() == job.job_id
    with Engine(lm.workspace) as engine:
        artifacts = engine.store.db.conn.execute("SELECT uri FROM analysis_results ORDER BY uri").fetchall()
        assert len(artifacts) == 8  # the practice pass's outputs
        engine.freeze(eid, lm.base)
        final = engine.final_evaluate(eid)  # on the host's engine device, mps when available
        assert final.same_checkpoint and engine.get_experiment(eid).state == "CLOSED"
        for report in (final.candidate, final.control):
            assert report.split == "final" and report.sample_count == 3
            assert report.evaluator.extension_id == "generation"
            assert report.problems is None and report.scores is None and report.cluster_ids is None
            assert set(report.metrics) == {"accuracy", "limit_hit_rate", "mean_generated_tokens"}
            assert report.execution.device == report.runtime["device"] in ("mps", "cpu")
        assert final.comparison.sample_count == 3 and final.comparison.difference == 0
        # The final outputs are never stored as artifacts.
        assert engine.store.db.conn.execute("SELECT uri FROM analysis_results ORDER BY uri").fetchall() == artifacts
    with Engine(lm.workspace, run_id=run_id) as bound:
        closed = bound.get_final_report(eid)
        visible = [canonical_json(closed.model_dump(mode="json"))]
        visible += [report.canonical_json() for report in bound.list_reports(eid)]
        visible += [canonical_json(bound.read_analysis_result(row["uri"])) for row in artifacts]
        text = "\n".join(visible)
        assert not [row for row in MATH_ROWS if row["problem"] in text or row["unique_id"] in text]


def test_development_text_suites_overlapping_a_final_suite_are_refused(lm):
    from wse.schemas import TextItem, TextSuiteData

    def inline(suite_id: str, items: list[TextItem], split: str = "development", task_id: str = TASK) -> dict:
        return {"suite_id": suite_id, "task_id": task_id, "split": split, "evaluator": "generation@1",
                "parameters": {"profile": PROFILE, "grader": "math_boxed@1", "max_new_tokens": 8},
                "sample_counts": [len(items)], "seeds": [0],
                "data": TextSuiteData(items=items).model_dump(mode="json")}

    # Fullwidth letters, case, and whitespace normalize to a final problem ("what is 2 times 3?").
    leaked = TextItem(item_id="leak", problem="ＷＨＡＴ  is 2\ttimes 3? ", reference="6")
    fresh = TextItem(item_id="fresh", problem="what is 3 times 3?", reference="9")
    with Engine(lm.workspace) as engine:
        for suite_id, task_id in (("inline_leak", TASK), ("other_task_leak", "another_task")):
            error = _expect(ErrorCode.INVALID_ARGUMENT, engine.register_suite,
                            inline(suite_id, [fresh, leaked], task_id=task_id))
            assert error.details["overlapping_items"] == ["leak"] and error.details["final_suite_id"] == "final"
        assert [suite.suite_id for suite in engine.jobs.list_suites(TASK)] == ["practice", "final"]
        assert engine.register_suite(inline("inline_ok", [fresh])).dataset is not None  # stored, not inline
        # In the other order, a final suite that overlaps a registered practice item is refused too.
        error = _expect(ErrorCode.INVALID_ARGUMENT, engine.register_suite,
                        inline("final_two", [TextItem(item_id="f", problem="What is 3 times 3?", reference="9")],
                               split="final", task_id="third_task"))
        assert error.details["overlapping_items"] == ["fresh"] and error.details["development_suite_id"] == "inline_ok"
        assert "final_two" not in [suite.suite_id for suite in engine.jobs.list_suites("third_task")]
