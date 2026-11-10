"""Resumable paired generation runs (``wse.evaluators.generation_runs``, M2 plan Task 7).

The runner evaluates a base and a candidate checkpoint on a seeded problem
subset through ``generation@1``'s plan, run, and result steps, records one
line per finished (checkpoint, problem), resumes from those lines, and ends
with the paired comparison. Models are the tiny offline fixtures of
``tests/hf_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import HF_SKIP_REASON
from wse.api import Engine
from wse.schemas import ErrorCode, WSEError

pytest.importorskip("transformers", reason=HF_SKIP_REASON)

REPO = "example-org/Llama-3.2-1B-Instruct"
REVISION = "0123456789abcdef0123456789abcdef01234567"
DOWN = "model.layers.0.mlp.down_proj.weight"
# Short problems: the boxed-answer instruction plus 6 new tokens must fit the fixture's 128 positions.
ROWS = [{"problem": f"{a}+{b}?", "solution": "", "answer": str(a + b), "subject": "s", "level": 1,
         "unique_id": f"test/s/{a}{b}.json"} for a, b in ((1, 1), (2, 3), (4, 4), (5, 1), (6, 2), (7, 0))]


def _expect(code, func, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        func(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


@pytest.fixture(scope="module")
def pair(hf_snapshots, tmp_path_factory):
    """A workspace with the Llama fixture and a candidate that scales one row of an MLP projection."""
    from wse.policy import SMALL_MODEL_POLICY_ID

    root = tmp_path_factory.mktemp("generation_runs")
    dataset = root / "math.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in ROWS), encoding="utf-8")
    with Engine(root / "workspace") as engine:
        base = engine.import_hf_snapshot(hf_snapshots["llama_untied"], repo=REPO, revision=REVISION).checkpoint_id
        grant = engine.grant_run("runner_v1", SMALL_MODEL_POLICY_ID, base)
        experiment = engine.create_experiment(grant.run_id, "runner_v1", SMALL_MODEL_POLICY_ID, "create")
        spec = {"base_checkpoint_id": base, "hypothesis": "Scaling one row changes some samples.",
                "operations": [{"op": "scale_slice", "tensor": DOWN, "factor": -3.0,
                                "selection": [{"start": 0, "stop": 16, "step": 1}, {"start": 0, "stop": 32,
                                                                                   "step": 1}]}]}
        preview = engine.preview_patch(experiment.experiment_id, spec, "preview")
        candidate = engine.apply_patch(experiment.experiment_id, preview.patch_id, "apply").checkpoint_id
    return root, dataset, base, candidate


def _run(engine, pair, out, **overrides):
    from wse.evaluators.generation_runs import run_comparison

    root, dataset, base, candidate = pair
    options = dict(base=base, candidate=candidate, dataset=dataset, format="math500", count=4, subset_seed=7,
                   profile="qwen35think@1", grader="math_boxed@1", max_new_tokens=6, device="cpu", batch_size=2,
                   out_dir=out)
    return run_comparison(engine, **{**options, **overrides})


def test_select_subset_is_seeded_and_kept_in_file_order():
    from wse.evaluators.generation_runs import select_subset

    ids = [f"item_{i}" for i in range(50)]
    first = select_subset(ids, 10, 0)
    assert first == select_subset(ids, 10, 0) and first == sorted(first) and len(set(first)) == 10
    assert select_subset(ids, 10, 1) != first
    assert select_subset(ids, 50, 3) == list(range(50))
    for count, seed in ((0, 0), (51, 0), (5, -1)):
        _expect(ErrorCode.INVALID_ARGUMENT, select_subset, ids, count, seed)


class _Interrupt(Exception):
    pass


def test_the_runner_resumes_and_matches_single_evaluations(pair, tmp_path, monkeypatch):
    import wse.evaluators.generation_runs as runs
    from wse.evaluators.generation import compare_generation, evaluate_generation
    from wse.evaluators.text_suites import read_jsonl

    root, dataset, base, candidate = pair
    out = tmp_path / "out"
    calls: list[tuple[int, ...]] = []
    original = runs.run_problems

    def counting(plan, text_model, indices, **kwargs):
        calls.append(tuple(indices))
        return original(plan, text_model, indices, **kwargs)

    monkeypatch.setattr(runs, "run_problems", counting)
    lines: list[str] = []

    def interrupt_after_three(line: str) -> None:
        lines.append(line)
        if len(lines) == 3:
            raise _Interrupt

    with Engine(root / "workspace") as engine:
        with pytest.raises(_Interrupt):
            _run(engine, pair, out, log=interrupt_after_three)
        recorded = runs.read_log(out / runs.PROBLEMS_FILE)
        # The base's first chunk (2 problems) and the candidate's first problem were recorded before the stop.
        assert [(r["checkpoint"], r["problem"]["index"]) for r in recorded] == [("base", 0), ("base", 1),
                                                                                 ("candidate", 0)]
        assert not (out / runs.SUMMARY_FILE).exists()
        calls.clear()
        summary = _run(engine, pair, out)
        # Nothing recorded is generated again: the restart starts at the candidate's first chunk.
        assert calls == [(1,), (2, 3), (2, 3)]
        assert len(runs.read_log(out / runs.PROBLEMS_FILE)) == 8

        # The runner's results are those of single evaluations of the same subset, batch size, and device.
        data, _ = read_jsonl(dataset, format="math500")
        suite, indices = runs.subset_suite(data, count=4, seed=7, parameters={
            "profile": "qwen35think@1", "grader": "math_boxed@1", "samples_per_problem": 1, "max_new_tokens": 6})
        assert summary["configuration"]["subset"]["source_indices"] == indices
        assert summary["configuration"]["subset"]["data_hash"] == suite.data_hash
        direct = {}
        for label, checkpoint in (("base", base), ("candidate", candidate)):
            handle = engine.open_checkpoint(checkpoint)
            direct[label] = evaluate_generation(handle, suite, {"sample_count": 4, "seed": 0},
                                                files=engine.model_files(checkpoint), registry=engine.registry,
                                                batch_size=2, device="cpu")
            result = summary["results"][label]
            assert result["evaluator_config_hash"] == direct[label].evaluator_config_hash
            assert result["metrics"] == dict(direct[label].metrics)
            assert result["problems"] == [p.model_dump(mode="json") for p in direct[label].problems]
            assert result["model"] == handle.identity.model_dump(mode="json")
        expected = compare_generation(direct["base"], direct["candidate"]).model_dump(mode="json")
        assert summary["comparison"] == expected
        assert set(summary["comparison"]["differences"]) == {"accuracy", "limit_hit_rate", "mean_generated_tokens"}
        # The candidate generated different tokens for at least one problem: the edit is visible.
        base_hashes = [s["output_sha256"] for p in summary["results"]["base"]["problems"] for s in p["samples"]]
        candidate_hashes = [s["output_sha256"] for p in summary["results"]["candidate"]["problems"]
                            for s in p["samples"]]
        assert base_hashes != candidate_hashes
        assert json.loads((out / runs.SUMMARY_FILE).read_text())["comparison"] == expected

        # A torn last line (an interrupted write) is dropped; the summary is rebuilt from the recorded lines.
        with (out / runs.PROBLEMS_FILE).open("a", encoding="utf-8") as handle:
            handle.write('{"checkpoint": "ba')
        calls.clear()
        again = _run(engine, pair, out)
        assert calls == [] and again["comparison"] == expected
        assert len(runs.read_log(out / runs.PROBLEMS_FILE)) == 8

        # Another configuration never reuses the directory.
        error = _expect(ErrorCode.INVALID_ARGUMENT, _run, engine, pair, out, max_new_tokens=5)
        assert error.details["differing"] == ["parameters", "profile_hash"]
        error = _expect(ErrorCode.INVALID_ARGUMENT, _run, engine, pair, out, subset_seed=8)
        assert error.details["differing"] == ["subset", "suite_id"]


def test_recorded_problems_must_match_the_configuration(pair, tmp_path):
    import wse.evaluators.generation_runs as runs

    root, _, _, _ = pair
    out = tmp_path / "out"
    with Engine(root / "workspace") as engine:
        _run(engine, pair, out, count=2)
        path = out / runs.PROBLEMS_FILE
        records = runs.read_log(path)
        records[0]["problem"]["prompt_sha256"] = "sha256:" + "0" * 64
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        error = _expect(ErrorCode.INVALID_ARGUMENT, _run, engine, pair, out, count=2)
        assert error.details["item_id"] == records[0]["problem"]["item_id"]


def test_wse_compare_generation_cli(pair, tmp_path, capsys):
    from wse.cli import main

    root, dataset, base, candidate = pair
    out = tmp_path / "cli"
    argv = ["--workspace", str(root / "workspace"), "compare-generation", "--base", base, "--candidate", candidate,
            "--dataset", str(dataset), "--dataset-format", "math500", "--problems", "2", "--subset-seed", "7",
            "--profile", "qwen35think@1", "--grader", "math_boxed@1", "--max-new-tokens", "4", "--device", "cpu",
            "--out", str(out)]
    assert main(argv) == 0
    captured = capsys.readouterr()
    printed = json.loads(captured.out)
    assert printed["comparison"]["problems"] == 2 and set(printed["seconds"]) == {"base", "candidate"}
    assert captured.err.count("\n") == 4  # one progress line per (checkpoint, problem)
    assert (out / "summary.json").exists() and (out / "run.json").exists()
    assert main(argv[:-1] + [str(out)]) == 0  # rerun: everything is recorded, nothing is generated
    assert capsys.readouterr().err == ""
    assert main([a if a != "4" else "5" for a in argv]) == 2  # another configuration in the same directory
    assert '"INVALID_ARGUMENT"' in capsys.readouterr().err


def test_read_log_keeps_and_terminates_an_unterminated_last_record(tmp_path):
    import wse.evaluators.generation_runs as runs

    path = tmp_path / "log.jsonl"
    path.write_bytes(b'{"a": 1}\n{"a": 2}')  # a complete last record whose newline was not written
    assert runs.read_log(path) == [{"a": 1}, {"a": 2}]
    assert path.read_bytes() == b'{"a": 1}\n{"a": 2}\n'
    runs._append(path, {"a": 3})
    assert runs.read_log(path) == [{"a": 1}, {"a": 2}, {"a": 3}]
    # A last line that is not valid JSON is dropped and truncated away.
    path.write_bytes(b'{"a": 1}\n{"a": ')
    assert runs.read_log(path) == [{"a": 1}]
    assert path.read_bytes() == b'{"a": 1}\n'
    # Invalid JSON before the last record is corruption, and the file is left as it is.
    path.write_bytes(b'{"a": 1}\n{"a": \n{"a": 3}')
    _expect(ErrorCode.ARTIFACT_CORRUPT, runs.read_log, path)
    assert path.read_bytes() == b'{"a": 1}\n{"a": \n{"a": 3}'


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {str(path.relative_to(directory)): path.read_bytes() for path in sorted(directory.rglob("*"))
            if path.is_file()}


def _forbid(monkeypatch, name: str) -> None:
    import wse.evaluators.generation_runs as runs

    def forbidden(*args, **kwargs):
        raise AssertionError(f"{name} was called")

    monkeypatch.setattr(runs, name, forbidden)


def test_a_restart_under_other_library_versions_is_refused(pair, tmp_path, monkeypatch):
    import wse.evaluators.generation_runs as runs

    root = pair[0]
    out = tmp_path / "out"
    with Engine(root / "workspace") as engine:
        summary = _run(engine, pair, out, count=2)
        recorded = json.loads((out / runs.RUN_FILE).read_text())["configuration"]["libraries"]
        # The recorded versions are those the results report.
        assert {"torch", "transformers"} <= set(recorded)
        assert recorded == {name: summary["results"]["base"]["runtime"][name] for name in recorded}
        current = dict(recorded)
        monkeypatch.setattr(runs, "_run_libraries", lambda: {**current, "transformers": "0.0.1"})
        _forbid(monkeypatch, "load_text_model")
        before = _snapshot(out)
        error = _expect(ErrorCode.INVALID_ARGUMENT, _run, engine, pair, out, count=2)
    assert f"transformers {recorded['transformers']} recorded, 0.0.1 now" in error.message
    assert error.details["libraries"] == {"transformers": {"recorded": recorded["transformers"], "current": "0.0.1"}}
    assert _snapshot(out) == before


def test_a_run_recorded_without_library_versions_is_not_continued(pair, tmp_path, monkeypatch):
    import wse.evaluators.generation_runs as runs

    root = pair[0]
    out = tmp_path / "out"
    with Engine(root / "workspace") as engine:
        _run(engine, pair, out, count=2)
        run_file = out / runs.RUN_FILE
        run = json.loads(run_file.read_text())
        del run["configuration"]["libraries"]
        run_file.write_text(json.dumps(run), encoding="utf-8")
        with (out / runs.PROBLEMS_FILE).open("a", encoding="utf-8") as handle:
            handle.write('{"checkpoint": "ba')  # a torn line, which a continued run would truncate
        _forbid(monkeypatch, "load_text_model")
        before = _snapshot(out)
        error = _expect(ErrorCode.INVALID_ARGUMENT, _run, engine, pair, out, count=2)
    assert "recorded without library versions" in error.message
    assert "stay valid as recorded" in error.message and "new output directory" in error.message
    assert _snapshot(out) == before


def test_sides_that_render_different_prompts_are_refused_before_generation(pair, tmp_path, monkeypatch):
    import wse.evaluators.generation_runs as runs

    root = pair[0]
    out = tmp_path / "out"
    original = runs.encode_prompts
    sides: list[object] = []

    def encode(text_model, profile, items, indices):
        prompts = original(text_model, profile, items, indices)
        sides.append(text_model)
        if len(sides) == 2:  # the candidate renders problem 1 with one more token
            prompts[1] = [*prompts[1], prompts[1][-1]]
        return prompts

    monkeypatch.setattr(runs, "encode_prompts", encode)
    _forbid(monkeypatch, "run_problems")
    with Engine(root / "workspace") as engine:
        error = _expect(ErrorCode.INVALID_ARGUMENT, _run, engine, pair, out)
    assert len(sides) == 2 and "problem 1 " in error.message
    assert error.details["index"] == 1
    assert not out.exists()  # nothing generated, nothing recorded


@pytest.mark.parametrize("samples", [1, 2])
def test_an_uninterrupted_run_reports_the_usage_of_a_single_evaluation(pair, tmp_path, samples):
    import wse.evaluators.generation_runs as runs
    from wse.evaluators.generation import evaluate_generation
    from wse.evaluators.text_suites import read_jsonl

    root, dataset, base, candidate = pair
    out = tmp_path / "out"
    with Engine(root / "workspace") as engine:
        summary = _run(engine, pair, out, count=3, samples_per_problem=samples)
        data, _ = read_jsonl(dataset, format="math500")
        suite, _ = runs.subset_suite(data, count=3, seed=7, parameters={
            "profile": "qwen35think@1", "grader": "math_boxed@1", "samples_per_problem": samples, "max_new_tokens": 6})
        for label, checkpoint in (("base", base), ("candidate", candidate)):
            direct = evaluate_generation(engine.open_checkpoint(checkpoint), suite, {"sample_count": 3, "seed": 0},
                                         files=engine.model_files(checkpoint), registry=engine.registry,
                                         batch_size=2, device="cpu")
            result = summary["results"][label]
            assert result["problems"] == [p.model_dump(mode="json") for p in direct.problems]
            usage = result["usage"]
            # 3 problems x samples rows in batches of 2: one batch per sample of a full chunk, then the rest.
            assert usage["evaluation_batches"] == direct.usage.evaluation_batches == -(-3 * samples // 2)
            assert usage["evaluated_examples"] == direct.usage.evaluated_examples == 3 * samples
            assert usage["model_forward_calls"] == direct.usage.model_forward_calls
            assert result["split"] == direct.split == "final"
            assert result["errors"] == list(direct.errors)
