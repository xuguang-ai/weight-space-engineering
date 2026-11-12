"""Managed probes: probe jobs, their artifacts, capabilities, and the runner's language-model tools (M2 Task 11).

* **Probe jobs.** ``Engine.submit_probe`` queues a teacher-forced probe of a
  checkpoint of the experiment; the worker runs ``wse.probes.probe`` on its
  cached model. Texts are outputs of the run's own generation passes (by
  output URI), strings, or items of a development suite of the run's task.
  Each is cut to its last ``tokens_per_probe_text`` positions; its window is
  the output (or the text after its first token) inside them, never the first
  kept position.
* **Results.** The job carries an inline summary within the response limit;
  the full vectors are a run-scoped artifact whose JSON pointers are vector
  references (``<uri>#/sites/<i>/contrast/direction``).
* **Capabilities.** ``probe.hidden_states`` and ``weights.direction_edit``
  belong to ``hf_causal_lm@1``; M1's MLP adapter has neither.
* **Runner.** Policies that grant generation passes or probes expose them to
  controllers, start a worker, and describe them in the instructions.

Models are the tiny offline fixtures of ``tests/hf_fixtures.py``; generation
outputs are scripted where a test needs known outcomes (``_script``).
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import HF_SKIP_REASON
from test_llm_policy_mcp import (  # noqa: F401 - lm is a fixture
    REVISION,
    TASK,
    _budget,
    _expect,
    _generation_policy,
    _open_run,
    _samples,
    _script,
    _source_at,
    _worker,
    lm,
)
from wse.api import Engine
from wse.schemas import ErrorCode, json_hash

pytest.importorskip("transformers", reason=HF_SKIP_REASON)

DOWN0 = "model.layers.0.mlp.down_proj"  # [16, 32] in the Llama fixture: 32 inputs, 16 outputs
FULL = [{"start": 0, "stop": 16, "step": 1}, {"start": 0, "stop": 32, "step": 1}]
SITES = ["residual:1", f"{DOWN0}:input", {"role": "mlp.down", "layer": 0, "point": "output"}]
RESOLVED = ["residual:1", f"{DOWN0}:input", f"{DOWN0}:output"]
INLINE_RESULT_LIMIT = 31 * 1024
CONTEXT = 128  # the fixtures' max_position_embeddings


def _decode(values: list) -> np.ndarray:
    """A stored vector with its non-finite entries ("inf", "-inf", "nan") decoded."""
    special = {"inf": np.inf, "-inf": -np.inf, "nan": np.nan}
    return np.array([special[value] if isinstance(value, str) else value for value in values], dtype=np.float64)


def _six(value: float) -> float:
    return float(f"{value:.6g}")


def _loop_or_answer(prompt: str) -> str:
    """Scripted outputs: even problems answer and stop (10 tokens); odd problems loop to the 12-token limit."""
    index = int(re.search(r"p(\d+) plus", prompt).group(1))
    return f"\\boxed{{{2 * index}}}<STOP>" if index % 2 == 0 else "loop " * 10


def _pass(lm, monkeypatch, policy=None, key: str = "pass"):
    """A run with one scripted generation pass on the 4-problem practice suite (2 samples each): 8 outputs."""
    _script(monkeypatch, _loop_or_answer)
    run_id, eid = _open_run(lm.workspace, lm.base, policy)
    with Engine(lm.workspace, run_id=run_id) as bound:
        job = bound.submit_evaluation(eid, lm.base, "practice", {"sample_count": 4, "seed": 0}, key)
    with _worker(lm.workspace) as worker:
        assert worker.run_once() == job.job_id
    with Engine(lm.workspace, run_id=run_id) as bound:
        done = bound.get_job(eid, job.job_id)
        assert done.state == "succeeded", done.error
        report = bound.get_report(eid, done.report_id)
    return run_id, eid, report


def _probe(lm, run_id: str, eid: str, texts, sites, *, key: str = "probe", groups=None, **options):
    """Submit a probe through the run-bound engine, run it in a worker, and return the job and its artifact."""
    with Engine(lm.workspace, run_id=run_id) as bound:
        job = bound.submit_probe(eid, lm.base, texts, sites, key, groups=groups, **options)
        assert job.kind == "probe" and job.state == "queued" and job.suite_id is None and job.config is None
    with _worker(lm.workspace) as worker:
        assert worker.run_once() == job.job_id
    with Engine(lm.workspace, run_id=run_id) as bound:
        done = bound.get_job(eid, job.job_id)
        assert done.state == "succeeded", done.error
        document = bound.read_analysis_result(done.result["result_uri"])
    return done, document


def _groups_by_outcome(samples) -> dict[str, list[int]]:
    return {"a": [k for k, s in enumerate(samples) if s.limit_hit], "b": [k for k, s in enumerate(samples) if s.finished]}


def _expected_texts(lm, samples, limit: int, strings=(), items=()):
    """The documented probe texts, built independently of the worker: (tokens, window start) after cutting."""
    from wse.evaluators.generation import effective_profile, encode_prompts, load_text_model, suite_parameters, \
        token_hash

    with Engine(lm.workspace) as engine:
        handle = engine.open_checkpoint(lm.base)
        text_model = load_text_model(handle, engine.model_files(lm.base), device="cpu")
        suite = engine.jobs._read_suite("practice")
        suite = suite.resolved(engine.datasets.load(suite.dataset))
        profile = effective_profile(suite_parameters(suite), handle.source, engine.registry)
        prompts = encode_prompts(text_model, profile, suite.data.items, range(suite.data.example_count))
        outputs = [engine.read_analysis_result(sample.output_uri) for sample in samples]
    full = []
    for output in outputs:
        prompt = prompts[output["index"]]
        assert token_hash(prompt) == output["prompt_sha256"]
        tokens = output["tokens"]
        body = tokens[:-1] if output["finished"] and len(tokens) > 1 else tokens  # the stop token is left out
        full.append((prompt + body, len(prompt)))
    full += [(text_model.tokenizer(text)["input_ids"], 1) for text in strings]
    full += [(prompts[item], 1) for item in items]
    texts = []
    for tokens, start in full:
        dropped = max(0, len(tokens) - limit)
        texts.append((tuple(tokens[dropped:]), max(start - dropped, 1), dropped))
    return handle, texts


# ---------------------------------------------------------------------------
# Means, windows, and the cutting rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tokens_per_text", [None, 8])
def test_a_managed_probe_equals_the_direct_probe_on_the_same_texts_and_sites(lm, monkeypatch, tokens_per_text):
    from wse.evaluators.generation import token_hash
    from wse.probes import ProbeText, probe

    policy = None if tokens_per_text is None else _generation_policy("short_texts_v1",
                                                                      tokens_per_probe_text=tokens_per_text)
    run_id, eid, report = _pass(lm, monkeypatch, policy)
    samples = _samples(report)
    groups = _groups_by_outcome(samples)
    assert len(groups["a"]) == 4 and len(groups["b"]) == 4
    groups["b"].append(8)  # the string joins the finished outputs
    texts = [{"output_uri": sample.output_uri} for sample in samples] + [{"text": "loop loop loop"},
                                                                         {"suite_id": "practice", "item": 3}]
    done, document = _probe(lm, run_id, eid, texts, SITES, groups=groups)

    # The same texts and windows, by the documented rule: under qwen_loop_v1 the fixture's context (128) bounds a
    # text; with 8 tokens per text, every text is cut to its last 8 positions.
    limit = CONTEXT if tokens_per_text is None else tokens_per_text
    handle, expected = _expected_texts(lm, samples, limit, strings=["loop loop loop"], items=[3])
    direct = probe(handle, [ProbeText(tokens, start=start) for tokens, start, _ in expected], sites=RESOLVED,
                   device="cpu")
    assert document["kind"] == "probe_result" and document["checkpoint_id"] == lm.base
    assert document["experiment_id"] == eid and document["job_id"] == done.job_id
    assert document["model"] == handle.identity.model_dump(mode="json") and document["tokens_per_text"] == limit
    assert document["groups"] == groups
    for record, (tokens, start, dropped) in zip(document["texts"], expected, strict=True):
        assert (record["forward_tokens"], record["window_start"], record["window_stop"]) == (len(tokens), start,
                                                                                              len(tokens))
        assert record["dropped_tokens"] == dropped and record["tokens_sha256"] == token_hash(tokens)
    assert [record["source"] for record in document["texts"]] == ["output"] * 8 + ["text", "suite_item"]
    assert [record["output_uri"] for record in document["texts"][:8]] == [s.output_uri for s in samples]
    assert [record["limit_hit"] for record in document["texts"][:8]] == [s.limit_hit for s in samples]
    assert [record["mean_logprob"] for record in document["texts"]] == list(direct.mean_logprobs)
    if tokens_per_text is not None:
        # A limit-hit output (12 tokens) is longer than 8: its prompt and first output tokens are dropped, and the
        # window starts after the first kept position, which has no context.
        assert all(document["texts"][k]["window_start"] == 1 for k in groups["a"])
        assert all(record["dropped_tokens"] > 0 for record in document["texts"])
    else:
        assert all(record["dropped_tokens"] == 0 for record in document["texts"])
    for index, (site, requested) in enumerate(zip(RESOLVED, SITES, strict=True)):
        entry = document["sites"][index]
        assert entry["site"] == site and entry["requested"] == requested
        assert entry["features"] == direct.means[site].shape[1]
        for group in ("a", "b"):
            np.testing.assert_array_equal(np.array(entry["group_means"][group]), direct.group_mean(site, groups[group]))
            assert entry["group_mean_norms"][group] == float(np.linalg.norm(direct.group_mean(site, groups[group])))
        contrast = direct.contrast(site, groups["a"], groups["b"])
        np.testing.assert_array_equal(np.array(entry["contrast"]["direction"]), contrast.direction)
        np.testing.assert_array_equal(_decode(entry["contrast"]["standardized"]), contrast.standardized)
        assert entry["contrast"]["norm"] == contrast.norm and entry["contrast"]["sizes"] == [4, 5]
        # The inline summary: norms and the dimensions of largest |standardized difference|, to 6 digits.
        summary = done.result["sites"][index]
        assert summary["site"] == site and summary["features"] == entry["features"]
        assert summary["contrast_norm"] == _six(contrast.norm)
        assert summary["group_mean_norms"] == [_six(entry["group_mean_norms"]["a"]), _six(entry["group_mean_norms"]["b"])]
        finite = np.isfinite(contrast.standardized)
        order = sorted(range(entry["features"]), key=lambda d: (bool(finite[d]), -abs(contrast.standardized[d])
                                                               if finite[d] else 0.0, d))
        assert [dim for dim, _ in summary["top"]] == order[:done.result["top"]]
        assert done.result["top"] == 8
    assert done.result["texts"] == [
        {"source": record["source"], "tokens": record["forward_tokens"],
         "window": record["window_stop"] - record["window_start"], "dropped": record["dropped_tokens"],
         "logprob": _six(record["mean_logprob"])} for record in document["texts"]]
    assert done.result["groups"] == {"a": 4, "b": 5}
    # The artifact is content-addressed: its URI is the SHA-256 of its canonical JSON.
    stored = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    assert done.result["result_uri"] == "wse://analyses/" + json_hash(document).removeprefix("sha256:")
    assert done.result["result_bytes"] == len(stored) == done.usage.artifact_bytes
    assert done.result["teacher_forced_tokens"] == sum(len(tokens) for tokens, _, _ in expected)


def test_probes_without_groups_report_one_mean_per_site(lm, monkeypatch):
    from wse.probes import ProbeText, probe

    run_id, eid = _open_run(lm.workspace, lm.base)
    strings = ["abc def", "loop loop", "the answer is 4"]
    done, document = _probe(lm, run_id, eid, [{"text": text} for text in strings], ["residual:2", "model.norm:output"])
    handle, expected = _expected_texts(lm, [], CONTEXT, strings=strings)
    direct = probe(handle, [ProbeText(tokens, start=start) for tokens, start, _ in expected],
                   sites=["residual:2", "model.norm:output"], device="cpu")
    for index, site in enumerate(["residual:2", "model.norm:output"]):
        entry = document["sites"][index]
        assert set(entry) == {"site", "requested", "features", "mean", "mean_norm"}
        np.testing.assert_array_equal(np.array(entry["mean"]), direct.group_mean(site, [0, 1, 2]))
        assert done.result["sites"][index] == {"site": site, "features": 16, "mean_norm": _six(entry["mean_norm"])}
    assert document["groups"] is None and done.result["groups"] is None
    # A string's window starts after its first token, so every window token is scored from a prefix.
    assert [record["window_start"] for record in document["texts"]] == [1, 1, 1]


# ---------------------------------------------------------------------------
# Vector references
# ---------------------------------------------------------------------------


def test_a_probe_contrast_direction_by_reference_is_accepted_by_a_project_out_preview(lm, monkeypatch):
    from wse.operators.tensor import shortest_floats
    from wse.patches import materialize_patch
    from wse.policy import builtin_experiment_policies

    run_id, eid, report = _pass(lm, monkeypatch)
    samples = _samples(report)
    done, document = _probe(lm, run_id, eid, [{"output_uri": s.output_uri} for s in samples],
                            [f"{DOWN0}:input", f"{DOWN0}:output"], groups=_groups_by_outcome(samples))
    uri = done.result["result_uri"]
    # The inputs of mlp.down that separate the groups most are columns of its weight; the contrast at its output
    # lies on the weight's output axis, so it is a direction that project_out removes from those columns.
    columns = sorted(dim for dim, _ in done.result["sites"][0]["top"][:4])
    reference = f"{uri}#/sites/1/contrast/direction"

    def spec(direction: str) -> dict:
        return {"base_checkpoint_id": lm.base, "hypothesis": "The limit-hit outputs differ along this direction.",
                "operations": [{"op": "project_out", "tensor": f"{DOWN0}.weight", "selection": FULL,
                                "direction": direction, "columns": columns}]}

    with Engine(lm.workspace, run_id=run_id) as bound:
        preview = bound.preview_patch(eid, spec(reference), "project")
        assert preview.changed_elements > 0 and str(preview.operations[0].operator) == "project_out@1"
        # Analyses take the same references: the columns of mlp.down most aligned with the contrast.
        aligned = bound.analyze_model(eid, lm.base, "direction_projection", f"{DOWN0}.weight", FULL,
                                      parameters={"direction": reference, "axis": "columns", "measure": "cosine"})
        assert aligned.result["direction_length"] == 16 and aligned.usage.analysis_calls == 1
        # A vector of the wrong length (the 32 inputs of mlp.down) is refused by the operator, and a group mean is
        # a vector too.
        error = _expect(ErrorCode.INVALID_PATCH, bound.preview_patch, eid,
                        spec(f"{uri}#/sites/0/contrast/direction"), "wrong-length")
        assert "one value per row" in error.message
        assert bound.preview_patch(eid, spec(f"{uri}#/sites/1/group_means/a"), "mean").changed_elements > 0
        # The run's view of the patch records the reference with the length and hash of the vector it named.
        shown = bound.find_patch(preview.patch_id)
        vector = np.asarray(document["sites"][1]["contrast"]["direction"], dtype=np.float32)
        record = {"reference": reference, "length": 16, "sha256": json_hash(shortest_floats(vector))}
        assert shown.spec.operations[0].parameters["direction"] == record
    with Engine(lm.workspace) as engine:
        step = engine.read_patch(eid, preview.patch_id).patch.steps[0]
        assert step.parameters["direction"] == record
        handle = engine.open_checkpoint(lm.base)
    # The same edit with the vector inline (its float32 values) gives the same weights.
    inline = spec(reference)["operations"][0] | {"direction": shortest_floats(vector)}
    operations = builtin_experiment_policies()["qwen_loop_v1"].operations
    assert materialize_patch(handle, [inline], policy=operations).patch.result_weights_hash == \
        preview.result_weights_hash


def test_standardized_differences_encode_infinities_and_are_not_vectors(lm, monkeypatch):
    run_id, eid = _open_run(lm.workspace, lm.base)
    # Identical texts in each group: zero pooled variance, so every dimension that differs is an infinity.
    texts = [{"text": "abc abc"}, {"text": "abc abc"}, {"text": "xyz xyz"}]
    done, document = _probe(lm, run_id, eid, texts, ["residual:1"], groups={"a": [0, 1], "b": [2]})
    standardized = document["sites"][0]["contrast"]["standardized"]
    assert set(standardized) <= {"inf", "-inf"} and len(standardized) == 16
    assert document["sites"][0]["contrast"]["nonfinite_standardized"] == 16
    assert all(z in ("inf", "-inf") for _, z in done.result["sites"][0]["top"])
    with Engine(lm.workspace, run_id=run_id) as bound:
        spec = {"base_checkpoint_id": lm.base, "hypothesis": "Standardized differences are not a direction.",
                "operations": [{"op": "project_out", "tensor": f"{DOWN0}.weight", "selection": FULL,
                                "direction": f"{done.result['result_uri']}#/sites/0/contrast/standardized",
                                "columns": [0]}]}
        error = _expect(ErrorCode.INVALID_PATCH, bound.preview_patch, eid, spec, "standardized")
        assert "list of numbers" in error.message


# ---------------------------------------------------------------------------
# Run scoping
# ---------------------------------------------------------------------------


def test_probe_artifacts_and_probed_outputs_are_run_scoped(lm, monkeypatch):
    run_id, eid, report = _pass(lm, monkeypatch)
    samples = _samples(report)
    done, _ = _probe(lm, run_id, eid, [{"output_uri": s.output_uri} for s in samples[:3]], [f"{DOWN0}:output"],
                     groups={"a": [0], "b": [1, 2]})
    uri = done.result["result_uri"]
    missing = "wse://analyses/" + "0" * 64
    other_run, other_eid = _open_run(lm.workspace, lm.base)
    with Engine(lm.workspace, run_id=other_run) as other:
        before = _budget(other, other_eid)
        _expect(ErrorCode.NOT_FOUND, other.read_analysis_result, uri)
        # Another run's outputs are refused exactly like outputs that do not exist, before anything is reserved.
        absent = _expect(ErrorCode.NOT_FOUND, other.submit_probe, other_eid, lm.base, [{"output_uri": missing}],
                         ["residual:1"], "absent").message
        error = _expect(ErrorCode.NOT_FOUND, other.submit_probe, other_eid, lm.base,
                        [{"output_uri": samples[0].output_uri}], ["residual:1"], "foreign")
        assert error.message.replace(samples[0].output_uri, missing) == absent
        # Its probe vectors are not references this run can use.
        spec = {"base_checkpoint_id": lm.base, "hypothesis": "Another run's direction.",
                "operations": [{"op": "project_out", "tensor": f"{DOWN0}.weight", "selection": FULL,
                                "direction": f"{uri}#/sites/0/contrast/direction", "columns": [0]}]}
        _expect(ErrorCode.NOT_FOUND, other.preview_patch, other_eid, spec, "foreign-direction")
        assert _budget(other, other_eid) == before and other.list_jobs(other_eid) == []
    with Engine(lm.workspace, run_id=run_id) as bound:
        assert bound.read_analysis_result(uri)["kind"] == "probe_result"
        # A probe artifact is no generation output: it cannot be probed as a text.
        error = _expect(ErrorCode.INVALID_ARGUMENT, bound.submit_probe, eid, lm.base, [{"output_uri": uri}],
                        ["residual:1"], "not-an-output")
        assert "generation output" in error.message
    with Engine(lm.workspace) as engine:
        owners = engine.store.db.conn.execute("SELECT experiment_id FROM analysis_results WHERE uri = ?",
                                              (uri,)).fetchall()
        assert [row["experiment_id"] for row in owners] == [eid]


# ---------------------------------------------------------------------------
# Refusals, limits, and stops
# ---------------------------------------------------------------------------


def test_invalid_probe_requests_are_refused_before_anything_is_reserved(lm, monkeypatch):
    from wse.policy import SMALL_MODEL_POLICY_ID

    run_id, eid, report = _pass(lm, monkeypatch)
    output = _samples(report)[0].output_uri
    with Engine(lm.workspace, run_id=run_id) as bound:
        before = _budget(bound, eid)

        def refused(code: ErrorCode, texts, sites, key: str, **options):
            return _expect(code, bound.submit_probe, eid, lm.base, texts, sites, key, **options)

        good = [{"text": "abc"}, {"text": "def"}, {"output_uri": output}]
        # Malformed texts, sites, and groups.
        for texts in ([], "abc", [{"text": ""}], [{"text": "a" * 8193}], [{"text": "a", "output_uri": output}],
                      [{"output_uri": "wse://analyses/x"}], [{"suite_id": "practice"}], [{"suite_id": "practice",
                                                                                          "item": -1}], [{}], [1]):
            refused(ErrorCode.INVALID_ARGUMENT, texts, ["residual:1"], "bad-texts")
        for sites in ([], "residual:1", ["residual"], ["residual:1", "residual:1"], [{"role": "mlp.down"}],
                      [{"role": "mlp.down", "layer": 0, "point": "weight"}], [1]):
            refused(ErrorCode.INVALID_ARGUMENT, good, sites, "bad-sites")
        for groups in ({"a": [0]}, {"a": [0], "b": []}, {"a": [0], "b": [0, 1]}, {"a": [0], "b": [1]},
                       {"a": [0, 3], "b": [1]}, {"a": [0, 0], "b": [1, 2]}, {"a": [True], "b": [1, 2]}):
            refused(ErrorCode.INVALID_ARGUMENT, good, ["residual:1"], "bad-groups", groups=groups)
        refused(ErrorCode.INVALID_ARGUMENT, good, ["residual:1"], "bad-top", top=0)
        refused(ErrorCode.INVALID_ARGUMENT, good, ["residual:1"], "bad-top", top=65)
        # A string that encodes to one token has no scored window: refused at submission, with the checkpoint's
        # stored tokenizer, not in the worker after the reservation and the model load.
        error = refused(ErrorCode.INVALID_ARGUMENT, [*good, {"text": "a"}], ["residual:1"], "one-token")
        assert error.details == {"text": 3} and "encodes to 1 token" in error.message
        # Unknown sites, roles, layers, and suite items are NOT_FOUND, with candidates for sites.
        error = refused(ErrorCode.NOT_FOUND, good, ["model.layers.0.mlp.down_projj:input"], "unknown-site")
        assert "model.layers.0.mlp.down_proj" in error.details["candidates"]
        refused(ErrorCode.NOT_FOUND, good, ["residual:3"], "beyond")
        refused(ErrorCode.NOT_FOUND, good, [{"role": "mlp.down", "layer": 2, "point": "input"}], "no-layer")
        refused(ErrorCode.NOT_FOUND, [{"suite_id": "practice", "item": 4}, *good], ["residual:1"], "item")
        refused(ErrorCode.NOT_FOUND, [{"suite_id": "nope", "item": 0}, *good], ["residual:1"], "suite")
        # More texts than a probe may take under the policy (64).
        error = refused(ErrorCode.POLICY_DENIED, [{"text": f"t{i}"} for i in range(65)], ["residual:1"], "many")
        assert error.details["texts_per_probe_call"] == 64
        assert _budget(bound, eid) == before and [job.kind for job in bound.list_jobs(eid)] == ["evaluation"]
        failures = [event["event_type"] for event in bound.audit_log(eid) if event["event_type"] == "job.submit.failed"]
        assert failures
    # Under an M1 policy, which leaves the probe counters unset, no probe is permitted at all.
    other_run, other_eid = _open_run(lm.workspace, lm.base, policy_id=SMALL_MODEL_POLICY_ID)
    with Engine(lm.workspace, run_id=other_run) as bound:
        error = _expect(ErrorCode.POLICY_DENIED, bound.submit_probe, other_eid, lm.base, [{"text": "abc"}],
                        ["residual:1"], "m1")
        assert "probe" in error.message and bound.list_jobs(other_eid) == []


def test_overlong_sites_are_invalid_arguments_on_every_path(lm):
    """A residual boundary of more than 9 digits (Python converts at most 4,300 to an int), an overlong site name,
    and a role site's huge layer are INVALID_ARGUMENT, with a JSON body, through the Python API, a strategy's
    executor, and the MCP handler, before anything is reserved; no error echoes the whole site."""
    from mcp.server.mcpserver.exceptions import ToolError

    from wse.mcp_server import PROBE_SITES_SCHEMA, WSEMCPServer
    from wse.probes import MAX_SITE_CHARACTERS
    from wse.runner import ExperimentExecutor
    from wse.strategies.base import Action

    assert PROBE_SITES_SCHEMA["items"]["anyOf"][0]["maxLength"] == MAX_SITE_CHARACTERS
    run_id, eid = _open_run(lm.workspace, lm.base)
    texts = [{"text": "abc"}, {"text": "def"}]
    cases = (["residual:" + "1" * 5000], ["residual:1234567890"], ["residual:1", "m" * MAX_SITE_CHARACTERS + ":input"],
             ["x" * 100_000], [{"role": "mlp.down", "layer": 10 ** 9, "point": "input"}],
             [{"role": "mlp.down", "layer": 10 ** 5000, "point": "input"}])
    with Engine(lm.workspace, run_id=run_id) as bound:
        before = _budget(bound, eid)
        executor = ExperimentExecutor(bound, constraint_status="verified")
        server = WSEMCPServer(bound)
        for number, sites in enumerate(cases):
            error = _expect(ErrorCode.INVALID_ARGUMENT, bound.submit_probe, eid, lm.base, texts, sites, f"py-{number}")
            assert len(error.message) < 1000 and len(json.dumps(error.details)) < 1000
            observation = executor.execute(Action.call("wse_probe", checkpoint_id=lm.base, texts=texts, sites=sites,
                                                       idempotency_key=f"strategy-{number}"))
            assert observation.error_code == "INVALID_ARGUMENT"
            with pytest.raises(ToolError) as raised:
                server.wse_probe(experiment_id=eid, checkpoint_id=lm.base, texts=texts, sites=sites,
                                 idempotency_key=f"mcp-{number}")
            body = json.loads(str(raised.value))["error"]
            assert body["code"] == "INVALID_ARGUMENT" and len(str(raised.value)) < 2000
        # Within the bounds, a boundary beyond the model's layers is still NOT_FOUND.
        _expect(ErrorCode.NOT_FOUND, bound.submit_probe, eid, lm.base, texts, ["residual:999999999"], "beyond")
        assert _budget(bound, eid) == before and bound.list_jobs(eid) == []


def test_probe_retries_return_the_same_job_and_other_input_conflicts(lm):
    run_id, eid = _open_run(lm.workspace, lm.base)
    texts = [{"text": "abc"}, {"text": "def"}, {"text": "ghi"}]
    with Engine(lm.workspace, run_id=run_id) as bound:
        job = bound.submit_probe(eid, lm.base, texts, ["residual:1"], "k", groups={"a": [0], "b": [1, 2]})
        held = _budget(bound, eid)
        assert job.reservation.requested == {"probe_calls": 1, "teacher_forced_tokens": 3 * 1024,
                                             "compute_seconds": 7200}
        assert bound.submit_probe(eid, lm.base, texts, ["residual:1"], "k", groups={"a": [0], "b": [1, 2]}) == job
        assert _budget(bound, eid) == held
        _expect(ErrorCode.IDEMPOTENCY_CONFLICT, bound.submit_probe, eid, lm.base, texts, ["residual:2"], "k")
        assert [listed.job_id for listed in bound.list_jobs(eid)] == [job.job_id]


def test_a_probe_summary_stays_within_the_response_limit(lm):
    """64 texts at every capturable site of the fixture fit the inline limit; requests that cannot fit are refused."""
    layers = [f"model.layers.{i}" for i in range(2)]
    modules = [f"{layer}.{name}" for layer in layers
               for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "mlp", "input_layernorm",
                            "post_attention_layernorm")]
    sites = [f"{module}:{point}" for module in modules for point in ("input", "output")]
    sites += ["residual:0", "residual:1", "residual:2", "model.norm:input", "model.norm:output", "lm_head:output"]
    texts = [{"text": "abc " * (1 + i % 7)} for i in range(64)]
    groups = {"a": list(range(32)), "b": list(range(32, 64))}

    def limited(policy_id: str, inline_bytes: int):
        policy = _generation_policy(policy_id)
        return policy.model_copy(update={"operations": policy.operations.with_limits(
            max_inline_json_bytes=inline_bytes)})

    for policy, limit in ((None, INLINE_RESULT_LIMIT), (limited("smaller_inline_v1", 20 * 1024), 19 * 1024)):
        run_id, eid = _open_run(lm.workspace, lm.base, policy)
        done, document = _probe(lm, run_id, eid, texts, sites, groups=groups, top=64)
        assert len(done.canonical_json().encode("utf-8")) <= limit
        assert len(done.result["sites"]) == len(sites) == len(document["sites"])
        # Every site lists the same number of top dimensions, or all of its dimensions when it has fewer.
        top = done.result["top"]
        assert [len(site["top"]) for site in done.result["sites"]] == [min(top, site["features"])
                                                                       for site in done.result["sites"]]
        assert document["sites"][-1]["features"] == 64  # the output head's sites see every position's logits
        if policy is not None:
            assert 0 < top < 64  # the top entries shrank to fit the smaller limit
    # A policy whose inline limit leaves too little room refuses the request at submission.
    other_run, other_eid = _open_run(lm.workspace, lm.base, limited("tight_inline_v1", 8 * 1024))
    with Engine(lm.workspace, run_id=other_run) as bound:
        error = _expect(ErrorCode.INVALID_ARGUMENT, bound.submit_probe, other_eid, lm.base, texts, sites, "big",
                        groups=groups)
        assert "inline" in error.message and error.details["max_sites"] < len(sites)
        assert bound.list_jobs(other_eid) == []


def test_the_summary_bound_holds_for_the_widest_numbers_a_summary_writes():
    """Numbers to 6 significant digits print as up to 19 characters (-1234560000000000.0), and the bound allows that.

    A probe that passes the submission bound must never fail in the worker's ``fit_summary`` (after its forwards
    were charged), whatever norms and log-probabilities its activations give.
    """
    from wse.probes import _json_bytes, _summary, fit_summary, summary_size_bound

    names = [f"model.language_model.layers.{layer}.{module}:{point}" for layer in range(24)
             for module in ("post_attention_layernorm", "mlp.down_proj", "mlp.gate_proj") for point in ("input",
                                                                                                    "output")]
    values = (12.3456789, 1.2345678e15, 9.99999e15, 1e16, 1.23456e-300, 0.000123456)
    for grouped in (True, False):
        for count in (3, 64):
            bounds = summary_size_bound(texts=count, sites=names, grouped=grouped, tokens_per_text=1024)
            for norm in values:
                texts = [{"source": "suite_item", "forward_tokens": 1024, "window_start": 1, "window_stop": 1024,
                          "dropped_tokens": 123456789, "mean_logprob": -norm}] * count
                sites = [{"site": name, "features": 3584,
                          **({"group_mean_norms": {"a": norm, "b": norm}, "contrast": {"norm": norm}} if grouped
                             else {"mean_norm": norm})} for name in names]
                document = {"checkpoint_id": "ckpt_" + "a" * 32, "tokens_per_text": 1024, "texts": texts,
                            "sites": sites, "groups": {"a": [0], "b": [1, 2]} if grouped else None}
                for k in (0, 1, 47, len(names)):
                    shown = {**document, "sites": sites[:k]}
                    size = _json_bytes(_summary(shown, [[] if grouped else None] * k, 0, "wse://analyses/" + "a" * 64,
                                                10 ** 12))
                    assert size <= bounds[k], (grouped, count, norm, k, size, bounds[k])
                # At the bound, the worker's summary always fits (here without top dimensions to drop).
                fit_summary({**document, "sites": [{**site, "contrast": {**site["contrast"], "standardized": []}}
                                                   if grouped else site for site in sites]},
                            uri="wse://analyses/" + "a" * 64, size=10 ** 12, top=8, limit=bounds[-1])


def test_probes_stop_on_cancellation_and_at_the_time_limit_and_stay_charged(lm, monkeypatch):
    from wse.engines import hf_torch

    original = hf_torch.HFTorchModel.next_token_logprobs
    started = threading.Event()

    def slow(self, input_ids, **kwargs):
        started.set()
        time.sleep(0.15)
        return original(self, input_ids, **kwargs)

    monkeypatch.setattr(hf_torch.HFTorchModel, "next_token_logprobs", slow)
    texts = [{"text": f"text number {i}"} for i in range(30)]
    run_id, eid = _open_run(lm.workspace, lm.base)
    with Engine(lm.workspace, run_id=run_id) as bound:
        job = bound.submit_probe(eid, lm.base, texts, ["residual:1"], "slow")
        with _worker(lm.workspace) as worker:
            runner = threading.Thread(target=worker.run_once)
            runner.start()
            try:
                assert started.wait(30)
                assert bound.cancel_job(eid, job.job_id, "cancel").cancel_requested
                done = bound.get_job(eid, job.job_id, wait_seconds=10)
            finally:
                runner.join()
        assert done.state == "cancelled" and done.error.details["reason"] == "cancelled" and done.result is None
        assert done.usage.probe_calls == 1 and 0 < done.usage.teacher_forced_tokens < 30 * 14
        assert done.reservation.charged == {"probe_calls": 1, "teacher_forced_tokens": done.usage.teacher_forced_tokens,
                                            "compute_seconds": done.usage.compute_seconds}
        assert done.progress.batches_total == 30 and 0 < done.progress.batches_started < 30
        budget = _budget(bound, eid)
        assert budget["probe_calls"][1:] == (1, 0) and budget["teacher_forced_tokens"][1:] == (
            done.usage.teacher_forced_tokens, 0)
    with Engine(lm.workspace) as engine:  # a stopped probe leaves no artifact the run could read
        assert engine.store.db.conn.execute("SELECT COUNT(*) FROM analysis_results WHERE experiment_id = ?",
                                            (eid,)).fetchone()[0] == 0
    # The per-operation time limit stops a probe the same way.
    run_id, eid = _open_run(lm.workspace, lm.base, _generation_policy("short_probes_v1", seconds_per_operation=1.0))
    with Engine(lm.workspace, run_id=run_id) as bound:
        job = bound.submit_probe(eid, lm.base, texts, ["residual:1"], "slow")
        assert job.reservation.requested["compute_seconds"] == 1
        with _worker(lm.workspace) as worker:
            worker.run_once()
        done = bound.get_job(eid, job.job_id)
        assert done.state == "failed" and done.error.details["reason"] == "timeout"
        assert done.usage.compute_seconds == pytest.approx(1.0) and done.usage.probe_calls == 1
        assert 0 < done.usage.teacher_forced_tokens < 30 * 14


# ---------------------------------------------------------------------------
# The engine: logits for the window only, and one forward at a time
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def llama(hf_snapshots, tmp_path_factory):
    with Engine(tmp_path_factory.mktemp("probe_engine") / "workspace") as engine:
        checkpoint = engine.import_hf_snapshot(hf_snapshots["llama_untied"], repo="example-org/tiny-model",
                                               revision=REVISION).checkpoint_id
        return engine.open_checkpoint(checkpoint)


def test_teacher_forced_reads_compute_logits_only_for_the_scored_positions(llama, monkeypatch):
    runtime = llama.runtime("cpu")
    kept: list = []
    original = type(runtime.module).forward

    def recording(module, *args, **kwargs):
        output = original(module, *args, **kwargs)
        kept.append((kwargs.get("logits_to_keep"), tuple(output.logits.shape)))
        return output

    monkeypatch.setattr(type(runtime.module), "forward", recording)
    ids = torch.randint(3, 64, (1, 30), generator=torch.Generator().manual_seed(3))
    read = runtime.next_token_logprobs(ids, start=5)
    # Positions 5-29 have logits; the last one scores nothing, so 25 rows are kept, not 30.
    assert kept[-1] == (25, (1, 25, 64))
    logprobs = torch.log_softmax(runtime.logits(ids)[0], dim=-1)
    assert kept[-1][0] in (None, 0) and kept[-1][1] == (1, 30, 64)
    assert torch.equal(read, torch.stack([logprobs[t, ids[0, t + 1]] for t in range(5, 29)]))
    # A hook on the output head sees every position, so probe sites on the head stay whole.
    hook = runtime.module.get_output_embeddings().register_forward_hook(lambda module, args, output: None)
    try:
        assert torch.equal(runtime.next_token_logprobs(ids, start=5), read)
    finally:
        hook.remove()
    assert kept[-1][1] == (1, 30, 64)


def test_a_probe_and_a_concurrent_generation_on_one_runtime_do_not_interleave(llama):
    from wse.probes import ProbeText, probe

    runtime = llama.runtime("cpu")
    generator = torch.Generator().manual_seed(5)
    texts = [ProbeText(tuple(torch.randint(3, 64, (length,), generator=generator).tolist()), start=1)
             for length in (40, 60, 80)]
    sites = ["residual:1", f"{DOWN0}:input"]
    expected = probe(llama, texts, sites=sites, device="cpu")
    stop, errors = threading.Event(), []

    def generate() -> None:
        try:
            while not stop.is_set():
                runtime.generate(torch.tensor([[5, 6, 7, 8]]), max_new_tokens=6)
        except Exception as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    thread = threading.Thread(target=generate)
    thread.start()
    try:
        for _ in range(8):
            again = probe(llama, texts, sites=sites, device="cpu")
            for site in sites:
                np.testing.assert_array_equal(again.means[site], expected.means[site])
            assert again.mean_logprobs == expected.mean_logprobs
    finally:
        stop.set()
        thread.join()
    assert errors == []


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


def test_probes_and_direction_edits_need_capabilities_that_m1_models_lack(workspace, bundles, signed_sum):
    from wse.adapters.base import CAP_PROBE_HIDDEN_STATES, CAP_WEIGHTS_DIRECTION_EDIT
    from wse.adapters.mlp import MLPAdapter
    from wse.api import apply_operator, load_model
    from wse.policy import ExperimentPolicy, builtin_experiment_policies, default_operation_policy
    from wse.probes import ProbeText, probe
    from wse.registry import builtin_registry

    registry = builtin_registry()
    hf = registry.resolve("adapter", "hf_causal_lm", "1").capabilities()
    assert hf.supports(CAP_PROBE_HIDDEN_STATES) and hf.supports(CAP_WEIGHTS_DIRECTION_EDIT)
    assert MLPAdapter().capabilities().capabilities == frozenset({"weights.read", "weights.slice_edit",
                                                                   "forward.logits"})
    for ref in ("project_out@1", "add_low_rank@1"):
        assert CAP_WEIGHTS_DIRECTION_EDIT in registry.resolve_ref("operator", ref).spec.required_capabilities
    assert registry.resolve_ref("operator", "fill_slice@1").spec.required_capabilities == ["weights.read",
                                                                                          "weights.slice_edit"]
    # Direct use: M1's MLP adapter supports neither probes nor direction edits; slice edits still work.
    m1 = load_model(signed_sum.task_tensors(), "mlp@1", signed_sum.CONFIG)
    selection = [{"start": 0, "stop": 4, "step": 1}, {"start": 0, "stop": 2, "step": 1}]
    for operator, parameters in (("project_out@1", {"direction": [1.0, 0.0, 0.0, 0.0], "columns": [0]}),
                                 ("add_low_rank@1", {"u": [[1.0] * 4], "v": [[1.0, 0.0]]})):
        error = _expect(ErrorCode.UNSUPPORTED_CAPABILITY, apply_operator, m1, operator,
                        {"tensor": "layers.0.weight", "selection": selection, **parameters})
        assert error.details["missing_capabilities"] == [CAP_WEIGHTS_DIRECTION_EDIT]
    assert apply_operator(m1, "fill_slice@1", {"tensor": "layers.0.weight", "selection": selection,
                                               "value": 0.0}).patch.steps[0].changed_elements > 0
    error = _expect(ErrorCode.UNSUPPORTED_CAPABILITY, probe, m1, [ProbeText((1, 2, 3))], sites=["residual:0"],
                    device="cpu")
    assert CAP_PROBE_HIDDEN_STATES in error.message
    # Managed: a bounded_eval policy that grants probes and the direction edits is refused on an M1 checkpoint
    # before anything is reserved or stored.
    policy = ExperimentPolicy(
        policy_id="m1_probes_v1", mode="bounded_eval",
        budget={"probe_calls": 2, "texts_per_probe_call": 4, "tokens_per_probe_text": 16, "teacher_forced_tokens": 64},
        operations=default_operation_policy().with_edit("project_out", "project_out@1"))
    with Engine(workspace) as engine:
        engine.register_policy(policy)
        base = engine.import_bundle(bundles["task"]).checkpoint_id
        grant = engine.grant_run("signed_sum_v1", policy.policy_id, base)
    with Engine(workspace, run_id=grant.run_id) as bound:
        eid = bound.create_experiment(grant.run_id, "signed_sum_v1", policy.policy_id, "create").experiment_id
        before = _budget(bound, eid)
        error = _expect(ErrorCode.UNSUPPORTED_CAPABILITY, bound.submit_probe, eid, base, [{"text": "abc"}],
                        ["residual:0"], "probe")
        assert CAP_PROBE_HIDDEN_STATES in error.message
        spec = {"base_checkpoint_id": base, "hypothesis": "Removing a direction changes the outputs.",
                "operations": [{"op": "project_out", "tensor": "layers.0.weight", "selection": selection,
                                "direction": [1.0, 0.0, 0.0, 0.0], "columns": [0]}]}
        _expect(ErrorCode.UNSUPPORTED_CAPABILITY, bound.preview_patch, eid, spec, "project")
        assert _budget(bound, eid) == before and bound.list_jobs(eid) == []
    # The M1 policies keep their hashes.
    policies = builtin_experiment_policies()
    assert policies["small_model_v1"].policy_hash == \
        "sha256:d8761471bfb88e1d322da1ca06003729886daafa028963f1dcf7ef807cc218b5"
    assert policies["small_model_weights_only_v1"].policy_hash == \
        "sha256:88d3c1e4f73e0eb8204c30b65aceb753608114c4ed05e074302db630e665fa75"


# ---------------------------------------------------------------------------
# Migration 11
# ---------------------------------------------------------------------------

M10_COMMIT = "7d0b2c9"  # a commit whose code writes schema version 10 (migration 11 came after it)
M10_WRITER = """
import importlib.util, json, sys
from pathlib import Path
import wse
from wse.api import Engine
from wse.worker import Worker
src, fixture, workspace, bundles = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
assert Path(wse.__file__).resolve().is_relative_to(src.resolve()), wse.__file__
spec = importlib.util.spec_from_file_location("build_fixture", fixture)
signed_sum = importlib.util.module_from_spec(spec)
spec.loader.exec_module(signed_sum)
with Engine(workspace) as engine:
    assert engine.store.db.schema_version() == 10
    ids = signed_sum.setup_workspace(engine, bundles)
    grant = engine.grant_run(ids["task_id"], ids["policy_id"], ids["task_checkpoint_id"])
    eid = engine.create_experiment(grant.run_id, ids["task_id"], ids["policy_id"], "create").experiment_id
    first = engine.submit_evaluation(eid, ids["task_checkpoint_id"], "signed_sum_dev_v1",
                                     {"sample_count": 64, "seed": 0}, "first").job_id
with Worker(workspace) as worker:
    worker.run_once()
with Engine(workspace) as engine:
    second = engine.submit_evaluation(eid, ids["task_checkpoint_id"], "signed_sum_dev_v1",
                                      {"sample_count": 64, "seed": 0}, "second").job_id
print(json.dumps({"experiment_id": eid, "jobs": [first, second]}))
"""


def _table_rows(workspace: Path, table: str) -> list[tuple]:
    with sqlite3.connect(workspace / "metadata.sqlite") as conn:
        return conn.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()


def test_migration_11_keeps_jobs_and_reports_written_by_schema_10_code(workspace, tmp_path):
    import os

    from wse.database import LATEST_SCHEMA_VERSION

    source = _source_at(M10_COMMIT, tmp_path / "m10")
    if source is None:
        pytest.skip(f"the schema-10 code (commit {M10_COMMIT}) is not available from git history")
    fixture = Path(__file__).resolve().parents[1] / "examples" / "signed_sum" / "build_fixture.py"
    written = subprocess.run([sys.executable, "-c", M10_WRITER, str(source), str(fixture), str(workspace),
                              str(tmp_path / "bundles")], capture_output=True, text=True,
                             env={**os.environ, "PYTHONPATH": str(source)}, cwd=tmp_path, timeout=300)
    assert written.returncode == 0, written.stderr
    ids = json.loads(written.stdout)
    jobs, reports = _table_rows(workspace, "jobs"), _table_rows(workspace, "reports")
    assert [row[0] for row in jobs] == sorted(ids["jobs"]) and len(reports) == 1
    with Engine(workspace) as engine:
        assert engine.store.db.schema_version() == LATEST_SCHEMA_VERSION >= 11
        conn = engine.store.db.conn
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        # Every job and report is kept as it was, and the jobs read back as before.
        assert _table_rows(workspace, "jobs") == jobs and _table_rows(workspace, "reports") == reports
        listed = engine.list_jobs(ids["experiment_id"])
        assert [(job.kind, job.state) for job in listed] == [("evaluation", "succeeded"), ("evaluation", "queued")]
        assert listed[0].report_id == reports[0][0] and listed[0].config.sample_count == 64
        indexes = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index' AND "
                                                  "tbl_name = 'jobs' AND sql IS NOT NULL")}
        assert indexes == {"jobs_state", "jobs_experiment"}
        # The table now accepts probe jobs, which name no suite; an evaluation job still needs one.
        sql = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'jobs'").fetchone()[0]
        assert "'probe'" in sql
        with pytest.raises(sqlite3.IntegrityError):
            with engine.store.db.transaction() as tx:
                tx.execute("UPDATE jobs SET suite_id = NULL WHERE job_id = ?", (ids["jobs"][1],))
        # The queued job still runs.
        assert engine.get_job(ids["experiment_id"], ids["jobs"][1]).state == "queued"
    from wse.worker import Worker

    with Worker(workspace) as worker:
        assert worker.run_once() == ids["jobs"][1]
    with Engine(workspace) as engine:
        assert engine.get_job(ids["experiment_id"], ids["jobs"][1]).state == "succeeded"


# ---------------------------------------------------------------------------
# The runner and strategies
# ---------------------------------------------------------------------------


def test_the_runner_exposes_generation_passes_and_probes_under_qwen_loop_v1():
    from wse.policy import builtin_experiment_policies
    from wse.runner import EVALUATION_TOOLS, allowed_tools, render_instructions
    from wse.schemas import TaskDefinition
    from wse.strategies.base import STRATEGY_TOOLS

    policies = builtin_experiment_policies()
    sandbox_tools = ("wse_run_python", "wse_submit_candidate")
    assert "wse_probe" in STRATEGY_TOOLS and set(sandbox_tools) <= set(STRATEGY_TOOLS)
    m1 = [tool for tool in STRATEGY_TOOLS if tool not in ("wse_probe", *sandbox_tools)]
    assert allowed_tools(policies["small_model_v1"]) == m1
    assert allowed_tools(policies["small_model_weights_only_v1"]) == [tool for tool in m1
                                                                      if tool not in EVALUATION_TOOLS]
    qwen = policies["qwen_loop_v1"]
    tools = allowed_tools(qwen)
    assert tools == [tool for tool in STRATEGY_TOOLS if tool not in sandbox_tools]  # the sandbox needs its policy
    task = TaskDefinition.model_validate({
        "task_id": TASK, "title": "Finish answers", "description": "Make the model finish its answers.",
        "objective": {"task_id": TASK, "description": "Fewer outputs at the token limit.",
                      "gates": [{"metric": "accuracy", "min_value": 0.5}]},
        "development_suite_id": "practice", "development_examples": 4})
    text = render_instructions(task, qwen, experiment_id="exp_0123", tools=tools)
    assert "not available" not in text and ", ".join(tools) in text
    for fragment in ("generation passes: 10", "at most 32 problems", "8192 new tokens", "probe calls: 25",
                     "at most 64 texts", "1024 tokens", "wse_probe", "wse_submit_evaluation", "limit"):
        assert fragment in text, fragment
    assert "evaluation batches" not in text
    m1_text = render_instructions(task, policies["small_model_v1"], experiment_id="exp_0123",
                                  tools=allowed_tools(policies["small_model_v1"]))
    assert "wse_probe" not in m1_text and "generation passes" not in m1_text and "evaluation batches" in m1_text


class LoopStrategy:
    """Through the runner's dispatch: a generation pass, a probe, an analysis by reference, and a preview by reference."""

    strategy_id, version = "loop_probe_test", "1"

    def __init__(self, checkpoint_id: str) -> None:
        self.checkpoint_id = checkpoint_id
        self.plan = iter(self._steps())
        self.last = None

    def config(self):
        return {"checkpoint_id": self.checkpoint_id}

    def _wait(self, eid, job_id):
        from wse.strategies.base import Action

        while True:
            observation = yield Action.call("wse_get_job", "Wait for the job.", experiment_id=eid, job_id=job_id,
                                            wait_seconds=10)
            if observation.result["state"] in ("succeeded", "failed", "cancelled"):
                return observation.result

    def _steps(self):
        from wse.strategies.base import Action

        eid = yield None
        submitted = yield Action.call("wse_submit_evaluation", "Generate on the practice problems.",
                                      experiment_id=eid, checkpoint_id=self.checkpoint_id, suite_id="practice",
                                      eval_config={"sample_count": 4, "seed": 0}, idempotency_key="pass")
        evaluated = yield from self._wait(eid, submitted.result["job_id"])
        assert evaluated["state"] == "succeeded", evaluated
        texts = [{"text": "loop loop loop loop"}, {"text": "loop loop"}, {"suite_id": "practice", "item": 0},
                 {"suite_id": "practice", "item": 1}]
        probed = yield Action.call("wse_probe", "Contrast looping text with the prompts.", experiment_id=eid,
                                   checkpoint_id=self.checkpoint_id, texts=texts,
                                   sites=[{"role": "mlp.down", "layer": 0, "point": "input"},
                                          {"role": "mlp.down", "layer": 0, "point": "output"}],
                                   groups={"a": [0, 1], "b": [2, 3]}, idempotency_key="probe")
        assert probed.ok, probed.error
        result = (yield from self._wait(eid, probed.result["job_id"]))["result"]
        reference = f"{result['result_uri']}#/sites/1/contrast/direction"
        analyzed = yield Action.call("wse_analyze", "Columns aligned with the contrast.", experiment_id=eid,
                                     checkpoint_id=self.checkpoint_id, kind="direction_projection",
                                     tensor=f"{DOWN0}.weight", selection=FULL,
                                     parameters={"direction": reference, "axis": "columns"})
        assert analyzed.ok, analyzed.error
        columns = sorted(entry["index"] for entry in analyzed.result["result"]["top"][:3])
        previewed = yield Action.call("wse_preview_patch", "Remove the contrast from the aligned columns.",
                                      experiment_id=eid, idempotency_key="preview", patch_spec={
                                          "base_checkpoint_id": self.checkpoint_id,
                                          "hypothesis": "Looping text differs along this direction.",
                                          "operations": [{"op": "project_out", "tensor": f"{DOWN0}.weight",
                                                          "selection": FULL, "direction": reference,
                                                          "columns": columns}]})
        assert previewed.ok, previewed.error
        yield Action.stop("done")

    def next_action(self, objective, observations, remaining_budget):
        if self.last is None:
            next(self.plan)
            self.last = self.plan.send(observations[0].arguments["experiment_id"])
            return self.last
        self.last = self.plan.send(observations[-1])
        return self.last


def test_a_strategy_runs_passes_probes_and_references_through_the_runner(lm):
    from wse.registry import builtin_registry
    from wse.runner import ReferenceRunner
    from wse.schemas import TaskDefinition

    with Engine(lm.workspace) as engine:
        engine.register_task(TaskDefinition.model_validate({
            "task_id": TASK, "title": "Finish answers", "description": "Make the model finish its answers.",
            "objective": {"task_id": TASK, "description": "Fewer outputs at the token limit.",
                          "gates": [{"metric": "accuracy", "min_value": 0.5}]},
            "development_suite_id": "practice", "development_examples": 4}))
        from wse.policy import QWEN_LOOP_POLICY_ID

        grant = engine.grant_run(TASK, QWEN_LOOP_POLICY_ID, lm.base)
    registry = builtin_registry()
    registry.register("strategy", "loop_probe_test", "1", LoopStrategy)
    runner = ReferenceRunner(lm.workspace, registry=registry, worker_args=["--poll-seconds", "0.02", "--device", "cpu"])
    eid = runner.run_experiment(grant.run_id, TASK, QWEN_LOOP_POLICY_ID, {
        "kind": "strategy", "strategy": "loop_probe_test@1", "parameters": {"checkpoint_id": lm.base}})
    result = runner.last_result["result"]
    assert result["status"] == "completed", result
    assert "wse_probe" in runner.last_result["config"]["allowed_tools"]
    with Engine(lm.workspace) as engine:
        experiment = engine.get_experiment(eid)
        assert experiment.state == "FROZEN"
        jobs = engine.list_jobs(eid)
        assert [(job.kind, job.state) for job in jobs] == [("evaluation", "succeeded"), ("probe", "succeeded")]
        budget = experiment.budget
        assert budget["generation_passes"].consumed == 1 and budget["probe_calls"].consumed == 1
        assert budget["analysis_calls"].consumed == 1 and budget["patch_previews"].consumed == 1
        assert budget["teacher_forced_tokens"].consumed == jobs[1].usage.teacher_forced_tokens > 0
        rows = engine.store.db.conn.execute("SELECT payload FROM controller_events WHERE experiment_id = ? AND "
                                            "kind = 'tool_call' ORDER BY seq", (eid,)).fetchall()
        tools = [json.loads(row["payload"])["tool"] for row in rows]
    assert tools[:2] == ["wse_get_experiment", "wse_submit_evaluation"]
    assert tools[-3:] == ["wse_get_job", "wse_analyze", "wse_preview_patch"] and "wse_probe" in tools


def test_action_shortfalls_know_generation_passes_and_probes():
    from wse.runner import _action_shortfall
    from wse.schemas import BudgetEntry
    from wse.strategies.base import Action

    def budget(**remaining):
        full = {"evaluation_batches": 0, "evaluated_examples": 0, "generation_passes": 10, "generated_tokens": 1000,
                "probe_calls": 25, "teacher_forced_tokens": 4096, "compute_seconds": 60.0, **remaining}
        return {name: BudgetEntry(limit=100, consumed=0, remaining=value) for name, value in full.items()}

    evaluate = Action.call("wse_submit_evaluation", experiment_id="e", suite_id="practice",
                           eval_config={"sample_count": 4, "seed": 0})
    probe = Action.call("wse_probe", experiment_id="e", texts=[{"text": "a"}] * 4)
    # A generation pass needs a pass and tokens, not evaluation batches (qwen_loop_v1 grants none).
    assert _action_shortfall(evaluate, budget(), generation=True) is None
    assert "needs generation_passes" in _action_shortfall(evaluate, budget(generation_passes=0), generation=True)
    assert "needs generated_tokens" in _action_shortfall(evaluate, budget(generated_tokens=0), generation=True)
    assert "needs evaluation_batches" in _action_shortfall(evaluate, budget())  # a forward-batch suite
    # A suite the run cannot resolve (a final suite, another task's, a typo): only compute time is checked, and the
    # submission itself reports the domain error, as it does through MCP.
    assert _action_shortfall(evaluate, budget(), generation=None) is None
    assert "needs compute_seconds" in _action_shortfall(evaluate, budget(compute_seconds=0), generation=None)
    # A probe needs a probe call and the tokens its texts reserve.
    assert _action_shortfall(probe, budget(), tokens_per_probe_text=1024) is None
    assert "needs probe_calls" in _action_shortfall(probe, budget(probe_calls=0), tokens_per_probe_text=1024)
    assert "needs teacher_forced_tokens" in _action_shortfall(probe, budget(teacher_forced_tokens=4095),
                                                              tokens_per_probe_text=1024)
    assert "needs compute_seconds" in _action_shortfall(probe, budget(compute_seconds=0), tokens_per_probe_text=1)


class WrongSuiteStrategy:
    """Submits a generation pass on a final suite and on a misspelled one, then stops with what it observed."""

    strategy_id, version = "wrong_suite_test", "1"

    def __init__(self, checkpoint_id: str) -> None:
        self.checkpoint_id = checkpoint_id

    def config(self):
        return {"checkpoint_id": self.checkpoint_id}

    def next_action(self, objective, observations, remaining_budget):
        from wse.strategies.base import Action

        suites = ("final", "practise")
        if len(observations) <= len(suites):
            suite = suites[len(observations) - 1]
            return Action.call("wse_submit_evaluation", f"Evaluate on {suite}.", checkpoint_id=self.checkpoint_id,
                               suite_id=suite, eval_config={"sample_count": 4, "seed": 0}, idempotency_key=suite)
        return Action.stop("observed " + ", ".join(str(o.error_code) for o in observations[1:]))


def test_a_strategy_submission_of_an_unresolvable_suite_gets_the_engine_refusal(lm):
    """Under qwen_loop_v1 (no evaluation batches), a final or misspelled suite is refused by the engine as through MCP
    (POLICY_DENIED, NOT_FOUND) and the run goes on; the runner's budget pre-check does not stop it."""
    from wse.policy import QWEN_LOOP_POLICY_ID
    from wse.registry import builtin_registry
    from wse.runner import ReferenceRunner, _generation_submission
    from wse.schemas import TaskDefinition
    from wse.strategies.base import Action

    with Engine(lm.workspace) as engine:
        engine.register_task(TaskDefinition.model_validate({
            "task_id": TASK, "title": "Finish answers", "description": "Make the model finish its answers.",
            "objective": {"task_id": TASK, "description": "Fewer outputs at the token limit.",
                          "gates": [{"metric": "accuracy", "min_value": 0.5}]},
            "development_suite_id": "practice", "development_examples": 4}))
        grant = engine.grant_run(TASK, QWEN_LOOP_POLICY_ID, lm.base)
    with Engine(lm.workspace, run_id=grant.run_id) as bound:
        assert [_generation_submission(bound, Action.call("wse_submit_evaluation", suite_id=suite))
                for suite in ("practice", "final", "practise")] == [True, None, None]
    registry = builtin_registry()
    registry.register("strategy", "wrong_suite_test", "1", WrongSuiteStrategy)
    runner = ReferenceRunner(lm.workspace, registry=registry, worker_args=["--poll-seconds", "0.02", "--device", "cpu"])
    eid = runner.run_experiment(grant.run_id, TASK, QWEN_LOOP_POLICY_ID, {
        "kind": "strategy", "strategy": "wrong_suite_test@1", "parameters": {"checkpoint_id": lm.base}})
    result = runner.last_result["result"]
    assert result["status"] == "completed" and result["stop_condition"] is None, result
    assert result["reason"] == "the strategy stopped: observed POLICY_DENIED, NOT_FOUND"
    with Engine(lm.workspace) as engine:
        assert engine.list_jobs(eid) == [] and engine.get_experiment(eid).budget["generation_passes"].consumed == 0


def test_the_stop_monitor_counts_generation_passes_as_evaluations_left(lm):
    from wse.runner import StopMonitor

    run_id, eid = _open_run(lm.workspace, lm.base)

    def consume(counter: str) -> None:
        with Engine(lm.workspace) as engine, engine.store.db.transaction() as conn:
            conn.execute("UPDATE budget_ledger SET consumed = limit_value WHERE experiment_id = ? AND counter = ?",
                         (eid, counter))

    with Engine(lm.workspace) as engine:
        monitor = StopMonitor(engine, eid, None)
        assert monitor.check() is None
        consume("candidate_commits")  # no commits left, but generation passes remain to evaluate candidates
        assert monitor.check() is None
        consume("generation_passes")
        assert monitor.check().startswith("insufficient_budget: no candidate commits remain")


def test_the_model_cache_releases_an_evicted_model_before_loading_the_next():
    import gc
    import weakref
    from types import SimpleNamespace

    from wse.worker import LoadedModel, ModelCache

    references: dict[str, list] = {}
    loads: list[tuple[str, list[str]]] = []

    def load(checkpoint_id: str) -> LoadedModel:
        gc.collect()
        loads.append((checkpoint_id, sorted(name for name, refs in references.items()
                                            if any(ref() is not None for ref in refs))))
        entry = LoadedModel(checkpoint_id=checkpoint_id, handle=SimpleNamespace(),
                            text_model=SimpleNamespace(runtime=SimpleNamespace(device=torch.device("cpu"))))
        references.setdefault(checkpoint_id, []).append(weakref.ref(entry))
        return entry

    cache = ModelCache(load, capacity=2)
    for checkpoint_id in ("a", "b", "c", "a"):
        cache.get(checkpoint_id)
    # At capacity 2 the least recently used model is evicted, and nothing holds it while the next one loads, so at
    # most two models are ever resident.
    assert loads == [("a", []), ("b", ["a"]), ("c", ["b"]), ("a", ["c"])]
    cache.discard("c")
    cache.clear()
    gc.collect()
    assert not any(ref() is not None for refs in references.values() for ref in refs)
