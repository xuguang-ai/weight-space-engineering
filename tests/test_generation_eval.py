"""Generation evaluation: profiles, presence penalty, graders, and suites (M2 plan Sections 4.2-4.3, Task 6).

* **Profiles.** A profile is a registered, versioned sampling configuration:
  the prompt format (chat template and its arguments, the user-message
  suffix) and the decoding settings. ``qwen35think@1`` reproduces the
  article's ``qwen35think`` profile; every imported model also carries a
  default profile frozen from its ``generation_config.json``.
* **Presence penalty.** vLLM's semantics: the penalty is subtracted once from
  each token present in the generated output (not the prompt), before
  temperature, top-k, and top-p.
* **Seeds.** Each (problem, sample) pair has its own seed, so a row's samples
  do not depend on the other rows of its batch.
* **Graders.** ``math_boxed@1`` grades the last ``\\boxed{}`` answer with
  math-verify; ``gsm8k_numeric@1`` compares the boxed number with the
  reference after ``####``.
* **Suites.** Text suites live in evaluator storage with their hashes; a
  development suite that shares a normalized problem with a final suite is
  rejected.

Models are the tiny offline fixtures of ``tests/hf_fixtures.py``; scripted
outputs come from a stand-in runtime.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from conftest import HF_SKIP_REASON
from wse.api import Engine
from wse.schemas import ErrorCode, EvalConfig, EvaluationSuite, WSEError

pytest.importorskip("transformers", reason=HF_SKIP_REASON)

REPO = "example-org/Llama-3.2-1B-Instruct"
REVISION = "0123456789abcdef0123456789abcdef01234567"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def _expect(code, func, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        func(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


MATH_ROWS = [
    {"problem": "what is 1+1?", "solution": "it is \\boxed{2}", "answer": "2", "subject": "a", "level": 1,
     "unique_id": "test/a/1.json"},
    {"problem": "what is 2+2?", "solution": "it is \\boxed{4}", "answer": "4", "subject": "a", "level": 1,
     "unique_id": "test/a/2.json"},
    {"problem": "half of one?", "solution": "\\boxed{\\frac{1}{2}}", "answer": "\\frac{1}{2}", "subject": "b",
     "level": 2, "unique_id": "test/b/3.json"},
]
GSM_ROWS = [
    {"question": "tom has 3 apples and buys 4. how many?", "answer": "3+4=7\n#### 7"},
    {"question": "a box holds 1,200 pens. how many pens are in two boxes?", "answer": "2*1200\n#### 2,400"},
]


@pytest.fixture(scope="module")
def hf_workspace(hf_snapshots, tmp_path_factory):
    workspace = tmp_path_factory.mktemp("generation") / "workspace"
    with Engine(workspace) as engine:
        manifest = engine.import_hf_snapshot(hf_snapshots["llama_untied"], repo=REPO, revision=REVISION)
    return workspace, manifest


@pytest.fixture(scope="module")
def llama(hf_workspace):
    workspace, manifest = hf_workspace
    with Engine(workspace) as engine:
        return engine.open_checkpoint(manifest.checkpoint_id), engine.model_files(manifest.checkpoint_id)


def _profile(**update):
    from wse.evaluators.sampling import QWEN35THINK

    return QWEN35THINK.model_copy(update=update)


# -- presence penalty and the sampling order ----------------------------------------------------------


def test_presence_penalty_subtracts_once_per_present_output_token_and_ignores_the_prompt():
    from wse.evaluators.sampling import PresencePenalty

    # Rows share one (left-padded) prompt length of 3; the generated output follows it.
    ids = torch.tensor([[5, 6, 7, 7, 7, 8],
                        [0, 9, 9, 3, 3, 4]])
    scores = torch.arange(24, dtype=torch.float32).reshape(2, 12) / 10
    before = scores.clone()
    out = PresencePenalty(1.5, prompt_len=3)(ids, scores)
    expected = before.clone()
    expected[0, [7, 8]] -= 1.5  # output [7, 7, 8]: 7 occurs twice and is penalized once (not per occurrence)
    expected[1, [3, 4]] -= 1.5  # output [3, 3, 4]; 9 and the pad 0 occur only in the prompt
    assert torch.equal(out, expected)
    assert torch.equal(scores, before) and out.dtype == torch.float32
    # Before the first token there is no output, so nothing changes.
    assert torch.equal(PresencePenalty(1.5, prompt_len=6)(ids, scores), before)
    # A zero penalty is the identity, and the penalty must be finite.
    assert torch.equal(PresencePenalty(0.0, prompt_len=3)(ids, scores), before)
    with pytest.raises(ValueError):
        PresencePenalty(float("nan"), prompt_len=3)


def _kept(scores: torch.Tensor) -> set[int]:
    return set(torch.nonzero(torch.isfinite(scores[0])).flatten().tolist())


def test_sampling_order_is_penalty_temperature_top_k_top_p(monkeypatch):
    import wse.engines.hf_torch as hf_torch
    from wse.evaluators.sampling import processed_scores, profile_sampling

    output = torch.tensor([[1, 0]])  # one prompt token (1), then the generated token 0
    # Penalty before top-k: token 0 leads by 0.5 but is present in the output; after a penalty of 2.5 it
    # falls out of the top 2. Top-k first would have kept it.
    kept = _kept(processed_scores(output, torch.tensor([[3.0, 2.5, 1.0, 0.0]]),
                                  _profile(presence_penalty=2.5, temperature=1.0, top_k=2, top_p=1.0), prompt_len=1))
    assert kept == {1, 2}
    # Temperature before top-p: at temperature 0.5 the probabilities [0.867, 0.117, 0.016] keep only token 0 at
    # top_p 0.8; top-p on the untempered [0.665, 0.245, 0.090] would keep tokens 0 and 1.
    kept = _kept(processed_scores(output, torch.tensor([[2.0, 1.0, 0.0]]),
                                  _profile(presence_penalty=0.0, temperature=0.5, top_k=None, top_p=0.8),
                                  prompt_len=2))
    assert kept == {0}
    # Top-k before top-p: within the top 2 the probabilities are [0.622, 0.378], so top_p 0.6 keeps token 0
    # only; top-p over all five tokens would keep tokens 0 and 1.
    kept = _kept(processed_scores(output, torch.tensor([[2.0, 1.5, 1.0, 1.0, 1.0]]),
                                  _profile(presence_penalty=0.0, temperature=1.0, top_k=2, top_p=0.6), prompt_len=2))
    assert kept == {0}
    # qwen35think@1 maps onto the engine's settings.
    sampling = profile_sampling(_profile())
    assert (sampling.temperature, sampling.top_k, sampling.top_p) == (0.6, 20, 0.95)
    assert profile_sampling(_profile(do_sample=False)) is None
    # The engine applies the profile's processors to the raw scores of each step, then warps them.
    calls = []
    real_warp = hf_torch.warp_scores
    monkeypatch.setattr(hf_torch, "warp_scores", lambda s, sampling: calls.append("warp") or real_warp(s, sampling))

    class Recorder:
        def __call__(self, ids, scores):
            calls.append("penalty")
            return scores

    class Module(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            from types import SimpleNamespace

            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 8)
            return SimpleNamespace(logits=logits, past_key_values=None)

    model = hf_torch.HFTorchModel.__new__(hf_torch.HFTorchModel)
    model.module, model.device, model._vocab_size, model._logits_to_keep, model._position_ids = (
        Module(), torch.device("cpu"), 8, False, False)
    model.generate(torch.tensor([[1, 2]]), max_new_tokens=3, sampling=sampling, seeds=[0],
                   logits_processors=[Recorder()])
    assert calls == ["penalty", "warp"] * 3


def test_each_problem_and_sample_has_its_own_reproducible_seed(llama):
    from wse.evaluators.generation import generate_samples, load_text_model
    from wse.evaluators.sampling import sample_seed
    from wse.schemas import TextItem

    seeds = {(item, sample): sample_seed(0, 0, item, sample) for item in ("p0", "p1") for sample in range(3)}
    assert len(set(seeds.values())) == 6 and all(0 <= seed < 2 ** 63 for seed in seeds.values())
    assert sample_seed(0, 0, "p0", 1) == seeds[("p0", 1)]  # reproducible
    assert sample_seed(1, 0, "p0", 1) != seeds[("p0", 1)] and sample_seed(0, 1, "p0", 1) != seeds[("p0", 1)]

    handle, files = llama
    text_model = load_text_model(handle, files, device="cpu")
    profile = _profile(max_new_tokens=10, chat_template_kwargs={})
    items = [TextItem(item_id="p0", problem="add two and two", reference="4"),
             TextItem(item_id="p1", problem="what is six", reference="6"),
             TextItem(item_id="p2", problem="a much longer problem text that pads the others", reference="1")]
    rows = [(index, sample) for index in range(3) for sample in range(2)]
    together = generate_samples(text_model, profile, items, rows, eval_seed=0, batch_size=6)
    alone = generate_samples(text_model, profile, items, rows, eval_seed=0, batch_size=1)
    reordered = generate_samples(text_model, profile, items, list(reversed(rows)), eval_seed=0, batch_size=4)
    by_row = {(o.index, o.sample): o.tokens for o in together}
    assert by_row == {(o.index, o.sample): o.tokens for o in alone}
    assert by_row == {(o.index, o.sample): o.tokens for o in reordered}
    # Different samples of one problem draw differently, and the recorded seed is the derived one.
    assert by_row[(0, 0)] != by_row[(0, 1)]
    assert [o.seed for o in together] == [sample_seed(0, 0, items[i].item_id, k) for i, k in rows]
    # Another evaluation seed changes the draws.
    other = generate_samples(text_model, profile, items, rows, eval_seed=1, batch_size=6)
    assert {(o.index, o.sample): o.tokens for o in other} != by_row


# -- graders ---------------------------------------------------------------------------------------------


def test_math_boxed_grades_the_last_boxed_answer():
    from wse.evaluators.graders import MathBoxedGrader, last_boxed

    grade = MathBoxedGrader().grade
    # math-verify's README example: {1,3} union {2,4} equals {1,2,3,4} (gold first, then the answer).
    assert grade("so the set is $\\boxed{\\{1,3\\} \\cup \\{2,4\\}}$", "\\{1,2,3,4\\}").correct
    # The last box decides; earlier boxes are ignored.
    assert grade("first \\boxed{3}, then corrected: \\boxed{4}", "4").correct
    assert not grade("first \\boxed{3}, then corrected: \\boxed{4}", "3").correct
    assert grade("\\boxed{3} then \\boxed{4}", "4").answer == "4"
    # Equivalent forms: a fraction and a decimal, nested braces, \\fbox, and '\\boxed 5' without braces.
    assert grade("the half is \\boxed{\\frac{1}{2}}", "0.5").correct
    assert grade("\\fbox{12}", "12").correct
    assert grade("so $\\boxed 5$.", "5").correct
    assert grade("\\boxed{(3, \\frac{\\pi}{2})}", "\\left( 3, \\frac{\\pi}{2} \\right)").correct
    # A thinking block is cut off at its last </think> before the answer is read.
    graded = grade("<think>maybe \\boxed{9}</think> the answer is \\boxed{7}", "7")
    assert graded.correct and graded.answer == "7"
    # No box, an empty box, or a box cut off by the token limit is no answer, and wrong.
    for text in ("the answer is 4", "\\boxed{}", "so \\boxed{\\frac{1}{"):
        graded = grade(text, "4")
        assert not graded.correct and graded.answer is None
    assert last_boxed("a \\boxed{x{y}z} b") == "x{y}z"
    assert MathBoxedGrader().grader_id == "math_boxed" and MathBoxedGrader().version == "1"


def test_gsm8k_numeric_reads_the_reference_after_hashes():
    from wse.evaluators.graders import GSM8KNumericGrader, gsm8k_reference

    reference = "She has 1,000 + 234 = <<1000+234=1234>>1234 marbles.\n#### 1,234"
    assert gsm8k_reference(reference) == "1234"
    grade = GSM8KNumericGrader().grade
    for text in ("\\boxed{1234}", "\\boxed{1,234}", "\\boxed{\\$1234.00}", "so \\boxed{3} no wait \\boxed{1234}",
                 "\\boxed{1234 \\text{ marbles}}"):
        assert grade(text, reference).correct, text
    for text in ("\\boxed{1235}", "the answer is 1234", "\\boxed{12 and 34}", "\\boxed{1234} then \\boxed{5}"):
        assert not grade(text, reference).correct, text
    assert grade("\\boxed{-3.5}", "x\n#### -3.5").correct
    assert grade("\\boxed{7}", "#### 7").answer == "7"
    with pytest.raises(ValueError):
        gsm8k_reference("no hashes here")


# -- per-problem results, suites, and report identities -------------------------------------------------


class ScriptedRuntime:
    """Stands in for the engine: each prompt has a scripted output; a row without a stop token hits the limit."""

    def __init__(self, tokenizer, script: dict[str, str], stop: int) -> None:
        self.tokenizer, self.script, self.stop = tokenizer, script, stop

    def info(self):
        return {"device": "cpu", "dtypes": ["bfloat16"], "model_class": "Scripted"}

    def generate(self, input_ids, attention_mask=None, *, max_new_tokens, stop_token_ids=(), pad_token_id=0,
                 sampling=None, seeds=None, logits_processors=()):
        from wse.engines.hf_torch import GenerationResult

        tokens, finished = [], []
        for row in range(input_ids.shape[0]):
            prompt = self.tokenizer.decode(input_ids[row][attention_mask[row].bool()].tolist())
            key = next(k for k in self.script if k in prompt)
            text = self.script[key]
            ids = self.tokenizer.encode(text.removesuffix("<STOP>"), add_special_tokens=False)
            ids = ids + [self.stop] if text.endswith("<STOP>") else ids
            tokens.append(ids[:max_new_tokens])
            finished.append(text.endswith("<STOP>") and len(ids) <= max_new_tokens)
        return GenerationResult(tokens=tokens, finished=finished)


def _text_suite(items, *, profile="qwen35think@1", grader="math_boxed@1", samples=1, split="development"):
    from wse.schemas import TextSuiteData

    return EvaluationSuite(suite_id="practice", task_id="task", split=split,
                           evaluator={"extension_id": "generation", "version": "1"},
                           parameters={"profile": profile, "grader": grader, "samples_per_problem": samples},
                           sample_counts=[len(items)], seeds=[0], data=TextSuiteData(items=items))


def test_limit_hits_and_lengths_are_reported_per_problem(llama):
    from wse.evaluators.generation import TextModel, evaluate_generation, load_text_model
    from wse.schemas import TextItem

    handle, files = llama
    real = load_text_model(handle, files, device="cpu")
    script = {"one": "so \\boxed{4}<STOP>", "two": "loop loop loop loop loop loop", "three": "no box here<STOP>"}
    runtime = ScriptedRuntime(real.tokenizer, script, real.stop_token_ids[0])
    text_model = TextModel(runtime=runtime, tokenizer=real.tokenizer, stop_token_ids=real.stop_token_ids,
                           pad_token_id=real.pad_token_id)
    items = [TextItem(item_id="a", problem="problem one", reference="4"),
             TextItem(item_id="b", problem="problem two", reference="4"),
             TextItem(item_id="c", problem="problem three", reference="4")]
    suite = _text_suite(items, samples=2)
    result = evaluate_generation(handle, suite, EvalConfig(sample_count=3, seed=0), text_model=text_model,
                                 profile_overrides={"max_new_tokens": 20})
    problems = {p.item_id: p for p in result.problems}
    a, b, c = problems["a"].samples, problems["b"].samples, problems["c"].samples
    assert [s.correct for s in a] == [True, True] and [s.answer for s in a] == ["4", "4"]
    assert all(s.finished and not s.limit_hit for s in a + c)
    assert all(s.limit_hit and not s.finished and s.generated_tokens == 20 and not s.correct for s in b)
    assert all(s.answer is None and not s.correct for s in c)
    length_a = len(real.tokenizer.encode("so \\boxed{4}", add_special_tokens=False)) + 1  # the stop token counts
    assert [s.generated_tokens for s in a] == [length_a, length_a]
    assert result.metrics["accuracy"] == pytest.approx(2 / 6)
    assert result.metrics["limit_hit_rate"] == pytest.approx(2 / 6)
    lengths = [s.generated_tokens for p in result.problems for s in p.samples]
    assert result.metrics["mean_generated_tokens"] == pytest.approx(sum(lengths) / 6)
    assert result.truncation_count == 2 and result.sample_count == 3
    # Scores are per sample, clustered by problem, with bootstrap and exact intervals per proportion.
    assert list(result.scores) == [1, 1, 0, 0, 0, 0] and list(result.cluster_ids) == [0, 0, 1, 1, 2, 2]
    assert result.clusters.unit == "problem" and result.clusters.count == 3 and result.clusters.fully_correct == 1
    for name in ("accuracy", "limit_hit_rate", "mean_generated_tokens"):
        assert result.intervals[name].method == "cluster_bootstrap" and result.intervals[name].cluster_unit == "problem"
    for name in ("accuracy_exact", "limit_hit_rate_exact"):
        assert result.intervals[name].method == "clopper_pearson" and result.intervals[name].trials == 6
    assert result.intervals["limit_hit_rate_exact"].successes == 2


def _register_text(engine, tmp_path, *, final_rows=MATH_ROWS, dev_rows=GSM_ROWS, dev_id="gsm8k_practice"):
    final_file = _jsonl(tmp_path / "math500.jsonl", final_rows)
    dev_file = _jsonl(tmp_path / f"{dev_id}.jsonl", dev_rows)
    final = engine.register_text_dataset(final_file, format="math500")
    dev = engine.register_text_dataset(dev_file, format="gsm8k", id_prefix="gsm8k_train_")
    final_suite = engine.register_suite({
        "suite_id": "math500", "task_id": "loops", "split": "final",
        "evaluator": {"extension_id": "generation", "version": "1"},
        "parameters": {"profile": "qwen35think@1", "grader": "math_boxed@1", "samples_per_problem": 1},
        "sample_counts": [final.example_count], "seeds": [0], "dataset": final.model_dump(mode="json")})
    return final_file, dev_file, final, dev, final_suite


def _dev_suite(dataset, suite_id="gsm8k_practice", task_id="loops"):
    return {"suite_id": suite_id, "task_id": task_id, "split": "development",
            "evaluator": {"extension_id": "generation", "version": "1"},
            "parameters": {"profile": "qwen35think@1", "grader": "gsm8k_numeric@1", "samples_per_problem": 1},
            "sample_counts": sorted({1, dataset.example_count}), "seeds": [0], "dataset": dataset.model_dump(mode="json")}


def test_suites_record_hashes_and_reject_overlap_with_the_final_suite(tmp_path):
    workspace = tmp_path / "workspace"
    with Engine(workspace) as engine:
        final_file, dev_file, final, dev, final_suite = _register_text(engine, tmp_path)
        # Hashes: the source file, the items, and the registered suite; items stay in evaluator storage.
        assert final.source_sha256 == "sha256:" + hashlib.sha256(final_file.read_bytes()).hexdigest()
        assert dev.source_sha256 == "sha256:" + hashlib.sha256(dev_file.read_bytes()).hexdigest()
        assert final.kind == "text" and final.example_count == 3 and final.cluster_unit == "problem"
        assert final_suite.data is None and final_suite.dataset == final and final_suite.data_hash == final.data_hash
        stored = engine.datasets.load(final)
        assert [item.item_id for item in stored.items] == [row["unique_id"] for row in MATH_ROWS]
        assert [item.reference for item in stored.items] == [row["answer"] for row in MATH_ROWS]
        gsm = engine.datasets.load(dev)
        assert [item.item_id for item in gsm.items] == ["gsm8k_train_0", "gsm8k_train_1"]
        assert gsm.items[1].reference == GSM_ROWS[1]["answer"]
        registered = engine.register_suite(_dev_suite(dev))
        assert registered.suite_hash == engine.get_suite("gsm8k_practice").suite_hash
        # Registering again is idempotent; a dataset stored twice is one file.
        assert engine.register_text_dataset(dev_file, format="gsm8k", id_prefix="gsm8k_train_") == dev
        # A practice item whose normalized problem (case, Unicode form, whitespace) is a final problem is refused.
        overlapping = GSM_ROWS + [{"question": "What  is 1+1?\n", "answer": "#### 2"}]
        leaked = engine.register_text_dataset(_jsonl(tmp_path / "leak.jsonl", overlapping), format="gsm8k",
                                              id_prefix="leak_")
        error = _expect(ErrorCode.INVALID_ARGUMENT, engine.register_suite, _dev_suite(leaked, "leaky"))
        assert error.details["overlapping_items"] == ["leak_2"] and "math500" in error.message
        assert "leaky" not in [s.suite_id for s in engine.jobs.list_suites("loops")]
        # The same holds across tasks, and in the other direction: a final suite that overlaps a registered
        # practice suite is refused.
        _expect(ErrorCode.INVALID_ARGUMENT, engine.register_suite, _dev_suite(leaked, "leaky2", task_id="other"))
        # Sample counts must not exceed the dataset.
        bad = _dev_suite(dev, "too_many")
        bad["sample_counts"] = [3]
        _expect(ErrorCode.INVALID_ARGUMENT, engine.register_suite, bad)
        # Duplicate item IDs and empty problems are refused at reading.
        _expect(ErrorCode.INVALID_ARGUMENT, engine.register_text_dataset,
                _jsonl(tmp_path / "dup.jsonl", [MATH_ROWS[0], MATH_ROWS[0]]), format="math500")
        _expect(ErrorCode.INVALID_ARGUMENT, engine.register_text_dataset,
                _jsonl(tmp_path / "empty.jsonl", [{"question": " ", "answer": "#### 1"}]), format="gsm8k")
    # A practice suite registered first blocks an overlapping final suite registered later.
    other = tmp_path / "other"
    with Engine(other) as engine:
        practice = engine.register_text_dataset(_jsonl(tmp_path / "p.jsonl", [MATH_ROWS[0]]), format="math500")
        engine.register_suite({**_dev_suite(practice, "practice"),
                               "parameters": {"profile": "qwen35think@1", "grader": "math_boxed@1",
                                              "samples_per_problem": 1}})
        final = engine.register_text_dataset(_jsonl(tmp_path / "f.jsonl", MATH_ROWS), format="math500")
        error = _expect(ErrorCode.INVALID_ARGUMENT, engine.register_suite, {
            "suite_id": "math500", "task_id": "loops", "split": "final",
            "evaluator": {"extension_id": "generation", "version": "1"},
            "parameters": {"profile": "qwen35think@1", "grader": "math_boxed@1", "samples_per_problem": 1},
            "sample_counts": [3], "seeds": [0], "dataset": final.model_dump(mode="json")})
        assert error.details["overlapping_items"] == ["test/a/1.json"]


def test_the_final_text_suite_is_unreachable_from_a_run_bound_engine(hf_workspace, tmp_path):
    from wse.policy import SMALL_MODEL_POLICY_ID

    workspace, manifest = hf_workspace
    with Engine(workspace) as engine:
        _, _, final, dev, _ = _register_text(engine, tmp_path)
        engine.register_suite(_dev_suite(dev))
        grant = engine.grant_run("loops", SMALL_MODEL_POLICY_ID, manifest.checkpoint_id)
    with Engine(workspace, run_id=grant.run_id) as bound:
        _expect(ErrorCode.NOT_FOUND, bound.get_suite, "math500")
        assert bound.get_suite("gsm8k_practice").data is None  # practice items stay in evaluator storage too
        _expect(ErrorCode.POLICY_DENIED, bound.datasets.load, final)
        _expect(ErrorCode.POLICY_DENIED, bound.register_text_dataset, tmp_path / "math500.jsonl", format="math500")
        _expect(ErrorCode.POLICY_DENIED, bound.evaluate_generation, manifest.checkpoint_id, "math500",
                {"sample_count": 3, "seed": 0})
        _expect(ErrorCode.POLICY_DENIED, bound.model_files, manifest.checkpoint_id)


def test_generation_reports_bind_model_profile_suite_and_grader_identities(hf_workspace, tmp_path):
    from wse.evaluators.generation import compare_generation
    from wse.evaluators.sampling import QWEN35THINK

    workspace, manifest = hf_workspace
    rows = [{"question": "tom has three apples", "answer": "#### 3"},
            {"question": "two and two", "answer": "#### 4"}]
    with Engine(workspace) as engine:
        dataset = engine.register_text_dataset(_jsonl(tmp_path / "ids.jsonl", rows), format="gsm8k", id_prefix="r_")
        suites = {}
        for suite_id, profile, grader in (("ids_think", "qwen35think@1", "gsm8k_numeric@1"),
                                          ("ids_default", "llama_3_2_1b_instruct_default@1", "gsm8k_numeric@1"),
                                          ("ids_math", "qwen35think@1", "math_boxed@1")):
            suites[suite_id] = engine.register_suite({
                **_dev_suite(dataset, suite_id, task_id="identity"),
                "parameters": {"profile": profile, "grader": grader, "samples_per_problem": 2,
                               "max_new_tokens": 6}})
        config = {"sample_count": 2, "seed": 0}
        results = {suite_id: engine.evaluate_generation(manifest.checkpoint_id, suite_id, config, device="cpu")
                   for suite_id in suites}
        think = results["ids_think"]
        assert think.model == engine.open_checkpoint(manifest.checkpoint_id).identity
        assert think.model.tokenizer_hash and think.model.chat_template_hash and think.model.generation_config_hash
        assert think.evaluator.extension_id == "generation" and think.suite_id == "ids_think"
        assert think.data_hash == dataset.data_hash
        parameters = think.execution.parameters
        assert parameters["profile"] == "qwen35think@1" and parameters["grader"] == "gsm8k_numeric@1"
        assert parameters["profile_hash"] == QWEN35THINK.model_copy(update={"max_new_tokens": 6}).profile_hash
        assert think.execution.device == "cpu" and think.runtime["device"] == "cpu"
        assert think.runtime["dtypes"] == ["bfloat16"] and "torch" in think.runtime and "transformers" in think.runtime
        default = results["ids_default"].execution.parameters
        assert default["profile_hash"] == manifest.source.default_profile.model_copy(
            update={"max_new_tokens": 6}).profile_hash
        hashes = {result.evaluator_config_hash for result in results.values()}
        assert len(hashes) == 3  # the profile and the grader are part of the configuration identity
        # The same evaluation again is identical (seeded per problem and sample).
        again = engine.evaluate_generation(manifest.checkpoint_id, "ids_think", config, device="cpu")
        assert again.evaluator_config_hash == think.evaluator_config_hash
        assert [s.output_sha256 for p in again.problems for s in p.samples] == \
            [s.output_sha256 for p in think.problems for s in p.samples]
        # A managed report built from the result carries the identities, the per-problem results, and the runtime.
        report = engine.evaluation.build_report(think, experiment_id="exp_x", job_id=None,
                                                checkpoint_id=manifest.checkpoint_id, suite=suites["ids_think"],
                                                usage=think.usage, keep_scores=True)
        assert report.model == think.model and report.suite_hash == suites["ids_think"].suite_hash
        assert report.execution == think.execution and report.evaluator_config_hash == think.evaluator_config_hash
        assert [p.item_id for p in report.problems] == ["r_0", "r_1"] and report.runtime == dict(think.runtime)
        final_style = engine.evaluation.build_report(think, experiment_id="exp_x", job_id=None,
                                                     checkpoint_id=manifest.checkpoint_id, suite=suites["ids_think"],
                                                     usage=think.usage, keep_scores=False)
        assert final_style.problems is None and final_style.scores is None
        # Paired comparisons accept only reports of one suite, data, and configuration.
        paired = compare_generation(think, again)
        assert paired.differences["accuracy"].estimate == 0 and paired.differences["limit_hit_rate"].estimate == 0
        assert set(paired.differences) == {"accuracy", "limit_hit_rate", "mean_generated_tokens"}
        error = _expect(ErrorCode.INCOMPATIBLE_REPORTS, compare_generation, think, results["ids_math"])
        assert "evaluator_config_hash" in error.details["mismatched"]


def test_default_profiles_are_frozen_from_generation_config(hf_workspace, llama):
    from wse.evaluators.sampling import default_profile_id, profile_from_generation_config, resolve_profile
    from wse.registry import builtin_registry

    workspace, manifest = hf_workspace
    profile = manifest.source.default_profile
    # The fixture's generation_config.json: do_sample, temperature 0.7, eos 2, pad 0; the rest are the
    # defaults transformers' generate applies (top_k 50, top_p 1.0).
    assert str(profile.ref) == "llama_3_2_1b_instruct_default@1"
    assert (profile.do_sample, profile.temperature, profile.top_k, profile.top_p) == (True, 0.7, 50, 1.0)
    assert profile.presence_penalty == 0.0 and profile.stop_token_ids == [2] and profile.seed == 0
    assert profile.chat_template and profile.user_suffix == "\n" + INSTRUCTION
    assert profile.source == f"generation_config.json {manifest.source.generation_config_hash}"
    # Frozen: a candidate of the model carries the same profile, and resolution finds it through the model.
    handle, _ = llama
    registry = builtin_registry()
    assert resolve_profile(registry, "llama_3_2_1b_instruct_default@1", handle.source) == profile
    assert str(resolve_profile(registry, "qwen35think@1", handle.source).ref) == "qwen35think@1"
    _expect(ErrorCode.NOT_FOUND, resolve_profile, registry, "other_model_default@1", handle.source)
    _expect(ErrorCode.NOT_FOUND, resolve_profile, registry, "llama_3_2_1b_instruct_default@2", handle.source)
    # Naming: lowercased, other characters become '_', 'm_' before a leading digit.
    assert default_profile_id("Qwen/Qwen3.5-0.8B") == "qwen3_5_0_8b_default"
    assert default_profile_id("meta-llama/Llama-3.2-1B-Instruct") == "llama_3_2_1b_instruct_default"
    assert default_profile_id("org/3B-Chat") == "m_3b_chat_default"
    assert len(default_profile_id("org/" + "x" * 100)) <= 64
    # Without a generation configuration the defaults are transformers' (greedy); unsupported settings are
    # listed, not applied.
    greedy = profile_from_generation_config("Qwen/Qwen3.5-0.8B", None, has_chat_template=True)
    assert greedy.do_sample is False and greedy.stop_token_ids is None and greedy.source == "transformers defaults"
    custom = profile_from_generation_config("org/m", {"do_sample": True, "top_k": 0, "max_new_tokens": 64,
                                                      "repetition_penalty": 1.1, "eos_token_id": 7},
                                            has_chat_template=False, source="x")
    assert custom.top_k is None and custom.max_new_tokens == 64 and custom.stop_token_ids == [7]
    assert custom.ignored_settings == {"repetition_penalty": 1.1} and custom.chat_template is False


def test_prompts_render_with_the_chat_template_and_the_profile_arguments(llama):
    from wse.evaluators.generation import load_text_model
    from wse.evaluators.sampling import render_prompt

    handle, files = llama
    tokenizer = load_text_model(handle, files, device="cpu").tokenizer
    thinking = render_prompt(tokenizer, _profile(), "what is 2+2?")
    assert thinking == "<user>what is 2+2?\n" + INSTRUCTION + "<assistant><think>"
    plain = render_prompt(tokenizer, _profile(chat_template_kwargs={"enable_thinking": False}), "x")
    assert plain.endswith("<assistant>") and "<think>" not in plain
    raw = render_prompt(tokenizer, _profile(chat_template=False), "x")
    assert raw == "x\n" + INSTRUCTION


def test_profiles_and_graders_are_registered_and_versioned():
    from wse.evaluators.sampling import QWEN35THINK
    from wse.registry import builtin_registry
    from wse.schemas import GenerationProfile

    registry = builtin_registry()
    assert registry.resolve_ref("profile", "qwen35think@1") == QWEN35THINK
    assert {str(ref) for ref in registry.list("grader")} == {"math_boxed@1", "gsm8k_numeric@1"}
    assert "generation@1" in {str(ref) for ref in registry.list("evaluator")}
    profile = registry.resolve_ref("profile", "qwen35think@1")
    assert (profile.temperature, profile.top_p, profile.top_k, profile.presence_penalty) == (0.6, 0.95, 20, 1.5)
    assert (profile.max_new_tokens, profile.seed, profile.do_sample) == (32768, 0, True)
    assert profile.chat_template and profile.chat_template_kwargs == {"enable_thinking": True}
    assert profile.user_suffix == "\n" + INSTRUCTION
    # A registered profile is immutable; another version must be registered under a new version.
    _expect(ErrorCode.VERSION_CONFLICT, registry.register, "profile", "qwen35think", "1", QWEN35THINK)
    _expect(ErrorCode.INVALID_ARGUMENT, registry.register, "profile", "other", "1", QWEN35THINK)
    with pytest.raises(ValueError):
        GenerationProfile(profile_id="x", version="1", do_sample=True, temperature=0.0, max_new_tokens=1)


def test_model_files_are_stored_at_import_and_verified_on_read(hf_snapshots, tmp_path):
    workspace = tmp_path / "workspace"
    snapshot = hf_snapshots["llama_untied"]
    with Engine(workspace) as engine:
        manifest = engine.import_hf_snapshot(snapshot, repo=REPO, revision=REVISION)
        files = engine.model_files(manifest.checkpoint_id)
        assert set(files) == {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                              "chat_template.jinja", "generation_config.json"}
        assert all(files[name] == (snapshot / name).read_bytes() for name in files)
        stored = sorted((workspace / "artifacts" / "model_files").iterdir())
        assert len(stored) == len(files)
        # A changed stored file is refused, not used.
        target = workspace / "artifacts" / "model_files" / manifest.source.files["tokenizer.json"].removeprefix(
            "sha256:")
        target.chmod(0o644)
        target.write_bytes(b"{}")
        error = _expect(ErrorCode.NOT_FOUND, engine.model_files, manifest.checkpoint_id)
        assert error.details["file"] == "tokenizer.json"
        # Importing the snapshot again restores it and creates no checkpoint.
        again = engine.import_hf_snapshot(snapshot, repo=REPO, revision=REVISION)
        assert again.checkpoint_id == manifest.checkpoint_id
        assert engine.model_files(manifest.checkpoint_id) == files
