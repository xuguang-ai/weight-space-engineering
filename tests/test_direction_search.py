"""The scripted discovery strategy ``direction_search@1`` (M2 plan Task 12).

* **Decisions.** Synthetic observations drive the strategy through its four
  steps, so each decision is checked on its own: grouping the outputs of the
  diagnosis pass, the probe's sites and groups, the layer of largest
  separation, the direction's reference, the columns, one candidate per
  alpha, the selection, and every stop (an empty group, a missing role, the
  budget, a failed operation).
* **End to end.** The strategy runs through ``ReferenceRunner`` on the tiny
  Llama fixture under ``qwen_loop_v1``. The worker's generation is scripted
  (``scripted_generation``, a worker setup hook) so that the base and each
  candidate have known outcomes; probes, analyses, and edits run on the real
  fixture weights.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from test_llm_policy_mcp import TASK, lm  # noqa: F401 - lm is a fixture
from wse.api import Engine
from wse.schemas import BudgetEntry
from wse.strategies.base import Action, Observation
from wse.strategies.direction_search import (
    DirectionSearchStrategy,
    choose_layer,
    decoder_layer_count,
    improves,
    output_groups,
    rank_key,
    role_tensor_of_site,
    separation,
    top_columns,
)

TESTS_DIR = Path(__file__).resolve().parent
EID = "exp_" + "e" * 32
BASE = "ckpt_" + "0" * 32
URI = "wse://analyses/" + "a" * 64
FULL_BUDGET = {"candidate_commits": 20, "patch_previews": 50, "analysis_calls": 50, "tensor_read_calls": 100,
               "evaluation_batches": 0, "evaluated_examples": 0, "model_forward_calls": 0, "compute_seconds": 21600.0,
               "generation_passes": 10, "generated_tokens": 10 * 32 * 8192, "probe_calls": 25,
               "teacher_forced_tokens": 25 * 64 * 1024}
# What each tool consumes in the synthetic world (a pass reserves 4 problems x 12 tokens and settles them all).
COSTS = {"wse_submit_evaluation": {"generation_passes": 1, "generated_tokens": 48},
         "wse_probe": {"probe_calls": 1, "teacher_forced_tokens": 1024},
         "wse_analyze": {"analysis_calls": 1}, "wse_preview_patch": {"patch_previews": 1},
         "wse_apply_patch": {"candidate_commits": 1}}


def _down(layer: int) -> str:
    return f"model.layers.{layer}.mlp.down_proj"


def _sample(uri: str | None, *, limit_hit: bool = False, finished: bool = False, correct: bool = False) -> dict:
    return {"sample": 0, "seed": 0, "correct": correct, "generated_tokens": 12 if limit_hit else 5,
            "finished": finished, "limit_hit": limit_hit, "output_sha256": "sha256:" + "0" * 64, "tail": "...",
            "output_uri": uri}


def _report(outcomes: str) -> dict:
    """A generation report with one sample per problem: ``L`` reached the limit, ``F`` finished, ``O`` neither."""
    problems = []
    for index, outcome in enumerate(outcomes):
        uri = f"wse://analyses/{index:064x}"
        sample = _sample(uri, limit_hit=outcome == "L", finished=outcome == "F", correct=outcome == "F")
        problems.append({"index": index, "item_id": f"p{index}", "prompt_sha256": "sha256:" + "1" * 64,
                         "samples": [sample]})
    return {"report_id": "report_base", "metrics": {}, "problems": problems}


class World:
    """Answers a strategy's actions with synthetic observations, and meters the budget as the engine would."""

    def __init__(self, *, layers: int = 3, outcomes: str = "LLFLFF", norms: dict | None = None,
                 metrics: dict | None = None, budget: dict | None = None, family: str | None = "llama@1",
                 operations: dict | None = None, refuse_layers: list[int] | None = None,
                 max_sites: int | None = None, analysis_top: list[tuple[int, float]] | None = None,
                 input_top: list | None = None, fail: dict | None = None, changed: dict | None = None,
                 config: dict | None = None, pages: int = 1, overview: bool = False) -> None:
        self.layers = layers
        self.report = _report(outcomes)
        # Per (layer, point): (|mean_a|, |mean_b|, |mean_a - mean_b|). By default layer 1 separates most.
        self.norms = norms or {(layer, point): (10.0 * (layer + 1), 10.0 * (layer + 1), [1.0, 4.0, 2.0][layer % 3])
                               for layer in range(layers) for point in ("input", "output")}
        # Metrics by checkpoint: the base, then candidates in commit order.
        self.metrics = metrics or {"base": (0.5, 0.5, 9.0), 1: (0.5, 0.25, 8.0), 2: (0.25, 0.0, 6.0)}
        self.budget = {name: float(value) for name, value in {**FULL_BUDGET, **(budget or {})}.items()}
        self.family, self.refuse_layers, self.max_sites = family, refuse_layers, max_sites
        self.operations = operations or {"analyses": {"direction_projection": {}}, "edits": {"project_out": {}},
                                         "limits": {}}
        self.analysis_top = analysis_top or [(30, -0.9), (2, 0.8), (17, 0.7), (5, -0.6), (9, 0.1)]
        self.input_top = input_top or [[11, "inf"], [3, -5.0], [29, 4.0], [7, 1.0]]
        self.fail = fail or {}
        self.changed = changed or {}
        self.config = config or {"num_hidden_layers": layers}
        self.pages, self.overview = pages, overview
        self.inspections: list[dict] = []
        self.jobs: dict[str, dict] = {}
        self.commits: list[str] = []
        self.selected = BASE
        self.probes: list[dict] = []
        self.counter = 0

    # -- state --------------------------------------------------------------------------

    def remaining(self) -> dict[str, BudgetEntry]:
        return {name: BudgetEntry(limit=FULL_BUDGET.get(name, value), consumed=0, remaining=value)
                for name, value in self.budget.items()}

    def description(self) -> dict:
        return {"experiment": {"experiment_id": EID, "base_checkpoint_id": BASE, "selected_checkpoint_id": BASE,
                               "mode": "bounded_eval"},
                "task": {"task_id": TASK, "development_suite_id": "practice", "development_examples": 6},
                "operations": self.operations, "evaluation_available": True,
                "suites": [{"suite_id": "practice", "task_id": TASK, "split": "development",
                            "sample_counts": [3, 6], "seeds": [0]}]}

    def manifest(self) -> dict:
        tensors = []
        for layer in range(self.layers):
            tensors += [{"name": f"{_down(layer)}.weight", "shape": [16, 32], "axes": ["out_features", "in_features"]},
                        {"name": f"model.layers.{layer}.mlp.up_proj.weight", "shape": [32, 16],
                         "axes": ["out_features", "in_features"]}]
        return {"config": {"model_config": self.config, "family": self.family}, "tensors": tensors}

    def _id(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter:032x}"

    def _job(self, kind: str, **fields: Any) -> dict:
        job = {"job_id": self._id("job"), "kind": kind, "state": "queued", "polls": 0, **fields}
        self.jobs[job["job_id"]] = job
        return {key: value for key, value in job.items() if key not in ("polls", "final")}

    # -- execution ------------------------------------------------------------------------

    def error(self, action: Action, code: str, message: str = "refused", **details: Any) -> Observation:
        return Observation(tool=action.tool, arguments=action.arguments, ok=False,
                           error={"code": code, "message": message, "retryable": False, "details": details})

    def execute(self, action: Action) -> Observation:
        tool, args = action.tool, action.arguments
        if tool in self.fail:
            return self.error(action, self.fail.pop(tool))
        for counter, amount in COSTS.get(tool, {}).items():
            if self.budget[counter] < amount:
                return self.error(action, "BUDGET_EXCEEDED", f"{counter} is exhausted")
        result = getattr(self, tool)(action)
        if isinstance(result, Observation):
            return result
        for counter, amount in COSTS.get(tool, {}).items():
            self.budget[counter] -= amount
        return Observation(tool=tool, arguments=args, ok=True, result=result)

    def wse_inspect_model(self, action: Action) -> dict:
        """One page of the inspection: the tensors split into ``pages`` pages, the rest of the manifest whole."""
        self.inspections.append(action.arguments)
        manifest = self.manifest()
        if self.pages == 1:
            return {"manifest": manifest}
        number = int(action.arguments.get("cursor", "page:0").partition(":")[2])
        size = -(-len(manifest["tensors"]) // self.pages)
        tensors = manifest["tensors"][number * size:(number + 1) * size]
        page = {"start": number * size, "stop": number * size + len(tensors), "page": number,
                "page_count": self.pages, "next_cursor": f"page:{number + 1}" if number + 1 < self.pages else None}
        return {"manifest": {**manifest, "tensors": tensors}, "tensor_page": page}

    def wse_submit_evaluation(self, action: Action) -> dict:
        checkpoint = action.arguments["checkpoint_id"]
        key = "base" if checkpoint == BASE else self.commits.index(checkpoint) + 1
        accuracy, limit_hit_rate, tokens = self.metrics[key]
        result = {"report_id": f"report_{key}", "metrics": {"accuracy": accuracy, "limit_hit_rate": limit_hit_rate,
                                                             "mean_generated_tokens": tokens}}
        return self._job("evaluation", checkpoint_id=checkpoint, report_id=f"report_{key}", final=result,
                         reservation={"requested": {"generation_passes": 1, "generated_tokens": 48}})

    def wse_probe(self, action: Action) -> dict | Observation:
        sites = action.arguments["sites"]
        self.probes.append(action.arguments)
        layers = sorted({site["layer"] for site in sites})
        missing = [layer for layer in layers if self.refuse_layers is not None and layer not in self.refuse_layers]
        if missing:
            return self.error(action, "NOT_FOUND", f"layer {missing[0]} has no tensor with role mlp.down",
                              family="llama@1", role="mlp.down", layer=missing[0], layers=self.refuse_layers)
        if self.max_sites is not None and len(sites) > self.max_sites:
            return self.error(action, "INVALID_ARGUMENT", "the inline summary could exceed the limit",
                              limit_bytes=28672, texts=len(action.arguments["texts"]), sites=len(sites),
                              max_sites=self.max_sites)
        uri = "wse://analyses/" + f"{len(self.probes):064x}"
        summary_sites = []
        for site in sites:
            a, b, contrast = self.norms[(site["layer"], site["point"])]
            entry = {"site": f"{_down(site['layer'])}:{site['point']}", "features": 16 if site["point"] == "output"
                     else 32, "group_mean_norms": [a, b], "contrast_norm": contrast, "top": []}
            if site["point"] == "input":
                entry["top"] = self.input_top[: action.arguments.get("top", 8)]
            summary_sites.append(entry)
        result = {"result_uri": uri, "groups": {name: len(v) for name, v in action.arguments["groups"].items()},
                  "sites": summary_sites, "top": action.arguments.get("top", 8)}
        return self._job("probe", final=result,
                         reservation={"requested": {"probe_calls": 1, "teacher_forced_tokens": 1024}})

    def wse_get_job(self, action: Action) -> dict:
        job = self.jobs[action.arguments["job_id"]]
        job["polls"] += 1
        job["state"] = "running" if job["polls"] == 1 else "succeeded"
        shown = {key: value for key, value in job.items() if key not in ("polls", "final")}
        if job["state"] == "succeeded":
            shown["result"] = job["final"]
        return shown

    def wse_get_report(self, action: Action) -> dict:
        if self.overview:  # a report too large for one result: everything but its per-output results
            return {**self.report, "problems": None, "omitted": ["problems", "scores", "cluster_ids"],
                    "report_bytes": 40960, "limit_bytes": 31744, "resource_uri": "wse://reports/report_base"}
        return self.report

    def wse_analyze(self, action: Action) -> dict:
        entries = [{"index": index, "projection": value * 2.0, "cosine": value} for index, value in self.analysis_top]
        return {"kind": action.arguments["kind"], "result": {"top": entries[: action.arguments["parameters"]["top"]]},
                "result_uri": "wse://analyses/" + "c" * 64}

    def wse_preview_patch(self, action: Action) -> dict:
        alpha = action.arguments["patch_spec"]["operations"][0]["alpha"]
        return {"patch_id": self._id("patch"), "changed_elements": self.changed.get(alpha, 512)}

    def wse_apply_patch(self, action: Action) -> dict:
        checkpoint = self._id("ckpt")
        self.commits.append(checkpoint)
        return {"checkpoint_id": checkpoint}

    def wse_compare(self, action: Action) -> dict:
        return {"metric": "accuracy", "difference": 0.0}

    def wse_select_checkpoint(self, action: Action) -> dict | Observation:
        if action.arguments["expected_selected_id"] != self.selected:
            return self.error(action, "VERSION_CONFLICT")
        self.selected = action.arguments["checkpoint_id"]
        return {"selected_checkpoint_id": self.selected}


def drive(strategy: DirectionSearchStrategy, world: World, limit: int = 300) -> list[Action]:
    """Run the strategy against ``world`` as the runner would, until it stops; return every action, the stop last."""
    observations = [Observation(tool="wse_get_experiment", arguments={"experiment_id": EID}, ok=True,
                                result=world.description())]
    actions: list[Action] = []
    for _ in range(limit):
        action = strategy.next_action(None, list(observations), world.remaining())
        actions.append(action)
        if action.kind == "stop":
            return actions
        observations.append(world.execute(action))
    raise AssertionError("the strategy did not stop")


def _calls(actions: list[Action], tool: str) -> list[Action]:
    return [action for action in actions if action.tool == tool]


def _tools(actions: list[Action]) -> list[str]:
    """The tools in order, with consecutive job polls collapsed into one."""
    tools: list[str] = []
    for action in actions:
        name = action.tool or "stop"
        if not (name == "wse_get_job" and tools and tools[-1] == "wse_get_job"):
            tools.append(name)
    return tools


# ---------------------------------------------------------------------------
# Registration and parameters
# ---------------------------------------------------------------------------


def test_the_strategy_is_registered_and_checks_its_parameters():
    from wse.registry import builtin_registry

    assert builtin_registry().resolve("strategy", "direction_search", "1") is DirectionSearchStrategy
    assert DirectionSearchStrategy().config() == {
        "suite_id": None, "sample_count": None, "seed": None, "columns": 64, "alphas": [1.0, 6.0], "layers": None,
        "last": None, "column_rule": "cosine", "max_texts": 64, "wait_seconds": 10.0, "compare": True}
    configured = DirectionSearchStrategy(suite_id="practice", sample_count=3, seed=0, columns=8, alphas=[2],
                                         layers=[3, 1], last=256, column_rule="input_contrast", max_texts=16,
                                         wait_seconds=0.5, compare=False).config()
    assert configured["alphas"] == [2.0] and configured["layers"] == [3, 1] and configured["last"] == 256
    for bad in ({"columns": 0}, {"columns": 65}, {"columns": 2.0}, {"alphas": []}, {"alphas": [0.0]},
                {"alphas": [float("nan")]}, {"alphas": "6"}, {"layers": []}, {"layers": [-1]}, {"layers": [1, 1]},
                {"last": 0}, {"column_rule": "norm"}, {"max_texts": 2}, {"wait_seconds": 0}, {"wait_seconds": 11},
                {"suite_id": ""}, {"sample_count": -1}):
        with pytest.raises(ValueError):
            DirectionSearchStrategy(**bad)


def test_engine_limits_the_plan_relies_on_match_their_sources():
    from wse import jobs, policy, probes
    from wse.strategies import direction_search

    assert direction_search.MAX_INLINE_TOP == policy.MAX_TOP_K == probes.MAX_SUMMARY_TOP
    assert direction_search.MAX_PROBE_SITES == probes.MAX_PROBE_SITES
    assert direction_search.MAX_WAIT_SECONDS == jobs.MAX_WAIT_SECONDS


# ---------------------------------------------------------------------------
# Decision helpers
# ---------------------------------------------------------------------------


def test_decoder_layers_come_from_the_configuration_or_its_text_config():
    def manifest(model_config):
        return {"config": {"model_config": model_config, "family": "x@1"}}

    assert decoder_layer_count(manifest({"num_hidden_layers": 16})) == 16
    assert decoder_layer_count(manifest({"n_layer": 12})) == 12
    # A multimodal wrapper: the language model's layers, never the vision tower's.
    assert decoder_layer_count(manifest({"text_config": {"num_hidden_layers": 24},
                                         "vision_config": {"num_hidden_layers": 27}})) == 24
    for config in ({}, {"num_hidden_layers": True}, {"num_hidden_layers": 0}, {"vision_config": {"depth": 12}}, None):
        assert decoder_layer_count(manifest(config)) is None
    assert decoder_layer_count({}) is None


def test_outputs_are_grouped_by_limit_hit_and_finished_within_the_text_limit():
    grouped = output_groups(_report("LFOLF"), 64)
    uri = [f"wse://analyses/{index:064x}" for index in range(5)]
    assert grouped == {"a": [uri[0], uri[3]], "b": [uri[1], uri[4]], "limit_hit": 2, "finished": 2, "other": 1}
    # An output without a URI is left out.
    report = _report("LF")
    report["problems"][0]["samples"][0]["output_uri"] = None
    assert output_groups(report, 64) | {"a": None} == {"a": None, "b": [uri[1]], "limit_hit": 0, "finished": 1,
                                                         "other": 1}
    # Over the limit, each group keeps its first outputs, and at least half of the probe when it has them.
    for outcomes, kept in (("L" * 50 + "F" * 40, (32, 32)), ("L" * 60 + "F" * 4, (60, 4)),
                           ("L" * 3 + "F" * 70, (3, 61)), ("L" * 10 + "F" * 10, (10, 10))):
        grouped = output_groups(_report(outcomes), 64)
        assert (len(grouped["a"]), len(grouped["b"])) == kept
        assert grouped["a"] == [f"wse://analyses/{index:064x}" for index in range(kept[0])]
    assert output_groups({"problems": None}, 64) == {"a": [], "b": [], "limit_hit": 0, "finished": 0, "other": 0}


def test_the_layer_of_largest_relative_contrast_is_chosen():
    assert separation({"group_mean_norms": [3.0, 5.0], "contrast_norm": 2.0}) == 0.5
    for site in ({"group_mean_norms": [0.0, 0.0], "contrast_norm": 1.0}, {"mean_norm": 3.0},
                 {"group_mean_norms": ["inf", 1.0], "contrast_norm": 1.0},
                 {"group_mean_norms": [1.0, 1.0], "contrast_norm": "nan"}):
        assert separation(site) is None
    records = [{"layer": 0, "separation": 0.2}, {"layer": 1, "separation": None}, {"layer": 2, "separation": 0.7},
               {"layer": 3, "separation": 0.7}]
    assert choose_layer(records)["layer"] == 2  # the lower layer on a tie
    assert choose_layer([{"layer": 0, "separation": None}]) is None
    # The contrast norm alone would pick the late layer, whose outputs are larger; relative to them it is smaller.
    world = World(layers=2, norms={(0, "input"): (1.0, 1.0, 0.1), (0, "output"): (2.0, 2.0, 1.0),
                                   (1, "input"): (1.0, 1.0, 0.1), (1, "output"): (40.0, 40.0, 4.0)})
    actions = drive(DirectionSearchStrategy(columns=4), world)
    assert _calls(actions, "wse_analyze")[0].arguments["tensor"] == f"{_down(0)}.weight"


def test_the_layer_and_the_direction_come_from_the_output_site_not_the_input():
    """The plan probes 'at the input of mlp.down'; the strategy probes input and output and ranks layers, and takes
    the direction, at the output (the residual space that project_out on mlp.down needs). Here the two sites rank the
    layers differently: the input would pick layer 0, the output picks layer 1."""
    norms = {(0, "input"): (1.0, 1.0, 1.5), (0, "output"): (10.0, 10.0, 1.0),
             (1, "input"): (1.0, 1.0, 0.1), (1, "output"): (10.0, 10.0, 5.0)}
    world = World(layers=2, norms=norms)
    actions = drive(DirectionSearchStrategy(columns=4), world)
    (probe,) = _calls(actions, "wse_probe")
    assert [(site["layer"], site["point"]) for site in probe.arguments["sites"]] == [
        (0, "input"), (0, "output"), (1, "input"), (1, "output")]
    (analysis,) = _calls(actions, "wse_analyze")
    reference = "wse://analyses/" + f"{1:064x}" + "#/sites/3/contrast/direction"  # layer 1's output site
    assert analysis.arguments["tensor"] == f"{_down(1)}.weight"
    assert analysis.arguments["parameters"]["direction"] == reference
    assert all(p.arguments["patch_spec"]["operations"][0]["direction"] == reference
               for p in _calls(actions, "wse_preview_patch"))
    assert actions[-1].reason.endswith("at layer 1 (limit-hit/accuracy: base 0.5/0.5, a1 0.25/0.5, a6 0/0.25)")


def test_helpers_read_tensors_columns_and_metrics():
    manifest = World(layers=2).manifest()
    assert role_tensor_of_site(f"{_down(1)}:output", manifest)["name"] == f"{_down(1)}.weight"
    assert role_tensor_of_site("model.layers.9.mlp.down_proj:output", manifest) is None
    biased = {"tensors": [{"name": "m.lin.weight", "shape": [4, 8]}, {"name": "m.lin.bias", "shape": [4]}]}
    assert role_tensor_of_site("m.lin:output", biased)["name"] == "m.lin.weight"
    analysis = {"result": {"top": [{"index": 9, "cosine": -0.9}, {"index": 2, "cosine": 0.5}, {"index": 4,
                                                                                               "cosine": 0.1}]}}
    assert top_columns(analysis, 2) == [2, 9] and top_columns(analysis, 9) == [2, 4, 9]
    base = {"accuracy": 0.5, "limit_hit_rate": 0.5}
    assert improves(base, {"accuracy": 0.5, "limit_hit_rate": 0.25})
    assert not improves(base, {"accuracy": 0.49, "limit_hit_rate": 0.0})
    assert not improves(base, {"accuracy": 0.9, "limit_hit_rate": 0.5})
    assert rank_key({"accuracy": 0.6, "limit_hit_rate": 0.2, "mean_generated_tokens": 5}) < \
        rank_key({"accuracy": 0.5, "limit_hit_rate": 0.2, "mean_generated_tokens": 1}) < \
        rank_key({"accuracy": 1.0, "limit_hit_rate": 0.3, "mean_generated_tokens": 1})


# ---------------------------------------------------------------------------
# The search, step by step
# ---------------------------------------------------------------------------


def test_a_search_diagnoses_probes_every_layer_edits_the_chosen_one_and_selects():
    world = World()
    actions = drive(DirectionSearchStrategy(columns=4), world)
    assert _tools(actions) == [
        "wse_inspect_model", "wse_submit_evaluation", "wse_get_job", "wse_get_report", "wse_probe", "wse_get_job",
        "wse_analyze", "wse_preview_patch", "wse_apply_patch", "wse_preview_patch", "wse_apply_patch",
        "wse_submit_evaluation", "wse_get_job", "wse_compare", "wse_select_checkpoint",
        "wse_submit_evaluation", "wse_get_job", "wse_compare", "stop"]
    # 1. The diagnosis pass evaluates the base on every practice problem with the suite's first seed.
    (base_pass, *candidate_passes) = _calls(actions, "wse_submit_evaluation")
    assert base_pass.arguments | {"idempotency_key": None} == {
        "experiment_id": EID, "checkpoint_id": BASE, "suite_id": "practice",
        "eval_config": {"sample_count": 6, "seed": 0}, "idempotency_key": None}
    assert _calls(actions, "wse_get_report")[0].arguments == {"experiment_id": EID, "report_id": "report_base"}
    # 2. The probe contrasts the limit-hit outputs (a) with the finished ones (b) at mlp.down input and output.
    (probe,) = _calls(actions, "wse_probe")
    uris = [f"wse://analyses/{index:064x}" for index in range(6)]
    assert probe.arguments["texts"] == [{"output_uri": uris[k]} for k in (0, 1, 3, 2, 4, 5)]
    assert probe.arguments["groups"] == {"a": [0, 1, 2], "b": [3, 4, 5]}
    assert probe.arguments["sites"] == [{"role": "mlp.down", "layer": layer, "point": point}
                                        for layer in range(3) for point in ("input", "output")]
    assert probe.arguments["checkpoint_id"] == BASE and "top" not in probe.arguments
    # The direction is layer 1's output contrast (relative contrast 4/20), by reference.
    reference = "wse://analyses/" + f"{1:064x}" + "#/sites/3/contrast/direction"
    (analysis,) = _calls(actions, "wse_analyze")
    assert analysis.arguments == {
        "experiment_id": EID, "checkpoint_id": BASE, "kind": "direction_projection", "tensor": f"{_down(1)}.weight",
        "selection": [{"start": 0, "stop": 16, "step": 1}, {"start": 0, "stop": 32, "step": 1}],
        "parameters": {"direction": reference, "axis": "columns", "measure": "cosine", "top": 4}}
    # 3. One project_out per alpha: the top 4 columns by |cosine|, sorted, from the whole weight.
    previews = _calls(actions, "wse_preview_patch")
    for preview, alpha in zip(previews, (1.0, 6.0), strict=True):
        spec = preview.arguments["patch_spec"]
        assert spec["base_checkpoint_id"] == BASE and spec["development_suite_id"] == "practice"
        assert spec["operations"] == [{"op": "project_out", "tensor": f"{_down(1)}.weight",
                                       "selection": analysis.arguments["selection"], "direction": reference,
                                       "columns": [2, 5, 17, 30], "alpha": alpha}]
        assert "layer 1" in spec["hypothesis"] and spec["expected_behavior"]
    # 4. Each candidate is evaluated on the same problems; the improving one is selected at once.
    assert [p.arguments["checkpoint_id"] for p in candidate_passes] == world.commits
    assert all(p.arguments["eval_config"] == base_pass.arguments["eval_config"] for p in candidate_passes)
    assert [c.arguments["candidate_report_id"] for c in _calls(actions, "wse_compare")] == ["report_1", "report_2"]
    (select,) = _calls(actions, "wse_select_checkpoint")
    assert select.arguments | {"idempotency_key": None} == {
        "experiment_id": EID, "checkpoint_id": world.commits[0], "expected_selected_id": BASE, "idempotency_key": None}
    assert world.selected == world.commits[0]
    stop = actions[-1]
    assert stop.reason == ("selected alpha 1 at layer 1 (limit-hit/accuracy: base 0.5/0.5, a1 0.25/0.5, "
                           "a6 0/0.25)")
    # Every call states an observable rationale, and every mutating call has its own idempotency key.
    calls = [action for action in actions if action.kind == "call"]
    assert all(0 < len(action.rationale) <= 500 for action in calls)
    keys = [action.arguments["idempotency_key"] for action in calls if "idempotency_key" in action.arguments]
    assert len(keys) == len(set(keys)) == 9  # three passes, a probe, two previews, two commits, a selection
    assert all(len(action.reason or "") <= 200 for action in actions)


@pytest.mark.parametrize("metrics, selected, reason", [
    # Both improve; the second has the lower limit-hit rate, so it replaces the first.
    ({"base": (0.4, 0.6, 9.0), 1: (0.4, 0.3, 8.0), 2: (0.5, 0.2, 7.0)}, [1, 2], "selected alpha 6"),
    # The same limit-hit rate: the higher accuracy wins; at equal accuracy too, the first is kept.
    ({"base": (0.4, 0.6, 9.0), 1: (0.4, 0.3, 8.0), 2: (0.5, 0.3, 9.0)}, [1, 2], "selected alpha 6"),
    ({"base": (0.4, 0.6, 9.0), 1: (0.4, 0.3, 8.0), 2: (0.4, 0.3, 8.0)}, [1], "selected alpha 1"),
    # Lower accuracy disqualifies, however few outputs reach the limit.
    ({"base": (0.4, 0.6, 9.0), 1: (0.3, 0.0, 2.0), 2: (0.4, 0.5, 8.0)}, [2], "selected alpha 6"),
    # No candidate lowers the limit-hit rate at no lower accuracy: the base stays selected.
    ({"base": (0.4, 0.6, 9.0), 1: (0.3, 0.0, 2.0), 2: (0.9, 0.6, 8.0)}, [], "kept the base"),
])
def test_selection_prefers_the_lowest_limit_hit_rate_at_no_lower_accuracy(metrics, selected, reason):
    world = World(metrics=metrics)
    actions = drive(DirectionSearchStrategy(columns=4), world)
    selects = _calls(actions, "wse_select_checkpoint")
    assert [world.commits.index(s.arguments["checkpoint_id"]) + 1 for s in selects] == selected
    # Each selection is a compare-and-swap on the previous one.
    expected = [BASE] + [s.arguments["checkpoint_id"] for s in selects]
    assert [s.arguments["expected_selected_id"] for s in selects] == expected[:len(selects)]
    assert world.selected == expected[-1]
    assert actions[-1].reason.startswith(reason)
    assert "limit-hit/accuracy: base 0.6/0.4" in actions[-1].reason


@pytest.mark.parametrize("outcomes", ["LLLL", "FFFF", "LF", "OOLO", ""])
def test_it_stops_before_probing_when_a_group_is_empty_or_too_small(outcomes):
    world = World(outcomes=outcomes)
    actions = drive(DirectionSearchStrategy(), world)
    assert _tools(actions) == ["wse_inspect_model", "wse_submit_evaluation", "wse_get_job", "wse_get_report", "stop"]
    assert actions[-1].reason.startswith(f"no contrast: {outcomes.count('L')} outputs reached the token limit and "
                                         f"{outcomes.count('F')} finished")
    assert world.probes == []


def test_the_window_the_text_limit_and_the_evaluation_configuration_are_the_operators():
    world = World(outcomes="LLLFFFL")
    actions = drive(DirectionSearchStrategy(columns=2, alphas=[3.0], last=256, max_texts=4, sample_count=3,
                                            compare=False, wait_seconds=0.25), world)
    (probe,) = _calls(actions, "wse_probe")
    assert probe.arguments["texts"] == [{"output_uri": f"wse://analyses/{k:064x}", "last": 256} for k in (0, 1, 3, 4)]
    assert probe.arguments["groups"] == {"a": [0, 1], "b": [2, 3]}
    assert {p.arguments["eval_config"]["sample_count"] for p in _calls(actions, "wse_submit_evaluation")} == {3}
    assert {p.arguments["wait_seconds"] for p in _calls(actions, "wse_get_job")} == {0.25}
    assert _calls(actions, "wse_compare") == [] and len(_calls(actions, "wse_preview_patch")) == 1
    assert _calls(actions, "wse_preview_patch")[0].arguments["patch_spec"]["operations"][0]["columns"] == [2, 30]


def test_explicit_layers_are_probed_and_layers_without_the_role_are_dropped_after_a_refusal():
    contrasts = [0.1, 0.9, 0.5, 0.2]  # unit group-mean norms: layer 1 would separate most, but it has no mlp.down
    norms = {(layer, point): (1.0, 1.0, contrasts[layer]) for layer in range(4) for point in ("input", "output")}
    world = World(layers=4, refuse_layers=[0, 2, 3], norms=norms)
    actions = drive(DirectionSearchStrategy(columns=4), world)
    first, second = world.probes
    assert [site["layer"] for site in first["sites"]] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [site["layer"] for site in second["sites"]] == [0, 0, 2, 2, 3, 3]
    # Layer 2 separates most among those probed; its output is site 3 of the second probe.
    reference = "wse://analyses/" + f"{2:064x}" + "#/sites/3/contrast/direction"
    assert _calls(actions, "wse_analyze")[0].arguments["parameters"]["direction"] == reference
    # Explicit layers are probed as given, and a layer without the role then stops the search.
    world = World(layers=4, refuse_layers=[0, 2, 3])
    actions = drive(DirectionSearchStrategy(layers=[3, 1]), world)
    assert [site["layer"] for site in world.probes[0]["sites"]] == [3, 3, 1, 1] and len(world.probes) == 1
    assert actions[-1].reason.startswith("the probe failed with NOT_FOUND")


def test_sites_that_do_not_fit_one_summary_are_split_over_several_probes():
    world = World(layers=3, max_sites=4)
    actions = drive(DirectionSearchStrategy(columns=4), world)
    assert [[site["layer"] for site in probe["sites"]] for probe in world.probes] == [
        [0, 0, 1, 1, 2, 2], [0, 0, 1, 1], [2, 2]]
    # Layer 1 is chosen from the second probe (the first was refused): its output is that probe's site 3.
    reference = "wse://analyses/" + f"{2:064x}" + "#/sites/3/contrast/direction"
    assert _calls(actions, "wse_analyze")[0].arguments["parameters"]["direction"] == reference
    # A layer chosen from the last probe is referenced in it.
    norms = {(layer, point): (1.0, 1.0, 0.1 * (layer + 1)) for layer in range(3) for point in ("input", "output")}
    world = World(layers=3, max_sites=4, norms=norms)
    actions = drive(DirectionSearchStrategy(columns=4), world)
    reference = "wse://analyses/" + f"{3:064x}" + "#/sites/1/contrast/direction"
    assert _calls(actions, "wse_analyze")[0].arguments["parameters"]["direction"] == reference
    assert _calls(actions, "wse_analyze")[0].arguments["tensor"] == f"{_down(2)}.weight"


def test_the_input_contrast_rule_takes_columns_from_a_probe_of_the_chosen_input():
    world = World()
    actions = drive(DirectionSearchStrategy(columns=3, column_rule="input_contrast"), world)
    assert _calls(actions, "wse_analyze") == []
    first, second = world.probes
    assert second["sites"] == [{"role": "mlp.down", "layer": 1, "point": "input"}] and second["top"] == 3
    assert second["texts"] == first["texts"] and second["groups"] == first["groups"]
    operation = _calls(actions, "wse_preview_patch")[0].arguments["patch_spec"]["operations"][0]
    assert operation["columns"] == [3, 11, 29]
    assert operation["direction"] == "wse://analyses/" + f"{1:064x}" + "#/sites/3/contrast/direction"
    projection = drive(DirectionSearchStrategy(columns=2, column_rule="projection"), World())
    assert _calls(projection, "wse_analyze")[0].arguments["parameters"]["measure"] == "projection"


def test_an_alpha_that_changes_nothing_is_not_committed():
    world = World(changed={1.0: 0})
    actions = drive(DirectionSearchStrategy(columns=4), world)
    assert len(_calls(actions, "wse_preview_patch")) == 2 and len(world.commits) == 1
    assert _calls(actions, "wse_apply_patch")[0].rationale.startswith("Commit the previewed candidate with alpha 6")
    world = World(changed={1.0: 0, 6.0: 0})
    actions = drive(DirectionSearchStrategy(columns=4), world)
    assert actions[-1].reason == "no alpha changed any weight of the chosen columns" and world.commits == []


def test_a_paged_inspection_is_read_to_its_last_page():
    # Layer 2 separates most; its mlp.down weight is on the last of three pages.
    norms = {(layer, point): (1.0, 1.0, 0.1 * (layer + 1)) for layer in range(3) for point in ("input", "output")}
    world = World(layers=3, pages=3, norms=norms)
    actions = drive(DirectionSearchStrategy(columns=4), world)
    assert [call.get("cursor") for call in world.inspections] == [None, "page:1", "page:2"]
    assert _tools(actions)[:4] == ["wse_inspect_model", "wse_inspect_model", "wse_inspect_model",
                                   "wse_submit_evaluation"]
    assert _calls(actions, "wse_analyze")[0].arguments["tensor"] == f"{_down(2)}.weight"
    assert actions[-1].reason.startswith("selected alpha 1 at layer 2")


def test_a_report_too_large_to_read_inline_stops_the_search():
    world = World(overview=True)
    actions = drive(DirectionSearchStrategy(), world)
    assert _tools(actions) == ["wse_inspect_model", "wse_submit_evaluation", "wse_get_job", "wse_get_report", "stop"]
    assert actions[-1].reason == ("the base report (40960 bytes) exceeds the 31744-byte inline limit, so its outputs "
                                  "cannot be read; use fewer problems")


@pytest.mark.parametrize("change, reason", [
    ({"family": None}, "the model has no family descriptor"),
    ({"config": {"hidden_size": 16}}, "the model configuration does not give its decoder layers"),
])
def test_a_model_without_roles_or_layers_is_refused_before_any_pass(change, reason):
    world = World(**change)
    actions = drive(DirectionSearchStrategy(), world)
    assert _tools(actions) == ["wse_inspect_model", "stop"] and actions[-1].reason.startswith(reason)


@pytest.mark.parametrize("operations, reason", [
    ({"analyses": {"direction_projection": {}}, "edits": {}, "limits": {}}, "the policy permits no project_out"),
    ({"analyses": {}, "edits": {"project_out": {}}, "limits": {}}, "the policy permits no direction_projection"),
])
def test_a_policy_without_the_operations_is_refused_before_anything_runs(operations, reason):
    actions = drive(DirectionSearchStrategy(), World(operations=operations))
    assert _tools(actions) == ["stop"] and actions[-1].reason.startswith(reason)
    # The input-contrast rule needs no analysis.
    without_analysis = World(operations={"analyses": {}, "edits": {"project_out": {}}, "limits": {}})
    if "direction_projection" in reason:
        assert drive(DirectionSearchStrategy(column_rule="input_contrast"), without_analysis)[-1].reason.startswith(
            "selected")


# ---------------------------------------------------------------------------
# Budget and failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("budget, reason", [
    ({"generation_passes": 1}, "insufficient budget for the smallest plan (a base pass, a probe, an edit, and a "
                               "candidate pass): generation_passes has 1 left and 2 is needed"),
    ({"probe_calls": 0}, "insufficient budget for the smallest plan"),
    ({"candidate_commits": 0}, "insufficient budget for the smallest plan"),
    ({"analysis_calls": 0}, "insufficient budget for the smallest plan"),
])
def test_a_budget_that_cannot_cover_the_smallest_plan_stops_before_anything_runs(budget, reason):
    actions = drive(DirectionSearchStrategy(), World(budget=budget))
    assert _tools(actions) == ["stop"] and actions[-1].reason.startswith(reason)


def test_only_affordable_candidates_are_built_and_evaluated():
    # Two passes left: the base pass and one candidate, so only the first alpha is built.
    world = World(budget={"generation_passes": 2})
    actions = drive(DirectionSearchStrategy(columns=4), world)
    assert len(_calls(actions, "wse_preview_patch")) == 1 and len(_calls(actions, "wse_submit_evaluation")) == 2
    assert actions[-1].reason.startswith("selected alpha 1")
    # Generated tokens for two passes (each reserves 48): the same.
    world = World(budget={"generated_tokens": 96})
    actions = drive(DirectionSearchStrategy(columns=4), world)
    assert len(world.commits) == 1 and actions[-1].reason.startswith("selected alpha 1")
    # One commit left: one candidate.
    world = World(budget={"candidate_commits": 1})
    assert len(_calls(drive(DirectionSearchStrategy(columns=4), world), "wse_apply_patch")) == 1


def test_a_budget_spent_between_candidates_ends_the_search_with_the_best_so_far():
    world = World(metrics={"base": (0.5, 0.5, 9.0), 1: (0.5, 0.25, 8.0), 2: (0.5, 0.0, 6.0)})
    strategy = DirectionSearchStrategy(columns=4)
    observations = [Observation(tool="wse_get_experiment", arguments={"experiment_id": EID}, ok=True,
                                result=world.description())]
    actions: list[Action] = []
    while True:
        action = strategy.next_action(None, list(observations), world.remaining())
        actions.append(action)
        if action.kind == "stop":
            break
        if action.tool == "wse_select_checkpoint":  # another consumer spends the last generated tokens
            world.budget["generated_tokens"] = 10
        observations.append(world.execute(action))
    assert len(_calls(actions, "wse_submit_evaluation")) == 2 and world.selected == world.commits[0]
    assert actions[-1].reason.startswith("budget ended after 1 of 2 candidates (generated_tokens has 10 left and 48 "
                                         "is needed); selected alpha 1")


@pytest.mark.parametrize("tool, code", [("wse_probe", "BUDGET_EXCEEDED"), ("wse_preview_patch", "BUDGET_EXCEEDED"),
                                        ("wse_get_report", "POLICY_DENIED"), ("wse_analyze", "INVALID_ARGUMENT")])
def test_a_failed_operation_stops_the_search_with_its_error(tool, code):
    world = World(fail={tool: code})
    actions = drive(DirectionSearchStrategy(columns=4), world)
    assert actions[-2].tool == tool and actions[-1].kind == "stop"
    assert re.fullmatch(rf".* failed with {code}: refused", actions[-1].reason)


def test_a_failed_job_stops_the_search():
    world = World()
    original = world.wse_get_job

    def failing(action):
        shown = original(action)
        if shown["state"] == "succeeded" and shown["kind"] == "probe":
            shown.update(state="failed", result=None, error={"code": "WORKER_FAILED", "message": "timeout"})
        return shown

    world.wse_get_job = failing
    actions = drive(DirectionSearchStrategy(), world)
    assert actions[-1].reason.startswith("the probe of layers 0-2 (job job_") and \
        actions[-1].reason.endswith("ended failed: WORKER_FAILED timeout")


def test_a_competing_selection_stops_the_search():
    world = World()
    world.selected = "ckpt_" + "f" * 32  # another controller selected a checkpoint meanwhile
    actions = drive(DirectionSearchStrategy(columns=4), world)
    assert actions[-1].reason.startswith("selecting candidate alpha 1 failed with VERSION_CONFLICT")


# ---------------------------------------------------------------------------
# End to end through the reference runner (tiny Llama fixture, qwen_loop_v1)
# ---------------------------------------------------------------------------


def _scripted_text(order: int):
    """The scripted outputs of the checkpoint the worker loaded ``order``-th (0 is the base, the first job's model).

    The base answers even problems and stops (10 tokens) and loops on odd ones
    to the 12-token limit; the first candidate answers only problem 0; every
    later candidate answers all four.
    """

    def text_for(prompt: str) -> str:
        index = int(re.search(r"p(\d+) plus", prompt).group(1))
        answer = f"\\boxed{{{2 * index}}}<STOP>"
        if order == 0:
            return answer if index % 2 == 0 else "loop " * 10
        if order == 1:
            return answer if index == 0 else "loop " * 10
        return answer

    return text_for


def _install_scripted_texts(texts) -> None:
    """Replace the worker's text models with scripted runtimes (``test_llm_policy_mcp.ScriptedRuntime``).

    ``texts(order)`` gives the outputs of the checkpoint the worker loaded
    ``order``-th (0 is the base). Probes still read the real fixture weights.
    """
    from test_llm_policy_mcp import ScriptedRuntime
    from wse.evaluators import generation

    original = generation.load_text_model
    order: dict[str, int] = {}

    def load(handle, files, *, device=None):
        real = original(handle, files, device="cpu")
        position = order.setdefault(handle.weights_hash, len(order))
        runtime = ScriptedRuntime(real.tokenizer, texts(position), real.stop_token_ids[0])
        return generation.TextModel(runtime=runtime, tokenizer=real.tokenizer, stop_token_ids=real.stop_token_ids,
                                    pad_token_id=real.pad_token_id)

    generation.load_text_model = load


def scripted_generation(registry) -> None:
    """Worker setup hook ``test_direction_search:scripted_generation`` (``tests`` must be on ``PYTHONPATH``).

    Scripted outputs that depend on the order in which the worker first loads
    each checkpoint (``_scripted_text``).
    """
    _install_scripted_texts(_scripted_text)


def _same_accuracy_until_third(order: int):
    """The base and the first two candidates answer even problems and loop on odd ones; from the third candidate on,
    odd problems finish with a wrong answer: the same accuracy, and no output reaches the limit."""

    def text_for(prompt: str) -> str:
        index = int(re.search(r"p(\d+) plus", prompt).group(1))
        if index % 2 == 0:
            return f"\\boxed{{{2 * index}}}<STOP>"
        return "\\boxed{999}<STOP>" if order >= 3 else "loop " * 10

    return text_for


def same_accuracy_until_third(registry) -> None:
    """Worker setup hook ``test_direction_search:same_accuracy_until_third`` (``_same_accuracy_until_third``)."""
    _install_scripted_texts(_same_accuracy_until_third)


def _register_loop_task(workspace) -> None:
    from wse.schemas import TaskDefinition

    with Engine(workspace) as engine:
        engine.register_task(TaskDefinition.model_validate({
            "task_id": TASK, "title": "Finish answers", "description": "Make the model finish its answers.",
            "objective": {"task_id": TASK, "description": "Fewer outputs at the token limit, no lower accuracy.",
                          "gates": [{"metric": "accuracy", "min_delta": 0.0}]},
            "development_suite_id": "practice", "development_examples": 4}))


def test_with_three_alphas_the_runner_lets_the_strategy_select_a_lower_limit_hit_rate(lm, monkeypatch):  # noqa: F811
    """Candidates 1 and 2 change nothing measurable; candidate 3 keeps the accuracy and no output reaches the limit.
    The runner's no-improvement patience (3 candidates) counts that as an improvement, so the strategy selects it."""
    from wse.policy import QWEN_LOOP_POLICY_ID
    from wse.registry import builtin_registry
    from wse.runner import ReferenceRunner

    monkeypatch.setenv("PYTHONPATH", str(TESTS_DIR))  # the worker imports this module's setup hook
    _register_loop_task(lm.workspace)
    with Engine(lm.workspace) as engine:
        grant = engine.grant_run(TASK, QWEN_LOOP_POLICY_ID, lm.base)
    runner = ReferenceRunner(lm.workspace, registry=builtin_registry(),
                             setup=["test_direction_search:same_accuracy_until_third"],
                             worker_args=["--poll-seconds", "0.02", "--device", "cpu"])
    eid = runner.run_experiment(grant.run_id, TASK, QWEN_LOOP_POLICY_ID, {
        "kind": "strategy", "strategy": "direction_search@1",
        "parameters": {"columns": 4, "alphas": [1.0, 2.0, 6.0]}, "max_steps": 400, "wall_clock_seconds": 600})
    result = runner.last_result["result"]
    assert result["status"] == "completed" and result["stop_condition"] is None, result
    assert re.fullmatch(r"the strategy stopped: selected alpha 6 at layer [01] \(limit-hit/accuracy: base "
                        r"0\.5/0\.5, a1 0\.5/0\.5, a2 0\.5/0\.5, a6 0/0\.5\)", result["reason"]), result["reason"]
    with Engine(lm.workspace) as engine:
        experiment = engine.get_experiment(eid)
        selected = next(c for c in engine.list_experiment_checkpoints(eid)
                        if c.checkpoint_id == experiment.selected_checkpoint_id)
        assert selected.role == "candidate"
        assert engine.find_patch(selected.patch_id).spec.operations[0].parameters["alpha"] == 6.0


def test_direction_search_runs_end_to_end_through_the_reference_runner(lm, monkeypatch):  # noqa: F811
    from wse.policy import QWEN_LOOP_POLICY_ID
    from wse.registry import builtin_registry
    from wse.runner import ReferenceRunner
    from wse.schemas import TaskDefinition
    from wse.strategies.base import STRATEGY_TOOLS

    assert "wse_get_report" in STRATEGY_TOOLS, "the runner's dispatch has no wse_get_report action yet"
    monkeypatch.setenv("PYTHONPATH", str(TESTS_DIR))  # the worker imports this module's setup hook
    with Engine(lm.workspace) as engine:
        engine.register_task(TaskDefinition.model_validate({
            "task_id": TASK, "title": "Finish answers", "description": "Make the model finish its answers.",
            "objective": {"task_id": TASK, "description": "Fewer outputs at the token limit, no lower accuracy.",
                          "gates": [{"metric": "accuracy", "max_drop": 0.0}, {"metric": "accuracy", "min_value": 0.0}]},
            "development_suite_id": "practice", "development_examples": 4}))
        grant = engine.grant_run(TASK, QWEN_LOOP_POLICY_ID, lm.base)
    runner = ReferenceRunner(lm.workspace, registry=builtin_registry(),
                             setup=["test_direction_search:scripted_generation"],
                             worker_args=["--poll-seconds", "0.02", "--device", "cpu"])
    eid = runner.run_experiment(grant.run_id, TASK, QWEN_LOOP_POLICY_ID, {
        "kind": "strategy", "strategy": "direction_search@1", "parameters": {"columns": 4},
        "max_steps": 400, "wall_clock_seconds": 600})
    record = runner.last_result
    result = record["result"]
    assert result["status"] == "completed", result
    assert record["config"]["resolved_config"]["alphas"] == [1.0, 6.0]
    assert re.fullmatch(r"the strategy stopped: selected alpha 6 at layer [01] \(limit-hit/accuracy: base 0\.5/0\.5, "
                        r"a1 0\.75/0\.25, a6 0/1\)", result["reason"]), result["reason"]
    with Engine(lm.workspace) as engine:
        experiment = engine.get_experiment(eid)
        assert experiment.state == "FROZEN"
        jobs = engine.list_jobs(eid)
        assert [(job.kind, job.state) for job in jobs] == [("evaluation", "succeeded"), ("probe", "succeeded"),
                                                           ("evaluation", "succeeded"), ("evaluation", "succeeded")]
        budget = experiment.budget
        consumed = {name: budget[name].consumed for name in ("generation_passes", "probe_calls", "analysis_calls",
                                                             "patch_previews", "candidate_commits")}
        assert consumed == {"generation_passes": 3, "probe_calls": 1, "analysis_calls": 1, "patch_previews": 2,
                            "candidate_commits": 2}
        # The probe contrasted the four looping outputs (odd problems) with the four finished ones, by output URI.
        request = json.loads(engine.store.db.conn.execute("SELECT config_json FROM jobs WHERE job_id = ?",
                                                          (jobs[1].job_id,)).fetchone()["config_json"])
        base_report = engine.get_report(eid, jobs[0].report_id)
        samples = [(problem.index, sample) for problem in base_report.problems for sample in problem.samples]
        looping = [s.output_uri for index, s in samples if index % 2 == 1]
        finished = [s.output_uri for index, s in samples if index % 2 == 0]
        assert all(s.limit_hit for index, s in samples if index % 2 == 1) and all(s.finished for index, s in samples
                                                                                if index % 2 == 0)
        assert [text["output_uri"] for text in request["texts"]] == looping + finished
        assert request["groups"] == {"a": [0, 1, 2, 3], "b": [4, 5, 6, 7]}
        assert [site["site"] for site in request["sites"]] == [f"{_down(layer)}:{point}" for layer in (0, 1)
                                                               for point in ("input", "output")]
        # The selected candidate projects the probe's output contrast of the chosen layer out of 4 columns, alpha 6.
        selected = next(c for c in engine.list_experiment_checkpoints(eid)
                        if c.checkpoint_id == experiment.selected_checkpoint_id)
        assert selected.role == "candidate"
        operation = engine.find_patch(selected.patch_id).spec.operations[0]
        layer = int(re.search(r"layer (\d)", result["reason"]).group(1))
        assert operation.op == "project_out" and operation.tensor == f"{_down(layer)}.weight"
        assert operation.parameters["alpha"] == 6.0 and len(operation.parameters["columns"]) == 4
        assert operation.parameters["direction"]["reference"] == \
            f"{jobs[1].result['result_uri']}#/sites/{2 * layer + 1}/contrast/direction"
        rows = engine.store.db.conn.execute("SELECT payload FROM controller_events WHERE experiment_id = ? AND "
                                            "kind = 'tool_call' ORDER BY seq", (eid,)).fetchall()
        calls = [json.loads(row["payload"]) for row in rows]
    tools = [call["tool"] for call in calls]
    assert tools[:3] == ["wse_get_experiment", "wse_inspect_model", "wse_submit_evaluation"]
    assert "wse_get_report" in tools and tools.count("wse_select_checkpoint") == 1
    assert all(call.get("details", {}).get("rationale") for call in calls)


def test_direction_search_runs_through_mcp_with_the_scripts_bound_strategy(lm, monkeypatch):  # noqa: F811
    """``--via-mcp``: Codex's stand-in plays the bound subclass the script registered in the runner's process only."""
    from wse.policy import QWEN_LOOP_POLICY_ID
    from wse.registry import builtin_registry
    from wse.runner import ReferenceRunner
    from wse.schemas import TaskDefinition

    loop_setup = _script("loop_setup")
    monkeypatch.setenv("PYTHONPATH", str(TESTS_DIR))
    with Engine(lm.workspace) as engine:
        engine.register_task(TaskDefinition.model_validate({
            "task_id": TASK, "title": "Finish answers", "description": "Make the model finish its answers.",
            "objective": {"task_id": TASK, "description": "Fewer outputs at the token limit, no lower accuracy.",
                          "gates": [{"metric": "accuracy", "max_drop": 0.0}, {"metric": "accuracy", "min_value": 0.0}]},
            "development_suite_id": "practice", "development_examples": 4}))
        grant = engine.grant_run(TASK, QWEN_LOOP_POLICY_ID, lm.base)
    registry = builtin_registry()
    config = loop_setup.controller(registry, DirectionSearchStrategy, {"columns": 4}, via_mcp=True, max_steps=400,
                                   wall_clock_seconds=600)
    runner = ReferenceRunner(lm.workspace, registry=registry, setup=["test_direction_search:scripted_generation"],
                             worker_args=["--poll-seconds", "0.02", "--device", "cpu"],
                             serve_env={"PYTHONPATH": str(TESTS_DIR)}, settle_timeout_seconds=60)
    eid = runner.run_experiment(grant.run_id, TASK, QWEN_LOOP_POLICY_ID, config)
    record = runner.last_result
    result = record["result"]
    assert result["status"] == "completed", result
    assert record["config"]["test_double"]["strategy"] == "direction_search_mcp@1"
    assert record["config"]["test_double"]["resolved_config"]["columns"] == 4  # the script's parameters applied
    assert re.fullmatch(r"WSE-STOP: selected alpha 6 at layer [01] \(limit-hit/accuracy: base 0\.5/0\.5, "
                        r"a1 0\.75/0\.25, a6 0/1\)", result["final_message"]), result["final_message"]
    with Engine(lm.workspace) as engine:
        experiment = engine.get_experiment(eid)
        assert experiment.state == "FROZEN"
        selected = next(c for c in engine.list_experiment_checkpoints(eid)
                        if c.checkpoint_id == experiment.selected_checkpoint_id)
        operation = engine.find_patch(selected.patch_id).spec.operations[0]
        assert operation.parameters["alpha"] == 6.0 and len(operation.parameters["columns"]) == 4


# ---------------------------------------------------------------------------
# The Q2 scripts' helpers (examples/qwen_loops)
# ---------------------------------------------------------------------------

QWEN_LOOPS = TESTS_DIR.parent / "examples" / "qwen_loops"


def _script(name: str):
    """A module of the Q2 scripts, imported as the scripts import each other (their directory on the path)."""
    import importlib
    import sys

    if str(QWEN_LOOPS) not in sys.path:
        sys.path.insert(0, str(QWEN_LOOPS))
    return importlib.import_module(name)


def test_the_controller_runs_the_strategy_by_dispatch_or_through_mcp_and_the_record_says_which():
    from wse.registry import builtin_registry
    from wse.runner import TEST_DOUBLE_PREFIX

    loop_setup = _script("loop_setup")
    registry = builtin_registry()
    parameters = {"columns": 4, "suite_id": "practice"}
    dispatch = loop_setup.controller(registry, DirectionSearchStrategy, parameters, via_mcp=False, max_steps=7,
                                     wall_clock_seconds=60.0)
    assert (dispatch.kind, dispatch.strategy, dispatch.parameters) == ("strategy", "direction_search@1", parameters)
    assert (dispatch.max_steps, dispatch.wall_clock_seconds) == (7, 60.0)
    mcp = loop_setup.controller(registry, DirectionSearchStrategy, parameters, via_mcp=True, max_steps=20_000,
                                wall_clock_seconds=60.0)
    assert (mcp.kind, mcp.model) == ("agent", f"{TEST_DOUBLE_PREFIX}direction_search_mcp@1")
    assert (mcp.max_turns, mcp.wall_clock_seconds) == (10_000, 60.0)
    # The test double creates its strategy without arguments: the registered subclass applies the parameters.
    bound = registry.resolve("strategy", "direction_search_mcp", "1")
    assert issubclass(bound, DirectionSearchStrategy) and (bound.strategy_id, bound.version) == ("direction_search_mcp",
                                                                                               "1")
    assert bound().config() == DirectionSearchStrategy(**parameters).config()
    assert bound(columns=8).config()["columns"] == 8
    for config, path in ((dispatch, "dispatch"), (mcp, "mcp")):
        assert loop_setup.strategy_record(DirectionSearchStrategy, parameters, config) == {
            "strategy": "direction_search@1", "path": path, "parameters": parameters,
            "controller": "direction_search@1" if path == "dispatch" else mcp.model,
            "resolved_config": DirectionSearchStrategy(**parameters).config()}


def test_step_summaries_collapse_the_polls_of_a_job_on_either_path():
    """Through MCP a model response precedes every call; the polls of one job still collapse into one step."""
    loop_setup = _script("loop_setup")
    numbers = iter(range(1, 100))

    def event(kind: str, **fields: Any) -> dict:
        number = next(numbers)
        return {"seq": number, "created_at": f"t{number}", "kind": kind, **fields}

    def poll() -> list[dict]:
        return [event("tool_call", tool="wse_get_job", arguments={"job_id": "job_1"}, details={"rationale": "Wait."}),
                event("tool_result", tool="wse_get_job", ok=True, error_code=None, summary=None)]

    dispatch = [event("run_started"), *poll(), *poll(), *poll(), event("message", output="the search is complete")]
    mcp = [event("run_started")]
    for _ in range(3):
        mcp += [event("model_response", usage={"input_tokens": 1}), *poll()]
    mcp += [event("model_response", usage={"input_tokens": 1}), event("message", output="Stopping: complete")]
    for events in (dispatch, mcp):
        steps = loop_setup._steps(events)
        assert [step.get("tool") or step.get("event") for step in steps] == ["run_started", "wse_get_job", "message"]
        last_poll = [event for event in events if event["kind"] == "tool_call"][-1]
        assert steps[1]["polls"] == 3 and steps[1]["rationale"] == "Wait." and steps[1]["last_at"] == \
            last_poll["created_at"]


def test_a_step_keeps_the_outcome_text_as_its_summary_not_as_an_error():
    """A successful call's outcome text is not an error: steps record it as ``summary``, beside ``ok``."""
    loop_setup = _script("loop_setup")
    events = [
        {"seq": 1, "created_at": "t1", "kind": "tool_call", "tool": "wse_get_experiment", "arguments": {}},
        {"seq": 2, "created_at": "t2", "kind": "tool_result", "tool": "wse_get_experiment", "ok": True,
         "error_code": None, "summary": "experiment exp_1 is OPEN"},
        {"seq": 3, "created_at": "t3", "kind": "tool_call", "tool": "wse_probe", "arguments": {}},
        {"seq": 4, "created_at": "t4", "kind": "tool_result", "tool": "wse_probe", "ok": False,
         "error_code": "BUDGET_EXCEEDED", "summary": "probe_calls: 0 remain"},
    ]
    succeeded, failed = loop_setup._steps(events)
    assert succeeded["summary"] == "experiment exp_1 is OPEN" and "error" not in succeeded
    assert (failed["ok"], failed["error_code"], failed["summary"]) == (False, "BUDGET_EXCEEDED",
                                                                       "probe_calls: 0 remain")
    assert "error" not in failed


def test_summarizing_a_run_again_keeps_its_record(lm, tmp_path):  # noqa: F811
    """--summarize (both scripts) updates the experiment's summary in the run's record and keeps its other keys."""
    from test_llm_policy_mcp import _open_run

    run_discovery, check = _script("run_discovery"), _script("check_expressiveness")
    run_id, eid = _open_run(lm.workspace, lm.base)
    out = tmp_path / "out"
    finished = {"script": "run_discovery.py", "setup": {"suite_id": "practice"}, "strategy": {"path": "mcp"},
                "controller_config": {"kind": "agent"}, "elapsed_seconds": 1.5, "error": None}
    run_discovery._summarize(out, lm.workspace, eid, finished)
    run_discovery._summarize(out, lm.workspace, eid, {"summarized_at": "later"})
    stored = json.loads((out / "runs" / f"{run_id}.json").read_text())
    assert {key: stored[key] for key in finished} == finished and stored["summarized_at"] == "later"
    assert stored["experiment"]["experiment_id"] == eid and stored["final_message"] is None
    assert json.loads((out / "summary.json").read_text()) == stored
    check.record_run(out, lm.workspace, eid, {"alpha": 6.0, "directions": {"used": "npz"}, "error": None})
    record = check.record_run(out, lm.workspace, eid, {"summarized_at": "later"})
    stored = json.loads((out / f"expressiveness-{run_id}.json").read_text())
    assert stored == record and stored["alpha"] == 6.0 and stored["summarized_at"] == "later"
    assert "verdict" not in stored  # no pass has run
    assert json.loads((out / "expressiveness.json").read_text()) == stored
