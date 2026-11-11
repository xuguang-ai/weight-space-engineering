"""The scripted discovery strategy ``direction_search@1`` (M2 plan Task 12).

Behavioral diagnosis, mechanism investigation, weight intervention, and
evaluation, done through the agent-facing tools alone and with no oracle
value: the layer, the direction, and the columns come from the run's own
generation pass, probe, and analysis, and no layer index, column list,
direction, or evaluation item of a published edit is used. The edit's form
and its size defaults do come from p50c's published edit: ``project_out`` of
a residual-space direction from columns of ``mlp.down`` (the form the M2
plan's Task 12 prescribes), 64 columns, and alphas 1 and 6 (which the plan
does not prescribe; alpha 6 is p50c's own, and alpha 1 removes the
component). A run therefore tests whether the tools can localize an edit of
p50c's shape from the model's behavior. The strategy works for any model
whose family descriptor gives its tensors the role ``mlp.down`` (the MLP's
output projection, ``[hidden, intermediate]``).

1. **Diagnose.** Read the architecture (``wse_inspect_model``, every page of a
   paged inspection), run one generation pass of the base on the practice
   suite (``wse_submit_evaluation`` and ``wse_get_job``), and read its report
   (``wse_get_report``; a report too large for one result, a
   ``ReportOverview``, stops the search). Its outputs form two groups:
   ``a``, the outputs that reached the token limit, and ``b``, those that
   finished. An output that did neither is left out. When
   both groups together hold more outputs than one probe may take
   (``max_texts``, 64 under ``qwen_loop_v1``), each group keeps its first
   outputs in report order, and each at least half of the probe when it has
   them. The strategy stops if either group is empty or the two hold fewer
   than three outputs, the minimum of a contrast.
2. **Investigate.** Probe the outputs (``wse_probe``, by output URI, each
   window the output's last ``last`` tokens, or the whole output within the
   policy's per-text limit when ``last`` is unset) with groups ``a`` and
   ``b``, at the role sites ``{"role": "mlp.down", "layer": i, "point":
   "input"}`` and ``"output"`` of every chosen layer: by default every
   decoder layer of the model's configuration (``num_hidden_layers``, read
   from a nested ``text_config`` for multimodal wrappers). If the probe is
   refused because some of those layers have no ``mlp.down`` role, it is
   repeated once with the layers the refusal lists; if the sites do not fit
   one probe's inline summary, they are split over several probes. The
   **separation** of a layer is the relative contrast at its output site,
   ``|mean_a - mean_b| / ((|mean_a| + |mean_b|) / 2)``: how much of the MLP's
   typical output differs between the groups. Dividing by the group-mean norms
   makes layers comparable, since the output norms of ``mlp.down`` grow with
   depth; for two groups of equal norm it equals ``sqrt(2 (1 - cos))`` of their
   means. The layer with the largest separation is chosen (the lower layer on
   a tie). Its direction is the contrast at its ``mlp.down`` output, one value
   per row of the weight (the residual space, which ``project_out`` on
   ``mlp.down`` needs), passed by reference
   (``<result_uri>#/sites/<i>/contrast/direction``).
3. **Intervene.** Choose ``columns`` input columns of that layer's
   ``mlp.down`` weight, by ``column_rule``:

   * ``cosine`` (default): ``wse_analyze`` ``direction_projection`` of the
     direction against every column (``axis="columns"``), the ``columns``
     columns of largest absolute cosine. A column's weights are what its
     intermediate unit writes into the residual stream; the columns most
     aligned with the direction are those through which the MLP writes it,
     and the cosine does not favour columns of large norm;
   * ``projection``: the same analysis ranked by absolute scalar projection,
     which favours the columns that write the most of the direction per unit
     of activation;
   * ``input_contrast``: the ``columns`` dimensions of largest absolute
     standardized difference at the same layer's ``mlp.down`` input, from a
     second probe of that one site: the intermediate units whose activations
     differ most between the groups, whatever they write.

   For each ``alpha``, preview one ``project_out`` of the direction (by
   reference) from those columns of the whole weight, ``W'[:, C] = W[:, C] -
   alpha d (d^T W[:, C]) / (d^T d)``, and apply it. A preview that changes
   nothing is not applied. Only as many candidates are built as the
   remaining budget can evaluate.
4. **Evaluate and select.** Run a generation pass of each candidate on the
   same practice suite and configuration, and compare its report with the
   base's (``wse_compare``, paired accuracy differences; a refused comparison
   is recorded and does not stop the search). A candidate **improves** on the
   base when its accuracy is at least the base's and its limit-hit rate is
   lower. Among improving candidates the best has the lowest limit-hit rate,
   then the higher accuracy, then the fewer generated tokens per output, then
   the earlier alpha. Each candidate that becomes the best is selected at
   once (``wse_select_checkpoint``, compare-and-swap), so a run that the
   runner stops between candidates keeps its best selection. When no
   candidate improves, the base stays selected. The strategy then stops with
   a reason that states the numbers.

**Budget.** Before the first operation the strategy checks that the
remaining budget covers the counts of its smallest complete plan: two
generation passes (the base and one candidate), the probes, an analysis, a
preview, and a commit, and at least one generated token. The tokens a pass
reserves are not visible before the base pass (a suite does not show its
``max_new_tokens``), so the generated tokens of the candidates' passes are
checked once the base pass has reserved its own: after the probe, only as
many candidates are built as the remaining tokens can evaluate, and before
every later pass it checks that a pass and the tokens the base pass reserved
remain; when they do not, it stops after the candidates evaluated so far. A
short token budget is therefore found only after the base pass and the
probe. Any operation that fails, a budget error included, stops the strategy
with the operation and the error code. The default plan (three passes, one
probe, one analysis, two previews, two commits) fits ``qwen_loop_v1``: 32
problems at 8,192 tokens take about 40 minutes per pass on the operator's
MPS, so the plan takes about 2 hours of its 6.

Every action carries a short observable rationale; idempotency keys derive
from the step number, so a run is reproducible. This module does not import
``transformers``.
"""

from __future__ import annotations

import math
from collections.abc import Generator, Mapping, Sequence
from typing import Any

from wse.schemas import BudgetEntry, ObjectiveSpec
from wse.strategies.base import Action, Observation

__all__ = [
    "COLUMN_RULES",
    "DirectionSearchStrategy",
    "choose_layer",
    "decoder_layer_count",
    "improves",
    "layer_records",
    "output_groups",
    "rank_key",
    "role_tensor_of_site",
    "separation",
    "top_columns",
]

ROLE = "mlp.down"
COLUMN_RULES = ("cosine", "projection", "input_contrast")
# Engine limits the plan depends on (the tests pin them to their sources): the most inline entries of an analysis
# (wse.policy.MAX_TOP_K) and of a probe summary (wse.probes.MAX_SUMMARY_TOP), the most sites one probe reads
# (wse.probes.MAX_PROBE_SITES), and the longest wait of one job poll (wse.jobs.MAX_WAIT_SECONDS).
MAX_INLINE_TOP = 64
MAX_PROBE_SITES = 128
MAX_WAIT_SECONDS = 10.0
ACTIVE_JOB_STATES = ("queued", "running")
# Metrics are means over samples; differences smaller than this are ties.
TOLERANCE = 1e-9
MAX_REASON = 200  # Action.reason
MAX_RATIONALE = 500  # Action.rationale


class _Stop(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# Decisions (pure functions of observations)
# ---------------------------------------------------------------------------


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value: Any) -> float | None:
    """``value`` as a finite float, or ``None`` (summaries write non-finite values as strings)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def decoder_layer_count(manifest: Mapping[str, Any]) -> int | None:
    """The number of decoder layers in a manifest's model configuration, or ``None`` when it does not say.

    The configuration is the snapshot's ``config.json``
    (``manifest["config"]["model_config"]``); ``num_hidden_layers`` (or GPT-2's
    ``n_layer``) is read from it, then from a multimodal wrapper's nested
    ``text_config``.
    """
    config = (manifest.get("config") or {}).get("model_config")
    if not isinstance(config, Mapping):
        return None
    for scope in (config, config.get("text_config")):
        if not isinstance(scope, Mapping):
            continue
        for key in ("num_hidden_layers", "n_layer"):
            value = scope.get(key)
            if _integer(value) and value >= 1:
                return int(value)
    return None


def output_groups(report: Mapping[str, Any], max_texts: int) -> dict[str, Any]:
    """The outputs of a generation report, grouped (module docstring, step 1).

    Returns ``a`` (URIs of outputs that reached the token limit) and ``b``
    (URIs of outputs that finished), each in report order and together at
    most ``max_texts``, and the counts before any cut: ``limit_hit``,
    ``finished``, and ``other`` (outputs that did neither, or have no URI).
    """
    limit_hit: list[str] = []
    finished: list[str] = []
    other = 0
    for problem in report.get("problems") or []:
        for sample in problem.get("samples") or []:
            uri = sample.get("output_uri")
            if not uri:
                other += 1
            elif sample.get("limit_hit"):
                limit_hit.append(uri)
            elif sample.get("finished"):
                finished.append(uri)
            else:
                other += 1
    a, b = limit_hit, finished
    if len(a) + len(b) > max_texts:
        half = max_texts // 2
        keep_a = min(len(a), max(half, max_texts - len(b)))
        a, b = a[:keep_a], b[: max_texts - keep_a]
    return {"a": a, "b": b, "limit_hit": len(limit_hit), "finished": len(finished), "other": other}


def separation(site: Mapping[str, Any]) -> float | None:
    """The relative contrast of a probe site's summary, ``|mean_a - mean_b| / ((|mean_a| + |mean_b|) / 2)``.

    ``None`` when the summary has no contrast or its numbers are not finite.
    """
    norms = site.get("group_mean_norms")
    contrast = _number(site.get("contrast_norm"))
    if not isinstance(norms, Sequence) or len(norms) != 2 or contrast is None:
        return None
    a, b = _number(norms[0]), _number(norms[1])
    if a is None or b is None or a + b <= 0:
        return None
    return contrast / ((a + b) / 2)


def layer_records(summary: Mapping[str, Any], layers: Sequence[int]) -> list[dict[str, Any]]:
    """Per layer, the output and input sites of a probe that read ``mlp.down`` input then output of each layer.

    Each record holds the layer, the probe's ``result_uri``, the sites' indices
    and resolved names, the output site's feature count, its separation, the
    input site's separation, and the output contrast norm.
    """
    sites = summary.get("sites") or []
    if len(sites) != 2 * len(layers):
        raise ValueError(f"the probe summary has {len(sites)} sites, not two for each of {len(layers)} layers")
    records = []
    for position, layer in enumerate(layers):
        inner, outer = sites[2 * position], sites[2 * position + 1]
        records.append({
            "layer": int(layer), "result_uri": summary.get("result_uri"),
            "input_index": 2 * position, "output_index": 2 * position + 1,
            "input_site": inner.get("site"), "output_site": outer.get("site"), "features": outer.get("features"),
            "separation": separation(outer), "input_separation": separation(inner),
            "contrast_norm": _number(outer.get("contrast_norm")),
        })
    return records


def choose_layer(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The record of largest separation (the lower layer on a tie), or ``None`` when no layer has one."""
    scored = [record for record in records if record.get("separation") is not None]
    if not scored:
        return None
    return max(scored, key=lambda record: (record["separation"], -record["layer"]))


def role_tensor_of_site(site: str, manifest: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The two-dimensional weight of the module a probe site names (``<module>:output``), from the manifest.

    A site's module path is the canonical name of its tensors without the last
    component, so the weight is the one two-dimensional tensor named
    ``<module>.<name>`` (``<module>.weight`` preferred).
    """
    module = site.rpartition(":")[0]
    matches = [info for info in manifest.get("tensors") or []
               if info.get("name", "").rpartition(".")[0] == module and len(info.get("shape") or []) == 2]
    preferred = [info for info in matches if info["name"] == f"{module}.weight"]
    if preferred:
        return preferred[0]
    return matches[0] if len(matches) == 1 else None


def top_columns(analysis: Mapping[str, Any], count: int) -> list[int]:
    """The first ``count`` inline entries of a ``direction_projection`` result (by descending |measure|), sorted."""
    entries = (analysis.get("result") or {}).get("top") or []
    return sorted({int(entry["index"]) for entry in entries[:count]})


def improves(base: Mapping[str, float], candidate: Mapping[str, float]) -> bool:
    """Whether a candidate's accuracy is at least the base's and its limit-hit rate is lower."""
    return (candidate["accuracy"] >= base["accuracy"] - TOLERANCE
            and candidate["limit_hit_rate"] < base["limit_hit_rate"] - TOLERANCE)


def rank_key(metrics: Mapping[str, float]) -> tuple[float, float, float]:
    """Order of improving candidates: lower limit-hit rate, then higher accuracy, then fewer generated tokens."""
    def rounded(value: float | None) -> float:
        return math.inf if value is None else round(float(value), 9)

    return (rounded(metrics.get("limit_hit_rate")), -rounded(metrics.get("accuracy")),
            rounded(metrics.get("mean_generated_tokens")))


def _metrics(job: Mapping[str, Any]) -> dict[str, float]:
    metrics = dict(((job.get("result") or {}).get("metrics")) or {})
    missing = [name for name in ("accuracy", "limit_hit_rate") if _number(metrics.get(name)) is None]
    if missing:
        raise _Stop(f"the report of job {job.get('job_id')} has no {', '.join(missing)}; a generation suite is needed")
    return metrics


def _fmt(value: Any) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:.3g}"


def _short(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# The strategy
# ---------------------------------------------------------------------------


class DirectionSearchStrategy:
    """Scripted discovery: diagnose, probe, project a contrast direction out of ``mlp.down``, evaluate, select.

    Parameters (all optional): ``suite_id`` (the practice suite; default the
    task's development suite), ``sample_count`` and ``seed`` (the evaluation
    configuration; default every problem and the suite's first seed),
    ``columns`` (default 64, at most 64), ``alphas`` (default ``[1.0, 6.0]``;
    both defaults follow p50c's published edit, module docstring),
    ``layers`` (default every decoder layer with an ``mlp.down`` role),
    ``last`` (the probe window in output tokens; default the whole output
    within the policy's per-text limit), ``column_rule`` (``cosine``,
    ``projection``, or ``input_contrast``), ``max_texts`` (texts per probe,
    default 64), ``wait_seconds`` (per job poll, default 10), and ``compare``
    (whether to compare each candidate's report with the base's, default
    true). The base checkpoint is the experiment's.
    """

    strategy_id = "direction_search"
    version = "1"

    def __init__(self, suite_id: str | None = None, sample_count: int | None = None, seed: int | None = None,
                 columns: int = 64, alphas: Sequence[float] = (1.0, 6.0), layers: Sequence[int] | None = None,
                 last: int | None = None, column_rule: str = "cosine", max_texts: int = 64,
                 wait_seconds: float = 10.0, compare: bool = True) -> None:
        if suite_id is not None and (not isinstance(suite_id, str) or not suite_id):
            raise ValueError("suite_id must be a suite ID")
        for name, value in (("sample_count", sample_count), ("seed", seed)):
            if value is not None and (not _integer(value) or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer")
        if not _integer(columns) or not 1 <= columns <= MAX_INLINE_TOP:
            raise ValueError(f"columns must be an integer from 1 to {MAX_INLINE_TOP} (an analysis's inline entries)")
        if isinstance(alphas, (str, bytes)) or not isinstance(alphas, Sequence) or not alphas \
                or any(_number(alpha) is None or alpha == 0 for alpha in alphas):
            raise ValueError("alphas must be a nonempty list of finite, nonzero numbers")
        if layers is not None:
            if isinstance(layers, (str, bytes)) or not isinstance(layers, Sequence) or not layers \
                    or any(not _integer(layer) or layer < 0 for layer in layers):
                raise ValueError("layers must be a nonempty list of layer indices")
            if len(set(layers)) != len(layers):
                raise ValueError("layers must not repeat a layer")
        if last is not None and (not _integer(last) or last < 1):
            raise ValueError("last must be a positive number of tokens")
        if column_rule not in COLUMN_RULES:
            raise ValueError(f"column_rule must be one of {list(COLUMN_RULES)}")
        if not _integer(max_texts) or max_texts < 3:
            raise ValueError("max_texts must be an integer of at least 3 (a contrast needs three texts)")
        if _number(wait_seconds) is None or not 0 < wait_seconds <= MAX_WAIT_SECONDS:
            raise ValueError(f"wait_seconds must lie in (0, {MAX_WAIT_SECONDS:g}]")
        self.suite_id = suite_id
        self.sample_count = sample_count
        self.seed = seed
        self.columns = int(columns)
        self.alphas = [float(alpha) for alpha in alphas]
        self.layers = None if layers is None else [int(layer) for layer in layers]
        self.last = None if last is None else int(last)
        self.column_rule = column_rule
        self.max_texts = int(max_texts)
        self.wait_seconds = float(wait_seconds)
        self.compare = bool(compare)
        self._plan: Generator[Action, Observation, None] | None = None
        self._seen = 0
        self._budget: Mapping[str, BudgetEntry] = {}
        self._step = 0

    def config(self) -> dict[str, Any]:
        return {"suite_id": self.suite_id, "sample_count": self.sample_count, "seed": self.seed,
                "columns": self.columns, "alphas": self.alphas, "layers": self.layers, "last": self.last,
                "column_rule": self.column_rule, "max_texts": self.max_texts, "wait_seconds": self.wait_seconds,
                "compare": self.compare}

    # -- Strategy contract -------------------------------------------------------------

    def next_action(self, objective: ObjectiveSpec | None, observations: Sequence[Observation],
                    remaining_budget: Mapping[str, BudgetEntry]) -> Action:
        self._budget = remaining_budget
        try:
            if self._plan is None:
                if not observations:
                    return Action.stop("no experiment description was observed")
                self._seen = len(observations)
                self._plan = self._run(observations[0])
                return next(self._plan)
            new = list(observations[self._seen:])
            self._seen = len(observations)
            if len(new) != 1:
                return Action.stop("expected exactly one new observation per action")
            return self._plan.send(new[0])
        except _Stop as stop:
            return Action.stop(_short(stop.reason, MAX_REASON))
        except StopIteration:
            return Action.stop("the search is complete")

    # -- helpers -----------------------------------------------------------------------

    def _key(self, label: str) -> str:
        self._step += 1
        return f"direction-search-{self._step:04d}-{label}"

    @staticmethod
    def _call(tool: str, rationale: str, **arguments: Any) -> Action:
        return Action.call(tool, _short(rationale, MAX_RATIONALE), **arguments)

    def _remaining(self, counter: str) -> float:
        entry = self._budget.get(counter)
        if entry is None:
            return math.inf
        return entry.remaining if isinstance(entry, BudgetEntry) else entry["remaining"]

    def _shortfall(self, **amounts: float) -> str | None:
        for counter, amount in amounts.items():
            if self._remaining(counter) < amount:
                return f"{counter} has {self._remaining(counter):g} left and {amount:g} is needed"
        return None

    def _require(self, what: str, **amounts: float) -> None:
        shortfall = self._shortfall(**amounts)
        if shortfall is not None:
            raise _Stop(f"insufficient budget for {what}: {shortfall}")

    @staticmethod
    def _ok(observation: Observation, what: str) -> Any:
        if not observation.ok:
            error = observation.error or {}
            raise _Stop(f"{what} failed with {error.get('code', 'ERROR')}: {error.get('message', '')}")
        return observation.result

    def _wait(self, eid: str, job: Mapping[str, Any], what: str) -> Generator[Action, Observation, dict[str, Any]]:
        while job["state"] in ACTIVE_JOB_STATES:
            job = self._ok((yield self._call("wse_get_job", f"Wait for {what}.", experiment_id=eid,
                                             job_id=job["job_id"], wait_seconds=self.wait_seconds)), "a job poll")
        if job["state"] != "succeeded":
            error = job.get("error") or {}
            raise _Stop(f"{what} (job {job['job_id']}) ended {job['state']}: {error.get('code', '')} "
                        f"{error.get('message', '')}".strip())
        return dict(job)

    def _evaluate(self, eid: str, checkpoint_id: str, suite_id: str, config: dict[str, int], what: str
                  ) -> Generator[Action, Observation, dict[str, Any]]:
        job = self._ok((yield self._call(
            "wse_submit_evaluation", f"Generate on the {config['sample_count']} practice problems with {what}, to "
            "measure accuracy and how many outputs reach the token limit.", experiment_id=eid,
            checkpoint_id=checkpoint_id, suite_id=suite_id, eval_config=config, idempotency_key=self._key("pass"))),
            f"the generation pass of {what}")
        return (yield from self._wait(eid, job, f"the generation pass of {what}"))

    def _probe(self, eid: str, base: str, texts: list[dict[str, Any]], groups: dict[str, list[int]],
               sites: list[dict[str, Any]], rationale: str, top: int | None = None
               ) -> Generator[Action, Observation, Observation]:
        arguments: dict[str, Any] = {"experiment_id": eid, "checkpoint_id": base, "texts": texts, "sites": sites,
                                     "groups": groups, "idempotency_key": self._key("probe")}
        if top is not None:
            arguments["top"] = top
        return (yield self._call("wse_probe", rationale, **arguments))

    # -- the steps ---------------------------------------------------------------------

    def _suite(self, description: Mapping[str, Any]) -> tuple[str, dict[str, int]]:
        suites = description.get("suites") or []
        task = description.get("task") or {}
        suite_id = self.suite_id or task.get("development_suite_id") or (suites[0]["suite_id"] if suites else None)
        suite = next((s for s in suites if s["suite_id"] == suite_id), None)
        if suite is None:
            raise _Stop(f"the practice suite {suite_id} is not a development suite of this experiment")
        count = self.sample_count if self.sample_count is not None else max(suite["sample_counts"])
        seed = self.seed if self.seed is not None else suite["seeds"][0]
        if count not in suite["sample_counts"] or seed not in suite["seeds"]:
            raise _Stop(f"suite {suite_id} preregisters sample counts {suite['sample_counts']} and seeds "
                        f"{suite['seeds']}, not {count} and {seed}")
        return suite_id, {"sample_count": count, "seed": seed}

    def _inspect(self, eid: str, base: str) -> Generator[Action, Observation, dict[str, Any]]:
        """The base's manifest with every tensor: a paged inspection is read to its last page."""
        tensors: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            arguments: dict[str, Any] = {"experiment_id": eid, "checkpoint_id": base}
            page_note = ""
            if cursor is not None:
                arguments["cursor"] = cursor
                page_note = " (next page)"
            inspection = self._ok((yield self._call(
                "wse_inspect_model", f"Read the architecture: the decoder layers and their mlp.down weights"
                f"{page_note}.", **arguments)), "the inspection")
            manifest = inspection["manifest"]
            tensors += manifest.get("tensors") or []
            cursor = (inspection.get("tensor_page") or {}).get("next_cursor")
            if not cursor:
                return {**manifest, "tensors": tensors}

    def _investigate(self, eid: str, base: str, texts: list[dict[str, Any]], groups: dict[str, list[int]],
                     layers: list[int], counts: str) -> Generator[Action, Observation, list[dict[str, Any]]]:
        """Probe ``mlp.down`` input and output of ``layers``; return one record per layer (``layer_records``)."""
        per_probe = MAX_PROBE_SITES // 2
        retried_layers = False
        records: list[dict[str, Any]] = []
        position = 0
        while position < len(layers):
            chunk = layers[position:position + per_probe]
            if records:
                self._require("another probe", probe_calls=1)
            sites = [{"role": ROLE, "layer": layer, "point": point} for layer in chunk for point in ("input", "output")]
            span = f"layer {chunk[0]}" if len(chunk) == 1 else f"layers {chunk[0]}-{chunk[-1]}"
            observation = yield from self._probe(
                eid, base, texts, groups, sites,
                f"Contrast {counts} at the input and output of mlp.down in {span}, to find the layer whose MLP output "
                "separates them most.")
            error = observation.error or {}
            details = error.get("details") or {}
            listed = details.get("layers")
            if not observation.ok and observation.error_code == "NOT_FOUND" and details.get("role") == ROLE \
                    and self.layers is None and not retried_layers and isinstance(listed, list) and listed \
                    and all(_integer(layer) for layer in listed):
                retried_layers = True  # some configured layers have no mlp.down: probe the layers that do
                layers, records, position = sorted(set(listed)), [], 0
                continue
            fitting = details.get("max_sites")
            if not observation.ok and observation.error_code == "INVALID_ARGUMENT" and _integer(fitting) \
                    and 2 <= fitting < len(sites):
                per_probe = fitting // 2  # the inline summary holds fewer sites: split the layers over more probes
                continue
            job = self._ok(observation, "the probe")
            job = yield from self._wait(eid, job, f"the probe of {span}")
            try:
                records += layer_records(job["result"], chunk)
            except ValueError as exc:
                raise _Stop(str(exc)) from None
            position += len(chunk)
        return records

    def _columns(self, eid: str, base: str, chosen: Mapping[str, Any], tensor: Mapping[str, Any], reference: str,
                 texts: list[dict[str, Any]], groups: dict[str, list[int]], rationale: str
                 ) -> Generator[Action, Observation, tuple[list[int], dict[str, Any]]]:
        """The edited columns by ``column_rule``, and evidence about them."""
        layer, width = chosen["layer"], tensor["shape"][1]
        count = min(self.columns, width)
        if self.column_rule == "input_contrast":
            self._require("the input probe", probe_calls=1)
            observation = yield from self._probe(
                eid, base, texts, groups, [{"role": ROLE, "layer": layer, "point": "input"}],
                f"Find the {count} intermediate units of layer {layer} whose mlp.down inputs differ most between "
                "the groups (standardized differences).", top=min(count, MAX_INLINE_TOP))
            job = self._ok(observation, "the input probe")
            job = yield from self._wait(eid, job, f"the input probe of layer {layer}")
            top = job["result"]["sites"][0].get("top") or []
            columns = sorted({int(dimension) for dimension, _ in top[:count]})
            return columns, {"rule": "input_contrast", "top": [list(entry) for entry in top[:count]]}
        selection = [{"start": 0, "stop": tensor["shape"][0], "step": 1}, {"start": 0, "stop": width, "step": 1}]
        analysis = self._ok((yield self._call(
            "wse_analyze", rationale, experiment_id=eid, checkpoint_id=base, kind="direction_projection",
            tensor=tensor["name"], selection=selection,
            parameters={"direction": reference, "axis": "columns", "measure": self.column_rule, "top": count})),
            "the direction projection")
        columns = top_columns(analysis, count)
        entries = (analysis.get("result") or {}).get("top") or []
        measure = [abs(float(entry[self.column_rule])) for entry in entries[:count]]
        return columns, {"rule": self.column_rule, "result_uri": analysis.get("result_uri"),
                         "largest": max(measure, default=None), "smallest": min(measure, default=None)}

    def _run(self, first: Observation) -> Generator[Action, Observation, None]:
        description = self._ok(first, "reading the experiment")
        experiment = description["experiment"]
        eid, base = experiment["experiment_id"], experiment["base_checkpoint_id"]
        selected = experiment["selected_checkpoint_id"]
        if experiment["mode"] != "bounded_eval" or not description.get("evaluation_available"):
            raise _Stop("development evaluation is not available, so no candidate can be compared with the base")
        operations = description.get("operations") or {}
        if "project_out" not in (operations.get("edits") or {}):
            raise _Stop("the policy permits no project_out edit")
        analysis_needed = self.column_rule != "input_contrast"
        if analysis_needed and "direction_projection" not in (operations.get("analyses") or {}):
            raise _Stop("the policy permits no direction_projection analysis")
        suite_id, config = self._suite(description)
        self._require("the smallest plan (a base pass, a probe, an edit, and a candidate pass)", generation_passes=2,
                      generated_tokens=1, probe_calls=1 + (not analysis_needed), analysis_calls=int(analysis_needed),
                      patch_previews=1, candidate_commits=1)

        manifest = yield from self._inspect(eid, base)
        if (manifest.get("config") or {}).get("family") is None:
            raise _Stop("the model has no family descriptor, so its tensors have no mlp.down role to probe")
        layers = self.layers
        if layers is None:
            count = decoder_layer_count(manifest)
            if count is None:
                raise _Stop("the model configuration does not give its decoder layers; pass layers explicitly")
            layers = list(range(count))

        # 1. Diagnose: which outputs reach the token limit, and which finish.
        job = yield from self._evaluate(eid, base, suite_id, config, "the base")
        base_metrics = _metrics(job)
        pass_tokens = (job.get("reservation") or {}).get("requested", {}).get("generated_tokens", 1)
        report = self._ok((yield self._call(
            "wse_get_report", "Read every output's grade, length, and whether it reached the token limit.",
            experiment_id=eid, report_id=job["report_id"])), "reading the base report")
        if "problems" in (report.get("omitted") or []):
            raise _Stop(f"the base report ({report.get('report_bytes')} bytes) exceeds the {report.get('limit_bytes')}"
                        "-byte inline limit, so its outputs cannot be read; use fewer problems")
        base_report_id = job["report_id"]
        grouped = output_groups(report, self.max_texts)
        a, b = grouped["a"], grouped["b"]
        if not a or not b or len(a) + len(b) < 3:
            raise _Stop(f"no contrast: {grouped['limit_hit']} outputs reached the token limit and "
                        f"{grouped['finished']} finished; a contrast needs both groups and three outputs")
        window = {} if self.last is None else {"last": self.last}
        texts = [{"output_uri": uri, **window} for uri in a + b]
        groups = {"a": list(range(len(a))), "b": list(range(len(a), len(a) + len(b)))}
        counts = f"the {len(a)} outputs that reached the limit with the {len(b)} that finished"

        # 2. Investigate: the layer whose mlp.down output separates the groups most.
        records = yield from self._investigate(eid, base, texts, groups, layers, counts)
        chosen = choose_layer(records)
        if chosen is None:
            raise _Stop("no probed layer has a finite contrast between the groups")
        if chosen["contrast_norm"] is None or chosen["contrast_norm"] <= 0:
            raise _Stop(f"layer {chosen['layer']}'s mlp.down output does not differ between the groups")
        tensor = role_tensor_of_site(chosen["output_site"] or "", manifest)
        if tensor is None or tensor.get("axes") != ["out_features", "in_features"] \
                or tensor["shape"][0] != chosen["features"]:
            raise _Stop(f"site {chosen['output_site']} has no [out_features, in_features] weight with one row per "
                        "feature of the probed output")
        reference = f"{chosen['result_uri']}#/sites/{chosen['output_index']}/contrast/direction"
        ranked = sorted((r for r in records if r["separation"] is not None), key=lambda r: -r["separation"])
        runner_up = next((r for r in ranked if r["layer"] != chosen["layer"]), None)
        evidence = (f"Layer {chosen['layer']}'s mlp.down output separates the groups most (relative contrast "
                    f"{_fmt(chosen['separation'])}" + ("" if runner_up is None else
                                                       f"; next layer {runner_up['layer']}: "
                                                       f"{_fmt(runner_up['separation'])}") + ").")

        # 3. Intervene: project the direction out of the chosen columns, once per affordable alpha.
        covered = {"patch_previews": self._remaining("patch_previews"),
                   "candidate_commits": self._remaining("candidate_commits"),
                   "generation_passes": self._remaining("generation_passes"),
                   "generated_tokens": self._remaining("generated_tokens") // max(pass_tokens, 1)}
        affordable = int(max(min(len(self.alphas), *covered.values()), 0))
        if affordable < 1:
            binding = min(covered, key=lambda counter: covered[counter])
            raise _Stop(f"insufficient budget for any candidate: {binding} has {self._remaining(binding):g} left "
                        f"(a pass reserves {pass_tokens} generated tokens)")
        alphas = self.alphas[:affordable]
        columns, column_evidence = yield from self._columns(
            eid, base, chosen, tensor, reference, texts, groups,
            f"{evidence} Find the columns of its weight most aligned with that direction.")
        if not columns:
            raise _Stop("no column was chosen")
        if column_evidence.get("largest") is not None:
            chosen_by = (f"|{self.column_rule}| {_fmt(column_evidence['smallest'])} to "
                         f"{_fmt(column_evidence['largest'])}")
        else:
            chosen_by = "the largest standardized input differences"
        budget_note = "" if affordable == len(self.alphas) else \
            f" The budget covers {affordable} of the {len(self.alphas)} alphas."
        selection = [{"start": 0, "stop": tensor["shape"][0], "step": 1},
                     {"start": 0, "stop": tensor["shape"][1], "step": 1}]
        candidates: list[dict[str, Any]] = []
        for alpha in alphas:
            hypothesis = (f"Outputs that reach the token limit differ from finished ones along the contrast at layer "
                          f"{chosen['layer']}'s mlp.down output (relative contrast {_fmt(chosen['separation'])}); "
                          f"projecting it out of the {len(columns)} columns chosen by {self.column_rule} with alpha "
                          f"{alpha:g} may let more outputs finish.")
            rationale = f"{hypothesis} Chosen columns: {chosen_by}.{budget_note}"
            preview = self._ok((yield self._call(
                "wse_preview_patch", rationale, experiment_id=eid, idempotency_key=self._key("preview"),
                patch_spec={"base_checkpoint_id": base, "hypothesis": hypothesis,
                            "expected_behavior": "The development limit-hit rate falls and accuracy does not.",
                            "development_suite_id": suite_id,
                            "operations": [{"op": "project_out", "tensor": tensor["name"], "selection": selection,
                                            "direction": reference, "columns": columns, "alpha": alpha}]})),
                f"the preview with alpha {alpha:g}")
            if preview["changed_elements"] == 0:
                continue
            applied = self._ok((yield self._call(
                "wse_apply_patch", f"Commit the previewed candidate with alpha {alpha:g} "
                f"({preview['changed_elements']} changed weights).", experiment_id=eid, patch_id=preview["patch_id"],
                idempotency_key=self._key("apply"))), f"committing alpha {alpha:g}")
            candidates.append({"alpha": alpha, "checkpoint_id": applied["checkpoint_id"],
                               "patch_id": preview["patch_id"], "changed_elements": preview["changed_elements"]})
        if not candidates:
            raise _Stop("no alpha changed any weight of the chosen columns")

        # 4. Evaluate and select.
        best: dict[str, Any] | None = None
        evaluated: list[dict[str, Any]] = []
        ended = ""
        for candidate in candidates:
            shortfall = self._shortfall(generation_passes=1, generated_tokens=pass_tokens)
            if shortfall is not None:
                ended = f"budget ended after {len(evaluated)} of {len(candidates)} candidates ({shortfall}); "
                break
            label = f"candidate alpha {candidate['alpha']:g}"
            job = yield from self._evaluate(eid, candidate["checkpoint_id"], suite_id, config, label)
            metrics = _metrics(job)
            evaluated.append({**candidate, "metrics": metrics})
            if self.compare:
                yield self._call("wse_compare", f"Pair {label}'s grades with the base's on the same problems.",
                                 experiment_id=eid, base_report_id=base_report_id,
                                 candidate_report_id=job["report_id"])
            if improves(base_metrics, metrics) and (best is None or rank_key(metrics) < rank_key(best["metrics"])):
                result = self._ok((yield self._call(
                    "wse_select_checkpoint",
                    f"{label} lowers the limit-hit rate from {_fmt(base_metrics['limit_hit_rate'])} to "
                    f"{_fmt(metrics['limit_hit_rate'])} at accuracy {_fmt(metrics['accuracy'])} (base "
                    f"{_fmt(base_metrics['accuracy'])}); select it.", experiment_id=eid,
                    checkpoint_id=candidate["checkpoint_id"], expected_selected_id=selected,
                    idempotency_key=self._key("select"))), f"selecting {label}")
                selected = result["selected_checkpoint_id"]
                best = evaluated[-1]
        scores = ", ".join([f"base {_fmt(base_metrics['limit_hit_rate'])}/{_fmt(base_metrics['accuracy'])}"] + [
            f"a{c['alpha']:g} {_fmt(c['metrics']['limit_hit_rate'])}/{_fmt(c['metrics']['accuracy'])}"
            for c in evaluated])
        if best is None:
            raise _Stop(f"{ended}kept the base: no candidate lowered the limit-hit rate without lower accuracy "
                        f"(limit-hit/accuracy: {scores}) at layer {chosen['layer']}")
        raise _Stop(f"{ended}selected alpha {best['alpha']:g} at layer {chosen['layer']} (limit-hit/accuracy: "
                    f"{scores})")
