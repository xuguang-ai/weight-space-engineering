"""``Model``: one model version to inspect, analyze, edit, and evaluate; ``Edit``: a change that makes another.

    from wse import Model, compare_models

    original = Model.from_pretrained("Qwen/Qwen3.5-0.8B", workspace="runs/qwen")
    print(original.inspect())
    edit = original.edit("layers.15.self_attn.o_proj[:, 1792:2048]", "fill_slice", value=0.0)
    assert edit.verify()
    edited = edit.save().model
    print(compare_models(base=original, candidate=edited, data="data/math_dev.jsonl", evaluator="math",
                         max_new_tokens=8192).table())

A ``Model`` is one checkpoint: ``checkpoint_id`` (``None`` for an edit kept
in memory), ``weights_hash``, and ``source`` (the snapshot's repository and
the commit its revision resolved to). Its methods serve the steps of an
experiment:

* **Inspect:** ``inspect`` (identity, architecture, parameters, tensor
  groups, and protected tensors), ``tensors`` (names, shapes, dtypes),
  ``tensor`` (a copy: changing it changes nothing in the model), ``roles``.
* **Diagnose:** ``generate`` (text, generated tokens, and finish reason) and
  ``evaluate`` (``evaluator="math"``: accuracy, limit-hit rate, and mean
  generated tokens, with intervals and every sample's output).
* **Investigate:** ``probe`` (activation means at named sites; a probe of an
  evaluation keeps each text's sample: item, grade, limit hit, and text).
* **Intervene:** ``edit`` applies a registered edit operator, and
  ``replace_weights`` submits tensors computed any other way, after checking
  their names, shapes, dtypes, finiteness, and the protected tensors; both
  return an ``Edit`` whose ``model`` is the new version, kept in memory until
  ``Edit.save`` stores it with its exact patch. ``apply_patch`` imports a
  published or exported edit. The original model never changes.
* **Evaluate:** ``compare_models`` (or ``Model.compare``) gives paired
  differences on the same problems, prompts, sampling settings, and seeds.
* **Deliver:** ``Model.export`` writes a Hugging Face model directory, which
  ``Model.from_pretrained`` imports again with the same weights hash, and
  ``Edit.export`` writes the exact patch with its provenance.

The ``limit_hit_rate`` metric is the fraction of outputs that reached the
token limit without finishing. It is not a repetition rate: an output can
repeat itself and still finish, or reach the limit without repeating.

Underneath is the low-level API. ``Model.engine`` is the operator ``Engine``
of the workspace and ``Model.handle`` the ``ModelHandle``. Every edit is an
exact patch with before and after hashes, which ``Edit.verify`` replays.
Short tensor names and NumPy-style indices are resolved here, so patches
always record full names and explicit ranges. Replaying weights exactly does
not make generated text identical on other hardware.

Version-less operator names resolve to the versions pinned in ``OPERATORS``,
never to the latest registered version.

A ``Model`` is for direct use. Its work is unmanaged: it belongs to no
experiment and no budget. Imports (``from_pretrained``, ``apply_patch``) and
saved edits (``Edit.save``) are recorded in the workspace's audit log; an
edit kept in memory is not. Agents work through managed experiments
(``wse.Experiment``, or the MCP server, ``wse serve``).

**Downloads.** ``from_pretrained`` with a Hub repository ID resolves the
revision (default ``main``) to a commit, which ``source.revision`` records,
and lists the repository's files. It then downloads only the allow-listed
files at the snapshot root (``wse.hf_snapshot``) into the Hugging Face
cache. Gated repositories use the user's own Hugging Face login; WSE never
reads or stores a token.
"""

from __future__ import annotations

import difflib
import fnmatch
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from wse.api import Engine, EvaluationResult, ModelHandle, apply_operator
from wse.schemas import (
    ErrorCode,
    EvalConfig,
    GenerationComparison,
    MaterializedPatch,
    ModelSource,
    TensorDiff,
    TextItem,
    WSEError,
)

__all__ = ["EVALUATORS", "GRADERS", "METRICS", "OPERATORS", "WORKSPACE_ENV", "Comparison", "Edit", "Evaluation",
           "Inspection", "Model", "Probe", "ProbeSample", "TensorEntry", "compare_models",
           "default_workspace"]

WORKSPACE_ENV = "WSE_WORKSPACE"
DEFAULT_WORKSPACE = "workspace"
# Version-less edit operator names and the exact versions they mean here.
OPERATORS = MappingProxyType({
    "scale_slice": "scale_slice@1",
    "set_slice": "set_slice@1",
    "fill_slice": "fill_slice@1",
    "project_out": "project_out@1",
    "add_low_rank": "add_low_rank@1",
})
_HUB_CACHE_DIR = re.compile(r"^models--(?:(?P<org>[A-Za-z0-9][A-Za-z0-9._-]*)--)?(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)$")
# The default grader of each dataset format.
GRADERS = MappingProxyType({"math500": "math_boxed@1", "math": "math_boxed@1", "gsm8k": "gsm8k_numeric@1"})
# Evaluators by name: the built-in math protocol reads GSM8K or MATH rows (wse.evaluators.text_suites.detect_format).
EVALUATORS = ("math",)
# The metrics a generation evaluation reports.
METRICS = ("accuracy", "limit_hit_rate", "mean_generated_tokens")
EXPORT_PROVENANCE = "wse_provenance.json"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPO_ID = re.compile(r"^(?:[A-Za-z0-9][A-Za-z0-9._-]{0,95}/)?[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_TARGET = re.compile(r"^\s*(?P<name>[^\[\]\s]+)\s*(?:\[(?P<index>[^\[\]]*)\])?\s*$")


def default_workspace() -> Path:
    """The workspace of the CLI and of ``WSE``: ``$WSE_WORKSPACE``, or ``./workspace``."""
    return Path(os.environ.get(WORKSPACE_ENV, DEFAULT_WORKSPACE))


def _invalid(message: str, **details: Any) -> WSEError:
    return WSEError(ErrorCode.INVALID_ARGUMENT, message, details=details)


def _policy() -> Any:
    from wse.policy import default_operation_policy

    policy = default_operation_policy()
    for name, ref in OPERATORS.items():
        if name not in policy.edits:
            policy = policy.with_edit(name, ref)
    return policy


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def _cache_labels(path: Path) -> tuple[str, str] | None:
    """The repository and commit of a Hugging Face cache snapshot (``models--ORG--NAME/snapshots/SHA``)."""
    parts = path.absolute().parts
    if len(parts) >= 3 and parts[-2] == "snapshots" and _COMMIT.fullmatch(parts[-1]):
        match = _HUB_CACHE_DIR.fullmatch(parts[-3])
        if match:
            name = match["name"] if match["org"] is None else f"{match['org']}/{match['name']}"
            return name, parts[-1]
    return None


def _local_labels(path: Path, repo: str | None, revision: str | None) -> tuple[str, str]:
    inferred = _cache_labels(path)
    if inferred is not None:
        for label, given, value in (("repo", repo, inferred[0]), ("revision", revision, inferred[1])):
            if given is not None and given != value:
                raise _invalid(f"{label}={given!r} contradicts the Hugging Face cache path, which names {value!r}",
                               **{label: given, "inferred": value})
        return inferred
    if repo is None or revision is None:
        raise _invalid(
            f"the local snapshot {path} needs repo= and revision= (the repository ID and the commit SHA it was "
            "downloaded from), which WSE records as its provenance; a Hugging Face cache snapshot "
            "(models--ORG--NAME/snapshots/SHA) names both itself", snapshot=str(path))
    return repo, revision


def _download(repo_id: str, revision: str | None) -> tuple[Path, str, str]:
    """Download the allow-listed root files of ``repo_id`` at ``revision``; return the snapshot, repo, and commit."""
    try:
        import huggingface_hub
    except ImportError as exc:  # pragma: no cover - installed with the models-hf extra
        raise WSEError(ErrorCode.UNSUPPORTED_CAPABILITY, "downloading from the Hugging Face Hub needs the models-hf "
                       "extra; install it with 'uv sync --extra models-hf'") from exc
    from wse.hf_snapshot import is_allowed

    try:
        api = huggingface_hub.HfApi()
        commit = api.model_info(repo_id, revision=revision).sha
        files = sorted(name for name in api.list_repo_files(repo_id, revision=commit) if is_allowed(name))
        if not any(name.endswith(".safetensors") for name in files):
            raise WSEError(ErrorCode.NOT_FOUND, f"{repo_id}@{commit} has no safetensors weights at its root; WSE "
                           "imports safetensors only", details={"repo": repo_id, "revision": commit})
        snapshot = huggingface_hub.snapshot_download(repo_id, revision=commit, allow_patterns=files)
    except WSEError:
        raise
    except Exception as exc:  # network, authorization, and missing-repository errors of huggingface_hub
        raise WSEError(ErrorCode.NOT_FOUND, f"cannot download {repo_id} at revision {revision or 'main'}: {exc}",
                       details={"repo": repo_id, "revision": revision}) from exc
    return Path(snapshot), repo_id, commit


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def _index(entry: str, size: int, axis: int, target: str) -> dict[str, int]:
    """One axis of a NumPy-style index, as an explicit ``{start, stop, step}`` range with step 1."""
    text = entry.strip()

    def integer(value: str) -> int | None:
        value = value.strip()
        if value == "":
            return None
        if not re.fullmatch(r"[+-]?[0-9]+", value):
            raise _invalid(f"{target}: index {entry.strip()!r} on axis {axis} is not an integer or a slice",
                           target=target, axis=axis)
        return int(value)

    if ":" in text:
        parts = text.split(":")
        if len(parts) > 3:
            raise _invalid(f"{target}: index {text!r} on axis {axis} is not a slice", target=target, axis=axis)
        start, stop = integer(parts[0]), integer(parts[1])
        step = integer(parts[2]) if len(parts) == 3 else None
        if step not in (None, 1):
            raise _invalid(f"{target}: axis {axis} has step {step}; selections take step 1", target=target, axis=axis)
        first, last, _ = slice(start, stop).indices(size)
    else:
        position = integer(text)
        if position is None:
            raise _invalid(f"{target}: axis {axis} has an empty index", target=target, axis=axis)
        if not -size <= position < size:
            raise _invalid(f"{target}: index {position} is outside axis {axis} of size {size}", target=target,
                           axis=axis, size=size)
        first = position % size
        last = first + 1
    if last <= first:
        raise _invalid(f"{target}: the range on axis {axis} is empty", target=target, axis=axis)
    return {"start": first, "stop": last, "step": 1}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    """``value`` with NumPy arrays, tensors, and NumPy scalars as Python lists and numbers (operator parameters)."""
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _short_hash(value: str) -> str:
    return value.removeprefix("sha256:")[:12]


def _wse_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("wse")
    except PackageNotFoundError:  # pragma: no cover - a source tree that is not installed
        return "unknown"


def _new_directory(path: str | Path) -> Path:
    """``path`` as a new or empty directory, created; an existing non-empty directory is refused."""
    directory = Path(path).expanduser()
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise _invalid(f"{directory} exists and is not an empty directory; choose a new one", path=str(directory))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _evaluation_format(data: str | Path, evaluator: str, format: str | None) -> str:
    """The data format of an evaluation: ``format`` as given, or the one ``evaluator`` reads from ``data``."""
    from wse.evaluators.text_suites import FORMATS, detect_format

    if evaluator not in EVALUATORS:
        raise _invalid(f"unknown evaluator {evaluator!r}; the built-in evaluators are {list(EVALUATORS)}",
                       evaluator=str(evaluator))
    if format is not None:
        if format not in FORMATS:
            raise _invalid(f"unknown data format {format!r}; formats are {list(FORMATS)}", format=str(format))
        return format
    return detect_format(data)


def _metric_names(metrics: Sequence[str] | None) -> tuple[str, ...]:
    if metrics is None:
        return METRICS
    if isinstance(metrics, str) or not metrics or any(name not in METRICS for name in metrics):
        raise _invalid(f"metrics must be a non-empty list of {list(METRICS)}", metrics=str(metrics))
    return tuple(dict.fromkeys(metrics))


@dataclass(frozen=True)
class TensorEntry:
    """One tensor of a model: its canonical name, shape, dtype, and whether every edit of it is refused."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    protected: bool

    def __str__(self) -> str:
        return f"{self.name} {list(self.shape)} {self.dtype}" + (" (protected)" if self.protected else "")


@dataclass(frozen=True)
class Inspection:
    """What ``Model.inspect`` reports; ``str()`` gives a readable summary.

    ``groups`` collapses layer indices (``layers.{i}``) into one line per
    tensor of a layer: the pattern, how many tensors match it, its shape and
    dtype, and its parameters in all. ``protected`` lists the tensors every
    edit refuses (the family's protected roles), and ``aliases`` the groups of
    names that share one parameter.
    """

    checkpoint_id: str | None
    weights_hash: str
    source: str | None
    adapter: str
    family: str | None
    model_type: str | None
    layers: int | None
    parameter_count: int
    tensor_count: int
    dtypes: Mapping[str, int]
    groups: tuple[tuple[str, int, tuple[int, ...], str, int], ...]
    protected: tuple[str, ...]
    aliases: tuple[tuple[str, ...], ...]
    dropped_tensors: int

    def __str__(self) -> str:
        where = f"checkpoint {self.checkpoint_id}" if self.checkpoint_id else "in memory"
        lines = [f"model {self.source or self.adapter}, {where}, weights {_short_hash(self.weights_hash)}",
                 f"adapter {self.adapter}, family {self.family or 'none (Level 0)'}, model type {self.model_type}, "
                 f"{self.layers if self.layers is not None else '?'} layers",
                 f"{self.parameter_count:,} parameters in {self.tensor_count} tensors ("
                 + ", ".join(f"{count} {dtype}" for dtype, count in sorted(self.dtypes.items())) + ")"]
        if self.dropped_tensors:
            lines.append(f"{self.dropped_tensors} tensors of the snapshot were not imported (outside the language "
                         "model)")
        lines.append("tensors:")
        for pattern, count, shape, dtype, parameters in self.groups:
            lines.append(f"  {pattern}  x{count}  {list(shape)} {dtype}  {parameters:,} parameters")
        lines.append("protected (every edit is refused): " + (", ".join(self.protected) or "none"))
        if self.aliases:
            lines.append("shared parameters: " + "; ".join(" = ".join(group) for group in self.aliases))
        return "\n".join(lines)


@dataclass(eq=False)
class Edit:
    """An edit of ``base`` into ``model``: the exact patch, its per-tensor diff, and its binary blocks.

    ``model`` is the new version, usable before it is saved (``generate``,
    ``evaluate``, ``probe``). ``patch_id`` names the stored patch in the
    workspace (``apply_patch``, or ``save``, which fills it in together with
    ``model.checkpoint_id``); an edit kept in memory has neither.
    """

    base: Model
    model: Model
    patch: MaterializedPatch
    diff: tuple[TensorDiff, ...]
    blocks: Mapping[str, bytes] = field(default_factory=dict, repr=False)
    patch_id: str | None = None

    @property
    def changed(self) -> tuple[TensorDiff, ...]:
        """The diffs of the tensors the edit changed."""
        return tuple(d for d in self.diff if d.changed_elements)

    @property
    def changed_elements(self) -> int:
        return sum(d.changed_elements for d in self.diff)

    def verify(self) -> bool:
        """Replay the patch on the base without running any operator; true when it gives ``model``'s weights.

        The replay checks every step's before and after hashes and raises if
        one does not match.
        """
        from wse.patches import replay_patch

        return replay_patch(self.base.handle, self.patch, blocks=self.blocks).weights_hash == self.model.weights_hash

    def save(self, label: str = "direct") -> Edit:
        """Store the edited model in the workspace with its exact patch; return this edit, now stored.

        ``Engine.store_patch`` replays the patch on the stored base, which
        checks every hash, then commits the candidate with its patch file and
        an audit event. A stored model can be reopened (``Model.open``) and
        compared on record (``compare(..., out_dir=...)``). The base must be
        stored: for an edit of an edit, save the earlier edit first. Saving
        fills in ``patch_id`` and ``model.checkpoint_id``, so
        ``edit.save().model`` is the stored version; saving the same edit again
        changes nothing, and ``label`` names it in the record.
        """
        if self.patch_id is not None and self.model.checkpoint_id is not None:
            return self
        if self.patch.result_weights_hash != self.model.weights_hash:
            raise _invalid("this edit's model is not its patch's result, so it is not stored; make edits with "
                           "Model.edit, Model.replace_weights, or Model.apply_patch")
        base_id = self.base.checkpoint_id or self.base._stored_checkpoint_id()
        if base_id is None:
            raise _invalid("this edit's base is an edit kept in memory, so it is not stored; save the edit that "
                           "made the base first")
        record = self.base.engine.store_patch(self.patch, self.blocks, base_checkpoint_id=base_id, label=label)
        self.model.checkpoint_id = record.checkpoint_id
        self.patch_id = record.patch_id
        return self

    def export(self, path: str | Path) -> Path:
        """Write the exact patch and its provenance into the directory ``path`` (new or empty); return it.

        ``patch.wsepatch`` is the binary patch file: the materialized steps and
        the blocks of their after-values. ``patch.json`` is its provenance:
        the base (checkpoint, weights hash, snapshot source), the result's
        weights hash, each step's operator, tensor, selection, and changed
        elements, the stored IDs when the edit was saved, and the patch file's
        SHA-256. ``Model.apply_patch(path, format="wse-patch-v1")`` replays it
        exactly onto the same base, in this or another workspace.
        """
        from wse.patch_blocks import write_patch_file

        directory = _new_directory(path)
        data = write_patch_file(self.patch.model_dump(mode="json"), dict(self.blocks))
        (directory / "patch.wsepatch").write_bytes(data)
        source = self.base.source
        provenance = {
            "format": "wse-patch-v1",
            "base": {"checkpoint_id": self.base.checkpoint_id, "weights_hash": self.base.weights_hash,
                     "source": None if source is None else {"repository": source.repository,
                                                            "revision": source.revision}},
            "result_weights_hash": self.model.weights_hash,
            "patch_id": self.patch_id, "checkpoint_id": self.model.checkpoint_id,
            "steps": [{"operator": str(step.operator), "tensor": step.tensor,
                       "selection": [entry.model_dump(mode="json") for entry in step.selection],
                       "changed_elements": step.changed_elements} for step in self.patch.steps],
            "changed_elements": self.changed_elements,
            "patch_sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
            "wse_version": _wse_version(),
        }
        (directory / "patch.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                                              encoding="utf-8")
        return directory

    def table(self) -> str:
        """One line per changed tensor: changed entries, relative L2 change, and the before and after hashes."""
        lines = [f"{len(self.changed)} tensors changed, {self.changed_elements:,} entries; "
                 f"weights {_short_hash(self.base.weights_hash)} -> {_short_hash(self.model.weights_hash)}"]
        for d in self.changed:
            lines.append(f"  {d.tensor}: {d.changed_elements:,} of {d.element_count:,} entries, relative L2 change "
                         f"{d.relative_l2_delta:.4g}, {_short_hash(d.before_hash)} -> {_short_hash(d.after_hash)}")
        return "\n".join(lines)


@dataclass(frozen=True)
class Comparison:
    """A paired comparison of two models on the same problems, prompts, and seeds.

    ``result`` holds both models' metrics and the candidate-minus-base
    differences with paired cluster-bootstrap intervals. ``base`` and
    ``candidate`` are each model's result as the runner's ``summary.json``
    records it: metrics, intervals, identities, and per-problem samples.
    ``out_dir`` is the recorded run's directory, if there is one; an
    in-memory comparison instead keeps both ``evaluations`` (base, then
    candidate), with every output.
    """

    result: GenerationComparison
    base: Mapping[str, Any]
    candidate: Mapping[str, Any]
    out_dir: Path | None = None
    evaluations: tuple[Evaluation, Evaluation] | None = None
    metrics: tuple[str, ...] = METRICS

    def table(self) -> str:
        """Base, candidate, and difference (with its 95% paired interval) for each requested metric."""
        r = self.result
        lines = [f"{r.problems} problems, {r.samples} samples per model",
                 f"{'metric':<24}{'base':>10}{'candidate':>12}   difference [95% interval]"]
        for name in self.metrics:
            d = r.differences[name]
            digits = 1 if name == "mean_generated_tokens" else 3
            lines.append(f"{name:<24}{r.base[name]:>10.{digits}f}{r.candidate[name]:>12.{digits}f}   "
                         f"{d.estimate:+.{digits}f} [{d.lower:+.{digits}f}, {d.upper:+.{digits}f}]")
        return "\n".join(lines)


@dataclass(frozen=True, eq=False)
class Evaluation:
    """A generation evaluation with every sample's output: the input of a probe of the model's own answers.

    ``result`` is the ``EvaluationResult`` (metrics, intervals, per-problem
    samples, identities). ``outputs`` holds each sample's ``SampleOutput``
    (generated token IDs, text, whether it finished), in plan order, and
    ``prompts`` the rendered prompt's token IDs of each problem index.
    """

    result: EvaluationResult
    outputs: tuple[Any, ...]
    prompts: Mapping[int, tuple[int, ...]]
    max_new_tokens: int
    metric_names: tuple[str, ...] = METRICS

    @property
    def metrics(self) -> Mapping[str, float]:
        return self.result.metrics

    def limit_hit(self) -> list[int]:
        """Indices into ``outputs`` of the samples that reached the token limit without finishing."""
        return [k for k, o in enumerate(self.outputs) if not o.finished and len(o.tokens) >= self.max_new_tokens]

    def finished(self) -> list[int]:
        """Indices into ``outputs`` of the samples that ended with a stop token."""
        return [k for k, o in enumerate(self.outputs) if o.finished]

    def probe_texts(self, *, last: int | None = None, context: int | None = None) -> list[Any]:
        """One ``ProbeText`` per output: its prompt and output tokens, averaged over the output.

        A finished output's stop token is left out, so that finished and
        limit-hit windows hold the same kind of tokens; an output that is
        only its stop token keeps it, as its one-token window. With ``last``, the
        window is the output's last ``last`` tokens (all of them when it is
        shorter); the forward still reads the whole prompt and output before
        them, so every position has its full context. With ``context``, only
        the last ``context`` positions are kept: earlier tokens are dropped,
        which removes their context, so that a long output fits a probe's
        forward limit (``wse.probes.MAX_PROBE_TOKENS``).
        """
        from wse.probes import ProbeText

        for name, value in (("last", last), ("context", context)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise _invalid(f"{name} must be a positive integer")
        texts = []
        for output in self.outputs:
            drop = output.finished and len(output.tokens) > 1
            generated = tuple(output.tokens[:-1] if drop else output.tokens)
            tokens = tuple(self.prompts[output.index]) + generated
            size = len(generated) if last is None else min(last, len(generated))
            start = len(tokens) - size
            if context is not None and len(tokens) > context:
                dropped = len(tokens) - context
                tokens, start = tokens[dropped:], max(0, start - dropped)
            texts.append(ProbeText(tokens, start=start, stop=len(tokens)))
        return texts

    def samples(self) -> list[dict[str, Any]]:
        """One row per output, in ``outputs`` order: problem, item, sample, grade, answer, finish, tokens, text."""
        grades = {(problem.index, sample.sample): sample for problem in self.result.problems or []
                  for sample in problem.samples}
        rows = []
        for output in self.outputs:
            graded = grades.get((output.index, output.sample))
            rows.append({"problem": output.index, "item_id": output.item_id, "sample": output.sample,
                         "correct": None if graded is None else graded.correct,
                         "answer": None if graded is None else graded.answer,
                         "finished": output.finished,
                         "limit_hit": not output.finished and len(output.tokens) >= self.max_new_tokens,
                         "generated_tokens": len(output.tokens), "text": output.text})
        return rows

    def table(self) -> str:
        """Each requested metric with its 95% interval."""
        lines = [f"{self.result.sample_count} problems, {len(self.outputs)} samples"]
        for name in self.metric_names:
            interval = self.result.intervals.get(name)
            digits = 1 if name == "mean_generated_tokens" else 3
            bounds = f" [{interval.lower:.{digits}f}, {interval.upper:.{digits}f}]" if interval is not None else ""
            lines.append(f"{name:<24}{self.metrics[name]:>10.{digits}f}{bounds}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ProbeSample:
    """The sample behind one text of a probe of an evaluation: its problem, item, grade, finish, and text."""

    index: int
    problem: int
    item_id: str
    sample: int
    correct: bool | None
    finished: bool
    limit_hit: bool
    generated_tokens: int
    text: str


@dataclass(frozen=True, eq=False)
class Probe:
    """A probe's result (``wse.probes.ProbeResult``) that also answers to the site names it was asked with.

    ``names`` maps each requested site name to its canonical site; the result
    records canonical sites only. A probe of an evaluation keeps its texts'
    ``samples``: row ``i`` of ``means(site)`` is ``samples[i]``, so
    activations line up with generated texts, grades, and limit hits.
    """

    result: Any
    names: Mapping[str, str]
    samples: tuple[ProbeSample, ...] = ()

    def _canonical(self, site: str) -> str:
        return self.names.get(site, site)

    def means(self, site: str) -> np.ndarray:
        """Per-text window means of ``site``: float64 ``[texts, features]``."""
        canonical = self._canonical(site)
        if canonical not in self.result.means:
            raise WSEError(ErrorCode.NOT_FOUND, f"this probe holds no site {site!r}",
                           details={"site": site, "candidates": list(self.names)})
        return self.result.means[canonical]

    def group_mean(self, site: str, indices: Sequence[int]) -> np.ndarray:
        return self.result.group_mean(self._canonical(site), indices)

    def contrast(self, site: str, group_a: Sequence[int], group_b: Sequence[int]) -> Any:
        """The contrast of ``site`` between two groups of texts (``wse.probes.ProbeResult.contrast``)."""
        return self.result.contrast(self._canonical(site), group_a, group_b)


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


class Model:
    """A model in a workspace, for direct use: diagnose, investigate, intervene, evaluate (see the module docstring).

    Create it with ``from_pretrained`` or ``open``; edits create more. The
    object that created the engine owns it: ``close`` (or a ``with`` block)
    closes the workspace for it and every edit made from it.
    """

    def __init__(self, engine: Engine, handle: ModelHandle, files: Mapping[str, bytes], *,
                 checkpoint_id: str | None, owner: Model | None = None) -> None:
        self._engine = engine
        self._owner = owner
        self.handle = handle
        self.files: Mapping[str, bytes] = MappingProxyType(dict(files))
        self.checkpoint_id = checkpoint_id
        self._text_models: dict[str, Any] = {}
        self._closed = False

    # -- creation ---------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, source: str | Path, *, revision: str | None = None, repo: str | None = None,
                        workspace: str | Path | None = None, family: str | None = None,
                        license: str = "unspecified") -> Model:
        """Import a causal LM snapshot into the workspace (once) and open it.

        ``source`` is a local snapshot directory or a Hugging Face repository
        ID. A local directory needs ``repo`` and ``revision`` for its
        provenance, unless it is a Hugging Face cache snapshot, whose path
        names both. A repository ID is downloaded at ``revision`` (default
        ``main``), resolved to its commit (see the module docstring).
        ``workspace`` defaults to ``$WSE_WORKSPACE`` or ``./workspace``;
        importing the same snapshot again opens the stored checkpoint.
        """
        path = Path(source).expanduser()
        if path.is_dir() and (path / EXPORT_PROVENANCE).is_file() and repo is None and revision is None:
            # A directory that Model.export wrote: its labels name the export, not the snapshot it derives from.
            exported = json.loads((path / EXPORT_PROVENANCE).read_text(encoding="utf-8"))
            repo = f"wse-export/{exported.get('checkpoint_id') or 'in-memory'}"
            revision = "weights-" + str(exported.get("weights_hash", "")).removeprefix("sha256:")[:40]
        if path.is_dir():
            repo, revision = _local_labels(path, repo, revision)
        else:
            name = str(source)
            # A Path, or a string that starts like a path, names a local directory, never a repository.
            if isinstance(source, Path) or name.startswith((".", "/", "~")):
                raise WSEError(ErrorCode.NOT_FOUND, f"no such directory: {path}", details={"source": name})
            if repo is not None:
                raise _invalid("repo= names the source of a local snapshot; a repository ID is its own source")
            if not _REPO_ID.fullmatch(name):
                raise WSEError(ErrorCode.NOT_FOUND, f"{name!r} is not a snapshot directory, nor a Hugging Face "
                               "repository ID (ORG/NAME)", details={"source": name})
            path, repo, revision = _download(name, revision)
        engine = Engine(default_workspace() if workspace is None else workspace)
        try:
            manifest = engine.import_hf_snapshot(path, repo=repo, revision=revision, family=family, license=license)
            return cls._open(engine, manifest.checkpoint_id)
        except BaseException:
            engine.close()
            raise

    @classmethod
    def open(cls, checkpoint_id: str, *, workspace: str | Path | None = None) -> Model:
        """Open a stored checkpoint of the workspace (an import, or a candidate that ``apply_patch`` stored)."""
        engine = Engine(default_workspace() if workspace is None else workspace)
        try:
            return cls._open(engine, checkpoint_id)
        except BaseException:
            engine.close()
            raise

    @classmethod
    def _open(cls, engine: Engine, checkpoint_id: str) -> Model:
        manifest = engine.load_manifest(checkpoint_id)
        files = engine.model_files(checkpoint_id) if manifest.source is not None else {}
        return cls(engine, engine.open_checkpoint(checkpoint_id), files, checkpoint_id=checkpoint_id)

    def _child(self, handle: ModelHandle, checkpoint_id: str | None) -> Model:
        return Model(self._engine, handle, self.files, checkpoint_id=checkpoint_id, owner=self._owner or self)

    def close(self) -> None:
        """Free this model's device copies and, if this object opened the workspace, close it.

        Edits share their creator's workspace: once it is closed, they keep
        their in-memory work (``Edit.verify``, ``tensor``) and refuse
        workspace use with ``INVALID_ARGUMENT``.
        """
        self._text_models.clear()
        self.handle._runtimes.clear()
        if self._owner is None and not self._closed:
            self._closed = True
            self._engine.close()

    def __enter__(self) -> Model:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        source = self.source
        label = f"{source.repository}@{source.revision[:12]}" if source is not None else str(self.handle.adapter_ref)
        where = self.checkpoint_id or "in memory"
        return f"Model({label}, {where}, weights {_short_hash(self.weights_hash)})"

    # -- identity ---------------------------------------------------------------

    @property
    def engine(self) -> Engine:
        """The operator ``Engine`` of the workspace."""
        if (self._owner or self)._closed:
            raise _invalid("this model's workspace is closed; open it again with Model.open or Model.from_pretrained")
        return self._engine

    def _stored_checkpoint_id(self) -> str | None:
        """The stored checkpoint whose identity is this model's, if the workspace holds one."""
        from wse.schemas import model_identity

        identity = self.handle.identity
        for manifest in self.engine.list_models():
            if model_identity(manifest.adapter, manifest.config_hash, manifest.weights_hash,
                              manifest.source) == identity:
                return manifest.checkpoint_id
        return None

    @property
    def weights_hash(self) -> str:
        return self.handle.weights_hash

    @property
    def source(self) -> ModelSource | None:
        """The snapshot source: repository, revision, files read, and the frozen default profile."""
        return self.handle.source

    # -- names and targets ------------------------------------------------------

    def tensor_names(self, contains: str | None = None) -> list[str]:
        """Canonical tensor names, optionally only those containing ``contains``."""
        return [n for n in self.handle.tensor_names() if contains is None or contains in n]

    def name(self, short: str) -> str:
        """The one canonical tensor name that equals ``short`` or ends with ``.`` + ``short``.

        A name that matches nothing is tried again with ``.weight`` appended,
        so a module name means its weight. Several matches raise
        ``INVALID_ARGUMENT``, and no match raises ``NOT_FOUND``. Both list
        candidates in ``details["candidates"]``.
        """
        if not isinstance(short, str) or not short.strip():
            raise _invalid("a tensor name must be a nonempty string")
        short = short.strip()
        names = self.handle.tensor_names()
        if short in names:
            return short
        for query in (short, f"{short}.weight"):
            found = [n for n in names if n == query or n.endswith("." + query)]
            if len(found) == 1:
                return found[0]
            if found:
                raise _invalid(f"{short!r} matches {len(found)} tensors; give more of the name, for example "
                               f"{found[0]!r}", name=short, candidates=found[:20])
        width = len(short.split("."))
        keys: dict[str, str] = {}
        for n in names:
            for form in (n, n.removesuffix(".weight")):
                keys.setdefault(".".join(form.split(".")[-width:]), n)
        close = list(dict.fromkeys(keys[k] for k in difflib.get_close_matches(short, list(keys), n=5, cutoff=0.6)))
        raise WSEError(ErrorCode.NOT_FOUND, f"no tensor is named {short!r} or ends with it"
                       + (f"; close names: {close}" if close else ""), details={"name": short, "candidates": close})

    def tensor(self, name: str) -> torch.Tensor:
        """A copy of the tensor named ``name`` (short names allowed, see ``name``): changing it changes nothing here."""
        return self.handle.tensor(self.name(name))

    def _protected(self) -> set[str]:
        from wse.adapters.families import model_family

        family = model_family(self.handle)
        names = set(self.handle.tensor_names())
        protected = {name for name in names if family is not None and family.protected_role(name) is not None}
        for group in self.handle.architecture.alias_groups:
            protected |= set(group)
        return protected

    def tensors(self, pattern: str | None = None) -> list[TensorEntry]:
        """The tensors whose names match ``pattern``: name, shape, dtype, and whether edits of it are refused.

        ``pattern`` is a glob (``"layers.15.*"``) when it holds ``*``, ``?`` or
        ``[``, matched against the whole name or any dotted suffix of it, and
        otherwise a substring; ``None`` lists every tensor.
        """
        protected = self._protected()
        entries = []
        for name in self.handle.tensor_names():
            if pattern is not None:
                if any(char in pattern for char in "*?["):
                    if not (fnmatch.fnmatchcase(name, pattern) or fnmatch.fnmatchcase(name, "*." + pattern)):
                        continue
                elif pattern not in name:
                    continue
            tensor = self.handle._tensors[name]
            entries.append(TensorEntry(name=name, shape=tuple(tensor.shape),
                                       dtype=str(tensor.dtype).removeprefix("torch."), protected=name in protected))
        return entries

    def inspect(self) -> Inspection:
        """Identity, architecture, parameters, tensors by layer pattern, and protected tensors (``Inspection``)."""
        from wse.adapters.families import model_family

        stored = self.handle.config
        config = stored.get("model_config", stored) if isinstance(stored.get("model_config"), dict) else stored
        text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
        family = model_family(self.handle)
        groups: dict[str, list[str]] = {}
        for name in self.handle.tensor_names():
            groups.setdefault(re.sub(r"(?<=layers\.)\d+(?=\.)", "{i}", name), []).append(name)
        rows = []
        for pattern, names in groups.items():
            first = self.handle._tensors[names[0]]
            rows.append((pattern, len(names), tuple(first.shape), str(first.dtype).removeprefix("torch."),
                         sum(self.handle._tensors[n].numel() for n in names)))
        dtypes: dict[str, int] = {}
        for name in self.handle.tensor_names():
            key = str(self.handle._tensors[name].dtype).removeprefix("torch.")
            dtypes[key] = dtypes.get(key, 0) + 1
        source = self.source
        layers = text.get("num_hidden_layers") if isinstance(text, dict) else None
        return Inspection(
            checkpoint_id=self.checkpoint_id, weights_hash=self.weights_hash,
            source=None if source is None else f"{source.repository}@{source.revision}",
            adapter=str(self.handle.adapter_ref), family=None if family is None else str(family.ref),
            model_type=config.get("model_type"), layers=layers if isinstance(layers, int) else None,
            parameter_count=self.handle.parameter_count, tensor_count=len(self.handle.tensor_names()),
            dtypes=MappingProxyType(dtypes), groups=tuple(rows), protected=tuple(sorted(self._protected() - {
                name for group in self.handle.architecture.alias_groups for name in group})),
            aliases=tuple(tuple(sorted(group)) for group in self.handle.architecture.alias_groups),
            dropped_tensors=0 if source is None else source.dropped_tensor_count)

    def replace_weights(self, weights: Mapping[str, Any]) -> Edit:
        """An edit that replaces whole tensors with values computed any other way; this model is unchanged.

        ``weights`` maps tensor names (short names allowed, see ``name``) to
        their new values: tensors (or arrays) of the stored shape and dtype
        (a bfloat16 tensor stays bfloat16: convert with ``.to(torch.bfloat16)``),
        every value finite. Protected tensors and shared parameters are
        refused, as for every edit. The edit is an exact patch of
        ``replace_tensor@1`` steps, whose records name the submitted values by
        their content hash; ``Edit.save`` stores it.
        """
        from safetensors.torch import save as safetensors_bytes

        from wse.patches import materialize_patch
        from wse.tensors import dtype_name

        if not isinstance(weights, Mapping) or not weights:
            raise _invalid("weights must map tensor names to their new values")
        tensors: dict[str, torch.Tensor] = {}
        for name, value in weights.items():
            canonical = self.name(name)
            if canonical in tensors:
                raise _invalid(f"{name!r} and another key both name {canonical!r}", tensor=canonical)
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value))
            tensors[canonical] = tensor.detach().to("cpu").contiguous()
        for name, tensor in tensors.items():
            stored = self.handle._tensors[name]
            if dtype_name(tensor) != dtype_name(stored):
                raise _invalid(f"{name!r} is stored as {dtype_name(stored)}; the new values are "
                               f"{dtype_name(tensor)}: convert them with .to(torch.{dtype_name(stored)})",
                               tensor=name, dtype=dtype_name(tensor), stored_dtype=dtype_name(stored))
            if tuple(tensor.shape) != tuple(stored.shape):
                raise _invalid(f"{name!r} has shape {list(stored.shape)}; the new values have shape "
                               f"{list(tensor.shape)}", tensor=name)
        digest = hashlib.sha256(safetensors_bytes(dict(sorted(tensors.items())))).hexdigest()
        references: dict[str, np.ndarray] = {}
        operations = []
        for name, tensor in tensors.items():
            values = tensor.to(torch.float32).numpy()
            bad = int(np.count_nonzero(~np.isfinite(values)))
            if bad:
                raise _invalid(f"the new values of {name!r} hold {bad} nonfinite value(s); every weight must be "
                               "finite", tensor=name, nonfinite_count=bad)
            values.flags.writeable = False
            reference = f"wse://tensors/{digest}#{name}"
            references[reference] = values
            operations.append({"op": "replace_tensor", "tensor": name,
                               "selection": [{"start": 0, "stop": size, "step": 1} for size in tensor.shape],
                               "source": reference})
        result = materialize_patch(self.handle, operations, registry=self.engine.registry,
                                   policy=_policy().with_edit("replace_tensor", "replace_tensor@1"),
                                   references=references)
        return Edit(base=self, model=self._child(result.candidate, None), patch=result.patch, diff=result.diff,
                    blocks=result.blocks)

    def export(self, path: str | Path) -> Path:
        """Write this model as a Hugging Face model directory into ``path`` (new or empty); return it.

        The directory holds ``config.json``, the model's tokenizer, chat
        template, and generation configuration files as imported, and
        ``model.safetensors`` with every tensor under its checkpoint name
        (each name of a shared parameter as its own copy), so
        ``Model.from_pretrained(path)`` imports it again with this weights
        hash, and ``transformers`` loads it when the snapshot held nothing
        that WSE left out (``inspect().dropped_tensors``, for example a vision
        encoder). ``wse_provenance.json`` records the checkpoint, its weights
        hash, and the snapshot and patch it derives from.
        """
        from safetensors.torch import save_file

        if self.source is None or not self.files:
            raise WSEError(ErrorCode.UNSUPPORTED_CAPABILITY, "only imported open-weight models export as a Hugging "
                           "Face directory")
        directory = _new_directory(path)
        tensors = {name: self.handle._tensors[name].clone().contiguous() for name in sorted(self.handle.tensor_names())}
        save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})
        stored = self.handle.config
        config = stored["model_config"] if isinstance(stored.get("model_config"), dict) else stored
        (directory / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8")
        for name, data in sorted(self.files.items()):
            (directory / name).write_bytes(data)
        manifest = None if self.checkpoint_id is None else self.engine.load_manifest(self.checkpoint_id)
        provenance = {
            "format": "wse-model-v1", "checkpoint_id": self.checkpoint_id, "weights_hash": self.weights_hash,
            "source": {"repository": self.source.repository, "revision": self.source.revision,
                       "dropped_tensors": self.source.dropped_tensor_count},
            "parent_checkpoint_id": None if manifest is None else manifest.provenance.parent_checkpoint_id,
            "patch_id": None if manifest is None or manifest.provenance.kind != "patch" else manifest.provenance.source,
            "wse_version": _wse_version(),
        }
        (directory / EXPORT_PROVENANCE).write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
        return directory

    def roles(self) -> tuple[Any, ...]:
        """The semantic roles of this model's tensors (``wse.adapters.families.model_roles``); none at Level 0."""
        from wse.adapters.families import model_roles

        return model_roles(self.handle)

    def role(self, role: str, layer: int | None = None) -> Any:
        """The tensor that holds ``role`` in ``layer``, for example ``role("mlp.down", 15)``.

        It is a ``TensorRole``: ``.tensor`` is an edit target, ``.site`` the
        module for probe sites (``.site + ":input"``), and ``.protected`` tells
        whether every edit of it is refused. Model-wide roles (``embed``,
        ``lm_head``, ``final_norm``) take no layer.
        """
        from wse.adapters.families import role_tensor

        return role_tensor(self.handle, role, layer)

    def target(self, target: Any) -> tuple[str, list[dict[str, int]]]:
        """Resolve a target to the canonical tensor name and an explicit selection.

        A target is ``"<name>[<index>]"``, a ``TensorRole`` (``role``; the whole
        tensor), or a pair of a name or role and an explicit selection (a list
        of ``{start, stop, step}``, as ``TensorRole.selection(head=...)``
        gives). ``<name>`` is resolved by ``name``. The optional index is
        NumPy-style: one entry per leading axis, each an integer or a slice
        with step 1, with an optional ``...`` and a trailing comma; missing
        axes are taken whole. ``"layers.15.mlp.down_proj[:, :64]"`` selects
        every row and the first 64 columns.
        """
        from wse.adapters.families import TensorRole

        if isinstance(target, TensorRole):
            target = target.tensor
        elif isinstance(target, tuple) and len(target) == 2:
            name, selection = target
            tensor = self.name(name.tensor if isinstance(name, TensorRole) else name)
            if not isinstance(selection, Sequence) or isinstance(selection, (str, bytes)):
                raise _invalid("an explicit selection is a list of {start, stop, step}, one per axis")
            return tensor, [dict(entry) for entry in selection]
        if not isinstance(target, str):
            raise _invalid("a target is a string such as 'layers.0.mlp.down_proj[:, :8]', a TensorRole, or a "
                           "(name, selection) pair")
        if target.count("[") != target.count("]"):
            raise _invalid(f"{target}: unbalanced bracket in the index", target=target)
        match = _TARGET.fullmatch(target)
        if match is None:
            raise _invalid(f"{target}: expected '<tensor name>' or '<tensor name>[<index>]'", target=target)
        tensor = self.name(match["name"])
        shape = list(self.handle._tensors[tensor].shape)
        index = match["index"]
        if index is not None and not index.strip():
            raise _invalid(f"{target}: the index is empty", target=target)
        entries = [] if index is None else [entry.strip() for entry in index.split(",")]
        if len(entries) > 1 and entries[-1] == "":
            entries.pop()  # a trailing comma, as in "[0,]"
        if entries.count("...") > 1:
            raise _invalid(f"{target}: an index has at most one '...'", target=target)
        if "..." in entries:
            at = entries.index("...")
            fill = len(shape) - (len(entries) - 1)
            if fill < 0:
                raise _invalid(f"{target}: {len(entries) - 1} indices for a tensor with {len(shape)} axes {shape}",
                               target=target, shape=shape)
            entries = entries[:at] + [":"] * fill + entries[at + 1:]
        if len(entries) > len(shape):
            raise _invalid(f"{target}: {len(entries)} indices for a tensor with {len(shape)} axes {shape}",
                           target=target, shape=shape)
        entries += [":"] * (len(shape) - len(entries))
        return tensor, [_index(entry, size, axis, target) for axis, (entry, size) in enumerate(zip(entries, shape))]

    def site(self, site: str) -> str:
        """The canonical probe site for ``site``: ``residual:<i>``, or ``<module>:input`` / ``<module>:output``.

        ``<module>`` is a canonical module path (``wse.probes.module_paths``)
        or a unique dotted suffix of one, as for tensor names.
        """
        from wse.probes import module_paths

        if not isinstance(site, str) or not site.strip():
            raise _invalid("a site must be a nonempty string such as 'layers.15.mlp.down_proj:input' or 'residual:15'")
        if site.startswith("residual:"):
            return site
        path, colon, point = site.rpartition(":")
        if not colon or point not in ("input", "output") or not path:
            raise _invalid(f"site {site!r} must be 'residual:<i>' or '<module>:input' or '<module>:output'",
                           site=site)
        paths = list(module_paths(self.handle))
        found = [p for p in paths if p == path or p.endswith("." + path)]
        if len(found) == 1:
            return f"{found[0]}:{point}"
        if found:
            raise _invalid(f"{path!r} matches {len(found)} modules; give more of the name", site=site,
                           candidates=found[:20])
        close = difflib.get_close_matches(path, paths, n=5, cutoff=0.0)
        raise WSEError(ErrorCode.NOT_FOUND, f"no module path is {path!r} or ends with it; close paths: {close}",
                       details={"site": site, "candidates": close})

    # -- investigate ------------------------------------------------------------

    def probe(self, texts: Any, *, sites: Sequence[str], device: str | None = None,
              max_tokens: int | None = None, last: int | None = None, context: int | None = None) -> Probe:
        """Teacher-force texts and average each site's activations over each text's window (``wse.probes``).

        ``texts`` is an ``Evaluation`` (its outputs after their prompts,
        averaged over each output, or over its ``last`` tokens, with at most
        ``context`` positions read: ``Evaluation.probe_texts``) or a sequence
        of ``ProbeText``, token-ID sequences, or strings. A string is encoded
        as the tokenizer encodes plain text (Llama's tokenizer prepends its BOS
        token; no chat template). Site names may be short (``site``).
        ``max_tokens`` bounds each text's forward; by default it is the
        longest text, within ``wse.probes.MAX_PROBE_TOKENS``. The cost grows
        with the forward length (``wse.probes`` gives measured times and
        memory).
        """
        from wse.probes import MAX_PROBE_TOKENS, ProbeText, probe

        text_model = self._text_model(device)
        if isinstance(texts, Evaluation):
            items = texts.probe_texts(last=last, context=context)
        elif last is not None or context is not None:
            raise _invalid("last and context shape the windows of an Evaluation; give ProbeText windows instead")
        elif isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise _invalid("texts must be an Evaluation or a sequence of texts")
        else:
            items = []
            for text in texts:
                if isinstance(text, ProbeText):
                    items.append(text)
                elif isinstance(text, str):
                    items.append(ProbeText(tuple(text_model.tokenizer(text)["input_ids"])))
                else:
                    items.append(ProbeText(text))
        names = {name: self.site(name) for name in sites}
        if max_tokens is None:
            lengths = [len(item.tokens) if item.stop is None else item.stop for item in items
                       if isinstance(item.tokens, tuple)]
            max_tokens = min(max(lengths, default=1), MAX_PROBE_TOKENS)
        result = probe(self.handle, items, sites=list(names.values()), device=text_model.runtime.device.type,
                       max_tokens=max_tokens)
        samples: tuple[ProbeSample, ...] = ()
        if isinstance(texts, Evaluation):
            samples = tuple(ProbeSample(index=i, problem=row["problem"], item_id=row["item_id"], sample=row["sample"],
                                        correct=row["correct"], finished=row["finished"], limit_hit=row["limit_hit"],
                                        generated_tokens=row["generated_tokens"], text=row["text"])
                            for i, row in enumerate(texts.samples()))
        return Probe(result=result, names=MappingProxyType(names), samples=samples)

    # -- intervene --------------------------------------------------------------

    def edit(self, target: Any, operator: str, **parameters: Any) -> Edit:
        """Apply one registered edit operator in memory; return the edit (this model is unchanged).

        ``target`` is resolved by ``target``; ``operator`` is an exact
        reference (``"project_out@1"``) or a name in ``OPERATORS``; the
        keyword arguments are the operator's parameters, for example
        ``factor=0.5`` for ``scale_slice``, ``value=0.0`` for ``fill_slice``,
        ``direction=..., columns=..., alpha=...`` for ``project_out``, or
        ``u=..., v=..., alpha=...`` for ``add_low_rank``, where ``u`` and ``v``
        list the ``r`` columns of ``U`` and ``V`` (arrays of shape
        ``[r, rows]`` and ``[r, columns]``: the factors transposed).
        NumPy arrays and tensors among them
        become lists, and NumPy scalars Python numbers.
        """
        if {"tensor", "selection"} & set(parameters):
            raise _invalid("give the tensor and its index as the target (the first argument), not as parameters")
        if not isinstance(operator, str) or ("@" not in operator and operator not in OPERATORS):
            raise WSEError(ErrorCode.NOT_FOUND, f"unknown edit operator {operator!r}; the known ones are "
                           f"{sorted(OPERATORS)}, or give an exact reference such as 'project_out@1'",
                           details={"operator": str(operator), "known": sorted(OPERATORS)})
        tensor, selection = self.target(target)
        result = apply_operator(self.handle, OPERATORS.get(operator, operator),
                                {"tensor": tensor, "selection": selection,
                                 **{key: _plain(value) for key, value in parameters.items()}},
                                registry=self.engine.registry, policy=_policy())
        return Edit(base=self, model=self._child(result.candidate, None), patch=result.patch, diff=result.diff,
                    blocks=result.blocks)

    def apply_patch(self, path: str | Path, *, repo: str | None = None, revision: str | None = None,
                    format: str = "lm-understand-v1") -> Edit:
        """Import a published or exported edit as an exact patch onto this stored model, store it, and return it.

        ``format="lm-understand-v1"`` (the article's published edits): every
        base hash is verified before anything is rebuilt and every rebuilt
        tensor against its published hash (``Engine.import_patch``); ``repo``
        and ``revision`` name the edit's source for its provenance.
        ``format="wse-patch-v1"``: a directory that ``Edit.export`` wrote,
        replayed exactly on this base (``Engine.store_patch`` checks every
        step's hashes). Importing the same edit again returns the stored
        record.
        """
        from wse.patches import tensor_diffs

        if self.checkpoint_id is None:
            raise _invalid("apply_patch needs a stored base model; this model is an in-memory edit, so apply the "
                           "published edit to the model loaded with from_pretrained or open")
        if format == "wse-patch-v1":
            from wse.patch_blocks import read_patch_file

            document, blocks = read_patch_file((Path(path).expanduser() / "patch.wsepatch").read_bytes())
            patch = MaterializedPatch.model_validate(document)
            record = self.engine.store_patch(patch, blocks, base_checkpoint_id=self.checkpoint_id,
                                             label="import of an exported patch")
            model = self._child(self.engine.open_checkpoint(record.checkpoint_id), record.checkpoint_id)
            return Edit(base=self, model=model, patch=patch, diff=tensor_diffs(self.handle, model.handle),
                        blocks=blocks, patch_id=record.patch_id)
        if repo is None or revision is None:
            raise _invalid("a published edit needs repo= and revision=, the source WSE records as its provenance")
        record = self.engine.import_patch(path, base_checkpoint_id=self.checkpoint_id, format=format, repo=repo,
                                          revision=revision)
        patch, blocks = self.engine.read_imported_patch(record.patch_id)
        model = self._child(self.engine.open_checkpoint(record.checkpoint_id), record.checkpoint_id)
        return Edit(base=self, model=model, patch=patch, diff=tensor_diffs(self.handle, model.handle), blocks=blocks,
                    patch_id=record.patch_id)

    def diff_from(self, other: Model) -> tuple[TensorDiff, ...]:
        """Per-tensor differences from ``other`` (a model of the same architecture) to this model."""
        from wse.patches import tensor_diffs

        return tensor_diffs(other.handle, self.handle)

    # -- diagnose ---------------------------------------------------------------

    def _text_model(self, device: str | None) -> Any:
        from wse.engines.hf_torch import select_device
        from wse.evaluators.generation import load_text_model

        if not self.files:
            raise WSEError(ErrorCode.UNSUPPORTED_CAPABILITY, "generation needs an imported open-weight model with "
                           "its tokenizer files")
        key = select_device(device).type
        if key not in self._text_models:
            self._text_models[key] = load_text_model(self.handle, self.files, device=key)
        return self._text_models[key]

    def _profile_ref(self, profile: str | None) -> str:
        if profile is not None:
            return profile
        default = self.source.default_profile if self.source is not None else None
        if default is None:
            raise _invalid("this model has no default generation profile; name one, for example 'qwen35think@1'")
        return str(default.ref)

    def generate(self, prompt: str, *, max_new_tokens: int | None = None, profile: str | None = None,
                 seed: int = 0, device: str | None = None, **overrides: Any) -> Any:
        """Generate one answer to ``prompt``; return its ``SampleOutput`` (text, tokens, finished, seed).

        The prompt is rendered and sampled by ``profile`` (default: the
        model's own default profile from its ``generation_config.json``), with
        ``max_new_tokens`` and any other profile fields overridden. The output
        is a function of the prompt, the profile, and ``seed``.
        """
        from wse.evaluators.generation import generate_samples
        from wse.evaluators.sampling import resolve_profile, with_overrides

        if max_new_tokens is not None:
            overrides["max_new_tokens"] = max_new_tokens
        resolved = with_overrides(resolve_profile(self.engine.registry, self._profile_ref(profile), self.source),
                                  overrides)
        (output,) = generate_samples(self._text_model(device), resolved,
                                     [TextItem(item_id="prompt", problem=prompt, reference="")], [(0, 0)],
                                     eval_seed=seed)
        return output

    def _suite(self, dataset: str | Path, *, format: str, problems: int | None, subset_seed: int,
               profile: str | None, grader: str | None, samples: int, max_new_tokens: int | None, eval_seed: int,
               id_prefix: str | None) -> tuple[Any, dict[str, Any], int]:
        from wse.evaluators.generation_runs import subset_suite
        from wse.evaluators.text_suites import read_jsonl

        data, _ = read_jsonl(dataset, format=format, id_prefix=id_prefix)
        count = len(data.items) if problems is None else problems
        parameters: dict[str, Any] = {"profile": self._profile_ref(profile), "grader": grader or GRADERS[format],
                                      "samples_per_problem": samples}
        if max_new_tokens is not None:
            parameters["max_new_tokens"] = max_new_tokens
        suite, _ = subset_suite(data, count=count, seed=subset_seed, parameters=parameters, eval_seed=eval_seed)
        return suite, parameters, count

    def evaluate(self, data: str | Path, *, evaluator: str = "math", metrics: Sequence[str] | None = None,
                 max_new_tokens: int | None = None, problems: int | None = None, subset_seed: int = 0,
                 profile: str | None = None, samples: int = 1, eval_seed: int = 0, batch_size: int = 8,
                 device: str | None = None, format: str | None = None, grader: str | None = None,
                 id_prefix: str | None = None, log: Any = None) -> Evaluation:
        """Evaluate generation on the problems of a JSON Lines file, keeping every output (``Evaluation``).

        ``evaluator="math"`` reads GSM8K rows (``question``, ``answer`` ending
        in ``#### <number>``) or MATH rows (``problem`` and ``answer``, or
        ``solution`` with a final ``\\boxed{}``), renders each problem with the
        boxed-answer instruction through ``profile`` (default: the model's
        own), and grades the final boxed answer; ``format`` and ``grader``
        override the detected format and its grader (``GRADERS``).
        ``problems`` problems are chosen by ``subset_seed`` (all of them when
        ``None``). The result's metrics are accuracy, limit-hit rate, and mean
        generated tokens, with intervals, ``metrics`` naming the ones the
        table shows; ``Evaluation.samples()`` lists every output with its
        grade and text, and ``Evaluation.outputs`` holds its tokens, for
        probing. ``log`` receives one line per graded problem.
        """
        from wse.evaluators.generation import encode_prompts, run_generation

        dataset = data
        names = _metric_names(metrics)
        format = _evaluation_format(dataset, evaluator, format)
        suite, _, count = self._suite(dataset, format=format, problems=problems, subset_seed=subset_seed,
                                      profile=profile, grader=grader, samples=samples, max_new_tokens=max_new_tokens,
                                      eval_seed=eval_seed, id_prefix=id_prefix)
        text_model = self._text_model(device)

        def progress(problem: Any, outputs: Any) -> None:
            sample = problem.samples[0]
            log(f"{problem.index + 1}/{count} {problem.item_id}: correct={sample.correct} "
                f"tokens={sample.generated_tokens} limit_hit={sample.limit_hit}")

        run = run_generation(self.handle, suite, EvalConfig(sample_count=count, seed=eval_seed),
                             text_model=text_model, registry=self.engine.registry, batch_size=batch_size,
                             on_problem=None if log is None else progress)
        prompts = encode_prompts(text_model, run.plan.profile, run.plan.items, range(len(run.plan.items)))
        return Evaluation(result=run.result, outputs=run.outputs,
                          prompts=MappingProxyType({index: tuple(ids) for index, ids in prompts.items()}),
                          max_new_tokens=run.plan.profile.max_new_tokens, metric_names=names)

    # -- evaluate ---------------------------------------------------------------

    def compare(self, candidate: Model | Edit, data: str | Path, *, evaluator: str = "math",
                metrics: Sequence[str] | None = None, out_dir: str | Path | None = None, format: str | None = None,
                problems: int | None = None, subset_seed: int = 0, profile: str | None = None,
                grader: str | None = None, samples: int = 1, max_new_tokens: int | None = None, eval_seed: int = 0,
                batch_size: int = 8, device: str | None = None, id_prefix: str | None = None,
                log: Any = None) -> Comparison:
        """Evaluate this model (the base) and ``candidate`` on the same problems; return the paired comparison.

        The arguments are those of ``evaluate`` (``compare_models`` is the same
        with both models named). Both models use one profile,
        the base's default unless ``profile`` names one, resolved for both
        before anything is generated. Without ``out_dir`` both evaluations
        run in memory, and the comparison keeps them. With ``out_dir`` the
        resumable runner (``wse.evaluators.generation_runs.run_comparison``)
        records every problem there and continues an interrupted run. That
        needs both models stored as checkpoints: save an in-memory edit
        first (``Edit.save``). ``log`` receives one line per graded problem
        and model.
        """
        from wse.evaluators.generation import compare_generation
        from wse.evaluators.generation_runs import result_summary, run_comparison
        from wse.evaluators.sampling import resolve_profile

        other = candidate.model if isinstance(candidate, Edit) else candidate
        dataset = data
        names = _metric_names(metrics)
        format = _evaluation_format(dataset, evaluator, format)
        profile = self._profile_ref(profile)
        for model in (self, other):  # fail before generating if either model cannot use the profile
            resolve_profile(self.engine.registry, profile, model.source)
        if out_dir is None:
            options = dict(evaluator=evaluator, metrics=names, format=format, problems=problems,
                           subset_seed=subset_seed, profile=profile, grader=grader, samples=samples,
                           max_new_tokens=max_new_tokens, eval_seed=eval_seed, batch_size=batch_size, device=device,
                           id_prefix=id_prefix)
            base = self.evaluate(dataset, **options, log=None if log is None else lambda line: log(f"base {line}"))
            other_evaluation = other.evaluate(dataset, **options,
                                              log=None if log is None else lambda line: log(f"candidate {line}"))
            return Comparison(result=compare_generation(base.result, other_evaluation.result),
                              base=result_summary(base.result), candidate=result_summary(other_evaluation.result),
                              evaluations=(base, other_evaluation), metrics=names)
        if self.checkpoint_id is None or other.checkpoint_id is None:
            raise _invalid("a recorded comparison (out_dir) needs both models stored as checkpoints; save an "
                           "in-memory edit first (edit.save()), or compare it without out_dir")
        _, parameters, count = self._suite(dataset, format=format, problems=problems, subset_seed=subset_seed,
                                           profile=profile, grader=grader, samples=samples,
                                           max_new_tokens=max_new_tokens, eval_seed=eval_seed, id_prefix=id_prefix)
        summary = run_comparison(self.engine, base=self.checkpoint_id, candidate=other.checkpoint_id, dataset=dataset,
                                 format=format, count=count, subset_seed=subset_seed, profile=parameters["profile"],
                                 grader=parameters["grader"], out_dir=out_dir, samples_per_problem=samples,
                                 max_new_tokens=max_new_tokens, eval_seed=eval_seed, device=device,
                                 batch_size=batch_size, id_prefix=id_prefix, log=log)
        return Comparison(result=GenerationComparison.model_validate(summary["comparison"]),
                          base=summary["results"]["base"], candidate=summary["results"]["candidate"],
                          out_dir=Path(out_dir), metrics=names)


def compare_models(*, base: Model, candidate: Model | Edit, data: str | Path, evaluator: str = "math",
                   metrics: Sequence[str] | None = None, max_new_tokens: int | None = None,
                   out_dir: str | Path | None = None, **options: Any) -> Comparison:
    """Evaluate ``base`` and ``candidate`` on the same data, prompts, sampling settings, and seeds; pair them.

    ``evaluator`` defines how the data is read, prompted, and graded
    (``"math"``: see ``Model.evaluate``); ``metrics`` names what the table
    reports (default: accuracy, limit-hit rate, and mean generated tokens);
    the result gives each metric's raw values, their difference, and a 95%
    paired interval. With ``out_dir``, the resumable runner records every
    problem there (both models must be stored). Other options are those of
    ``Model.compare``: ``problems``, ``subset_seed``, ``profile``,
    ``samples``, ``eval_seed``, ``batch_size``, ``device``, ``log``.
    """
    if not isinstance(base, Model):
        raise _invalid("base must be a Model")
    return base.compare(candidate, data, evaluator=evaluator, metrics=metrics, max_new_tokens=max_new_tokens,
                        out_dir=out_dir, **options)
