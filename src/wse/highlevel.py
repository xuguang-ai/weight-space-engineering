"""``WSE``: one object per model for the four steps of a weight-space experiment.

    from wse import WSE

    base = WSE.from_pretrained("Qwen/Qwen3.5-0.8B")
    p50c = base.apply_patch("lm_understand-logs/exp8_math/runs/p50c/edit",
                            repo="RikiyaT/lm_understand-logs", revision="4d09ba6d6a7e891dd8063974dfe81b3d78e3494f")
    print(p50c.table())
    print(base.compare(p50c, "MATH-500/test.jsonl", problems=2, profile="qwen35think@1").table())

The steps, and the methods that serve them:

* **Diagnose:** ``generate`` and ``evaluate`` (``generation@1``): accuracy,
  limit-hit rate, and generated tokens, with intervals.
* **Investigate:** ``tensor_names``, ``tensor``, and ``probe`` (activation
  means at named sites, and contrasts between groups of texts).
* **Intervene:** ``edit`` applies a registered edit operator in memory, and
  ``Edit.save`` stores its candidate with the exact patch; ``apply_patch``
  imports a published edit as an exact patch and stores its candidate.
* **Evaluate:** ``compare`` gives paired differences on the same problems,
  prompts, and seeds. With ``out_dir``, it runs the resumable runner, which
  records every problem; that needs stored models.

Underneath is the low-level API. ``WSE.engine`` is the operator ``Engine`` of
the workspace and ``WSE.handle`` the ``ModelHandle``. Every edit is an exact
patch with before and after hashes, which ``Edit.verify`` replays. Short
tensor names and NumPy-style indices are resolved here, so patches always
record full names and explicit ranges.

Version-less operator names resolve to the versions pinned in ``OPERATORS``,
never to the latest registered version.

A ``WSE`` object is for direct use by the operator. Its work is unmanaged: it
belongs to no experiment and no budget. Imports (``from_pretrained``,
``apply_patch``) and saved edits (``Edit.save``) are recorded in the
workspace's audit log; an edit kept in memory is not. Agents work through the
managed experiments of the MCP server (``wse serve``).

**Downloads.** ``from_pretrained`` with a Hub repository ID resolves the
revision to a commit and lists the repository's files. It then downloads
only the allow-listed files at the snapshot root (``wse.hf_snapshot``) into
the Hugging Face cache. Gated repositories use the user's own Hugging Face
login; WSE never reads or stores a token.
"""

from __future__ import annotations

import difflib
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

__all__ = ["GRADERS", "OPERATORS", "WORKSPACE_ENV", "Comparison", "Edit", "Evaluation", "Probe", "WSE",
           "default_workspace"]

WORKSPACE_ENV = "WSE_WORKSPACE"
DEFAULT_WORKSPACE = "workspace"
# Version-less edit operator names and the exact versions they mean here.
OPERATORS = MappingProxyType({
    "scale_slice": "scale_slice@1",
    "set_slice": "set_slice@1",
    "project_out": "project_out@1",
})
_HUB_CACHE_DIR = re.compile(r"^models--(?:(?P<org>[A-Za-z0-9][A-Za-z0-9._-]*)--)?(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)$")
# The default grader of each dataset format.
GRADERS = MappingProxyType({"math500": "math_boxed@1", "math": "math_boxed@1", "gsm8k": "gsm8k_numeric@1"})
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


@dataclass(frozen=True, eq=False)
class Edit:
    """An edit of ``base`` into ``model``: the exact patch, its per-tensor diff, and its binary blocks.

    ``patch_id`` names the stored patch in the workspace (``apply_patch``, or
    ``save``); an edit kept in memory has none, and its ``model`` has no
    checkpoint.
    """

    base: WSE
    model: WSE
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
        """Store the edited model in the workspace with its exact patch; return the stored edit.

        ``Engine.store_patch`` replays the patch on the stored base, which
        checks every hash, then commits the candidate with its patch file and
        an audit event. A stored model can be reopened (``WSE.open``) and
        compared on record (``compare(..., out_dir=...)``). The base must be
        stored: for an edit of an edit, save the earlier edit first. Saving
        the same edit again returns the stored one; ``label`` names it in the
        record.
        """
        if self.patch_id is not None and self.model.checkpoint_id is not None:
            return self
        base_id = self.base.checkpoint_id or self.base._stored_checkpoint_id()
        if base_id is None:
            raise _invalid("this edit's base is an edit kept in memory, so it is not stored; save the edit that "
                           "made the base first")
        record = self.base.engine.store_patch(self.patch, self.blocks, base_checkpoint_id=base_id, label=label)
        if record.result_weights_hash != self.model.weights_hash:  # pragma: no cover - store_patch replays the patch
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, "the stored candidate's weights differ from the edit's")
        model = self.base._child(self.model.handle, record.checkpoint_id)
        return Edit(base=self.base, model=model, patch=self.patch, diff=self.diff, blocks=self.blocks,
                    patch_id=record.patch_id)

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

    def table(self) -> str:
        """Base, candidate, and difference (with its 95% interval) for each metric."""
        r = self.result
        lines = [f"{r.problems} problems, {r.samples} samples per model",
                 f"{'metric':<24}{'base':>10}{'candidate':>12}   difference [95% interval]"]
        for name in ("accuracy", "limit_hit_rate", "mean_generated_tokens"):
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
        limit-hit windows hold the same kind of tokens. With ``last``, the
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
            generated = tuple(output.tokens[:-1] if output.finished and output.tokens else output.tokens)
            tokens = tuple(self.prompts[output.index]) + generated
            size = len(generated) if last is None else min(last, len(generated))
            start = len(tokens) - size
            if context is not None and len(tokens) > context:
                dropped = len(tokens) - context
                tokens, start = tokens[dropped:], max(0, start - dropped)
            texts.append(ProbeText(tokens, start=start, stop=len(tokens)))
        return texts

    def table(self) -> str:
        """Each metric with its 95% interval."""
        lines = [f"{self.result.sample_count} problems, {len(self.outputs)} samples"]
        for name in ("accuracy", "limit_hit_rate", "mean_generated_tokens"):
            interval = self.result.intervals.get(name)
            digits = 1 if name == "mean_generated_tokens" else 3
            bounds = f" [{interval.lower:.{digits}f}, {interval.upper:.{digits}f}]" if interval is not None else ""
            lines.append(f"{name:<24}{self.metrics[name]:>10.{digits}f}{bounds}")
        return "\n".join(lines)


@dataclass(frozen=True, eq=False)
class Probe:
    """A probe's result (``wse.probes.ProbeResult``) that also answers to the site names it was asked with.

    ``names`` maps each requested site name to its canonical site; the result
    records canonical sites only.
    """

    result: Any
    names: Mapping[str, str]

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


class WSE:
    """A model in a workspace, for direct use: diagnose, investigate, intervene, evaluate (see the module docstring).

    Create it with ``from_pretrained`` or ``open``; edits create more. The
    object that created the engine owns it: ``close`` (or a ``with`` block)
    closes the workspace for it and every edit made from it.
    """

    def __init__(self, engine: Engine, handle: ModelHandle, files: Mapping[str, bytes], *,
                 checkpoint_id: str | None, owner: WSE | None = None) -> None:
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
                        license: str = "unspecified") -> WSE:
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
    def open(cls, checkpoint_id: str, *, workspace: str | Path | None = None) -> WSE:
        """Open a stored checkpoint of the workspace (an import, or a candidate that ``apply_patch`` stored)."""
        engine = Engine(default_workspace() if workspace is None else workspace)
        try:
            return cls._open(engine, checkpoint_id)
        except BaseException:
            engine.close()
            raise

    @classmethod
    def _open(cls, engine: Engine, checkpoint_id: str) -> WSE:
        manifest = engine.load_manifest(checkpoint_id)
        files = engine.model_files(checkpoint_id) if manifest.source is not None else {}
        return cls(engine, engine.open_checkpoint(checkpoint_id), files, checkpoint_id=checkpoint_id)

    def _child(self, handle: ModelHandle, checkpoint_id: str | None) -> WSE:
        return WSE(self._engine, handle, self.files, checkpoint_id=checkpoint_id, owner=self._owner or self)

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

    def __enter__(self) -> WSE:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        source = self.source
        label = f"{source.repository}@{source.revision[:12]}" if source is not None else str(self.handle.adapter_ref)
        where = self.checkpoint_id or "in memory"
        return f"WSE({label}, {where}, weights {_short_hash(self.weights_hash)})"

    # -- identity ---------------------------------------------------------------

    @property
    def engine(self) -> Engine:
        """The operator ``Engine`` of the workspace."""
        if (self._owner or self)._closed:
            raise _invalid("this model's workspace is closed; open it again with WSE.open or WSE.from_pretrained")
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
        """A copy of the tensor named ``name`` (short names allowed, see ``name``)."""
        return self.handle.tensor(self.name(name))

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
        return Probe(result=result, names=MappingProxyType(names))

    # -- intervene --------------------------------------------------------------

    def edit(self, target: Any, operator: str, **parameters: Any) -> Edit:
        """Apply one registered edit operator in memory; return the edit (this model is unchanged).

        ``target`` is resolved by ``target``; ``operator`` is an exact
        reference (``"project_out@1"``) or a name in ``OPERATORS``; the
        keyword arguments are the operator's parameters, for example
        ``factor=0.5`` for ``scale_slice`` or ``direction=..., columns=...,
        alpha=...`` for ``project_out``. NumPy arrays and tensors among them
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

    def apply_patch(self, path: str | Path, *, repo: str, revision: str, format: str = "lm-understand-v1") -> Edit:
        """Import a published edit as an exact patch onto this stored model, store its candidate, and return it.

        Every base hash is verified before anything is rebuilt and every
        rebuilt tensor against its published hash (``Engine.import_patch``).
        ``repo`` and ``revision`` name the edit's source for its provenance.
        Importing the same edit again returns the stored record.
        """
        from wse.patches import tensor_diffs

        if self.checkpoint_id is None:
            raise _invalid("apply_patch needs a stored base model; this model is an in-memory edit, so apply the "
                           "published edit to the model loaded with from_pretrained or open")
        record = self.engine.import_patch(path, base_checkpoint_id=self.checkpoint_id, format=format, repo=repo,
                                          revision=revision)
        patch, blocks = self.engine.read_imported_patch(record.patch_id)
        model = self._child(self.engine.open_checkpoint(record.checkpoint_id), record.checkpoint_id)
        return Edit(base=self, model=model, patch=patch, diff=tensor_diffs(self.handle, model.handle), blocks=blocks,
                    patch_id=record.patch_id)

    def diff_from(self, other: WSE) -> tuple[TensorDiff, ...]:
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

    def evaluate(self, dataset: str | Path, *, format: str = "math500", problems: int | None = None,
                 subset_seed: int = 0, profile: str | None = None, grader: str | None = None, samples: int = 1,
                 max_new_tokens: int | None = None, eval_seed: int = 0, batch_size: int = 8,
                 device: str | None = None, id_prefix: str | None = None, log: Any = None) -> Evaluation:
        """Evaluate generation on ``problems`` problems of a JSON Lines dataset, keeping every output.

        The problems are chosen by ``subset_seed`` (all of them when
        ``problems`` is ``None``); ``format`` is ``math500``, ``gsm8k``, or
        ``math``, and ``grader`` defaults to the format's (``GRADERS``). The
        result's metrics are accuracy, limit-hit rate, and mean generated
        tokens, with intervals; ``Evaluation.outputs`` holds every sample's
        tokens and text, for probing. ``log`` receives one line per graded
        problem.
        """
        from wse.evaluators.generation import encode_prompts, run_generation

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
                          max_new_tokens=run.plan.profile.max_new_tokens)

    # -- evaluate ---------------------------------------------------------------

    def compare(self, candidate: WSE | Edit, dataset: str | Path, *, out_dir: str | Path | None = None,
                format: str = "math500", problems: int | None = None, subset_seed: int = 0,
                profile: str | None = None, grader: str | None = None, samples: int = 1,
                max_new_tokens: int | None = None, eval_seed: int = 0, batch_size: int = 8,
                device: str | None = None, id_prefix: str | None = None, log: Any = None) -> Comparison:
        """Evaluate this model (the base) and ``candidate`` on the same problems; return the paired comparison.

        The arguments are those of ``evaluate``. Both models use one profile,
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
        profile = self._profile_ref(profile)
        for model in (self, other):  # fail before generating if either model cannot use the profile
            resolve_profile(self.engine.registry, profile, model.source)
        if out_dir is None:
            options = dict(format=format, problems=problems, subset_seed=subset_seed, profile=profile, grader=grader,
                           samples=samples, max_new_tokens=max_new_tokens, eval_seed=eval_seed,
                           batch_size=batch_size, device=device, id_prefix=id_prefix)
            base = self.evaluate(dataset, **options, log=None if log is None else lambda line: log(f"base {line}"))
            other_evaluation = other.evaluate(dataset, **options,
                                              log=None if log is None else lambda line: log(f"candidate {line}"))
            return Comparison(result=compare_generation(base.result, other_evaluation.result),
                              base=result_summary(base.result), candidate=result_summary(other_evaluation.result),
                              evaluations=(base, other_evaluation))
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
                          out_dir=Path(out_dir))
