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
* **Intervene:** ``edit`` applies a registered edit operator in memory;
  ``apply_patch`` imports a published edit as an exact patch and stores its
  candidate.
* **Evaluate:** ``compare`` gives paired differences on the same problems,
  prompts, and seeds. With ``out_dir``, it runs the resumable runner, which
  records every problem.

Underneath is the low-level API. ``WSE.engine`` is the operator ``Engine`` of
the workspace and ``WSE.handle`` the ``ModelHandle``. Every edit is an exact
patch with before and after hashes, which ``Edit.verify`` replays. Short
tensor names and NumPy-style indices are resolved here, so patches always
record full names and explicit ranges.

Version-less operator names resolve to the versions pinned in ``OPERATORS``,
never to the latest registered version.

A ``WSE`` object is for direct use by the operator. Its edits are unmanaged:
they belong to no experiment, budget, or audit log. Agents work through the
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

import torch

from wse.api import Engine, ModelHandle, apply_operator
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

__all__ = ["OPERATORS", "WORKSPACE_ENV", "Comparison", "Edit", "WSE", "default_workspace"]

WORKSPACE_ENV = "WSE_WORKSPACE"
DEFAULT_WORKSPACE = "workspace"
# Version-less edit operator names and the exact versions they mean here.
OPERATORS = MappingProxyType({
    "scale_slice": "scale_slice@1",
    "set_slice": "set_slice@1",
    "project_out": "project_out@1",
})
_HUB_CACHE_DIR = re.compile(r"^models--(?P<org>[A-Za-z0-9][A-Za-z0-9._-]*)--(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)$")
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
            return f"{match['org']}/{match['name']}", parts[-1]
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
        if not re.fullmatch(r"[+-]?\d+", value):
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


def _short_hash(value: str) -> str:
    return value.removeprefix("sha256:")[:12]


@dataclass(frozen=True, eq=False)
class Edit:
    """An edit of ``base`` into ``model``: the exact patch, its per-tensor diff, and its binary blocks.

    ``patch_id`` names an imported patch in the workspace (``apply_patch``);
    an in-memory edit has none, and its ``model`` has no checkpoint.
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
    ``out_dir`` is the recorded run's directory, if there is one.
    """

    result: GenerationComparison
    base: Mapping[str, Any]
    candidate: Mapping[str, Any]
    out_dir: Path | None = None

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
            if repo is not None:
                raise _invalid("repo= names the source of a local snapshot; a repository ID is its own source")
            if not _REPO_ID.fullmatch(name):
                raise WSEError(ErrorCode.NOT_FOUND, f"{name!r} is neither a snapshot directory nor a Hugging Face "
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
        """Close the workspace if this object opened it; edits share their creator's workspace."""
        self._text_models.clear()
        if self._owner is None:
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
        return self._engine

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
        names = self.handle.tensor_names()
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

    def target(self, target: str) -> tuple[str, list[dict[str, int]]]:
        """Resolve ``"<name>[<index>]"`` to the canonical tensor name and an explicit selection.

        ``<name>`` is resolved by ``name``. The optional index is NumPy-style:
        one entry per leading axis, each an integer or a slice with step 1;
        missing axes are taken whole. ``"layers.15.mlp.down_proj[:, :64]"``
        selects every row and the first 64 columns.
        """
        if not isinstance(target, str):
            raise _invalid("a target must be a string such as 'layers.0.mlp.down_proj[:, :8]'")
        if target.count("[") != target.count("]"):
            raise _invalid(f"{target}: unbalanced bracket in the index", target=target)
        match = _TARGET.fullmatch(target)
        if match is None:
            raise _invalid(f"{target}: expected '<tensor name>' or '<tensor name>[<index>]'", target=target)
        tensor = self.name(match["name"])
        shape = list(self.handle._tensors[tensor].shape)
        entries = [] if match["index"] is None else match["index"].split(",")
        if match["index"] is not None and not match["index"].strip():
            raise _invalid(f"{target}: the index is empty", target=target)
        if len(entries) > len(shape):
            raise _invalid(f"{target}: {len(entries)} indices for a tensor with {len(shape)} axes {shape}",
                           target=target, shape=shape)
        entries += [":"] * (len(shape) - len(entries))
        return tensor, [_index(entry, size, axis, target) for axis, (entry, size) in enumerate(zip(entries, shape))]

    # -- intervene --------------------------------------------------------------

    def edit(self, target: str, operator: str, **parameters: Any) -> Edit:
        """Apply one registered edit operator in memory; return the edit (this model is unchanged).

        ``target`` is resolved by ``target``; ``operator`` is an exact
        reference (``"project_out@1"``) or a name in ``OPERATORS``; the
        keyword arguments are the operator's parameters, for example
        ``factor=0.5`` for ``scale_slice`` or ``direction=..., columns=...,
        alpha=...`` for ``project_out``.
        """
        if {"tensor", "selection"} & set(parameters):
            raise _invalid("give the tensor and its index as the target (the first argument), not as parameters")
        tensor, selection = self.target(target)
        result = apply_operator(self.handle, OPERATORS.get(operator, operator),
                                {"tensor": tensor, "selection": selection, **parameters},
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
               profile: str | None, grader: str, samples: int, max_new_tokens: int | None, eval_seed: int,
               id_prefix: str | None) -> tuple[Any, dict[str, Any], int]:
        from wse.evaluators.generation_runs import subset_suite
        from wse.evaluators.text_suites import read_jsonl

        data, _ = read_jsonl(dataset, format=format, id_prefix=id_prefix)
        count = len(data.items) if problems is None else problems
        parameters: dict[str, Any] = {"profile": self._profile_ref(profile), "grader": grader,
                                      "samples_per_problem": samples}
        if max_new_tokens is not None:
            parameters["max_new_tokens"] = max_new_tokens
        suite, _ = subset_suite(data, count=count, seed=subset_seed, parameters=parameters, eval_seed=eval_seed)
        return suite, parameters, count

    def evaluate(self, dataset: str | Path, *, format: str = "math500", problems: int | None = None,
                 subset_seed: int = 0, profile: str | None = None, grader: str = "math_boxed@1", samples: int = 1,
                 max_new_tokens: int | None = None, eval_seed: int = 0, batch_size: int = 8,
                 device: str | None = None, id_prefix: str | None = None) -> Any:
        """Evaluate generation on ``problems`` problems of a JSON Lines dataset; return the ``EvaluationResult``.

        The problems are chosen by ``subset_seed`` (all of them when
        ``problems`` is ``None``); ``format`` is ``math500``, ``gsm8k``, or
        ``math``. Metrics: accuracy, limit-hit rate, and mean generated
        tokens, with intervals; ``problems`` holds every sample.
        """
        from wse.evaluators.generation import evaluate_generation

        suite, _, count = self._suite(dataset, format=format, problems=problems, subset_seed=subset_seed,
                                      profile=profile, grader=grader, samples=samples, max_new_tokens=max_new_tokens,
                                      eval_seed=eval_seed, id_prefix=id_prefix)
        return evaluate_generation(self.handle, suite, EvalConfig(sample_count=count, seed=eval_seed),
                                   text_model=self._text_model(device), registry=self.engine.registry,
                                   batch_size=batch_size)

    # -- evaluate ---------------------------------------------------------------

    def compare(self, candidate: WSE | Edit, dataset: str | Path, *, out_dir: str | Path | None = None,
                format: str = "math500", problems: int | None = None, subset_seed: int = 0,
                profile: str | None = None, grader: str = "math_boxed@1", samples: int = 1,
                max_new_tokens: int | None = None, eval_seed: int = 0, batch_size: int = 8,
                device: str | None = None, id_prefix: str | None = None, log: Any = None) -> Comparison:
        """Evaluate this model (the base) and ``candidate`` on the same problems; return the paired comparison.

        The arguments are those of ``evaluate``. Without ``out_dir`` both
        evaluations run in memory. With ``out_dir`` the resumable runner
        (``wse.evaluators.generation_runs.run_comparison``) records every
        problem there and continues an interrupted run. That needs both models
        stored as checkpoints, so an in-memory edit cannot be recorded.
        """
        from wse.evaluators.generation import compare_generation
        from wse.evaluators.generation_runs import result_summary, run_comparison

        other = candidate.model if isinstance(candidate, Edit) else candidate
        if out_dir is None:
            options = dict(format=format, problems=problems, subset_seed=subset_seed, profile=profile, grader=grader,
                           samples=samples, max_new_tokens=max_new_tokens, eval_seed=eval_seed,
                           batch_size=batch_size, device=device, id_prefix=id_prefix)
            base_result = self.evaluate(dataset, **options)
            candidate_result = other.evaluate(dataset, **options)
            return Comparison(result=compare_generation(base_result, candidate_result),
                              base=result_summary(base_result), candidate=result_summary(candidate_result))
        if self.checkpoint_id is None or other.checkpoint_id is None:
            raise _invalid("a recorded comparison (out_dir) needs both models stored as checkpoints; an in-memory "
                           "edit has none, so compare it without out_dir")
        _, parameters, count = self._suite(dataset, format=format, problems=problems, subset_seed=subset_seed,
                                           profile=profile, grader=grader, samples=samples,
                                           max_new_tokens=max_new_tokens, eval_seed=eval_seed, id_prefix=id_prefix)
        summary = run_comparison(self.engine, base=self.checkpoint_id, candidate=other.checkpoint_id, dataset=dataset,
                                 format=format, count=count, subset_seed=subset_seed, profile=parameters["profile"],
                                 grader=grader, out_dir=out_dir, samples_per_problem=samples,
                                 max_new_tokens=max_new_tokens, eval_seed=eval_seed, device=device,
                                 batch_size=batch_size, id_prefix=id_prefix, log=log)
        return Comparison(result=GenerationComparison.model_validate(summary["comparison"]),
                          base=summary["results"]["base"], candidate=summary["results"]["candidate"],
                          out_dir=Path(out_dir))
