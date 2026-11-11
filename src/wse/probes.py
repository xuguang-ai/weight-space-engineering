"""Activation probes: teacher-forced window means at residual and module sites (M2 plan Section 5.3).

``probe`` runs each text through a model's engine runtime
(``wse.engines.hf_torch``) in one teacher-forced forward pass and returns,
per text and site, the mean activation over a window of positions, and the
mean log-probability of the text's own next tokens over that window.
``ProbeResult.contrast`` compares two groups of texts. This is the in-memory
core; the managed probe (``Engine.submit_probe``, a worker job metered by
policy and budget) builds on it, and the end of this module holds its pure
parts: the cutting rule of its texts, its result artifact, and its inline
summary (**Managed probes** below).

* **Texts.** A ``ProbeText`` holds the token IDs to teacher-force (for example
  a prompt followed by an output) and the window ``[start, stop)`` of
  positions whose activations are averaged (by default the whole text). The
  forward covers positions ``[0, stop)``: one text per forward, batch 1, no
  padding, no cache, under inference mode. Tokens at or after ``stop`` are
  never fed to the model and change nothing; they must still be token IDs of
  the vocabulary.
* **Residual sites.** ``residual:<i>``, for ``0 <= i <= L`` with ``L`` the
  number of decoder layers (the module list of ``num_hidden_layers``
  entries), is the residual stream at layer boundary ``i``:

  * ``residual:0`` is the first positional input of decoder layer 0, what the
    decoder stack receives from the embeddings: the embedding module's output
    for most families (Gemma 3's embedding module scales by
    ``sqrt(hidden_size)``), and GPT-2's token plus learned position
    embeddings;
  * ``residual:i`` for ``1 <= i <= L`` is the output of decoder layer
    ``i - 1``; ``residual:L`` is the last layer's output, before the final
    norm.

* **Why hooks, not ``output_hidden_states``.** ``transformers`` 5.22.2
  returns ``L + 1`` hidden states, recorded by hooks on the same decoder
  layers, whose first ``L`` are the boundaries ``0`` to ``L - 1`` above; but
  its ``capture_outputs`` replaces the last one with ``last_hidden_state``,
  the final norm's output, so its index ``L`` is not the residual stream.
  Here ``residual:L`` is the stream before that norm, and the normed state is
  the site ``<final norm>:output`` (for example ``model.norm:output``). The
  test suite checks both on Llama, Qwen3, Gemma 3, GPT-2, and Qwen3.5
  fixtures. Hooks also reduce in place, so no full activation is stored.
* **Module sites.** ``<module>:input`` and ``<module>:output``, where
  ``<module>`` is a canonical module path: a dotted prefix of canonical tensor
  names, such as ``model.language_model.layers.15.mlp.down_proj`` for the
  tensor ``model.language_model.layers.15.mlp.down_proj.weight``. It maps to
  the runtime module through the tensors' module names (``module_paths``);
  at Level 0 the two are equal. ``:input`` is the first positional argument
  of the module's forward (a forward pre-hook) and ``:output`` the tensor the
  module returns (a forward hook).
* **What a site must capture.** A floating-point tensor ``[1, stop,
  features]`` (batch, sequence, features), exactly once per forward. Any
  other capture is refused with ``INVALID_ARGUMENT``, for example a module
  that returns a tuple (attention modules: probe their output projection
  instead), receives its input by keyword (attention again: probe the output
  of the norm before it), is applied to a reshaped tensor (per-head norms),
  never runs (the convolution of Qwen3.5's linear attention, whose kernel
  reads the weight directly), or runs more than once in one forward.
* **Numerics.** Inside the hook, the window of each capture is widened from
  the activation's dtype (BF16, F16, or F32) to float32, which is exact, and
  summed over positions on the model's device in float32 (PyTorch's
  reduction); only these per-text sums are kept. After each forward the sums
  move to the CPU, where they are widened to float64 and divided by the
  window length (MPS has no float64). Group means, contrast directions,
  norms, and standardized differences are float64 (NumPy); a group's mean is
  shifted by its first text (``x_0 + mean(x - x_0)``), so a group of identical
  texts has exactly that text's mean and zero variance. Log-probabilities
  are the engine's float32 ``log_softmax`` values
  (``HFTorchModel.next_token_logprobs``), averaged in float64; the output
  head runs on the scored positions only, so on MPS they can differ in their
  last digits from ``log_softmax`` of the full sequence's logits (by at most
  2.2e-4 over a 512-token Qwen3.5-0.8B output, measured), and on the CPU
  fixtures they agree bitwise. Results are bitwise reproducible within one
  device and library version. On
  Qwen3.5-0.8B on MPS, over a 3,600-position window, the float32 sums
  differed from float64 sums of the same activations by at most 1.2e-7 of
  each dimension's mean magnitude; activations themselves differ between MPS
  and CPU kernels far more (BF16 rounding).
* **Log-probabilities.** ``mean_logprobs[k]`` is the mean log-probability of
  the window's own tokens given everything before them: the mean of
  ``log p(tokens[t] | tokens[:t])`` over ``max(start, 1) <= t < stop``. It is
  ``None`` only for the window ``[0, 1)``, whose one token has no prefix. A
  window over an output therefore scores every output token.
* **Limits.** ``max_tokens`` (default ``DEFAULT_MAX_PROBE_TOKENS``, at most
  ``MAX_PROBE_TOKENS``) bounds each text's forward length ``stop``, and so
  does the model's context (``max_positions``). Cost grows with ``stop``:
  every position runs through the decoder. The output head runs only on the
  positions that score the window's tokens (the window and the position
  before it, through ``logits_to_keep``), unless a site reads the output
  head, which then sees every position; those logits are held in the
  module's dtype (on Qwen3.5-0.8B, with its 248,320-token vocabulary in
  BF16, 0.5 MB per position: 2 GB for 4,096 positions and 8 GB for 16,384).
  Measured on the operator's Mac (MPS) with every position's logits, one
  probe of Qwen3.5-0.8B took 1.7 s at 4,000 positions and 8.6 s at 16,384,
  with about 5 GB and 13 GB of MPS driver memory (2.1 GB of it the model). A
  probe reads at most ``MAX_PROBE_SITES`` distinct sites.
* **Contrasts.** ``ProbeResult.contrast(site, a, b)`` treats each text's mean
  as one observation, so texts weigh equally whatever their token counts.
  ``direction`` is the mean over group ``a`` minus the mean over group ``b``,
  and ``norm`` its Euclidean norm. ``standardized`` divides ``direction``,
  per dimension, by the pooled standard deviation
  ``s = sqrt((SS_a + SS_b) / (n_a + n_b - 2))``, where ``SS_g`` is the sum of
  squared deviations of group ``g``'s texts from the group mean. Where ``s``
  is 0, ``standardized`` is 0 if the difference is 0 and an infinity of the
  difference's sign otherwise (each group is constant there, and they
  differ); JSON surfaces must encode those infinities, which canonical JSON
  rejects. The groups must be disjoint and non-empty and hold at least three
  texts together, so that ``s`` has a degree of freedom.

* **Capability.** A model is probed only when its adapter declares
  ``probe.hidden_states`` (``hf_causal_lm@1``); others, such as M1's MLP
  adapter, are refused with ``UNSUPPORTED_CAPABILITY``.

The probe never changes weights: its hooks only read, and they are removed
when the call returns or raises. It holds the runtime's forward lock
(``HFTorchModel.exclusive``) from registering its hooks to removing them, so
no other thread's forward on the same runtime runs meanwhile, and its hooks
see only its own forwards. ``on_forward`` is called before each text's
forward with the text's index and forward length; it may raise to stop the
probe (the managed worker meters and stops probes through it).

**Managed probes** (``Engine.submit_probe``; M2 plan Section 5.3). A probe job
teacher-forces texts of three kinds, each built by the worker:

* an *output* of one of the run's generation passes, by its output URI (one
  URI names one sample): its prompt, rendered as the pass rendered it, then
  its generated tokens, with a finished output's stop token left out (unless
  it is the only token) so that finished and limit-hit windows hold the same
  kind of tokens; the window is the output, or its last ``last`` tokens;
* a *string*, encoded as the tokenizer encodes plain text (Llama's tokenizer
  prepends its BOS token; no chat template); the window is every token after
  the first;
* an *item* of a development suite of the run's task: its prompt as a pass
  renders it; the window is every token after the first.

**Cutting.** A text longer than ``limit`` positions (the policy's
``tokens_per_probe_text``, bounded by ``MAX_PROBE_TOKENS`` and the model's
context) keeps its last ``limit`` positions: earlier tokens, the prompt first,
are dropped, which removes their context (``cut_text``). The window keeps the
positions of its own tokens that remain, but never the first kept position,
which has no context once the text is cut. So no window holds position 0 of a
forward: every window token is scored from a prefix (``mean_logprobs``), and
attention-sink activations at the first position, which many models have,
stay out of the means.

**Artifact.** ``probe_document`` is the full result, stored as a run-scoped
artifact (``wse://analyses/<sha256>``): per text its source, forward length,
window, dropped tokens, token hash, and mean log-probability; per site its
mean (no groups) or the means of groups ``a`` and ``b``, their norms, and the
contrast ``a - b`` with its norm and standardized differences. Its vectors are
vector references: ``<uri>#/sites/<i>/contrast/direction``,
``<uri>#/sites/<i>/group_means/a`` (or ``/b``), and ``<uri>#/sites/<i>/mean``,
with ``i`` the site's position in the request. A non-finite value is written
as the string ``"inf"``, ``"-inf"``, or ``"nan"`` (canonical JSON has no
infinities): standardized differences are infinite where both groups are
constant and differ. A vector holding such a string is not a valid vector
reference.

**Summary.** The job's inline result (``fit_summary``) gives the artifact's
URI and size and, per text, its source, forward length, window length,
dropped tokens, and mean log-probability; per site its features, mean norms,
contrast norm, and the ``top`` dimensions of largest absolute standardized
difference (``top_dimensions``: infinities first); numbers to 6 significant
digits. Every site lists the same number of dimensions (all of them when it
has fewer): the largest number, at most the requested ``top``, whose summary
fits the policy's inline-result limit less ``PROBE_JOB_BYTES`` for the job
around it. Submission refuses a probe whose summary could not fit even
without top dimensions (``summary_size_bound``).

This module does not import ``transformers``.
"""

from __future__ import annotations

import contextlib
import difflib
import math
import operator
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from wse.schemas import MAX_LAYER_INDEX, ErrorCode, ModelIdentity, TensorInfo, WSEError, canonical_json

if TYPE_CHECKING:
    from wse.api import ModelHandle
    from wse.engines.hf_torch import HFTorchModel

DEFAULT_MAX_PROBE_TOKENS = 4096
# The hard limit on a text's forward length: twice the 8,192 new tokens of a generation pass (M2 plan Section 5.5),
# so a whole limit-hit output can be probed. On Qwen3.5-0.8B its BF16 logits alone are 8 GB at this length.
MAX_PROBE_TOKENS = 16384
# Enough for two sites in every layer of a 64-layer model.
MAX_PROBE_SITES = 128
MAX_SITE_CANDIDATES = 10
# The longest site name: a module path is a prefix of a tensor name (at most 256 characters in a patch operation)
# followed by ":input" or ":output". Longer names are refused before they are parsed or echoed.
MAX_SITE_CHARACTERS = 512
CAPTURE_POINTS = ("input", "output")
# A residual boundary has at most 9 digits (MAX_LAYER_INDEX), so parsing it never meets Python's integer-digit limit.
_RESIDUAL = re.compile(r"residual:(0|[1-9][0-9]{0,8})")
_RESIDUAL_DIGITS = re.compile(r"residual:[0-9]+")
# Managed probes: the inline summary's default and largest number of top standardized dimensions per site, and
# the allowance for the job's other fields around the summary.
DEFAULT_SUMMARY_TOP = 8
MAX_SUMMARY_TOP = 64
PROBE_JOB_BYTES = 3 * 1024


def _invalid(message: str, **details: Any) -> WSEError:
    return WSEError(ErrorCode.INVALID_ARGUMENT, message, details=details)


def residual_boundary(name: str, index: int) -> int | None:
    """The layer boundary of a residual site name, or ``None`` for any other name, after the bounds every site obeys.

    A name longer than ``MAX_SITE_CHARACTERS``, and ``residual:`` followed by
    digits that are not a boundary of at most 9 digits without leading
    zeros, are ``INVALID_ARGUMENT``; ``index`` is the site's position in the
    request, which the error names instead of echoing the site.
    """
    if len(name) > MAX_SITE_CHARACTERS:
        raise _invalid(f"site {index} has {len(name)} characters; a site name has at most {MAX_SITE_CHARACTERS}",
                       site=index)
    residual = _RESIDUAL.fullmatch(name)
    if residual:
        return int(residual.group(1))
    if _RESIDUAL_DIGITS.fullmatch(name):
        raise _invalid(f"site {index} ({name[:24]!r}{'...' if len(name) > 24 else ''}) is not a residual site: a "
                       f"layer boundary has no leading zeros and at most {len(str(MAX_LAYER_INDEX))} digits",
                       site=index)
    return None


def _integer(value: Any) -> int | None:
    """``value`` as an ``int`` when it is an integer other than a bool, else ``None``."""
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        return operator.index(value)
    except TypeError:
        return None


# ---------------------------------------------------------------------------
# Texts and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeText:
    """Token IDs to teacher-force, and the window ``[start, stop)`` of positions whose activations are averaged.

    ``stop`` defaults to ``len(tokens)``; the forward covers positions
    ``[0, stop)``. A list, NumPy array, or tensor of token IDs is stored as a
    tuple; ``probe`` checks every field.
    """

    tokens: tuple[int, ...]
    start: int = 0
    stop: int | None = None

    def __post_init__(self) -> None:
        tokens = self.tokens
        if isinstance(tokens, (np.ndarray, torch.Tensor)):
            tokens = tokens.tolist()
        if isinstance(tokens, Sequence) and not isinstance(tokens, (str, bytes)):
            object.__setattr__(self, "tokens", tuple(tokens))


@dataclass(frozen=True, eq=False)
class Contrast:
    """The contrast of one site between two groups of texts (float64; module docstring, **Contrasts**).

    It holds arrays, so it compares by identity, as ``ProbeResult`` does.
    """

    site: str
    direction: np.ndarray
    norm: float
    standardized: np.ndarray
    sizes: tuple[int, int]


def _shifted_mean(rows: np.ndarray) -> np.ndarray:
    """The mean of ``rows`` shifted by the first row: exact for identical rows, and better conditioned otherwise."""
    return rows[0] + (rows - rows[0]).mean(axis=0)


def _frozen(array: Any) -> np.ndarray:
    """A read-only float64 array: ``array`` itself when it already is one, otherwise a private copy."""
    result = np.asarray(array, dtype=np.float64)
    if result.flags.writeable:
        result = result.copy()
        result.setflags(write=False)
    return result


@dataclass(frozen=True, eq=False)
class ProbeResult:
    """Per-text window means of each site, with token counts, log-probabilities, and the model's identity.

    ``means[site]`` is float64 ``[len(texts), features]``: row ``k`` is text
    ``k``'s mean over its window. ``token_counts[k]`` is the window length and
    ``mean_logprobs[k]`` the mean log-probability of the window's tokens given
    their prefixes (``None`` for the window ``[0, 1)``).
    ``runtime`` records the engine's device, stored dtypes, and library
    versions. Results are immutable (the means are read-only arrays) and,
    since they hold arrays, compare by identity.
    """

    sites: tuple[str, ...]
    means: Mapping[str, np.ndarray]
    token_counts: tuple[int, ...]
    mean_logprobs: tuple[float | None, ...]
    model: ModelIdentity
    runtime: Mapping[str, Any]

    def __post_init__(self) -> None:
        sites = tuple(self.sites)
        if set(self.means) != set(sites) or len(set(sites)) != len(sites):
            raise ValueError("means must hold exactly one array per site")
        counts, logprobs = tuple(self.token_counts), tuple(self.mean_logprobs)
        means = {site: _frozen(self.means[site]) for site in sites}
        if len(logprobs) != len(counts) or any(m.ndim != 2 or m.shape[0] != len(counts) for m in means.values()):
            raise ValueError("every site's means, token_counts, and mean_logprobs must have one entry per text")
        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "means", MappingProxyType(means))
        object.__setattr__(self, "token_counts", counts)
        object.__setattr__(self, "mean_logprobs", logprobs)
        object.__setattr__(self, "runtime", MappingProxyType(dict(self.runtime)))

    def _site(self, site: str) -> np.ndarray:
        if not isinstance(site, str) or site not in self.means:
            raise WSEError(ErrorCode.NOT_FOUND, f"this result holds no site {site!r}",
                           details={"site": str(site), "candidates": list(self.sites[:MAX_SITE_CANDIDATES])})
        return self.means[site]

    def _group(self, indices: Any, label: str) -> list[int]:
        if isinstance(indices, (str, bytes)) or not isinstance(indices, (Sequence, np.ndarray)):
            raise _invalid(f"{label} must be a sequence of text indices", group=label)
        rows = [_integer(index) for index in indices]
        if not rows:
            raise _invalid(f"{label} is empty; a group holds at least one text", group=label)
        count = len(self.token_counts)
        if any(row is None or not 0 <= row < count for row in rows):
            raise _invalid(f"{label} must hold integer text indices in [0, {count})", group=label)
        if len(set(rows)) != len(rows):
            raise _invalid(f"{label} lists a text more than once", group=label)
        return rows  # type: ignore[return-value]

    def group_mean(self, site: str, indices: Sequence[int]) -> np.ndarray:
        """The mean of ``site``'s per-text means over the texts ``indices`` (float64; texts weigh equally)."""
        means = self._site(site)
        return _shifted_mean(means[self._group(indices, "indices")])

    def contrast(self, site: str, group_a: Sequence[int], group_b: Sequence[int]) -> Contrast:
        """The contrast of ``site`` between two disjoint groups of texts (module docstring, **Contrasts**)."""
        means = self._site(site)
        rows_a, rows_b = self._group(group_a, "group_a"), self._group(group_b, "group_b")
        shared = sorted(set(rows_a) & set(rows_b))
        if shared:
            raise _invalid(f"the groups share texts {shared[:MAX_SITE_CANDIDATES]}; a contrast compares disjoint "
                           "groups", shared=shared[:MAX_SITE_CANDIDATES])
        if len(rows_a) + len(rows_b) < 3:
            raise _invalid("a contrast needs at least three texts in its two groups, so that the pooled standard "
                           "deviation has a degree of freedom; compare group_mean values for single texts")
        a, b = means[rows_a], means[rows_b]
        mean_a, mean_b = _shifted_mean(a), _shifted_mean(b)
        direction = mean_a - mean_b
        squares = ((a - mean_a) ** 2).sum(axis=0) + ((b - mean_b) ** 2).sum(axis=0)
        pooled = np.sqrt(squares / (len(rows_a) + len(rows_b) - 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            standardized = np.where(pooled > 0, direction / pooled,
                                    np.where(direction == 0, 0.0, np.copysign(np.inf, direction)))
        return Contrast(site=site, direction=_frozen(direction), norm=float(np.linalg.norm(direction)),
                        standardized=_frozen(standardized), sizes=(len(rows_a), len(rows_b)))


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------


def module_paths(model: ModelHandle) -> dict[str, str]:
    """Canonical module path -> runtime module path, for every module that holds a stored tensor.

    A canonical module path is a dotted prefix of canonical tensor names; its
    runtime module is the same prefix of the tensors' module names
    (``TensorInfo.module_name``), which family name maps rewrite by prefix.
    For Qwen3.5, ``model.language_model.layers.15.mlp.down_proj`` maps to
    ``model.layers.15.mlp.down_proj``; at Level 0 every path maps to itself.
    The root module is not a site.
    """
    return canonical_module_paths(model.architecture.tensors)


def canonical_module_paths(tensors: Iterable[TensorInfo]) -> dict[str, str]:
    """``module_paths`` from a model's tensor entries (a manifest's ``tensors`` suffices; no weight is read)."""
    paths: dict[str, str] = {}
    for info in tensors:
        canonical, module = info.name.split("."), (info.module_name or info.name).split(".")
        for size in range(1, min(len(canonical), len(module))):
            if canonical[-size:] != module[-size:]:
                break
            key, value = ".".join(canonical[:-size]), ".".join(module[:-size])
            if paths.setdefault(key, value) != value:
                raise WSEError(ErrorCode.INVALID_MODEL, f"canonical module path {key!r} maps to two runtime modules, "
                               f"{paths[key]!r} and {value!r}", details={"path": key})
    return dict(sorted(paths.items()))


@dataclass(frozen=True)
class _Site:
    name: str
    module: torch.nn.Module
    point: str  # "input" or "output"
    residual: bool


def _decoder_layers(module: torch.nn.Module) -> torch.nn.ModuleList:
    """The decoder layers: the one ``ModuleList`` with ``num_hidden_layers`` entries."""
    config = getattr(module, "config", None)
    text_config = config.get_text_config() if callable(getattr(config, "get_text_config", None)) else config
    count = _integer(getattr(text_config, "num_hidden_layers", None))
    lists = [sub for sub in module.modules() if isinstance(sub, torch.nn.ModuleList) and len(sub) == count]
    if count is None or count < 1 or len(lists) != 1:
        raise WSEError(ErrorCode.UNSUPPORTED_CAPABILITY,
                       f"residual sites need the decoder layers, one module list of num_hidden_layers ({count}) "
                       f"entries; {type(module).__name__} has {len(lists)}; probe module sites instead")
    return lists[0]


def _resolve_sites(model: ModelHandle, runtime: HFTorchModel, sites: Any) -> list[_Site]:
    if isinstance(sites, (str, bytes)) or not isinstance(sites, Sequence):
        raise _invalid("sites must be a sequence of site names, such as ['residual:0', "
                       "'model.layers.0.self_attn.o_proj:output']")
    names = list(sites)
    if not names:
        raise _invalid("sites must name at least one site")
    if len(names) > MAX_PROBE_SITES:
        raise _invalid(f"a probe reads at most {MAX_PROBE_SITES} sites, not {len(names)}", sites=len(names))
    if not all(isinstance(name, str) for name in names):
        raise _invalid("every site must be a string")
    boundaries = [residual_boundary(name, index) for index, name in enumerate(names)]
    if len(set(names)) != len(names):
        repeated = sorted({name for name in names if names.count(name) > 1})
        raise _invalid(f"sites {repeated[:MAX_SITE_CANDIDATES]} are listed more than once",
                       sites=repeated[:MAX_SITE_CANDIDATES])
    paths: dict[str, str] | None = None
    layers: torch.nn.ModuleList | None = None
    resolved = []
    for name, boundary in zip(names, boundaries, strict=True):
        if boundary is not None:
            layers = layers if layers is not None else _decoder_layers(runtime.module)
            index, count = boundary, len(layers)
            if index > count:
                nearest = sorted(range(count + 1), key=lambda i: (abs(i - index), i))[:MAX_SITE_CANDIDATES]
                raise WSEError(ErrorCode.NOT_FOUND,
                               f"site {name!r} does not exist: the model has {count} decoder layers, so residual "
                               f"sites run from residual:0 to residual:{count}",
                               details={"site": name, "candidates": [f"residual:{i}" for i in nearest]})
            module, point = (layers[0], "input") if index == 0 else (layers[index - 1], "output")
            resolved.append(_Site(name, module, point, residual=True))
            continue
        path, colon, point = name.rpartition(":")
        if not colon or not path or point not in CAPTURE_POINTS:
            raise _invalid(f"site {name!r} is neither 'residual:<layer boundary>' nor '<canonical module path>:input' "
                           "or ':output'", site=name)
        paths = paths if paths is not None else module_paths(model)
        target = paths.get(path)
        try:
            module = runtime.module.get_submodule(target) if target is not None else None
        except AttributeError:
            module = None
        if module is None:
            candidates = difflib.get_close_matches(path, list(paths), n=MAX_SITE_CANDIDATES, cutoff=0.0)
            raise WSEError(ErrorCode.NOT_FOUND,
                           f"site {name!r} names no module: a module path is a dotted prefix of canonical tensor "
                           f"names (module_paths); the closest are {candidates[:3]}",
                           details={"site": name, "candidates": candidates})
        resolved.append(_Site(name, module, point, residual=False))
    return resolved


_KEYWORD_INPUT = object()  # a pre-hook's stand-in for a forward called without positional arguments


def _describe(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"a {str(value.dtype).removeprefix('torch.')} tensor of shape {list(value.shape)}"
    if value is _KEYWORD_INPUT:
        return "no positional argument (the module's input is passed by keyword)"
    return f"a {type(value).__name__}"


class _Capture:
    """One site's hook: the float32 sum over the current forward's window, kept on the device."""

    def __init__(self, site: _Site) -> None:
        self.site = site
        self.window = (0, 0)
        self.calls = 0
        self.total: torch.Tensor | None = None

    def begin(self, start: int, stop: int) -> None:
        self.window, self.calls, self.total = (start, stop), 0, None

    def pre_hook(self, module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        self._add(args[0] if args else _KEYWORD_INPUT)

    def hook(self, module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> None:
        self._add(output)

    def _add(self, value: Any) -> None:
        name = self.site.name
        self.calls += 1
        if self.calls > 1:
            raise _invalid(f"site {name!r} ran more than once in one forward, so its capture is ambiguous", site=name)
        start, stop = self.window
        if not (isinstance(value, torch.Tensor) and value.is_floating_point() and value.dim() == 3
                and value.shape[0] == 1 and value.shape[1] == stop):
            message = (f"site {name!r} captured {_describe(value)}; a probe site must hold one floating-point vector "
                       f"per position, shaped [1, {stop}, features] for this forward of {stop} positions")
            if self.site.residual:
                raise WSEError(ErrorCode.UNSUPPORTED_CAPABILITY, message, details={"site": name})
            raise _invalid(message, site=name)
        self.total = value[0, start:stop].to(torch.float32).sum(dim=0)

    def finish(self) -> torch.Tensor:
        """The window sum of the forward that just ran, as float64 on the CPU."""
        if self.total is None:
            raise _invalid(f"site {self.site.name!r} did not run in the forward: its module is never called (a "
                           "container, or a module whose weights a kernel reads directly)", site=self.site.name)
        return self.total.cpu().to(torch.float64)


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------


def _windows(texts: Any, *, vocab_size: int, context: int | None, max_tokens: int) -> list[tuple[torch.Tensor, int, int]]:
    """Check every text; return its forward input ``[1, stop]``, ``start``, and ``stop``."""
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence) or not texts:
        raise _invalid("texts must be a non-empty sequence of ProbeText")
    windows = []
    for index, text in enumerate(texts):
        if not isinstance(text, ProbeText):
            raise _invalid(f"text {index} is a {type(text).__name__}, not a ProbeText", text=index)
        tokens = text.tokens
        if not isinstance(tokens, tuple) or not tokens:
            raise _invalid(f"text {index} has no tokens; tokens must be a non-empty sequence of token IDs", text=index)
        if not all(type(token) is int for token in tokens):
            converted = [_integer(token) for token in tokens]
            if any(token is None for token in converted):
                raise _invalid(f"text {index}: every token must be an integer token ID", text=index)
            tokens = tuple(converted)  # type: ignore[arg-type]
        try:
            ids = torch.tensor(tokens, dtype=torch.int64)
        except (OverflowError, RuntimeError):
            raise _invalid(f"text {index}: token IDs must lie in [0, {vocab_size})", text=index) from None
        outside = torch.nonzero((ids < 0) | (ids >= vocab_size)).flatten()
        if outside.numel():
            position = int(outside[0])
            raise _invalid(f"text {index}: token {int(ids[position])} at position {position} lies outside the "
                           f"vocabulary [0, {vocab_size})", text=index, position=position)
        start, stop = _integer(text.start), len(tokens) if text.stop is None else _integer(text.stop)
        if start is None or stop is None or not 0 <= start < stop <= len(tokens):
            raise _invalid(f"text {index}: start and stop must be integers with 0 <= start < stop <= {len(tokens)} "
                           f"(the number of tokens), not start={text.start!r}, stop={text.stop!r}", text=index)
        if stop > max_tokens:
            advice = ("raise max_tokens, up to " + str(MAX_PROBE_TOKENS) if max_tokens < MAX_PROBE_TOKENS
                      else f"the forward is at most {MAX_PROBE_TOKENS} positions")
            raise _invalid(f"text {index}: its forward of {stop} positions exceeds max_tokens {max_tokens}; {advice}, "
                           "or shorten the text (keep fewer tokens before stop, which removes context)",
                           text=index, stop=stop, max_tokens=max_tokens)
        if context is not None and stop > context:
            raise _invalid(f"text {index}: its forward of {stop} positions exceeds the model's context of {context} "
                           "positions", text=index, stop=stop, context=context)
        windows.append((ids[None, :stop], start, stop))
    return windows


def probe(
    model: ModelHandle,
    texts: Sequence[ProbeText],
    *,
    sites: Sequence[str],
    device: str | None = None,
    max_tokens: int = DEFAULT_MAX_PROBE_TOKENS,
    on_forward: Callable[[int, int], None] | None = None,
) -> ProbeResult:
    """Teacher-force each text and average each site's activations over the text's window (module docstring).

    ``device`` is the engine device (``mps`` when available, otherwise
    ``cpu``). Arguments are checked before any forward; a site that captures
    something other than ``[1, stop, features]`` is refused during the first
    forward. ``on_forward(index, stop)`` runs before each text's forward and
    may raise to stop the probe. Errors: ``UNSUPPORTED_CAPABILITY`` for a
    model whose adapter lacks ``probe.hidden_states`` (M1 models),
    ``NOT_FOUND`` for an unknown site (with candidate paths), and
    ``INVALID_ARGUMENT`` for malformed sites, texts, and limits.
    """
    from wse.adapters.base import CAP_PROBE_HIDDEN_STATES
    from wse.api import ModelHandle

    if not isinstance(model, ModelHandle):
        raise _invalid(f"model must be a ModelHandle, not {type(model).__name__}")
    if not model.adapter.capabilities().supports(CAP_PROBE_HIDDEN_STATES):
        raise WSEError(ErrorCode.UNSUPPORTED_CAPABILITY,
                       f"adapter {model.adapter_ref} does not support {CAP_PROBE_HIDDEN_STATES}, so its models cannot "
                       "be probed", details={"adapter": str(model.adapter_ref),
                                             "missing_capabilities": [CAP_PROBE_HIDDEN_STATES]})
    runtime = model.runtime(device)
    limit = _integer(max_tokens)
    if limit is None or not 1 <= limit <= MAX_PROBE_TOKENS:
        raise _invalid(f"max_tokens must be an integer in [1, {MAX_PROBE_TOKENS}], not {max_tokens!r}")
    captures = [_Capture(site) for site in _resolve_sites(model, runtime, sites)]
    vocab_size = int(runtime.module.get_input_embeddings().weight.shape[0])
    windows = _windows(texts, vocab_size=vocab_size, context=runtime.max_positions, max_tokens=limit)

    totals: dict[str, list[torch.Tensor]] = {capture.site.name: [] for capture in captures}
    logprobs: list[float | None] = []
    hooks = []
    exclusive = getattr(runtime, "exclusive", None)
    with exclusive() if callable(exclusive) else contextlib.nullcontext():
        try:
            for capture in captures:
                module = capture.site.module
                hooks.append(module.register_forward_pre_hook(capture.pre_hook) if capture.site.point == "input"
                             else module.register_forward_hook(capture.hook))
            for index, (ids, start, stop) in enumerate(windows):
                if on_forward is not None:
                    on_forward(index, stop)
                for capture in captures:
                    capture.begin(start, stop)
                # The window's tokens are scored from the positions before them; position 0 has no prefix.
                values = runtime.next_token_logprobs(ids, start=max(start, 1) - 1)
                for capture in captures:
                    totals[capture.site.name].append(capture.finish())
                logprobs.append(float(values.to(torch.float64).mean()) if values.numel() else None)
        finally:
            for hook in hooks:
                hook.remove()

    counts = tuple(stop - start for _, start, stop in windows)
    divisor = torch.tensor(counts, dtype=torch.float64)[:, None]
    means = {}
    for name, rows in totals.items():
        array = (torch.stack(rows) / divisor).numpy()
        array.setflags(write=False)
        means[name] = array
    return ProbeResult(sites=tuple(totals), means=means, token_counts=counts, mean_logprobs=tuple(logprobs),
                       model=model.identity, runtime=runtime.info())


# ---------------------------------------------------------------------------
# Managed probes: cutting, result documents, and summaries (module docstring, **Managed probes**)
# ---------------------------------------------------------------------------

# How the summary names the stored vectors: a JSON pointer per vector, with <i> the site's index in the request.
VECTOR_POINTERS = ("<result_uri>#/sites/<i>/contrast/direction, #/sites/<i>/group_means/a and /b, or "
                   "#/sites/<i>/mean without groups; <i> is the site's index in the request")


@dataclass(frozen=True)
class CutText:
    """A managed probe's text after cutting: the forward's tokens, its window ``[start, stop)``, and the tokens dropped."""

    tokens: tuple[int, ...]
    start: int
    stop: int
    dropped: int

    def probe_text(self) -> ProbeText:
        return ProbeText(self.tokens, start=self.start, stop=self.stop)


def cut_text(tokens: Sequence[int], start: int, stop: int, limit: int) -> CutText:
    """Keep the last ``limit`` positions of ``tokens``; the window ``[start, stop)`` keeps its remaining positions.

    The window never holds the first kept position of a cut text, which has
    no context. Raises ``INVALID_ARGUMENT`` when no window position remains
    (a window that lies before the kept positions).
    """
    tokens = tuple(int(token) for token in tokens)
    if not 0 <= start < stop <= len(tokens) or limit < 2:
        raise _invalid(f"a window [{start}, {stop}) of {len(tokens)} tokens, cut to {limit} positions, is not valid")
    dropped = max(0, len(tokens) - limit)
    first = max(start - dropped, 1 if dropped else 0)
    last = stop - dropped
    if first >= last:
        raise _invalid(f"the text's window [{start}, {stop}) lies before its last {limit} positions, which a probe "
                       "reads", start=start, stop=stop, limit=limit)
    return CutText(tokens=tokens[dropped:], start=first, stop=last, dropped=dropped)


def json_number(value: Any) -> float | str:
    """``value`` as a float for canonical JSON, or ``"inf"``, ``"-inf"``, or ``"nan"`` when it is not finite."""
    number = float(value)
    if math.isfinite(number):
        return number
    return "nan" if math.isnan(number) else ("inf" if number > 0 else "-inf")


def json_vector(values: Any) -> list[float | str]:
    """A vector for canonical JSON: its float64 values, with non-finite entries as ``json_number`` writes them."""
    array = np.asarray(values, dtype=np.float64)
    if bool(np.isfinite(array).all()):
        return array.tolist()
    return [json_number(value) for value in array.tolist()]


def decode_vector(values: Sequence[float | str]) -> np.ndarray:
    """The float64 array of a stored vector, ``"inf"``, ``"-inf"``, and ``"nan"`` decoded."""
    special = {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}
    return np.array([special[value] if isinstance(value, str) else value for value in values], dtype=np.float64)


def _short(value: Any) -> float | str | None:
    """A summary number: 6 significant digits, or the string of a non-finite value."""
    if value is None:
        return None
    number = json_number(value)
    return float(f"{number:.6g}") if isinstance(number, float) else number


def probe_document(result: ProbeResult, *, texts: Sequence[Mapping[str, Any]], sites: Sequence[Any],
                   groups: Mapping[str, Sequence[int]] | None, header: Mapping[str, Any]) -> dict[str, Any]:
    """The full result of a managed probe, stored as its artifact (module docstring, **Artifact**).

    ``texts`` holds one record per text (source, references, forward length,
    window, dropped tokens, token hash), to which its mean log-probability is
    added; ``sites`` the requested form of each site of ``result.sites``;
    ``header`` the identifying fields (experiment, job, checkpoint, model,
    runtime, and tokens per text). Every vector is float64.
    """
    if len(texts) != len(result.token_counts) or len(sites) != len(result.sites):
        raise ValueError("one text record per text and one requested site per site are needed")
    records = [{**record, "mean_logprob": None if logprob is None else json_number(logprob)}
               for record, logprob in zip(texts, result.mean_logprobs, strict=True)]
    everything = list(range(len(records)))
    entries = []
    for site, requested in zip(result.sites, sites, strict=True):
        entry: dict[str, Any] = {"site": site, "requested": requested, "features": int(result.means[site].shape[1])}
        if groups is None:
            mean = result.group_mean(site, everything)
            entry.update(mean=json_vector(mean), mean_norm=json_number(np.linalg.norm(mean)))
        else:
            mean_a, mean_b = result.group_mean(site, groups["a"]), result.group_mean(site, groups["b"])
            contrast = result.contrast(site, groups["a"], groups["b"])
            entry.update(
                group_means={"a": json_vector(mean_a), "b": json_vector(mean_b)},
                group_mean_norms={"a": json_number(np.linalg.norm(mean_a)), "b": json_number(np.linalg.norm(mean_b))},
                contrast={"direction": json_vector(contrast.direction), "norm": json_number(contrast.norm),
                          "standardized": json_vector(contrast.standardized),
                          "nonfinite_standardized": int(np.count_nonzero(~np.isfinite(contrast.standardized))),
                          "sizes": list(contrast.sizes)})
        entries.append(entry)
    shown_groups = None if groups is None else {"a": list(groups["a"]), "b": list(groups["b"])}
    return {"schema_version": "1", "kind": "probe_result", **dict(header), "groups": shown_groups,
            "texts": records, "sites": entries}


def top_dimensions(standardized: Any, count: int) -> list[list[Any]]:
    """The ``count`` dimensions of largest absolute standardized difference, as ``[dimension, value]``.

    Infinities come first, then finite values by descending magnitude, then
    NaN; ties go to the lower dimension.
    """
    values = np.asarray(standardized, dtype=np.float64)
    finite, missing = np.isfinite(values), np.isnan(values)
    rank = np.where(missing, 2, np.where(finite, 1, 0))
    magnitude = np.where(finite, -np.abs(np.where(finite, values, 0.0)), 0.0)
    order = np.lexsort((np.arange(values.size), magnitude, rank))[:max(count, 0)]
    return [[int(dimension), _short(values[dimension])] for dimension in order]


def _summary(document: Mapping[str, Any], tops: Sequence[Sequence[list[Any]] | None], top: int, uri: str,
             size: int) -> dict[str, Any]:
    texts = [{"source": record["source"], "tokens": record["forward_tokens"],
              "window": record["window_stop"] - record["window_start"], "dropped": record["dropped_tokens"],
              "logprob": _short(record["mean_logprob"])} for record in document["texts"]]
    sites = []
    for entry, ranked in zip(document["sites"], tops, strict=True):
        item: dict[str, Any] = {"site": entry["site"], "features": entry["features"]}
        if ranked is None:
            item["mean_norm"] = _short(entry["mean_norm"])
        else:
            norms = entry["group_mean_norms"]
            item.update(group_mean_norms=[_short(norms["a"]), _short(norms["b"])],
                        contrast_norm=_short(entry["contrast"]["norm"]), top=[list(pair) for pair in ranked[:top]])
        sites.append(item)
    groups = document["groups"]
    return {"result_uri": uri, "result_bytes": size, "checkpoint_id": document["checkpoint_id"],
            "tokens_per_text": document["tokens_per_text"],
            "teacher_forced_tokens": sum(record["forward_tokens"] for record in document["texts"]),
            "groups": None if groups is None else {"a": len(groups["a"]), "b": len(groups["b"])},
            "top": top if groups is not None else 0, "texts": texts, "sites": sites, "vectors": VECTOR_POINTERS}


def _json_bytes(value: Any) -> int:
    return len(canonical_json(value).encode("utf-8"))


def fit_summary(document: Mapping[str, Any], *, uri: str, size: int, top: int, limit: int) -> dict[str, Any]:
    """The inline summary of a probe's ``document`` with at most ``top`` entries per site, within ``limit`` bytes.

    Every site lists the same number of top dimensions: the largest number,
    at most ``top``, whose summary fits ``limit`` bytes of canonical JSON.
    """
    tops: list[list[list[Any]] | None] = []
    for entry in document["sites"]:
        contrast = entry.get("contrast")
        tops.append(None if contrast is None else top_dimensions(decode_vector(contrast["standardized"]), top))
    low, high = 0, max(0, top)
    if _json_bytes(_summary(document, tops, low, uri, size)) > limit:
        raise WSEError(ErrorCode.INVALID_ARGUMENT,
                       f"the probe's inline summary exceeds the {limit}-byte limit even without top dimensions; "
                       "probe fewer texts or sites", details={"limit_bytes": limit})
    while low < high:  # the largest count that fits
        middle = (low + high + 1) // 2
        if _json_bytes(_summary(document, tops, middle, uri, size)) <= limit:
            low = middle
        else:
            high = middle - 1
    return _summary(document, tops, low, uri, size)


def summary_size_bound(*, texts: int, sites: Sequence[str], grouped: bool, tokens_per_text: int) -> list[int]:
    """Upper bounds on the bytes of a summary without top dimensions: the total for each prefix of ``sites``.

    Element ``k`` bounds the summary of ``texts`` texts and the first ``k``
    sites (so the list has ``len(sites) + 1`` entries); numbers take their
    widest forms. Submission refuses a probe whose bound exceeds the limit,
    so the worker's ``fit_summary`` never fails.
    """
    # The widest summary number (``_short``): 6 significant digits print in positional notation below 1e16, so a
    # value in [1e15, 1e16) takes 18 characters and its negative 19 (-1234560000000000.0); exponents take at most 13.
    wide = -1.23456e15
    record = {"source": "suite_item", "forward_tokens": tokens_per_text, "window_start": 0,
              "window_stop": tokens_per_text, "dropped_tokens": 10 ** 9, "mean_logprob": wide}
    entry: dict[str, Any] = {"group_mean_norms": {"a": wide, "b": wide}, "contrast": {"norm": wide}} if grouped \
        else {"mean_norm": wide}
    document = {"checkpoint_id": "ckpt_" + "0" * 32, "tokens_per_text": tokens_per_text,
                "texts": [record] * texts, "groups": {"a": [0], "b": [0]} if grouped else None, "sites": []}
    base = _json_bytes(_summary(document, [], 0, "wse://analyses/" + "0" * 64, 10 ** 12))
    base += len(str(texts * tokens_per_text)) + 8  # teacher_forced_tokens and the group sizes at their widest
    bounds = [base]
    for site in sites:
        item = _summary({**document, "texts": [], "sites": [{"site": site, "features": 10 ** 9, **entry}]},
                        [[] if grouped else None], 0, "", 0)["sites"][0]
        bounds.append(bounds[-1] + _json_bytes(item) + 1)  # and its separating comma
    return bounds
