"""The PyTorch engine for causal LMs imported through ``hf_causal_lm@1`` (M2 plan Sections 2.1 and 4.1).

* **Construction.** The ``transformers`` module is built from the model's
  configuration on the meta device, so no weight memory is allocated. Its
  non-persistent buffers (rotary frequencies, embedding scales) are created on
  the target device and initialized by the model's own initializer, as
  ``from_pretrained`` does, before any weight is attached; the initializer
  therefore never touches a stored tensor. The stored WSE tensors are then
  attached with ``load_state_dict(..., assign=True)`` under the module's
  parameter names: on CPU the module's parameters are the stored tensors
  themselves, and on another device each distinct tensor is copied once.
  Modules run in the stored dtypes.
* **Devices.** Selection is explicit: ``mps`` when available, otherwise
  ``cpu``, or the device a caller names. Bitwise reproducibility holds within
  one device and library version, not across them (M1 plan Section 7.3).
* **Forward.** ``logits`` returns float32 next-token logits for every
  position of a batch of token IDs.
* **Positions.** Position IDs come from the attention mask, as
  ``transformers``' ``generate`` derives them: the cumulative sum of the mask
  minus one, with masked positions set to 0, at prefill (and in ``logits``),
  then the last position plus one at each decode step. A left-padded row
  therefore computes the logits of the same prompt run alone (M2 plan
  Section 4.1); counting from the first pad would shift it by its padding.
* **Generation.** ``generate`` decodes a batch with a KV cache: the prompt is
  prefilled once, then each step feeds one token per row. Greedy decoding
  takes the first maximal logit, as ``transformers``' ``generate`` does.
  Sampling applies the caller's logits processors, then temperature, top-k,
  and top-p, and draws each row from its own seeded CPU generator, so a row's
  tokens depend on its seed and not on its batch position. Rows that emit a
  stop token finish; later positions are padding and are not reported.

In managed experiments only the worker runs this engine; a run-bound engine
never obtains a model handle (M1 plan Section 2). ``transformers`` is imported
lazily, so this module imports without the ``models-hf`` extra.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from wse.schemas import ErrorCode, WSEError

if TYPE_CHECKING:
    from wse.adapters.families import FamilyDescriptor

DEVICES = ("mps", "cpu")
LogitsProcessor = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _invalid_argument(message: str, **details: Any) -> WSEError:
    return WSEError(ErrorCode.INVALID_ARGUMENT, message, details=details)


def mps_available() -> bool:
    return bool(torch.backends.mps.is_available())


def select_device(device: str | torch.device | None = None) -> torch.device:
    """``mps`` when available, otherwise ``cpu``; or the named device, which must be available."""
    if device is None:
        return torch.device("mps" if mps_available() else "cpu")
    name = device.type if isinstance(device, torch.device) else device
    if name not in DEVICES:
        raise _invalid_argument(f"device must be one of {list(DEVICES)}, not {name!r}", device=str(name))
    if name == "mps" and not mps_available():
        raise WSEError(ErrorCode.UNSUPPORTED_CAPABILITY, "PyTorch MPS is not available on this host; use 'cpu'")
    return torch.device(name)


@dataclass(frozen=True)
class Sampling:
    """Sampling settings: temperature, then top-k (``None`` keeps all), then top-p."""

    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.temperature) and self.temperature > 0):
            raise ValueError("temperature must be a finite number above 0")
        if self.top_k is not None and (isinstance(self.top_k, bool) or not isinstance(self.top_k, int)
                                       or self.top_k < 1):
            raise ValueError("top_k must be a positive integer or None")
        if not (math.isfinite(self.top_p) and 0 < self.top_p <= 1):
            raise ValueError("top_p must lie in (0, 1]")


@dataclass(frozen=True)
class GenerationResult:
    """Generated token IDs per row, up to and including a stop token.

    ``finished[i]`` is true when row ``i`` emitted a stop token, and false
    when it reached ``max_new_tokens``.
    """

    tokens: list[list[int]]
    finished: list[bool]

    @property
    def generated_tokens(self) -> list[int]:
        return [len(row) for row in self.tokens]


def _transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise WSEError(
            ErrorCode.UNSUPPORTED_CAPABILITY,
            "running hf_causal_lm@1 models needs the models-hf extra (transformers); install it with "
            "'uv sync --extra models-hf'",
        ) from exc
    return transformers


def _dtype(tensors: Mapping[str, torch.Tensor]) -> torch.dtype:
    """The dtype that holds most elements: the module's nominal dtype."""
    counts: dict[torch.dtype, int] = {}
    for tensor in tensors.values():
        counts[tensor.dtype] = counts.get(tensor.dtype, 0) + tensor.numel()
    return max(sorted(counts, key=str), key=lambda dtype: counts[dtype]) if counts else torch.float32


def _owner(module: torch.nn.Module, name: str) -> tuple[torch.nn.Module, str]:
    prefix, _, local = name.rpartition(".")
    return (module.get_submodule(prefix) if prefix else module), local


def build_model(
    model_config: Mapping[str, Any],
    descriptor: FamilyDescriptor | None,
    tensors: Mapping[str, torch.Tensor],
    *,
    device: str | torch.device | None = None,
) -> HFTorchModel:
    """Build the causal LM for ``model_config`` and attach ``tensors`` (module parameter name -> tensor).

    Every parameter of the module must be supplied, and nothing else.
    """
    _transformers()
    from wse.adapters.hf_causal_lm import causal_lm_class

    target = select_device(device)
    model_class, config = causal_lm_class(dict(model_config), descriptor)
    with torch.device("meta"):
        module = model_class._from_config(config, dtype=_dtype(tensors))
    module.requires_grad_(False)
    module.eval()

    # Non-persistent buffers: materialize on the target device, then let the model's initializer fill them
    # while every parameter is still on the meta device (where initialization is a no-op).
    for name, buffer in list(module.named_buffers()):
        owner, local = _owner(module, name)
        if buffer.is_meta and local in owner._non_persistent_buffers_set:
            owner._buffers[local] = torch.empty_like(buffer, device=target)
    module.initialize_weights()

    expected = set(module.state_dict(keep_vars=True))
    missing, unexpected = sorted(expected - set(tensors)), sorted(set(tensors) - expected)
    if missing or unexpected:
        raise WSEError(ErrorCode.INVALID_MODEL,
                       f"tensors do not match the parameters of {model_class.__name__}"
                       + (f"; missing: {missing[:20]}" if missing else "")
                       + (f"; unexpected: {unexpected[:20]}" if unexpected else ""),
                       details={"missing": missing[:20], "unexpected": unexpected[:20]})
    moved: dict[tuple[int, torch.dtype, tuple[int, ...]], torch.Tensor] = {}
    state = {}
    for name, tensor in tensors.items():
        key = (tensor.untyped_storage().data_ptr() + tensor.storage_offset() * tensor.element_size(),
               tensor.dtype, tuple(tensor.shape))
        if key not in moved:  # one device copy per distinct tensor; on CPU the tensor itself
            moved[key] = tensor.detach().to(target)
        state[name] = moved[key]
    module.load_state_dict(state, strict=True, assign=True)
    module.requires_grad_(False)
    leftover = sorted(name for name, value in [*module.named_parameters(), *module.named_buffers()]
                      if value.is_meta)
    if leftover:
        raise WSEError(ErrorCode.INVALID_MODEL, f"{model_class.__name__} keeps uninitialized tensors after "
                       f"loading: {leftover[:20]}", details={"tensors": leftover[:20]})
    return HFTorchModel(module, target)


@dataclass
class HFTorchModel:
    """A ``transformers`` causal LM bound to stored WSE tensors on one device."""

    module: Any
    device: torch.device
    _vocab_size: int = field(init=False)
    _logits_to_keep: bool = field(init=False)

    def __post_init__(self) -> None:
        import inspect

        self._vocab_size = int(self.module.get_input_embeddings().weight.shape[0])
        parameters = inspect.signature(self.module.forward).parameters
        self._logits_to_keep = "logits_to_keep" in parameters
        self._position_ids = "position_ids" in parameters

    def info(self) -> dict[str, Any]:
        """Device, stored dtypes, model class, attention implementation, and library versions."""
        transformers = _transformers()
        dtypes = sorted({str(p.dtype).removeprefix("torch.") for p in self.module.parameters()})
        return {"device": self.device.type, "dtypes": dtypes, "model_class": type(self.module).__name__,
                "attn_implementation": self.module.config._attn_implementation, "torch": torch.__version__,
                "transformers": transformers.__version__}

    # -- inputs ---------------------------------------------------------------

    def _check(self, input_ids: Any, attention_mask: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.int64 or input_ids.dim() != 2 \
                or 0 in input_ids.shape:
            raise _invalid_argument("input_ids must be a non-empty int64 tensor of shape [batch, sequence]")
        if int(input_ids.min()) < 0 or int(input_ids.max()) >= self._vocab_size:
            raise _invalid_argument(f"token IDs must lie in [0, {self._vocab_size})")
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape \
                    or attention_mask.dtype not in (torch.int64, torch.bool):
                raise _invalid_argument("attention_mask must be an int64 or bool tensor shaped like input_ids")
            attention_mask = attention_mask.to(torch.int64)
            if not bool(((attention_mask == 0) | (attention_mask == 1)).all()) or not bool(attention_mask.any(1).all()):
                raise _invalid_argument("attention_mask must hold 0 and 1 and attend to at least one token per row")
        return input_ids.cpu(), None if attention_mask is None else attention_mask.cpu()

    def _forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None, *, last_only: bool,
                 position_ids: torch.Tensor | None = None, cache: Any = None, use_cache: bool = False) -> Any:
        kwargs: dict[str, Any] = {"input_ids": input_ids.to(self.device), "use_cache": use_cache}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask.to(self.device)
        if position_ids is not None and self._position_ids:
            kwargs["position_ids"] = position_ids.to(self.device)
        if cache is not None:
            kwargs["past_key_values"] = cache
        if last_only and self._logits_to_keep:
            kwargs["logits_to_keep"] = 1
        return self.module(**kwargs, return_dict=True)

    # -- forward --------------------------------------------------------------

    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Float32 logits ``[batch, sequence, vocab]`` on the CPU."""
        input_ids, attention_mask = self._check(input_ids, attention_mask)
        positions = None if attention_mask is None else positions_from_mask(attention_mask)
        with torch.inference_mode():
            output = self._forward(input_ids, attention_mask, last_only=False, position_ids=positions)
            return output.logits.to(device="cpu", dtype=torch.float32, copy=True)

    # -- generation -----------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        max_new_tokens: int,
        stop_token_ids: Sequence[int] = (),
        pad_token_id: int = 0,
        sampling: Sampling | None = None,
        seeds: Sequence[int] | None = None,
        logits_processors: Sequence[LogitsProcessor] = (),
    ) -> GenerationResult:
        """Decode up to ``max_new_tokens`` per row with a KV cache; greedy unless ``sampling`` is given.

        Prompts in a batch are left-padded and masked by ``attention_mask``.
        Sampling needs one seed per row. Each logits processor receives the
        token IDs so far (prompt and output, ``[batch, length]``) and float32
        scores ``[batch, vocab]`` and returns new scores.
        """
        input_ids, attention_mask = self._check(input_ids, attention_mask)
        batch = input_ids.shape[0]
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise _invalid_argument("max_new_tokens must be a positive integer")
        stops = [int(token) for token in stop_token_ids]
        if any(not 0 <= token < self._vocab_size for token in [*stops, pad_token_id]):
            raise _invalid_argument(f"stop and pad token IDs must lie in [0, {self._vocab_size})")
        if sampling is None and seeds is not None:
            raise _invalid_argument("seeds apply only to sampling; greedy decoding takes none")
        generators: list[torch.Generator] = []
        if sampling is not None:
            if seeds is None or len(seeds) != batch:
                raise _invalid_argument(f"sampling needs one seed per row ({batch})")
            generators = [torch.Generator(device="cpu").manual_seed(int(seed)) for seed in seeds]
        mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
        stop_ids = torch.tensor(stops, dtype=torch.int64)
        sequences = input_ids
        unfinished = torch.ones(batch, dtype=torch.bool)
        tokens: list[list[int]] = [[] for _ in range(batch)]
        finished = [False] * batch
        step_ids, cache = input_ids, None
        positions = positions_from_mask(mask)  # the prompt's positions at prefill; then one per step
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                output = self._forward(step_ids, mask, last_only=True, position_ids=positions, cache=cache,
                                       use_cache=True)
                cache = output.past_key_values
                scores = output.logits[:, -1].to(device="cpu", dtype=torch.float32, copy=True)
                for processor in logits_processors:
                    scores = processor(sequences, scores)
                if sampling is None:
                    chosen = torch.argmax(scores, dim=-1)
                else:
                    chosen = _sample(scores, sampling, generators)
                chosen = torch.where(unfinished, chosen, torch.full_like(chosen, pad_token_id))
                for row in torch.nonzero(unfinished).flatten().tolist():
                    tokens[row].append(int(chosen[row]))
                stopped = torch.isin(chosen, stop_ids) & unfinished
                for row in torch.nonzero(stopped).flatten().tolist():
                    finished[row] = True
                unfinished &= ~stopped
                sequences = torch.cat([sequences, chosen[:, None]], dim=1)
                mask = torch.cat([mask, torch.ones((batch, 1), dtype=mask.dtype)], dim=1)
                if not bool(unfinished.any()):
                    break
                step_ids = chosen[:, None]
                positions = positions[:, -1:] + 1
        return GenerationResult(tokens=tokens, finished=finished)


def positions_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Position IDs ``[batch, sequence]`` from a 0/1 attention mask, as ``transformers``' ``generate`` derives them.

    Each attended token's position counts the attended tokens before it
    (the cumulative sum minus one); masked positions get 0.
    """
    mask = mask.to(torch.int64)
    return (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)


def _sample(scores: torch.Tensor, sampling: Sampling, generators: Sequence[torch.Generator]) -> torch.Tensor:
    """Temperature, top-k, and top-p (as ``transformers``' warpers), then one draw per row from its generator."""
    scores = scores / sampling.temperature
    if sampling.top_k is not None and sampling.top_k < scores.shape[-1]:
        kth = torch.topk(scores, sampling.top_k, dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < kth, float("-inf"))
    if sampling.top_p < 1.0:
        ordered, order = torch.sort(scores, descending=False, dim=-1)
        cumulative = ordered.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative <= (1 - sampling.top_p)
        remove[..., -1:] = False  # always keep the most likely token
        scores = scores.masked_fill(remove.scatter(1, order, remove), float("-inf"))
    probabilities = scores.softmax(dim=-1)
    return torch.stack([torch.multinomial(probabilities[row], 1, generator=generators[row])[0]
                        for row in range(scores.shape[0])])
