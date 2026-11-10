"""Generation profiles, the presence penalty, per-row seeds, and prompt rendering (M2 plan Section 4.2).

* **Profiles.** A ``GenerationProfile`` (``wse.schemas``) is a registered,
  versioned sampling configuration. ``qwen35think@1`` reproduces the
  article's ``qwen35think`` profile (``exp8_math/code/profiles.py`` and
  ``mathqa.py`` of its logs): the chat template with thinking enabled, the
  user message ``problem + "\\n" + BOXED_INSTRUCTION``, temperature 0.6,
  top_p 0.95, top_k 20, presence penalty 1.5, at most 32,768 new tokens,
  seed 0.
* **Default profiles.** Every imported model carries a default profile,
  frozen at import from its ``generation_config.json``
  (``profile_from_generation_config``) and named after the repository
  (``default_profile_id``). Settings the file leaves out take the defaults
  that ``transformers``' ``generate`` applies (temperature 1.0, top_k 50,
  top_p 1.0, greedy). The model's default profile is resolved through the
  model (``resolve_profile``); other profiles through the registry.
* **Presence penalty.** ``transformers`` has none, so ``PresencePenalty``
  implements vLLM's semantics as a logits processor; the engine applies it to
  the raw scores before temperature, top-k, and top-p (``processed_scores``
  is the same composition, for checking).
* **Seeds.** ``sample_seed`` derives each (problem, sample) pair's seed from
  the profile seed, the evaluation seed, the problem ID, and the sample index,
  so a row's draws do not depend on the other rows of its batch.
* **Prompts.** ``render_prompt`` applies the model's chat template with the
  profile's arguments and a fixed clock (``template_clock``): a template that
  prints the current date through ``strftime_now`` (a "Today Date" line, for
  example) prints ``TEMPLATE_DATE`` instead, so one profile renders one
  prompt on every day.

This module imports neither ``transformers`` nor ``math_verify``.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, tzinfo
from typing import TYPE_CHECKING, Any

import torch

from wse.schemas import ErrorCode, ExtensionRef, GenerationProfile, WSEError, canonical_json

if TYPE_CHECKING:
    from wse.engines.hf_torch import Sampling
    from wse.registry import Registry
    from wse.schemas import ModelSource

# The math instruction the Qwen3.5 model card recommends; the article's user message is the problem, a newline, and it.
BOXED_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
BOXED_SUFFIX = "\n" + BOXED_INSTRUCTION

QWEN35THINK = GenerationProfile(
    profile_id="qwen35think",
    version="1",
    description="The article's qwen35think profile for Qwen3.5-0.8B: thinking on, the model card's thinking-mode "
                "sampling, and the boxed-answer math instruction.",
    chat_template=True,
    chat_template_kwargs={"enable_thinking": True},
    user_suffix=BOXED_SUFFIX,
    do_sample=True,
    temperature=0.6,
    top_k=20,
    top_p=0.95,
    presence_penalty=1.5,
    max_new_tokens=32768,
    seed=0,
    source="lm_understand-logs exp8_math/code/profiles.py (qwen35think)",
)
BUILTIN_PROFILES = (QWEN35THINK,)

# The clock a chat template sees through strftime_now: fixed, so a rendered prompt never depends on the day it is
# rendered (the M2 plan's start date; changing it changes every prompt that reads the date).
TEMPLATE_DATE = datetime(2026, 11, 9)

DEFAULT_PROFILE_VERSION = "1"
DEFAULT_PROFILE_SUFFIX = "_default"
# A generation_config.json rarely bounds new tokens; default profiles then allow this many.
DEFAULT_MAX_NEW_TOKENS = 2048
# The defaults transformers 5.22.2 applies when neither the model nor the caller sets a value
# (GenerationConfig._get_default_generation_params).
TRANSFORMERS_DEFAULTS = {"do_sample": False, "temperature": 1.0, "top_k": 50, "top_p": 1.0}
# generation_config.json keys that do not change which tokens are sampled, or that WSE derives elsewhere.
_NEUTRAL_KEYS = {"bos_token_id", "pad_token_id", "eos_token_id", "transformers_version", "_from_model_config",
                 "use_cache", "max_length", "num_return_sequences", "output_scores", "return_dict_in_generate",
                 "decoder_start_token_id", "cache_implementation"}
# Keys that change sampling but match their neutral value here (as transformers' defaults).
_NEUTRAL_VALUES = {"repetition_penalty": 1.0, "num_beams": 1, "typical_p": 1.0, "min_p": None,
                   "no_repeat_ngram_size": 0, "length_penalty": 1.0, "epsilon_cutoff": 0.0, "eta_cutoff": 0.0,
                   "encoder_repetition_penalty": 1.0, "min_length": 0, "min_new_tokens": None,
                   "bad_words_ids": None, "suppress_tokens": None, "begin_suppress_tokens": None}


def default_profile_id(repository: str) -> str:
    """The default profile ID of a repository: ``org/Chat-3.2-1B-Instruct`` gives ``chat_3_2_1b_instruct_default``.

    The repository's name (after the last ``/``) is lowercased, every
    character other than a letter or digit becomes ``_``, and ``m_`` is
    prefixed when it would start with a digit. Long names are shortened so
    the ID, with ``_default``, stays within M1's 64-character extension IDs.
    """
    name = repository.rsplit("/", 1)[-1].lower()
    stem = re.sub(r"[^a-z0-9]", "_", name)
    if not stem or not stem[0].isalpha():
        stem = "m_" + stem
    return stem[: 64 - len(DEFAULT_PROFILE_SUFFIX)] + DEFAULT_PROFILE_SUFFIX


def _token_ids(value: Any) -> list[int] | None:
    if value is None:
        return None
    values = value if isinstance(value, list) else [value]
    if not values or not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in values):
        return None
    return sorted(dict.fromkeys(values))


def profile_from_generation_config(repository: str, generation_config: Mapping[str, Any] | None, *,
                                   has_chat_template: bool, source: str | None = None) -> GenerationProfile:
    """Freeze a model's default profile from its ``generation_config.json`` (``None`` when it has none).

    The prompt is the problem with ``BOXED_SUFFIX``, through the chat template
    when the model has one, because WSE's built-in graders read boxed answers.
    ``top_k`` 0 means no top-k, as in ``transformers``. Settings that change
    sampling but that WSE does not implement (for example a repetition
    penalty) are recorded in ``ignored_settings`` instead of being applied.
    """
    config = dict(generation_config or {})
    settings = {key: config[key] if config.get(key) is not None else default
                for key, default in TRANSFORMERS_DEFAULTS.items()}
    top_k = settings["top_k"]
    max_new_tokens = config.get("max_new_tokens")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
        max_new_tokens = DEFAULT_MAX_NEW_TOKENS
    ignored = {key: value for key, value in sorted(config.items())
               if key not in _NEUTRAL_KEYS and key not in TRANSFORMERS_DEFAULTS and key != "max_new_tokens"
               and not (key in _NEUTRAL_VALUES and value == _NEUTRAL_VALUES[key])}
    temperature = float(settings["temperature"])
    try:
        return GenerationProfile(
            profile_id=default_profile_id(repository),
            version=DEFAULT_PROFILE_VERSION,
            description=f"The default profile of {repository}, frozen from its generation configuration at import.",
            chat_template=has_chat_template,
            user_suffix=BOXED_SUFFIX,
            do_sample=bool(settings["do_sample"]),
            temperature=temperature if temperature > 0 else 1.0,
            top_k=None if not top_k else int(top_k),
            top_p=float(settings["top_p"]),
            max_new_tokens=max_new_tokens,
            seed=0,
            stop_token_ids=_token_ids(config.get("eos_token_id")),
            source=source if generation_config is not None else "transformers defaults",
            ignored_settings=ignored,
        )
    except ValueError as exc:
        raise WSEError(ErrorCode.INVALID_MODEL, f"generation_config.json has settings outside the supported ranges: "
                       f"{exc}") from exc


def resolve_profile(registry: Registry, reference: str | ExtensionRef, source: ModelSource | None) -> GenerationProfile:
    """The exact profile ``reference`` names: the model's frozen default profile, or a registered one."""
    ref = ExtensionRef.parse(reference)
    default = source.default_profile if source is not None else None
    if default is not None and default.ref == ref:
        return default
    if ref.extension_id.endswith(DEFAULT_PROFILE_SUFFIX):
        registered = {str(r) for r in registry.list("profile")}
        if str(ref) not in registered:
            raise WSEError(ErrorCode.NOT_FOUND,
                           f"profile {ref} is not this model's default profile"
                           + (f" ({default.ref})" if default is not None else " (it has none)")
                           + " and is not registered", details={"profile": str(ref)})
    return registry.resolve_ref("profile", ref)


def with_overrides(profile: GenerationProfile, overrides: Mapping[str, Any] | None) -> GenerationProfile:
    """``profile`` with suite-preregistered overrides (for example a smaller ``max_new_tokens``), revalidated."""
    if not overrides:
        return profile
    unknown = sorted(set(overrides) - set(GenerationProfile.model_fields) - {"schema_version"})
    if unknown or {"profile_id", "version"} & set(overrides):
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"cannot override profile fields {sorted(overrides)}")
    try:
        return GenerationProfile.model_validate({**profile.model_dump(mode="json"), **dict(overrides)})
    except ValueError as exc:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"invalid profile overrides: {exc}") from exc


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


class PresencePenalty:
    """Subtract ``penalty`` once from every token that occurs in the generated output (vLLM's presence penalty).

    The processor receives the token IDs so far, ``[batch, prompt_len +
    generated]`` (left-padded prompts share one length), and the float32
    scores of the next token, ``[batch, vocab]``. Prompt tokens are ignored.

    Compared with vLLM's sampler source (main branch, read 2026-11-09):

    * ``vllm/model_executor/layers/utils.py``, ``apply_penalties``: the
      presence penalty subtracts ``presence_penalties * output_mask``, where
      ``output_mask`` marks tokens present in the *output* tokens; prompt
      tokens feed only the repetition penalty. It is therefore subtracted once
      per distinct output token; the per-occurrence term is the frequency
      penalty, which WSE does not implement. This class does the same.
    * ``vllm/v1/worker/gpu/sample/sampler.py``: the order is logit bias, then
      penalties, bad words, thinking-budget forcing, temperature, min-p, then
      top-k and top-p, then sampling. WSE profiles use none of logit bias, bad
      words, thinking budgets, or min-p, so the effective order is penalty,
      temperature, top-k, top-p, as here (``processed_scores``, and the
      engine's ``generate``, which applies processors before
      ``warp_scores``).

    One difference remains: after a row stops, the engine pads it, and the pad
    token then counts as present in that row; finished rows are never
    reported, so this changes no result.
    """

    def __init__(self, penalty: float, prompt_len: int) -> None:
        if not isinstance(penalty, (int, float)) or isinstance(penalty, bool) or not math.isfinite(penalty):
            raise ValueError("the presence penalty must be a finite number")
        if isinstance(prompt_len, bool) or not isinstance(prompt_len, int) or prompt_len < 0:
            raise ValueError("prompt_len must be a nonnegative integer")
        self.penalty, self.prompt_len = float(penalty), prompt_len

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        generated = input_ids[:, self.prompt_len:]  # left-padded batches share one prompt length
        if generated.shape[1] == 0 or self.penalty == 0.0:
            return scores
        present = torch.zeros_like(scores, dtype=torch.bool)
        present.scatter_(1, generated.to(torch.int64), True)
        return scores - self.penalty * present.to(scores.dtype)


def profile_sampling(profile: GenerationProfile) -> Sampling | None:
    """The engine's sampling settings for ``profile``; ``None`` for greedy decoding."""
    from wse.engines.hf_torch import Sampling

    if not profile.do_sample:
        return None
    return Sampling(temperature=profile.temperature, top_k=profile.top_k, top_p=profile.top_p)


def logits_processors(profile: GenerationProfile, prompt_len: int) -> list[PresencePenalty]:
    """The processors the engine applies to raw scores before warping, for a batch of prompt length ``prompt_len``.

    As in vLLM, the penalty applies to greedy decoding too (the argmax is
    taken over the penalized scores).
    """
    if profile.presence_penalty == 0.0:
        return []
    return [PresencePenalty(profile.presence_penalty, prompt_len)]


def processed_scores(input_ids: torch.Tensor, scores: torch.Tensor, profile: GenerationProfile, *,
                     prompt_len: int) -> torch.Tensor:
    """The scores the engine samples from: the profile's processors, then temperature, top-k, and top-p."""
    from wse.engines.hf_torch import warp_scores

    for processor in logits_processors(profile, prompt_len):
        scores = processor(input_ids, scores)
    sampling = profile_sampling(profile)
    return scores if sampling is None else warp_scores(scores, sampling)


def sample_seed(profile_seed: int, eval_seed: int, item_id: str, sample: int) -> int:
    """The seed of one (problem, sample) pair: the first 63 bits of a SHA-256 over its inputs."""
    material = canonical_json({"profile_seed": profile_seed, "eval_seed": eval_seed, "item_id": item_id,
                               "sample": sample})
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big") >> 1


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def user_message(profile: GenerationProfile, problem: str) -> str:
    return problem + profile.user_suffix


class _TemplateClock(datetime):
    """A ``datetime`` whose ``now()`` is ``TEMPLATE_DATE``: the clock chat templates see while WSE renders them."""

    @classmethod
    def now(cls, tz: tzinfo | None = None) -> datetime:  # type: ignore[override]
        return TEMPLATE_DATE if tz is None else TEMPLATE_DATE.replace(tzinfo=tz)


_CLOCK_LOCK = threading.Lock()


@contextmanager
def template_clock(chat_template: str) -> Iterator[None]:
    """Render ``chat_template`` with its ``strftime_now`` reading ``TEMPLATE_DATE`` instead of the current time.

    ``transformers`` gives templates a ``strftime_now`` global that formats
    ``datetime.now()`` of its ``chat_template_utils`` module, and it turns
    every template argument into plain JSON data, so the function cannot be
    passed as an argument. While a template that mentions ``strftime_now``
    renders, that module's ``datetime`` is therefore this module's
    ``_TemplateClock`` (under a lock, restored afterwards). Templates that do
    not mention it render unchanged.
    """
    if "strftime_now" not in chat_template:
        yield
        return
    from transformers.utils import chat_template_utils

    with _CLOCK_LOCK:
        original = getattr(chat_template_utils, "datetime", None)
        if not (isinstance(original, type) and issubclass(original, datetime)):
            raise WSEError(ErrorCode.UNSUPPORTED_CAPABILITY,
                           "the chat template reads the current date, and this transformers version does not let WSE "
                           "fix the date it sees")
        chat_template_utils.datetime = _TemplateClock
        try:
            yield
        finally:
            chat_template_utils.datetime = original


def render_prompt(tokenizer: Any, profile: GenerationProfile, problem: str) -> str:
    """The prompt text: the user message through the model's chat template with the profile's arguments.

    The template sees ``TEMPLATE_DATE`` as the current date
    (``template_clock``), so a template that prints today's date renders the
    same prompt on every day; a profile may still set a template variable
    such as ``date_string`` through ``chat_template_kwargs``.
    """
    content = user_message(profile, problem)
    if not profile.chat_template:
        return content
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        raise WSEError(ErrorCode.UNSUPPORTED_CAPABILITY,
                       f"profile {profile.ref} uses a chat template, but the model has none")
    messages = ([{"role": "system", "content": profile.system_prompt}] if profile.system_prompt is not None else []) \
        + [{"role": "user", "content": content}]
    with template_clock(template if isinstance(template, str) else str(template)):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             **profile.chat_template_kwargs)


__all__ = ["BOXED_INSTRUCTION", "BOXED_SUFFIX", "BUILTIN_PROFILES", "DEFAULT_MAX_NEW_TOKENS", "PresencePenalty",
           "QWEN35THINK", "TEMPLATE_DATE", "default_profile_id", "logits_processors", "processed_scores",
           "profile_from_generation_config", "profile_sampling", "render_prompt", "resolve_profile", "sample_seed",
           "template_clock", "user_message", "with_overrides"]
