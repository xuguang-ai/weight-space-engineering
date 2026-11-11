"""Activation probes: residual and module sites, window means, contrasts, and log-probabilities (M2 plan Section 5.3).

``wse.probes`` teacher-forces each text through the engine runtime in one
forward pass (batch 1, no cache) and reduces activations to per-text window
sums inside forward hooks. These tests compare every probe output with an
independent computation on the same runtime module: ``output_hidden_states``,
a hook of the test's own on another module, or the module's arithmetic
itself. Snapshots are tiny random checkpoints built offline by
``tests/hf_fixtures.py`` (hidden size 16, two decoder layers, vocabulary 64,
context 128); every test runs on the CPU.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from wse.api import Engine, load_model
from wse.probes import (
    DEFAULT_MAX_PROBE_TOKENS,
    MAX_PROBE_SITES,
    MAX_PROBE_TOKENS,
    MAX_SITE_CANDIDATES,
    Contrast,
    ProbeResult,
    ProbeText,
    module_paths,
    probe,
)
from wse.schemas import ErrorCode, WSEError
from wse.tensors import hash_weights, weights_hash_scheme

REPO = "example-org/tiny-model"
REVISION = "0123456789abcdef0123456789abcdef01234567"
MODELS = ("llama_untied", "qwen3", "gemma3", "gpt2", "qwen3_5")  # every fixture family
LAYERS = 2  # decoder layers in every fixture
VOCAB = 64
CONTEXT = 128
FINAL_NORMS = {"gpt2": "transformer.ln_f"}  # runtime path of the final norm; the other fixtures use model.norm
# Canonical prefix of the decoder stack: Qwen3.5 checkpoints name the language model model.language_model.*.
PREFIXES = {"qwen3_5": "model.language_model."}


def _expect(code: ErrorCode, call, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        call(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


@pytest.fixture(scope="module")
def handles(hf_snapshots, tmp_path_factory):
    """Every fixture family imported into one workspace and opened as an in-memory handle."""
    workspace = tmp_path_factory.mktemp("probes") / "workspace"
    with Engine(workspace) as engine:
        ids = {name: engine.import_hf_snapshot(hf_snapshots[name], repo=REPO, revision=REVISION).checkpoint_id
               for name in MODELS}
        return {name: engine.open_checkpoint(checkpoint_id) for name, checkpoint_id in ids.items()}


def _tokens(seed: int, length: int) -> tuple[int, ...]:
    generator = torch.Generator().manual_seed(seed)
    return tuple(torch.randint(3, VOCAB, (length,), generator=generator).tolist())


def _stop(text: ProbeText) -> int:
    return len(text.tokens) if text.stop is None else text.stop


def _documented_mean(full: torch.Tensor, start: int, stop: int) -> np.ndarray:
    """The documented reduction of a ``[1, sequence, dim]`` capture: a float32 sum over the window, divided in float64."""
    total = full[0, start:stop].to(torch.float32).sum(dim=0)
    return total.to(torch.float64).numpy() / (stop - start)


def _float64_mean(full: torch.Tensor, start: int, stop: int) -> np.ndarray:
    return full[0, start:stop].to(torch.float64).mean(dim=0).numpy()


def _capture_input(runtime, path: str, tokens) -> torch.Tensor:
    """The full first positional input of runtime module ``path`` in a forward of ``tokens``, by a hook of the test's own."""
    seen: dict[str, torch.Tensor] = {}
    hook = runtime.module.get_submodule(path).register_forward_pre_hook(
        lambda module, args: seen.setdefault("input", args[0]))
    try:
        runtime.logits(torch.tensor([list(tokens)]))
    finally:
        hook.remove()
    return seen["input"]


def _hooks(module: torch.nn.Module) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    return {name: (tuple(sub._forward_hooks), tuple(sub._forward_pre_hooks)) for name, sub in module.named_modules()}


# -- residual sites --------------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODELS)
def test_probe_hidden_states_match_a_direct_forward_on_the_fixture(name, handles):
    handle = handles[name]
    runtime = handle.runtime("cpu")
    texts = [ProbeText(_tokens(0, 13)), ProbeText(_tokens(1, 13), start=4, stop=10)]
    sites = [f"residual:{index}" for index in range(LAYERS + 1)]
    result = probe(handle, texts, sites=sites, device="cpu")
    assert isinstance(result, ProbeResult) and result.sites == tuple(sites) and result.token_counts == (13, 6)
    assert result.model == handle.identity and dict(result.runtime) == runtime.info()
    norm = runtime.module.get_submodule(FINAL_NORMS.get(name, "model.norm"))
    for row, text in enumerate(texts):
        stop = _stop(text)
        seen: dict[str, torch.Tensor] = {}
        hook = norm.register_forward_pre_hook(lambda module, args: seen.setdefault("input", args[0]))
        try:
            with torch.inference_mode():
                states = runtime.module(input_ids=torch.tensor([text.tokens[:stop]]), use_cache=False,
                                        output_hidden_states=True).hidden_states
                normed = norm(seen["input"])
        finally:
            hook.remove()
        # transformers returns L + 1 states, but its last one is the final norm's output (last_hidden_state),
        # not the residual stream after the last layer, which is the final norm's input.
        assert len(states) == LAYERS + 1
        assert torch.equal(states[LAYERS], normed) and not torch.equal(states[LAYERS], seen["input"])
        for index, state in enumerate([*states[:LAYERS], seen["input"]]):
            actual = result.means[f"residual:{index}"][row]
            assert actual.dtype == np.float64 and actual.shape == (16,)
            np.testing.assert_array_equal(actual, _documented_mean(state, text.start, stop))
            np.testing.assert_allclose(actual, _float64_mean(state, text.start, stop), rtol=1e-6, atol=1e-7)
    # Results are immutable.
    with pytest.raises(TypeError):
        result.means["residual:0"] = np.zeros((2, 16))  # type: ignore[index]
    with pytest.raises(ValueError):
        result.means["residual:0"][0, 0] = 1.0


# -- module sites ----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ("llama_untied", "qwen3_5"))
def test_probe_role_sites_capture_the_input_of_mlp_down(name, handles):
    handle = handles[name]
    runtime = handle.runtime("cpu")
    prefix = PREFIXES.get(name, "model.")
    down, mlp_site = f"{prefix}layers.1.mlp.down_proj", f"{prefix}layers.1.mlp"
    sites = [f"{down}:input", f"{down}:output", f"{mlp_site}:input"]
    texts = [ProbeText(_tokens(2, 12)), ProbeText(_tokens(3, 12), start=5)]
    result = probe(handle, texts, sites=sites, device="cpu")
    mlp = runtime.module.get_submodule("model.layers.1.mlp")  # the runtime module of both families
    weight = mlp.down_proj.weight.to(torch.float64).numpy()
    for row, text in enumerate(texts):
        x = _capture_input(runtime, "model.layers.1.mlp", text.tokens)
        with torch.inference_mode():
            hidden = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)  # the input of down_proj, computed independently
            output = torch.nn.functional.linear(hidden, mlp.down_proj.weight)  # that input times the weight transposed
        for site, full in ((f"{down}:input", hidden), (f"{down}:output", output), (f"{mlp_site}:input", x)):
            np.testing.assert_array_equal(result.means[site][row], _documented_mean(full, text.start, 12))
        # The mean commutes with the linear map, up to the rounding of each BF16 output element to its stored
        # dtype (at most half a unit in the last place, 2^-9 of its magnitude); the bound allows twice that.
        bound = 2.0 ** -8 * float(output[0, text.start:].abs().max())
        np.testing.assert_allclose(result.means[f"{down}:output"][row], result.means[f"{down}:input"][row] @ weight.T,
                                   rtol=0, atol=bound)
    assert result.means[f"{down}:input"].shape == (2, 32) and result.means[f"{down}:output"].shape == (2, 16)
    assert result.token_counts == (12, 7)


def test_sites_use_canonical_names_and_map_to_runtime_modules_on_qwen3_5(handles):
    handle = handles["qwen3_5"]
    runtime = handle.runtime("cpu")
    paths = module_paths(handle)
    assert paths["model.language_model.layers.1.mlp.down_proj"] == "model.layers.1.mlp.down_proj"
    assert paths["model.language_model.layers.1"] == "model.layers.1"
    assert paths["model.language_model.norm"] == "model.norm" and paths["lm_head"] == "lm_head"
    assert not any(path.startswith(("model.visual", "mtp", "model.layers")) for path in paths)
    assert all(isinstance(runtime.module.get_submodule(module), torch.nn.Module) for module in paths.values())
    # At Level 0 canonical names are the module names.
    level0 = module_paths(handles["llama_untied"])
    assert "model.layers.1.mlp.down_proj" in level0 and all(c == m for c, m in level0.items())
    # A canonical site reads the runtime module it maps to.
    tokens = _tokens(4, 10)
    result = probe(handle, [ProbeText(tokens, start=1)], sites=["model.language_model.layers.1.mlp.down_proj:input"],
                   device="cpu")
    expected = _documented_mean(_capture_input(runtime, "model.layers.1.mlp.down_proj", tokens), 1, 10)
    np.testing.assert_array_equal(result.means["model.language_model.layers.1.mlp.down_proj:input"][0], expected)
    # A runtime module name is not a site name: NOT_FOUND, with the canonical path among bounded candidates.
    error = _expect(ErrorCode.NOT_FOUND, probe, handle, [ProbeText(tokens)],
                    sites=["model.layers.1.mlp.down_proj:input"], device="cpu")
    assert "model.language_model.layers.1.mlp.down_proj" in error.details["candidates"]
    assert 0 < len(error.details["candidates"]) <= MAX_SITE_CANDIDATES


# -- causality and log-probabilities ----------------------------------------------------------------


@pytest.mark.parametrize("name", ("llama_untied", "qwen3_5"))
def test_tokens_at_or_after_stop_change_nothing(name, handles):
    handle = handles[name]
    prefix = PREFIXES.get(name, "model.")
    sites = ["residual:0", f"residual:{LAYERS}", f"{prefix}layers.0.mlp.down_proj:input"]
    tokens = _tokens(5, 16)
    base = probe(handle, [ProbeText(tokens, start=2, stop=9)], sites=sites, device="cpu")
    changed = tokens[:9] + tuple((token + 1) % VOCAB for token in tokens[9:])  # every token from stop on
    longer = tokens[:9] + _tokens(6, 30)
    for other in (changed, longer):
        again = probe(handle, [ProbeText(other, start=2, stop=9)], sites=sites, device="cpu")
        assert again.token_counts == base.token_counts == (7,)
        assert again.mean_logprobs == base.mean_logprobs
        for site in sites:
            np.testing.assert_array_equal(again.means[site], base.means[site])
    # A change before stop does change the probe, so the check above is not vacuous.
    earlier = tokens[:8] + ((tokens[8] + 1) % VOCAB,) + tokens[9:]
    moved = probe(handle, [ProbeText(earlier, start=2, stop=9)], sites=sites, device="cpu")
    assert moved.mean_logprobs != base.mean_logprobs
    assert not np.array_equal(moved.means[f"residual:{LAYERS}"], base.means[f"residual:{LAYERS}"])


@pytest.mark.parametrize("name", ("llama_untied", "gemma3", "qwen3_5"))
def test_mean_logprobs_agree_with_log_softmax_of_the_engine_logits(name, handles, monkeypatch):
    from wse.engines import hf_torch

    handle = handles[name]
    runtime = handle.runtime("cpu")
    tokens = _tokens(7, 14)
    texts = [ProbeText(tokens), ProbeText(tokens, start=3, stop=11), ProbeText(tokens, start=10, stop=11),
             ProbeText(tokens, start=13), ProbeText(tokens, start=0, stop=1)]
    result = probe(handle, texts, sites=["residual:1"], device="cpu")
    for text, actual in zip(texts, result.mean_logprobs, strict=True):
        stop = _stop(text)
        logprobs = torch.log_softmax(runtime.logits(torch.tensor([tokens[:stop]]))[0], dim=-1)
        targets = range(max(text.start, 1), stop)  # the window's tokens, each scored from the position before it
        if not targets:
            assert actual is None
            continue
        expected = torch.stack([logprobs[t - 1, tokens[t]] for t in targets]).to(torch.float64).mean().item()
        assert isinstance(actual, float) and actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
    # A one-token window scores its token from the position before it; only position 0 has no prefix.
    assert result.mean_logprobs[2] is not None and result.mean_logprobs[3] is not None
    assert result.mean_logprobs[4] is None
    # The engine's teacher-forced read returns the same per-token values, bitwise, in chunks of any size.
    ids = torch.tensor([tokens])
    logprobs = torch.log_softmax(runtime.logits(ids)[0], dim=-1)
    expected = torch.stack([logprobs[t, tokens[t + 1]] for t in range(3, 13)])
    for chunk in (hf_torch.LOGPROB_CHUNK, 4):
        monkeypatch.setattr(hf_torch, "LOGPROB_CHUNK", chunk)
        read = runtime.next_token_logprobs(ids, start=3)
        assert read.dtype == torch.float32 and read.device.type == "cpu" and torch.equal(read, expected)
    assert runtime.next_token_logprobs(ids, start=13).shape == (0,)
    for kwargs in (dict(start=14), dict(start=-1), dict(start=True)):
        _expect(ErrorCode.INVALID_ARGUMENT, runtime.next_token_logprobs, ids, **kwargs)
    _expect(ErrorCode.INVALID_ARGUMENT, runtime.next_token_logprobs, ids.repeat(2, 1))
    _expect(ErrorCode.INVALID_ARGUMENT, runtime.next_token_logprobs, ids + VOCAB)


# -- contrasts -------------------------------------------------------------------------------------


def test_the_contrast_of_identical_texts_is_exactly_zero():
    from wse.probes import ProbeResult
    from wse.schemas import ModelIdentity

    row = np.array([0.1, 0.2, 0.30000000000000004, -7.123456789, 1e-9])
    identity = ModelIdentity(adapter={"extension_id": "mlp", "version": "1"}, config_hash="sha256:" + "0" * 64,
                             weights_hash="sha256:" + "1" * 64)
    for n_a, n_b in ((3, 2), (5, 7), (2, 1)):
        means = np.tile(row, (n_a + n_b, 1))
        result = ProbeResult(sites=("s",), means={"s": means}, token_counts=(1,) * (n_a + n_b),
                             mean_logprobs=(None,) * (n_a + n_b), model=identity, runtime={})
        contrast = result.contrast("s", list(range(n_a)), list(range(n_a, n_a + n_b)))
        assert np.array_equal(contrast.direction, np.zeros_like(row)) and contrast.norm == 0.0
        assert np.array_equal(contrast.standardized, np.zeros_like(row))
        assert np.array_equal(result.group_mean("s", list(range(n_a))), row)


def test_contrast_direction_is_the_difference_of_group_means(handles):
    handle = handles["llama_untied"]
    site = "model.layers.0.mlp.down_proj:input"
    texts = [ProbeText(_tokens(10 + i, 9 + i), start=i) for i in range(5)]
    result = probe(handle, texts, sites=["residual:1", site], device="cpu")
    means = result.means[site]
    a, b = means[[0, 3]], means[[1, 2, 4]]
    difference = a.mean(axis=0) - b.mean(axis=0)
    np.testing.assert_allclose(result.group_mean(site, [0, 3]), a.mean(axis=0), rtol=1e-12, atol=0)
    contrast = result.contrast(site, [0, 3], [1, 2, 4])
    assert isinstance(contrast, Contrast) and contrast.site == site and contrast.sizes == (2, 3)
    np.testing.assert_allclose(contrast.direction, difference, rtol=1e-12, atol=1e-15)
    assert contrast.norm == pytest.approx(float(np.linalg.norm(difference)), rel=1e-12)
    # Standardized: the difference over the pooled per-text standard deviation, (n_a - 1) + (n_b - 1) = 3 degrees of freedom.
    pooled = np.sqrt((((a - a.mean(axis=0)) ** 2).sum(axis=0) + ((b - b.mean(axis=0)) ** 2).sum(axis=0)) / 3)
    np.testing.assert_allclose(contrast.standardized, difference / pooled, rtol=1e-12)
    for value in (contrast.direction, contrast.standardized):
        assert value.dtype == np.float64 and value.shape == (32,)
    # Texts weigh equally, whatever their token counts; swapping the groups negates the direction.
    np.testing.assert_allclose(result.contrast(site, [1, 2, 4], [0, 3]).direction, -difference, rtol=1e-12,
                               atol=1e-15)

    # Zero pooled variance: 0 where the groups agree, a signed infinity where they differ.
    synthetic = dataclasses.replace(result, sites=("s",), means={"s": np.array(
        [[1.0, 2.0, 0.5], [1.0, 2.0, 0.7], [1.0, 3.0, 0.2]])}, token_counts=(1, 1, 1), mean_logprobs=(None,) * 3)
    degenerate = synthetic.contrast("s", [0, 1], [2])
    np.testing.assert_allclose(degenerate.direction, [0.0, -1.0, 0.4], rtol=1e-12, atol=1e-15)
    # Dimension 0 is constant (no difference), dimension 1 is constant within each group but differs between them,
    # and dimension 2 pools (0.5 - 0.6)^2 + (0.7 - 0.6)^2 over 2 + 1 - 2 = 1 degree of freedom.
    np.testing.assert_array_equal(degenerate.standardized[:2], [0.0, -np.inf])
    assert degenerate.standardized[2] == pytest.approx(0.4 / np.sqrt(0.02), rel=1e-12)

    for group_a, group_b in (([0], [1]), ([], [1, 2]), ([0, 1], [1, 2]), ([0, 0], [1, 2]), ([0, 5], [1]),
                             ([0, -1], [1]), ([True, 2], [3]), ([0.0, 2], [3]), ("01", [3])):
        _expect(ErrorCode.INVALID_ARGUMENT, result.contrast, site, group_a, group_b)
    _expect(ErrorCode.INVALID_ARGUMENT, result.group_mean, site, [])
    _expect(ErrorCode.NOT_FOUND, result.contrast, "residual:2", [0, 1], [2])
    _expect(ErrorCode.NOT_FOUND, result.group_mean, "residual:2", [0])
    # Results hold arrays, so they compare by identity (an element-wise comparison would be ambiguous).
    assert result == result and result != dataclasses.replace(result) and len({result, contrast}) == 2


# -- refusals --------------------------------------------------------------------------------------


def test_probe_refuses_unknown_and_malformed_sites(handles):
    handle = handles["qwen3"]
    texts = [ProbeText(_tokens(9, 8))]

    def refused(code: ErrorCode, sites) -> WSEError:
        return _expect(code, probe, handle, texts, sites=sites, device="cpu")

    # Unknown sites: NOT_FOUND with a bounded list of candidates.
    error = refused(ErrorCode.NOT_FOUND, ["model.layers.1.mlp.down_projj:input"])
    assert error.details["candidates"][0] == "model.layers.1.mlp.down_proj"
    assert 0 < len(error.details["candidates"]) <= MAX_SITE_CANDIDATES
    error = refused(ErrorCode.NOT_FOUND, [f"residual:{LAYERS + 1}"])
    assert error.details["candidates"][0] == f"residual:{LAYERS}"
    assert 0 < len(error.details["candidates"]) <= MAX_SITE_CANDIDATES
    # Malformed site names and site lists: INVALID_ARGUMENT.
    too_many = [f"model.layers.{i}:output" for i in range(MAX_PROBE_SITES + 1)]
    for sites in (["residual"], ["residual:-1"], ["residual:01"], ["residual: 1"], ["model.layers.1.mlp"],
                  ["model.layers.1.mlp:weight"], [":input"], [""], [1], "residual:1", [], too_many,
                  ["residual:1", "residual:1"]):
        refused(ErrorCode.INVALID_ARGUMENT, sites)
    _expect(ErrorCode.INVALID_ARGUMENT, probe, handle, texts, sites=["residual:1"], device="cuda")
    _expect(ErrorCode.INVALID_ARGUMENT, probe, object(), texts, sites=["residual:1"], device="cpu")


def test_probe_refuses_sites_whose_capture_is_not_a_sequence_of_vectors(handles):
    texts = [ProbeText(_tokens(9, 8))]
    cases = {
        "qwen3": ("model.layers.0.self_attn.q_norm:output",  # [batch, sequence, heads, head_dim]
                  "model.embed_tokens:input",  # int64 token IDs [batch, sequence]
                  "model.layers.0.self_attn:input",  # called with keyword arguments only
                  "model.layers.0.self_attn:output",  # a tuple (attention output, weights)
                  "model:output"),  # a model output object
        "qwen3_5": ("model.language_model.layers.0.linear_attn.conv1d:input",  # the kernel reads its weight; never runs
                    "model.language_model.layers.0.linear_attn.norm:input",  # flattened to [tokens * heads, head_dim]
                    "model.language_model.layers:output"),  # a module list, which never runs
    }
    for name, sites in cases.items():
        for site in sites:
            error = _expect(ErrorCode.INVALID_ARGUMENT, probe, handles[name], texts, sites=["residual:1", site],
                            device="cpu")
            assert error.details["site"] == site, (site, error)
    # A three-dimensional capture whose second axis is not the sequence (channels first, as nn.Conv1d takes its
    # input) is refused too: a hook of the test's own, registered before the probe's, hands later hooks the final
    # norm's output transposed to [1, features, sequence].
    norm = handles["llama_untied"].runtime("cpu").module.get_submodule("model.norm")
    hook = norm.register_forward_hook(lambda module, args, output: output.transpose(1, 2))
    try:
        error = _expect(ErrorCode.INVALID_ARGUMENT, probe, handles["llama_untied"], texts, sites=["model.norm:output"],
                        device="cpu")
    finally:
        hook.remove()
    assert "[1, 16, 8]" in error.message and "[1, 8, features]" in error.message


def test_residual_sites_need_decoder_layers_that_pass_on_one_tensor(handles, monkeypatch):
    handle = handles["llama_untied"]
    runtime = handle.runtime("cpu")
    texts = [ProbeText(_tokens(16, 8))]
    # A decoder layer that hands on a tuple (as layers of older transformers versions did) is not read as a boundary.
    layer = runtime.module.get_submodule("model.layers.0")
    hook = layer.register_forward_hook(lambda module, args, output: (output,))
    try:
        error = _expect(ErrorCode.UNSUPPORTED_CAPABILITY, probe, handle, texts, sites=["residual:1"], device="cpu")
    finally:
        hook.remove()
    assert "a tuple" in error.message
    # A model without exactly one module list of num_hidden_layers decoder layers has no residual sites, but its
    # module sites still work.
    monkeypatch.setattr(runtime.module.config, "num_hidden_layers", LAYERS + 1)
    _expect(ErrorCode.UNSUPPORTED_CAPABILITY, probe, handle, texts, sites=["residual:0"], device="cpu")
    assert probe(handle, texts, sites=["model.layers.1:output"], device="cpu").token_counts == (8,)


def test_a_site_that_runs_twice_in_one_forward_is_refused(handles):
    runtime = handles["llama_untied"].runtime("cpu")
    mlp = runtime.module.get_submodule("model.layers.0.mlp")

    def call_down_proj_again(module, args, output) -> None:  # a hook of the test's own; returns None
        module.down_proj(module.act_fn(module.gate_proj(args[0])) * module.up_proj(args[0]))

    hook = mlp.register_forward_hook(call_down_proj_again)
    try:
        error = _expect(ErrorCode.INVALID_ARGUMENT, probe, handles["llama_untied"], [ProbeText(_tokens(8, 6))],
                        sites=["model.layers.0.mlp.down_proj:output"], device="cpu")
    finally:
        hook.remove()
    assert "more than once" in error.message and error.details["site"] == "model.layers.0.mlp.down_proj:output"


def test_probe_texts_accept_lists_and_arrays_of_token_ids(handles):
    tokens = _tokens(15, 6)
    texts = [ProbeText(list(tokens)), ProbeText(np.array(tokens)), ProbeText(torch.tensor(tokens)),
             ProbeText(tuple(np.int64(token) for token in tokens))]
    assert all(type(text.tokens) is tuple and text.tokens == tokens for text in texts)
    result = probe(handles["qwen3"], texts, sites=["residual:1"], device="cpu")
    assert all(np.array_equal(row, result.means["residual:1"][0]) for row in result.means["residual:1"])


def test_probe_refuses_bad_texts_and_limits(handles):
    handle = handles["qwen3"]
    good = ProbeText(_tokens(9, 8))

    def refused(texts, **kwargs) -> WSEError:
        return _expect(ErrorCode.INVALID_ARGUMENT, probe, handle, texts, sites=["residual:1"], device="cpu", **kwargs)

    for texts in ([], [ProbeText(())], [ProbeText((1, VOCAB))], [ProbeText((1, -1))], [ProbeText((1, 2.0))],
                  [ProbeText((1, True))], [ProbeText((1, 2, 3), start=-1)], [ProbeText((1, 2, 3), start=3)],
                  [ProbeText((1, 2, 3), start=2, stop=2)], [ProbeText((1, 2, 3), start=2, stop=1)],
                  [ProbeText((1, 2, 3), stop=4)], [ProbeText((1, 2, 3), stop=0)], [ProbeText((1, 2, 3), start=True)],
                  [ProbeText((1, 2, 3), stop=2.0)], [(1, 2, 3)], "abc"):
        refused(texts)
    assert refused([good, ProbeText((1, VOCAB))]).details["text"] == 1
    # max_tokens caps each text's forward length, stop; tokens after stop do not count.
    long = ProbeText(_tokens(10, 20))
    assert refused([good, long], max_tokens=19).details["text"] == 1
    assert probe(handle, [long], sites=["residual:1"], device="cpu", max_tokens=20).token_counts == (20,)
    assert probe(handle, [ProbeText(long.tokens, stop=19)], sites=["residual:1"], device="cpu",
                 max_tokens=19).token_counts == (19,)
    for bad in (0, -1, True, 2.5, MAX_PROBE_TOKENS + 1):
        refused([good], max_tokens=bad)
    assert 0 < DEFAULT_MAX_PROBE_TOKENS <= MAX_PROBE_TOKENS
    # A forward longer than the model's context is refused at any max_tokens.
    beyond = ProbeText(_tokens(11, CONTEXT + 1))
    assert refused([beyond]).details["context"] == CONTEXT
    assert probe(handle, [ProbeText(beyond.tokens, stop=CONTEXT)], sites=["residual:1"],
                 device="cpu").token_counts == (CONTEXT,)


def test_a_model_without_an_engine_runtime_is_unsupported(signed_sum):
    handle = load_model(signed_sum.task_tensors(), "mlp@1", signed_sum.CONFIG)
    _expect(ErrorCode.UNSUPPORTED_CAPABILITY, probe, handle, [ProbeText((1, 2, 3))], sites=["residual:0"],
            device="cpu")


# -- weights, hooks, and module boundaries ----------------------------------------------------------


def test_a_probe_changes_no_weights_and_removes_its_hooks(handles):
    handle = handles["qwen3_5"]
    runtime = handle.runtime("cpu")
    before = {name: parameter.detach().clone() for name, parameter in runtime.module.named_parameters()}
    hooks = _hooks(runtime.module)
    texts = [ProbeText(_tokens(12, 10)), ProbeText(_tokens(13, 7), start=2)]
    sites = [*(f"residual:{index}" for index in range(LAYERS + 1)), "model.language_model.layers.0.mlp.down_proj:input",
             "model.language_model.layers.1.mlp.down_proj:output", "lm_head:output"]
    result = probe(handle, texts, sites=sites, device="cpu")
    assert result.means["lm_head:output"].shape == (2, VOCAB)
    assert _hooks(runtime.module) == hooks
    # A refusal raised inside a hook in the middle of a forward (in layer 1, after residual:1 was read) leaves no
    # hook behind either.
    _expect(ErrorCode.INVALID_ARGUMENT, probe, handle, texts,
            sites=["residual:1", "model.language_model.layers.1.self_attn.q_norm:output"], device="cpu")
    assert _hooks(runtime.module) == hooks
    assert all(torch.equal(parameter, before[name]) for name, parameter in runtime.module.named_parameters())
    assert hash_weights(handle.tensors(), weights_hash_scheme(handle.adapter)) == handle.weights_hash


def test_the_probe_module_imports_no_transformers():
    code = textwrap.dedent(
        """
        import sys
        import wse.probes
        print(sorted(m for m in sys.modules if m.split(".")[0] in ("transformers", "mcp", "agents", "openai")))
        """
    )
    loaded = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert loaded.returncode == 0, loaded.stderr
    assert loaded.stdout.strip() == "[]"
