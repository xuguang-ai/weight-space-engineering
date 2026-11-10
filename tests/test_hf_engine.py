"""The PyTorch engine for imported causal LMs: logits parity and generation (M2 plan Task 4).

The engine (``wse.engines.hf_torch``) builds the ``transformers`` module on
the meta device and assigns the stored WSE tensors to it. These tests compare
it with ``from_pretrained`` on the same snapshot: logits bitwise on CPU, within
stated tolerances on MPS, and greedy generation token for token. Snapshots are
tiny random checkpoints built offline by ``tests/hf_fixtures.py``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from wse.api import Engine
from wse.schemas import ErrorCode, WSEError

REPO = "example-org/tiny-model"
REVISION = "0123456789abcdef0123456789abcdef01234567"
# gpt2 has learned absolute position embeddings, which expose a padding-shifted position that
# rotary embeddings (relative in attention) can hide.
MODELS = ("llama_tied", "llama_untied", "qwen3", "gemma3", "qwen3_5", "gpt2")
# The reference class per fixture: the snapshot's own causal-LM class, except for the Qwen3.5
# multimodal wrapper, whose language model is Qwen3_5ForCausalLM (as the family descriptor states).
REFERENCE_CLASSES = {"qwen3_5": "Qwen3_5ForCausalLM"}
STOP = 2  # </s> in the fixtures' tokenizer
PAD = 0
# MPS against CPU: matmul and attention kernels differ, so logits agree only within a tolerance
# that scales with the precision of the stored dtype: 1.6e-2 is 2 units in the last place of a BF16
# value near 1 (2^-7 each), 2e-3 is 2 of F16's (2^-10 each), and 1e-5 is 84 of F32's (2^-23 each).
# On the fixtures (|logits| < 0.5) the measured maximum absolute differences were 2.0e-3 (BF16),
# 1.8e-4 (F16), and 1.2e-7 (F32). These tolerances hold for the fixtures only and are no parity
# criterion for real models: on Llama-3.2-1B-Instruct (BF16) MPS and CPU logits differed by up
# to 0.25, with equal argmax.
MPS_TOLERANCES = {torch.float32: dict(atol=1e-5, rtol=1e-5), torch.float16: dict(atol=2e-3, rtol=2e-3),
                  torch.bfloat16: dict(atol=1.6e-2, rtol=1.6e-2)}
MPS_SKIP_REASON = "PyTorch MPS is not available on this host"


def _expect(code: ErrorCode, call, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        call(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


@pytest.fixture(scope="module")
def hf_workspace(hf_snapshots, tmp_path_factory) -> tuple[Path, dict[str, str]]:
    """One workspace with every fixture imported; returns it and fixture name -> checkpoint ID."""
    workspace = tmp_path_factory.mktemp("hf_engine") / "workspace"
    with Engine(workspace) as engine:
        ids = {name: engine.import_hf_snapshot(hf_snapshots[name], repo=REPO, revision=REVISION).checkpoint_id
               for name in MODELS}
    return workspace, ids


@pytest.fixture(scope="module")
def handles(hf_workspace):
    workspace, ids = hf_workspace
    with Engine(workspace) as engine:
        return {name: engine.open_checkpoint(checkpoint_id) for name, checkpoint_id in ids.items()}


def _reference(snapshot: Path, name: str):
    """``from_pretrained`` on the snapshot, with every parameter kept in its stored dtype.

    ``from_pretrained`` casts parameters to the configuration's dtype, so the Qwen3.5 fixture's F32
    tensors (``A_log`` and ``linear_attn.norm.weight``, F32 in the published checkpoint too) would
    become BF16. WSE keeps stored dtypes (M2 plan Section 2.1), and so does this reference: each
    parameter whose stored dtype differs is replaced by the stored tensor itself.
    """
    import transformers
    from safetensors import safe_open

    model_class = getattr(transformers, REFERENCE_CLASSES.get(name, "AutoModelForCausalLM"))
    model = model_class.from_pretrained(str(snapshot)).eval()
    parameters = dict(model.named_parameters())
    for file in sorted(snapshot.glob("*.safetensors")):
        with safe_open(str(file), framework="pt") as weights:
            for checkpoint_name in weights.keys():
                module_name = checkpoint_name.replace("model.language_model.", "model.", 1)
                if module_name not in parameters:
                    continue
                stored = weights.get_tensor(checkpoint_name)
                if stored.dtype != parameters[module_name].dtype:
                    owner, _, local = module_name.rpartition(".")
                    model.get_submodule(owner)._parameters[local] = torch.nn.Parameter(stored, requires_grad=False)
    return model


def _prompts(seed: int = 0, rows: int = 3, length: int = 9) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(3, 40, (rows, length), generator=generator)


def _left_padded() -> tuple[torch.Tensor, torch.Tensor]:
    """Two prompts of lengths 9 and 5, left-padded to one length, with their attention mask."""
    ids = _prompts(seed=1, rows=2)
    mask = torch.ones_like(ids)
    ids[1, :4], mask[1, :4] = PAD, 0
    return ids, mask


def _positions(mask: torch.Tensor) -> torch.Tensor:
    """generate's position IDs: the cumulative sum of the mask minus one, with padded positions at 0."""
    return (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)


class _Recorder:
    """A logits processor that records each step's float32 scores and changes nothing."""

    def __init__(self) -> None:
        self.steps: list[torch.Tensor] = []

    def __call__(self, sequences: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        self.steps.append(scores.clone())
        return scores


# -- logits parity ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODELS)
def test_wse_built_module_matches_from_pretrained_logits_bitwise_on_cpu(name, handles, hf_snapshots):
    handle = handles[name]
    runtime = handle.runtime("cpu")
    reference = _reference(hf_snapshots[name], name)
    assert {p.dtype for p in runtime.module.parameters()} == {p.dtype for p in reference.parameters()}
    for ids in (_prompts(), _prompts(seed=7, rows=1, length=16)):
        ours = runtime.logits(ids)
        with torch.inference_mode():
            expected = reference(input_ids=ids).logits.float()
        assert ours.dtype == torch.float32 and ours.shape == (*ids.shape, 64)
        assert torch.equal(ours, expected), (ours - expected).abs().max().item()
    # The same holds with an attention mask over a left-padded batch, whose positions the engine
    # derives from the mask as generate does (M2 plan Section 4.1).
    ids, mask = _left_padded()
    with torch.inference_mode():
        expected = reference(input_ids=ids, attention_mask=mask, position_ids=_positions(mask)).logits.float()
    assert torch.equal(runtime.logits(ids, mask)[mask.bool()], expected[mask.bool()])


@pytest.mark.parametrize("name", ("llama_tied", "qwen3_5"))
def test_cpu_modules_assign_the_stored_tensors_without_a_second_copy(name, handles):
    handle = handles[name]
    runtime = handle.runtime("cpu")
    stored = {tensor.untyped_storage().data_ptr() for tensor in handle._tensors.values()}
    parameters = dict(runtime.module.named_parameters(remove_duplicate=False))
    assert {p.untyped_storage().data_ptr() for p in parameters.values()} <= stored
    assert not any(p.requires_grad or p.is_meta for p in parameters.values())
    assert not any(b.is_meta for b in runtime.module.buffers())
    # Alias-group members share one tensor in the module as in storage.
    for group in handle.architecture.alias_groups:
        modules = {info.name: info.module_name for info in handle.architecture.tensors}
        assert len({parameters[modules[member]].data_ptr() for member in group}) == 1
    # A handle builds its runtime once per device.
    assert handle.runtime("cpu") is runtime


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason=MPS_SKIP_REASON)
@pytest.mark.parametrize("name", MODELS)
def test_mps_logits_match_cpu_within_tolerance(name, handles):
    handle = handles[name]
    cpu, mps = handle.runtime("cpu"), handle.runtime("mps")
    assert mps.info()["device"] == "mps" and next(mps.module.parameters()).device.type == "mps"
    dtype = handle.runtime("cpu").module.get_input_embeddings().weight.dtype
    ids = _prompts()
    torch.testing.assert_close(mps.logits(ids), cpu.logits(ids), **MPS_TOLERANCES[dtype])
    # Unmanaged forward through the M1 handle contract runs on the default device (MPS here).
    torch.testing.assert_close(handle.forward(ids), cpu.logits(ids), **MPS_TOLERANCES[dtype])


# -- generation ------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODELS)
def test_greedy_generation_matches_transformers_generate(name, handles, hf_snapshots):
    runtime = handles[name].runtime("cpu")
    reference = _reference(hf_snapshots[name], name)
    ids, mask = _left_padded()
    width = ids.shape[1]

    def expected(stop: int | None, steps: int, **kwargs):
        with torch.inference_mode():
            out = reference.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=steps,
                                     eos_token_id=stop, pad_token_id=PAD, **kwargs)
        return out[:, width:] if isinstance(out, torch.Tensor) else out

    unstopped = expected(None, 10)
    scores = _Recorder()
    ours = runtime.generate(ids, mask, max_new_tokens=10, pad_token_id=PAD, logits_processors=[scores])
    assert ours.tokens == unstopped.tolist() and ours.finished == [False, False]
    assert ours.generated_tokens == [10, 10]
    # Every step's float32 logits equal generate's, bitwise, for the padded row too.
    logits = expected(None, 10, output_logits=True, return_dict_in_generate=True).logits
    assert len(logits) == len(scores.steps) == 10
    for step, (theirs, mine) in enumerate(zip(logits, scores.steps)):
        assert torch.equal(mine, theirs.float()), (step, (mine - theirs.float()).abs().max().item())
    # Stop on the token row 0 emits third: row 0 finishes early, and later positions hold the pad token.
    stop = int(unstopped[0, 2])
    stopped = expected(stop, 10)
    ours = runtime.generate(ids, mask, max_new_tokens=10, stop_token_ids=[stop], pad_token_id=PAD)
    for row, tokens in enumerate(ours.tokens):
        assert tokens == stopped[row, :len(tokens)].tolist()
        assert all(token == PAD for token in stopped[row, len(tokens):].tolist())
        assert ours.finished[row] == (stop in tokens) and (not ours.finished[row] or tokens[-1] == stop)
    assert ours.finished[0] and len(ours.tokens[0]) <= 3


# A padded row runs in a batch of a different shape than the row alone, so CPU matmul and attention
# kernels may reduce in a different order. In BF16 and F16 the rounding of every layer's output to
# the stored dtype absorbs that, and the fixtures agree bitwise. In F32 the last bits differ: the
# measured maximum over six prompt pairs was 8.9e-8 (gemma3) and 4.5e-8 (gpt2), 3 units in the last
# place of the fixtures' logits (|logits| < 0.5, unit 2^-25), so F32 rows agree within 2^-22 (8 units).
# A position shifted by the padding moved the gpt2 fixture's logits by 0.17, far outside it.
PADDED_ROW_TOLERANCES = {torch.float32: dict(atol=2**-22, rtol=0.0), torch.float16: dict(atol=0.0, rtol=0.0),
                         torch.bfloat16: dict(atol=0.0, rtol=0.0)}


@pytest.mark.parametrize("name", MODELS)
def test_a_left_padded_row_computes_the_logits_of_the_same_prompt_run_alone(name, handles):
    runtime = handles[name].runtime("cpu")
    ids, mask = _left_padded()
    pads = int((mask[1] == 0).sum())
    assert pads == 4
    batched, alone = _Recorder(), _Recorder()
    together = runtime.generate(ids, mask, max_new_tokens=8, pad_token_id=PAD, logits_processors=[batched])
    single = runtime.generate(ids[1:, pads:], max_new_tokens=8, logits_processors=[alone])
    assert together.tokens[1] == single.tokens[0]
    dtype = runtime.module.get_input_embeddings().weight.dtype
    assert len(batched.steps) == len(alone.steps) == 8
    for step, (padded_row, row_alone) in enumerate(zip(batched.steps, alone.steps)):
        torch.testing.assert_close(padded_row[1], row_alone[0], **PADDED_ROW_TOLERANCES[dtype],
                                   msg=lambda message, step=step: f"step {step}: {message}")
    # Teacher-forced logits of the padded row equal those of the row alone in the same way.
    torch.testing.assert_close(runtime.logits(ids, mask)[1, pads:], runtime.logits(ids[1:, pads:])[0],
                               **PADDED_ROW_TOLERANCES[dtype])


def test_sampled_generation_is_reproducible_per_row_seed(handles):
    from wse.engines.hf_torch import Sampling

    runtime = handles["llama_untied"].runtime("cpu")
    prompt = _prompts(seed=3, rows=1)
    ids = prompt.repeat(3, 1)
    sampling = Sampling(temperature=1.3, top_k=40, top_p=0.95)
    first = runtime.generate(ids, max_new_tokens=16, sampling=sampling, seeds=[11, 11, 12])
    again = runtime.generate(ids, max_new_tokens=16, sampling=sampling, seeds=[11, 11, 12])
    assert first == again
    assert first.tokens[0] == first.tokens[1] != first.tokens[2]
    # A row's tokens depend on its own seed, not on its batch position or neighbors.
    alone = runtime.generate(prompt, max_new_tokens=16, sampling=sampling, seeds=[12])
    assert alone.tokens[0] == first.tokens[2]
    # top_k=1 keeps only the most likely token, which is greedy decoding.
    greedy = runtime.generate(ids, max_new_tokens=16)
    assert runtime.generate(ids, max_new_tokens=16, sampling=Sampling(top_k=1), seeds=[1, 2, 3]) == greedy
    # Logits processors run on the scores before sampling; this one forbids every token but 7.
    def only_seven(sequences: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        assert sequences.shape[0] == scores.shape[0] == 3
        allowed = torch.full_like(scores, float("-inf"))
        allowed[:, 7] = 0
        return scores + allowed

    forced = runtime.generate(ids, max_new_tokens=4, sampling=sampling, seeds=[1, 2, 3],
                              logits_processors=[only_seven])
    assert forced.tokens == [[7] * 4] * 3


def test_engine_arguments_are_validated(handles):
    from wse.engines.hf_torch import Sampling, select_device

    runtime = handles["llama_untied"].runtime("cpu")
    ids = _prompts(rows=2)
    _expect(ErrorCode.INVALID_ARGUMENT, runtime.logits, ids.float())
    _expect(ErrorCode.INVALID_ARGUMENT, runtime.logits, ids[0])
    _expect(ErrorCode.INVALID_ARGUMENT, runtime.logits, ids + 64)  # outside the vocabulary
    _expect(ErrorCode.INVALID_ARGUMENT, runtime.logits, ids, torch.ones(2, 3, dtype=torch.long))
    _expect(ErrorCode.INVALID_ARGUMENT, runtime.generate, ids, max_new_tokens=0)
    _expect(ErrorCode.INVALID_ARGUMENT, runtime.generate, ids, max_new_tokens=4, sampling=Sampling(), seeds=[1])
    _expect(ErrorCode.INVALID_ARGUMENT, runtime.generate, ids, max_new_tokens=4, seeds=[1, 2])
    for bad in (dict(temperature=0.0), dict(top_k=-1), dict(top_p=0.0), dict(top_p=1.5)):
        with pytest.raises(ValueError):
            Sampling(**bad)
    _expect(ErrorCode.INVALID_ARGUMENT, select_device, "cuda")
    assert select_device("cpu") == torch.device("cpu")
    assert select_device().type == ("mps" if torch.backends.mps.is_available() else "cpu")


def test_runtime_reports_device_dtypes_and_library_versions(handles):
    import transformers

    info = handles["qwen3_5"].runtime("cpu").info()
    # The fixture stores A_log and the gated norm weight in F32, as the published checkpoint does,
    # and the module keeps them in F32 (M2 plan Section 2.1).
    assert info == {"device": "cpu", "dtypes": ["bfloat16", "float32"], "model_class": "Qwen3_5ForCausalLM",
                    "attn_implementation": info["attn_implementation"], "torch": torch.__version__,
                    "transformers": transformers.__version__}
    assert info["attn_implementation"] in ("sdpa", "eager")


def test_the_adapter_advertises_forward_and_generation(handles):
    from wse.adapters.base import CAP_FORWARD_LOGITS, CAP_GENERATE_TEXT

    capabilities = handles["qwen3"].adapter.capabilities()
    assert capabilities.supports(CAP_FORWARD_LOGITS) and capabilities.supports(CAP_GENERATE_TEXT)


def test_a_run_bound_engine_never_reaches_forward(hf_workspace):
    from wse.policy import SMALL_MODEL_POLICY_ID

    workspace, ids = hf_workspace
    with Engine(workspace) as operator:
        grant = operator.grant_run("hf_engine_v1", SMALL_MODEL_POLICY_ID, ids["qwen3"])
    with Engine(workspace, run_id=grant.run_id) as bound:
        _expect(ErrorCode.POLICY_DENIED, bound.open_checkpoint, ids["qwen3"])
        assert not any(hasattr(bound, name) for name in ("runtime", "forward", "generate"))


# -- module boundaries -----------------------------------------------------------------------------

CORE_MODULES = (
    "wse, wse.api, wse.cli, wse.store, wse.database, wse.registry, wse.tensors, wse.schemas, wse.analysis, "
    "wse.policy, wse.patches, wse.experiments, wse.audit, wse.evaluation, wse.evaluators.classification, "
    "wse.evaluators.datasets, wse.jobs, wse.worker, wse.budget, wse.runner, wse.hf_snapshot, "
    "wse.adapters.mlp, wse.adapters.families, wse.adapters.hf_causal_lm, wse.operators.base, "
    "wse.operators.tensor, wse.strategies.scripted, wse.engines, wse.engines.hf_torch, wse.patch_blocks, "
    "wse.external_patches, wse.evaluators.sampling, wse.evaluators.graders, wse.evaluators.text_suites, "
    "wse.evaluators.generation, wse.evaluators.generation_runs"
)


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=120)


def test_core_modules_import_no_transformers():
    # With transformers installed, importing the core and engine modules loads none of it ...
    loaded = _run(
        f"""
        import sys
        import {CORE_MODULES}
        print(sorted(m for m in sys.modules
                     if m.split(".")[0] in ("transformers", "math_verify", "mcp", "agents", "openai")))
        """
    )
    assert loaded.returncode == 0, loaded.stderr
    assert loaded.stdout.strip() == "[]"
    # ... and without it they still import; building a runtime then fails with an install hint.
    blocked = _run(
        f"""
        import sys
        sys.modules["transformers"] = None
        import {CORE_MODULES}
        from wse.engines.hf_torch import build_model, select_device
        from wse.schemas import WSEError
        assert select_device("cpu").type == "cpu"
        try:
            build_model({{"model_type": "llama"}}, None, {{}}, device="cpu")
        except WSEError as exc:
            print(exc.code.value, "models-hf" in exc.message)
        """
    )
    assert blocked.returncode == 0, blocked.stderr
    assert blocked.stdout.strip() == "UNSUPPORTED_CAPABILITY True"
