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
MODELS = ("llama_tied", "llama_untied", "qwen3", "gemma3", "qwen3_5")
# The reference class per fixture: the snapshot's own causal-LM class, except for the Qwen3.5
# multimodal wrapper, whose language model is Qwen3_5ForCausalLM (as the family descriptor states).
REFERENCE_CLASSES = {"qwen3_5": "Qwen3_5ForCausalLM"}
STOP = 2  # </s> in the fixtures' tokenizer
PAD = 0
# MPS against CPU: matmul and attention kernels differ, so logits agree only within a tolerance
# that scales with the precision of the stored dtype: about 2 units of BF16's last place (2^-7),
# 16 of F16's (2^-10), and 80 of F32's (2^-23). On the fixtures (|logits| < 0.5) the measured maximum
# absolute differences were 2.0e-3 (BF16), 1.8e-4 (F16), and 1.2e-7 (F32).
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
    import transformers

    model_class = getattr(transformers, REFERENCE_CLASSES.get(name, "AutoModelForCausalLM"))
    return model_class.from_pretrained(str(snapshot)).eval()


def _prompts(seed: int = 0, rows: int = 3, length: int = 9) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(3, 40, (rows, length), generator=generator)


def _left_padded() -> tuple[torch.Tensor, torch.Tensor]:
    """Two prompts of lengths 9 and 5, left-padded to one length, with their attention mask."""
    ids = _prompts(seed=1, rows=2)
    mask = torch.ones_like(ids)
    ids[1, :4], mask[1, :4] = PAD, 0
    return ids, mask


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
    # The same holds with an attention mask over a left-padded batch.
    ids, mask = _left_padded()
    with torch.inference_mode():
        expected = reference(input_ids=ids, attention_mask=mask).logits.float()
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

    def expected(stop: int | None, steps: int) -> torch.Tensor:
        with torch.inference_mode():
            out = reference.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=steps,
                                     eos_token_id=stop, pad_token_id=PAD)
        return out[:, width:]

    unstopped = expected(None, 10)
    ours = runtime.generate(ids, mask, max_new_tokens=10, pad_token_id=PAD)
    assert ours.tokens == unstopped.tolist() and ours.finished == [False, False]
    assert ours.generated_tokens == [10, 10]
    # Stop on the token row 0 emits third: row 0 finishes early, and later positions hold the pad token.
    stop = int(unstopped[0, 2])
    stopped = expected(stop, 10)
    ours = runtime.generate(ids, mask, max_new_tokens=10, stop_token_ids=[stop], pad_token_id=PAD)
    for row, tokens in enumerate(ours.tokens):
        assert tokens == stopped[row, :len(tokens)].tolist()
        assert all(token == PAD for token in stopped[row, len(tokens):].tolist())
        assert ours.finished[row] == (stop in tokens) and (not ours.finished[row] or tokens[-1] == stop)
    assert ours.finished[0] and len(ours.tokens[0]) <= 3


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
    assert info == {"device": "cpu", "dtypes": ["bfloat16"], "model_class": "Qwen3_5ForCausalLM",
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
    "wse.operators.tensor, wse.strategies.scripted, wse.engines, wse.engines.hf_torch"
)


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=120)


def test_core_modules_import_no_transformers():
    # With transformers installed, importing the core and engine modules loads none of it ...
    loaded = _run(
        f"""
        import sys
        import {CORE_MODULES}
        print(sorted(m for m in sys.modules if m.split(".")[0] in ("transformers", "mcp", "agents", "openai")))
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
