"""BF16 edits and binary patch blocks (M2 plan Sections 3.2 and 3.3, Task 5).

Edits compute in FP32 and write back to the stored dtype with
round-to-nearest-even; changed elements are counted on the stored bits after
write-back. A materialized step of an open-weight model holds a binary block
(sparse: flat row-major ``uint64`` indices and stored-dtype bit patterns;
dense: the bit patterns of a contiguous slice) bound to the tensor's base and
result blob IDs, and replay reads those blocks without running any operator.
The models are the tiny offline fixtures of ``tests/hf_fixtures.py``.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from wse.api import Engine
from wse.schemas import ErrorCode, WSEError
from wse.tensors import blob_id

REPO = "example-org/tiny-model"
REVISION = "0123456789abcdef0123456789abcdef01234567"
DOWN = "model.layers.0.mlp.down_proj.weight"  # [16, 32] in the Llama fixtures


def _sel(*ranges):
    return [{"start": start, "stop": stop, "step": 1} for start, stop in ranges]


def _expect(code, func, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        func(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


def _bits(tensor: torch.Tensor) -> list[int]:
    """Stored bit patterns of a 16-bit tensor, row-major."""
    return tensor.contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1).tolist()


def _bf16(bits: list[int]) -> torch.Tensor:
    return torch.tensor(np.array(bits, dtype=np.uint16).view(np.int16)).view(torch.bfloat16)


@pytest.fixture(scope="module")
def hf_workspace(hf_snapshots, tmp_path_factory):
    workspace = tmp_path_factory.mktemp("patch_blocks") / "workspace"
    with Engine(workspace) as engine:
        ids = {name: engine.import_hf_snapshot(hf_snapshots[name], repo=REPO, revision=REVISION).checkpoint_id
               for name in ("llama_untied", "qwen3", "gemma3")}
    return workspace, ids


@pytest.fixture(scope="module")
def bf16_model(hf_workspace):
    workspace, ids = hf_workspace
    with Engine(workspace) as engine:
        return engine.open_checkpoint(ids["llama_untied"])


def _with_tensor(model, name: str, tensor: torch.Tensor):
    from wse.patches import _derive

    return _derive(model, {**model._tensors, name: tensor})


# -- BF16 arithmetic ---------------------------------------------------------------------------------


def test_bf16_helpers_round_the_float32_bit_pattern_to_nearest_even():
    from wse.tensors import bf16_to_f32, f32_to_bf16_bits_rne

    cases = {  # float32 bits -> bfloat16 bits
        0x3F800000: 0x3F80,  # 1.0, exact
        0x3F808000: 0x3F80,  # halfway between 0x3F80 and 0x3F81: to the even 0x3F80
        0x3F818000: 0x3F82,  # halfway between 0x3F81 and 0x3F82: to the even 0x3F82
        0x3F808001: 0x3F81,  # just above halfway: up
        0x3F807FFF: 0x3F80,  # just below halfway: down
        0xBF818000: 0xBF82,  # negative ties round the magnitude to even
        0x80000000: 0x8000,  # -0.0 keeps its sign
        0x00018000: 0x0002,  # a subnormal tie, to even
        0x00008000: 0x0000,  # a subnormal tie, to even (zero)
        0x7F7FFFFF: 0x7F80,  # the largest float32 rounds up to infinity
        0x7F800000: 0x7F80,  # infinity stays infinity
    }
    source = np.array(list(cases), dtype=np.uint32).view(np.float32)
    assert f32_to_bf16_bits_rne(source).tolist() == list(cases.values())
    assert f32_to_bf16_bits_rne(source).dtype == np.uint16
    # The same as torch's float32 -> bfloat16 conversion.
    assert _bits(torch.from_numpy(source.copy()).to(torch.bfloat16)) == list(cases.values())
    # A NaN stays a NaN (quiet, with its sign).
    nan = f32_to_bf16_bits_rne(np.array([np.nan, -np.nan], dtype=np.float32))
    assert np.isnan(bf16_to_f32(nan)).all() and (nan >> 15).tolist() == [0, 1]
    # Widening is exact: the bit pattern moves to the high half.
    widened = bf16_to_f32(np.array([0x3F80, 0xC000, 0x8000, 0x0001], dtype=np.uint16))
    assert widened.dtype == np.float32
    assert widened.view(np.uint32).tolist() == [0x3F800000, 0xC0000000, 0x80000000, 0x00010000]


def test_bf16_scale_computes_in_fp32_and_rounds_to_nearest_even(bf16_model):
    from wse.patches import materialize_patch

    original = bf16_model._tensors[DOWN].clone()
    edited = original.clone()
    # Row 0: 1.0, 1 + 2^-7, 1 + 3 * 2^-7, -2.0, -0.0; row 1: 2 - 2^-7, 1 + 2^-7.
    edited[0, :5] = _bf16([0x3F80, 0x3F81, 0x3F83, 0xC000, 0x8000])
    edited[1, :2] = _bf16([0x3FFF, 0x3F81])
    model = _with_tensor(bf16_model, DOWN, edited)
    result = materialize_patch(model, [
        {"op": "scale_slice", "tensor": DOWN, "selection": _sel((0, 1), (0, 5)), "factor": 1.5},
        {"op": "scale_slice", "tensor": DOWN, "selection": _sel((1, 2), (0, 1)), "factor": 1.004},
        {"op": "scale_slice", "tensor": DOWN, "selection": _sel((1, 2), (1, 2)), "factor": 1.25},
    ])
    after = result.candidate._tensors[DOWN]
    # 1.5 * (1 + 2^-7) = 193.5 units of 2^-7: a tie, to the even 194 (0x3FC2).
    # 1.5 * (1 + 3 * 2^-7) = 196.5 units: a tie, to the even 196 (0x3FC4), not up.
    # -0.0 * 1.5 stays -0.0 and is not a change.
    assert _bits(after[0, :5]) == [0x3FC0, 0x3FC2, 0x3FC4, 0xC040, 0x8000]
    # (2 - 2^-7) * 1.004 = 2.00016 in FP32 rounds to 2.0. Rounding the factor to BF16 first
    # (1.0) would have kept 0x3FFF, so the product is computed in FP32.
    assert _bits(after[1, :1]) == [0x4000]
    # 1.25 * (1 + 2^-7) = 161.25 units of 2^-7: down to 161 (0x3FA1).
    assert _bits(after[1, 1:2]) == [0x3FA1]
    assert [step.changed_elements for step in result.patch.steps] == [4, 1, 1]
    # Nothing outside the selections changed.
    mask = torch.ones(after.shape, dtype=torch.bool)
    mask[0, :5] = mask[1, :2] = False
    assert _bits(after[mask]) == _bits(edited[mask])
    # The same as torch computing in FP32 and casting back.
    for row, cols, factor in ((0, slice(0, 5), 1.5), (1, slice(0, 1), 1.004), (1, slice(1, 2), 1.25)):
        expected = (edited[row, cols].float() * torch.tensor(factor, dtype=torch.float32)).to(torch.bfloat16)
        assert _bits(after[row, cols]) == _bits(expected)
    # The input model never changes.
    assert _bits(model._tensors[DOWN]) == _bits(edited) and _bits(bf16_model._tensors[DOWN]) == _bits(original)


def test_changed_elements_are_counted_after_write_back(bf16_model):
    from wse.patches import apply_operator

    before = bf16_model._tensors[DOWN]
    result = apply_operator(bf16_model, "scale_slice", {"tensor": DOWN, "selection": _sel((0, 16), (0, 32)),
                                                       "factor": 1.003})
    after = result.candidate._tensors[DOWN]
    # Half a unit in the last place of a BF16 value is between 2^-9 and 2^-8 of its magnitude, so a
    # factor of 1.003 moves only values whose significand is large enough (about 1.3 or more) to
    # another BF16 value; the rest round back to their own bits and do not count.
    expected = int((before.contiguous().view(torch.int16) != after.contiguous().view(torch.int16)).sum())
    reference = (before.float() * torch.tensor(1.003)).to(torch.bfloat16)
    assert _bits(after) == _bits(reference)
    assert 0 < expected < before.numel()
    step = result.patch.steps[0]
    assert step.changed_elements == expected
    diff = {d.tensor: d for d in result.diff}
    assert diff[DOWN].changed_elements == expected and diff[DOWN].element_count == before.numel()
    assert all(d.changed_elements == 0 for name, d in diff.items() if name != DOWN)
    # The block records exactly the changed elements when that is the smaller encoding.
    assert step.values is None and step.block is not None
    if step.block.kind == "sparse":
        assert step.block.count == expected
    # A no-op edit is recorded and counts nothing.
    noop = apply_operator(bf16_model, "scale_slice", {"tensor": DOWN, "selection": _sel((0, 1), (0, 4)),
                                                     "factor": 1.0})
    assert noop.patch.steps[0].changed_elements == 0 and noop.candidate.weights_hash == bf16_model.weights_hash


# -- blocks ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_sparse_and_dense_blocks_round_trip_and_bind_base_and_result_ids(dtype):
    from wse.patch_blocks import apply_block, decode_block, make_block

    generator = torch.Generator().manual_seed(0)
    base = torch.randn(6, 10, generator=generator).to(dtype)
    # A few scattered changes: a sparse block of exactly the changed elements.
    result = base.clone()
    result[1, 3], result[4, 9], result[0, 0] = 0.5, -0.25, -0.0
    block, data = make_block(base, result)
    assert block.kind == "sparse" and block.count == 3 and block.dtype == str(dtype).removeprefix("torch.")
    assert (block.base_blob_id, block.result_blob_id) == (blob_id(base), blob_id(result))
    import hashlib

    assert block.block_id == hashlib.sha256(data).hexdigest()
    width = base.element_size()
    assert len(data) == 3 * (8 + width)
    indices, bits = decode_block(block, data)
    assert indices.dtype == np.uint64 and indices.tolist() == [0, 13, 49]
    assert bits.dtype == np.dtype(f"<u{width}")
    # Indices are little-endian uint64; the bit patterns follow, little-endian in the stored dtype.
    assert data[:8] == struct.pack("<Q", 0)
    rebuilt = apply_block(base, block, data)
    assert blob_id(rebuilt) == block.result_blob_id and rebuilt.dtype == dtype
    assert torch.equal(rebuilt.view(torch.uint8), result.contiguous().view(torch.uint8))
    assert blob_id(base) == block.base_blob_id  # the base is never modified
    # A changed contiguous slice: a dense block of that slice's bit patterns.
    dense_result = base.clone()
    dense_result[2:4, 1:9] = torch.arange(16, dtype=torch.float32).reshape(2, 8).to(dtype)
    dense, dense_data = make_block(base, dense_result, selection=_sel((2, 4), (1, 9)))
    assert dense.kind == "dense" and dense.count == 16 and len(dense_data) == 16 * width
    assert [s.model_dump() for s in dense.selection] == _sel((2, 4), (1, 9))
    assert torch.equal(apply_block(base, dense, dense_data).view(torch.uint8),
                       dense_result.contiguous().view(torch.uint8))
    # Bound to the base: another tensor is refused, and so is a result that does not match.
    _expect(ErrorCode.VERSION_CONFLICT, apply_block, result, block, data)
    forged = block.model_copy(update={"result_blob_id": "0" * 64})
    _expect(ErrorCode.INVALID_PATCH, apply_block, base, forged, data)
    # Bound to its bytes: tampered or truncated data is refused before anything is written.
    tampered = bytearray(data)
    tampered[-1] ^= 0x01
    _expect(ErrorCode.INVALID_PATCH, apply_block, base, block, bytes(tampered))
    _expect(ErrorCode.INVALID_PATCH, apply_block, base, block, data[:-1])


def test_binary_patch_files_round_trip_and_detect_tampering():
    from wse.patch_blocks import read_patch_file, write_patch_file

    blocks = {"a" * 64: b"\x01\x02", "b" * 64: b"\x03"}
    blocks = {__import__("hashlib").sha256(v).hexdigest(): v for v in blocks.values()}
    document = {"kind": "example", "steps": [1, 2]}
    data = write_patch_file(document, blocks)
    assert data == write_patch_file(document, dict(reversed(list(blocks.items()))))  # deterministic
    assert read_patch_file(data) == (document, blocks)
    # A plain JSON patch (M1's format) reads back with no blocks.
    assert read_patch_file(b'{"kind":"json"}') == ({"kind": "json"}, {})
    tampered = bytearray(data)
    tampered[-1] ^= 0xFF
    _expect(ErrorCode.ARTIFACT_CORRUPT, read_patch_file, bytes(tampered))
    _expect(ErrorCode.ARTIFACT_CORRUPT, read_patch_file, data[:-1])


# -- replay ------------------------------------------------------------------------------------------


def test_replay_of_a_binary_patch_is_hash_exact_and_runs_no_operator(bf16_model, hf_workspace, monkeypatch):
    from wse.operators.tensor import ScaleSliceOperator, SetSliceOperator
    from wse.patches import materialize_patch, replay_patch
    from wse.policy import SMALL_MODEL_POLICY_ID

    q = "model.layers.1.self_attn.q_proj.weight"
    result = materialize_patch(bf16_model, [
        {"op": "scale_slice", "tensor": DOWN, "selection": _sel((0, 16), (0, 32)), "factor": -0.75},
        {"op": "set_slice", "tensor": q, "selection": _sel((3, 4), (0, 4)), "values": [[0.1, -0.2, 0.3, 1e-3]]},
        {"op": "scale_slice", "tensor": DOWN, "selection": _sel((2, 3), (0, 8)), "factor": 3.0},
    ])
    steps = result.patch.steps
    assert all(step.values is None and step.block is not None for step in steps)
    assert {step.block.block_id for step in steps} == set(result.blocks)
    # Each block is bound to the tensor state around its own step.
    assert steps[0].block.base_blob_id == blob_id(bf16_model._tensors[DOWN])
    assert steps[2].block.base_blob_id == steps[0].block.result_blob_id
    assert steps[2].block.result_blob_id == blob_id(result.candidate._tensors[DOWN])

    def never(*args, **kwargs):
        raise AssertionError("replay must not run an editing operator")

    monkeypatch.setattr(ScaleSliceOperator, "apply", never)
    monkeypatch.setattr(SetSliceOperator, "apply", never)
    replayed = replay_patch(bf16_model, result.patch, blocks=result.blocks)
    assert replayed.weights_hash == result.candidate.weights_hash
    for name in (DOWN, q):
        assert blob_id(replayed._tensors[name]) == blob_id(result.candidate._tensors[name])
    # Replay reads the binary blocks: without them it fails, and so it does with a wrong block.
    _expect(ErrorCode.INVALID_PATCH, replay_patch, bf16_model, result.patch)
    wrong = {key: value[::-1] for key, value in result.blocks.items()}
    _expect(ErrorCode.INVALID_PATCH, replay_patch, bf16_model, result.patch, blocks=wrong)
    monkeypatch.undo()

    # Managed: the preview stores one binary patch file, and apply replays it without operators.
    workspace, ids = hf_workspace
    with Engine(workspace) as engine:
        grant = engine.grant_run("bf16_patch_v1", SMALL_MODEL_POLICY_ID, ids["llama_untied"])
        experiment = engine.create_experiment(grant.run_id, "bf16_patch_v1", SMALL_MODEL_POLICY_ID, "create")
        spec = {"base_checkpoint_id": ids["llama_untied"], "hypothesis": "Halving one row changes nothing.",
                "operations": [{"op": "scale_slice", "tensor": DOWN, "selection": _sel((5, 6), (0, 32)),
                                "factor": 0.5}]}
        preview = engine.preview_patch(experiment.experiment_id, spec, "preview")
        assert preview.storage.patch_bytes > 0
        artifact = engine.read_patch(experiment.experiment_id, preview.patch_id)
        assert artifact.patch.steps[0].block is not None and artifact.patch.steps[0].values is None
        monkeypatch.setattr(ScaleSliceOperator, "apply", never)
        candidate = engine.apply_patch(experiment.experiment_id, preview.patch_id, "apply")
        monkeypatch.undo()
        handle = engine.open_checkpoint(candidate.checkpoint_id)
        direct = materialize_patch(bf16_model, [{"op": "scale_slice", "tensor": DOWN,
                                                 "selection": _sel((5, 6), (0, 32)), "factor": 0.5}])
        assert handle.weights_hash == direct.candidate.weights_hash == preview.result_weights_hash
        # The run-facing view summarizes the step and omits its values and block bytes.
        document = engine.find_patch(preview.patch_id).model_dump(mode="json")
        assert "values" not in str(document) and "block_id" not in str(document)


@pytest.mark.parametrize("name", ["qwen3", "gemma3"])
def test_f16_and_f32_edits_of_open_weight_models_use_blocks_too(name, hf_workspace):
    from wse.patches import apply_operator, replay_patch

    workspace, ids = hf_workspace
    with Engine(workspace) as engine:
        model = engine.open_checkpoint(ids[name])
    tensor = "model.layers.0.self_attn.o_proj.weight"
    result = apply_operator(model, "scale_slice", {"tensor": tensor, "selection": _sel((0, 2), (0, 16)),
                                                  "factor": -2.0})
    step = result.patch.steps[0]
    assert step.block is not None and step.values is None
    stored = model._tensors[tensor]
    expected = (stored[:2].float() * -2.0).to(stored.dtype)
    assert torch.equal(result.candidate._tensors[tensor][:2].view(torch.uint8), expected.view(torch.uint8))
    assert replay_patch(model, result.patch, blocks=result.blocks).weights_hash == result.candidate.weights_hash


def test_m1_patches_keep_inline_values(signed_sum):
    from wse.api import load_model
    from wse.patches import apply_operator

    model = load_model(signed_sum.task_tensors(), "mlp@1", signed_sum.CONFIG)
    result = apply_operator(model, "scale_slice", {"tensor": "layers.1.weight", "selection": _sel((1, 2), (0, 4)),
                                                  "factor": -1.0})
    step = result.patch.steps[0]
    assert step.block is None and len(step.values) == 4 and result.blocks == {}
    assert "block" not in step.model_dump(mode="json")
