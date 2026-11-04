"""Explicit, exact-version extension registration."""

from __future__ import annotations

import pytest
import torch

from wse.adapters.base import CAP_WEIGHTS_READ, AdapterCapabilities, ModelAdapter
from wse.adapters.mlp import MLPAdapter
from wse.api import load_model
from wse.registry import EXTENSION_KINDS, Registry, builtin_registry
from wse.schemas import ErrorCode, ExtensionRef, WSEError


class ScaleAdapter(MLPAdapter):
    """A separately supplied adapter: an MLP whose logits are doubled."""

    adapter_id = "scaled_mlp"
    version = "2"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(capabilities=frozenset({CAP_WEIGHTS_READ}))

    def validate(self, manifest, tensors):
        if str(manifest.adapter) != "scaled_mlp@2":
            raise WSEError(ErrorCode.INVALID_MODEL, "wrong adapter")
        architecture = self.describe(manifest.config)
        from wse.adapters.base import check_tensors_match

        check_tensors_match(architecture, tensors)
        return architecture

    def forward(self, tensors, inputs):
        return 2 * super().forward(tensors, inputs)


def test_builtin_registry_contains_mlp_adapter():
    registry = builtin_registry()
    adapter = registry.resolve("adapter", "mlp", "1")
    assert isinstance(adapter, MLPAdapter)
    assert isinstance(adapter, ModelAdapter)
    assert registry.list("adapter") == [ExtensionRef.parse("mlp@1")]
    assert registry.list("operator") == []


def test_resolve_is_exact_and_never_substitutes_versions():
    registry = builtin_registry()
    with pytest.raises(WSEError) as excinfo:
        registry.resolve("adapter", "mlp", "2")
    assert excinfo.value.code == ErrorCode.NOT_FOUND
    assert excinfo.value.details["registered_versions"] == ["1"]
    with pytest.raises(WSEError) as excinfo:
        registry.resolve("adapter", "unknown", "1")
    assert excinfo.value.code == ErrorCode.NOT_FOUND


def test_registering_new_version_keeps_old_one_resolvable():
    registry = builtin_registry()

    class MLPv2(MLPAdapter):
        version = "2"

    registry.register("adapter", "mlp", "2", MLPv2())
    assert registry.resolve("adapter", "mlp", "1").version == "1"
    assert registry.resolve("adapter", "mlp", "2").version == "2"


def test_duplicate_registration_is_rejected():
    registry = builtin_registry()
    with pytest.raises(WSEError) as excinfo:
        registry.register("adapter", "mlp", "1", MLPAdapter())
    assert excinfo.value.code == ErrorCode.VERSION_CONFLICT


def test_unknown_kind_and_malformed_ids_are_rejected():
    registry = Registry()
    with pytest.raises(WSEError):
        registry.register("plugin", "x", "1", object())
    with pytest.raises(WSEError):
        registry.register("operator", "Bad-Name", "1", object())
    with pytest.raises(WSEError):
        registry.register("operator", "ok", "latest", object())
    assert set(EXTENSION_KINDS) == {"adapter", "operator", "evaluator", "strategy"}


def test_adapter_registration_validates_contract():
    registry = Registry()

    class Incomplete:
        adapter_id = "incomplete"
        version = "1"

        def capabilities(self):
            return AdapterCapabilities(capabilities=frozenset())

    with pytest.raises(WSEError) as excinfo:
        registry.register("adapter", "incomplete", "1", Incomplete())
    assert "forward" in excinfo.value.details["missing_methods"]

    with pytest.raises(WSEError):
        registry.register("adapter", "mlp", "9", MLPAdapter())  # declares mlp@1


def test_registries_are_independent():
    first, second = builtin_registry(), builtin_registry()
    first.register("adapter", "scaled_mlp", "2", ScaleAdapter())
    assert first.resolve("adapter", "scaled_mlp", "2")
    with pytest.raises(WSEError):
        second.resolve("adapter", "scaled_mlp", "2")


def test_separately_supplied_adapter_works_without_core_changes(signed_sum):
    registry = builtin_registry()
    registry.register("adapter", "scaled_mlp", "2", ScaleAdapter())
    handle = load_model(
        signed_sum.reference_tensors(), "scaled_mlp@2", signed_sum.CONFIG, registry=registry
    )
    assert str(handle.adapter_ref) == "scaled_mlp@2"
    assert not handle.adapter.capabilities().supports("weights.slice_edit")
    logits = handle.forward(torch.tensor([[1.0, 0.5]]))
    assert torch.equal(logits, torch.tensor([[-3.0, 3.0]]))
    # The same weights under another adapter have a different model identity.
    plain = load_model(signed_sum.reference_tensors(), "mlp@1", signed_sum.CONFIG)
    assert plain.weights_hash == handle.weights_hash
    assert plain.identity != handle.identity
