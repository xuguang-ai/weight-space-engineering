"""Explicit, exact-version extension registration."""

from __future__ import annotations

import pytest
import torch

from wse.adapters.base import CAP_WEIGHTS_READ, AdapterCapabilities, ModelAdapter
from wse.adapters.mlp import MLPAdapter
from wse.api import load_model
from wse.registry import EXTENSION_KINDS, Registry, builtin_registry
from wse.operators.base import BaseAnalysisOperator, OperatorParameters, OperatorResult, make_spec
from wse.operators.tensor import TopKParameters, TopKResult
from wse.schemas import ErrorCode, ExtensionRef, OperatorSpec, PatchStep, UsageContract, WSEError


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
    assert registry.list("operator") == [
        ExtensionRef.parse("scale_slice@1"),
        ExtensionRef.parse("set_slice@1"),
        ExtensionRef.parse("tensor_norm@1"),
        ExtensionRef.parse("tensor_stats@1"),
        ExtensionRef.parse("tensor_top_k@1"),
    ]


def test_builtin_edits_are_registered_through_operator_specs():
    registry = builtin_registry()
    for ref in ("set_slice@1", "scale_slice@1"):
        spec = registry.resolve_ref("operator", ref).spec
        assert spec.effect == "edit"
        assert spec.required_capabilities == sorted([CAP_WEIGHTS_READ, "weights.slice_edit"])
        assert spec.usage == UsageContract(counter="candidate_commits", units_per_call=1, model_forward_calls=0)
        assert spec.output_schema == PatchStep.model_json_schema()


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


def test_aliases_are_declared_explicitly_and_resolve_exactly():
    registry = builtin_registry()
    assert registry.resolve_reference("adapter", "mlp") == ExtensionRef.parse("mlp@1")
    assert registry.resolve_reference("adapter", "mlp@1") == ExtensionRef.parse("mlp@1")
    assert registry.aliases("adapter") == {"mlp": ExtensionRef.parse("mlp@1")}

    class MLPv2(MLPAdapter):
        version = "2"

    registry.register("adapter", "mlp", "2", MLPv2())
    assert registry.resolve_reference("adapter", "mlp") == ExtensionRef.parse("mlp@1")
    registry.register_alias("adapter", "mlp", "mlp@1")  # identical declaration: no-op
    for alias, target, code in (
        ("mlp", "mlp@2", ErrorCode.VERSION_CONFLICT),  # aliases are immutable
        ("fresh", "mlp@9", ErrorCode.NOT_FOUND),  # target must be registered
        ("Bad-Alias", "mlp@1", ErrorCode.INVALID_ARGUMENT),
        ("fresh", "mlp", ErrorCode.INVALID_ARGUMENT),  # target must be exact
    ):
        with pytest.raises(WSEError) as excinfo:
            registry.register_alias("adapter", alias, target)
        assert excinfo.value.code == code, (alias, target)
    for reference, code in (("other", ErrorCode.NOT_FOUND), ("mlp@9", ErrorCode.NOT_FOUND),
                            ("mlp@x", ErrorCode.INVALID_ARGUMENT)):
        with pytest.raises(WSEError) as excinfo:
            registry.resolve_reference("adapter", reference)
        assert excinfo.value.code == code, reference


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


# -- operators ---------------------------------------------------------------------------


def test_builtin_analyses_are_registered_through_operator_specs():
    registry = builtin_registry()
    for ref in ("tensor_stats@1", "tensor_norm@1", "tensor_top_k@1"):
        operator = registry.resolve_ref("operator", ref)
        spec = operator.spec
        assert isinstance(spec, OperatorSpec) and str(spec.ref) == ref
        assert spec.effect == "analysis"
        assert spec.required_capabilities == [CAP_WEIGHTS_READ]
        assert spec.usage == UsageContract(counter="analysis_calls", units_per_call=1, model_forward_calls=0)
        assert spec.input_schema == operator.parameters_model.model_json_schema()
        assert spec.output_schema == operator.result_model.model_json_schema()
    top_k = registry.resolve_ref("operator", "tensor_top_k@1").spec
    assert top_k.input_schema == TopKParameters.model_json_schema()
    assert "k" in top_k.input_schema["properties"]
    assert top_k.output_schema == TopKResult.model_json_schema()
    stats = registry.resolve_ref("operator", "tensor_stats@1").spec
    assert stats.input_schema["properties"] == {}
    with pytest.raises(WSEError) as excinfo:
        registry.resolve("operator", "tensor_stats", "2")
    assert excinfo.value.code == ErrorCode.NOT_FOUND


class _Result(OperatorResult):
    value: float


def _spec(operator_id="custom", version="1", effect="analysis", parameters=OperatorParameters, result=_Result,
          usage=None):
    return make_spec(operator_id=operator_id, version=version, effect=effect, description="custom op",
                     required_capabilities=[CAP_WEIGHTS_READ], parameters_model=parameters,
                     result_model=result, usage=usage or UsageContract(counter="analysis_calls"))


class _Custom(BaseAnalysisOperator):
    result_model = _Result
    spec = _spec()

    def analyze(self, region, parameters):
        return _Result(value=0.0)


def test_separately_supplied_operator_registers_without_core_changes():
    registry = builtin_registry()
    registry.register("operator", "custom", "1", _Custom())
    assert ExtensionRef.parse("custom@1") in registry.list("operator")
    with pytest.raises(WSEError) as excinfo:
        registry.register("operator", "custom", "1", _Custom())
    assert excinfo.value.code == ErrorCode.VERSION_CONFLICT


def test_operator_registration_validates_the_contract():
    registry = Registry()

    def rejected(implementation, extension_id="custom", version="1"):
        with pytest.raises(WSEError) as excinfo:
            registry.register("operator", extension_id, version, implementation)
        assert excinfo.value.code == ErrorCode.INVALID_ARGUMENT
        return excinfo.value

    rejected(object())  # no spec
    rejected(_Custom(), version="2")  # spec declares custom@1
    rejected(_Custom(), extension_id="other")

    class NoAnalyze:
        spec = _spec()
        parameters_model = OperatorParameters
        result_model = _Result

        def resolve_parameters(self, parameters, element_count):
            return OperatorParameters()

    assert "analyze" in rejected(NoAnalyze()).details["missing_methods"]

    class DriftingSchema(_Custom):
        spec = _spec(parameters=TopKParameters)  # advertises k but validates no parameters

    rejected(DriftingSchema())

    class NotPydantic(_Custom):
        result_model = dict

    rejected(NotPydantic())

    class EditWithoutApply(_Custom):
        result_model = PatchStep
        spec = _spec(effect="edit", result=PatchStep, usage=UsageContract(counter="candidate_commits"))

    assert "apply" in rejected(EditWithoutApply()).details["missing_methods"]

    class EditWithOwnOutput(EditWithoutApply):
        result_model = _Result
        spec = _spec(effect="edit", usage=UsageContract(counter="candidate_commits"))

        def apply(self, region, parameters):  # pragma: no cover - never registered
            return region.values

    assert "PatchStep" in rejected(EditWithOwnOutput()).message


def test_operator_usage_contract_must_match_the_effect():
    registry = Registry()

    def rejected(implementation):
        with pytest.raises(WSEError) as excinfo:
            registry.register("operator", "custom", "1", implementation)
        assert excinfo.value.code == ErrorCode.INVALID_ARGUMENT
        return excinfo.value.message

    class ChargesCommits(_Custom):
        spec = _spec(usage=UsageContract(counter="candidate_commits"))

    assert "analysis_calls" in rejected(ChargesCommits())

    class MayRunForward(_Custom):
        spec = _spec(usage=UsageContract(counter="analysis_calls", model_forward_calls=None))

    assert "model_forward_calls=0" in rejected(MayRunForward())

    class EditChargedAsAnalysis(_Custom):
        result_model = PatchStep
        spec = _spec(effect="edit", result=PatchStep)

        def apply(self, region, parameters):  # pragma: no cover - never registered
            return region.values

    assert "candidate_commits" in rejected(EditChargedAsAnalysis())

    class EditChargedFiveCommits(EditChargedAsAnalysis):
        spec = _spec(effect="edit", result=PatchStep,
                     usage=UsageContract(counter="candidate_commits", units_per_call=5))

    # The managed path charges exactly one candidate commit per applied patch.
    assert "units_per_call=1" in rejected(EditChargedFiveCommits())

    class ChargesReads(_Custom):
        spec = _spec(usage=UsageContract(counter="tensor_read_calls", units_per_call=3))

    registry.register("operator", "custom", "1", ChargesReads())


def test_operator_spec_rejects_malformed_identity_and_effects():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _spec(operator_id="Bad-Name")
    with pytest.raises(ValidationError):
        _spec(version="latest")
    with pytest.raises(ValidationError):
        _spec(effect="train")
    with pytest.raises(ValidationError):
        UsageContract(counter="free_lunch")
