"""Family descriptors: roles, head groupings, and protections (M2 plan Section 2.2, Task 8).

A family descriptor (``wse.adapters.families``) names the semantic roles of a
family's tensors with regular expressions over canonical names, groups the head
axis of attention projections from configuration keys, and lists the roles that
every policy refuses to edit. The built-in descriptors are ``llama@1`` and
``qwen3_5@1``. Snapshots are the tiny offline checkpoints of
``tests/hf_fixtures.py``; tests that need ``transformers`` request the
``hf_snapshots`` fixture, which skips when the ``models-hf`` extra is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from wse.adapters.families import (
    LLAMA,
    QWEN3_5,
    FamilyDescriptor,
    HeadGrouping,
    RoleRule,
    model_family,
    model_roles,
    role_tensor,
)
from wse.api import Engine, apply_operator, load_model
from wse.patches import materialize_patch
from wse.policy import SMALL_MODEL_POLICY_ID, default_operation_policy
from wse.registry import builtin_registry
from wse.schemas import ErrorCode, ExtensionRef, WSEError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_fixtures  # noqa: E402

REPO = "example-org/tiny-model"
REVISION = "0123456789abcdef0123456789abcdef01234567"
PROTECTED = ["embed", "final_norm", "lm_head"]
GLOBAL = ("embed", "lm_head", "final_norm")
FAMILIES = {"llama_untied": "llama@1", "llama_tied": "llama@1", "qwen3_5": "qwen3_5@1"}
LINEAR = ["out_features", "in_features"]


def _expect(code: ErrorCode, call, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        call(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


def _full(shape) -> list[dict[str, int]]:
    return [{"start": 0, "stop": size, "step": 1} for size in shape]


def _text_config(name: str) -> dict:
    return hf_fixtures.QWEN3_5_TEXT if name == "qwen3_5" else hf_fixtures.LLAMA


def _import(engine: Engine, snapshot: Path, **kwargs):
    return engine.import_hf_snapshot(snapshot, repo=REPO, revision=REVISION, **kwargs)


@pytest.fixture(scope="module")
def hf_workspace(hf_snapshots, tmp_path_factory) -> tuple[Path, dict[str, str]]:
    workspace = tmp_path_factory.mktemp("families") / "workspace"
    with Engine(workspace) as engine:
        ids = {name: _import(engine, hf_snapshots[name]).checkpoint_id
               for name in ("llama_untied", "llama_tied", "qwen3_5", "qwen3")}
    return workspace, ids


@pytest.fixture(scope="module")
def handles(hf_workspace):
    workspace, ids = hf_workspace
    with Engine(workspace) as engine:
        return {name: engine.open_checkpoint(checkpoint_id) for name, checkpoint_id in ids.items()}


def _expected_shapes(name: str) -> dict[str, tuple[list[int], list[str]]]:
    """Each role's shape and axes, derived from the fixture's configuration."""
    text = _text_config(name)
    hidden, intermediate, vocab = text["hidden_size"], text["intermediate_size"], text["vocab_size"]
    heads, kv_heads = text["num_attention_heads"], text["num_key_value_heads"]
    head_dim = text.get("head_dim", hidden // heads)
    gate = 2 if name == "qwen3_5" else 1  # Qwen3.5's q_proj holds each head's query and output gate
    shapes = {
        "embed": ([vocab, hidden], ["num_embeddings", "embedding_dim"]),
        "lm_head": ([vocab, hidden], LINEAR),
        "final_norm": ([hidden], ["features"]),
        "attn.q": ([gate * heads * head_dim, hidden], LINEAR),
        "attn.k": ([kv_heads * head_dim, hidden], LINEAR),
        "attn.v": ([kv_heads * head_dim, hidden], LINEAR),
        "attn.o": ([hidden, heads * head_dim], LINEAR),
        "mlp.gate": ([intermediate, hidden], LINEAR),
        "mlp.up": ([intermediate, hidden], LINEAR),
        "mlp.down": ([hidden, intermediate], LINEAR),
        "norm.input": ([hidden], ["features"]),
        "norm.post_attention": ([hidden], ["features"]),
    }
    if name == "qwen3_5":
        key_dim = text["linear_num_key_heads"] * text["linear_key_head_dim"]
        value_dim = text["linear_num_value_heads"] * text["linear_value_head_dim"]
        shapes["linear_attn.in"] = ([2 * key_dim + value_dim, hidden], LINEAR)  # queries, keys, values
        shapes["linear_attn.out"] = ([hidden, value_dim], LINEAR)
    return shapes


def _expected_layers(name: str) -> dict[str, list[int | None]]:
    """The layers each role appears in: attention roles follow the configuration's layer types."""
    text = _text_config(name)
    layers = list(range(text["num_hidden_layers"]))
    types = text.get("layer_types", ["full_attention"] * len(layers))
    expected: dict[str, list[int | None]] = {role: [None] for role in GLOBAL}
    for role in ("mlp.gate", "mlp.up", "mlp.down", "norm.input", "norm.post_attention"):
        expected[role] = layers
    for role in ("attn.q", "attn.k", "attn.v", "attn.o"):
        expected[role] = [layer for layer in layers if types[layer] == "full_attention"]
    if name == "qwen3_5":
        for role in ("linear_attn.in", "linear_attn.out"):
            expected[role] = [layer for layer in layers if types[layer] == "linear_attention"]
    return expected


# -- roles --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_every_role_rule_resolves_to_tensors_with_expected_shapes(name, handles, hf_workspace):
    handle = handles[name]
    descriptor = builtin_registry().resolve_ref("family", FAMILIES[name])
    assert model_family(handle) is descriptor
    roles = model_roles(handle)
    # Every rule resolves, in exactly the layers that the configuration gives its role.
    assert {rule.role for rule in descriptor.roles} == {role.role for role in roles}
    layers: dict[str, list[int | None]] = {}
    for role in roles:
        layers.setdefault(role.role, []).append(role.layer)
    assert layers == _expected_layers(name)
    expected = _expected_shapes(name)
    infos = {info.name: info for info in handle.architecture.tensors}
    for role in roles:
        assert (role.shape, role.axes) == expected[role.role], role
        info = infos[role.tensor]
        assert (role.shape, role.axes, role.module_name) == (info.shape, info.axes, info.module_name)
        assert list(handle.tensor(role.tensor).shape) == role.shape
        assert role.protected == (role.role in PROTECTED)
    # Tensors without a role stay readable and editable by name.
    unassigned = sorted(tensor.split(".layers.")[-1] for tensor in set(infos) - {role.tensor for role in roles})
    if name == "qwen3_5":
        assert unassigned == ["0.linear_attn.A_log", "0.linear_attn.conv1d.weight", "0.linear_attn.dt_bias",
                              "0.linear_attn.in_proj_a.weight", "0.linear_attn.in_proj_b.weight",
                              "0.linear_attn.in_proj_z.weight", "0.linear_attn.norm.weight",
                              "1.self_attn.k_norm.weight", "1.self_attn.q_norm.weight"]
    else:
        assert unassigned == []
    # Canonical names stay the checkpoint names; module names are those of the transformers module.
    down = role_tensor(handle, "mlp.down", layer=1)
    prefix = "model.language_model." if name == "qwen3_5" else "model."
    assert down.tensor == f"{prefix}layers.1.mlp.down_proj.weight"
    assert down.site == f"{prefix}layers.1.mlp.down_proj"  # the module whose input is a probe's role site
    assert down.module_name == "model.layers.1.mlp.down_proj.weight"
    # The manifest alone gives the same roles, without loading any weight.
    workspace, ids = hf_workspace
    with Engine(workspace) as engine:
        assert model_roles(engine.load_manifest(ids[name]), registry=engine.registry) == roles


@pytest.mark.parametrize("name", ["llama_untied", "qwen3_5"])
def test_head_groups_partition_the_head_axis(name, handles):
    handle = handles[name]
    text = _text_config(name)
    heads, kv_heads = text["num_attention_heads"], text["num_key_value_heads"]
    head_dim = text.get("head_dim", text["hidden_size"] // heads)
    expected = {"attn.q": (0, heads, (2 if name == "qwen3_5" else 1) * head_dim), "attn.k": (0, kv_heads, head_dim),
                "attn.v": (0, kv_heads, head_dim), "attn.o": (1, heads, head_dim)}
    if name == "qwen3_5":
        expected["linear_attn.out"] = (1, text["linear_num_value_heads"], text["linear_value_head_dim"])
    grouped = [role for role in model_roles(handle) if role.heads is not None]
    assert {role.role for role in grouped} == set(expected)
    for role in grouped:
        layout = role.heads
        assert (layout.axis, layout.num_heads, layout.size) == expected[role.role], role
        ranges = [layout.head_range(head) for head in range(layout.num_heads)]
        # Consecutive, nonempty, and disjoint ranges that cover the whole axis.
        assert ranges[0][0] == 0 and ranges[-1][1] == role.shape[layout.axis]
        assert all(start < stop for start, stop in ranges)
        assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
        for head, (start, stop) in enumerate(ranges):
            selection = role.selection(head=head)
            expected_selection = _full(role.shape)
            expected_selection[layout.axis] = {"start": start, "stop": stop, "step": 1}
            assert selection == expected_selection
        for bad in (-1, layout.num_heads, True):
            _expect(ErrorCode.INVALID_ARGUMENT, role.selection, head=bad)
    down = role_tensor(handle, "mlp.down", layer=0)
    assert down.heads is None and down.selection() == _full(down.shape)
    error = _expect(ErrorCode.INVALID_ARGUMENT, down.selection, head=0)
    assert "mlp.down" in error.message and "head" in error.message


def _inputs_of(handle, module_path: str, ids: torch.Tensor) -> torch.Tensor:
    """The input of one runtime module during a CPU forward pass over ``ids``."""
    runtime = handle.runtime("cpu")
    captured: list[torch.Tensor] = []
    hook = runtime.module.get_submodule(module_path).register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().float().clone()))
    try:
        runtime.logits(ids)
    finally:
        hook.remove()
    assert len(captured) == 1
    return captured[0]


def _changed_features(handle, candidate, module_path: str, ids: torch.Tensor) -> set[int]:
    before, after = _inputs_of(handle, module_path, ids), _inputs_of(candidate, module_path, ids)
    changed = (before != after).reshape(-1, before.shape[-1]).any(dim=0)
    return set(torch.nonzero(changed).reshape(-1).tolist())


def _zeroed(handle, tensor: str, selection) -> object:
    return apply_operator(handle, "scale_slice", {"tensor": tensor, "selection": selection, "factor": 0.0}).candidate


@pytest.mark.parametrize("name, layer", [("llama_untied", 0), ("qwen3_5", 1)])
def test_head_slices_select_the_heads_of_the_computation(name, layer, handles):
    """Zeroing one head's slice of a projection changes only that head's slice of the output projection's input."""
    handle = handles[name]
    ids = torch.randint(3, 40, (2, 7), generator=torch.Generator().manual_seed(0))
    query, value, output = (role_tensor(handle, role, layer=layer) for role in ("attn.q", "attn.v", "attn.o"))
    runtime_output = output.module_name.rpartition(".")[0]
    for head in range(query.heads.num_heads):
        changed = _changed_features(handle, _zeroed(handle, query.tensor, query.selection(head=head)),
                                    runtime_output, ids)
        assert changed and changed <= set(range(*output.heads.head_range(head))), (head, changed)
    # Grouped-query attention: key-value head g serves query heads g * group ... (g + 1) * group - 1.
    group = output.heads.num_heads // value.heads.num_heads
    for kv_head in range(value.heads.num_heads):
        changed = _changed_features(handle, _zeroed(handle, value.tensor, value.selection(head=kv_head)),
                                    runtime_output, ids)
        start, stop = output.heads.head_range(kv_head * group)[0], output.heads.head_range((kv_head + 1) * group - 1)[1]
        assert changed and changed <= set(range(start, stop)), (kv_head, changed)
    if name == "qwen3_5":  # the value heads of the linear-attention layer: zero one head's output gate
        out = role_tensor(handle, "linear_attn.out", layer=0)
        gate = "model.language_model.layers.0.linear_attn.in_proj_z.weight"  # [value_dim, hidden], no role
        for head in range(out.heads.num_heads):
            start, stop = out.heads.head_range(head)
            selection = [{"start": start, "stop": stop, "step": 1}, {"start": 0, "stop": 16, "step": 1}]
            changed = _changed_features(handle, _zeroed(handle, gate, selection), out.module_name.rpartition(".")[0],
                                        ids)
            assert changed and changed <= set(range(start, stop)), (head, changed)


def test_role_tensor_names_edit_targets_and_probe_sites(handles, signed_sum):
    qwen, llama = handles["qwen3_5"], handles["llama_untied"]
    attention = role_tensor(qwen, "attn.o", layer=1)
    assert attention.tensor == "model.language_model.layers.1.self_attn.o_proj.weight"
    assert attention.site == "model.language_model.layers.1.self_attn.o_proj"
    assert role_tensor(qwen, "embed").tensor == "model.language_model.embed_tokens.weight"
    assert role_tensor(qwen, "lm_head").tensor == "lm_head.weight"
    # A head named by role is an edit target: one head's input columns of attn.o.
    edit = apply_operator(qwen, "scale_slice", {"tensor": attention.tensor, "selection": attention.selection(head=2),
                                                "factor": 0.5})
    assert [entry.model_dump() for entry in edit.patch.steps[0].selection] == [
        {"start": 0, "stop": 16, "step": 1}, {"start": 8, "stop": 12, "step": 1}]
    # Layer 0 of the fixture is a linear-attention layer: it has no attn.o.
    error = _expect(ErrorCode.NOT_FOUND, role_tensor, qwen, "attn.o", layer=0)
    assert "attn.o" in error.message and error.details["layers"] == [1]
    error = _expect(ErrorCode.NOT_FOUND, role_tensor, llama, "linear_attn.in", layer=0)
    assert "llama@1" in error.message
    for role, layer in (("embed", 0), ("mlp.down", None), ("mlp.middle", 0)):
        _expect(ErrorCode.INVALID_ARGUMENT, role_tensor, llama, role, layer=layer)
    # Level 0 models (no family descriptor) and M1 models have no roles.
    assert model_family(handles["qwen3"]) is None and model_roles(handles["qwen3"]) == ()
    error = _expect(ErrorCode.NOT_FOUND, role_tensor, handles["qwen3"], "mlp.down", layer=0)
    assert "family descriptor" in error.message
    m1 = load_model(signed_sum.task_tensors(), "mlp@1", signed_sum.CONFIG)
    assert model_family(m1) is None and model_roles(m1) == ()


# -- protections --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_protected_roles_reject_edits_with_an_actionable_policy_error(name, handles):
    handle = handles[name]
    family = FAMILIES[name]
    descriptor = model_family(handle)
    assert descriptor.protected_roles == PROTECTED
    editable = sorted({rule.role for rule in descriptor.roles} - set(PROTECTED))
    projection = default_operation_policy().with_edit("project_out", "project_out@1")
    down = role_tensor(handle, "mlp.down", layer=1)
    for role in PROTECTED:
        target = role_tensor(handle, role)
        assert target.protected
        request = {"tensor": target.tensor, "selection": target.selection(), "factor": 2.0}
        for operator in ("scale_slice", "scale_slice@1"):
            error = _expect(ErrorCode.POLICY_DENIED, apply_operator, handle, operator, request)
            # What is refused, why, and what to edit instead.
            for fragment in (repr(target.tensor), role, family, "protect", "mlp.down"):
                assert fragment in error.message, (fragment, error.message)
            assert error.details == {"tensor": target.tensor, "role": role, "layer": None, "family": family,
                                     "protected_roles": PROTECTED, "editable_roles": editable}
        # Every edit operator is refused alike, before its parameters are checked.
        _expect(ErrorCode.POLICY_DENIED, apply_operator, handle, "project_out", {
            "tensor": target.tensor, "selection": target.selection(), "direction": [1.0] * target.shape[0],
            "columns": [0]}, policy=projection)
        # In a patch, the refused step keeps POLICY_DENIED and reports its position.
        first = [{"start": 0, "stop": 1, "step": 1}] * len(target.shape)
        error = _expect(ErrorCode.POLICY_DENIED, materialize_patch, handle, [
            {"op": "scale_slice", "tensor": down.tensor, "selection": down.selection(), "factor": 2.0},
            {"op": "set_slice", "tensor": target.tensor, "selection": first,
             "values": [[0.5]] if len(target.shape) == 2 else [0.5]},
        ])
        assert error.details["operation_index"] == 1 and error.details["role"] == role
    # Untied embeddings are no alias group: the protection alone refuses them.
    assert handle.architecture.alias_groups == ([] if name == "llama_untied" else [
        sorted([role_tensor(handle, "embed").tensor, "lm_head.weight"])])
    # Unprotected roles stay editable.
    edit = apply_operator(handle, "scale_slice", {"tensor": down.tensor, "selection": down.selection(), "factor": 2.0})
    assert edit.patch.steps[0].changed_elements > 0


def test_protections_hold_in_managed_previews_and_only_for_families(hf_snapshots, engine):
    base = _import(engine, hf_snapshots["llama_untied"])
    grant = engine.grant_run("families_v1", SMALL_MODEL_POLICY_ID, base.checkpoint_id)
    experiment = engine.create_experiment(grant.run_id, "families_v1", SMALL_MODEL_POLICY_ID, "create-1")

    def spec(tensor: str, shape: list[int]) -> dict:
        return {"base_checkpoint_id": base.checkpoint_id, "hypothesis": "Scaling this tensor changes the output.",
                "operations": [{"op": "scale_slice", "tensor": tensor, "selection": _full(shape), "factor": 2.0}]}

    for tensor in ("model.norm.weight", "lm_head.weight", "model.embed_tokens.weight"):
        shape = next(info.shape for info in base.tensors if info.name == tensor)
        error = _expect(ErrorCode.POLICY_DENIED, engine.preview_patch, experiment.experiment_id, spec(tensor, shape),
                        f"preview-{tensor}")
        assert tensor in error.message and "llama@1" in error.message and error.details["operation_index"] == 0
    preview = engine.preview_patch(experiment.experiment_id, spec("model.layers.1.mlp.down_proj.weight", [16, 32]),
                                   "preview-down")
    assert preview.changed_elements > 0
    # A Level 0 model has no roles, so only its alias groups are protected.
    level0 = _import(engine, hf_snapshots["qwen3"])
    handle = engine.open_checkpoint(level0.checkpoint_id)
    edit = apply_operator(handle, "scale_slice", {"tensor": "model.norm.weight", "selection": _full([16]),
                                                  "factor": 2.0})
    assert edit.patch.steps[0].changed_elements == 16


# -- descriptors as registered, versioned data --------------------------------------------------------


def _qwen3_descriptor(version: str = "1") -> FamilyDescriptor:
    """A descriptor supplied by the operator for Qwen3 checkpoints, which have no built-in descriptor."""
    layer = r"model\.layers\.(?P<layer>\d+)\."
    return FamilyDescriptor(
        family_id="qwen3_operator", version=version, model_types=["qwen3"],
        roles=[
            RoleRule(pattern=r"model\.embed_tokens\.weight", role="embed"),
            RoleRule(pattern=r"lm_head\.weight", role="lm_head"),
            RoleRule(pattern=r"model\.norm\.weight", role="final_norm"),
            RoleRule(pattern=layer + r"self_attn\.o_proj\.weight", role="attn.o",
                     heads=HeadGrouping(num_heads_key="num_attention_heads", head_dim_key="head_dim", axis=1)),
            RoleRule(pattern=layer + r"mlp\.down_proj\.weight", role="mlp.down"),
        ],
        protected_roles=["lm_head", "mlp.down"],  # protections are the descriptor's data, not fixed roles
    )


def test_a_separately_supplied_descriptor_registers_without_core_changes(hf_snapshots, tmp_path):
    registry = builtin_registry()
    descriptor = _qwen3_descriptor()
    registry.register("family", "qwen3_operator", "1", descriptor)
    assert builtin_registry().list("family") == [ExtensionRef.parse("llama@1"), ExtensionRef.parse("qwen3_5@1")]
    with Engine(tmp_path / "operator", registry=registry) as engine:
        manifest = _import(engine, hf_snapshots["qwen3"])
        assert manifest.config["family"] == "qwen3_operator@1"
        assert manifest.source.family == ExtensionRef.parse("qwen3_operator@1")
        handle = engine.open_checkpoint(manifest.checkpoint_id)
    assert model_family(handle) is descriptor
    roles = model_roles(handle)
    assert [(role.role, role.layer) for role in roles] == [
        ("embed", None), ("lm_head", None), ("final_norm", None),
        ("attn.o", 0), ("mlp.down", 0), ("attn.o", 1), ("mlp.down", 1)]
    output = role_tensor(handle, "attn.o", layer=1)
    assert (output.heads.axis, output.heads.num_heads, output.heads.size) == (1, 4, 4)
    # The descriptor's protections apply: mlp.down is refused, and the final norm (unprotected here) is not.
    down = role_tensor(handle, "mlp.down", layer=0)
    error = _expect(ErrorCode.POLICY_DENIED, apply_operator, handle, "scale_slice",
                    {"tensor": down.tensor, "selection": down.selection(), "factor": 2.0})
    assert "qwen3_operator@1" in error.message and error.details["layer"] == 0
    norm = role_tensor(handle, "final_norm")
    assert apply_operator(handle, "scale_slice", {"tensor": norm.tensor, "selection": norm.selection(),
                                                  "factor": 2.0}).patch.steps[0].changed_elements == 16
    # Roles change nothing else: the weights and the module are the Level 0 import's.
    with Engine(tmp_path / "default") as engine:
        level0 = engine.open_checkpoint(_import(engine, hf_snapshots["qwen3"]).checkpoint_id)
    assert level0.source.family is None and model_roles(level0) == ()
    assert handle.weights_hash == level0.weights_hash and handle.config_hash != level0.config_hash
    ids = torch.randint(3, 40, (2, 7), generator=torch.Generator().manual_seed(1))
    assert torch.equal(handle.runtime("cpu").logits(ids), level0.runtime("cpu").logits(ids))


def test_descriptors_are_versioned_and_recorded_in_manifests(hf_snapshots, tmp_path):
    assert (LLAMA.ref, QWEN3_5.ref) == (ExtensionRef.parse("llama@1"), ExtensionRef.parse("qwen3_5@1"))
    with pytest.raises(ValidationError):  # a version is exact: never "latest"
        FamilyDescriptor.model_validate({**LLAMA.model_dump(), "version": "latest"})
    registry = builtin_registry()
    stricter = FamilyDescriptor.model_validate({**LLAMA.model_dump(), "version": "2",
                                                "protected_roles": [*PROTECTED, "mlp.down"]})
    registry.register("family", "llama", "2", stricter)
    with Engine(tmp_path / "workspace", registry=registry) as engine:
        # Two registered versions accept model type "llama": an import must name one, and nothing is stored.
        error = _expect(ErrorCode.INVALID_ARGUMENT, _import, engine, hf_snapshots["llama_untied"])
        assert "llama@1" in error.message and "llama@2" in error.message
        assert engine.store.list_ready() == []
        first = _import(engine, hf_snapshots["llama_untied"], family="llama@1")
        second = _import(engine, hf_snapshots["llama_untied"], family="llama@2")
        for manifest, family in ((first, "llama@1"), (second, "llama@2")):
            stored = engine.load_manifest(manifest.checkpoint_id)
            assert stored.config["family"] == family and stored.source.family == ExtensionRef.parse(family)
        # The descriptor version is part of the identity: same weights, different configuration hash.
        assert first.checkpoint_id != second.checkpoint_id
        assert first.weights_hash == second.weights_hash and first.config_hash != second.config_hash
        handles = {family: engine.open_checkpoint(manifest.checkpoint_id)
                   for family, manifest in (("llama@1", first), ("llama@2", second))}
        # Each checkpoint's roles and protections follow the version its manifest records.
        down = {family: role_tensor(handle, "mlp.down", layer=0) for family, handle in handles.items()}
        assert not down["llama@1"].protected and down["llama@2"].protected
        request = {"tensor": down["llama@2"].tensor, "selection": down["llama@2"].selection(), "factor": 2.0}
        assert apply_operator(handles["llama@1"], "scale_slice", request).patch.steps[0].changed_elements > 0
        error = _expect(ErrorCode.POLICY_DENIED, apply_operator, handles["llama@2"], "scale_slice", request)
        assert "llama@2" in error.message
    # A recorded version is never substituted: without llama@2, its checkpoint does not open.
    with Engine(tmp_path / "workspace") as engine:
        engine.open_checkpoint(first.checkpoint_id)
        error = _expect(ErrorCode.NOT_FOUND, engine.open_checkpoint, second.checkpoint_id)
        assert "llama@2" in error.message


def test_descriptor_rules_are_validated_when_constructed():
    layer = r"model\.layers\.(?P<layer>\d+)\.mlp\.down_proj\.weight"
    assert RoleRule(pattern=layer, role="mlp.down").role == "mlp.down"
    invalid = [
        dict(pattern=r"model\.layers\.\d+\.mlp\.down_proj\.weight", role="mlp.down"),  # a layer role needs the group
        dict(pattern=r"model\.(?P<layer>\d+)\.norm\.weight", role="final_norm"),  # a model-wide role has none
        dict(pattern=r"model\.layers\.(?P<layer>\d+\.", role="mlp.down"),  # not a regular expression
        dict(pattern=layer, role="mlp.middle"),  # not a role
    ]
    for arguments in invalid:
        with pytest.raises(ValidationError):
            RoleRule(**arguments)
    with pytest.raises(ValidationError):
        HeadGrouping(num_heads_key="num_attention_heads", head_dim_key="head_dim", axis=1, head_dim_factor=0)
    base = dict(family_id="checked", version="1", model_types=["llama"],
                roles=[RoleRule(pattern=layer, role="mlp.down")])
    assert FamilyDescriptor(**base, protected_roles=["mlp.down"]).protected_roles == ["mlp.down"]
    for protected in (["final_norm"], ["not_a_role"]):  # a protected role needs a rule
        with pytest.raises(ValidationError):
            FamilyDescriptor(**base, protected_roles=protected)


@pytest.mark.parametrize("label, rules, fragment", [
    ("heads", [RoleRule(pattern=r"model\.layers\.(?P<layer>\d+)\.self_attn\.o_proj\.weight", role="attn.o",
                        heads=HeadGrouping(num_heads_key="num_key_value_heads", head_dim_key="head_dim", axis=1))],
     "num_key_value_heads"),
    ("missing_key", [RoleRule(pattern=r"model\.layers\.(?P<layer>\d+)\.self_attn\.o_proj\.weight", role="attn.o",
                              heads=HeadGrouping(num_heads_key="num_heads", head_dim_key="head_dim", axis=1))],
     "num_heads"),
    ("axis", [RoleRule(pattern=r"model\.layers\.(?P<layer>\d+)\.input_layernorm\.weight", role="norm.input",
                       heads=HeadGrouping(num_heads_key="num_attention_heads", head_dim_key="head_dim", axis=1))],
     "axis 1"),
    ("two_rules", [RoleRule(pattern=r"model\.layers\.(?P<layer>\d+)\.mlp\.down_proj\.weight", role="mlp.down"),
                   RoleRule(pattern=r"model\.layers\.(?P<layer>\d+)\.mlp\.\w+\.weight", role="mlp.up")],
     "mlp.down"),
    ("two_tensors", [RoleRule(pattern=r"model\.layers\.(?P<layer>\d+)\.mlp\.(gate|up)_proj\.weight", role="mlp.up")],
     "gate_proj"),
])
def test_a_descriptor_that_does_not_fit_the_checkpoint_is_refused_at_import(label, rules, fragment, hf_snapshots,
                                                                             tmp_path, monkeypatch):
    from wse.hf_snapshot import SnapshotReader

    registry = builtin_registry()
    registry.register("family", "misfit", "1", FamilyDescriptor(family_id="misfit", version="1",
                                                                  model_types=["llama"], roles=rules))

    def never(*args, **kwargs):
        raise AssertionError("the roles are checked before any tensor data is read")

    monkeypatch.setattr(SnapshotReader, "open_weights", never)
    with Engine(tmp_path / label, registry=registry) as engine:
        error = _expect(ErrorCode.INVALID_MODEL, _import, engine, hf_snapshots["llama_untied"], family="misfit@1")
        assert "misfit@1" in error.message and fragment in error.message, error.message
        assert engine.store.list_ready() == []
