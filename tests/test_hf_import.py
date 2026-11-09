"""Generic import of Hugging Face causal LMs, model identity, and the Qwen3.5 name map (M2 plan Task 3).

Snapshots are tiny random checkpoints built offline by ``tests/hf_fixtures.py``.
Tests that need ``transformers`` request the ``snapshots`` fixture, which skips
when the ``models-hf`` extra is not installed.
"""

from __future__ import annotations

import builtins
import io
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from wse.api import Engine
from wse.schemas import ErrorCode, ExtensionRef, WSEError
from wse.tensors import blob_id, blob_weights_hash, weights_hash

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_fixtures  # noqa: E402

REPO = "example-org/tiny-model"
REVISION = "0123456789abcdef0123456789abcdef01234567"
SKIP_REASON = "the models-hf extra (transformers) is not installed"


@pytest.fixture(scope="session")
def snapshots(tmp_path_factory) -> dict[str, Path]:
    pytest.importorskip("transformers", reason=SKIP_REASON)
    return hf_fixtures.build_all(tmp_path_factory.mktemp("hf_snapshots"))


@pytest.fixture
def copy_snapshot(snapshots, tmp_path):
    def copy(name: str, label: str | None = None) -> Path:
        destination = tmp_path / "copies" / (label or name)
        shutil.copytree(snapshots[name], destination)
        return destination

    return copy


def _import(engine: Engine, snapshot: Path, **kwargs):
    return engine.import_hf_snapshot(snapshot, repo=kwargs.pop("repo", REPO),
                                     revision=kwargs.pop("revision", REVISION), **kwargs)


def _module_parameters(model_class: str, config) -> dict[str, object]:
    import transformers

    with torch.device("meta"):
        module = getattr(transformers, model_class)(config)
    return dict(module.named_parameters(remove_duplicate=False))


def _blob_files(workspace: Path) -> list[str]:
    return sorted(path.name for path in (workspace / "artifacts" / "blobs").iterdir())


def _expect_error(code: ErrorCode, call, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        call(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


# -- safe import -----------------------------------------------------------------------------------


def test_only_allow_listed_files_are_read(snapshots, engine, monkeypatch):
    from wse.hf_snapshot import is_allowed

    snapshot = snapshots["llama_tied"].resolve()
    opened: list[str] = []
    real_open, real_os_open = builtins.open, os.open

    def record(path) -> None:
        try:
            resolved = Path(os.fsdecode(path)).resolve()
        except (TypeError, ValueError):
            return
        if resolved.is_relative_to(snapshot):
            opened.append(resolved.relative_to(snapshot).as_posix())

    def spy_open(file, *args, **kwargs):
        if not isinstance(file, int):
            record(file)
        return real_open(file, *args, **kwargs)

    def spy_os_open(path, *args, **kwargs):
        record(path)
        return real_os_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(io, "open", spy_open)
    monkeypatch.setattr(os, "open", spy_os_open)
    manifest = _import(engine, snapshot)
    monkeypatch.undo()

    read = sorted(manifest.source.files)
    assert read == ["chat_template.jinja", "config.json", "generation_config.json", "model.safetensors",
                    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]
    assert all(is_allowed(name) for name in read)
    for decoy in hf_fixtures.DECOYS:
        assert decoy not in read and decoy not in opened
        assert not is_allowed(decoy)
    assert set(opened) <= set(read)
    # Every file read is recorded with its SHA-256.
    for name, digest in manifest.source.files.items():
        import hashlib

        assert digest == "sha256:" + hashlib.sha256((snapshot / name).read_bytes()).hexdigest()


def test_a_snapshot_with_only_pickled_weights_is_rejected(snapshots, engine):
    from wse.adapters.hf_causal_lm import read_snapshot

    error = _expect_error(ErrorCode.INVALID_MODEL, _import, engine, snapshots["pickled_only"])
    assert "safetensors" in error.message and "pickle" in error.message
    with pytest.raises(WSEError) as excinfo:
        read_snapshot(snapshots["pickled_only"], repo=REPO, revision=REVISION, registry=engine.registry)
    assert "pytorch_model.bin" not in excinfo.value.details.get("read_files", [])
    assert engine.store.list_ready() == []


def test_auto_map_in_config_is_rejected(copy_snapshot, engine):
    for nested in (False, True):
        snapshot = copy_snapshot("llama_untied", f"auto_map_{nested}")
        config = json.loads((snapshot / "config.json").read_text())
        auto_map = {"AutoModelForCausalLM": "modeling_custom.CustomForCausalLM"}
        if nested:
            config["rope_scaling"] = {"rope_type": "default", "auto_map": auto_map}
        else:
            config["auto_map"] = auto_map
        (snapshot / "config.json").write_text(json.dumps(config))
        error = _expect_error(ErrorCode.INVALID_MODEL, _import, engine, snapshot)
        assert "auto_map" in error.message and "remote code" in error.message
    assert engine.store.list_ready() == []


def test_an_architecture_transformers_does_not_implement_is_rejected(copy_snapshot, engine):
    snapshot = copy_snapshot("llama_untied", "unknown")
    config = json.loads((snapshot / "config.json").read_text())
    (snapshot / "config.json").write_text(json.dumps({**config, "model_type": "wse_unknown_architecture",
                                                      "architectures": ["WseUnknownForCausalLM"]}))
    error = _expect_error(ErrorCode.INVALID_MODEL, _import, engine, snapshot)
    assert "wse_unknown_architecture" in error.message and "transformers" in error.message

    from transformers import T5Config

    snapshot = copy_snapshot("llama_untied", "encoder_decoder")
    (snapshot / "config.json").write_text(json.dumps(T5Config(vocab_size=64, d_model=16).to_dict()))
    error = _expect_error(ErrorCode.INVALID_MODEL, _import, engine, snapshot)
    assert "encoder" in error.message

    snapshot = copy_snapshot("llama_untied", "quantized")
    (snapshot / "config.json").write_text(json.dumps({**config, "quantization_config": {"quant_method": "gptq"}}))
    error = _expect_error(ErrorCode.INVALID_MODEL, _import, engine, snapshot)
    assert "quantiz" in error.message
    assert engine.store.list_ready() == []


def test_unsupported_weight_dtypes_are_rejected(copy_snapshot, engine):
    snapshot = copy_snapshot("llama_untied", "float64")
    tensors = load_file(str(snapshot / "model.safetensors"))
    tensors["model.norm.weight"] = tensors["model.norm.weight"].to(torch.float64)
    save_file(tensors, str(snapshot / "model.safetensors"), metadata={"format": "pt"})
    error = _expect_error(ErrorCode.INVALID_MODEL, _import, engine, snapshot)
    assert "model.norm.weight" in error.message and "F64" in error.message


# -- layout ----------------------------------------------------------------------------------------


def test_axes_follow_module_types(snapshots, engine):
    llama = _import(engine, snapshots["llama_untied"])
    info = {t.name: t for t in llama.tensors}
    q = info["model.layers.0.self_attn.q_proj.weight"]
    assert (q.shape, q.axes) == ([16, 16], ["out_features", "in_features"])
    k = info["model.layers.1.self_attn.k_proj.weight"]
    assert (k.shape, k.axes) == ([8, 16], ["out_features", "in_features"])
    embed = info["model.embed_tokens.weight"]
    assert (embed.shape, embed.axes) == ([64, 16], ["num_embeddings", "embedding_dim"])
    head = info["lm_head.weight"]
    assert (head.shape, head.axes) == ([64, 16], ["out_features", "in_features"])
    for name in ("model.norm.weight", "model.layers.0.input_layernorm.weight"):
        assert (info[name].shape, info[name].axes) == ([16], ["features"])
    assert {t.dtype for t in llama.tensors} == {"bfloat16"}

    qwen = {t.name: t for t in _import(engine, snapshots["qwen3_5"]).tensors}
    prefix = "model.language_model.layers.0.linear_attn."
    conv = qwen[prefix + "conv1d.weight"]
    assert conv.axes == ["out_channels", "in_channels_per_group", "kernel"] and conv.shape[1] == 1
    assert qwen[prefix + "norm.weight"].axes == ["features"]  # a gated RMS norm
    # Parameters of other module types are readable with unknown axes.
    assert qwen[prefix + "A_log"].axes == ["unknown"]
    assert qwen[prefix + "dt_bias"].axes == ["unknown"]
    gemma = {t.name: t for t in _import(engine, snapshots["gemma3"]).tensors}
    # A subclass of nn.Embedding is an embedding.
    assert gemma["model.embed_tokens.weight"].axes == ["num_embeddings", "embedding_dim"]


def test_tied_embeddings_form_one_alias_group_and_one_blob(snapshots, engine, workspace):
    tied_file = load_file(str(snapshots["llama_tied"] / "model.safetensors"))
    assert "lm_head.weight" not in tied_file  # save_pretrained stores a tied group once
    tied = _import(engine, snapshots["llama_tied"])
    assert tied.alias_groups == [["lm_head.weight", "model.embed_tokens.weight"]]
    names = [t.name for t in tied.tensors]
    assert names == sorted(names) and len(names) == len(tied_file) + 1
    ids = engine.store.tensor_blob_ids(tied.checkpoint_id)
    assert ids["lm_head.weight"] == ids["model.embed_tokens.weight"] == blob_id(tied_file["model.embed_tokens.weight"])
    assert engine.store.blob_reference_counts()[ids["lm_head.weight"]] == 2
    # One blob per distinct tensor content (freshly initialized norms are all ones and share one).
    assert _blob_files(workspace) == sorted(set(ids.values())) == sorted({blob_id(t) for t in tied_file.values()})
    assert {t.name: t.blob_id for t in tied.tensors} == ids

    untied = _import(engine, snapshots["llama_untied"])
    assert untied.alias_groups == []
    untied_ids = engine.store.tensor_blob_ids(untied.checkpoint_id)
    assert untied_ids["lm_head.weight"] != untied_ids["model.embed_tokens.weight"]

    # The identity's weights hash is over blob IDs in canonical name order, and the stored tensors are the file's.
    loaded = engine.store.load_tensors(tied.checkpoint_id)
    assert tied.weights_hash == blob_weights_hash(loaded) != weights_hash(loaded)
    for name, tensor in tied_file.items():
        assert loaded[name].view(torch.int16).numpy().tobytes() == tensor.view(torch.int16).numpy().tobytes()
    assert torch.equal(loaded["lm_head.weight"], loaded["model.embed_tokens.weight"])
    # The adapter's describe/validate contract agrees with the import, and handles hash by blob IDs.
    handle = engine.open_checkpoint(tied.checkpoint_id)
    assert handle.weights_hash == tied.weights_hash
    assert handle.architecture.alias_groups == tied.alias_groups
    assert [(t.name, t.shape, t.dtype, t.axes, t.module_name) for t in handle.architecture.tensors] == \
        [(t.name, t.shape, t.dtype, t.axes, t.module_name) for t in tied.tensors]


def test_tied_members_with_different_bytes_are_rejected(copy_snapshot, engine):
    snapshot = copy_snapshot("llama_tied", "both_tied_members")
    tensors = load_file(str(snapshot / "model.safetensors"))
    tensors["lm_head.weight"] = -tensors["model.embed_tokens.weight"]
    save_file(tensors, str(snapshot / "model.safetensors"), metadata={"format": "pt"})
    error = _expect_error(ErrorCode.INVALID_MODEL, _import, engine, snapshot)
    assert "lm_head.weight" in error.message and "alias" in error.message


# -- identity --------------------------------------------------------------------------------------


def test_identity_changes_with_tokenizer_chat_template_and_generation_config(copy_snapshot, engine):
    base = _import(engine, copy_snapshot("llama_untied", "base"))
    identity = base.identity
    for field in ("tokenizer_hash", "chat_template_hash", "generation_config_hash"):
        assert getattr(identity, field).startswith("sha256:")
    assert (identity.source_repository, identity.source_revision) == (REPO, REVISION)

    def variant(label: str, edit) -> object:
        snapshot = copy_snapshot("llama_untied", label)
        edit(snapshot)
        return _import(engine, snapshot).identity

    def edit_json(name: str, update: dict):
        def edit(snapshot: Path) -> None:
            path = snapshot / name
            path.write_text(json.dumps({**json.loads(path.read_text()), **update}))
        return edit

    changed = {
        "tokenizer_hash": variant("tokenizer", edit_json("tokenizer_config.json", {"model_max_length": 7})),
        "chat_template_hash": variant("template", lambda s: (s / "chat_template.jinja").write_text("{{ messages }}")),
        "generation_config_hash": variant("generation", edit_json("generation_config.json", {"temperature": 0.1})),
    }
    for field, other in changed.items():
        differing = {name for name in type(identity).model_fields
                     if getattr(identity, name) != getattr(other, name)}
        assert differing == {field}, (field, differing)
    revision = variant("revision", lambda s: None)
    assert revision == identity  # identical files and labels: the same identity
    other_revision = _import(engine, copy_snapshot("llama_untied", "revision2"), revision="f" * 40).identity
    assert {name for name in type(identity).model_fields
            if getattr(identity, name) != getattr(other_revision, name)} == {"source_revision"}
    config = variant("config", edit_json("config.json", {"rms_norm_eps": 1e-3}))
    assert config.config_hash != identity.config_hash and config.weights_hash == identity.weights_hash
    # A missing chat template has no hash.
    snapshot = copy_snapshot("llama_untied", "no_template")
    (snapshot / "chat_template.jinja").unlink()
    assert _import(engine, snapshot).identity.chat_template_hash is None


def test_m1_manifests_and_identities_serialize_unchanged(engine, bundles):
    manifest = engine.import_bundle(bundles["task"])
    dumped = manifest.model_dump(mode="json")
    assert "source" not in dumped
    assert all(set(t) == {"name", "shape", "dtype", "axes", "element_count"} for t in dumped["tensors"])
    assert set(manifest.identity.model_dump(mode="json")) == {"adapter", "config_hash", "weights_hash"}


# -- family descriptors and name maps --------------------------------------------------------------


def test_qwen3_5_descriptor_maps_the_language_model_and_drops_vision_and_mtp(snapshots, engine, workspace):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    snapshot = snapshots["qwen3_5"]
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    language = sorted(name for name in weight_map if name.startswith("model.language_model."))
    assert any(name.startswith("model.visual.") for name in weight_map)
    assert any(name.startswith("mtp.") for name in weight_map)

    manifest = _import(engine, snapshot)
    assert manifest.source.family == ExtensionRef.parse("qwen3_5@1")
    assert manifest.config["family"] == "qwen3_5@1"
    names = [t.name for t in manifest.tensors]
    # Canonical names stay the checkpoint names; the tied output head is added under its module name.
    assert names == sorted([*language, "lm_head.weight"])
    assert not any(name.startswith(("model.visual.", "mtp.")) for name in names)
    modules = {t.name: t.module_name for t in manifest.tensors}
    for name in language:
        assert modules[name] == "model." + name.removeprefix("model.language_model.")
    assert modules["lm_head.weight"] == "lm_head.weight"
    assert manifest.alias_groups == [["lm_head.weight", "model.language_model.embed_tokens.weight"]]
    assert manifest.source.dropped_tensor_count == len(weight_map) - len(language)
    handle = engine.open_checkpoint(manifest.checkpoint_id)
    assert handle.weights_hash == manifest.weights_hash
    assert {t.name: t.module_name for t in handle.architecture.tensors} == modules
    # Pin the prefix that Qwen3_5ForCausalLM actually uses: its parameters are exactly the mapped module names.
    text_config = Qwen3_5TextConfig.from_dict(json.loads((snapshot / "config.json").read_text())["text_config"])
    parameters = _module_parameters(Qwen3_5ForCausalLM.__name__, text_config)
    assert sorted(parameters) == sorted(modules.values())
    assert all(name.startswith("model.layers.") or name in ("model.embed_tokens.weight", "model.norm.weight",
                                                            "lm_head.weight") for name in parameters)
    # Only language-model tensors are stored: the blobs are exactly the manifest's.
    from safetensors import safe_open

    language_blobs = {t.blob_id for t in manifest.tensors}
    assert set(_blob_files(workspace)) == language_blobs
    dropped_only = set()
    for name, file in weight_map.items():
        if name.startswith(("model.visual.", "mtp.")):
            with safe_open(str(snapshot / file), "pt") as handle:
                dropped_only.add(blob_id(handle.get_tensor(name)))
    assert dropped_only - language_blobs  # dropped tensors with their own content exist and are not stored


def test_unmapped_names_fail_with_the_list_of_names(copy_snapshot, engine):
    snapshot = copy_snapshot("llama_untied", "unmapped")
    tensors = load_file(str(snapshot / "model.safetensors"))
    extra = {"model.extra_head.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
             "vision_tower.proj.weight": torch.zeros(3, dtype=torch.bfloat16)}
    del tensors["model.layers.1.mlp.up_proj.weight"]
    save_file({**tensors, **extra}, str(snapshot / "model.safetensors"), metadata={"format": "pt"})
    error = _expect_error(ErrorCode.INVALID_MODEL, _import, engine, snapshot)
    assert error.details["unmapped"] == sorted(extra)
    assert error.details["missing"] == ["model.layers.1.mlp.up_proj.weight"]
    for name in [*extra, "model.layers.1.mlp.up_proj.weight"]:
        assert name in error.message

    snapshot = copy_snapshot("llama_untied", "wrong_shape")
    tensors = load_file(str(snapshot / "model.safetensors"))
    tensors["model.norm.weight"] = torch.zeros(15, dtype=torch.bfloat16)
    save_file(tensors, str(snapshot / "model.safetensors"), metadata={"format": "pt"})
    error = _expect_error(ErrorCode.INVALID_MODEL, _import, engine, snapshot)
    assert "model.norm.weight" in error.message and "[16]" in error.message
    assert engine.store.list_ready() == []


def test_reimporting_a_snapshot_creates_no_new_checkpoint_or_blob(snapshots, workspace, capsys):
    from wse.cli import main

    arguments = ["--workspace", str(workspace), "import-hf", str(snapshots["qwen3_5"]), "--repo", REPO,
                 "--revision", REVISION]
    assert main(arguments) == 0
    first = json.loads(capsys.readouterr().out)
    blobs = _blob_files(workspace)
    weights = sorted((workspace / "artifacts" / "weights").iterdir())
    assert main(arguments) == 0
    second = json.loads(capsys.readouterr().out)
    assert second == first
    assert _blob_files(workspace) == blobs and sorted((workspace / "artifacts" / "weights").iterdir()) == weights
    with Engine(workspace) as engine:
        assert engine.store.list_ready() == [first["checkpoint_id"]]
        manifest = engine.load_manifest(first["checkpoint_id"])
        assert manifest.adapter == ExtensionRef.parse("hf_causal_lm@1")
        assert manifest.provenance.source == f"{REPO}@{REVISION}"
    # CLI errors are JSON on stderr with exit code 2.
    assert main(["--workspace", str(workspace), "import-hf", str(snapshots["pickled_only"]), "--repo", REPO,
                 "--revision", REVISION]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "INVALID_MODEL"


@pytest.mark.parametrize("name, model_class", [("llama_tied", "LlamaForCausalLM"),
                                               ("llama_untied", "LlamaForCausalLM"),
                                               ("qwen3", "Qwen3ForCausalLM"),
                                               ("gemma3", "Gemma3ForCausalLM")])
def test_llama_style_checkpoints_import_without_a_descriptor(snapshots, engine, name, model_class):
    import transformers

    snapshot = snapshots[name]
    manifest = _import(engine, snapshot)
    assert manifest.source.family is None and manifest.config["family"] is None
    assert manifest.source.dropped_tensor_count == 0
    config_json = json.loads((snapshot / "config.json").read_text())
    config = transformers.AutoConfig.for_model(**config_json)
    parameters = _module_parameters(model_class, config)
    # Level 0: canonical names equal the module's parameter names.
    assert sorted(t.name for t in manifest.tensors) == sorted(parameters)
    assert all(t.module_name == t.name for t in manifest.tensors)
    # The parameter count counts an alias group once, as transformers does.
    with torch.device("meta"):
        module = getattr(transformers, model_class)(config)
    assert manifest.parameter_count == sum(p.numel() for p in module.parameters())
    tied = name in ("llama_tied", "gemma3")
    assert manifest.parameter_count == sum(t.element_count for t in manifest.tensors) - (64 * 16 if tied else 0)
    assert manifest.config["model_config"] == config_json
    stored = engine.store.load_tensors(manifest.checkpoint_id)
    on_disk = load_file(str(snapshot / "model.safetensors"))
    for tensor_name, tensor in on_disk.items():
        assert stored[tensor_name].dtype == tensor.dtype
        assert torch.equal(stored[tensor_name].reshape(-1).view(torch.uint8), tensor.reshape(-1).view(torch.uint8))


def test_family_descriptors_register_by_exact_version():
    from wse.adapters.families import QWEN3_5, FamilyDescriptor, PrefixRule, select_family
    from wse.registry import Registry, builtin_registry

    registry = builtin_registry()
    assert registry.list("family") == [ExtensionRef.parse("qwen3_5@1")]
    assert registry.resolve("family", "qwen3_5", "1") is QWEN3_5
    assert select_family(registry, "qwen3_5") == (ExtensionRef.parse("qwen3_5@1"), QWEN3_5)
    assert select_family(registry, "llama") is None
    _expect_error(ErrorCode.INVALID_ARGUMENT, Registry().register, "family", "qwen3_5", "1", object())
    _expect_error(ErrorCode.INVALID_ARGUMENT, Registry().register, "family", "other", "1", QWEN3_5)
    second = FamilyDescriptor(family_id="qwen3_5", version="2", model_types=["qwen3_5"], text_config_key="text_config",
                              causal_lm_class="Qwen3_5ForCausalLM",
                              name_map=[PrefixRule(checkpoint_prefix="model.language_model.", module_prefix="model.")])
    registry.register("family", "qwen3_5", "2", second)
    error = _expect_error(ErrorCode.INVALID_ARGUMENT, select_family, registry, "qwen3_5")
    assert "qwen3_5@1" in error.message and "qwen3_5@2" in error.message
    assert select_family(registry, "qwen3_5", "qwen3_5@2") == (ExtensionRef.parse("qwen3_5@2"), second)
    _expect_error(ErrorCode.INVALID_ARGUMENT, select_family, registry, "llama", "qwen3_5@2")


def test_hf_modules_import_without_transformers():
    code = textwrap.dedent(
        """
        import sys
        sys.modules["transformers"] = None
        import wse, wse.api, wse.cli, wse.hf_snapshot, wse.adapters.families, wse.adapters.hf_causal_lm
        from wse.registry import builtin_registry
        registry = builtin_registry()
        assert str(registry.list("adapter")[0]) == "hf_causal_lm@1"
        loaded = sorted(m for m, module in sys.modules.items()
                        if module is not None and m.split(".")[0] in ("transformers", "mcp", "agents"))
        print(loaded)
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
