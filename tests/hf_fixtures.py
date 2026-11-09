"""Tiny random Hugging Face checkpoints, built offline from configuration classes (M2 plan Task 3).

Every builder writes a snapshot directory as ``save_pretrained`` writes it
(safetensors weights, ``config.json``, ``generation_config.json``), plus a tiny
character-level tokenizer, a chat template, and decoy files that the importer
must never read: ``training_args.bin``, ``original/consolidated.00.pth``, and
``onnx/model.onnx``. Nothing is downloaded. Builders import ``transformers``
lazily, so test modules can skip cleanly when the ``models-hf`` extra is absent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DECOYS = ("training_args.bin", "original/consolidated.00.pth", "onnx/model.onnx")
VOCAB = ["<pad>", "<s>", "</s>", "<unk>"] + [chr(c) for c in range(97, 123)] + [str(d) for d in range(10)] + [" "]
VOCAB_SIZE = 64  # model vocabulary; the tokenizer uses the first len(VOCAB) entries
CHAT_TEMPLATE = "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}"

LLAMA = dict(vocab_size=VOCAB_SIZE, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
QWEN3 = dict(vocab_size=VOCAB_SIZE, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
             num_attention_heads=4, num_key_value_heads=2, head_dim=4, max_position_embeddings=128)
GEMMA3 = dict(vocab_size=VOCAB_SIZE, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
              num_attention_heads=4, num_key_value_heads=2, head_dim=4, sliding_window=8,
              max_position_embeddings=128)
# GPT-2 has absolute (learned) position embeddings, so a position shifted by padding changes its logits.
GPT2 = dict(vocab_size=VOCAB_SIZE, n_embd=16, n_layer=2, n_head=4, n_positions=128, bos_token_id=1, eos_token_id=2)
QWEN3_5_TEXT = dict(vocab_size=VOCAB_SIZE, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=4, max_position_embeddings=128,
                    layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2,
                    linear_num_value_heads=4, linear_key_head_dim=4, linear_value_head_dim=4,
                    linear_conv_kernel_dim=4, tie_word_embeddings=True)
QWEN3_5_VISION = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=16,
                      patch_size=2, temporal_patch_size=2, spatial_merge_size=2, num_position_embeddings=16)
# Multi-token-prediction tensors that Qwen3.5 checkpoints carry and transformers ignores.
QWEN3_5_MTP = {"mtp.fc.weight": (16, 32), "mtp.layers.0.mlp.down_proj.weight": (16, 32), "mtp.norm.weight": (16,)}


def _seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)


def write_tokenizer(snapshot: Path) -> None:
    """A character-level WordLevel tokenizer with a chat template and a special-tokens map."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.WordLevel(vocab={t: i for i, t in enumerate(VOCAB)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<s>", eos_token="</s>",
                                   unk_token="<unk>", pad_token="<pad>")
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(str(snapshot))
    (snapshot / "special_tokens_map.json").write_text(json.dumps(
        {"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<pad>"}, indent=2))


def add_decoys(snapshot: Path) -> None:
    for relative in DECOYS:
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x80\x04decoy: the importer must never read this file\x2e")


def _finish(model: Any, snapshot: Path, *, dtype: Any, generation: dict[str, Any] | None = None,
            max_shard_size: str | int | None = None, f32_suffixes: tuple[str, ...] = ()) -> Path:
    """Cast to ``dtype`` and save; parameters whose names end with one of ``f32_suffixes`` stay F32."""
    import torch
    from transformers import GenerationConfig

    model = model.to(dtype)
    for name, parameter in model.named_parameters():
        if name.endswith(f32_suffixes):
            parameter.data = parameter.data.to(torch.float32)
    model.generation_config = GenerationConfig(bos_token_id=1, eos_token_id=2, pad_token_id=0,
                                               **(generation or {"do_sample": True, "temperature": 0.7}))
    kwargs = {} if max_shard_size is None else {"max_shard_size": max_shard_size}
    model.save_pretrained(str(snapshot), **kwargs)
    write_tokenizer(snapshot)
    add_decoys(snapshot)
    return snapshot


def llama(snapshot: Path, *, tied: bool, seed: int = 0) -> Path:
    """A Llama-style model: 2 layers, hidden 16, 4 heads, 2 KV heads, BF16."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    _seed(seed)
    model = LlamaForCausalLM(LlamaConfig(**LLAMA, tie_word_embeddings=tied, bos_token_id=1, eos_token_id=2))
    return _finish(model, snapshot, dtype=torch.bfloat16)


def qwen3(snapshot: Path, *, seed: int = 1) -> Path:
    """A Qwen3 model in F16 with untied embeddings."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    _seed(seed)
    model = Qwen3ForCausalLM(Qwen3Config(**QWEN3, tie_word_embeddings=False))
    return _finish(model, snapshot, dtype=torch.float16)


def gemma3(snapshot: Path, *, seed: int = 2) -> Path:
    """A Gemma 3 text model in F32 with tied embeddings."""
    import torch
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    _seed(seed)
    model = Gemma3ForCausalLM(Gemma3TextConfig(**GEMMA3, tie_word_embeddings=True))
    return _finish(model, snapshot, dtype=torch.float32)


def gpt2(snapshot: Path, *, seed: int = 4) -> Path:
    """A GPT-2 model in F32: learned absolute position embeddings and tied input and output embeddings."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    _seed(seed)
    model = GPT2LMHeadModel(GPT2Config(**GPT2, tie_word_embeddings=True))
    return _finish(model, snapshot, dtype=torch.float32)


def qwen3_5(snapshot: Path, *, seed: int = 3) -> Path:
    """A Qwen3.5 multimodal wrapper: a 2-layer text model (linear then full attention) and a 1-block vision model.

    The checkpoint is sharded with an index, as the published snapshot is, and
    an extra shard carries ``mtp.*`` tensors, which the transformers class
    does not define. As in the published checkpoint, the linear-attention
    layer's ``A_log`` and gated norm weight are stored in F32 and the rest in BF16.
    """
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    _seed(seed)
    config = Qwen3_5Config(text_config=QWEN3_5_TEXT, vision_config=QWEN3_5_VISION, tie_word_embeddings=True)
    model = Qwen3_5ForConditionalGeneration(config)
    _finish(model, snapshot, dtype=torch.bfloat16, max_shard_size="20KB",
            f32_suffixes=(".linear_attn.A_log", ".linear_attn.norm.weight"))
    index_path = snapshot / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    mtp_file = "model-mtp.safetensors"
    save_file({name: torch.randn(shape).to(torch.bfloat16) for name, shape in QWEN3_5_MTP.items()},
              str(snapshot / mtp_file), metadata={"format": "pt"})
    index["weight_map"].update({name: mtp_file for name in QWEN3_5_MTP})
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
    return snapshot


def pickled_only(snapshot: Path) -> Path:
    """A snapshot whose only weights are a pickled ``pytorch_model.bin`` (never loaded)."""
    from transformers import LlamaConfig

    snapshot.mkdir(parents=True, exist_ok=True)
    LlamaConfig(**LLAMA).save_pretrained(str(snapshot))
    (snapshot / "pytorch_model.bin").write_bytes(b"\x80\x04pickled weights: never unpickled\x2e")
    write_tokenizer(snapshot)
    add_decoys(snapshot)
    return snapshot


BUILDERS = {
    "llama_tied": lambda path: llama(path, tied=True),
    "llama_untied": lambda path: llama(path, tied=False),
    "qwen3": qwen3,
    "gemma3": gemma3,
    "qwen3_5": qwen3_5,
    "gpt2": gpt2,
    "pickled_only": pickled_only,
}


def build_all(root: Path) -> dict[str, Path]:
    """Build every fixture snapshot under ``root``; return name to snapshot directory."""
    return {name: build(root / name) for name, build in BUILDERS.items()}
