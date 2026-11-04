"""The registered FP32 multilayer-perceptron adapter (``mlp@1``).

Layer ``i`` is a PyTorch-style linear layer: ``layers.{i}.weight`` has shape
``[out_features, in_features]`` and, when ``bias`` is true, ``layers.{i}.bias``
has shape ``[out_features]``. The activation is applied between layers, not
after the last one, whose outputs are class logits. Axis semantics come from
this adapter, never from tensor names in a file. Parameter sharing is not
supported.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, ValidationError

from wse.adapters.base import (
    CAP_FORWARD_LOGITS,
    CAP_WEIGHTS_READ,
    CAP_WEIGHTS_SLICE_EDIT,
    AdapterCapabilities,
    Architecture,
    check_tensors_match,
)
from wse.schemas import ErrorCode, ImportManifest, IOSpec, ModelManifest, TensorInfo, WSEError
from wse.tensors import require_finite

MAX_LAYER_WIDTH = 4096
MAX_LAYERS = 16
MAX_PARAMETERS = 4_000_000
MAX_FORWARD_BATCH = 65_536


class MLPConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    layer_sizes: list[StrictInt] = Field(min_length=2, max_length=MAX_LAYERS + 1)
    activation: Literal["relu"] = "relu"
    bias: StrictBool = True


class MLPAdapter:
    """Small dense FP32 MLP classifier."""

    adapter_id = "mlp"
    version = "1"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            capabilities=frozenset({CAP_WEIGHTS_READ, CAP_WEIGHTS_SLICE_EDIT, CAP_FORWARD_LOGITS}),
            shared_parameters=False,
        )

    def parse_config(self, config: Mapping[str, Any]) -> MLPConfig:
        try:
            parsed = MLPConfig.model_validate(dict(config))
        except (ValidationError, TypeError) as exc:
            raise WSEError(
                ErrorCode.INVALID_MODEL,
                f"invalid mlp@1 configuration: {exc}",
                details={"expected": {"layer_sizes": "[int, ...]", "activation": "relu", "bias": "bool"}},
            ) from exc
        for size in parsed.layer_sizes:
            if not 1 <= size <= MAX_LAYER_WIDTH:
                raise WSEError(
                    ErrorCode.INVALID_MODEL,
                    f"mlp@1 layer sizes must be between 1 and {MAX_LAYER_WIDTH}, got {size}",
                )
        return parsed

    def canonical_config(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """The configuration with defaults filled in, used for model identity."""
        return self.parse_config(config).model_dump(mode="json")

    def describe(self, config: Mapping[str, Any]) -> Architecture:
        parsed = self.parse_config(config)
        tensors: list[TensorInfo] = []
        sizes = parsed.layer_sizes
        for index, (fan_in, fan_out) in enumerate(zip(sizes, sizes[1:])):
            tensors.append(
                TensorInfo(
                    name=f"layers.{index}.weight",
                    shape=[fan_out, fan_in],
                    dtype="float32",
                    axes=["out_features", "in_features"],
                    element_count=fan_out * fan_in,
                )
            )
            if parsed.bias:
                tensors.append(
                    TensorInfo(
                        name=f"layers.{index}.bias",
                        shape=[fan_out],
                        dtype="float32",
                        axes=["out_features"],
                        element_count=fan_out,
                    )
                )
        architecture = Architecture(
            tensors=tensors,
            input_semantics=IOSpec(shape=["batch", sizes[0]], dtype="float32"),
            output_semantics=IOSpec(
                shape=["batch", sizes[-1]],
                dtype="float32",
                description="class logits; the predicted class is the argmax",
            ),
        )
        if architecture.parameter_count > MAX_PARAMETERS:
            raise WSEError(
                ErrorCode.INVALID_MODEL,
                f"mlp@1 models are limited to {MAX_PARAMETERS} parameters "
                f"(configuration has {architecture.parameter_count})",
            )
        return architecture

    def validate(
        self, manifest: ImportManifest | ModelManifest, tensors: Mapping[str, torch.Tensor]
    ) -> Architecture:
        if str(manifest.adapter) != f"{self.adapter_id}@{self.version}":
            raise WSEError(
                ErrorCode.INVALID_MODEL,
                f"manifest names adapter {manifest.adapter}, not {self.adapter_id}@{self.version}",
            )
        architecture = self.describe(manifest.config)
        check_tensors_match(architecture, tensors)
        require_finite(tensors)
        return architecture

    def forward(self, tensors: Mapping[str, torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
        weights = [key for key in tensors if key.endswith(".weight")]
        layer_count = len(weights)
        if layer_count == 0:
            raise WSEError(ErrorCode.INVALID_MODEL, "mlp@1 forward requires layer weights")
        has_bias = "layers.0.bias" in tensors
        in_features = tensors["layers.0.weight"].shape[1]
        if not isinstance(inputs, torch.Tensor) or inputs.dtype != torch.float32:
            raise WSEError(ErrorCode.INVALID_ARGUMENT, "mlp@1 inputs must be a float32 tensor")
        if inputs.dim() != 2 or inputs.shape[1] != in_features or inputs.shape[0] > MAX_FORWARD_BATCH:
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"mlp@1 inputs must have shape [batch <= {MAX_FORWARD_BATCH}, {in_features}], "
                f"got {list(inputs.shape)}",
            )
        with torch.inference_mode():
            hidden = inputs
            for index in range(layer_count):
                hidden = hidden @ tensors[f"layers.{index}.weight"].T
                if has_bias:
                    hidden = hidden + tensors[f"layers.{index}.bias"]
                if index < layer_count - 1:
                    hidden = torch.relu(hidden)
            return hidden.clone()
