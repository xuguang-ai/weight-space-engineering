"""The ``ModelAdapter`` contract.

An adapter owns one architecture: its configuration schema, exact tensor
names, shapes, dtypes, and axis semantics, and controlled inference. It
advertises capabilities explicitly; tensor access does not imply support for
any other operation. Adapters never execute code shipped with a model.

In managed experiments only the worker may call ``forward``.

Adapters may also implement ``canonical_config(config) -> dict``, returning the
configuration with defaults filled in; model identity hashes that normalized
configuration so equivalent configurations share one identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import torch
from pydantic import BaseModel, ConfigDict, Field

from wse.schemas import ErrorCode, ImportManifest, IOSpec, ModelManifest, TensorInfo, WSEError
from wse.tensors import dtype_name

# Capability names advertised by adapters and required by operators.
CAP_WEIGHTS_READ = "weights.read"
CAP_WEIGHTS_SLICE_EDIT = "weights.slice_edit"
CAP_FORWARD_LOGITS = "forward.logits"
CAP_GENERATE_TEXT = "generate.text"


class AdapterCapabilities(BaseModel):
    """Capabilities an adapter supports, plus its parameter-sharing semantics.

    ``shared_parameters`` declares that models of the adapter may share
    parameters. Such an adapter lists every alias group in the architecture it
    describes; edits refuse members of an alias group and allow every other
    tensor.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    capabilities: frozenset[str]
    shared_parameters: bool = False

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities


class Architecture(BaseModel):
    """The adapter-derived layout of a model with a given configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tensors: list[TensorInfo]
    input_semantics: IOSpec
    output_semantics: IOSpec
    alias_groups: list[list[str]] = Field(default_factory=list)

    @property
    def parameter_count(self) -> int:
        """Distinct parameters: an alias group (one shared object, such as tied embeddings) counts once."""
        aliases = {name for group in self.alias_groups for name in sorted(group)[1:]}
        return sum(info.element_count for info in self.tensors if info.name not in aliases)


@runtime_checkable
class ModelAdapter(Protocol):
    """Contract every registered architecture adapter implements."""

    adapter_id: str
    version: str

    def capabilities(self) -> AdapterCapabilities:
        """Return the explicitly supported capabilities."""
        ...

    def describe(self, config: Mapping[str, Any]) -> Architecture:
        """Validate ``config`` and return the exact expected tensor layout."""
        ...

    def validate(
        self, manifest: ImportManifest | ModelManifest, tensors: Mapping[str, torch.Tensor]
    ) -> Architecture:
        """Check ``tensors`` against the architecture; raise ``INVALID_MODEL``."""
        ...

    def forward(self, tensors: Mapping[str, torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
        """Run controlled inference and return logits."""
        ...


REQUIRED_ADAPTER_METHODS = ("capabilities", "describe", "validate", "forward")


def check_tensors_match(architecture: Architecture, tensors: Mapping[str, torch.Tensor]) -> None:
    """Require exactly the expected tensor names, shapes, and dtypes."""
    expected = {info.name: info for info in architecture.tensors}
    missing = sorted(set(expected) - set(tensors))
    unexpected = sorted(set(tensors) - set(expected))
    if missing or unexpected:
        raise WSEError(
            ErrorCode.INVALID_MODEL,
            "tensor names do not match the architecture"
            + (f"; missing: {missing}" if missing else "")
            + (f"; unexpected: {unexpected}" if unexpected else ""),
            details={"missing": missing, "unexpected": unexpected},
        )
    for name in sorted(expected):
        info, tensor = expected[name], tensors[name]
        if list(tensor.shape) != info.shape:
            raise WSEError(
                ErrorCode.INVALID_MODEL,
                f"tensor {name!r} has shape {list(tensor.shape)}; the architecture requires {info.shape}",
                details={"tensor": name, "shape": list(tensor.shape), "expected_shape": info.shape},
            )
        if dtype_name(tensor) != info.dtype:
            raise WSEError(
                ErrorCode.INVALID_MODEL,
                f"tensor {name!r} has dtype {dtype_name(tensor)}; the architecture requires {info.dtype}",
                details={"tensor": name, "dtype": dtype_name(tensor), "expected_dtype": info.dtype},
            )
