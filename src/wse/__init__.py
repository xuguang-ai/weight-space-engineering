"""Weight-space engineering: inspect, edit, and evaluate neural network weights.

The core works in memory through ``load_model``; ``Engine`` adds managed,
persistent storage. Neither imports MCP or any agent SDK.
"""

from wse.api import Engine, EditResult, EvaluationResult, ModelHandle, load_model, read_bundle
from wse.registry import Registry, builtin_registry
from wse.schemas import (
    ErrorCode,
    ExtensionRef,
    ImportManifest,
    MetricGate,
    ModelManifest,
    ObjectiveSpec,
    WSEError,
)

__version__ = "0.1.0"

__all__ = [
    "EditResult",
    "Engine",
    "ErrorCode",
    "EvaluationResult",
    "ExtensionRef",
    "ImportManifest",
    "MetricGate",
    "ModelHandle",
    "ModelManifest",
    "ObjectiveSpec",
    "Registry",
    "WSEError",
    "builtin_registry",
    "load_model",
    "read_bundle",
]
