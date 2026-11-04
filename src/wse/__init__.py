"""Weight-space engineering: inspect, edit, and evaluate neural network weights.

The core works in memory through ``load_model``; ``Engine`` adds managed,
persistent storage. Neither imports MCP or any agent SDK.
"""

from wse.analysis import analyze_tensor, read_tensor_slice
from wse.api import Engine, EditResult, EvaluationResult, ModelHandle, load_model, read_bundle
from wse.policy import OperationPolicy, default_operation_policy
from wse.registry import Registry, builtin_registry
from wse.schemas import (
    AnalysisResult,
    ErrorCode,
    ExtensionRef,
    ImportManifest,
    MetricGate,
    ModelManifest,
    ObjectiveSpec,
    OperatorSpec,
    TensorSlice,
    WSEError,
)

__version__ = "0.1.0"

__all__ = [
    "AnalysisResult",
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
    "OperationPolicy",
    "OperatorSpec",
    "Registry",
    "TensorSlice",
    "WSEError",
    "analyze_tensor",
    "builtin_registry",
    "default_operation_policy",
    "load_model",
    "read_bundle",
    "read_tensor_slice",
]
