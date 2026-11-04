"""Weight-space engineering: inspect, edit, and evaluate neural network weights.

The core works in memory through ``load_model``; ``Engine`` adds managed,
persistent storage. Neither imports MCP or any agent SDK.
"""

from wse.analysis import analyze_tensor, read_tensor_slice
from wse.api import (
    EditResult,
    Engine,
    EvaluationResult,
    ModelHandle,
    apply_operator,
    load_model,
    read_bundle,
)
from wse.patches import materialize_patch, replay_patch
from wse.policy import (
    BudgetLimits,
    ExperimentPolicy,
    OperationPolicy,
    builtin_experiment_policies,
    default_operation_policy,
)
from wse.registry import Registry, builtin_registry
from wse.schemas import (
    AnalysisResult,
    ErrorCode,
    ExtensionRef,
    ImportManifest,
    MetricGate,
    ModelManifest,
    MaterializedPatch,
    ObjectiveSpec,
    OperatorSpec,
    PatchPreview,
    PatchSpec,
    RunGrant,
    TensorSlice,
    WSEError,
)

__version__ = "0.1.0"

__all__ = [
    "AnalysisResult",
    "BudgetLimits",
    "EditResult",
    "Engine",
    "ErrorCode",
    "EvaluationResult",
    "ExperimentPolicy",
    "ExtensionRef",
    "ImportManifest",
    "MaterializedPatch",
    "MetricGate",
    "ModelHandle",
    "ModelManifest",
    "ObjectiveSpec",
    "OperationPolicy",
    "OperatorSpec",
    "PatchPreview",
    "PatchSpec",
    "Registry",
    "RunGrant",
    "TensorSlice",
    "WSEError",
    "analyze_tensor",
    "apply_operator",
    "builtin_experiment_policies",
    "builtin_registry",
    "default_operation_policy",
    "load_model",
    "materialize_patch",
    "read_bundle",
    "read_tensor_slice",
    "replay_patch",
]
