"""Weight-space engineering: inspect, edit, and evaluate neural network weights.

``WSE`` is the high-level entry point for open-weight language models
(``wse.highlevel``). Beneath it, the core works in memory through
``load_model``, and ``Engine`` adds managed, persistent storage. None of them
imports MCP or any agent SDK.
"""

from wse.analysis import analyze_tensor, read_tensor_slice
from wse.api import (
    EditResult,
    Engine,
    EvaluationResult,
    ModelHandle,
    apply_operator,
    evaluate_candidate,
    load_model,
    read_bundle,
)
from wse.highlevel import WSE, Comparison, Edit, Evaluation, Probe
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
    ComparisonReport,
    DatasetManifest,
    ErrorCode,
    EvalConfig,
    EvaluationReport,
    EvaluationSuite,
    ExtensionRef,
    FinalEvaluationReport,
    ImportManifest,
    MetricGate,
    ModelManifest,
    MaterializedPatch,
    ObjectiveSpec,
    OperatorSpec,
    PatchPreview,
    PatchSpec,
    RunGrant,
    SuiteData,
    TaskDefinition,
    TensorSlice,
    WSEError,
)

__version__ = "0.1.1"

__all__ = [
    "AnalysisResult",
    "BudgetLimits",
    "Comparison",
    "ComparisonReport",
    "DatasetManifest",
    "Edit",
    "EditResult",
    "Engine",
    "Evaluation",
    "ErrorCode",
    "EvalConfig",
    "EvaluationReport",
    "EvaluationResult",
    "EvaluationSuite",
    "ExperimentPolicy",
    "ExtensionRef",
    "FinalEvaluationReport",
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
    "Probe",
    "Registry",
    "RunGrant",
    "SuiteData",
    "TaskDefinition",
    "TensorSlice",
    "WSE",
    "WSEError",
    "analyze_tensor",
    "apply_operator",
    "builtin_experiment_policies",
    "builtin_registry",
    "default_operation_policy",
    "evaluate_candidate",
    "load_model",
    "materialize_patch",
    "read_bundle",
    "read_tensor_slice",
    "replay_patch",
]
