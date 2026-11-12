"""Weight-space engineering: inspect, edit, and evaluate neural network weights.

``Model`` is one version of an open-weight language model to inspect,
analyze, edit, and evaluate (``wse.highlevel``; ``compare_models`` pairs two),
and ``Experiment`` lets an agent investigate and edit one under a budget,
then freezes its choice for an independent test (``wse.experiment``).
Beneath them, the core works in memory through ``load_model``, and
``Engine`` adds managed, persistent storage. None of them imports MCP or any
agent SDK until an agent runs.
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
from wse.highlevel import (
    Comparison,
    Edit,
    Evaluation,
    Inspection,
    Model,
    Probe,
    ProbeSample,
    TensorEntry,
    compare_models,
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

__version__ = "0.1.2"

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
    "compare_models",
    "TensorEntry",
    "ProbeSample",
    "Model",
    "Inspection",
    "Registry",
    "RunGrant",
    "SuiteData",
    "TaskDefinition",
    "TensorSlice",
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
