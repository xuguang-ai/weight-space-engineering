# WSE extension contracts

The framework's reusable parts are registered extensions: model adapters,
operators, evaluators, and strategies, plus agent adapters for complete SDK
tool loops. This document describes each contract, how an operator registers
a separately supplied implementation without changing core code, and what
"exact patch replay" and "recipe re-execution" mean. Examples use a
hypothetical 8-input MLP, never a model or task used in an agent trial.

## Registration

`wse.registry.Registry` is a small in-process registry with explicit,
exact-version entries:

```python
registry.register(kind, extension_id, version, implementation)  # kind: adapter, operator, evaluator, strategy
registry.resolve(kind, extension_id, version)                    # exact; never substitutes the latest version
registry.register_alias(kind, "mlp", "mlp@1")                    # a version-less name only through a declared alias
```

Every kind has a registration validator that checks the declared identity
and the required methods before anything is stored; a duplicate
registration fails with `VERSION_CONFLICT`. `builtin_registry()` contains the
M1 extensions: adapter `mlp@1` (alias `mlp`), analyses `tensor_stats@1`,
`tensor_norm@1`, `tensor_top_k@1`, edits `set_slice@1` and `scale_slice@1`,
evaluators `classification@1` and `forward_batch@1`, and strategy
`scripted_search@1`.

Operators register extensions; agents only invoke what a policy permits.
Processes pick up operator registrations through setup hooks, functions
`module:function` that receive the registry:

```sh
wse --workspace WORKSPACE serve --run-id RUN --setup my_extensions:setup   # server and its worker
wse --workspace WORKSPACE worker --setup my_extensions:setup
wse --workspace WORKSPACE run --run-id RUN --setup my_extensions:setup     # runner, server, and worker
```

## Model adapters (`wse.adapters.base.ModelAdapter`)

An adapter declares `adapter_id` and `version` and implements
`capabilities() -> AdapterCapabilities`, `validate(manifest, tensors) ->
Architecture` (tensor names, shapes, dtypes, axis semantics, input/output
semantics, alias groups), and `forward(tensors, inputs) -> logits`. In a
managed experiment only the evaluation worker calls `forward`; a run-bound
Engine cannot open a checkpoint handle.

Capabilities are explicit. `weights.read`, `weights.slice_edit`, and
`forward.logits` are the M1 capabilities; tensor access does not imply
support for probes, layer surgery, quantization, or other architectures. An
operation whose `required_capabilities` the adapter lacks fails with
`UNSUPPORTED_CAPABILITY` before any candidate is created.

## Operators (`wse.operators.base`)

An operator carries an `OperatorSpec`: ID and version, input and output JSON
schemas (generated from its Pydantic models), required capabilities, effect
(`analysis`, `probe`, or `edit`), and a usage contract naming the budget
counter it charges. The engine dispatches by spec; MCP handlers contain no
operator-specific code. An analysis implements `analyze(region,
parameters)`; an edit implements `apply(region, parameters) -> values` and is
materialized by the engine into the standard patch representation (exact
after-values per step, with before/after tensor hashes), so storage,
transport, and replay need no per-operator code.

A separately supplied analysis:

```python
from wse.adapters.base import CAP_WEIGHTS_READ
from wse.operators.base import BaseAnalysisOperator, OperatorParameters, OperatorResult, make_spec
from wse.schemas import UsageContract


class AbsMeanResult(OperatorResult):
    value: float


class AbsMean(BaseAnalysisOperator):
    result_model = AbsMeanResult
    spec = make_spec(operator_id="abs_mean", version="1", effect="analysis",
                     description="Mean absolute value of a selection.",
                     required_capabilities=[CAP_WEIGHTS_READ], parameters_model=OperatorParameters,
                     result_model=AbsMeanResult, usage=UsageContract(counter="analysis_calls"))

    def analyze(self, region, parameters):
        return AbsMeanResult(value=float(abs(region.values).mean()))


def setup(registry):
    registry.register("operator", "abs_mean", "1", AbsMean())
```

Registering an operator does not permit it. A policy maps public names to
exact versions (`OperationPolicy.with_analysis("abs_mean", "abs_mean@1")` or
`with_edit(...)`), and an experiment policy is immutable once registered:
an operation that the experiment's policy does not list is rejected with
`POLICY_DENIED`, and an unknown version with `NOT_FOUND`.

Unsupported operations in M1: low-rank and directional edits (`add_low_rank`,
`project_out`) are M2 operators; models with shared parameters (alias
groups) are rejected; no operator may run the target network, compute
gradients, or execute agent-supplied code under the M1 benchmark policies.

## Evaluators (`wse.evaluators.base`)

An evaluator declares `evaluator_id` and `version` and implements
`evaluate(forward, suite, config, *, batch_size) -> {"metrics",
"sample_count", "errors"}`. `forward` is a metered callable: the worker
counts batches, examples, and forward calls against the reservation and
stops the evaluator at the policy's limits. Suites bind an exact evaluator,
its parameters, the data, and the preregistered sample counts and seeds.

## Strategies (`wse.strategies.base.Strategy`)

A strategy is a stepwise controller:

```python
next_action(objective, observations, remaining_budget) -> Action
```

`Action.call(tool, rationale, **arguments)` names one of the WSE tools
(`wse_get_experiment`, `wse_inspect_model`, `wse_read_tensor`,
`wse_analyze`, `wse_preview_patch`, `wse_apply_patch`,
`wse_submit_evaluation`, `wse_get_job`, `wse_cancel_job`, `wse_compare`,
`wse_select_checkpoint`, `wse_list_models`) with the MCP tool's arguments;
`Action.stop(reason)` ends the run. Each `Observation` carries the same
`result` object the MCP tool returns, or the domain error body. The runner
passes the experiment description as the first observation and executes each
action through an Engine bound to the run grant, so a strategy uses exactly
the agent-facing, metered operations and cannot change its evaluator, grant,
or budget. Rationales are brief and observable; strategies must not record
private reasoning.

The registered implementation is a factory, usually the class, that takes
the configured parameters as keyword arguments, declares `strategy_id` and
`version`, and whose instances implement `next_action` and `config()` (the
configuration recorded with the experiment):

```python
from wse.strategies.base import Action


class InspectOnce:
    strategy_id, version = "inspect_once", "1"

    def config(self):
        return {}

    def next_action(self, objective, observations, remaining_budget):
        experiment = observations[0].result["experiment"]
        if len(observations) == 1:
            return Action.call("wse_inspect_model", "Read the architecture.",
                               experiment_id=experiment["experiment_id"],
                               checkpoint_id=experiment["base_checkpoint_id"])
        return Action.stop("inspection done")


def setup(registry):
    registry.register("strategy", "inspect_once", "1", InspectOnce)
```

```sh
wse --workspace WORKSPACE run --run-id RUN --controller scripted --strategy inspect_once@1 \
    --setup my_extensions:setup
```

`scripted_search@1` is the deterministic reference strategy: a task-agnostic
search over single-unit `scale_slice` edits (parameters `factors`,
`patience`, `max_candidates`, `tensors`).

## Agent adapters (`wse.agent_adapters.base.AgentAdapter`)

Controllers that run a complete SDK tool loop implement the vendor-neutral
adapter contract instead of `Strategy`:

```python
await adapter.start(connection, agent_config)          # start `wse serve` for the run grant and connect
result = await adapter.run(experiment_id, agent_config, event_sink)   # -> AgentRunResult
adapter.cancel(experiment_id, reason)                   # thread-safe; the SDK loop stops
await adapter.close(experiment_id)                      # after the runner has settled and frozen
```

`AgentRunResult` holds the termination status and reason, the final
message, an optional conversation reference, turn and tool-call counts,
elapsed time, normalized usage (`AgentUsage`: token counts, provider, model;
fields the host cannot report are `None` with a reason), and provider
metadata. An adapter never selects, freezes, or closes. Events reach the
runner as `AgentEvent` records (model responses with usage, tool calls, tool
results with error codes and bounded excerpts, messages, cancellations).
Agent SDKs are optional extras imported lazily inside the adapter module.
`wse.agent_adapters.openai_agents.OpenAIAgentsAdapter` is the M1 adapter; see
[experiments.md](experiments.md) for its configuration.

## Direct Python use

The core works in memory, without a database, service, or MCP. Results are
labeled `unmanaged` and support no managed-enforcement claim:

```python
import torch

from wse import apply_operator, evaluate_candidate, load_model, replay_patch

tensors = {"layers.0.weight": torch.randn(16, 8), "layers.0.bias": torch.zeros(16),
           "layers.1.weight": torch.randn(2, 16), "layers.1.bias": torch.zeros(2)}
model = load_model(tensors, adapter="mlp", config={"layer_sizes": [8, 16, 2], "activation": "relu", "bias": True})

edit = apply_operator(model, "scale_slice", {
    "tensor": "layers.0.weight",
    "selection": [{"start": 3, "stop": 4, "step": 1}, {"start": 0, "stop": 8, "step": 1}],
    "factor": 0.5,
})  # edit.candidate (a new handle), edit.patch (exact), edit.diff (per tensor)

inputs = torch.randn(32, 8)
suite = {"suite_id": "example_dev_v1", "task_id": "example_task_v1", "split": "development",
         "evaluator": "classification@1", "sample_counts": [32], "seeds": [0],
         "data": {"inputs": inputs.tolist(), "labels": (inputs.sum(dim=1) > 0).long().tolist()}}
result = evaluate_candidate(edit.candidate, suite, {"sample_count": 32, "seed": 0})
print(result.metrics["accuracy"], result.management)   # ... unmanaged

assert replay_patch(model, edit.patch).weights_hash == edit.candidate.weights_hash
```

## Exact patch replay and recipe re-execution

An **exact patch** (`MaterializedPatch`) records, for every step, the
operator version, the selection, the after-values written back to the
stored dtype, and the tensor hashes before and after. Replaying it
(`replay_patch`, `Engine.replay_export`, `wse replay`) writes those stored
values without running the editing operator and verifies every hash, so it
reproduces byte-identical weights or fails. It applies only to its own base
revision: a different base identity (adapter, configuration, or weights)
fails with `VERSION_CONFLICT`.

A **recipe** is the operation list of a patch spec (operator references and
parameters). Re-executing it runs the registered operators again, on the
same or another compatible base. That requires the exact operator versions
to be registered (missing or incompatible versions are rejected, never
replaced by the latest), and the result is a new candidate that needs fresh
evaluation: a recipe that helped one base proves nothing about another.
Exports carry both, the exact patch for replay and the frozen spec as the
recipe, and their replay instructions say which one was used.
