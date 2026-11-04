# Weight-Space Engineering — Vision and Product Direction

**A proposed foundation for inspecting, transforming, and validating neural network weights—with humans, algorithms, and agents using the same engineering primitives.**

Status: design stage. The repository currently contains documentation; the library, runtime, and MCP integration have not been implemented. Direction updated on 2026-11-04.

**Purpose:** This document defines what we are building, who it serves, and why it should exist. The [implementation plan](./agentic-weight-space-engineering-plan.md) is the authoritative source for architecture, SDK choices, interfaces, tasks, and acceptance criteria.

## Vision

Make model weights a programmable engineering object: something developers can inspect, transform, test, version, and share through stable interfaces. A behavioral objective should be able to drive a sequence of observations, interventions, and evaluations whose result is a verifiable model change.

The long-term ambition is to become a common foundation for weight-space engineering, analogous to the role that shared model interfaces and training APIs play in the Transformers ecosystem. Transformers provides reusable model/configuration conventions and the Trainer API; we aim to establish similarly reusable contracts for model inspection, interventions, editing strategies, and evidence about their effects. This is an architectural ambition, not a claim of comparable coverage or adoption. [Transformers model customization](https://huggingface.co/docs/transformers/en/custom_models), [Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer).

The original Weight-Space Programming article is an inspiration and a source of possible benchmarks. Its experiments do not define the framework's scope. See the [article summary](./weight-space-programming-summary.md) for the research findings and limitations.

## What the Framework Should Make Possible

Researchers should be able to:

- Bring a supported model and describe a behavioral objective with explicit evaluation criteria.
- Inspect parameters and, when permitted, activations or other execution observations.
- Apply an existing editing method or implement a new one without rebuilding experiment infrastructure.
- Let a human, a deterministic algorithm, or an agent choose the next intervention.
- Compare candidates under recorded budgets and evaluation protocols.
- Export an exact patch, a reusable editing recipe, and evidence that explains what changed.

Potential applications include diagnosis, behavioral repair, controlled specialization, model construction, compression, and compatible model composition. Each requires its own algorithms, adapters, and evaluation protocols; naming an application does not imply that the framework already implements it.

## Design Principles

- **A usable library first:** Researchers can inspect, edit, and evaluate models directly in Python. Managed experiments add persistence, budgets, and recovery when needed.
- **Independent of any agent vendor:** Humans, algorithms, and agents share the same weight-engineering capabilities. MCP is an integration interface, and agent SDKs are optional adapters.
- **Extensible methods and models:** New adapters, operators, strategies, and evaluators should integrate without changing the engine. Unsupported capabilities must be explicit.
- **Build on existing execution libraries:** Start with PyTorch and reuse supported Transformers facilities. Claim additional backend support only after implementing and testing it.
- **Evidence defines success:** Behavioral objectives need measurable acceptance criteria. A completed edit is not evidence of improvement, and unrestricted library use does not establish managed budget enforcement.

Detailed contracts, architecture diagrams, and the accepted OpenAI Agents SDK → Claude Agent SDK integration sequence are maintained in the [implementation plan](./agentic-weight-space-engineering-plan.md).

## Extensibility and Reuse

Built-in operations are conveniences, not the permanent vocabulary of weight engineering. Operator definitions and strategy implementations should be independently registered and versioned. Installation is an operator action, separate from an agent's permission to invoke a registered capability.

The initial benchmark policies permit only a small declarative operation set. Later research modes can permit custom Python, gradient-based methods, or hybrid workflows when explicitly configured and measured. Gradient-free execution is one experimental condition, not a universal requirement of the framework. These broader modes are future capabilities, not part of the initial implementation.

Reusable outputs have different meanings:

- **Exact patch:** Rebuilds a candidate from its specified base revision and verifies the resulting tensor hashes.
- **Editing recipe:** Re-runs a method on models with compatible capabilities; new behavioral evaluation is required for each target.
- **Experiment bundle:** Preserves the model, recipe/operator versions, policy, evaluator configuration, permitted data references, and results needed for independent verification.

A patch for one checkpoint does not automatically transfer to another architecture. Recipe portability and exact patch replay must be tested and reported separately. Materialized exact patches should be replayable with the standard patch format without executing the original editing plugin; re-running a recipe requires compatible plugin versions and its declared environment.

## How We Will Establish Value

The initial small-model repair is an engineering acceptance test. A foundational framework also needs evidence of useful reuse:

1. Add a second model family, a second task, and a second agent through adapters and configuration without changing the engine.
2. Add a substantive editing method through the operator/strategy interfaces without changing MCP handlers or artifact storage.
3. Reproduce an exported experiment in a clean environment and evaluate its candidate with a separately supplied evaluator.
4. Measure integration effort, task success, computational cost, and failure rates against a direct Python workflow and a thin MCP wrapper. Comparisons must preserve editing permissions and information access.

The earliest adoption target is researchers who repeatedly change models, methods, and agents. A successful framework makes these combinations easier to build and compare while preserving their scientific meaning.

## From Vision to Implementation

Begin with a small-model workflow to validate the engineering foundation, then demonstrate extensibility across models, methods, and agents before expanding to useful pretrained-model workflows. The [implementation plan](./agentic-weight-space-engineering-plan.md) contains the milestone definitions, concrete tasks, and verification requirements.

For an illustrative developer workflow, see the [SDK-usage repair API sketch](./examples/repair_sdk_usage_api_sketch.py). It proposes a convenience API for a later code-model experiment; it is not executable and does not represent existing functionality.
