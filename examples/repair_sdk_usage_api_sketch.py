"""Proposed WSE user API: repair outdated SDK usage in a code model.

This is an API design sketch, not executable software. Every import from `wse`
and every WSE method below is proposed. No WSE package is implemented here.
OpenAIAgentsAdapter is our planned wrapper around the real OpenAI Agents SDK
for Python; it would connect to the managed WSE runtime through local MCP.

This illustrates a later pretrained-model workflow, beyond the small-MLP MVP.
It requires the following operator-prepared components, none provided here:

* A local model bundle and registered `hf_causal_lm` adapter.
* Registered probe/edit operators compatible with that model's architecture.
* An `sdk_migration_v1` task with development suites `sdk_v2` and `general_code`,
  and a sealed final suite containing independent examples for both metrics.
* A code evaluator with the exact target SDK version installed in an isolated
  execution environment. It checks API behavior and output assertions, not
  merely syntax, successful imports, or the presence of a new API name.

The Experiment convenience facade would register the operator's objective and
budget, create the run grant, and manage the worker/MCP lifecycle. The agent
cannot change these settings or access the sealed final evaluator. Core
operations must remain available without installing an agent SDK.

Freeze waits for or cancels outstanding work, settles usage, and fixes the
selected checkpoint. Final evaluation compares that checkpoint and the original
under the same protocol. A failed final result does not reopen the experiment.

Thresholds and budgets below are illustrative choices, not measured results.
Rates are fractions: 0.20 is 20 percentage points, and 0.01 is 1 percentage point.
WSE_AGENT_MODEL selects the editing agent; it is separate from the code model
whose weights are edited. Credentials are supplied through the adapter's normal
environment configuration and are never written into an experiment bundle.
"""

import asyncio
import os

from wse import Budget, Experiment, MetricGate, ObjectiveSpec, load_model
from wse.agent_adapters.openai_agents import OpenAIAgentsAdapter


async def main():
    # 1. Load the model to edit, using a compatible registered adapter.
    model = load_model(
        source="./models/internal-code-model/model.safetensors",
        adapter="hf_causal_lm",
        config="./models/internal-code-model/config.json",
    )

    # 2. Set acceptance criteria before the agent sees evaluation feedback.
    objective = ObjectiveSpec(
        task_id="sdk_migration_v1",
        description=(
            "Correct outdated SDK calls in generated code while preserving "
            "performance on general coding tasks."
        ),
        gates=[
            MetricGate("sdk_v2.pass_rate", min_delta=0.20),
            MetricGate("general_code.pass_rate", max_drop=0.01),
        ],
    )

    # 3. Select the editing agent, not the target model.
    agent = OpenAIAgentsAdapter(model=os.environ["WSE_AGENT_MODEL"])

    # 4. Give it only registered capabilities and a fixed development budget.
    async with Experiment(
        workspace="./runs/sdk-migration",
        model=model,
        objective=objective,
        policy_id="code_repair_probe_v1",
        operators=["tensor_stats@1", "activation_probe@1", "add_low_rank@1"],
        budget=Budget(
            max_candidates=8,
            max_evaluation_batches=20,
            max_generated_tokens=500_000,
        ),
    ) as experiment:
        # The task resolves development data; final data is evaluator-only.
        baseline = await experiment.evaluate_development(
            checkpoint_id=experiment.base_checkpoint_id,
        )
        print("Development baseline:", baseline.metrics)

        # 5. The SDK runs the tool loop; WSE executes and records the operations.
        # A valid outcome is that no useful edit is discovered.
        await experiment.run(agent=agent)

        # 6. Freeze first, then evaluate both models on sealed final examples.
        frozen = await experiment.freeze_selected()
        comparison = await experiment.evaluate_final(
            checkpoint_id=frozen.checkpoint_id,
            include_baseline=True,
        )

        # 7. Export a candidate for review only if it meets the declared gates.
        # Nothing in this example deploys a model or publishes a real PR.
        if comparison.accepted:
            bundle = experiment.export_bundle(
                output_dir="./artifacts/sdk-migration-candidate",
                include_patch=True,
                include_report=True,
                include_final_data=False,
            )
            print("Candidate available for review:", bundle.path)
        else:
            report = experiment.export_report(
                output_dir="./artifacts/sdk-migration-failed",
            )
            print("Acceptance criteria not met:", report.path)


if __name__ == "__main__":
    asyncio.run(main())
