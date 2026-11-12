"""An agent edits a small model's weights toward a goal; WSE evaluates, selects, freezes, and then tests the choice.

The experiment on Qwen3-0.6B by default, at a short token limit so that it runs on a laptop:

1. Load the original model at a resolved commit into the experiment's workspace.
2. Define the experiment: the goal (the agent's research brief), the development problems, the evaluator, the
   metric to minimize, and the metric to preserve. The development problems, prompts, sampling settings, seeds,
   and token limit are fixed for the whole experiment, and the agent cannot change how answers are graded.
3. Run the agent (the Codex CLI, connected to the run's WSE tools, writing Python under Codex's sandbox). WSE
   evaluates the original on the development problems first; ``--max-evaluations`` counts that evaluation and
   every evaluation of a candidate. When the agent stops, WSE keeps the candidate with the lowest development
   limit-hit rate among those that lower it without lowering development accuracy, and freezes the choice; when
   no candidate qualifies, the original is kept.
4. Test: only the frozen choice and the original, on test problems that are registered only now.
5. Export the experiment's record, the selected model, and the summary.

Status: run once with a live agent (codex:gpt-5.6-luna) on Qwen3-0.6B on 2026-11-12; docs/results.md, Section 6,
gives the result.

Needs the Codex CLI (its path in ``WSE_CODEX_PATH``, else ``codex`` on PATH, else OpenAI's VS Code extension's)
and your model provider's key and endpoint in ``.env``, copied from ``.env.example`` (git ignores ``.env``). With
your model's name, or your Azure deployment's name, in place of ``YOUR_AGENT_MODEL``:

    cp .env.example .env
    uv run --extra models-hf --env-file .env \\
      python examples/experiment_quickstart.py \\
      --agent codex:YOUR_AGENT_MODEL

A local snapshot names the repository and the commit it was downloaded from (``--model ~/models/Qwen3-0.6B
--repo Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca``).
"""

import argparse
from pathlib import Path

from wse import Experiment, Model

parser = argparse.ArgumentParser(description="An agent experiment on a small model's weights, tested after a freeze.")
parser.add_argument("--agent", required=True, help="codex:YOUR_AGENT_MODEL, your model or deployment name after codex:")
parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="a Hugging Face repository ID or a local snapshot")
parser.add_argument("--repo", help="a local snapshot's source repository")
parser.add_argument("--revision", help="the revision to download (default main), or a local snapshot's commit")
parser.add_argument("--dev", type=Path, default=Path("examples/data/math_dev.jsonl"), help="development problems")
parser.add_argument("--test", type=Path, default=Path("examples/data/math_test.jsonl"), help="test problems")
parser.add_argument("--max-new-tokens", type=int, default=1024, help="the token limit of each answer")
parser.add_argument("--max-evaluations", type=int, default=10, help="development evaluations, the original's included")
parser.add_argument("--workspace", type=Path, default=Path("runs/experiment-quickstart"), help="the workspace")
parser.add_argument("--export", type=Path, default=Path("artifacts/experiment-quickstart"), help="a new directory")
args = parser.parse_args()

# 1. Load the original model; the experiment uses its workspace.
original = Model.from_pretrained(
    args.model,
    repo=args.repo,
    revision=args.revision,
    workspace=args.workspace,
)

# 2. Define the experiment: the goal, the development problems, and the selection rule.
experiment = Experiment(
    model=original,
    goal="Reduce repetitive reasoning so the model finishes more math answers.",
    dev=args.dev,
    evaluator="math",
    minimize="limit_hit_rate",
    preserve=["accuracy"],
    max_new_tokens=args.max_new_tokens,
)

# 3. Run the agent; WSE selects by the rule and freezes the choice.
result = experiment.run(
    agent=args.agent,
    sandbox="python",
    max_evaluations=args.max_evaluations,
)
print(result.summary())
selected = result.model
if selected.checkpoint_id == original.checkpoint_id:
    print("No candidate met the selection rule; the original model is kept.")

# 4. Test the frozen choice and the original on test problems the agent never saw.
report = result.test(args.test)
print(report.compare())

# 5. Export the experiment's record, the selected model, and the summary.
print(f"export: {result.export(args.export)}")
