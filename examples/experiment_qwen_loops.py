"""An agent tries to stop Qwen3.5-0.8B's reasoning loops by editing its weights, under WSE's fixed evaluation.

Asked GSM8K problems with room for 8,192 new tokens, Qwen3.5-0.8B often keeps reasoning until the limit: in the
recorded diagnosis on the 32 development problems used here (``examples/data/qwen_loops_dev.jsonl``), 30 of its 32
answers reached the limit and 2 were correct (docs/results.md, Section 3). This script gives an agent that goal:

1. Load Qwen3.5-0.8B at the commit of every recorded run into the experiment's workspace.
2. Define the experiment: the goal, the 32 development problems, the article's sampling profile
   (``qwen35think@1``) at 8,192 new tokens, the limit-hit rate to minimize, and accuracy to preserve.
3. Run the agent. WSE evaluates the original on the 32 problems first; ``--max-evaluations`` counts that
   evaluation and every evaluation of a candidate. When the agent stops, WSE keeps the candidate with the lowest
   development limit-hit rate among those that lower it without lowering development accuracy, and freezes the
   choice; when no candidate qualifies, the original is kept.
4. Test only the frozen choice and the original, on test problems registered only now (16 GSM8K test problems
   by default).
5. Export the experiment's record, the selected model, and the summary.

The limit-hit rate is the share of answers cut off at the token limit. It is not a repetition rate: an answer
can repeat itself and still finish, or reach the limit without repeating.

Status: not run yet. The same API has run once with a live agent, on Qwen3-0.6B (docs/results.md, Section 6).

Time: on an M4 Pro (MPS), one evaluation of the 32 problems took 37 to 40 minutes in the recorded runs, so ten
evaluations take 6.2 to 6.6 hours, and the test (two models on 16 problems, as many answers as one evaluation)
about as long as one more. The agent's time limit includes its waits for evaluations; ``run``'s default, 12 hours,
leaves room for ten. Keep the Mac awake (``caffeinate -is``). Disk: the download is 1.7 GB and the
workspace's copy of the model 1.5 GB; the export adds the selected model and the experiment's record, which holds
the original model again.

Needs the Codex CLI (its path in ``WSE_CODEX_PATH``, else ``codex`` on PATH, else OpenAI's VS Code extension's)
and your model provider's key and endpoint in ``.env``, copied from ``.env.example``. With your model's name, or
your Azure deployment's name, in place of ``YOUR_AGENT_MODEL``:

    caffeinate -is uv run --extra models-hf --env-file .env \\
      python examples/experiment_qwen_loops.py \\
      --agent codex:YOUR_AGENT_MODEL

A local snapshot names the repository it was downloaded from (``--model ~/models/Qwen3.5-0.8B
--repo Qwen/Qwen3.5-0.8B``); ``--revision`` defaults to the recorded commit.
"""

import argparse
from pathlib import Path

from wse import Experiment, Model

parser = argparse.ArgumentParser(description="An agent experiment on Qwen3.5-0.8B's reasoning loops.")
parser.add_argument("--agent", required=True, help="codex:YOUR_AGENT_MODEL, your model or deployment name after codex:")
parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="a Hugging Face repository ID or a local snapshot")
parser.add_argument("--repo", help="a local snapshot's source repository")
parser.add_argument("--revision", default="2fc06364715b967f1860aea9cf38778875588b17", help="the commit")
parser.add_argument("--dev", type=Path, default=Path("examples/data/qwen_loops_dev.jsonl"), help="development problems")
parser.add_argument("--test", type=Path, default=Path("examples/data/math_test.jsonl"), help="test problems")
parser.add_argument("--max-evaluations", type=int, default=10, help="development evaluations, the original's included")
parser.add_argument("--wall-clock-hours", type=float, help="the agent's time limit, its waits included (default 12)")
parser.add_argument("--workspace", type=Path, default=Path("runs/qwen-loops"), help="the workspace")
parser.add_argument("--export", type=Path, default=Path("artifacts/qwen-loops"), help="a new directory")
args = parser.parse_args()

# 1. Load the original model at a fixed commit; the experiment uses its workspace.
original = Model.from_pretrained(
    args.model,
    repo=args.repo,
    revision=args.revision,
    workspace=args.workspace,
)

# 2. Define the experiment: the goal, the development problems, the sampling, and the selection rule.
experiment = Experiment(
    model=original,
    goal="Reduce repetitive reasoning so the model finishes more math answers.",
    dev=args.dev,
    evaluator="math",
    minimize="limit_hit_rate",
    preserve=["accuracy"],
    max_new_tokens=8192,
    profile="qwen35think@1",
)

# 3. Run the agent; WSE selects by the rule and freezes the choice.
result = experiment.run(
    agent=args.agent,
    sandbox="python",
    max_evaluations=args.max_evaluations,
    wall_clock_hours=args.wall_clock_hours,
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
