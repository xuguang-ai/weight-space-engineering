"""An agent edits Qwen3-0.6B's weights so that it finishes more math answers; WSE selects, freezes, and then tests.

This is the README's first example, line for line. Run it in a directory that holds the problems, math_dev.jsonl
and math_test.jsonl (examples/data/ in this repository), with your model provider's key and endpoint in
OPENAI_API_KEY and OPENAI_BASE_URL, and with your model's name, or your Azure deployment's, in place of
YOUR_AGENT_MODEL. The agent is the Codex CLI, 0.167 or later.

1. Define the experiment: the model, the goal (the agent's research brief), the development problems, the
   evaluator, the metric to minimize, and the metric to preserve. The development problems, prompts, sampling
   settings, seeds, and token limit are fixed for the whole experiment, and the agent cannot change how answers
   are graded.
2. Run the agent: Codex, connected to the run's WSE tools, writing Python under Codex's sandbox. WSE evaluates
   the original on the development problems first; max_evaluations counts that evaluation and every evaluation
   of a candidate. When the agent stops, WSE keeps the candidate with the lowest development limit-hit rate
   among those that lower it without lowering development accuracy, and freezes the choice; when no candidate
   qualifies, result.model is the original.
3. Test: only the frozen choice and the original, on test problems that are registered only now.
4. Export the experiment's record, the selected model, and the summary.

The runs recorded in docs/results.md, Section 6, used this configuration with codex:gpt-5.6-luna.
"""

from wse import Experiment

experiment = Experiment(
    model="Qwen/Qwen3-0.6B",
    goal="Reduce repetitive reasoning so the model finishes more math answers.",
    dev="math_dev.jsonl",
    evaluator="math",
    minimize="limit_hit_rate",
    preserve=["accuracy"],
    max_new_tokens=1024,
    workspace="runs/fix-loops",
)
result = experiment.run(
    agent="codex:YOUR_AGENT_MODEL",
    sandbox="python",
    max_evaluations=10,
)
print(result.summary())

report = result.test("math_test.jsonl")  # only the frozen choice and the original, on new problems
print(report.compare())
result.export("artifacts/fix-loops")
