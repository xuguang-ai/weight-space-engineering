# Development Documents

How WSE is being built: plans, milestone reports, and handovers. The user documentation is in [docs/](../docs), and [docs/results.md](../docs/results.md) collects every measurement. These documents record the project as it went: an early document can describe a state that later work changed.

| Document | What it is |
| --- | --- |
| [weight-space-programming-summary.md](weight-space-programming-summary.md) | A summary of *Sparks of Weight-Space Programming*, the article behind WSE's reference case (2026-11-03) |
| [weight-space-engineering-vision.md](weight-space-engineering-vision.md) | The vision and product direction, written before any code (2026-11-04) |
| [agentic-weight-space-engineering-plan.md](agentic-weight-space-engineering-plan.md) | The first implementation plan (milestone M1) and the roadmap |
| [m1-agent-trials.md](m1-agent-trials.md) | M1's live trials of an LLM agent on a small model |
| [agentic-weight-space-engineering-m2-plan.md](agentic-weight-space-engineering-m2-plan.md) | The M2 plan: open-weight language models, Qwen3.5-0.8B first |
| [m2-q1-report.md](m2-q1-report.md) | M2 stage Q1: loading, exact replay of published edits, and generation evaluation |
| [m2-q2-report.md](m2-q2-report.md) | M2 stage Q2: agent-usable tools for language models |
| [handover-cuda.md](handover-cuda.md) | The handover for CUDA support |

## Milestones

| Milestone | Status |
| --- | --- |
| **M1:** contracts, managed experiments, MCP, and an LLM agent repairing a small model | Done |
| **M2:** open-weight language models: import, exact replay of published edits, generation evaluation, and agent-usable tools for probes, edits, and budgets | Built and measured ([M2 Q1](m2-q1-report.md) and [Q2](m2-q2-report.md) reports) |
| **CUDA:** generation and probes on NVIDIA GPUs | Planned; [handover](handover-cuda.md) written |
| **Agent trials on Qwen3.5-0.8B:** preregistered trials of LLM agents | Next |
