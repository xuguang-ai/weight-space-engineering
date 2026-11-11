# Weight-Space Programming: Article Summary and Engineering Implications

- **Original article:** [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
- **Authors:** Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab (MIT CSAIL)
- **Published:** 2026-11-02; summary prepared: 2026-11-03.
- **Sources reviewed:** The article, the authors' experiment READMEs, and their log documentation. Results below are reported by the authors; no experiments were run or independently reproduced for this summary.

## 1. Core Idea

Weight-Space Programming means using a general-purpose LLM to analyze and directly modify neural network parameters according to a natural-language goal, without making those modifications through a gradient optimizer. The article uses GPT-6 Astra to explore three capabilities: **understanding behavior from weights, repairing existing networks, and constructing networks from scratch**. Small networks can fit their parameters into context; larger networks require tools to read, compute over, and save tensors. [Original article](https://rikiyat.github.io/blog/weight-space-programming/)

Several distinctions matter for comparing experiments: using no gradients does not mean using no feedback; a one-shot edit does not mean the entire model was never trained; and writing a target network from scratch does not mean the editing LLM lacks pretrained knowledge.

## 2. What the Eight Experiments Show

| Experiment | Inputs and permitted operations | Authors' reported results and limitations |
| --- | --- | --- |
| **1. Identify game policies** | Four identically structured MLPs with 17,253 parameters; weights and input/output semantics provided, without candidate game names or tools. | All four games were scored as correct. The answer for Freeway was Frogger, accepted under the predefined keyword rule. Identical architectures rule out classification by model size alone, but task interface semantics were still provided. [Experiment 1](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp1_which_game/README.md) |
| **2. Identify diffusion model outputs** | Three DDPMs with 23,824 parameters; no tools or forward computations. | Identified circles, squares, and triangles, but described filled shapes as hollow outlines. Identifying a category does not imply complete understanding of generation behavior. [Experiment 2](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp2_diffusion/README.md) |
| **3. Read intermediate Transformer layers** | Character models with 7.16M parameters; only four intermediate blocks provided. Weight matrix calculations allowed, but running the network or any layer prohibited. | Correctly identified all three training-text categories in two attempts each; identified copying heads matched the authors' threshold labels. The README notes that the article uses induction score >0.2, while some scoring-file fields use 0.3. [Experiment 3](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp3_language/README.md) |
| **4. Repair a car-control policy** | PPO actor with 171,664 parameters; scene, failure result, and one trajectory provided. One reply, without tools or subsequent feedback. | Edited 6,073 parameters (3.54%) to reverse before accelerating. The baseline succeeded in 0/256 episodes; the edited policy succeeded in 1,024/1,024 independent test episodes. This is a repair for a specific scenario. [Experiment 4](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp4_fix_car/README.md) |
| **5. Write MinAtar policies from scratch** | Three games; a 600→32→32→6 network initialized entirely to zero. Rules and source code provided, without execution feedback. | All three exceeded PPO trained for 100M steps. Against PPO trained for 500M steps, Breakout scored 517.4 versus 25.3, while Seaquest and Asterix did not exceed PPO. Each was evaluated over 1,000 episodes; this does not establish general superiority over reinforcement learning. [Experiment 5](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp5_write_from_scratch/README.md) |
| **6. Write a Go network from scratch** | 9×9 Go, approximately 1.48M parameters; Python editing with 0/10/25/50 feedback batches. Training and tree search prohibited. | With no feedback, win rates against two small PPO policies were approximately 54%. After 50 batches, win rates against three seeds of an equally sized PPO policy were approximately 70%–79%. This demonstrates iterative editing assisted by tools. [Experiment 6](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp6_go_from_scratch/README.md) |
| **7. Identify sycophancy** | Pairs of Gemma 3 270M models; weights-only analysis compared with access to attention observations. | Four weights-only attempts failed to distinguish the models. With attention observations on self-selected texts, all three model pairs were identified correctly. Success depended on dynamic observations rather than purely static auditing. [Experiment 7](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp7_sycophancy/README.md) |
| **8. Repair language-model loops** | Qwen3.5-0.8B; CPU weight editing, practice data, and a restricted inference tool. Training prohibited. | Local edits substantially improved answer completion and several metrics, detailed below. [Experiment 8](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md) |

## 3. Qwen's Improvements and the Gemma Counterexample

The 50-batch edit primarily removes a looping direction from the MLP outputs in layers 15 and 19. It changes 131,072 parameters, approximately 0.016% of the model. The authors published parameter diffs and reconstruction verification information. [Edited model documentation](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming)

| Configuration | MATH-500 | GSM8K | IFEval | MATH-500 answers reaching the token limit |
| --- | ---: | ---: | ---: | ---: |
| Base model | 7.9% | 13.8% | 21.0% | 93.0% |
| 10 batches | 12.3% | 25.7% | 49.0% | 87.2% |
| 25 batches | 14.1% | 25.1% | 39.9% | 86.4% |
| 50 batches | 48.1% | 67.1% | 62.3% | 33.1% |

The table reports the authors' results under the same evaluation protocol. MATH-500 uses four samples per problem, with a generation limit of 32,768 tokens. The 10-, 25-, and 50-batch configurations are separate editing runs, not points along a single continuous optimization trajectory. [Experiment 8 run and evaluation documentation](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md)

**Key counterexample:** For Gemma 3 270M, which rarely loops, MATH-500 changed from 6.8% to 7.0%–7.7% across budgets. The authors did not establish an effective improvement from these results. This control supports a narrower interpretation: editing can remove obstacles to existing capabilities, but has not yet demonstrated reliable addition of missing reasoning abilities. [Gemma follow-up experiment](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/gemma_followup/README.md)

## 4. Limits on Interpreting the Results

- **A pass is a tool-budget unit:** One Go pass can include up to 128 games or 256 positions. One Qwen pass can generate answers to up to 512 practice problems or read up to 1,024 texts. These involve multiple underlying network computations; autoregressive generation in particular cannot be accounted for as a single model forward call. [Go protocol](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp6_go_from_scratch/README.md), [Qwen protocol](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md)
- **The released records have a defined scope:** The repository includes runs reported in the article, but excludes trials of earlier prompt versions. Some weights and intermediate artifacts were not released. These results warrant reproduction, but do not establish an overall success rate across all attempts. [Log documentation](https://github.com/RikiyaT/lm_understand-logs)
- **My assessment:** Changing few parameters does not automatically guarantee preservation of other capabilities. Attention differences alone do not establish a complete causal explanation. Avoiding gradients during editing does not by itself establish lower total cost than training. Regressions, intervention effects, and end-to-end costs should be measured separately.

## 5. Implications for Our Harness (Engineering Inferences)

Translate natural-language goals into an inspectable loop: **inspect the model → propose a falsifiable hypothesis → create a weight patch → evaluate within a fixed budget → accept or roll back**.

MCP interfaces should return tensor summaries, bounded slices, and artifact references rather than placing an entire large model in context. Each patch should record the baseline hash, modified regions, numerical changes, a short rationale, and evidence. Static analysis, dynamic probes, development evaluation, and final blind evaluation should have separate permissions and accounting. Validate this loop on controlled small models before extending it to real LLMs. Harness engineering correctness and the Agent's ability to discover effective edits should be separate acceptance criteria.

See the [Weight-Space Engineering Framework — Initial Implementation Plan](./agentic-weight-space-engineering-plan.md) for the construction roadmap.

The project direction was broadened on 2026-11-04 to a reusable weight-engineering framework with a standalone Python core, managed experiments, and CLI/MCP interfaces. The article remains a research reference rather than the limit of the framework's scope. See the [framework vision](./weight-space-engineering-vision.md).

## Sources and Reuse

This document is an English summary and engineering analysis. Experimental details were cross-checked against the [experiment log repository](https://github.com/RikiyaT/lm_understand-logs) published by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab. The repository provides logs, data, and networks under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and the authors' scripts under MIT. Third-party materials retain their own licenses; see [LICENSE](https://github.com/RikiyaT/lm_understand-logs/blob/main/LICENSE). This document does not include models, datasets, or verbatim logs.
