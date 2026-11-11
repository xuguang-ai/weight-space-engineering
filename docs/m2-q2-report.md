# M2 Q2 Report: Agent-Usable Tools for Open-Weight Language Models

Q2 of the [M2 plan](../agentic-weight-space-engineering-m2-plan.md) gives agents the tools to work on open-weight language models:

- family roles and protections;
- direction, low-rank, and fill edits;
- analyses of large tensors;
- activation probes and generation passes;
- token budgets and the policy `qwen_loop_v1`;
- the MCP tools.

This report covers Tasks 8–12 and Q2's exit conditions. Everything ran on the operator's Mac described in the [Q1 report](m2-q1-report.md): an Apple M4 Pro with 48 GB, PyTorch 2.15.0 on MPS, and `transformers` 5.22.2. The runs took place from 2026-11-10 to 2026-11-11.

## 1. Q2 Exit

| Condition (plan Section 1) | Result |
| --- | --- |
| The known repair, expressed through the tools with a direction recovered from the published diff, lowers the development limit-hit rate | **Holds.** The check ran through MCP on 32 GSM8K practice problems. Answers cut off at the limit fell from 30/32 to 19/32, a paired difference of −0.34 [−0.50, −0.19] (Section 3). |
| A scripted discovery attempt that uses only the tools is reported | **Holds; the attempt found no repair.** Through MCP, `direction_search@1` chose layer 9. Its better candidate moved cut-off answers from 30/32 to 27/32, a paired difference of −0.09 [−0.22, +0.03]; the interval includes zero (Section 4). |
| Qwen3-0.6B and Gemma-3-1b-it pass the generic smoke checks with no new code | **Holds** (Section 5). |

All three conditions hold as written. The second asks for a report, not for a repair.

Section 8 lists the open items, including those that need the author's decision before Q3. Q3 is paused until after WSE's open-source release (user decision, 2026-11-10).

## 2. What Was Built

| Task | What | Commits |
| --- | --- | --- |
| 8 | Roles, head groupings, and protected roles in the family descriptors `qwen3_5@1` and `llama@1`. Every edit operator refuses the protected roles `embed`, `lm_head`, and `final_norm`. | be203d3, 37c35d7, 5eac44b |
| 9 | Edit operators `project_out@1`, `fill_slice@1`, and `add_low_rank@1`, with platform-independent arithmetic (exact products, correctly rounded sums); vector references for directions and factors | 07ccd3a, 9b77a4d |
| 10 | Analyses `tensor_norms@1`, `svd_top_k@1`, and `direction_projection@1`. Their vector results are run-scoped, and other operations accept pointers to them. | 98b564d |
| 11 | Activation probes, in memory and as managed jobs. Token budgets: `generation_passes`, `generated_tokens`, `probe_calls`, `teacher_forced_tokens` (migrations 10 and 11). The policy `qwen_loop_v1`. Generation passes as development evaluations. The new MCP tools `wse_probe` and `wse_get_report`, and extensions to `wse_analyze`, `wse_compare`, `wse_inspect_model`, and the previews. | ce5affe, 7d0b2c9, 0a7f670, f0c3191 |
| 12 | The strategy `direction_search@1`. The scripts `examples/qwen_loops/check_expressiveness.py` and `run_discovery.py`, which drive the tools through a real `wse serve` MCP session (`--via-mcp`). | 2495c95, fb060b8 |
| Reviews | Fixes from the six reviews (Section 7) | 19 commits from 6c46436 to 73e9b41, and 36fcbfb |

Alongside Q2, the high-level Python entry point `wse.WSE` gives the operator the same four steps (e6f2dd9, 4f9e1d8, 3394514, 51b0558):

- `evaluate`;
- `probe`;
- `edit`, `apply_patch`, and `Edit.save`;
- `compare`.

## 3. Expressiveness Check

**Setup:**

- **Practice suite.**
  - 32 GSM8K training problems, chosen by subset seed 0 (`gsm8k_train_n32_s0_t8192`).
  - Profile `qwen35think@1` at 8,192 new tokens, 1 sample each.
  - MPS at batch size 8.
- **Run.** A run grant under `qwen_loop_v1` on the Qwen3.5-0.8B base: run `run_23e8f36f14379754782d07d4ea19c771`, experiment `exp_47f77d80e210bf67e5cd3381ff5fa7eb`.
- **Controller.** A scripted client drove the tools through a real MCP session: the OpenAI Agents SDK's tool loop ran against `wse serve`, and the script chose each call in place of a model.
- **The edit.**
  - One patch of two `project_out` operations, one in layer 15 and one in layer 19.
  - Each layer's direction is the top left singular vector of p50c's published change to the 64 changed columns of that layer's `mlp.down`.
  - The edit uses those 64 columns and alpha 6, which is p50c's form (Section 6.1).
  - The directions entered the run as inline patch values, an operator-only artifact. Directions recomputed from the stored checkpoints equal the saved ones exactly (cosine 1.0, the same columns).
  - The patch changes 130,134 weights; p50c itself changes 130,188.

| 32 problems | Base | Edited | Paired difference [95% interval] |
| --- | --- | --- | --- |
| Accuracy | 3/32 (9.4%) | 12/32 (37.5%) | +0.28 [+0.13, +0.44] |
| Answers cut off at the limit | 30/32 (93.8%) | 19/32 (59.4%) | −0.34 [−0.50, −0.19] |
| Mean generated tokens | 7,924 | 6,849 | −1,075 [−1,858, −408] |

- **Cost.** Each pass took about 39 minutes. The run made 477 tool calls, most of them job polls, and used 472,727 of its 2,621,440 generated tokens.
- **A second, disjoint sample.**
  - An earlier operator-path run (`WSE.compare`, the same edit) drew 32 different problems. Its problem IDs had another prefix, and the subset is chosen by a hash of the IDs, so the two sets do not overlap.
  - It found the same effect. Accuracy rose from 2/32 to 10/32 (+0.25 [+0.13, +0.41]), and cut-off answers fell from 30/32 to 21/32 (−0.28 [−0.44, −0.13]).

The command, with the default snapshot `~/Desktop/wse-models/Qwen3.5-0.8B` of Q1:

- `--recompute` derives the directions from the stored base and p50c checkpoints, and the run uses them.
- `--directions` records their agreement with Task 9's saved copy.

```bash
caffeinate -is uv run --no-sync python examples/qwen_loops/check_expressiveness.py OUT \
    --workspace WORKSPACE --problems 32 --max-new-tokens 8192 --subset-seed 0 \
    --alpha 6.0 --via-mcp --recompute --directions p50c_directions.npz
```

## 4. Scripted Discovery

### 4.1 What ran

The strategy `direction_search@1` ran once, as planned, on the same 32 practice problems as Section 3, under `qwen_loop_v1`:

- run `run_b5297da814bf1c75b3c0c229d5309d2a`;
- experiment `exp_641407d3a653316b629ecddfa4d6b81e`;
- from 23:52 EST on 2026-11-10 to 01:44 EST on 2026-11-11.

It drove a real `wse serve` session through the OpenAI Agents SDK, with the strategy choosing every call in place of a model. It took the four steps of WSE's workflow:

1. **Behavioral diagnosis.**
   - It read the architecture: `wse_inspect_model`, 9 pages.
   - It ran one generation pass of the base: `wse_submit_evaluation`, then `wse_get_report`. The base got 3/32 correct, and 30/32 answers were cut off at 8,192 tokens.
   - The outputs formed two groups: the 30 cut-off outputs (a) and the 2 finished ones (b).
2. **Mechanism investigation.**
   - **Probe.** One probe job (`wse_probe`) read the last 1,024 tokens of each of the 32 outputs at the input and output of `mlp.down` in all 24 layers. That is 48 sites and 32,768 teacher-forced tokens.
   - **Layer score.** For each output site, the strategy scored the separation: the norm of the contrast (mean of a minus mean of b), divided by the mean of the two group means' norms.
   - **Layer.** Layer 9 scored highest, at 1.18. Next came layers 12, 11, 7, and 8, at 1.13, 1.10, 1.09, and 1.08.
   - **Columns.** One analysis (`wse_analyze`, `direction_projection`) picked the 64 columns of layer 9's `mlp.down` whose cosine with the contrast is largest in magnitude.
3. **Weight intervention.**
   - It built two candidates. Each is one `project_out` of the contrast from those 64 columns: one with alpha 1, one with alpha 6.
   - Each was previewed and applied (`wse_preview_patch`, `wse_apply_patch`).
   - The direction travelled as a vector reference, `wse://analyses/34ca4033…#/sites/19/contrast/direction`, so the strategy never handled its 1,024 numbers.
4. **Evaluation.**
   - It ran one generation pass per candidate.
   - It compared each pass with the base pass, problem by problem (`wse_compare`).
   - It selected a candidate by the point estimates (`wse_select_checkpoint`).

**No oracle values, but a p50c-shaped form.** The strategy reads no published weights, directions, columns, or evaluation items. Its edit form, the 64 columns, and alpha ∈ {1, 6} are taken from p50c's published form. The run therefore tests whether the tools can localize a p50c-shaped edit from behavior alone.

### 4.2 Results

| 32 problems | Base | Alpha 1 | Alpha 6 (selected) |
| --- | --- | --- | --- |
| Accuracy | 3/32 | 3/32; +0.00 [−0.09, +0.09] | 5/32; +0.06 [−0.06, +0.19] |
| Answers cut off at the limit | 30/32 | 29/32; −0.03 [−0.16, +0.06] | 27/32; −0.09 [−0.22, +0.03] |
| Mean generated tokens | 7,924 | 7,596; −328 [−1,022, +333] | 7,849; −75 [−511, +351] |
| Changed weights | | 64,588 | 65,308 |

Paired differences with the base, with 95% cluster-bootstrap intervals over problems (10,000 resamples).

**Outcome:**

- The attempt ran end to end through MCP.
- It stopped on its own after its two candidates: "Stopping: selected alpha 6 at layer 9 (limit-hit/accuracy: base 0.938/0.0938, a1 0.906/0.0938, a6 0.844/0.156)".
- It did not find a repair. Every paired interval includes zero.
- The selection used point estimates on the same 32 problems that defined the groups.
- On the same problems, the expressiveness check's edit (Section 3) kept 11 answers from being cut off: −0.34 [−0.50, −0.19].

### 4.3 Why it missed

This analysis ran after the attempt, by the operator, with knowledge of p50c. It is not part of the attempt.

- **Layer choice.** By the strategy's score, p50c's layers rank 10th (layer 15, 0.81) and 17th (layer 19, 0.60) of 24.
- **Direction.** At those layers, the probe's contrast is aligned with p50c's direction, though less strongly than a comparison with random directions suggests.
  - Its cosine with p50c's per-layer direction is +0.35 at layer 15 and −0.44 at layer 19. The sign does not matter to `project_out`.
  - Random directions are the wrong baseline. In 1,024 dimensions their cosine stays below 0.103 in magnitude in 99.9% of draws, but the outputs' activations vary strongly along p50c's direction itself.
  - The right baseline compares the actual split with arbitrary splits of the same outputs. This run's stored probe keeps only the group means, so the operator ran that check on the sample of the four-step example (README, Section 5): 32 other problems, probed with full context, with the per-output means of `Probe.means`.
  - There, p50c's direction carries 6.7% (layer 15) and 14.4% (layer 19) of the outputs' variance, against 0.1% for a random direction. Over all 496 ways to set 2 of the 32 outputs apart, the median cosine magnitude is 0.24 and 0.36. The actual split gives 0.37 and 0.45 and ranks 4th and 7th (p = 0.008 and 0.014). In each layer, all 32 outputs' components along the direction have the same sign, and the 2 finished outputs have the smallest in magnitude at layer 15 and two of the three smallest at layer 19.
  - The actual split also has the largest contrast norm of all 496: the finished outputs differ from the rest in many directions, not only in p50c's (Section 4.4, position confound).
- **Columns.**
  - With the probe's contrast at those layers, the cosine rule picks 27 (layer 15) and 19 (layer 19) of p50c's 64 columns.
  - With p50c's own direction, it picks 33 and 20.
  - Chance would give about 1.
  - Even with the exact direction, the column rule recovers at most about half of p50c's columns.

The miss is first a layer-selection miss: at p50c's layers the contrast was aligned with p50c's direction, and the score did not rank those layers high. The score has two known weaknesses:

- it ignores the spread within each group (review C6);
- group b has only 2 outputs.

### 4.4 Caveats

- **Group size.** Group b has 2 outputs. The contrast is the difference between a 30-output mean and a 2-output mean.
- **Position confound.**
  - Each window is the last 1,024 tokens of an output. Cut-off outputs' windows end at the limit; finished outputs' windows end at their answers.
  - The contrast therefore mixes "will be cut off" with "where in the output".
- **Output site.**
  - The layer and the direction come from the output of `mlp.down`, while the plan says to probe its input.
  - The output is the space in which `project_out` removes a direction. A test now pins this choice (cb533bd).
- **This run's record** was written by the code at fb060b8.
  - Its steps have no rationales; the test double records them since 067ade1.
  - Its steps store each call's outcome text under `error`; they store it as `summary` since 36fcbfb.
  - The strategy's code is unchanged since the run; its documentation changed.

### 4.5 Cost

- **Wall-clock time:** 1 h 52 min.
- **Budget used** (of `qwen_loop_v1`'s limits):
  - 3 of 10 generation passes;
  - 747,821 of 2,621,440 generated tokens;
  - 1 of 25 probes, with 32,768 teacher-forced tokens;
  - 1 of 50 analyses;
  - 2 of 50 previews;
  - 2 of 20 commits;
  - 6,708 of 21,600 compute seconds.
- **Tool calls:** 693, of which 669 were job polls.

**Polling cost for a model agent:**

- `wse_get_job` waits at most 10 seconds, and one pass takes about 37 minutes. The client therefore polled 669 times.
- The test double estimates the conversation a model would carry: characters divided by 4, without the tool definitions. By that estimate, the conversation passes 128,000 tokens at request 93 and 200,000 tokens at request 202, still within the base pass. It reaches 548,000 tokens at the end.
- A model agent that polls the same way would exceed its context window during the first pass.
- This needs a design decision before Q3 (Section 8).

The command:

```bash
caffeinate -is uv run --no-sync python examples/qwen_loops/run_discovery.py OUT \
    --workspace WORKSPACE --problems 32 --max-new-tokens 8192 --subset-seed 0 --via-mcp
```

## 5. Smoke Matrix

Both models went through Level 0 with no new code. Each used its own default generation profile.

| Check | `Qwen/Qwen3-0.6B` | `google/gemma-3-1b-it` |
| --- | --- | --- |
| Revision | `c1899de289a04d12100db370d81485cdf75e47ca` | `dcc83ea841ab6100d6b47a070329e1ba4cf78752` |
| Import (download allow-list of Task 1) | 311 tensors, 596,049,920 parameters, 7.7 s; no family descriptor | 341 tensors, 999,885,952 parameters, 10.6 s; no family descriptor |
| Tied embedding | `lm_head.weight` and `model.embed_tokens.weight` form one alias group | The same |
| Logits against `transformers` | Bitwise equal on CPU and on MPS | Bitwise equal on CPU and on MPS |
| Patch round trip | 8,192 weights in layer 0's `mlp.down_proj`. Verify and saved verify pass, and the reopened checkpoint has the same hash. | 9,216 weights; the same results |
| Analyses | `stats`, `tensor_norms`, `svd_top_k` (0.38 s), and `direction_projection` succeed | The same (`svd_top_k` 0.07 s) |
| Probe | Residual sites equal `hidden_states` exactly, in all 28 layers | The same, in all 26 layers |
| Generation (MPS) | 48 tokens in 1.3 s, cut off at the 48-token smoke limit | 13 tokens in 1.3 s, finished: "Two plus three is five. \boxed{5}" |

The smoke check found one bug. The parameter counts counted the tied embedding twice: 751,632,384 and 1,301,875,840 in the raw output. 737bd27 counts each alias group once.

## 6. Other Measurements

### 6.1 Recipe re-execution versus exact replay (Task 9)

**p50c's form, recovered from the published diff:**

- In each layer, the change to the 64 changed columns of `mlp.down` is rank 1. Its top singular value holds 0.999995 (layer 15) and 0.999991 (layer 19) of the change's energy.
- The least-squares alphas are 6.0003 and 6.0002.
- The two layers' directions have a cosine of −0.85.

**Re-execution.** `project_out@1` re-executed the edit on the base, with each layer's own direction and alpha 6. In the changed columns (65,536 entries per layer), the result differs from p50c's published weights as follows:

| Layer | Differing entries | By at most one BF16 spacing | By one to two spacings | By more | Largest difference |
| --- | --- | --- | --- | --- | --- |
| 15 | 5,924 | 4,961 | 407 | 556 | 0.00049 |
| 19 | 6,162 | 5,123 | 408 | 631 | 0.00049 |

- About 91% of the entries are byte-exact.
- Five rounding paths were tried. The two that compute in float64 or float32 and round to BF16 once come closest: 5,922–5,924 and 6,161–6,162 differing entries. The all-BF16 path differs in 21,995 and 19,369 entries. So p50c's published weights match none of the five exactly.
- The behavioral effect is the same (Section 3).
- Exact replay (Task 5) remains the hash-exact path.

**`fill_slice@1` reproduces p25c hash-exactly.**

- The edit: zero columns 1,792–2,047 of layer 15's `self_attn.o_proj`, the input of attention head 7.
- The edited tensor's hash and the whole checkpoint's weights hash equal those of p25c's exact replay, `sha256:dde0821e…` (Q1 report, Section 3).

### 6.2 Reproducibility of a generation pass

Section 3 and Section 4 each ran a base pass in a separate experiment, about 1.5 hours apart. The two passes produced the same 32 outputs, token for token: 253,567 generated tokens in all. Only the experiment and job IDs in the output records differ.

The four-step example (README, Section 5) ran in full twice, in separate processes, with the library code of b3534a4.

- **The runs.** The two comparisons started 2 h 38 min apart. The second run started from a fresh download of the model and the data into new directories.
- **What changed between them.** Only the example script:
  - a guard that skips step 2 when either group is empty (both groups were non-empty);
  - the sign convention of the recovered direction. It flips the printed cosine at layer 19 from +0.45 to −0.45 and leaves the edit bit-identical, because negating a direction is exact;
  - comments.
- **What matched.**
  - Both records hold the same weights hashes for the base and the edited model, and the same data and profile hashes.
  - The 64 outputs of the comparison are the same, token for token, with the same answers.
  - Only the random checkpoint IDs and the creation times differ.
- **A third process.** The rerun behind the permutation check of Section 4.3 reproduced the 32 base outputs token for token as well.

On this machine, a pass with a fixed suite, profile, seed, and batch size is reproducible to the token.

### 6.3 Throughput and costs

- **Generation pass:** 32 problems at up to 8,192 tokens on MPS at batch size 8.
  - Each of the five passes took 2,211–2,378 seconds, about 37–40 minutes.
  - The discovery's base pass generated 253,567 tokens in 2,243 s, about 113 tokens per second.
- **Probe:** 32 texts of 1,024 tokens at 48 sites took about 15 seconds as a job.
- **Other tool calls on Qwen3.5-0.8B** (from the discovery's step times):
  - a `project_out` preview took 3.6–5.2 s, and applying it took 2.9–3.1 s;
  - a `direction_projection` over a whole `mlp.down` (3,670,016 entries) took 0.14 s;
  - an inspection page took about 0.2 s, and a comparison under 0.1 s.
- **MCP response sizes:** measured on Qwen3.5-0.8B and listed in [protocol.md](protocol.md#language-model-tools).

## 7. Reviews

Six reviews checked the Q2 code, C1 to C6. In each, an agent read the commits, ran scripts against them, and reported findings with evidence.

| Review | Commits reviewed | Findings | Critical and major findings, and their fixes |
| --- | --- | --- | --- |
| C1 | 8fb1eeb, 5987a4a, 07ccd3a, ce5affe | 14: 1 critical, 1 major, 7 minor, 5 nits | **Critical.** A managed preview accepted an exact operator reference (`project_out@1`) that the policy does not list. Fixed in 9b77a4d. **Major.** `run_agent_experiment` final-evaluated every trial by default. Fixed in 37c35d7, where it is off by default. |
| C2 | be203d3, e6f2dd9, 4f9e1d8 | 15: 2 major, 9 minor, 4 nits | **Major.** The stored `model_config` was not the snapshot's `config.json`: `transformers` changed a shallow copy. Fixed in 37c35d7. **Major.** A user's own intervention could not be stored. Fixed in 3394514 (`Engine.store_patch`). |
| C3 | 37c35d7, 3394514, 9b77a4d, 98b564d | 13: 1 major, 5 minor, 7 nits | **Major.** `refuse_whole_tensor_scaling` was checked one operation at a time, so a whole-tensor scaling split across operations passed. Fixed in 7d0b2c9, which checks the patch's net effect against the experiment's base. C4 found the limits of that check (Section 8). |
| C4 | 7d0b2c9 | 7: 1 major, 3 minor, 3 nits | **Major.** The reference runner gated evaluation on `evaluation_batches`, which `qwen_loop_v1` sets to 0. Fixed in 0a7f670. |
| C5 | 0a7f670, f0c3191 | 9: 1 major, 4 minor, 4 nits | **Major.** Bounded views budgeted only the result, not the MCP envelope, so constructed requests reached 65–74 KB on the wire. Fixed in 729be0c, 9560fdf, and 73e9b41. |
| C6 | 2495c95, fb060b8 | 10: 1 major, 6 minor, 3 nits | **Major.** The strategy's "no oracle" claim was overstated, because its defaults copy p50c's form. The docstrings now say what comes from p50c (e8f9d52), and Section 4 uses the same wording. |

In all, the reviews reported 68 findings: 1 critical, 7 major, 34 minor, and 26 nits.

- Every critical and major finding is fixed. C3's fix has limits, which C4 found.
- C4's two minor findings on the scope of the whole-tensor scaling check need the author's decision (Section 8).
- Other findings stay as recorded limits rather than changes: Sections 4.3, 4.4, and 8.

## 8. Open Items

**Needs the author's decision before Q3:**

1. **Waiting on long jobs.**
   - `wse_get_job` waits at most 10 seconds, and one generation pass takes about 37 minutes, so an agent polls hundreds of times per pass. Its conversation outgrows a model's context window during the first pass (Section 4.5).
   - Options:
     - a longer wait, which needs a longer MCP client timeout than the default 30 seconds;
     - progress notifications;
     - a harness that drops old poll results;
     - a tool that blocks until a job ends or a wall-clock limit passes.
2. **The scope of the whole-tensor scaling protection** (review C4).
   - The check refuses only exact constant scalings of one stored tensor.
   - Near-constant scalings, and an up-scaling compensated by a down-scaling in the next tensor, pass.
   - For zero-centered RMSNorm weights (Qwen3.5, Gemma-3), the gain is 1 + w. A stored-tensor scaling is therefore not a gain scaling.
3. **`qwen_loop_v1`'s compute limit.** Ten passes at 37–40 minutes take 6.2–6.6 hours, more than the policy's 6 hours of computation, so an agent cannot use all ten passes. Task 13 adjusts the policy limits from measurements; these are the numbers. Lower the passes to nine, or raise the limit.
4. **The Q3 preregistration text.** The review fixes changed `qwen_loop_v1`'s instructions (neutral probe-site examples). The preregistration must quote the current text.

**Needs the author's decision, not blocking Q3:**

- **The discovery strategy.** Q3's controls do not use it. Keep `direction_search@1` as Q2's record. Any change to its layer score goes into a new version. The current score ignores within-group spread and uses position-mismatched windows (Sections 4.3 and 4.4).

**Recorded limits:**

- Managed experiments on open-weight models cannot yet be exported or replayed as M1 experiments can.
- Generation usage records generated tokens, but not prompt tokens or prefill and decode counts, which M1 plan Section 6.2 lists.
- Replaying or importing a recorded patch checks no protected role. Protections apply to edits made through the operators (`wse.patches` documents this).
- A result that can fill the inline-result limit now lists only the counters it charges in `usage.cumulative`. The rule applies to M1's tools too, and `docs/protocol.md` documents it.
- Which grader Llama-3.2-1B-Instruct's GSM8K tasks should use, strict or with a numeric fallback, is still open from Q1 (Q1 report, Section 9).

## 9. Status

- **Tests:** `uv run --no-sync pytest -q` passes in full at 36fcbfb on the operator's Mac, MPS included (819 tests).
- **Q2:** its exit conditions hold (Section 1).
- **Q3** (Tasks 13–15) is paused until after WSE's open-source release.
