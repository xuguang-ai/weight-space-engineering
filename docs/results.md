# Results

Every measurement so far, with its setup and caveats. Unless a section says otherwise, the runs used Qwen3.5-0.8B at revision `2fc06364715b`, Python 3.14.8, PyTorch 2.15.0, and `transformers` 5.22.2, on an M4 Pro with MPS at batch size 8. The milestone reports in [dev/](../dev) hold the full records.

The sections:

1. Published edits import exactly.
2. p50c against the base on 25 MATH-500 problems.
3. Reconstructing p50c in four steps, on 32 GSM8K problems.
4. Through the MCP tools, with scripted controllers.
5. A live LLM agent on a small model.
6. Other models.
7. Reproducibility and speed.

## 1. Published edits import exactly

The article's published logs hold three Qwen3.5-0.8B edits, named after Astra's budgets of 10, 25, and 50 forward passes, with SHA-256 hashes of every base and edited tensor. `WSE.apply_patch` and `wse import-patch` check every base hash before rebuilding. They rebuild each tensor as the article's `apply_edit.py` does, and require every edited hash and changed-entry count to match.

| Edit | Tensors | Form | Changed entries | WSE weights hash of the edited model |
| --- | ---: | --- | ---: | --- |
| p10c | 30 | low-rank | 72,986,982 | `sha256:77f20a38ee8e…` |
| p25c | 1 | sparse | 262,144 | `sha256:dde0821ecc99…` |
| p50c | 2 | sparse | 130,188 | `sha256:b8dccf8ca463…` |

WSE's own `fill_slice@1` operator reproduces p25c hash-exactly. Zeroing columns 1,792–2,047 of layer 15's `self_attn.o_proj`, the input of attention head 7, gives the same tensor hash and the same weights hash, `sha256:dde0821ecc99…`.

Records: [M2 Q1 report](../dev/m2-q1-report.md), Section 3, and [M2 Q2 report](../dev/m2-q2-report.md), Section 6.1.

## 2. p50c against the base on 25 MATH-500 problems

**Setup:**

- 25 problems chosen by subset seed 0 (`sha256_rank@1`), 1 sample each;
- the profile `qwen35think@1`, the article's sampling (thinking on, a boxed-answer instruction, temperature 0.6, top-k 20, top-p 0.95, presence penalty 1.5), at its full limit of 32,768 new tokens, evaluation seed 0, and the grader `math_boxed@1`;
- one process, 6 h 5 min in all.

The recorded run used `wse compare-generation`. The README's reproduction of it, `compare` with `problems=25`, no `max_new_tokens`, and an `out_dir` (which records every problem and resumes an interrupted run), calls the same runner with the same defaults: subset seed 0, evaluation seed 0, and batch size 8.

| Metric | Base | p50c | Paired difference, p50c − base (95% cluster bootstrap) | Exact 95% intervals (Clopper–Pearson) |
| --- | ---: | ---: | --- | --- |
| Accuracy | 12% (3/25) | 52% (13/25) | +0.40 [+0.16, +0.60] | base [0.025, 0.312]; p50c [0.313, 0.722] |
| Answers cut off at the limit | 92% (23/25) | 36% (9/25) | −0.56 [−0.76, −0.32] | base [0.740, 0.990]; p50c [0.180, 0.575] |
| Mean generated tokens | 30,824 | 21,554 | −9,270 [−13,854, −4,753] | – |

Per problem:

| | p50c correct | p50c wrong |
| --- | ---: | ---: |
| **Base correct** | 2 | 1 |
| **Base wrong** | 11 | 11 |

An exact two-sided sign test on the 12 problems where the two models disagree (11 against 1) gives p = 0.0063.

- **Against the article.** The article reports 7.9% → 48.1% accuracy and 93.0% → 33.1% of answers cut off, with 4 samples on all 500 problems. Each of the four values lies inside the matching exact interval above. Twenty-five problems confirm the direction of the change; they cannot tell whether its size matches.
- **A cut-off answer can count as correct.** One base answer reached the limit and still counts as correct: its last `\boxed{9}` sits inside a thinking block that never closed. The article's evaluation grades cut-off answers the same way.

Record: [M2 Q1 report](../dev/m2-q1-report.md), Section 6.

## 3. Reconstructing p50c in four steps

An experiment follows four steps, each with its own tools:

| Step | Question | WSE |
| --- | --- | --- |
| **Diagnose** | How does the model fail? | Generation evaluation: accuracy, the rate of answers cut off at the token limit, and generated tokens, with intervals; every output's text for measures such as loops |
| **Probe** | Where inside the model? | Weight analyses (norms, singular vectors, projections) and activation probes (mean activations over the last tokens of each output, at the residual stream or any module, and contrasts between groups of outputs) |
| **Edit** | What change fixes it? | Exact patches from registered operators (scale, set, fill, project out a direction, add a low-rank term), or published edits imported with hash checks |
| **Evaluate** | Did it help, and at what cost? | A paired comparison with the base on the same problems, prompts, and seeds, resumable and recorded |

[examples/p50c_four_steps.py](../examples/p50c_four_steps.py) runs them on Qwen3.5-0.8B. It reconstructs a published intervention; it does not rediscover it. The article's agent, GPT-6 Astra, found p50c itself, by analyzing the weights in a Python kernel and running the model on practice problems. In each of layers 15 and 19, p50c subtracts six times the projection of 64 columns of `mlp.down_proj` onto one direction of that layer. Here, the direction and the columns come from p50c's published change: the columns it changes, and the top singular vector of its change to them, signed so that its largest-magnitude entry is positive. WSE's own `project_out` operator then re-executes the edit.

The README's Python example walks through the script step by step, with the downloads it needs: p50c's directory and GSM8K's training set as practice problems, so that the MATH-500 problems stay unseen. The data directory defaults to `~/wse-data`:

```bash
uv run --extra models-hf python examples/p50c_four_steps.py ~/wse-data
```

On an M4 Pro the script took 1 h 54 min, almost all of it in three generation passes of about 37 minutes each; a generation pass generates one answer to each problem, and `compare` makes two. The published version, which also counts loops and prints p50c's change, was run again on a copy of that run's data. Steps 1 to 3 ran in full, and step 1 reproduced the 32 base answers token for token; the edit was stored as the same candidate checkpoint as before, so step 4 resumed from the recorded comparison instead of generating again. It printed:

```text
32 problems, 32 samples
accuracy                     0.062 [0.000, 0.156]
limit_hit_rate               0.938 [0.844, 1.000]
mean_generated_tokens       8063.5 [7843.9, 8192.0]
30 answers cut off at the limit, 2 finished
24 of the 30 cut-off answers end in a loop; the end of the first:
        *   Wait, is it possible the 400 files were the *number of files* he had *downloaded* in the *second* round?
        *   Yes.
        *   Wait, is it possible the 400 files were the *number of files* he had *downloaded* in the *first* round?
        *   No, he downloaded 800 files.
        *   Wait, is it possible the 400 files were the *number of files* he had *downloaded* in the *second* round?
        *   Yes.
        *   Wait, is it possible the 400 files were the *number of files* he had *downloaded* in the *first* round?
        *   No, he downloaded 800 files.
2 tensors changed, 130,188 entries; weights cb5a704b1fd2 -> b8dccf8ca463
  model.language_model.layers.15.mlp.down_proj.weight: 65,260 of 3,670,016 entries, relative L2 change 0.1064, db89c2e60dc5 -> c8df99feb499
  model.language_model.layers.19.mlp.down_proj.weight: 64,928 of 3,670,016 entries, relative L2 change 0.08106, 2c8c10a14a67 -> d0ae5310479d
layer 15: p50c changes 64 columns; cosine of the cut-off minus finished contrast with its direction +0.37
layer 19: p50c changes 64 columns; cosine of the cut-off minus finished contrast with its direction -0.45
130,134 weights changed
32 problems, 32 samples per model
metric                        base   candidate   difference [95% interval]
accuracy                     0.062       0.312   +0.250 [+0.125, +0.406]
limit_hit_rate               0.938       0.656   -0.281 [-0.438, -0.125]
mean_generated_tokens       8063.5      7082.0   -981.5 [-1597.1, -434.7]
```

Answers cut off at the limit fell from 30 to 21 of 32, and accuracy rose from 2 to 10; all three intervals exclude zero. The probe's contrast leans toward p50c's direction, but with only 2 finished answers it points to where to look; it does not establish a mechanism. The re-executed edit is close to p50c's weights but not identical, because the two round differently. The subsections below take each step in turn.

**Setup:**

- 32 problems from GSM8K's training set, chosen by subset seed 0 (`sha256_rank@1`), so that the MATH-500 problems stay unseen;
- `qwen35think@1` at 8,192 new tokens, 1 sample each, and the grader `gsm8k_numeric@1`.

### Step 1: Diagnose

| Metric | Base | 95% interval |
| --- | ---: | --- |
| Accuracy | 2/32 (0.062) | [0.000, 0.156] |
| Answers cut off at the limit | 30/32 (0.938) | [0.844, 1.000] |
| Mean generated tokens | 8,063.5 | [7,843.9, 8,192.0] |

The 30 cut-off answers form one group and the 2 finished ones the other.

**Loops.** An answer ends in a loop when at least half of the lines in its last 2,000 characters repeat an earlier line there; the window's first and last lines, which may be partial, are left out. By this measure 24 of the 30 cut-off answers end in a loop, and neither finished answer does. The first, to `gsm8k_27`, enters a four-line loop at line 250 of its 369 lines (the last 8,709 of its 26,544 characters) and never leaves it. The count depends a little on the window: 21 at 1,000 characters, 24 at 4,000, and 21 at 8,000. Two other measures agree. Ignoring digits gives the same 24. Counting repeated text instead of repeated lines, at least half of the 40-character strings in the last 2,000 characters occur more than once in 25 of the 30 cut-off answers (median share 1.00), and in neither finished answer (0.00 each).

### Step 2: Probe

**What was measured.** The probe reads the last 1,024 tokens of each answer at the output of `mlp.down` in layers 15 and 19, with the answer's full context, and averages them per answer. The contrast is the mean of the cut-off answers minus the mean of the finished ones. Its cosine with p50c's direction is +0.37 in layer 15 and −0.45 in layer 19. A singular vector's sign is arbitrary, so the size is what counts.

**Random directions are the wrong baseline.** In 1,024 dimensions, the cosine of a random direction stays below 0.103 in magnitude in 99.9% of draws. But these answers vary strongly along p50c's direction itself: it carries 6.7% of their variance in layer 15 and 14.4% in layer 19, against 0.1% for a random direction. So any split of the answers tends to produce a contrast that leans toward it.

**Arbitrary splits are the right baseline.** There are 496 ways to set 2 of the 32 answers apart.

| | Layer 15 | Layer 19 |
| --- | ---: | ---: |
| Median cosine magnitude over the 496 splits | 0.24 | 0.36 |
| 95th percentile | 0.34 | 0.43 |
| The actual split | 0.37 | 0.45 |
| Its rank among the 496 | 4th (p = 0.008) | 7th (p = 0.014) |

In each layer, all 32 answers' components along the direction have the same sign. The 2 finished answers have the smallest in magnitude in layer 15, and two of the three smallest in layer 19.

**Two caveats:**

- **Group size.** The finished group has 2 answers, so its mean can follow one unusual answer.
- **Position.** Each window is the last 1,024 tokens of an answer. A cut-off answer's window ends inside a loop, and a finished answer's window ends at its final answer. The contrast therefore mixes "will be cut off" with "where in the answer". The actual split also has the largest contrast norm of all 496 splits: the finished answers differ from the rest in many directions, not only in p50c's.

**Reading.** The probe points to where to look. It does not establish that this direction is the mechanism of the looping. That would take more finished answers, windows aligned by position, and controls such as random directions and other layers.

### Step 3: Edit

**p50c's form,** measured from its published change. In each layer, the change to the 64 changed columns is rank 1: its top singular value holds 0.999995 of the change's energy in layer 15 and 0.999991 in layer 19. The least-squares alphas are 6.0003 and 6.0002, and the two layers' directions have a cosine of −0.85.

**The edit.** Two `project_out` edits with alpha 6, one per layer, on p50c's 64 columns of `mlp.down`, with the recovered direction. Alpha 6 subtracts six times each column's projection onto the direction, which turns that component into −5 times itself; alpha 1 would remove it. The edits change 130,134 weights; p50c changes 130,188.

**Against p50c's published weights,** in the 64 edited columns (65,536 entries per layer), counting how many representable BF16 values apart the two entries are:

| Layer | Differing entries | One BF16 step apart | Two steps | More | Largest difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| 15 | 5,924 | 4,960 | 406 | 558 | 0.00049 |
| 19 | 6,162 | 5,117 | 409 | 636 | 0.00049 |

About 91% of the entries (90.8%) are byte-identical. The rest differ by rounding: p50c's published weights match none of five rounding paths tried. The two that compute in float64 or float32 and round to BF16 once come closest. Exact replay (Section 1) is the hash-exact path; re-executing a recipe is not. (The M2 Q2 report buckets the same differences by a spacing rule instead of steps, which moves up to 6 entries between buckets.)

### Step 4: Evaluate

| Metric | Base | Edited | Paired difference [95% interval] |
| --- | ---: | ---: | --- |
| Accuracy | 2/32 (0.062) | 10/32 (0.312) | +0.250 [+0.125, +0.406] |
| Answers cut off at the limit | 30/32 (0.938) | 21/32 (0.656) | −0.281 [−0.438, −0.125] |
| Mean generated tokens | 8,063.5 | 7,082.0 | −981.5 [−1,597.1, −434.7] |

Per problem:

- **Accuracy.** 8 problems were right only with the edit, none only with the base, and 2 with both.
- **Cut-off answers.** 9 answers that the base ran to the limit finished with the edit, and none went the other way.
- **Loops.** By step 1's measure, none of the 21 answers that the edited model still runs to the limit ends in a loop (0 at 1,000 characters, 2 at 4,000, and 1 at 8,000). By the repeated-text measure, their median share is 0.04, and 2 of the 21 reach 0.5. They keep re-checking the problem in new words until the limit, with sentences such as "Let's try a different interpretation." and "Let's verify the wording again." The edit removed the verbatim loops but not the over-long reasoning: 9 more answers finished, and 21 still did not.

The loop counts come from the recorded comparison, which keeps every output's text, with step 1's `ends_in_loop`:

```python
import json
from pathlib import Path


def ends_in_loop(text: str, window: int = 2000) -> bool:
    """True when at least half of the lines in the last ``window`` characters repeat an earlier line there."""
    lines = [line.strip() for line in text[-window:].split("\n")[1:-1]]
    lines = [line for line in lines if line]
    return bool(lines) and 2 * (len(lines) - len(set(lines))) >= len(lines)


rows = [json.loads(line) for line in Path("~/wse-data/runs/four-steps/problems.jsonl").expanduser().open()]
for checkpoint in ("base", "candidate"):
    cut_off = [row["outputs"][0]["text"] for row in rows
               if row["checkpoint"] == checkpoint and row["problem"]["samples"][0]["limit_hit"]]
    loops = sum(ends_in_loop(text) for text in cut_off)
    print(f"{checkpoint}: {loops} of {len(cut_off)} cut-off answers end in a loop")
```

```text
base: 24 of 30 cut-off answers end in a loop
candidate: 0 of 21 cut-off answers end in a loop
```

### Reproducibility

- **Two full runs.** They ran in separate processes, and their comparisons started 2 h 38 min apart. The second started from a fresh download of the model and the data into new directories.
  - Both records hold the same weights hashes for the base and the edited model (`sha256:cb5a704b1fd2…` and `sha256:44918f4ac03d…`), and the same data and profile hashes.
  - They hold the same 64 answers, token for token, with the same grades. Apart from the random checkpoint and batch IDs, the timestamps, and the measured times, the records are the same.
- **What changed between them.** Only the script:
  - a guard that skips step 2 when either group is empty (both groups were non-empty);
  - the sign convention of the direction, which flips the printed layer-19 cosine from +0.45 to −0.45 and leaves the edit bit-identical, because negating a direction is exact;
  - comments.
- **A third process.** The rerun behind the permutation check of step 2 reproduced the 32 base answers token for token.
- **The published script.** On 2026-11-11 it ran on a copy of the second run's data and repeated steps 1 to 3, in 42.5 minutes while other jobs shared the machine. Step 1 reproduced the 32 base answers token for token, with the same finished flags. `save()` returned the stored candidate, `ckpt_74cd8078…`, so step 4 resumed from the recorded comparison and printed the same table.
- **An earlier operator run.** An earlier run of the same edit through `WSE.compare`, on the same 32 problems, gave the same aggregates.

**Time.** The whole script took 1 h 54 min. Each generation pass took about 37 minutes: 2,224 s per model in the second run's comparison, and 2,235 s in the first. The probe and the edits took about 1.6 minutes together.

Record: [M2 Q2 report](../dev/m2-q2-report.md), Sections 4.3, 6.1, and 6.2.

## 4. Through the MCP tools, with scripted controllers

Both runs drove a real `wse serve` session under the policy `qwen_loop_v1` (10 generation passes of at most 32 problems and 8,192 new tokens per answer, 25 probe calls, and 6 hours of computation; [all of its budget](experiments.md#budgets)), through the OpenAI Agents SDK's tool loop, with a script choosing each call in place of a model.

Both used another 32 GSM8K training problems, chosen by subset seed 0. Their problem IDs had another prefix, and the subset is chosen by a hash of the IDs, so these problems do not overlap Section 3's.

### 4.1 The reconstruction through the tools

One patch of two `project_out` operations, with the directions recovered from p50c's published change, its 64 columns per layer, and alpha 6; it changes 130,134 weights.

| 32 problems | Base | Edited | Paired difference [95% interval] |
| --- | --- | --- | --- |
| Accuracy | 3/32 (9.4%) | 12/32 (37.5%) | +0.28 [+0.13, +0.44] |
| Answers cut off at the limit | 30/32 (93.8%) | 19/32 (59.4%) | −0.34 [−0.50, −0.19] |
| Mean generated tokens | 7,924 | 6,849 | −1,075 [−1,858, −408] |

Each generation pass took about 39 minutes. The run made 477 tool calls, most of them job polls.

### 4.2 A scripted discovery

**What it reads.** The strategy `direction_search@1` reads no published weights, directions, columns, or evaluation items. It does take p50c's form: `project_out` at the output of `mlp.down`, 64 columns, and alpha 1 or 6.

**Its four steps:**

1. **Diagnosis.** One generation pass of the base: 3/32 correct, 30/32 cut off.
2. **Probe and choices.**
   - **Probe.** One probe of the last 1,024 tokens of each answer at the input and output of `mlp.down` in all 24 layers.
   - **Layer.** A layer score: the norm of the contrast divided by the mean of the two group means' norms. Layer 9 scored highest, at 1.18.
   - **Columns.** The 64 columns of layer 9's `mlp.down` whose cosine with the contrast is largest in magnitude.
3. **Edit.** Two candidates, `project_out` of the contrast from those columns with alpha 1 and with alpha 6.
4. **Evaluation.** One generation pass per candidate, a paired comparison with the base, and a selection by point estimates.

| 32 problems | Base | Alpha 1 | Alpha 6 (selected) |
| --- | --- | --- | --- |
| Accuracy | 3/32 | 3/32; +0.00 [−0.09, +0.09] | 5/32; +0.06 [−0.06, +0.19] |
| Answers cut off at the limit | 30/32 | 29/32; −0.03 [−0.16, +0.06] | 27/32; −0.09 [−0.22, +0.03] |
| Mean generated tokens | 7,924 | 7,596; −328 [−1,022, +333] | 7,849; −75 [−511, +351] |

It found no repair: every paired interval includes zero.

**Why it missed** (analysis after the run, with knowledge of p50c):

- **Layer choice.** The score ranked p50c's layers 10th (layer 15, 0.81) and 17th (layer 19, 0.60) of 24. It ignores the spread within each group, and the finished group has 2 answers.
- **Direction.** At p50c's layers the contrast is aligned with p50c's direction (cosines +0.35 and −0.44). These cosines must be read against arbitrary splits of the answers, as in Section 3.
- **Columns.** The column rule picks 27 and 19 of p50c's 64 columns; with p50c's own direction, 33 and 20.

### 4.3 Waiting on long jobs

`wse_get_job` waits at most 10 seconds, and a generation pass takes about 37 minutes, so the discovery polled 669 times over its three passes. By an estimate of the conversation a model would carry (characters divided by 4, without the tool definitions), it passes 128,000 tokens at request 93 and 200,000 at request 202, within the first pass. Long jobs need another way to wait before LLM agents run them.

Record: [M2 Q2 report](../dev/m2-q2-report.md), Sections 3 and 4.

## 5. A live LLM agent on a small model

**Task.**

- `signed_sum_v1`: classify (x0, x1) by the sign of x0 + x1 with an FP32 2 → 4 (ReLU) → 2 MLP of 22 parameters, evaluated on the CPU.
- The planted fault multiplies row 1 of `layers.1.weight` by −1, so the logits tie and accuracy is 0.5.
- The development suite has 32 pairs; the final suite has 512 pairs that the agent never sees.

**Controller.** `openai_agents@1` with openai-agents 0.24.0 and the model `gpt-5.6-luna`, at most 100 turns and 900 s per trial, under the policy `small_model_v1`.

**Result.**

- All 5 trials of the acceptance set found and committed the fix, `set_slice layers.1.weight[1:2, 0:4] = [[1, -1, 1, -1]]`.
- Final accuracy was 1.0 on 512 pairs (Clopper–Pearson 95% lower bound 0.9928), against 0.5 for the unedited control.
- The trials used 9–12 turns and 12–16 tool calls each, and every export replayed to byte-identical weights.
- An earlier set reached 3 of 5 because of a harness defect, fixed before the acceptance set.

**Single runs on 2026-11-11.**

- **The README's agent example,** [examples/signed_sum/agent_api.py](../examples/signed_sum/agent_api.py) with `--model gpt-5.6-luna`, which writes out the model, task, data, and policy, under `signed_sum_policy_v1` (the values of `small_model_v1`). It met the target in 8 turns and 11 tool calls, with 68,866 input and 1,428 output tokens, in 20 s. The agent read the four tensors and stated that the two output rows are identical, and that setting class 1's weights to the negation of class 0's gives the logits (−s, s), where s = x0 + x1. It set class 1's row to [1, −1, 1, −1] with `set_slice`, as the acceptance trials did, changing 4 of the 22 parameters, and its first development evaluation met the target. Final accuracy was 1.0 on 512 pairs against 0.5 for the control, and the export replayed to byte-identical weights. Of its budget it used 1 of 10 development evaluation batches, 1 of 20 candidate commits, and 4 of 100 tensor reads. An earlier run of the example that day, before a change that only moved its task and policy definitions into variables, made the same edit in 8 turns and 11 tool calls, with 69,191 input and 1,425 output tokens.
- **The example's previous version,** which prepared the same task through `build_fixture.py` under `small_model_v1` (run from a fresh clone). It met the target in 8 turns and 11 tool calls, with 68,681 input and 1,402 output tokens, in about 30 s. Its one edit, `scale_slice` by −1 on `layers.1.weight[1:2, 0:4]`, gives the same weights, with the same final result and replay.

The instructions of these runs are in [experiments.md](experiments.md#what-the-agent-is-told).

One small fixture says little about agent capability in general.

Record: [M1 agent trials](../dev/m1-agent-trials.md).

## 6. Other models

- **Qwen3-0.6B and gemma-3-1b-it** at Level 0, with no new code:
  - import;
  - bitwise-equal logits against `from_pretrained` on CPU and MPS;
  - a patch round trip;
  - the analyses;
  - probes equal to the hidden states;
  - generation.
- **Llama-3.2-1B-Instruct:**
  - bitwise-equal logits on CPU and MPS, and a patch round trip;
  - on 25 GSM8K test problems, the strict boxed-answer grader scores 0 of 25, because the model never writes a boxed answer. In 8 of the 25 outputs the last number equals the reference. Which grader its GSM8K tasks should use is still open.

Records: [M2 Q1 report](../dev/m2-q1-report.md), Section 4, and [M2 Q2 report](../dev/m2-q2-report.md), Section 5.

## 7. Reproducibility and speed

- **Within one machine.** A generation pass with a fixed suite, profile, seed, and batch size is reproducible to the token. Two base passes in separate experiments of Section 4 produced the same 32 outputs, 253,567 generated tokens in all. Section 3's two full runs recorded the same answers, and its permutation rerun reproduced the base answers.
- **Across devices or library versions.** Bitwise reproducibility is not claimed. Edits are computed on the CPU in a platform-independent way, so a patch should hash the same on every platform; so far this has been checked on macOS only.
- **Speed on the M4 Pro (MPS).**
  - A pass of 32 problems at up to 8,192 tokens at batch size 8 takes 37–40 minutes, about 113 generated tokens per second.
  - At 32,768 tokens, a chunk of 8 rows took 95–108 ms per step.
