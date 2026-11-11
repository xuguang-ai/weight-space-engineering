# M2 Q1 Report: Load, Replay, and Evaluate Open-Weight Models

This report covers Task 7 of the [M2 plan](agentic-weight-space-engineering-m2-plan.md). It records:

- the measurements from the operator's Mac;
- the exact replay of the article's published edits;
- Llama-3.2-1B-Instruct through Level 0 with no new code;
- the pilot of the base-versus-p50c comparison, and the 25-problem comparison itself;
- cost estimates for the three protocols of Section 4.4.

The 25-problem base-versus-p50c comparison ran from 19:31 EST on 2026-11-09 to 01:36 EST on 2026-11-10. It moved in the article's direction (Section 6). The decision gate (Section 8) is the user's.

Status on 2026-11-10: every Task 7 run is done, and Task 7's review is complete (Section 9). The decision gate is open.

## 1. Environment

| Item | Value |
| --- | --- |
| Host | Apple M4 Pro (14 CPU cores, 20-core GPU), 48 GB, macOS 26.7.1 |
| Python | 3.14.8 (project `.venv`, `uv run --no-sync`) |
| Libraries | torch 2.15.0, transformers 5.22.2, tokenizers 0.23.3, safetensors 0.8.0, numpy 2.5.4, math-verify 0.9.0, huggingface-hub 2.4.0 |
| Devices | `mps` (PyTorch MPS) and `cpu`; attention implementation `sdpa` |
| Qwen3.5-0.8B | `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17` |
| Llama-3.2-1B-Instruct | `meta-llama/Llama-3.2-1B-Instruct` at `9213176726f574b556790deb65791e0c5aa438b6` |
| Article logs | `RikiyaT/lm_understand-logs` at `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f` (sparse checkout) |
| MATH-500 | `test.jsonl`: 500 problems, data hash `sha256:850072a7b4733bbe81d5651f2255837f7ab38da6a5442f4b4771485aa692f9f3`, file `sha256:35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132` |
| GSM8K test | `test.jsonl`: 1,319 problems, data hash `sha256:82c9bf913b4e4cc1c52eb6bfc5f92cbd78ed2f88926f6ab86df5507a70256e0e`, file `sha256:3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` |

Transformers warns at load that Qwen3.5's Gated DeltaNet layers fall back to its reference PyTorch implementations, because `causal_conv1d` and `flash-linear-attention` are not installed (they are CUDA kernels). Every Qwen3.5 measurement below uses these fallbacks.

### Imported models

Both models were imported with `wse import-hf` into a fresh workspace.

| Model | Tensors | Parameters | Stored dtypes | Alias groups | `weights_hash` |
| --- | ---: | ---: | --- | --- | --- |
| Qwen3.5-0.8B (text model) | 321 | 752,393,024 | 285 BF16, 36 F32 | `lm_head.weight` = `model.language_model.embed_tokens.weight` | `sha256:cb5a704b1fd2b2649fab2005f630c88eb4a7dd60b80b08183ca7e592ce6b53a5` |
| Llama-3.2-1B-Instruct | 147 | 1,235,814,400 | 147 BF16 | `lm_head.weight` = `model.embed_tokens.weight` | `sha256:359815b46de606c4a754b9bce78e77f41bff4d4b38bc9abf8e58b7f96562bb23` |

Identity hashes:

- **Qwen3.5-0.8B:** config `sha256:5d179f2308164eadf8082febd23e6c01ce636721e5b15ca539074d0902d8d31f`, tokenizer `sha256:89c311e61febfaf933944ad719158c6c2c45dbf74067de0f15265ae5410a87fc`, chat template `sha256:273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80`. The snapshot has no `generation_config.json`.
- **Llama-3.2-1B-Instruct:** config `sha256:2ff3facc5554c49c17cf89d91e3bb7a798683bab2a258b987937d95617088680` (see the note below), tokenizer `sha256:ee50c999c9968897333d6ab4ea48fcb978d42a08858e2ece171e4177687ef261`, chat template `sha256:5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4`, generation configuration `sha256:a63b7e5c108ea78be8b33c22a7bfcbf7430a0cd8ffb7de53bc9ca436df195edf`.

  Note on this configuration hash. Task 7 imported Llama at Level 0, and its stored configuration carried a `rope_theta` that `transformers` had written into `config.json`'s `rope_scaling` while building the module. Since M2 Task 8, a Llama import selects the `llama@1` descriptor, and since that task's review fixes it stores `config.json` unchanged, so the same snapshot now gets configuration hash `sha256:756364057c62e1ece05d2137b31bc0c2b355ff2f59c4ad2be31035ed565d3b99`. Its weights, tokenizer, chat-template, and generation-configuration hashes are unchanged. Checkpoints imported before still open with their recorded configuration. Patches recorded on them replay only on them.

Imports took 7 s for Qwen and 10 s for Llama. Importing either snapshot again returned the same checkpoint and stored nothing.

## 2. Throughput

Each configuration ran in its own process through WSE's engine (`HFTorchModel.generate`) with the decoding path of `qwen35think@1`:

- temperature 0.6, top-k 20, and top-p 0.95, with one seeded generator per row;
- a presence penalty of 1.5;
- no stop tokens, so every row generates the requested number of tokens.

Prompts are random token IDs of the stated length.

How each column was measured:

- **Prefill:** the time of a one-token generation. It includes the single forward pass over the prompt and one sampling step.
- **Decode:** (total time − prefill) / (new tokens − 1) per step. Tokens per second count every row of the batch.
- **Memory:** peak process RSS and the peak MPS driver allocation, polled every 50 ms. Model weights are included.

A warm-up generation ran before timing. CPU rows used fewer new tokens because each CPU step of Qwen3.5 takes seconds.

| Model | Device | Batch | Prompt tokens | New tokens | Prefill (s) | Decode (ms/step) | Decode tokens/s | Peak RSS (GB) | Peak MPS (GB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | mps | 1 | 256 | 64 | 0.09 | 29.1 | 34.4 | 2.48 | 2.12 |
| Qwen3.5-0.8B | mps | 8 | 256 | 64 | 0.60 | 60.9 | 131.3 | 2.53 | 2.10 |
| Qwen3.5-0.8B | mps | 32 | 256 | 64 | 2.41 | 212.9 | 150.3 | 2.68 | 4.17 |
| Qwen3.5-0.8B | cpu | 1 | 256 | 32 | 6.65 | 5,030 | 0.20 | 2.39 | – |
| Qwen3.5-0.8B | cpu | 8 | 256 | 8 | 16.4 | 9,936 | 0.81 | 2.83 | – |
| Qwen3.5-0.8B | cpu | 32 | 256 | 8 | 37.3 | 9,297 | 3.44 | 4.30 | – |
| Llama-3.2-1B-Instruct | mps | 1 | 256 | 64 | 0.10 | 21.1 | 47.4 | 3.35 | 3.02 |
| Llama-3.2-1B-Instruct | mps | 8 | 256 | 64 | 0.66 | 32.6 | 245.5 | 3.38 | 3.01 |
| Llama-3.2-1B-Instruct | mps | 32 | 256 | 64 | 2.59 | 102.3 | 312.9 | 3.45 | 4.06 |
| Llama-3.2-1B-Instruct | cpu | 1 | 256 | 8 | 2.14 | 30.0 | 33.4 | 3.23 | – |
| Llama-3.2-1B-Instruct | cpu | 8 | 256 | 8 | 14.1 | 137.7 | 58.1 | 3.62 | – |
| Llama-3.2-1B-Instruct | cpu | 32 | 256 | 8 | 60.0 | 719.5 | 44.5 | 4.64 | – |

**Long contexts (Qwen3.5-0.8B).** The prompt is prefilled once, then 64 tokens are decoded.

| Device | Batch | Prompt tokens | Prefill (s) | Decode (ms/step) | Decode tokens/s | Peak MPS (GB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mps | 1 | 4,096 | 1.18 | 30.4 | 32.9 | 2.17 |
| mps | 1 | 30,000 | 11.8 | 36.2 | 27.6 | 8.11 |
| mps | 8 | 4,096 | 9.50 | 66.9 | 119.6 | 8.10 |
| cpu | 1 | 4,096 | 32.1 | 4,940 | 0.20 | – (RSS 2.94) |

Findings:

- **MPS is the only practical device for Qwen3.5.**
  - On CPU a decoding step takes 5 to 10 s, about 170 times slower than on MPS.
  - Llama decodes on the same CPU at 30 ms per step, so the cost lies in Qwen3.5's own CPU path, not in BF16 matrix products generally. The Gated DeltaNet reference fallbacks are the only Qwen-specific code on that path.
  - CPU stays usable for parity checks and short reads, not for generation.
- **Decode cost grows slowly with context.** At batch 1 on MPS, a step takes 29.1 ms after 256 prompt tokens, 30.4 ms after 4,096, and 36.2 ms after 30,000. Only 6 of Qwen3.5's 24 layers use full attention. Its KV cache is 12 KiB per token per row, so a row at 32,768 tokens holds 0.4 GB.
- **Batching helps up to about 8 rows.** Aggregate decode throughput rises 3.8 times from batch 1 to batch 8 (34 to 131 tokens/s), and only 15% more at batch 32 (150 tokens/s). Per-step work outside the model (the vocabulary of 248,320 and sampling on the CPU) grows with the batch.
- **Not measured: decode at batch 8 with a 32,768-token context.** Prefilling 8 rows of 30,000 tokens would need far more attention memory than a real run, whose prompts are about 250 tokens. The estimates below therefore extrapolate:
  - the batch-1 slope from 4,096 to 30,000 tokens is +0.23 ms per 1,000 tokens;
  - for 8 rows, assumed to grow eight times, it is +1.8 ms per 1,000 tokens;
  - this gives about 118 ms per step at 32,768 tokens, and a mean of about 90 ms per step over a full-length generation (from 61 ms);
  - measured afterwards (Section 6): full-length chunks of 8 took 94.9–108.4 ms per step, 100.8 ms on average, about 12% above this extrapolation.

## 3. Replay of the Published Edits (Task 5)

`wse import-patch --format lm-understand-v1` was re-run in this environment on the imported Qwen3.5-0.8B for p10c, p25c, and p50c. Each import:

- verified every published `base_sha256` before rebuilding;
- rebuilt each tensor as the article's `apply_edit.py` does;
- matched every published `edited_sha256` and changed-entry count.

All three hashes agree with the edit manifests in the logs.

| Run | Tensors | Form | Changed elements | Patch bytes | Candidate `weights_hash` |
| --- | ---: | --- | ---: | ---: | --- |
| p10c | 30 | low-rank | 72,986,982 | 173,052,513 | `sha256:77f20a38ee8e061706a78d07ad95fb1d398380fe6ac017cb65c5400720bd3d77` |
| p25c | 1 | sparse | 262,144 | 2,623,594 | `sha256:dde0821ecc9998559bd1ba5847d62222a3a4f2da8bea27be1fca5266884eb9f4` |
| p50c | 2 | sparse | 130,188 | 1,305,161 | `sha256:b8dccf8ca463a97d9f6a783651f17b02fecdd740b154bb5ca53983e39852259f` |

The p25c and p50c hashes in full:

| Tensor | `base_sha256` | `edited_sha256` | Changed entries |
| --- | --- | --- | ---: |
| p25c `model.language_model.layers.15.self_attn.o_proj.weight` | `2f5be1b0f59d3bf45a992e18c26b3658ce5112527ea570b00d05fe1a48140092` | `3ece11f84f90d983f9a5e47e4c4c26906b4d7b8aa0e6f3ff56308033ba4e3caf` | 262,144 |
| p50c `model.language_model.layers.15.mlp.down_proj.weight` | `ef60d2c687bab19464dca9c8bfd7777e46ad226df0c9e25643b0bab04b5a1cca` | `82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53` | 65,260 |
| p50c `model.language_model.layers.19.mlp.down_proj.weight` | `638eedeb701b94aa861f4438e615c07ec6c60ee703db1bfee2de577bb6498c3e` | `8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2` | 64,928 |

p10c changes 30 tensors in low-rank form, all in layers 8–22:

- 15 `mlp.down_proj`, in layers 8–22;
- 12 `linear_attn.out_proj`;
- 3 `self_attn.o_proj`, in layers 11, 15, and 19.

Its 30 base and edited hashes are those of `exp8_math/runs/p10c/edit/manifest.json`. Each import ran in about 9–11 s. The p50c candidate adds exactly two blobs (Task 5). Importing p50c again returns the same patch and candidate.

## 4. Llama-3.2-1B-Instruct Through Level 0

No Llama-specific code was used or added.

- **Grep check:** `grep -rn -i llama src/wse` finds nothing. One docstring of `wse.evaluators.sampling` used the Llama repository name as an example; it now uses a neutral name.
- **Import:** as in Section 1, with no family descriptor (Level 0).
- **Logits parity:** three prompts compared with `AutoModelForCausalLM.from_pretrained(snapshot, dtype="auto")`. The maximum absolute difference was **0.0** on CPU and **0.0** on MPS (bitwise equal).
- **Patch:** a managed `scale_slice` with factor 0.5 on rows 0–15 of `model.layers.7.mlp.down_proj.weight`, all 8,192 columns. It used the M1 small-model policy, a run grant, and an experiment.
  - The preview stored one dense binary block.
  - Apply replayed the stored block with no operator.
  - Replaying the stored patch file on the base, applying the operator directly, and replaying its blocks gave the same weights.

  | Check | Hash |
  | --- | --- |
  | Preview result, applied candidate, stored-patch replay, direct operator, direct replay | `sha256:07efc1dc6a41f8cf48438f5d92d786d17087dea0534272e9e66c90637ac96c33` |
  | Edited tensor blob (applied candidate and stored-patch replay) | `d1ab779dc73a01f0b39e43920a7fe4ab7120b5c23de2e39add2e81760753548f` |
  | Base tensor blob | `a7c35432c958b20627cb04fa50b56429affc65085baebc8ba3603af31bb928f2` |

  M1's export-and-replay path (`wse export` then `wse replay`) does not yet carry open-weight models. An export without base weights has nothing to replay against. An export with base weights writes a 3.0 GB `model.safetensors` bundle, which the M1 bundle reader refuses above 64 MiB. Section 9 lists this.
- **GSM8K:** the first 25 GSM8K test problems with `llama_3_2_1b_instruct_default@1` and `gsm8k_numeric@1`, one sample each, on MPS at batch size 8:
  - the profile was frozen at import: temperature 0.6, top-p 0.9, top-k 50, stops 128001, 128008, and 128009, and at most 2,048 new tokens;
  - the suite was registered from the file in evaluator storage, and the run used `Engine.evaluate_generation`.

  | Metric | Value | Interval (95%) |
  | --- | ---: | --- |
  | Accuracy | 0.0 (0 of 25) | Clopper-Pearson [0.000, 0.137] |
  | Limit-hit rate | 0.0 | Clopper-Pearson [0.000, 0.137] |
  | Mean generated tokens | 232.6 | cluster bootstrap [197.4, 270.2] |

  The run took 51.5 s for 5,815 generated tokens. Its evaluator configuration hash is `sha256:98e3a2fb7b017b48a80e45325130398a0ae6760cf6fcecc4db9eb04ce8a1bfc4`, and its prompts hash is `sha256:023a741227da54d86520d12a74019c7185c0ea826a728fc69c39aac51ae70bfc`.

  The zero comes from the answer format, not from the pipeline:
  - The rendered prompt ends with the boxed-answer instruction and the assistant header.
  - No output contains `\boxed`, so `gsm8k_numeric@1`, which reads only boxed answers, finds no answer in any of the 25.
  - As a diagnostic only, in 8 of the 25 outputs the last number in the text equals the reference. In the first problem, for example, the output ends "So, Janet makes $18 every day at the farmers' market."
  - Regenerating the 25 rows gave identical lengths, so the diagnostic describes the reported outputs.

  Whether a GSM8K grader should fall back to the last number for models that ignore the instruction is a decision for the plan's author (Section 9).

## 5. Pilot of the Comparison Runner

`wse compare-generation` (`wse.evaluators.generation_runs`) runs the base-versus-candidate comparison through generation@1's own steps, the same ones `evaluate_generation` runs:

1. `plan_generation`;
2. `run_problems` (prompt rendering, per-row seeds, batching, grading);
3. `generation_result` (metrics, intervals, identities);
4. then `compare_generation`.

It works through the problems in chunks of `--batch-size`, base first and then candidate for each chunk. It writes:

- `run.json`: the fixed configuration;
- `problems.jsonl`: one fsynced line per finished (checkpoint, problem), with output texts;
- `summary.json`: the final results.

A restart skips the recorded problems. A test checks that the runner's results equal single evaluations of the same subset, batch size, and device.

Pilot settings:

- 2 problems chosen with subset seed 0: `test/geometry/711.json` and `test/geometry/1108.json`, at file positions 308 and 314;
- `qwen35think@1` with `--max-new-tokens 1024`;
- MPS, batch size 8, so one chunk of 2 rows per checkpoint.

| Checkpoint | Accuracy | Limit-hit rate | Mean generated tokens | Generation time |
| --- | ---: | ---: | ---: | ---: |
| base | 0/2 | 2/2 | 1,024 | 36.6 s |
| p50c | 0/2 | 2/2 | 1,024 | 36.5 s |

- **Time:** the whole command took 82.5 s, including model loading. At batch 2 that is 35.7 ms per step, consistent with Section 2.
- **Outputs:** all four outputs were still thinking at 1,024 tokens, as expected under this limit. Each begins "Here's a thinking process …". Base and p50c outputs first differ at characters 171 and 431.
- **Prompts and configuration:** both checkpoints used the same prompt hashes, seeds, and evaluator configuration (`sha256:17f47446b122e9ff4ea5b8f4b7ce52686efda2069576a24e86039ba74cc2e259`).
- **Restart:** rerunning the command found everything recorded and finished in 9.8 s without generating.

## 6. Base Versus p50c on 25 MATH-500 Problems

The comparison ran with these settings:

- 25 problems chosen by subset seed 0 (`sha256_rank@1`);
- 1 sample each, `qwen35think@1` at its full 32,768-token limit;
- evaluation seed 0;
- MPS at batch size 8.

Running the block again:

- import and patch import return the existing checkpoints;
- `compare-generation` resumes from its output directory. A run directory written before commit 8fb1eeb records no library versions, so it is read but not continued; continue in a new directory.

After an interruption, run the same block again. `W` and `OUT` must lie outside the repository.

```bash
# from the repository root
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
M=~/Desktop/wse-models
W=/path/outside/the/repository/q1-workspace
OUT=/path/outside/the/repository/q1-math500-25-seed0
field() { uv run --no-sync python -c "import json, sys; print(json.load(sys.stdin)['$1'])"; }
BASE=$(uv run --no-sync wse --workspace "$W" import-hf "$M/Qwen3.5-0.8B" --repo Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 2>/dev/null | field checkpoint_id)
P50C=$(uv run --no-sync wse --workspace "$W" import-patch "$M/lm_understand-logs/exp8_math/runs/p50c/edit" \
  --format lm-understand-v1 --base "$BASE" --repo RikiyaT/lm_understand-logs \
  --revision 4d09ba6d6a7e891dd8063974dfe81b3d78e3494f 2>/dev/null | field checkpoint_id)
uv run --no-sync wse --workspace "$W" compare-generation --base "$BASE" --candidate "$P50C" \
  --dataset "$M/datasets/MATH-500/test.jsonl" --dataset-format math500 --problems 25 --subset-seed 0 \
  --profile qwen35think@1 --grader math_boxed@1 --device mps --batch-size 8 --out "$OUT" 2>> "$OUT.log"
```

Progress goes to `$OUT.log`, one line per finished (checkpoint, problem). Results are written to `$OUT/summary.json`.

The run used `W=~/Desktop/wse-models/workspaces/q1` and `OUT=~/Desktop/wse-models/runs/q1-math500-25-seed0`, under `caffeinate -is` so that the Mac did not sleep. It ran as one process, from 2026-11-10T00:31:07Z to 06:36:17Z, with no interruption or restart.

**Expected duration: about 5.5 hours (range 4.5–6.5 hours).** The estimate:

- The 25 problems form chunks of 8, 8, 8, and 1 per checkpoint.
- A chunk runs until its longest row stops. The base reaches the limit on about 93% of answers, and p50c on 33% (article). So nearly every chunk of 8 runs the full 32,768 steps: with p50c, the chance that none of 8 rows reaches the limit is 0.67⁸ ≈ 4%.
- A chunk of 8 takes about 32,768 × 90 ms ≈ 49 min, using the extrapolated mean step time of Section 2.
- The single-row chunk takes at most 32,768 × 33 ms ≈ 18 min.
- Per checkpoint that is about 2.7 h, and for both about 5.5 h.

At batch size 1 the same comparison would take about 12 hours: 25 × 31,000 tokens for the base and 25 × 21,700 for p50c, at 33 ms per token. An interruption loses at most the chunk in progress, up to about 49 min.

The expected direction, from the article's MATH-500 protocol:

| | Base | p50c |
| --- | ---: | ---: |
| Accuracy | 7.9% | 48.1% |
| Limit-hit rate | 93.0% | 33.1% |

With 25 single samples and a different engine, the size of the change is not expected to match.

### Results

The numbers come from `$OUT/summary.json`. They were also recomputed from `$OUT/problems.jsonl`: re-grading the 50 stored outputs with `math_boxed@1` reproduced every recorded grade, and every limit hit is an unfinished output of 32,768 tokens.

| Metric | Base | p50c | Paired difference, p50c − base (95% cluster bootstrap) | Exact 95% intervals (Clopper-Pearson) |
| --- | ---: | ---: | --- | --- |
| Accuracy | 12% (3/25) | 52% (13/25) | +0.40 [+0.16, +0.60] | base [0.025, 0.312]; p50c [0.313, 0.722] |
| Limit-hit rate | 92% (23/25) | 36% (9/25) | −0.56 [−0.76, −0.32] | base [0.740, 0.990]; p50c [0.180, 0.575] |
| Mean generated tokens | 30,824 | 21,554 | −9,270 [−13,854, −4,753] | – |

The bootstrap resamples problems 10,000 times with seed 0.

**Direction.** The change has the article's direction on both measures, and both paired intervals exclude zero. Per problem:

| | p50c correct | p50c wrong |
| --- | ---: | ---: |
| **Base correct** | 2 | 1 |
| **Base wrong** | 11 | 11 |

An exact two-sided sign test on the 12 discordant problems (11 against 1) gives p = 0.0063. Each of the article's four values (7.9%, 48.1%, 93.0%, 33.1%; `exp8_math/README.md`) lies inside the matching exact interval above. With 25 problems, the run cannot tell whether the size of the change matches.

**Outcomes**, in the categories of the article's `mathqa.diagnostics`:

| Outcome | Base | p50c |
| --- | ---: | ---: |
| Correct | 3 | 13 |
| Finished with a wrong boxed answer | 0 | 1 |
| Finished without a boxed answer | 0 | 2 |
| Cut at the limit, not correct | 22 | 9 |

- One base answer, `test/algebra/1960.json`, reached the limit and still counts as correct: its last `\boxed{9}` sits inside a thinking block that never closed. The article's `evaluate.py` grades cut answers the same way, calling `Q.extract` on every output whatever its finish reason.
- The base closed its thinking block (`</think>`) in 3 of 25 outputs, and 22 of its 23 cut answers never left thinking. p50c closed it in 19 of 25, and 6 of its 9 cut answers never left thinking.

**Time and tokens.**

| Checkpoint | Generation time | Generated tokens | Chunk times (s) |
| --- | ---: | ---: | --- |
| base | 10,995 s (3.05 h) | 770,598 | 3,214; 3,224; 3,462; 1,094 |
| p50c | 10,912 s (3.03 h) | 538,857 | 3,108; 3,552; 3,249; 1,003 |

- The whole run took 6 h 5 min (21,910 s from the creation of `run.json` to the completion of `summary.json`), against the estimate of 5.5 h (range 4.5–6.5 h).
- All six chunks of 8 ran the full 32,768 steps, as expected. They took 94.9–108.4 ms per step, 100.8 ms on average, instead of the extrapolated 90 ms. Chunks of identical length differed by up to 14% in step time; the cause was not measured.
- The single-row chunks took 33.4 ms per step (base, 32,768 steps) and 33.1 ms (p50c, 30,311 steps), as assumed.

**Identities.** Both checkpoints were evaluated under the same:

- evaluator configuration, `sha256:f3ef721d9cef4326aff898c7cd73128a30579e15baffa4a55da9e3b1c9355cda`;
- prompts, `sha256:f200d94d0234e1432cba00d199d3c11928ea9c0bed5d6d25896a632bd9b69301`;
- profile, `sha256:c4bf7159c15ef0c9de4a5a096f1e81478381023b0095d9a3ec6b439e693ce85b`;
- subset (`subset_25_seed_0`), data hash `sha256:6a6eb9708748677ece795bede12812cb9c62fffa432333d4fb4030ff3c8940df`.

The candidate's `weights_hash`, `sha256:b8dccf8ca463a97d9f6a783651f17b02fecdd740b154bb5ca53983e39852259f`, is the p50c hash of Section 3. `summary.json` lists the 25 item IDs and their positions in `test.jsonl`.

## 7. Cost Estimates for the Protocols of Section 4.4

All figures are for this Mac on MPS at batch size 8, under these assumptions:

- per-row seeds keep results independent of the batch;
- a chunk of 8 runs about 55 min, the mean of the six chunks measured in Section 6 (51.8–59.2 min), because nearly every chunk contains a row that reaches the limit;
- the CPU is not an option for Qwen3.5 (Section 2).

| Protocol | Samples per checkpoint | Time per checkpoint | Base + p50c | Q3 final suite (12 checkpoints: base, p50c, 5 random controls, 5 trials) |
| --- | ---: | ---: | ---: | ---: |
| 25 problems × 1 sample (Section 6) | 25 | 3.0 h (measured) | 6.1 h (measured) | ≈ 36 h |
| 100 problems × 1 sample | 100 | ≈ 11.5 h (13 chunks) | ≈ 23 h | ≈ 6 days |
| Full MATH-500 × 4 samples, 32,768 tokens | 2,000 | ≈ 9.6 days (250 chunks; ≈ 62 M tokens for the base at ≈ 75 tokens/s) | ≈ 19 days | ≈ 115 days |

For reference, the article's re-run of the edited model under the full protocol took about 51 minutes on one H200 with vLLM, for 43.4 million generated tokens. That is about 14,000 tokens/s, roughly 190 times the 74.5 tokens/s that the base's full chunks reached in Section 6.

The Q2 generation passes of policy `qwen_loop_v1` allow at most 32 practice problems at 8,192 new tokens each, so each pass needs at most 4 chunks of 8 × 8,192 steps. Starting from 61 ms per step, it takes about 4 × 8,192 × 70 ms ≈ 38 min. Ten such passes per run would take about 6.4 hours, more than the policy's 6-hour computation limit, which Task 13 may need to reconcile. The 70 ms is not measured, and Section 6 found the full-length extrapolation 12% low.

## 8. Decision Gate (user)

Q3's final evaluation can run in one of two ways:

- **Locally on a subset:** about 36 hours on this Mac for 12 checkpoints × 25 problems × 1 sample, from the times measured in Section 6. About 6 days at 100 problems.
- **On a GPU worker:** a vLLM worker, inserted as its own task before Task 13. Section 10 of the plan defers GPU workers to M3 unless this gate brings one forward.

Locally, the full protocol (Section 7) is out of reach. The choice is the user's and is not made here.

## 9. Open Items From Q1

- **Q1 exit (plan Section 1):**
  - the replay condition holds (Section 3);
  - the comparison condition now holds (Section 6);
  - Section 4 reports the Llama condition as met: import, logits parity, a byte-identical patch round trip, and a completed GSM8K evaluation, with no Llama-specific code.

  Task 7's review confirmed the replay and comparison conditions, and Llama's import and logits parity. Whether the patch-level round trip and the strictly scored GSM8K evaluation satisfy the Llama condition is the plan author's reading (next items).
- **Needs the plan author's decision:** for Llama-3.2-1B-Instruct, `gsm8k_numeric@1` scores 0 of 25 because the model never writes `\boxed{}`. One option is a numeric fallback (the last number) in a new grader version, with the profile unchanged. The other is to keep the strict grader and report the format failure.
- **Not yet supported:** M1's experiment export and `wse replay` do not carry open-weight base models (bundle size limit and format). Patch-level replay works and is byte-identical (Section 4).
- **Measured since Section 2:** decode at batch 8 over full-length generations took 94.9–108.4 ms per step (Section 6). Section 7 uses these times.
- **Not measured:**
  - per-step time at batch sizes between 8 and 32;
  - why step times differed by up to 14% between chunks of identical length (Section 6).
