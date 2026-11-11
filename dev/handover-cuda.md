# Handover: CUDA Support

For a Claude Code session on a Linux server with an NVIDIA GPU, starting from a fresh clone of this repository. Written on 2026-11-11, when WSE 0.1.0 ran only on Apple's MPS and the CPU.

## 1. Goal

Make generation, probes, and evaluation run on CUDA with the guarantees WSE already gives on MPS:

- logits equal to `transformers`' own forward on the same device, with the stored dtypes kept;
- a left-padded row equal to the same prompt run alone;
- probes equal to the model's hidden states;
- token-identical generation when a pass is repeated on the same machine;
- run records that identify the device and the settings.

Edits must stay exactly as they are: they are computed on the CPU in a platform-independent way, so a patch made on the GPU server must hash the same as one made on a Mac.

**Out of scope** unless the maintainer asks: multiple GPUs, quantization, `torch.compile`, new MCP tools, and changes to edit arithmetic.

## 2. Read first

| Document | What you need from it |
| --- | --- |
| `README.md` | What WSE does, the Quick Start, and the four-step example (`examples/p50c_four_steps.py`) |
| `docs/results.md` | The MPS measurements to compare against: logits, reproducibility, and pass times |
| `docs/models.md` | Support levels, the verified models, and Known limits (devices, reproducibility) |
| `dev/agentic-weight-space-engineering-m2-plan.md` | The M2 plan: Section 2.1 on stored dtypes, Section 4.1 on the inference engine and its device rules |
| `dev/agentic-weight-space-engineering-plan.md` | The M1 plan: Section 7.3 is the reproducibility contract |
| `dev/m2-q1-report.md`, `dev/m2-q2-report.md` | How the MPS numbers were measured, and the full model revisions (Q1 report Section 1, Q2 report Section 5) |

Code comments cite "M1 plan Section x" and "M2 plan Section x"; those are the two plans in `dev/`.

## 3. Where devices are handled today

**Device selection and the engine:**

| Place | What it does now | What CUDA needs |
| --- | --- | --- |
| `src/wse/engines/hf_torch.py:68` | `DEVICES = ("mps", "cpu")` | Add `cuda` |
| `src/wse/engines/hf_torch.py:79–92` | `mps_available()`; `select_device()` picks `mps` when available, otherwise `cpu`, and rejects any other name | A CUDA check, and the automatic order (Section 5.1) |
| `src/wse/engines/hf_torch.py`, module docstring, **Devices** and **Positions** | States the device rule, and that a left-padded row computes the logits of the prompt run alone | Update the device rule; keep the padded-row guarantee (Section 5.2) |
| `src/wse/engines/hf_torch.py`, `build_model` | Builds the module on the meta device, creates non-persistent buffers on the target device, copies each distinct stored tensor to it once, and attaches the tensors with `load_state_dict(assign=True)` | Should work as is; verify memory use and dtypes on the GPU |
| `src/wse/engines/hf_torch.py:383–401, 455` | Every decoding step copies the last position's logits to the CPU as float32, for greedy decoding too. The logits processors (the presence penalty), temperature, top-k, and top-p (a sort over the 248,320-token vocabulary) run there, and each row draws from its own seeded CPU generator, so a row's tokens depend on its seed, not its batch position | Keep these semantics. This per-step CPU work may dominate on a fast GPU; profile before changing it |
| `src/wse/engines/hf_torch.py:257–263` | `info()` returns exactly the device type, dtypes, model class, attention implementation, and library versions | Add the CUDA identity under new keys, on CUDA only (Section 5.1) |
| `src/wse/api.py:178–189` | `ModelHandle.runtime()` keys its runtimes by device type; `ModelHandle.forward` uses the automatic device | Follows the automatic order |
| `src/wse/highlevel.py:705, 743, 805–815` | `WSE.probe(device=)`, and `_text_model`, which caches one text model per device type | Passes the device through |
| `src/wse/evaluators/generation.py:177` | `load_text_model(handle, files, device=)` | Passes the device through |
| `src/wse/evaluation.py:489–498` | `_final_result` takes no device: final evaluations run on the automatic device | Follows the automatic order |
| `src/wse/probes.py` (docstring lines 58–77; lines 502–539) | Window sums run on the device in float32, then move to the CPU as float64, because MPS has no float64 | Device-generic; verify on CUDA |
| `src/wse/worker.py:320–328` | `_release_device_memory` calls `torch.mps.empty_cache()` for MPS | Also call `torch.cuda.empty_cache()` for CUDA |
| `src/wse/worker.py:482, 1039`; `src/wse/cli.py:150` | `wse worker --device` and `wse compare-generation --device` accept `mps` and `cpu` | Add `cuda` |

**Records, hashes, and resumption.** Three contracts constrain what a CUDA record may add:

| Place | Contract | Consequence |
| --- | --- | --- |
| `src/wse/schemas.py:1526–1538`; `src/wse/evaluators/generation.py:609–614` | `ExecutionConfig.device: Literal["cpu", "mps"]` is copied from `info()["device"]`, and `ExecutionConfig.config_hash` is a report's `evaluator_config_hash`. Sealed final reports are re-verified against their hashes on read (`src/wse/evaluation.py:407–409`) | Add `"cuda"` to the Literal; existing hashes stay unchanged. `info()["device"]` must stay the bare type, so a value like `cuda:0` would fail validation after a whole pass. Do not add fields to `ExecutionConfig`: a plain new field changes every hash and makes old sealed reports read as corrupt. An unavoidable new field must be declared with `_optional()` (`src/wse/schemas.py:250–256`) |
| `src/wse/evaluators/generation_runs.py:213–232, 307, 320–335` | `run.json`'s `configuration` is compared key by key when a run resumes | A new key with a value for existing devices makes every existing run directory non-resumable. Add CUDA keys for CUDA runs only. A driver version in a compared key would also refuse to resume after a driver update; that choice is the maintainer's |
| `tests/test_hf_engine.py:294–303` | Asserts `info()` as an exact dict on the CPU | Keep CPU's `info()` unchanged |

**Tests and help texts:**

| Place | What it does now | What CUDA needs |
| --- | --- | --- |
| `tests/test_hf_engine.py:40, 165–175, 289–291` | `MPS_TOLERANCES`; an MPS-only test of logits against the CPU; assertions that `select_device("cuda")` is rejected and that the automatic choice is `mps` or `cpu` | A CUDA-only test, skipped without CUDA, with tolerances set from measurements; update the assertions |
| `tests/test_hf_engine.py:215–242` | A left-padded row equals the row alone, bitwise for BF16 and F16 on the CPU (`PADDED_ROW_TOLERANCES`) | The same check on CUDA (Section 5.2) |
| `tests/test_llm_policy_mcp.py:921` | Asserts that a final report's device is `mps` or `cpu` | Update when the automatic order changes |
| Docstrings and help texts that state the automatic device, for example `src/wse/worker.py:494, 1040`, `src/wse/cli.py:151`, `src/wse/api.py:162`, and `src/wse/engines/hf_torch.py:13, 84` (search with `grep -rn mps src/wse`) | Say "mps when available" | Update together with the automatic order |
| `examples/p50c_four_steps.py:62, 70, 89` | Passes no device, so it runs on the automatic device | Add a `--device` option, or rely on the new automatic order |
| `README.md`; `docs/models.md`, Known limits | State that WSE runs on MPS or the CPU only | Update once CUDA is measured |

## 4. Setting up the server

1. **Check the hardware and driver.**
   - Record `nvidia-smi`: the GPU model, its memory, the driver, and the highest CUDA version the driver supports.
   - `uv.lock` resolves `torch` 2.15.0 with CUDA 13 libraries on Linux (`cuda-toolkit` 13.2, `nvidia-cudnn-cu13`, `triton`), so the driver must support CUDA 13.
   - If it does not, stop and ask the maintainer before you change where `torch` comes from. Never commit machine-specific index settings.
2. **Install.**
   - Install [uv](https://docs.astral.sh/uv/). The repository pins Python 3.14 in `.python-version`.
   - Run `uv sync --extra models-hf --extra agents-openai`.
   - Then run `uv run --no-sync python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())"`. The GPU's compute capability must be among the architectures the build targets: CUDA 13 builds no longer target GPUs older than Turing.
3. **Run the test suite on the CPU first.**
   - Command: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENAI_AGENTS_DISABLE_TRACING=1 uv run --no-sync pytest -q -p no:cacheprovider`.
   - On macOS it passed 819 tests in about four minutes. On Linux without MPS, expect 813 passed and 6 skipped: the 6 MPS-only cases skip.
   - Clone with full history: a shallow clone skips 3 more tests, which check migrations against older commits.
   - Record any Linux failure before you change code: a failure on Linux that has nothing to do with the GPU is not a CUDA bug.
4. **Get the models and the data.**
   - Keep them outside the repository. The README's commands fetch the p50c edit, MATH-500, and GSM8K. `WSE.from_pretrained` downloads a model and needs the network even with a commit hash, because it resolves the revision through the Hub API.
   - Llama-3.2-1B-Instruct and gemma-3-1b-it are gated (`docs/models.md`). Downloading them needs the maintainer's own Hugging Face login on the server, `hf auth login`; ask them to run it. Never store a token in the repository or in a file.
   - The full revision hashes of the four verified models are in the Q1 report, Section 1, and the Q2 report, Section 5; `docs/models.md` shows 12-character prefixes.

## 5. Work plan

### 5.1 Device selection and records

- **Accept `cuda`**, everywhere a device is named (Section 3), with an error that names the available devices when CUDA is not available.
- **Accept only `cuda`, not `cuda:N`.** Choose the GPU with `CUDA_VISIBLE_DEVICES`. An index would be dropped at several places: `src/wse/api.py:180–183`, `src/wse/highlevel.py:743`, `src/wse/worker.py:482`, `src/wse/evaluators/generation_runs.py:307`, and `info()`, which reports the type.
- **Settle the automatic order with the maintainer before Section 5.6.** The proposal is `cuda`, then `mps`, then `cpu`. Until it changes, pass `device="cuda"` explicitly wherever an entry point takes a device. Final evaluations (`_final_result`) and `ModelHandle.forward` take no device and follow the automatic order.
- **Record the CUDA identity on CUDA only**, under new keys of `info()`. The runtime record is not part of `evaluator_config_hash`, but `info()["device"]` is, so it stays `"cuda"`. In `run.json`, add the identity for CUDA runs only (Section 3, records). Record:
  - the GPU name and compute capability;
  - the CUDA runtime, cuDNN, and driver versions;
  - whether deterministic algorithms are on, `torch.backends.cudnn.deterministic`, and `torch.backends.cudnn.benchmark`;
  - `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction` and `allow_fp16_reduced_precision_reduction` (both default to True). These matter for BF16 models; TF32 affects only float32 matrix products and convolutions;
  - `torch.backends.cuda.matmul.fp32_precision` and `torch.backends.cudnn.fp32_precision`;
  - which scaled-dot-product-attention backends are enabled: `torch.backends.cuda.flash_sdp_enabled()`, `mem_efficient_sdp_enabled()`, `math_sdp_enabled()`, and `cudnn_sdp_enabled()`;
  - `torch.backends.cuda.preferred_blas_library()`;
  - which optional kernels `transformers` could use (Section 5.4).

  On CUDA, two runs are comparable only when these agree.

### 5.2 Correctness

- **Logits against `transformers`.** Compare WSE's CUDA runtime with `from_pretrained` on the same GPU, with every parameter kept in its stored dtype, as `_reference` in `tests/test_hf_engine.py:69–93` builds it.
  - Do this for the tiny test models and the four verified models: Qwen3.5-0.8B, Qwen3-0.6B, gemma-3-1b-it, and Llama-3.2-1B-Instruct.
  - Do not use `from_pretrained(dtype="auto")` alone. It casts Qwen3.5-0.8B's 36 F32 tensors to BF16, which moves logits by up to 0.53 (M2 plan, Section 2.1).
  - The expectation is a maximum absolute difference of 0.0, as on the CPU and MPS.
- **Logits against the CPU.** Measure the largest differences, then set the CUDA test's tolerances from those measurements, as `MPS_TOLERANCES` does for MPS.
- **Padded rows.** A left-padded row must give the same logits as its prompt run alone (Section 3; bitwise for BF16 and F16 on the CPU).
  - On CUDA, scaled-dot-product attention chooses among flash, memory-efficient, cuDNN, and math kernels by mask, dtype, and shape, so a masked padded batch and an unmasked single row may take different kernels. `attn_implementation: "sdpa"` does not say which one ran.
  - Measure the check on CUDA. For the determinism runs, pin the kernel with `torch.nn.attention.sdpa_kernel`.
- **Probes.** Residual sites must equal `output_hidden_states` exactly on CUDA, as on MPS.
- **Edits**, checked in this order:
  1. **Base.** The imported base: weights hash `sha256:cb5a704b1fd2b2649fab2005f630c88eb4a7dd60b80b08183ca7e592ce6b53a5`.
  2. **p50c.** Its import is pure replay, so it is the strict check: weights hash `sha256:b8dccf8ca463a97d9f6a783651f17b02fecdd740b154bb5ca53983e39852259f`.
  3. **The four-step example's edit:** `sha256:44918f4ac03d71cc87ae59a13af13570cb17b834f1466e4e70cd10dc0c568b95` on the Mac. This hash also depends on the example's `np.linalg.svd`, which runs on the host's LAPACK, and `project_out` takes the direction as float32 values (`src/wse/operators/directions.py:22–28`). If the hash differs, compare the float32 directions before you suspect WSE.

### 5.3 Reproducibility

- **Repeated passes.** Run the same generation pass twice, in separate processes, and compare the per-answer `output_sha256` in the records, as `docs/results.md` (Section 3, Reproducibility) did on MPS.
- **Settings.** Do this with the default settings, then with deterministic settings:
  - `torch.use_deterministic_algorithms(True)`;
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
  - a pinned attention kernel (Section 5.2).

  Report which settings give token-identical passes and what they cost in speed. If an operation has no deterministic implementation, PyTorch raises an error; record which operation.
- **Where settings apply.**
  - `CUBLAS_WORKSPACE_CONFIG` must be in the environment before CUDA initializes.
  - In-process scripts, such as the four-step example and `WSE.compare`, set the flags themselves.
  - `wse compare-generation` is its own process.
  - Managed passes and probes run in the worker subprocess (`python -m wse.worker`), where flags set in the parent have no effect. Pass settings through `wse serve --worker-args` (`src/wse/cli.py:77`) or a `--setup module:function` hook (`src/wse/worker.py:1028`).
- **Batch size.** Numerics may depend on the batch size; measure and record the effect.
- **Across devices.** Token identity between CUDA and MPS is not expected. Compare aggregates instead.

### 5.4 Kernels for Qwen3.5

Qwen3.5 mixes Gated DeltaNet layers with full attention. On the Mac, `transformers` logged that `causal_conv1d` and `flash-linear-attention` were not installed. It fell back to reference PyTorch code for `causal_conv1d_fn`, `causal_conv1d_update`, `chunk_gated_delta_rule`, and `fused_recurrent_gated_delta_rule`.

On CUDA these packages provide fast kernels, which may change numerics and reproducibility. Measure both paths: speed, the logits difference from the reference path, and run-to-run token identity. Do not add either package as a dependency without the maintainer's decision. Whichever path runs must be recorded (Section 5.1).

### 5.5 Speed

- **Generation passes.** Measure a pass of 32 problems at 8,192 tokens at batch size 8, and at larger batches. On the M4 Pro (MPS) the same pass takes 37–40 minutes, about 113 generated tokens per second.
- **Where the time goes.** Profile one decoding step before you optimize, including the per-step CPU work (Section 3).
- **GPU memory.** The worker's `peak_memory_bytes` is the process's resident memory only (`src/wse/worker.py:115–118`). Record `torch.cuda.max_memory_allocated()` and `max_memory_reserved()` as well, to compare with the Q1 report's "Peak MPS" column.

### 5.6 End-to-end check

- **The four-step example.** Run `examples/p50c_four_steps.py` on CUDA and record its output and time. Pass the device explicitly, unless the new automatic order is approved (Section 5.1). Expect the edit hashes of Section 5.2 and aggregates near MPS's, which need not be identical.
- **Long runs.** The four-step example took 1 h 54 min on MPS, and the larger-batch runs of Section 5.5 may also take long. Time one pass first, estimate the whole run, and ask the maintainer before any run longer than about an hour, such as the 25-problem MATH-500 comparison at 32,768 tokens.

## 6. Definition of done

- The test suite passes on the server's CPU. New CUDA tests pass on the GPU and skip without one.
- Logits, with the stored dtypes kept, are bitwise equal to `transformers` on CUDA for the four verified models.
- Padded rows and probes hold on CUDA, or the report says where and by how much they differ.
- A repeated pass is token-identical on CUDA under documented settings, or the report says why not.
- Run records identify the GPU and the settings, without changing existing hashes or breaking the resumption of existing runs.
- Documentation:
  - `README.md` and `docs/models.md` are updated;
  - `docs/results.md` has a CUDA section with measured numbers;
  - a short report, `dev/cuda-report.md`, gives the measurements, what holds, what does not, and the open decisions.

## 7. Working rules

- **Language.** Talk with the maintainer in Simplified Chinese. Write code, comments, commit messages, and documents in English.
- **Evidence.**
  - Measure before you claim, and compare a change against an unchanged control run.
  - Report in three groups: what holds; what does not hold, and why; what needs the maintainer's decision.
- **Ask first** before you:
  - change a default that affects existing records, such as the automatic device or anything hashed;
  - add a dependency;
  - start a GPU run longer than about an hour;
  - push, tag, or open a pull request.
- **Git.**
  - Work on a branch, for example `cuda`, and commit in small steps, ending each message with the attribution line your harness gives.
  - Before the first commit, compare `git config user.name` and `user.email` with `git log -1 --format='%an <%ae>'`, the maintainer's identity. If they differ, ask.
  - When a pull request is merged, use `gh pr merge --delete-branch`.
- **Files.**
  - Keep models, workspaces, datasets, and run outputs outside the repository. `.gitignore` covers `workspace/`, `*.safetensors`, `*.sqlite`, and `.env`.
  - Never commit credentials or absolute local paths.
  - Treat downloaded files as untrusted data.
- **uv.** Run commands with `uv run --no-sync`: `uv sync` without the `--extra` flags removes the extras.
