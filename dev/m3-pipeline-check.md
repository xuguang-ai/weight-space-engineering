# M3 Stage S2: Pipeline Check on Qwen3.5-0.8B

The live check of M3's stage S2 ([plan](./agentic-weight-space-engineering-m3-plan.md), Task 9): one LLM agent works on Qwen3.5-0.8B's reasoning loops with code execution in the sandbox. Its purpose is to confirm that the whole pipeline works and is recorded, not to repair the model. The S2 exit asks that one run exercise and record five functions:

1. analysis in the sandbox;
2. an edit whose parameters come from a sandbox output;
3. a generation pass waited for in a few long waits;
4. selection and freeze;
5. export (and its replay).

A repair is not required. This document was written in two parts: the run sheet before the run, and the results after it.

## 1. Run sheet (written before the run)

**Task.** The existing Qwen loop task of M2, `qwen_loops_gsm8k_train_n32_s0_t8192`, registered by `examples/qwen_loops/loop_setup.py`. Its practice suite holds 32 GSM8K training problems (subset seed 0, `sha256_rank@1`), one sample each, at most 8,192 new tokens, profile `qwen35think@1`, grader `gsm8k_numeric@1`. The agent is told: "The model often keeps reasoning until it reaches the token limit, without giving an answer. Change its weights so that more of its answers to practice problems finish before the limit, without lowering their accuracy." The objective is to lower the development limit-hit rate without lowering development accuracy. The task registers no final suite, so this run has no final test; the preregistered trials of stage S3 bring MATH-500.

**Model.** Qwen3.5-0.8B at revision `2fc06364715b967f1860aea9cf38778875588b17`, imported from the operator's snapshot into a fresh workspace, `~/Desktop/wse-models/workspaces/m3-s2`. The workspace holds no other checkpoint, so no published edit exists in it.

**Policy.** `qwen_loop_sandbox_v1`: `qwen_loop_v1`'s limits (10 generation passes of at most 32 problems and 8,192 new tokens per answer; 25 probe calls of at most 64 texts of up to 1,024 tokens; 50 analyses; 100 tensor reads; 50 previews; 20 commits; 6 h of computation, at most 2 h per operation), plus 100 sandbox calls of at most 300 s of wall-clock time each and 7,200 s in all, 16 KiB of standard output and error each per call, and job waits of up to 600 s. Every run under it is `unverified`.

**Agent.** The OpenAI Agents SDK adapter (`openai_agents@1`, SDK 0.24.0) with `gpt-5.6-luna` on the operator's Azure OpenAI resource, at most 400 turns and 7 hours of wall-clock time, SDK tracing disabled, and `truncation: auto`, so that the provider drops the oldest conversation items rather than fail if the conversation outgrows the model's context. The runner raises the MCP session timeout to 620 s, above the longest permitted call (a 600 s wait). Its instructions are generated from the task and the policy only.

**Hardware.** The worker runs generation and probes on the Mac's MPS device; the sandbox computes on its CPU. M2 measured about 37 minutes per pass of the 32 problems, so the computation budget allows at most nine passes.

**What the agent cannot reach.** Checked before the run:

- The sandbox reads its run's inputs (the base checkpoint's permitted tensors, the run's development outputs, its probe results) and, of the Python environment, the standard library and the files of NumPy, PyTorch, safetensors, and their declared dependencies. While preparing this run we found that the sandbox could read the whole environment directory, including WSE's own distribution metadata, which holds the README, and the README describes p50c and the signed-sum fault. Since then each `site-packages` directory may only be listed (`tests/test_sandbox.py::test_sandbox_reads_only_the_analysis_libraries_of_its_environment`).
- The repository, the operator's model storage (`~/Desktop/wse-models`, which holds the article's logs), the workspace outside the sandbox directory, and the network are denied by the profile; the MCP tools list only this run's checkpoints.
- The agent's instructions name no p50c layer, value, or article (`tests/test_sandbox_records.py::test_instructions_describe_the_sandbox_and_name_no_answer`).

**Expected tokens.** About 150 to 300 model calls. Each call sends the whole conversation, which starts near 15,000 tokens (instructions and tool schemas) and grows with every result, so input should total roughly 10 to 25 million tokens, most of them cached, and output (reasoning included) roughly 0.2 to 0.6 million. The cost follows the deployment's prices; the user waived a cost confirmation for this run.

**What will be checked afterwards.** From the export and the run record: each of the five functions was exercised and recorded; how many `wse_get_job` calls each pass took; the selection and the freeze; that the export replays to identical weights; and a scan of every output for the key, the endpoint, and any word about p50c or the article.

## 2. Results

(Written after the run.)
