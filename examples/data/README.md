# Sample math problems

Math word problems in the format of WSE's `math` evaluator, one JSON object per line with the problem and its final answer. All of them come from [GSM8K](https://github.com/openai/grade-school-math) (Cobbe et al., 2021, [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)).

| File | Problems | GSM8K split | Rows (0-based line indexes, in file order) |
| --- | ---: | --- | --- |
| `math_dev.jsonl` | 16 | training, `train.jsonl` | 0 to 15, the file's first 16 lines |
| `math_test.jsonl` | 16 | test, `test.jsonl` | 0 to 15, the file's first 16 lines |
| `qwen_loops_dev.jsonl` | 32 | training, `train.jsonl` | 27, 85, 383, 456, 751, 1069, 1247, 1378, 1656, 2258, 2335, 2804, 3248, 3367, 3432, 3555, 3611, 3810, 3826, 4450, 4900, 4909, 5060, 5121, 5340, 5428, 6022, 6096, 6408, 6615, 7248, 7424 |

The rows of `qwen_loops_dev.jsonl` are the 32 problems of the recorded Qwen3.5-0.8B diagnosis ([docs/results.md](../../docs/results.md#3-reconstructing-p50c-in-four-steps)), which WSE's subset rule `sha256_rank@1` chose with seed 0 over the IDs `gsm8k_<line index>` (`wse.evaluators.generation_runs.select_subset`). They keep those IDs, so that results can be matched with the recorded ones problem by problem; the other two files have no IDs.

Each row's `problem` is GSM8K's `question`, and its `answer` is the number after the `####` that ends GSM8K's worked answer, with thousands separators removed (none of the chosen answers had one).

**Checks.**

- All three files parse with `wse.evaluators.text_suites.read_jsonl(path, format="math")`, and `detect_format` reads them as `math`.
- The test file shares no problem with either development file, and the two development files share none, compared as WSE compares problems (Unicode NFKC, lowercase, whitespace collapsed).
- Graded by the `math` evaluator's grader (`math_boxed@1`) against these answers, the 64 recorded outputs of that diagnosis's comparison keep the grades that GSM8K's numeric grader gave them: 2 of 32 correct for the base model and 10 of 32 for the edited model.

**License.** GSM8K is released under the [MIT License](https://github.com/openai/grade-school-math/blob/master/LICENSE). These rows keep its problems unchanged and shorten its answers to the final number.

## Rebuilding the files

The copies of GSM8K used had these SHA-256 hashes; the training file is also the one that the recorded runs read:

```text
17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465  train.jsonl
3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14  test.jsonl
```

Download both from the repository's commit `3101c7d5072418e28b9008a6636bde82a006892c`, and compare their hashes with the list above:

```bash
mkdir -p ~/wse-data/gsm8k
for split in train test; do
  curl -L -o ~/wse-data/gsm8k/$split.jsonl \
    https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/$split.jsonl
done
shasum -a 256 ~/wse-data/gsm8k/*.jsonl
```

Then, in this directory, `uv run python rebuild.py` with this script as `rebuild.py` writes the three files again, byte for byte:

```python
import json
from pathlib import Path

from wse.evaluators.generation_runs import select_subset

source = Path("~/wse-data/gsm8k").expanduser()


def rows(split):
    return [json.loads(line) for line in (source / f"{split}.jsonl").open(encoding="utf-8")]


def math_row(row, **extra):
    answer = row["answer"].rsplit("####", 1)[1].strip().replace(",", "")
    return {**extra, "problem": row["question"], "answer": answer}


def write(name, data):
    Path(name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data), encoding="utf-8")


train, test = rows("train"), rows("test")
write("math_dev.jsonl", [math_row(r) for r in train[:16]])
write("math_test.jsonl", [math_row(r) for r in test[:16]])
chosen = select_subset([f"gsm8k_{i}" for i in range(len(train))], 32, 0)
write("qwen_loops_dev.jsonl", [math_row(train[i], id=f"gsm8k_{i}") for i in chosen])
```

```text
59ad8acd621284fb908c563f0296845a991470f558a9aabef39398ed959da193  math_dev.jsonl
d2164a7d9035915bfa3dbcb383ac5d1b564a6344a01960f5e1875601b8915a50  math_test.jsonl
48f8837df5ecf886c626953d2bba48baa20f8d4467daf9cac6e7a9959337bb6d  qwen_loops_dev.jsonl
```
