"""Text suites from JSON Lines files: reading, hashes, and overlap checks (M2 plan Section 4.3).

Formats (one JSON object per line):

* ``math500``: ``HuggingFaceH4/MATH-500``'s ``test.jsonl``. The item ID is
  ``unique_id``, the problem ``problem``, and the reference ``answer``.
* ``gsm8k``: ``openai/grade-school-math``'s ``train.jsonl`` or ``test.jsonl``.
  The item ID is ``id_prefix`` plus the line's index, the problem
  ``question``, and the reference the full ``answer`` (its grader reads the
  number after ``####``).
* ``math``: MATH problems the operator supplies: ``problem`` and ``answer``,
  or ``solution`` whose last ``\\boxed{}`` is the reference; the item ID is
  ``unique_id``, ``id``, or ``id_prefix`` plus the line's index.

Problems compare by their normalized text: Unicode NFKC, lowercased, and
whitespace runs collapsed to one space. Suite registration
(``wse.jobs.JobService.register_suite``) rejects a development suite that
shares a normalized problem with any final suite, and a final suite that
shares one with any development suite.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from wse.schemas import ErrorCode, TextDatasetManifest, TextItem, TextSuiteData, WSEError

FORMATS = ("math500", "gsm8k", "math")
MAX_FILE_BYTES = 256 * 1024 * 1024


def normalize_problem(text: str) -> str:
    """NFKC, lowercase, and single spaces: the form in which problems are compared."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def _invalid(message: str, **details: Any) -> WSEError:
    return WSEError(ErrorCode.INVALID_ARGUMENT, message, details=details)


def _string(row: dict[str, Any], key: str, line: int) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise _invalid(f"line {line}: '{key}' must be a string", line=line)
    return value


def _item(row: Any, line: int, index: int, format: str, id_prefix: str | None) -> TextItem:
    from wse.evaluators.graders import last_boxed

    if not isinstance(row, dict):
        raise _invalid(f"line {line} is not a JSON object", line=line)
    if format == "math500":
        item_id, problem, reference = _string(row, "unique_id", line), _string(row, "problem", line), \
            _string(row, "answer", line)
    elif format == "gsm8k":
        item_id = f"{id_prefix or 'gsm8k_'}{index}"
        problem, reference = _string(row, "question", line), _string(row, "answer", line)
        if "####" not in reference:
            raise _invalid(f"line {line}: a GSM8K answer ends with '#### <number>'", line=line)
    else:
        identifier = row.get("unique_id", row.get("id"))
        item_id = str(identifier) if identifier is not None else f"{id_prefix or 'math_'}{index}"
        problem = _string(row, "problem", line)
        if isinstance(row.get("answer"), str):
            reference = row["answer"]
        else:
            reference = last_boxed(_string(row, "solution", line)) or ""
            if not reference:
                raise _invalid(f"line {line}: the solution has no \\boxed{{}} answer", line=line)
    try:
        return TextItem(item_id=item_id, problem=problem, reference=reference)
    except ValidationError as exc:
        raise _invalid(f"line {line}: {exc.errors()[0]['msg']}", line=line) from exc


def read_jsonl(path: str | Path, *, format: str, id_prefix: str | None = None
               ) -> tuple[TextSuiteData, TextDatasetManifest]:
    """Read a JSON Lines file in ``format``; return its problems and their manifest (with the file's SHA-256)."""
    if format not in FORMATS:
        raise _invalid(f"unknown text suite format {format!r}; supported: {list(FORMATS)}")
    if id_prefix is not None and not re.fullmatch(r"[A-Za-z0-9_.:\-/]{1,64}", id_prefix):
        raise _invalid("id_prefix must be 1-64 letters, digits, or '_.:-/'")
    file = Path(path)
    try:
        if file.stat().st_size > MAX_FILE_BYTES:
            raise _invalid(f"{file.name} is larger than {MAX_FILE_BYTES} bytes")
        data = file.read_bytes()
    except FileNotFoundError as exc:
        raise WSEError(ErrorCode.NOT_FOUND, f"text suite file {file} does not exist") from exc
    items = []
    lines = data.decode("utf-8").splitlines()
    for number, text in enumerate(lines, start=1):
        if not text.strip():
            continue
        try:
            row = json.loads(text)
        except ValueError as exc:
            raise _invalid(f"line {number} is not JSON: {exc}", line=number) from exc
        items.append(_item(row, number, len(items), format, id_prefix))
    if not items:
        raise _invalid(f"{file.name} holds no problems")
    ids = [item.item_id for item in items]
    duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1}) if len(set(ids)) != len(ids) else []
    if duplicates:
        raise _invalid(f"{file.name} repeats item IDs: {duplicates[:20]}", duplicate_items=duplicates[:20])
    suite_data = TextSuiteData(items=items)
    manifest = TextDatasetManifest.from_data(suite_data, source_format=format,
                                             source_sha256="sha256:" + hashlib.sha256(data).hexdigest())
    return suite_data, manifest


def detect_format(path: str | Path) -> str:
    """The format of a math JSON Lines file, read from its first row (the built-in ``math`` evaluator).

    ``problem`` with ``answer`` or ``solution`` is ``math``; ``question`` with
    an ``answer`` that ends in ``#### <number>`` is ``gsm8k``.
    """
    try:
        with open(Path(path).expanduser(), encoding="utf-8") as handle:
            line = next((line for line in handle if line.strip()), None)
    except OSError as exc:
        raise WSEError(ErrorCode.NOT_FOUND, f"cannot read {path}: {exc}", details={"path": str(path)}) from exc
    if line is None:
        raise _invalid(f"{path} holds no rows", path=str(path))
    try:
        row = json.loads(line)
    except ValueError as exc:
        raise _invalid(f"{path}: the first row is not JSON: {exc}", path=str(path)) from exc
    if isinstance(row, dict) and isinstance(row.get("problem"), str):
        return "math"
    if isinstance(row, dict) and isinstance(row.get("question"), str) and "####" in str(row.get("answer", "")):
        return "gsm8k"
    keys = sorted(row) if isinstance(row, dict) else type(row).__name__
    raise _invalid(f"{path}: a math row holds 'problem' with 'answer' (or 'solution' with a final \\boxed{{}}), or "
                   f"GSM8K's 'question' with an 'answer' ending in '#### <number>'; the first row holds {keys}",
                   path=str(path))


def overlapping_items(data: TextSuiteData, other: TextSuiteData) -> list[str]:
    """IDs of the items of ``data`` whose normalized problem also occurs in ``other``, in order."""
    problems = {normalize_problem(item.problem) for item in other.items}
    return [item.item_id for item in data.items if normalize_problem(item.problem) in problems]


def exclude_overlap(data: TextSuiteData, other: TextSuiteData) -> TextSuiteData:
    """``data`` without the items whose normalized problem occurs in ``other`` (for operator-supplied practice sets)."""
    excluded = set(overlapping_items(data, other))
    kept = [item for item in data.items if item.item_id not in excluded]
    if not kept:
        raise _invalid("every item overlaps the other suite")
    return TextSuiteData(items=kept)


__all__ = ["FORMATS", "exclude_overlap", "normalize_problem", "overlapping_items", "read_jsonl"]
