"""Graders of generated answers (M2 plan Section 4.3).

* ``math_boxed@1`` grades the last ``\\boxed{}`` answer with math-verify, as
  the article does (``exp8_math/code/mathqa.py``): a thinking block is cut
  off at its last ``</think>``; the answer is the content of the last
  ``\\boxed`` (else the last ``\\fbox``), brace-matched, or, for
  ``\\boxed 5`` without braces, the text up to the next ``$`` or line end; an
  empty or unclosed box is no answer. The reference and the answer are each
  wrapped in ``\\boxed{}``, parsed with ``math_verify.parse``, and compared
  with ``math_verify.verify(gold, answer)``.
* ``gsm8k_numeric@1`` compares the number in the last ``\\boxed{}`` answer
  with the number after the reference's last ``####`` (GSM8K's format).
  Thousands separators, dollar signs, a trailing ``.``, and ``\\text{...}``
  units are ignored; the answer must hold exactly one number, and numbers
  compare exactly as decimals.

A grader is a registered extension of kind ``grader`` with ``grader_id``,
``version``, and ``grade(output, reference) -> Grade``. ``math_verify`` is
imported lazily, only when ``math_boxed@1`` grades.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol, runtime_checkable

from wse.schemas import ErrorCode, WSEError

ANSWER_EXCERPT_CHARS = 200


@dataclass(frozen=True)
class Grade:
    """Whether an output is correct, and the answer the grader read (``None`` when there was none)."""

    correct: bool
    answer: str | None


@runtime_checkable
class Grader(Protocol):
    grader_id: str
    version: str

    def grade(self, output: str, reference: str) -> Grade: ...


def validate_grader(extension_id: str, version: str, implementation: Any) -> None:
    declared = (getattr(implementation, "grader_id", None), getattr(implementation, "version", None))
    if declared != (extension_id, version):
        raise WSEError(ErrorCode.INVALID_ARGUMENT,
                       f"grader declares {declared[0]}@{declared[1]} but is being registered as {extension_id}@{version}")
    if not callable(getattr(implementation, "grade", None)):
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"grader {extension_id}@{version} does not implement grade")


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def answer_text(output: str) -> str:
    """The part of an output that holds the answer: after the last ``</think>``, if there is one."""
    return output.rsplit("</think>", 1)[-1].strip() if "</think>" in output else output


def _box_content(text: str, start: int) -> str | None:
    """The content of the box whose command starts at ``text[start]``, or ``None``."""
    command = "\\boxed" if text.startswith("\\boxed", start) else "\\fbox"
    rest = text[start + len(command):]
    body = rest.lstrip()
    if body.startswith("{"):
        depth = 0
        for position, char in enumerate(body):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return body[1:position].strip() or None
        return None  # unclosed, for example cut off by the token limit
    if command == "\\boxed" and rest.startswith(" "):  # '\boxed 5': up to the next '$' or line end
        return rest.strip().split("$", 1)[0].split("\n", 1)[0].strip() or None
    return None


def last_boxed(text: str) -> str | None:
    """The content of the last ``\\boxed`` of ``text``, else of the last ``\\fbox``; ``None`` if none or empty."""
    start = text.rfind("\\boxed")
    if start < 0:
        start = text.rfind("\\fbox")
        if start < 0:
            return None
    return _box_content(text, start)


def _excerpt(answer: str | None) -> str | None:
    if answer is None or len(answer) <= ANSWER_EXCERPT_CHARS:
        return answer
    return answer[:ANSWER_EXCERPT_CHARS] + "..."


# ---------------------------------------------------------------------------
# math_boxed@1
# ---------------------------------------------------------------------------


def _math_verify() -> Any:
    try:
        import math_verify
    except ImportError as exc:
        raise WSEError(ErrorCode.UNSUPPORTED_CAPABILITY, "math_boxed@1 needs math-verify, part of the models-hf extra; "
                       "install it with 'uv sync --extra models-hf'") from exc
    return math_verify


def _timeout() -> int | None:
    # math-verify's timeouts use signal.alarm, which works only in the main thread; elsewhere it would raise,
    # and the library would report "not equal". Off the main thread the checks run without its timeouts.
    return 5 if threading.current_thread() is threading.main_thread() else None


class MathBoxedGrader:
    """``math_boxed@1``: math-verify equivalence of the last boxed answer and the reference answer."""

    grader_id = "math_boxed"
    version = "1"

    def grade(self, output: str, reference: str) -> Grade:
        answer = last_boxed(answer_text(output))
        if answer is None:
            return Grade(correct=False, answer=None)
        math_verify = _math_verify()
        timeout = _timeout()
        try:
            gold = math_verify.parse("\\boxed{" + reference + "}", parsing_timeout=timeout)
            target = math_verify.parse("\\boxed{" + answer + "}", parsing_timeout=timeout)
            correct = bool(gold) and bool(target) and bool(math_verify.verify(gold, target, timeout_seconds=timeout))
        except Exception:  # noqa: BLE001 - an unparseable answer is wrong, never an evaluation failure
            correct = False
        return Grade(correct=correct, answer=_excerpt(answer))


# ---------------------------------------------------------------------------
# gsm8k_numeric@1
# ---------------------------------------------------------------------------

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _clean_number_text(text: str) -> str:
    text = re.sub(r"\\text\{[^{}]*\}", " ", text)  # units such as \text{ marbles}
    for token in ("\\$", "$", "\\%", "%", "\\!", "\\,"):
        text = text.replace(token, " ")
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # thousands separators


def gsm8k_reference(answer: str) -> str:
    """The reference number of a GSM8K answer: the text after its last ``####``, without commas."""
    if "####" not in answer:
        raise ValueError("a GSM8K reference answer ends with '#### <number>'")
    return answer.rsplit("####", 1)[-1].strip().replace(",", "")


def _decimal(text: str) -> Decimal | None:
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


class GSM8KNumericGrader:
    """``gsm8k_numeric@1``: the number in the last boxed answer equals the reference after ``####``."""

    grader_id = "gsm8k_numeric"
    version = "1"

    def grade(self, output: str, reference: str) -> Grade:
        try:
            expected = _decimal(gsm8k_reference(reference))
        except ValueError as exc:
            raise WSEError(ErrorCode.INVALID_ARGUMENT, f"gsm8k_numeric@1: {exc}") from exc
        answer = last_boxed(answer_text(output))
        if answer is None:
            return Grade(correct=False, answer=None)
        numbers = _NUMBER.findall(_clean_number_text(answer))
        value = _decimal(numbers[0]) if len(numbers) == 1 else None
        return Grade(correct=expected is not None and value is not None and value == expected,
                     answer=_excerpt(answer))


BUILTIN_GRADERS = (MathBoxedGrader(), GSM8KNumericGrader())

__all__ = ["BUILTIN_GRADERS", "GSM8KNumericGrader", "Grade", "Grader", "MathBoxedGrader", "answer_text",
           "gsm8k_reference", "last_boxed", "validate_grader"]
