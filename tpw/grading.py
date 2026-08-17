# tpw/grading.py
import re

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_ANSWER = re.compile(r"ANSWER:\s*(-?[\d,]+(?:\.\d+)?)", re.IGNORECASE)
_FALLBACK = re.compile(r"(-?[\d,]+(?:\.\d+)?)")

PROMPT_DIRECT = (
    "{question}\n\n"
    "Give only the final numeric answer, with no explanation.\n"
    "End your response with: ANSWER: <number>"
)

PROMPT_COT = (
    "{question}\n\n"
    "Think step by step, then give the final numeric answer.\n"
    "End your response with: ANSWER: <number>"
)


def split_thinking(raw: str) -> tuple[str, str]:
    """Separate reasoning-model <think> blocks from the visible answer."""
    thinking = " ".join(_THINK.findall(raw))
    return thinking, _THINK.sub("", raw)


def parse_answer(raw: str) -> str | None:
    """Extract the numeric answer. Prefers the declared ANSWER: marker;
    falls back to the last number so that format non-compliance is not
    scored as an incorrect answer."""
    _, visible = split_thinking(raw)
    match = _ANSWER.search(visible)
    if match is None:
        candidates = _FALLBACK.findall(visible)
        if not candidates:
            return None
        value = candidates[-1]
    else:
        value = match.group(1)
    return value.replace(",", "").rstrip(".")


def followed_format(raw: str) -> bool:
    """Whether the model honoured the requested output format. Tracked
    separately: format compliance is a result, not measurement noise."""
    _, visible = split_thinking(raw)
    return _ANSWER.search(visible) is not None


def is_correct(parsed: str | None, expected: str) -> bool:
    if parsed is None:
        return False
    try:
        return abs(float(parsed) - float(expected)) < 1e-6
    except ValueError:
        return parsed.strip() == expected.strip()