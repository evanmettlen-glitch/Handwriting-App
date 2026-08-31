"""A curated enrollment set: enough handwriting to adapt the model, in < 5 min.

The order builds momentum (rote alphabet first), then covers every lowercase and
uppercase letter twice, all ten digits, common punctuation and contractions, and
a handful of natural sentences. ~40 short prompts, roughly 4 minutes of writing.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Dict, Iterable, List

ENROLLMENT_PROMPTS: List[str] = [
    # rote alphabet — fast, full a-z / A-Z coverage
    "abcdefghijklm",
    "nopqrstuvwxyz",
    "ABCDEFGHIJKLM",
    "NOPQRSTUVWXYZ",
    # warm-up common words
    "the", "and", "you", "was", "for", "with",
    # pangram, in short chunks (covers a-z again, in real words)
    "the quick brown",
    "fox jumps over",
    "the lazy dog",
    "pack my box with",
    "five dozen liquor jugs",
    # digits
    "0123456789",
    "3.14159",
    "call 867 5309",
    "$42.99 or 15%",
    # punctuation and contractions
    "Hello, world!",
    "it's, don't, can't",
    "yes / no?",
    "one (two) three",
    "wait - stop.",
    "5:30 pm - 6:00 pm",
    # more everyday words for letterform variety
    "they", "have", "from", "what", "were", "there",
    "when", "your", "said", "which",
    # natural sentences (real usage)
    "the meeting is at noon",
    "please send it today",
    "I will call you later",
    "turn the lights off",
    "add milk to the list",
]

DEFAULT_TARGET = len(ENROLLMENT_PROMPTS)

# Prompts that exist for character coverage, not to be read as language.
# A language-model decoder (TrOCR, and any CTC model with an LM) will mangle
# these however neatly they are written, so aggregate accuracy over them is
# misleading. Evaluation reports them separately.
ROTE_PROMPTS = frozenset(
    {
        "abcdefghijklm",
        "nopqrstuvwxyz",
        "ABCDEFGHIJKLM",
        "NOPQRSTUVWXYZ",
        "0123456789",
        "3.14159",
        "0 1 2 3 4 5 6 7 8 9",
    }
)


def is_rote(label: str) -> bool:
    return label in ROTE_PROMPTS


@dataclass
class Coverage:
    lower: int
    upper: int
    digit: int
    missing_lower: str
    missing_upper: str
    missing_digit: str

    @property
    def complete(self) -> bool:
        return self.lower == 26 and self.upper == 26 and self.digit == 10

    def summary(self) -> str:
        def part(name: str, have: int, total: int) -> str:
            return f"{name} {'OK' if have == total else f'{have}/{total}'}"

        return "   ".join(
            (
                part("a-z", self.lower, 26),
                part("A-Z", self.upper, 26),
                part("0-9", self.digit, 10),
            )
        )


def char_coverage(labels: Iterable[str]) -> Coverage:
    seen = set("".join(labels))
    lower = set(string.ascii_lowercase)
    upper = set(string.ascii_uppercase)
    digit = set(string.digits)
    return Coverage(
        lower=len(lower & seen),
        upper=len(upper & seen),
        digit=len(digit & seen),
        missing_lower="".join(sorted(lower - seen)),
        missing_upper="".join(sorted(upper - seen)),
        missing_digit="".join(sorted(digit - seen)),
    )


def is_enrolled(session_saved: int, coverage: Coverage, target: int) -> bool:
    """Enough data collected: either the full target, or a solid partial pass
    that already covers every letter and digit."""
    if session_saved >= target:
        return True
    return session_saved >= max(20, int(target * 0.6)) and coverage.complete
