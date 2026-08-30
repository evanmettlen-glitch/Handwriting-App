"""The list of words/phrases to elicit during training mode.

A built-in list is always available. ``handwriting_app/data/prompts.txt`` (if
present) and ``--prompts-file`` replace it.
"""

from __future__ import annotations

import importlib.resources as resources
from pathlib import Path
from typing import List, Optional

# Built-in fallback: pangrams, common words, digits, punctuation, names.
DEFAULT_PROMPTS: List[str] = [
    "the quick brown fox",
    "jumps over the lazy dog",
    "pack my box with five dozen liquor jugs",
    "how vexingly quick daft zebras jump",
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come",
    "over", "think", "also", "back", "after", "use", "two", "how", "our",
    "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "us",
    "Monday", "Tuesday", "Wednesday", "Saturday",
    "January", "August", "December",
    "London", "Raspberry Pi", "English",
    "0 1 2 3 4 5 6 7 8 9", "2026", "room 214", "order 4417",
    "version 0.1.0", "call 555 0139", "$19.95", "75%",
    "Hello, world!", "yes or no?", "it's over there.", "wait - stop!",
    "one (two) three", "email me later", "see you tomorrow",
    "turn it off and on", "the meeting is at noon", "please and thank you",
]


def _parse(text: str) -> List[str]:
    out: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def load_prompts(path: Optional[str] = None) -> List[str]:
    if path:
        return _parse(Path(path).read_text(encoding="utf-8")) or list(DEFAULT_PROMPTS)

    try:
        resource = resources.files("handwriting_app") / "data" / "prompts.txt"
        prompts = _parse(resource.read_text(encoding="utf-8"))
        if prompts:
            return prompts
    except (OSError, ModuleNotFoundError):
        pass
    return list(DEFAULT_PROMPTS)
