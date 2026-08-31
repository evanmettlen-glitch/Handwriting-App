"""Character alignment helpers for scoring and mining recognizer errors."""

from __future__ import annotations

import difflib
from typing import List, Tuple


def cer(pred: str, gold: str) -> float:
    """Character error rate: edit distance / len(gold)."""
    m, n = len(pred), len(gold)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (pred[i - 1] != gold[j - 1]))
            prev = cur
    return dp[n] / n


def char_confusions(pred: str, gold: str) -> List[Tuple[str, str]]:
    """(read_as, should_be) fragments where the two strings differ."""
    matcher = difflib.SequenceMatcher(None, pred, gold, autojunk=False)
    out: List[Tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        out.append((pred[i1:i2], gold[j1:j2]))
    return out
