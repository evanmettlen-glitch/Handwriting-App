"""Build a personal word list from collected training samples.

The words a user actually writes (names, jargon, place names) are exactly the
ones a generic dictionary "corrects" into something wrong. Feeding these into the
spell corrector as known terms stops that, with no model training required.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import List

from handwriting_app.dataset import iter_samples
from handwriting_app.naming import user_slug

_WORD = re.compile(r"[A-Za-z][A-Za-z']*")


def _counts_in_dir(samples_dir: Path) -> Counter:
    counts: Counter = Counter()
    if not samples_dir.is_dir():
        return counts
    for sample in iter_samples(samples_dir):
        for word in _WORD.findall(sample.label):
            if len(word) >= 2:
                counts[word.lower()] += 1
    return counts


def personal_word_counts(samples_root: str, user: str = "") -> Counter:
    """Word -> times written, across the relevant sample folder(s)."""
    root = Path(samples_root)
    dirs: List[Path] = []
    if user:
        dirs.append(root / user_slug(user))
    else:
        dirs.append(root)
        dirs.extend(p for p in sorted(root.glob("*")) if p.is_dir())

    total: Counter = Counter()
    for directory in dirs:
        total.update(_counts_in_dir(directory))
    return total
