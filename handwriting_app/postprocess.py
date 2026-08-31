"""Map recognizer output onto real English words with SymSpell.

SymSpell is optional. When ``symspellpy`` and its bundled dictionaries are not
installed, :class:`SpellCorrector` is a no-op and reports itself unavailable.
"""

from __future__ import annotations

import re
from typing import List, Mapping, Optional

_TOKEN = re.compile(r"\S+|\s+")
_AFFIX = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)

# How strongly a personally-collected word outranks dictionary corrections.
# ~mid-frequency in the bundled English dictionary, so common words still win
# but the corrector stops "fixing" the user's own vocabulary.
_PERSONAL_WEIGHT = 5_000_000


def _match_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


class SpellCorrector:
    def __init__(
        self,
        max_edit_distance: int = 2,
        boost: Optional[Mapping[str, int]] = None,
    ) -> None:
        self._max_edit_distance = max_edit_distance
        self._sym = None
        self._verbosity = None
        self.boosted = 0
        try:
            import importlib.resources as resources

            from symspellpy import SymSpell, Verbosity

            sym = SymSpell(
                max_dictionary_edit_distance=max_edit_distance, prefix_length=7
            )
            unigrams = resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
            bigrams = (
                resources.files("symspellpy")
                / "frequency_bigramdictionary_en_243_342.txt"
            )
            with resources.as_file(unigrams) as path:
                sym.load_dictionary(str(path), term_index=0, count_index=1)
            with resources.as_file(bigrams) as path:
                sym.load_bigram_dictionary(str(path), term_index=0, count_index=2)
        except Exception:  # noqa: BLE001 - any failure just disables correction
            return
        self._sym = sym
        self._verbosity = Verbosity.TOP

        for term, times in (boost or {}).items():
            term = term.lower().strip()
            if len(term) < 2 or not term.replace("'", "").isalpha():
                continue
            sym.create_dictionary_entry(term, max(1, times) * _PERSONAL_WEIGHT)
            self.boosted += 1

    @property
    def available(self) -> bool:
        return self._sym is not None

    def knows(self, word: str) -> bool:
        """True when ``word`` is in the dictionary exactly (no fuzzy match)."""
        if self._sym is None:
            return False
        return word.lower() in self._sym.words

    def join_split_letters(self, text: str, max_run: int = 24) -> str:
        """Glue runs of single letters back into words: ``a n d`` -> ``and``.

        Widely-spaced printing reads to the recognizer as separate one-letter
        words. SymSpell's compound mode mangles these (``w i t h`` -> ``a it a``),
        so join them explicitly and only keep a join the dictionary confirms.
        """
        if self._sym is None or not text:
            return text

        tokens = text.split(" ")
        out: List[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if not (len(token) == 1 and token.isalpha()):
                out.append(token)
                index += 1
                continue

            # Collect the maximal run of single letters starting here.
            end = index
            while (
                end < len(tokens)
                and len(tokens[end]) == 1
                and tokens[end].isalpha()
                and end - index < max_run
            ):
                end += 1
            run = tokens[index:end]

            if len(run) < 2:
                out.append(token)
                index += 1
                continue

            # Greedily consume the longest prefix that forms a known word.
            cursor = 0
            while cursor < len(run):
                for size in range(len(run) - cursor, 1, -1):
                    candidate = "".join(run[cursor : cursor + size])
                    if self.knows(candidate):
                        out.append(_match_case(run[cursor], candidate))
                        cursor += size
                        break
                else:
                    out.append(run[cursor])
                    cursor += 1
            index = end

        return " ".join(out)

    def correct_word(self, word: str) -> str:
        if self._sym is None or len(word) < 2 or not word.isalpha():
            return word
        suggestions = self._sym.lookup(
            word.lower(),
            self._verbosity,
            max_edit_distance=self._max_edit_distance,
            include_unknown=True,
        )
        if not suggestions or suggestions[0].distance == 0:
            return word
        return _match_case(word, suggestions[0].term)

    def correct_line(self, text: str, *, compound: bool = False) -> str:
        if self._sym is None or not text.strip():
            return text
        if compound:
            result = self._sym.lookup_compound(
                text, max_edit_distance=self._max_edit_distance, transfer_casing=True
            )
            return result[0].term if result else text

        out = []
        for token in _TOKEN.findall(text):
            if token.isspace():
                out.append(token)
                continue
            prefix, core, suffix = _AFFIX.match(token).groups()
            out.append(prefix + self.correct_word(core) + suffix)
        return "".join(out)
