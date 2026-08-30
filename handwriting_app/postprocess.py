"""Map recognizer output onto real English words with SymSpell.

SymSpell is optional. When ``symspellpy`` and its bundled dictionaries are not
installed, :class:`SpellCorrector` is a no-op and reports itself unavailable.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"\S+|\s+")
_AFFIX = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)


def _match_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


class SpellCorrector:
    def __init__(self, max_edit_distance: int = 2) -> None:
        self._max_edit_distance = max_edit_distance
        self._sym = None
        self._verbosity = None
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

    @property
    def available(self) -> bool:
        return self._sym is not None

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
