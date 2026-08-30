from handwriting_app.postprocess import SpellCorrector, _match_case


def test_match_case():
    assert _match_case("HELLO", "world") == "WORLD"
    assert _match_case("Hello", "world") == "World"
    assert _match_case("hello", "world") == "world"


def test_corrector_is_a_noop_when_unavailable():
    corrector = SpellCorrector()
    if corrector.available:
        return  # covered by the dictionary test below
    text = "teh quikc bronw fox"
    assert corrector.correct_line(text) == text


def test_corrector_fixes_typos_when_dictionary_present():
    corrector = SpellCorrector()
    if not corrector.available:
        return
    assert corrector.correct_word("recieve") == "receive"
    assert corrector.correct_line("teh cat") == "the cat"
    # unknown-but-plausible tokens and casing are preserved
    assert corrector.correct_word("Xyzzy").istitle()
