from handwriting_app.enrollment import (
    ENROLLMENT_PROMPTS,
    ROTE_PROMPTS,
    char_coverage,
    is_enrolled,
    is_rote,
)


def test_enrollment_set_is_reasonably_sized():
    assert 30 <= len(ENROLLMENT_PROMPTS) <= 50


def test_enrollment_set_covers_every_letter_and_digit():
    cov = char_coverage(ENROLLMENT_PROMPTS)
    assert cov.lower == 26, cov.missing_lower
    assert cov.upper == 26, cov.missing_upper
    assert cov.digit == 10, cov.missing_digit
    assert cov.complete


def test_partial_coverage_reports_missing():
    cov = char_coverage(["abc", "XYZ", "123"])
    assert cov.lower == 3
    assert "d" in cov.missing_lower
    assert not cov.complete
    assert "3/10" in cov.summary()


def test_is_enrolled_on_full_target():
    cov = char_coverage(["nothing"])
    assert is_enrolled(40, cov, 40)
    assert not is_enrolled(39, cov, 40)


def test_rote_prompts_are_a_minority_of_the_set():
    rote = [p for p in ENROLLMENT_PROMPTS if is_rote(p)]
    assert 3 <= len(rote) <= 10
    # the natural-language prompts must still dominate, or evaluation is skewed
    assert len(rote) < len(ENROLLMENT_PROMPTS) / 3


def test_is_rote():
    assert is_rote("abcdefghijklm")
    assert is_rote("0123456789")
    assert not is_rote("the meeting is at noon")
    assert not is_rote("Hello, world!")
    assert ROTE_PROMPTS <= set(ENROLLMENT_PROMPTS) | {"0 1 2 3 4 5 6 7 8 9"}


def test_is_enrolled_on_solid_partial_pass_with_full_coverage():
    full = char_coverage(ENROLLMENT_PROMPTS)
    assert is_enrolled(24, full, 40)      # >= 60% and every char seen
    assert not is_enrolled(10, full, 40)  # too few samples


def test_first_undone_skips_a_gap_not_just_a_leading_run():
    """Regression: the old resume logic skipped a leading run of done prompts
    once at startup, then walked the rest linearly — so deleting one sample
    from the middle of a finished enrollment made every prompt AFTER it get
    re-served and re-saved on resume, not just the one that was actually
    missing. This has to keep skipping past done prompts wherever they are,
    every time index advances, not only at the start."""
    from handwriting_app.training import _first_undone

    prompts = ["a", "b", "c", "d", "e"]
    # "c" is the one missing sample -- deleted from the middle of a finished set.
    done = {"a", "b", "d", "e"}

    assert _first_undone(prompts, done) == 2          # lands on "c", correctly
    # Simulating what used to be the bug: after resuming at "c" and saving it,
    # the walk must land past "d" and "e" too, not re-serve them.
    done_after_saving_c = done | {"c"}
    assert _first_undone(prompts, done_after_saving_c, start=3) == 5  # finished


def test_first_undone_with_no_gap_lands_on_the_first_new_prompt():
    from handwriting_app.training import _first_undone

    prompts = ["a", "b", "c", "d"]
    assert _first_undone(prompts, {"a", "b"}) == 2


def test_first_undone_all_done_reaches_the_end():
    from handwriting_app.training import _first_undone

    prompts = ["a", "b", "c"]
    assert _first_undone(prompts, {"a", "b", "c"}) == 3


def test_first_undone_nothing_done_starts_at_zero():
    from handwriting_app.training import _first_undone

    assert _first_undone(["a", "b"], set()) == 0
