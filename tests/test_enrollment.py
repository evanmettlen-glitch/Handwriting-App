from handwriting_app.enrollment import (
    ENROLLMENT_PROMPTS,
    char_coverage,
    is_enrolled,
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


def test_is_enrolled_on_solid_partial_pass_with_full_coverage():
    full = char_coverage(ENROLLMENT_PROMPTS)
    assert is_enrolled(24, full, 40)      # >= 60% and every char seen
    assert not is_enrolled(10, full, 40)  # too few samples
