import math

from handwriting_app.cleanup import (
    CleanupConfig,
    clean_ink,
    find_traverses,
    writing_height,
)
from handwriting_app.ink import Ink, Stroke
from handwriting_app.segmentation import segment_words

TOP, BOTTOM = 0.0, 100.0


def _sample(a, b, step=4.0):
    """Points along the segment a..b, roughly ``step`` pixels apart."""
    count = max(1, int(math.dist(a, b) / step))
    return [
        (a[0] + (b[0] - a[0]) * i / count, a[1] + (b[1] - a[1]) * i / count)
        for i in range(count + 1)
    ]


def _letter(x0, width=40.0):
    """A caret standing in for a letter: full writing height, starts and ends
    on the baseline, and no step in it is anywhere near horizontal."""
    peak = (x0 + width / 2, TOP)
    return _sample((x0, BOTTOM), peak) + _sample(peak, (x0 + width, BOTTOM))[1:]


def _diagonal_x(stroke):
    xs = [x for x, _ in stroke.points]
    return max(xs) - min(xs)


def _ink(*point_lists):
    return Ink([Stroke(list(points)) for points in point_lists])


def _drag(x0, x1, y=BOTTOM, step=30.0):
    """A fast flat traverse — a finger sliding across the pad."""
    return _sample((x0, y), (x1, y), step=step)


# -- writing height ---------------------------------------------------------


def test_writing_height_ignores_a_stray_tap():
    writing = _ink(_letter(0), _letter(60))
    with_tap = _ink(_letter(0), _letter(60), [(300.0, -400.0), (303.0, -397.0)])

    plain = writing_height(writing)
    assert abs(writing_height(with_tap) - plain) < 0.1 * plain
    # Every threshold is a fraction of this, and the plain bounding box — the
    # obvious measure — would have been five times too tall.
    top, bottom = with_tap.bounds()[1], with_tap.bounds()[3]
    assert bottom - top > 4 * plain


def test_writing_height_of_empty_ink_is_zero():
    assert writing_height(Ink()) == 0.0


# -- traverse detection -----------------------------------------------------


def test_traverse_is_found_between_two_letters():
    points = _letter(0) + _drag(40, 240)[1:] + _letter(240)[1:]
    height = writing_height(_ink(points))
    runs = find_traverses(points, height, CleanupConfig())
    assert len(runs) == 1
    start, end = runs[0]
    # The run may creep a little way up the baseline curl at either end, but it
    # must cover the drag and nothing like the whole letter.
    assert 30 <= points[start][0] <= 40
    assert 240 <= points[end][0] <= 250


def test_a_single_jittery_sample_does_not_break_a_traverse():
    flat = _drag(40, 240)
    middle = len(flat) // 2
    # One sample 4px off the line: under the jitter tolerance, so the run holds.
    jittered = flat[:middle] + [(flat[middle][0] + 1, BOTTOM - 4)] + flat[middle:]
    points = _letter(0) + jittered[1:] + _letter(240)[1:]
    height = writing_height(_ink(points))
    assert len(find_traverses(points, height, CleanupConfig())) == 1


def test_a_cursive_ligature_is_not_a_traverse():
    # A connector a quarter of the line height long — normal joined writing.
    points = _letter(0) + _drag(40, 65, step=4.0)[1:] + _letter(65)[1:]
    height = writing_height(_ink(points))
    assert find_traverses(points, height, CleanupConfig()) == []


def test_a_sloping_move_is_not_a_traverse():
    # Long, straight and fast, but it climbs — that is writing, not a drag.
    points = _sample((0.0, BOTTOM), (200.0, TOP), step=25.0)
    height = writing_height(_ink(_letter(0), points))
    assert find_traverses(points, height, CleanupConfig()) == []


def test_a_short_gap_crossed_quickly_is_a_traverse():
    """Words written close together still leave a slide when the pen never
    lifts — it is only half a line height, but it was covered in one stride."""
    points = _letter(0) + _drag(40, 100, step=30.0)[1:] + _letter(100)[1:]
    cleaned, report = clean_ink(_ink(points))
    assert report.traverses_cut == 1
    assert len(cleaned.strokes) == 2


def test_the_same_short_gap_drawn_slowly_is_left_alone():
    """Same geometry at writing speed is a deliberate mark — a dash, a
    flourish, an underscore — and deleting it would be a bug."""
    points = _letter(0) + _drag(40, 100, step=4.0)[1:] + _letter(100)[1:]
    cleaned, report = clean_ink(_ink(points))
    assert report.traverses_cut == 0
    assert len(cleaned.strokes) == 1


# -- no pen lift ------------------------------------------------------------


def test_no_lift_writing_is_cut_back_into_words():
    """One unbroken stroke across two words becomes two strokes again."""
    points = _letter(0) + _drag(40, 240)[1:] + _letter(240)[1:]
    cleaned, report = clean_ink(_ink(points))

    assert report.traverses_cut == 1
    assert report.strokes_in == 1 and report.strokes_out == 2
    assert report.traverse_pixels >= 190
    # ...and the word splitter, which needs pen lifts, can see them again.
    assert len(segment_words(cleaned)) == 2


def test_the_letters_either_side_of_a_cut_are_kept_whole():
    points = _letter(0) + _drag(40, 240)[1:] + _letter(240)[1:]
    cleaned, _ = clean_ink(_ink(points))
    first, second = cleaned.strokes
    assert first.points[0] == (0.0, BOTTOM)
    assert second.points[-1] == (280.0, BOTTOM)
    # Each letter keeps essentially all of its width...
    assert _diagonal_x(first) >= 35 and _diagonal_x(second) >= 35
    # ...and none of the drag between them survives.
    assert not [x for s in cleaned.strokes for x, _ in s.points if 50 < x < 230]


def test_several_words_written_without_lifting_all_split():
    points = _letter(0)
    for x0 in (240.0, 480.0):
        points = points + _drag(points[-1][0], x0)[1:] + _letter(x0)[1:]
    cleaned, report = clean_ink(_ink(points))
    assert report.traverses_cut == 2
    assert len(cleaned.strokes) == 3


# -- accidental drags -------------------------------------------------------


def test_a_drag_across_the_writing_is_removed_entirely():
    writing = [_letter(0), _letter(60), _letter(120)]
    cleaned, report = clean_ink(_ink(*writing, _drag(-20, 400, y=50.0)))
    assert report.traverses_cut == 1
    assert [s.points for s in cleaned.strokes] == writing


def test_a_run_in_hook_is_trimmed_off_the_front_of_a_letter():
    points = _drag(-200, 0) + _letter(0)[1:]
    cleaned, report = clean_ink(_ink(points))
    assert report.traverses_cut == 1
    assert len(cleaned.strokes) == 1
    kept = cleaned.strokes[0].points
    assert min(x for x, _ in kept) >= 0  # the whole run-in is gone
    assert max(x for x, _ in kept) == 40  # the letter itself is not


def test_cleanup_never_empties_the_pad():
    """A pad holding nothing but a drag is returned untouched, not wiped."""
    ink = _ink(_drag(0, 400))
    cleaned, report = clean_ink(ink)
    assert cleaned is ink
    assert report.reverted
    assert report.summary() == "cleanup skipped (would have erased everything)"


# -- stray specks -----------------------------------------------------------


def test_an_isolated_speck_is_dropped_but_an_i_dot_is_kept():
    dot = [(20.0, -15.0), (23.0, -12.0)]       # over the letter at x 0..40
    stray = [(400.0, 60.0), (403.0, 63.0)]     # far to the right of everything
    cleaned, report = clean_ink(_ink(_letter(0), _letter(60), dot, stray))
    assert report.specks_dropped == 1
    assert [s.points for s in cleaned.strokes][-1] == dot


def test_a_speck_far_below_the_writing_is_dropped():
    # Horizontally it sits under the writing, so only the vertical drift catches it.
    smudge = [(50.0, 300.0), (53.0, 303.0)]
    cleaned, report = clean_ink(_ink(_letter(0), _letter(60), smudge))
    assert report.specks_dropped == 1
    assert len(cleaned.strokes) == 2


def test_a_lone_speck_is_never_dropped():
    """A single full stop is the whole message, not a stray mark."""
    ink = _ink([(10.0, 10.0), (12.0, 12.0)])
    cleaned, report = clean_ink(ink)
    assert report.specks_dropped == 0
    assert len(cleaned.strokes) == 1


# -- general properties -----------------------------------------------------


def test_clean_writing_is_passed_through_untouched():
    ink = _ink(_letter(0), _letter(60), _letter(120))
    cleaned, report = clean_ink(ink)
    assert not report.changed
    assert report.summary() == ""
    assert cleaned.strokes == ink.strokes


def test_cleanup_is_idempotent():
    ink = _ink(_letter(0) + _drag(40, 240)[1:] + _letter(240)[1:])
    once, _ = clean_ink(ink)
    twice, report = clean_ink(once)
    assert not report.changed
    assert [s.points for s in twice.strokes] == [s.points for s in once.strokes]


def test_thresholds_are_scale_invariant():
    """Thresholds are fractions of the writing height, so tiny writing and
    large writing must clean the same way."""
    points = _letter(0) + _drag(40, 240)[1:] + _letter(240)[1:]
    big = [(x * 3, y * 3) for x, y in points]
    small = [(x / 4, y / 4) for x, y in points]
    assert clean_ink(_ink(big))[1].traverses_cut == 1
    assert clean_ink(_ink(small))[1].traverses_cut == 1


def test_cleanup_does_not_mutate_the_input():
    points = _letter(0) + _drag(40, 240)[1:] + _letter(240)[1:]
    ink = _ink(points)
    clean_ink(ink)
    assert ink.strokes[0].points == points


def test_empty_ink_is_handled():
    cleaned, report = clean_ink(Ink())
    assert cleaned.is_empty
    assert not report.changed


# -- pipeline wiring --------------------------------------------------------


def _pipeline(**kw):
    from handwriting_app.pipeline import PipelineConfig, RecognitionPipeline
    from handwriting_app.recognizer.base import Recognizer

    class Spy(Recognizer):
        name = "spy"

        def __init__(self):
            self.images = 0

        def recognize(self, image, *, hint="line"):
            self.images += 1
            return "x"

    return RecognitionPipeline(Spy(), PipelineConfig(spellcheck=False, **kw))


def test_the_cleanup_flag_reaches_the_config():
    from handwriting_app.config import parse_args

    assert parse_args([]).cleanup is True
    assert parse_args(["--no-cleanup"]).cleanup is False


def test_pipeline_splits_no_lift_ink_into_two_words():
    ink = _ink(_letter(0) + _drag(40, 240)[1:] + _letter(240)[1:])
    pipe = _pipeline(segment=True)
    pipe.run(ink)
    assert pipe.recognizer.images == 2
    assert pipe.last_cleanup.traverses_cut == 1


def test_disabling_cleanup_leaves_the_ink_alone():
    ink = _ink(_letter(0) + _drag(40, 240)[1:] + _letter(240)[1:])
    pipe = _pipeline(segment=True, cleanup=False)
    pipe.run(ink)
    assert pipe.recognizer.images == 1  # one unbroken stroke, so one "word"
    assert pipe.last_cleanup is None
    assert "ink cleanup off" in pipe.notes
