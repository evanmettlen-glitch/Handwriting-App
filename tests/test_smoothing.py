import math

from handwriting_app.ink import Ink, Stroke


def test_short_strokes_pass_through_unchanged():
    assert Stroke([(0, 0)]).smoothed() == [(0.0, 0.0)]
    assert Stroke([(0, 0), (10, 0)]).smoothed() == [(0.0, 0.0), (10.0, 0.0)]


def test_sparse_stroke_is_densified():
    sparse = Stroke([(0, 0), (50, 0), (100, 0), (150, 0)])
    dense = sparse.smoothed(max_spacing=3.0)
    assert len(dense) > 4 * len(sparse.points)
    gaps = [math.dist(a, b) for a, b in zip(dense, dense[1:])]
    assert max(gaps) <= 4.0


def test_already_dense_stroke_is_left_alone():
    dense = Stroke([(float(x), 0.0) for x in range(0, 30, 2)])
    out = dense.smoothed(max_spacing=3.0)
    assert len(out) == len(dense.points)


def test_spline_passes_through_the_captured_points():
    pts = [(0.0, 0.0), (30.0, 40.0), (60.0, 0.0), (90.0, 40.0)]
    out = Stroke(pts).smoothed(max_spacing=5.0)
    for original in pts:
        assert any(math.dist(original, p) < 1e-6 for p in out), original


def test_smoothing_curves_away_from_the_chord():
    """Interpolated points must leave the straight line, or nothing changed.

    Straight-line rendering puts every point of the first segment on y == x;
    the spline arrives at the apex horizontally, so it bulges above that chord.
    """
    pts = [(0.0, 0.0), (50.0, 50.0), (100.0, 0.0)]
    out = Stroke(pts).smoothed(max_spacing=2.0)
    first_segment = [(x, y) for x, y in out if 0 < x < 50]
    assert first_segment
    assert max(y - x for x, y in first_segment) > 1.0


def test_render_smooth_flag_changes_the_image():
    ink = Ink()
    stroke = ink.start_stroke()
    for x, y in [(0, 0), (40, 60), (80, 0), (120, 60)]:
        stroke.add(x, y)

    smooth = ink.render(smooth=True, supersample=1, stroke_width=4)
    plain = ink.render(smooth=False, supersample=1, stroke_width=4)
    assert smooth.size == plain.size
    assert list(smooth.getdata()) != list(plain.getdata())


def test_render_defaults_to_smoothing():
    ink = Ink()
    stroke = ink.start_stroke()
    for x, y in [(0, 0), (40, 60), (80, 0)]:
        stroke.add(x, y)
    assert list(ink.render(supersample=1).getdata()) == list(
        ink.render(smooth=True, supersample=1).getdata()
    )
