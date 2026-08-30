from PIL import Image

from handwriting_app.ink import Ink


def test_empty_ink_has_no_bounds_or_image():
    ink = Ink()
    assert ink.is_empty
    assert ink.bounds() is None
    assert ink.render() is None


def test_stroke_bounds_and_non_empty():
    ink = Ink()
    stroke = ink.start_stroke()
    stroke.add(10, 20)
    stroke.add(30, 45)
    assert not ink.is_empty
    assert ink.bounds() == (10.0, 20.0, 30.0, 45.0)


def test_started_but_unused_stroke_is_still_empty():
    ink = Ink()
    ink.start_stroke()
    assert ink.is_empty
    assert ink.render() is None


def test_render_produces_white_background_image_with_dark_ink():
    ink = Ink()
    stroke = ink.start_stroke()
    for x in range(0, 60, 4):
        stroke.add(x, 30)
    image = ink.render(pad=8, stroke_width=6, supersample=1)
    assert isinstance(image, Image.Image)
    assert image.mode == "L"
    extrema = image.getextrema()
    assert extrema[0] == 0  # some fully dark ink pixels
    assert extrema[1] == 255  # some white background


def test_render_clamps_width():
    ink = Ink()
    stroke = ink.start_stroke()
    stroke.add(0, 0)
    stroke.add(5000, 10)
    image = ink.render(max_width=800, supersample=1)
    assert image is not None
    assert image.width == 800


def test_clear_resets():
    ink = Ink()
    ink.start_stroke().add(1, 1)
    ink.clear()
    assert ink.is_empty
