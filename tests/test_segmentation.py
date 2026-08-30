from handwriting_app.ink import Ink
from handwriting_app.segmentation import segment_words


def _stroke(ink, xs, y0=0, y1=40):
    s = ink.start_stroke()
    for x in xs:
        s.add(x, y0)
        s.add(x, y1)
    return s


def test_empty_ink_yields_no_words():
    assert segment_words(Ink()) == []


def test_two_clusters_split_by_a_wide_gap():
    ink = Ink()
    _stroke(ink, [0, 10, 20])       # word 1
    _stroke(ink, [25, 35])          # still word 1 (small gap)
    _stroke(ink, [200, 210, 220])   # word 2 (big gap)
    words = segment_words(ink, gap_ratio=0.4)
    assert len(words) == 2
    assert len(words[0].strokes) == 2
    assert len(words[1].strokes) == 1


def test_single_connected_word_stays_together():
    ink = Ink()
    _stroke(ink, list(range(0, 120, 3)))
    words = segment_words(ink)
    assert len(words) == 1


def test_stroke_order_is_preserved_within_a_word():
    ink = Ink()
    first = _stroke(ink, [0, 5])
    second = _stroke(ink, [6, 11])
    words = segment_words(ink)
    assert words[0].strokes == [first, second]
