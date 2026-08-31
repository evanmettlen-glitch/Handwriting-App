from handwriting_app.calibration import Calibration, load, save
from handwriting_app.textalign import cer, char_confusions


def test_cer():
    assert cer("abc", "abc") == 0.0
    assert cer("abd", "abc") == 1 / 3
    assert cer("", "abc") == 1.0
    assert cer("", "") == 0.0


def test_char_confusions_finds_the_differing_span():
    pairs = char_confusions("rn", "m")
    assert ("rn", "m") in pairs
    assert char_confusions("same", "same") == []


def test_apply_fixes_is_whole_word_and_case_tolerant():
    cal = Calibration(fixes={"tne": "the", "l": "I"})
    assert cal.apply_fixes("tne cat") == "the cat"
    assert cal.apply_fixes("Tne cat") == "the cat"      # lowercase fallback
    assert cal.apply_fixes("tnet cat") == "tnet cat"    # not a substring match
    assert Calibration().apply_fixes("unchanged") == "unchanged"


def test_roundtrip(tmp_path):
    cal = Calibration(
        deslant=False, stroke_width=6, render_pad=16,
        fixes={"a": "b"}, baseline_cer=0.5, tuned_cer=0.3, samples=40,
    )
    save(cal, tmp_path)
    loaded = load(tmp_path)
    assert loaded.deslant is False
    assert loaded.stroke_width == 6
    assert loaded.render_pad == 16
    assert loaded.fixes == {"a": "b"}
    assert loaded.tuned_cer == 0.3
    assert loaded.samples == 40


def test_load_missing_or_corrupt_returns_none(tmp_path):
    assert load(tmp_path) is None
    (tmp_path / "calibration.json").write_text("{not json", encoding="utf-8")
    assert load(tmp_path) is None


def test_pipeline_uses_calibration_render_settings_and_fixes():
    from handwriting_app.ink import Ink
    from handwriting_app.pipeline import PipelineConfig, RecognitionPipeline
    from handwriting_app.recognizer.base import Recognizer

    class Fake(Recognizer):
        name = "fake"

        def __init__(self):
            self.sizes = []

        def recognize(self, image, *, hint="line"):
            self.sizes.append(image.size)
            return "tne"

    ink = Ink()
    stroke = ink.start_stroke()
    for x in range(0, 40, 4):
        stroke.add(x, 20)

    cal = Calibration(deslant=False, stroke_width=4, render_pad=8, fixes={"tne": "the"})
    pipe = RecognitionPipeline(
        Fake(), PipelineConfig(spellcheck=False, calibration=cal)
    )
    assert pipe.run(ink) == "the"
    assert "calibrated on 0 samples" in " ".join(pipe.notes)
