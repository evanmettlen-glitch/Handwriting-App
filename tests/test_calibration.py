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
        deslant=False, stroke_width=6, render_pad=16, smooth=False,
        word_gap_ratio=1.3,
        fixes={"a": "b"}, baseline_cer=0.5, tuned_cer=0.3, samples=40,
    )
    save(cal, tmp_path)
    loaded = load(tmp_path)
    assert loaded.deslant is False
    assert loaded.stroke_width == 6
    assert loaded.render_pad == 16
    assert loaded.smooth is False
    assert loaded.word_gap_ratio == 1.3
    assert loaded.fixes == {"a": "b"}
    assert loaded.tuned_cer == 0.3
    assert loaded.samples == 40


def test_resolve_segment_defaults_by_model_kind():
    from handwriting_app.pipeline import resolve_segment

    # TrOCR is line-trained: whole lines beat word fragments.
    assert resolve_segment(None, "trocr-torch:trocr-base-handwritten") is False
    assert resolve_segment(None, "trocr-onnx:personal") is False
    # tesseract is weak on multi-word images, so split for it.
    assert resolve_segment(None, "tesseract") is True
    # explicit settings always win
    assert resolve_segment(True, "trocr-torch:x") is True
    assert resolve_segment(False, "tesseract") is False


def test_calibrated_gap_ratio_overrides_the_config():
    from handwriting_app.ink import Ink
    from handwriting_app.pipeline import PipelineConfig, RecognitionPipeline
    from handwriting_app.recognizer.base import Recognizer

    class Counter(Recognizer):
        name = "counter"

        def __init__(self):
            self.calls = 0

        def recognize(self, image, *, hint="line"):
            self.calls += 1
            return "x"

    # two clusters ~90px apart, ink height 40 -> splits below ratio ~2.2
    ink = Ink()
    for x0 in (0, 90):
        stroke = ink.start_stroke()
        for x in range(x0, x0 + 20, 4):
            stroke.add(x, 0)
            stroke.add(x, 40)

    tight = RecognitionPipeline(
        Counter(), PipelineConfig(spellcheck=False, word_gap_ratio=0.4)
    )
    tight.run(ink)
    assert tight.recognizer.calls == 2  # split into two words

    merged = RecognitionPipeline(
        Counter(),
        PipelineConfig(
            spellcheck=False,
            word_gap_ratio=0.4,
            calibration=Calibration(word_gap_ratio=5.0),
        ),
    )
    merged.run(ink)
    assert merged.recognizer.calls == 1  # calibration widened the threshold


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


def test_calibration_overrides_an_explicitly_set_config_value_too():
    """Documents real, deliberate-but-undiscoverable behavior: calibration
    wins even over a render setting the caller set on purpose (what
    --stroke-width, --word-gap-ratio, --no-deslant, --no-smooth become on the
    CLI), with nothing telling the caller their value was ignored. See the
    override comment in RecognitionPipeline.__init__ and the --no-calibration
    help text, which is the only way to make an explicit value stick."""
    from handwriting_app.pipeline import PipelineConfig, RecognitionPipeline
    from handwriting_app.recognizer.base import Recognizer

    class Fake(Recognizer):
        name = "fake"

        def recognize(self, image, *, hint="line"):
            return ""

    # Every one of these differs from what the calibration below carries --
    # standing in for a user who explicitly chose these on the command line.
    explicit = PipelineConfig(
        stroke_width=99,
        render_pad=99,
        deslant=False,
        smooth=False,
        word_gap_ratio=9.0,
        calibration=Calibration(
            stroke_width=4, render_pad=8, deslant=True, smooth=True,
            word_gap_ratio=0.4,
        ),
    )
    pipe = RecognitionPipeline(Fake(), explicit)
    assert pipe._stroke_width == 4
    assert pipe._render_pad == 8
    assert pipe._deslant is True
    assert pipe._smooth is True
    assert pipe._word_gap_ratio == 0.4
