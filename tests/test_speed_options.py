"""The latency knobs: flag parsing, plumbing, and the pipeline split."""

from PIL import Image

from handwriting_app.calibration import Calibration
from handwriting_app.config import AppConfig, parse_args
from handwriting_app.ink import Ink, Stroke
from handwriting_app.pipeline import PipelineConfig, RecognitionPipeline
from handwriting_app.recognizer import Recognizer, build_recognizer


class StubRecognizer(Recognizer):
    name = "trocr-stub"

    def __init__(self, text: str = "hi") -> None:
        self.text = text
        self.images = []

    def recognize(self, image, *, hint: str = "line") -> str:
        self.images.append(image)
        return self.text


def _ink() -> Ink:
    return Ink([Stroke([(10.0, 10.0), (40.0, 30.0), (70.0, 10.0)])])


# -- flags ---------------------------------------------------------------
def test_speed_defaults_are_the_fast_ones():
    cfg = parse_args([])
    # Greedy by default: the TrOCR checkpoints ship num_beams=4, which costs
    # roughly 4x the decode time for marginal accuracy on this hardware.
    assert cfg.beams == 1
    # int8 changes accuracy, so it stays opt-in until measured.
    assert cfg.quantize is False
    assert cfg.max_new_tokens == 48


def test_resolve_quantized_engine_switches_arm_off_the_x86_default():
    """Regression: measured on a Pi 5 (aarch64), 2026-09-03. torch defaults
    ``quantized.engine`` to 'x86' everywhere, and quantize_dynamic() fails
    with 'unknown architecure' (torch's own typo) on any other machine."""
    from handwriting_app.recognizer.trocr_torch_recognizer import (
        resolve_quantized_engine,
    )

    supported = ("qnnpack", "onednn", "x86", "fbgemm")
    assert resolve_quantized_engine("x86", "aarch64", supported) == "qnnpack"
    assert resolve_quantized_engine("x86", "arm64", supported) == "qnnpack"
    assert resolve_quantized_engine("x86", "armv7l", supported) == "qnnpack"


def test_resolve_quantized_engine_leaves_x86_machines_alone():
    from handwriting_app.recognizer.trocr_torch_recognizer import (
        resolve_quantized_engine,
    )

    supported = ("qnnpack", "onednn", "x86", "fbgemm")
    assert resolve_quantized_engine("x86", "x86_64", supported) is None
    assert resolve_quantized_engine("x86", "AMD64", supported) is None


def test_resolve_quantized_engine_leaves_a_non_default_choice_alone():
    """Only the broken default gets overridden — someone who already picked
    an engine on purpose is left alone."""
    from handwriting_app.recognizer.trocr_torch_recognizer import (
        resolve_quantized_engine,
    )

    supported = ("qnnpack", "onednn", "x86", "fbgemm")
    assert resolve_quantized_engine("qnnpack", "aarch64", supported) is None
    assert resolve_quantized_engine("onednn", "aarch64", supported) is None


def test_resolve_quantized_engine_declines_without_qnnpack_available():
    from handwriting_app.recognizer.trocr_torch_recognizer import (
        resolve_quantized_engine,
    )

    assert resolve_quantized_engine("x86", "aarch64", ("x86", "fbgemm")) is None


def test_speed_flags_parse():
    cfg = parse_args(["--beams", "4", "--quantize", "--max-tokens", "12"])
    assert (cfg.beams, cfg.quantize, cfg.max_new_tokens) == (4, True, 12)


def test_image_size_defaults_to_the_checkpoints_native_resolution():
    # 0 means "whatever the processor says", i.e. 384. Interpolating the
    # position embeddings to anything else is opt-in until it is measured.
    assert parse_args([]).image_size == 0
    assert parse_args(["--image-size", "224"]).image_size == 224


def test_build_recognizer_passes_speed_options_to_trocr(monkeypatch):
    import handwriting_app.recognizer as pkg
    import handwriting_app.recognizer.trocr_torch_recognizer as mod

    captured = {}

    class FakeTrocr(StubRecognizer):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

    monkeypatch.setattr(mod, "TrocrTorchRecognizer", FakeTrocr)
    monkeypatch.setattr(pkg, "resolve_model_dir", lambda *_a, **_k: "some/model")

    build_recognizer(
        AppConfig(
            backend="trocr-torch",
            beams=4,
            quantize=True,
            max_new_tokens=12,
            image_size=224,
        )
    )

    assert captured["num_beams"] == 4
    assert captured["quantize"] is True
    assert captured["max_new_tokens"] == 12
    assert captured["image_size"] == 224


# -- warmup --------------------------------------------------------------
def test_warmup_defaults_to_a_no_op():
    assert StubRecognizer().warmup() == 0.0


# -- pipeline split ------------------------------------------------------
def test_render_matches_what_run_feeds_the_recognizer():
    recognizer = StubRecognizer()
    pipeline = RecognitionPipeline(
        recognizer, PipelineConfig(segment=False, spellcheck=False)
    )
    ink = _ink()

    standalone = pipeline.render(ink)
    pipeline.run(ink)

    assert isinstance(standalone, Image.Image)
    assert standalone.size == recognizer.images[0].size


def test_render_honours_calibration_overrides():
    pipeline = RecognitionPipeline(
        StubRecognizer(),
        PipelineConfig(
            segment=False,
            render_pad=4,
            calibration=Calibration(render_pad=40, samples=1),
        ),
    )
    image = pipeline.render(_ink())
    # 60px of ink plus the calibrated 40px padding on each side.
    assert image.width == 60 + 2 * 40


def test_postprocess_without_spellcheck_is_identity():
    pipeline = RecognitionPipeline(
        StubRecognizer(), PipelineConfig(segment=False, spellcheck=False)
    )
    assert pipeline.postprocess("a n d") == "a n d"
