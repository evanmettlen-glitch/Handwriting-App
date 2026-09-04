import pytest

from handwriting_app.config import AppConfig
from handwriting_app.recognizer import RecognitionError, build_recognizer


def test_unknown_backend_raises():
    with pytest.raises(RecognitionError):
        build_recognizer(AppConfig(backend="does-not-exist"))


def test_tesseract_missing_binary_raises(monkeypatch):
    import handwriting_app.recognizer.tesseract_recognizer as mod

    monkeypatch.setattr(mod.shutil, "which", lambda _name: None)
    with pytest.raises(RecognitionError) as excinfo:
        mod.TesseractRecognizer()
    assert "apt install tesseract-ocr" in str(excinfo.value)


def test_tesseract_builds_expected_args(monkeypatch):
    import handwriting_app.recognizer.tesseract_recognizer as mod

    monkeypatch.setattr(mod.shutil, "which", lambda _name: "/usr/bin/tesseract")

    captured = {}

    class _Result:
        stdout = b"hello world\n"
        stderr = b""

    def fake_run(args, **kwargs):
        captured["args"] = args
        return _Result()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    rec = mod.TesseractRecognizer(lang="eng", psm=7, whitelist="abc")
    from PIL import Image

    text = rec.recognize(Image.new("L", (40, 20), color=255))

    assert text == "hello world"
    assert "--psm" in captured["args"]
    assert "tessedit_char_whitelist=abc" in captured["args"]


def test_an_explicit_psm_is_used_even_for_word_images(monkeypatch):
    """Regression: segmentation is ON for tesseract, so every image arrives with
    hint="word" — and the word-mode override hardcoded psm 8, which meant --psm
    could never take effect in the default configuration."""
    import handwriting_app.recognizer.tesseract_recognizer as mod

    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        class R:
            returncode = 0
            stdout = b"hi"
            stderr = b""
        return R()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod.shutil, "which", lambda _n: "/usr/bin/tesseract")

    from PIL import Image

    image = Image.new("L", (40, 20), color=255)

    explicit = mod.TesseractRecognizer(lang="eng", psm=13)
    explicit.recognize(image, hint="word")
    assert "13" in captured["args"], captured["args"]

    # Left unset, the per-image choice still applies: 8 for a word.
    auto = mod.TesseractRecognizer(lang="eng")
    auto.recognize(image, hint="word")
    assert "8" in captured["args"], captured["args"]
    auto.recognize(image, hint="line")
    assert "7" in captured["args"], captured["args"]
