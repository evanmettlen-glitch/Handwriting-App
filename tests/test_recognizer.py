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
