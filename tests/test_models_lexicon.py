from handwriting_app.dataset import save_sample
from handwriting_app.ink import Ink
from handwriting_app.lexicon import personal_word_counts
from handwriting_app.models import (
    DEFAULT_HF_MODEL,
    is_local,
    is_onnx_dir,
    resolve_model_dir,
)
from handwriting_app.naming import user_slug


def _ink():
    ink = Ink()
    ink.start_stroke().add(1, 1)
    ink.start_stroke().add(2, 2)
    return ink


def test_user_slug():
    assert user_slug("Evan M.") == "Evan_M."
    assert user_slug("  !!!  ") == "user"
    assert user_slug("sam") == "sam"


def test_resolve_model_dir_explicit_wins(tmp_path):
    assert resolve_model_dir("some/explicit/path", "evan") == "some/explicit/path"


def test_resolve_model_dir_prefers_user_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models" / "evan-onnx").mkdir(parents=True)
    (tmp_path / "models" / "trocr-small-handwritten-onnx").mkdir(parents=True)
    assert resolve_model_dir("", "evan") == "models/evan-onnx"
    assert resolve_model_dir("", "nobody") == "models/trocr-small-handwritten-onnx"


def test_resolve_model_dir_finds_non_onnx_user_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models" / "evan").mkdir(parents=True)
    assert resolve_model_dir("", "evan") == "models/evan"


def test_resolve_model_dir_falls_back_to_hf_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_model_dir("", "") == DEFAULT_HF_MODEL
    assert "/" in DEFAULT_HF_MODEL  # a hub id, not a local path


def test_is_onnx_dir(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    assert is_local(str(plain))
    assert not is_onnx_dir(str(plain))

    exported = tmp_path / "exported"
    exported.mkdir()
    (exported / "encoder_model.onnx").write_bytes(b"")
    assert is_onnx_dir(str(exported))

    assert not is_local("microsoft/trocr-base-handwritten")


def test_personal_word_counts_from_samples(tmp_path):
    save_sample(_ink(), "the meeting is at noon", tmp_path)
    save_sample(_ink(), "call Priya today", tmp_path)
    counts = personal_word_counts(str(tmp_path))
    assert counts["priya"] == 1
    assert counts["the"] == 1
    assert counts["meeting"] == 1
    assert "a" not in counts  # single letters skipped


def test_personal_word_counts_scans_user_subdirs(tmp_path):
    save_sample(_ink(), "alpha", tmp_path / "evan")
    save_sample(_ink(), "beta", tmp_path / "sam")
    counts = personal_word_counts(str(tmp_path))
    assert counts["alpha"] == 1 and counts["beta"] == 1
