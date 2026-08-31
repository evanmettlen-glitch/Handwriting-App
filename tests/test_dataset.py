import json

from handwriting_app.dataset import (
    count_samples,
    iter_samples,
    load_sample,
    save_sample,
)
from handwriting_app.ink import Ink
from handwriting_app.prompts import load_prompts


def _ink():
    ink = Ink()
    s = ink.start_stroke()
    for x in range(0, 40, 4):
        s.add(x, 20 + (x % 8))
    ink.start_stroke().add(50, 10)
    return ink


def test_ink_dict_roundtrip():
    ink = _ink()
    restored = Ink.from_dict(ink.to_dict())
    assert restored.to_dict() == ink.to_dict()
    assert len(restored.strokes) == 2


def test_save_and_load_sample(tmp_path):
    path = save_sample(_ink(), "Hello, world!", tmp_path, stroke_width=9)
    assert path.exists()
    assert path.with_suffix(".png").exists()

    sample = load_sample(path)
    assert sample.label == "Hello, world!"
    assert sample.stroke_width == 9
    assert len(sample.ink.strokes) == 2

    manifest = (tmp_path / "manifest.jsonl").read_text().splitlines()
    assert json.loads(manifest[0])["label"] == "Hello, world!"


def test_count_and_iter(tmp_path):
    assert count_samples(tmp_path) == 0
    save_sample(_ink(), "one", tmp_path)
    save_sample(_ink(), "two", tmp_path)
    assert count_samples(tmp_path) == 2
    assert [s.label for s in iter_samples(tmp_path)] == ["one", "two"]


def test_filenames_are_indexed_and_slugged(tmp_path):
    p1 = save_sample(_ink(), "the quick brown fox", tmp_path)
    p2 = save_sample(_ink(), "2026", tmp_path)
    assert p1.name.startswith("0001_")
    assert p2.name.startswith("0002_")


def test_calibration_sidecar_is_not_treated_as_a_sample(tmp_path):
    """calibrate.py writes calibration.json into the samples dir."""
    from handwriting_app.calibration import Calibration
    from handwriting_app.calibration import save as save_calibration

    save_sample(_ink(), "real sample", tmp_path)
    save_calibration(Calibration(samples=1), tmp_path)

    assert count_samples(tmp_path) == 1
    assert [s.label for s in iter_samples(tmp_path)] == ["real sample"]


def test_unreadable_sample_is_skipped_not_fatal(tmp_path):
    save_sample(_ink(), "good", tmp_path)
    (tmp_path / "9999_broken.json").write_text("{}", encoding="utf-8")
    (tmp_path / "9998_garbage.json").write_text("not json", encoding="utf-8")
    assert [s.label for s in iter_samples(tmp_path)] == ["good"]


def test_user_flag_scopes_the_samples_dir():
    from handwriting_app.config import parse_args

    cfg = parse_args(["--train", "--user", "Alice B."])
    assert cfg.samples_dir.replace("\\", "/").endswith("data/samples/Alice_B.")


def test_builtin_prompts_load():
    prompts = load_prompts()
    assert len(prompts) > 100
    assert "the" in prompts
    assert all(not p.startswith("#") for p in prompts)
