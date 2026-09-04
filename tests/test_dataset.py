import json

from handwriting_app.dataset import (
    count_samples,
    iter_samples,
    load_sample,
    representative,
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


# -- representative(): an evenly-spaced subset, not the easiest slice -------


class _Labelled:
    def __init__(self, label):
        self.label = label


def test_representative_spreads_picks_across_the_whole_set():
    """Regression, measured 2026-09-03: bench_latency and eval_backend both
    used to slice the first N samples. iter_samples yields enrollment order —
    rote prompts, then single words, then multi-word lines — so that took the
    easiest samples in the set, and (with rote excluded from CER) scored
    accuracy on four one-word samples while every multi-word sample went
    unmeasured."""
    samples = [_Labelled(str(i)) for i in range(43)]
    picked = [s.label for s in representative(samples, 8)]

    assert len(picked) == 8
    assert picked[0] == "0"
    # Must reach the far end of the set, where the hard multi-word samples live.
    assert int(picked[-1]) > 30
    assert len(set(picked)) == 8  # no duplicates


def test_representative_returns_everything_when_unlimited():
    samples = [_Labelled(str(i)) for i in range(43)]
    assert len(representative(samples, 0)) == 43       # 0 means no limit
    assert len(representative(samples, 99)) == 43      # limit above the count
    assert len(representative(samples, 43)) == 43


def test_representative_never_indexes_past_the_end():
    for count in (1, 2, 3, 7):
        samples = [_Labelled(str(i)) for i in range(count)]
        for limit in (1, 2, 3, 5, 8, 100):
            picked = representative(samples, limit)
            assert len(picked) <= max(count, 1)
            assert all(p in samples for p in picked)
