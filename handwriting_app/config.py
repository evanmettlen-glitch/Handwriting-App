"""Command-line configuration for the app."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from handwriting_app.naming import user_slug


@dataclass
class AppConfig:
    backend: str = "auto"  # auto -> trocr if its model is present, else tesseract
    lang: str = "eng"
    psm: int = 7
    whitelist: str = ""
    model_dir: str = ""  # "" -> auto-discover (see handwriting_app/models.py)
    user: str = ""
    fullscreen: bool = False
    auto_recognize: bool = True
    auto_delay_ms: int = 1800
    stroke_width: int = 8
    render_pad: int = 32
    font_scale: float = 1.0
    clear_after_recognize: bool = True
    append_separator: str = " "
    # recognition pipeline
    segment: bool = True
    word_gap_ratio: float = 0.4
    deslant: bool = True
    spellcheck: bool = True
    spell_compound: bool = False
    personal_lexicon: bool = True
    # training mode
    train: bool = False
    samples_dir: str = "data/samples"
    prompts_file: str = ""
    freeform: bool = False
    enroll_target: int = 0  # 0 = use the full enrollment set length


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="handwriting-app",
        description="Touchscreen handwriting-to-text for Raspberry Pi 5.",
    )
    p.add_argument(
        "--backend",
        choices=["auto", "tesseract", "trocr"],
        default=AppConfig.backend,
        help="Recognition engine (default: auto).",
    )
    p.add_argument(
        "--lang",
        default=AppConfig.lang,
        help="Tesseract language(s), e.g. 'eng' or 'eng+deu'.",
    )
    p.add_argument(
        "--psm",
        type=int,
        default=AppConfig.psm,
        help="Tesseract page segmentation mode for line images (7=line, 13=raw line).",
    )
    p.add_argument(
        "--whitelist",
        default=AppConfig.whitelist,
        help="Restrict recognized output to these characters (tesseract only).",
    )
    p.add_argument(
        "--model-dir",
        default=AppConfig.model_dir,
        help="TrOCR ONNX model directory (default: auto-discover a personal or generic model).",
    )
    p.add_argument(
        "--user",
        default=AppConfig.user,
        help="Person to personalize for: loads models/<user>-onnx and their word list.",
    )
    p.add_argument(
        "--no-personal-lexicon",
        dest="personal_lexicon",
        action="store_false",
        help="Ignore words learned from collected samples.",
    )
    p.add_argument("--fullscreen", action="store_true", help="Start in fullscreen kiosk mode.")
    p.add_argument(
        "--no-auto",
        dest="auto_recognize",
        action="store_false",
        help="Disable auto-recognize after a writing pause.",
    )
    p.add_argument(
        "--auto-delay",
        type=int,
        default=AppConfig.auto_delay_ms,
        metavar="MS",
        help="Idle time before auto-recognize fires (default: 1800).",
    )
    p.add_argument("--stroke-width", type=int, default=AppConfig.stroke_width, help="Pen width in pixels.")
    p.add_argument(
        "--font-scale",
        type=float,
        default=AppConfig.font_scale,
        help="Scale factor for all UI text (e.g. 1.4 for small hi-dpi panels).",
    )
    p.add_argument(
        "--keep-ink",
        dest="clear_after_recognize",
        action="store_false",
        help="Do not clear the canvas after each recognition.",
    )
    p.add_argument(
        "--no-segment",
        dest="segment",
        action="store_false",
        help="Recognize the whole line at once instead of word by word.",
    )
    p.add_argument(
        "--word-gap-ratio",
        type=float,
        default=AppConfig.word_gap_ratio,
        help="Word-break gap as a fraction of writing height (default: 0.4).",
    )
    p.add_argument(
        "--no-deslant",
        dest="deslant",
        action="store_false",
        help="Do not straighten slanted writing before recognition.",
    )
    p.add_argument(
        "--no-spellcheck",
        dest="spellcheck",
        action="store_false",
        help="Do not correct output against an English dictionary.",
    )
    p.add_argument(
        "--spell-compound",
        action="store_true",
        help="Aggressive dictionary pass that also fixes wrong/missing spaces.",
    )

    train = p.add_argument_group("training mode")
    train.add_argument(
        "--train",
        action="store_true",
        help="Run data-collection mode instead of the recognizer.",
    )
    train.add_argument(
        "--samples-dir",
        default=AppConfig.samples_dir,
        help="Where training samples are read/written (default: data/samples). "
        "With --user, collection goes to <samples-dir>/<user>.",
    )
    train.add_argument(
        "--prompts-file",
        default=AppConfig.prompts_file,
        help="Custom prompt list (one word/phrase per line); implies --freeform.",
    )
    train.add_argument(
        "--freeform",
        action="store_true",
        help="Open-ended word list instead of the guided enrollment set.",
    )
    train.add_argument(
        "--enroll-target",
        type=int,
        default=AppConfig.enroll_target,
        metavar="N",
        help="Samples that count as 100%% on the enrollment bar (default: all ~40).",
    )
    return p


def parse_args(argv=None) -> AppConfig:
    args = build_parser().parse_args(argv)
    samples_dir = args.samples_dir
    if args.user:
        samples_dir = str(Path(samples_dir) / user_slug(args.user))
    return AppConfig(
        backend=args.backend,
        lang=args.lang,
        psm=args.psm,
        whitelist=args.whitelist,
        model_dir=args.model_dir,
        user=args.user,
        personal_lexicon=args.personal_lexicon,
        fullscreen=args.fullscreen,
        auto_recognize=args.auto_recognize,
        auto_delay_ms=args.auto_delay,
        stroke_width=args.stroke_width,
        font_scale=args.font_scale,
        clear_after_recognize=args.clear_after_recognize,
        segment=args.segment,
        word_gap_ratio=args.word_gap_ratio,
        deslant=args.deslant,
        spellcheck=args.spellcheck,
        spell_compound=args.spell_compound,
        train=args.train,
        samples_dir=samples_dir,
        prompts_file=args.prompts_file,
        freeform=args.freeform,
        enroll_target=args.enroll_target,
    )
