"""Command-line configuration for the app."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class AppConfig:
    backend: str = "auto"  # auto -> trocr if its model is present, else tesseract
    lang: str = "eng"
    psm: int = 7
    whitelist: str = ""
    model_dir: str = "models/trocr-small-handwritten-onnx"
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
    p.add_argument("--model-dir", default=AppConfig.model_dir, help="TrOCR ONNX model directory.")
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
    return p


def parse_args(argv=None) -> AppConfig:
    args = build_parser().parse_args(argv)
    return AppConfig(
        backend=args.backend,
        lang=args.lang,
        psm=args.psm,
        whitelist=args.whitelist,
        model_dir=args.model_dir,
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
    )
