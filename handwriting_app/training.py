"""Training / enrollment mode.

``./run.sh --train`` runs a guided ~40-prompt enrollment (progress bar, timer,
time-remaining estimate, letter/digit coverage). It's designed to gather enough
handwriting to adapt the model in under five minutes.

``--freeform`` or ``--prompts-file FILE`` switches to the open-ended word list.
"""

from __future__ import annotations

import math
import time
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from typing import List, Sequence, Set

from handwriting_app.canvas_widget import InkCanvas
from handwriting_app.config import AppConfig
from handwriting_app.dataset import iter_samples, save_sample
from handwriting_app.enrollment import (
    ENROLLMENT_PROMPTS,
    Coverage,
    char_coverage,
    is_enrolled,
)
from handwriting_app.naming import user_slug
from handwriting_app.prompts import load_prompts
from handwriting_app.widgets import ProgressBar

_BG = "#1e1e1e"
_FG = "#f0f0f0"
_MUTED = "#9a9a9a"


def _first_undone(prompts: Sequence[str], done_labels: Set[str], start: int = 0) -> int:
    """Index of the first prompt from ``start`` on that is not already collected.

    Called on resume *and* after every advance, not just once at startup —
    that one-time-only version used to skip a leading run of done prompts and
    then walk the rest linearly, so a gap anywhere past the first undone prompt
    (e.g. one sample deleted from the middle of a finished set) re-served and
    re-saved every prompt after it, duplicating an entire completed enrollment
    for the cost of restoring one label.
    """
    index = start
    while index < len(prompts) and prompts[index] in done_labels:
        index += 1
    return index


class TrainingApp(tk.Tk):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.cfg = config
        self.samples_dir = config.samples_dir

        self.enroll = not (config.freeform or config.prompts_file)
        if self.enroll:
            self.prompts: List[str] = list(ENROLLMENT_PROMPTS)
            self.target = config.enroll_target or len(self.prompts)
        else:
            try:
                self.prompts = load_prompts(config.prompts_file or None)
            except (OSError, ValueError) as exc:
                # A non-UTF-8 prompts file raises UnicodeDecodeError, which is a
                # ValueError — not an OSError — and used to escape as a traceback.
                raise SystemExit(f"Could not read prompts: {exc}")
            self.target = config.enroll_target or len(self.prompts)
        if not self.prompts:
            raise SystemExit("Prompt list is empty.")

        # Resume within the fixed prompt sequence by skipping prompts whose exact
        # label was already collected (a returning user), not by raw file count.
        # _done_labels is kept live (grown on every save, consulted on every
        # advance) rather than computed once — see _first_undone.
        self._done_labels: Set[str] = {s.label for s in iter_samples(self.samples_dir)}
        self._prior = min(
            sum(1 for p in self.prompts if p in self._done_labels), self.target
        )
        self.index = _first_undone(self.prompts, self._done_labels)

        self.session_saved = 0
        self._start = time.monotonic()
        self._tick_job = None

        self.title("Handwriting → Text · enrollment")
        self.configure(bg=_BG)
        self.geometry("1024x600")
        self.minsize(800, 480)

        self._build_fonts()
        self._build_ui()
        self._bind_keys()
        if config.fullscreen:
            self._set_fullscreen(True)
        self.protocol("WM_DELETE_WINDOW", self._quit)

        self._refresh()
        self._tick()

    # -- setup ------------------------------------------------------------
    def _build_fonts(self) -> None:
        scale = self.cfg.font_scale
        self.f_prompt = tkfont.Font(family="DejaVu Sans", size=max(20, int(34 * scale)), weight="bold")
        self.f_button = tkfont.Font(family="DejaVu Sans", size=max(9, int(15 * scale)), weight="bold")
        self.f_meta = tkfont.Font(family="DejaVu Sans", size=max(9, int(12 * scale)))
        self.f_bar = tkfont.Font(family="DejaVu Sans", size=max(9, int(13 * scale)), weight="bold")

    def _button(self, parent, text, command, bg="#3a3a3a"):
        return tk.Button(
            parent, text=text, command=command, font=self.f_button,
            bg=bg, fg="white", activebackground="#565656", activeforeground="white",
            relief="flat", bd=0, highlightthickness=0, padx=10, pady=14,
        )

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        top = tk.Frame(self, bg=_BG)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(10, 2))
        top.columnconfigure(0, weight=1)

        self.bar = ProgressBar(top, font=self.f_bar, bg=_BG)
        self.bar.grid(row=0, column=0, columnspan=2, sticky="ew")

        self.count_label = tk.Label(top, text="", font=self.f_meta, bg=_BG, fg=_MUTED, anchor="w")
        self.count_label.grid(row=1, column=0, sticky="w", pady=(3, 0))
        self.time_label = tk.Label(top, text="", font=self.f_meta, bg=_BG, fg=_MUTED, anchor="e")
        self.time_label.grid(row=1, column=1, sticky="e", pady=(3, 0))
        self.coverage_label = tk.Label(top, text="", font=self.f_meta, bg=_BG, fg=_MUTED, anchor="w")
        self.coverage_label.grid(row=2, column=0, columnspan=2, sticky="w")

        self.prompt_label = tk.Label(self, text="", font=self.f_prompt, bg=_BG, fg=_FG)
        self.prompt_label.grid(row=1, column=0, columnspan=2, pady=(14, 6))

        self.canvas = InkCanvas(self, stroke_width=self.cfg.stroke_width)
        self.canvas.grid(row=2, column=0, sticky="nsew", padx=(12, 6), pady=8)

        panel = tk.Frame(self, bg=_BG)
        panel.grid(row=2, column=1, sticky="ns", padx=(6, 12), pady=8)
        self.btn_save = self._button(panel, "Save & next", self._save_and_next, bg="#2f9e44")
        for widget in (
            self.btn_save,
            self._button(panel, "Skip", self._skip),
            self._button(panel, "Undo stroke", self._undo),
            self._button(panel, "Clear pad", self.canvas.clear),
            self._button(panel, "Finish", self._quit, bg="#555555"),
        ):
            widget.pack(fill="x", pady=4)

        self.status = tk.Label(
            self, text=f"Saving to {Path(self.samples_dir).resolve()}",
            font=self.f_meta, bg=_BG, fg=_MUTED, anchor="w",
        )
        self.status.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 8))

    def _bind_keys(self) -> None:
        self.bind("<Return>", lambda _e: self._save_and_next())
        self.bind("<space>", lambda _e: self._save_and_next())
        self.bind("<Control-z>", lambda _e: self._undo())
        self.bind("<F11>", lambda _e: self._set_fullscreen(not self._is_fullscreen()))
        self.bind("<Escape>", lambda _e: self._set_fullscreen(False))
        self.bind("<Control-q>", lambda _e: self._quit())

    def _is_fullscreen(self) -> bool:
        try:
            return bool(self.attributes("-fullscreen"))
        except tk.TclError:
            return False

    def _set_fullscreen(self, value: bool) -> None:
        try:
            self.attributes("-fullscreen", value)
        except tk.TclError:
            pass

    # -- state ----------------------------------------------------------
    def _finished(self) -> bool:
        return self.index >= len(self.prompts)

    def _coverage(self) -> Coverage:
        return char_coverage(s.label for s in iter_samples(self.samples_dir))

    def _progress(self) -> int:
        return min(self.target, self._prior + self.session_saved)

    def _refresh(self) -> None:
        done = self._progress()
        self.bar.set(done / self.target if self.target else 0.0)
        self.count_label.config(
            text=f"{done} / {self.target}"
            + ("  —  goal reached" if done >= self.target else "")
        )

        enrolled = False
        if self.enroll:
            coverage = self._coverage()
            enrolled = is_enrolled(done, coverage, self.target)
            self.coverage_label.config(
                text=("enrolled ✓   " if enrolled else "") + coverage.summary()
            )
            self.btn_save.config(bg="#2d6cdf" if enrolled else "#2f9e44")
        else:
            self.coverage_label.config(text="")

        if self._finished():
            self.prompt_label.config(text="All prompts done — thank you!")
            self.btn_save.config(state="disabled")
        else:
            self.prompt_label.config(text=f"“{self.prompts[self.index]}”")

        if enrolled and self.session_saved:
            name = user_slug(self.cfg.user) if self.cfg.user else "personal"
            self.status.config(
                text=f"Enrolled ✓  ·  next: ./scripts/train_personal.sh {name}"
            )

    def _tick(self) -> None:
        elapsed = time.monotonic() - self._start
        text = _fmt_mmss(elapsed)
        remaining = max(0, self.target - self._progress())
        if remaining and self.session_saved >= 3:
            per = elapsed / self.session_saved
            text += f"   ·   ~{max(1, math.ceil(remaining * per / 60))} min left"
        elif not remaining:
            text += "   ·   done"
        self.time_label.config(text=text)
        self._tick_job = self.after(1000, self._tick)

    # -- actions ------------------------------------------------------
    def _save_and_next(self) -> None:
        if self._finished():
            return
        if self.canvas.ink.is_empty:
            self.status.config(text="Write the prompt first")
            return
        label = self.prompts[self.index]
        try:
            path = save_sample(
                self.canvas.ink.copy(), label, self.samples_dir,
                stroke_width=self.cfg.stroke_width,
            )
        except OSError as exc:
            self.status.config(text=f"Save failed: {exc}")
            return
        self.session_saved += 1
        self._done_labels.add(label)
        self.index = _first_undone(self.prompts, self._done_labels, self.index + 1)
        self.canvas.clear()
        self.status.config(text=f"Saved {path.name}")
        self._refresh()

    def _skip(self) -> None:
        if self._finished():
            return
        self.index = _first_undone(self.prompts, self._done_labels, self.index + 1)
        self.canvas.clear()
        self.status.config(text="Skipped")
        self._refresh()

    def _undo(self) -> None:
        if not self.canvas.undo_last_stroke():
            self.status.config(text="Nothing to undo")

    def _quit(self) -> None:
        if self._tick_job is not None:
            self.after_cancel(self._tick_job)
        self.destroy()


def _fmt_mmss(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def main(config: AppConfig) -> None:
    TrainingApp(config).mainloop()
