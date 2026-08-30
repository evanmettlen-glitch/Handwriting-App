"""Training mode: prompt the user to write words and save (ink, label) samples.

Launch with ``./run.sh --train``. Collect ~150-300 samples, then fine-tune with
``scripts/finetune_trocr.py`` on a machine with a GPU (or patience).
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont

from handwriting_app.canvas_widget import InkCanvas
from handwriting_app.config import AppConfig
from handwriting_app.dataset import count_samples, save_sample
from handwriting_app.prompts import load_prompts

_BG = "#1e1e1e"
_FG = "#f0f0f0"


class TrainingApp(tk.Tk):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.cfg = config
        self.samples_dir = config.samples_dir

        try:
            self.prompts = load_prompts(config.prompts_file or None)
        except OSError as exc:
            raise SystemExit(f"Could not read prompts: {exc}")
        if not self.prompts:
            raise SystemExit("Prompt list is empty.")

        self.saved = count_samples(self.samples_dir)
        self.index = min(self.saved, len(self.prompts))

        self.title("Handwriting → Text · training")
        self.configure(bg=_BG)
        self.geometry("1024x600")
        self.minsize(800, 480)

        self._build_fonts()
        self._build_ui()
        self._bind_keys()
        if config.fullscreen:
            self._set_fullscreen(True)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self._show_current()

    # -- setup ---------------------------------------------------------------
    def _build_fonts(self) -> None:
        scale = self.cfg.font_scale
        self.f_prompt = tkfont.Font(family="DejaVu Sans", size=max(20, int(34 * scale)), weight="bold")
        self.f_button = tkfont.Font(family="DejaVu Sans", size=max(9, int(15 * scale)), weight="bold")
        self.f_meta = tkfont.Font(family="DejaVu Sans", size=max(9, int(12 * scale)))

    def _button(self, parent, text, command, bg="#3a3a3a"):
        return tk.Button(
            parent, text=text, command=command, font=self.f_button,
            bg=bg, fg="white", activebackground="#565656", activeforeground="white",
            relief="flat", bd=0, highlightthickness=0, padx=10, pady=14,
        )

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = tk.Frame(self, bg=_BG)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(10, 4))
        self.progress = tk.Label(header, text="", font=self.f_meta, bg=_BG, fg="#9a9a9a", anchor="w")
        self.progress.pack(side="left")
        self.prompt_label = tk.Label(self, text="", font=self.f_prompt, bg=_BG, fg=_FG)
        self.prompt_label.grid(row=0, column=0, columnspan=2, pady=(28, 6))

        self.canvas = InkCanvas(self, stroke_width=self.cfg.stroke_width)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=(12, 6), pady=8)

        panel = tk.Frame(self, bg=_BG)
        panel.grid(row=1, column=1, sticky="ns", padx=(6, 12), pady=8)
        self.btn_save = self._button(panel, "Save & next", self._save_and_next, bg="#2f9e44")
        widgets = [
            self.btn_save,
            self._button(panel, "Skip", self._skip),
            self._button(panel, "Undo stroke", self._undo),
            self._button(panel, "Clear pad", self.canvas.clear),
            self._button(panel, "Exit", self.destroy, bg="#555555"),
        ]
        for widget in widgets:
            widget.pack(fill="x", pady=4)

        self.status = tk.Label(
            self, text=f"Saving to {Path(self.samples_dir).resolve()}",
            font=self.f_meta, bg=_BG, fg="#9a9a9a", anchor="w",
        )
        self.status.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 8))

    def _bind_keys(self) -> None:
        self.bind("<Return>", lambda _e: self._save_and_next())
        self.bind("<Control-z>", lambda _e: self._undo())
        self.bind("<F11>", lambda _e: self._set_fullscreen(not self._is_fullscreen()))
        self.bind("<Escape>", lambda _e: self._set_fullscreen(False))
        self.bind("<Control-q>", lambda _e: self.destroy())

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

    # -- flow --------------------------------------------------------------
    def _finished(self) -> bool:
        return self.index >= len(self.prompts)

    def _show_current(self) -> None:
        total = len(self.prompts)
        if self._finished():
            self.prompt_label.config(text="All prompts done — thank you!")
            self.progress.config(text=f"{self.saved} samples saved")
            self.btn_save.config(state="disabled")
            return
        self.prompt_label.config(text=f"“{self.prompts[self.index]}”")
        self.progress.config(text=f"{self.index + 1} / {total}   ·   {self.saved} saved")

    def _save_and_next(self) -> None:
        if self._finished():
            return
        if self.canvas.ink.is_empty:
            self.status.config(text="Write the prompt first, then Save & next")
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
        self.saved += 1
        self.index += 1
        self.canvas.clear()
        self.status.config(text=f"Saved {path.name}")
        self._show_current()

    def _skip(self) -> None:
        if self._finished():
            return
        self.index += 1
        self.canvas.clear()
        self.status.config(text="Skipped")
        self._show_current()

    def _undo(self) -> None:
        if not self.canvas.undo_last_stroke():
            self.status.config(text="Nothing to undo")


def main(config: AppConfig) -> None:
    TrainingApp(config).mainloop()
