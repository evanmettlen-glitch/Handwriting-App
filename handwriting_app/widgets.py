"""Small reusable Tk widgets."""

from __future__ import annotations

import tkinter as tk
from typing import Optional


class ProgressBar(tk.Frame):
    """A flat determinate progress bar with a centered percentage label."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        height: int = 30,
        fill: str = "#2f9e44",
        trough: str = "#333333",
        text_color: str = "#ffffff",
        font=None,
        bg: str = "#1e1e1e",
        **kw,
    ) -> None:
        super().__init__(master, bg=bg, **kw)
        self._fraction = 0.0
        self._fill = fill
        self._trough = trough
        self._text_color = text_color
        self._font = font
        self.canvas = tk.Canvas(self, height=height, highlightthickness=0, bg=bg)
        self.canvas.pack(fill="x", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._draw())

    def set(self, fraction: float) -> None:
        self._fraction = 0.0 if fraction < 0 else 1.0 if fraction > 1 else fraction
        self._draw()

    def _draw(self) -> None:
        c = self.canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w <= 1:
            return
        c.create_rectangle(0, 0, w, h, fill=self._trough, outline="")
        filled = int(w * self._fraction)
        if filled > 0:
            c.create_rectangle(0, 0, filled, h, fill=self._fill, outline="")
        c.create_text(
            w // 2,
            h // 2,
            text=f"{round(self._fraction * 100)}%",
            fill=self._text_color,
            font=self._font,
        )
