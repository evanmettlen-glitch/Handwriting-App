"""A Tkinter canvas that captures pen/finger strokes into an :class:`Ink`."""

from __future__ import annotations

import tkinter as tk
from typing import Callable, Optional

from handwriting_app.ink import Ink, Stroke


class InkCanvas(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        *,
        stroke_width: int = 8,
        ink_color: str = "#0a0a0a",
        bg: str = "#fdfdfb",
        baseline: bool = True,
        on_stroke_start: Optional[Callable[[], None]] = None,
        on_stroke_end: Optional[Callable[[], None]] = None,
        **frame_kw,
    ) -> None:
        super().__init__(master, **frame_kw)
        self.stroke_width = stroke_width
        self.ink_color = ink_color
        self.baseline = baseline
        self._on_stroke_start = on_stroke_start
        self._on_stroke_end = on_stroke_end

        self.ink = Ink()
        self._active: Optional[Stroke] = None
        self._last: Optional[tuple[float, float]] = None

        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, cursor="pencil")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_down)
        self.canvas.bind("<B1-Motion>", self._on_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_up)
        self.canvas.bind("<Configure>", lambda _e: self._draw_guides())

    # -- guides -----------------------------------------------------------------
    def _draw_guides(self) -> None:
        self.canvas.delete("guide")
        if not self.baseline:
            return
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        y = int(h * 0.72)
        self.canvas.create_line(
            20, y, max(21, w - 20), y, fill="#dcdcd6", width=2, tags="guide"
        )
        self.canvas.tag_lower("guide")

    # -- pointer events -------------------------------------------------------
    def _on_down(self, event: tk.Event) -> None:
        if self._on_stroke_start:
            self._on_stroke_start()
        self._active = self.ink.start_stroke()
        self._active.add(event.x, event.y)
        self._last = (event.x, event.y)

    def _on_move(self, event: tk.Event) -> None:
        if self._active is None or self._last is None:
            return
        self._active.add(event.x, event.y)
        x0, y0 = self._last
        self.canvas.create_line(
            x0,
            y0,
            event.x,
            event.y,
            fill=self.ink_color,
            width=self.stroke_width,
            capstyle="round",
            joinstyle="round",
            tags="ink",
        )
        self._last = (event.x, event.y)

    def _on_up(self, event: tk.Event) -> None:
        if self._active is None:
            return
        if len(self._active) == 1:
            r = self.stroke_width / 2
            self.canvas.create_oval(
                event.x - r,
                event.y - r,
                event.x + r,
                event.y + r,
                fill=self.ink_color,
                outline=self.ink_color,
                tags="ink",
            )
        self._active = None
        self._last = None
        if self._on_stroke_end:
            self._on_stroke_end()

    # -- api ----------------------------------------------------------------
    def clear(self) -> None:
        self.ink.clear()
        self.canvas.delete("ink")
