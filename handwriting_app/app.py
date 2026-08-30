"""Tkinter UI: write on the canvas, get text out."""

from __future__ import annotations

import queue
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from tkinter import font as tkfont
from typing import Optional

from PIL import Image

from handwriting_app.canvas_widget import InkCanvas
from handwriting_app.config import AppConfig, parse_args
from handwriting_app.recognizer import RecognitionError, Recognizer, build_recognizer

_BG = "#1e1e1e"
_FG = "#f0f0f0"


class HandwritingApp(tk.Tk):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.cfg = config

        self.title("Handwriting → Text")
        self.configure(bg=_BG)
        self.geometry("1024x600")
        self.minsize(800, 480)

        self._results: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="recognizer")
        self._recognizer: Optional[Recognizer] = None
        self._busy = False
        self._pending_auto: Optional[str] = None

        self._build_fonts()
        self._build_ui()
        self._bind_keys()

        if config.fullscreen:
            self._set_fullscreen(True)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(50, self._load_recognizer_async)
        self.after(100, self._poll_results)

    # -- setup --------------------------------------------------------------
    def _build_fonts(self) -> None:
        scale = self.cfg.font_scale
        self.f_button = tkfont.Font(family="DejaVu Sans", size=max(9, int(15 * scale)), weight="bold")
        self.f_output = tkfont.Font(family="DejaVu Sans Mono", size=max(10, int(18 * scale)))
        self.f_status = tkfont.Font(family="DejaVu Sans", size=max(8, int(11 * scale)))

    def _mk_button(self, parent: tk.Misc, text: str, command, bg: str = "#3a3a3a") -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=self.f_button,
            bg=bg,
            fg="white",
            activebackground="#565656",
            activeforeground="white",
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=10,
            pady=14,
        )

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=3)
        self.rowconfigure(1, weight=2)

        self.canvas = InkCanvas(
            self,
            stroke_width=self.cfg.stroke_width,
            on_stroke_start=self._on_stroke_start,
            on_stroke_end=self._on_stroke_end,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(10, 6), pady=10)

        panel = tk.Frame(self, bg=_BG)
        panel.grid(row=0, column=1, rowspan=2, sticky="ns", padx=(6, 10), pady=10)

        self.btn_recognize = self._mk_button(panel, "Recognize", self._recognize_now, bg="#2d6cdf")
        self.btn_recognize.config(state="disabled")

        buttons = [
            self.btn_recognize,
            self._mk_button(panel, "Clear pad", self.canvas.clear),
            self._mk_button(panel, "Space", lambda: self._edit_output(" ")),
            self._mk_button(panel, "⌫  Back", self._backspace),
            self._mk_button(panel, "↵  Newline", lambda: self._edit_output("\n")),
            self._mk_button(panel, "Copy all", self._copy_all, bg="#2f9e44"),
            self._mk_button(panel, "Clear text", self._clear_output, bg="#c92a2a"),
        ]
        for button in buttons:
            button.pack(fill="x", pady=4)

        self.auto_var = tk.BooleanVar(value=self.cfg.auto_recognize)
        tk.Checkbutton(
            panel,
            text="Auto",
            variable=self.auto_var,
            font=self.f_button,
            bg=_BG,
            fg="white",
            selectcolor="#333333",
            activebackground=_BG,
            activeforeground="white",
            highlightthickness=0,
        ).pack(fill="x", pady=(12, 4))

        self.output = tk.Text(
            self,
            font=self.f_output,
            height=4,
            wrap="word",
            bg="#101010",
            fg=_FG,
            insertbackground=_FG,
            relief="flat",
            padx=10,
            pady=8,
        )
        self.output.grid(row=1, column=0, sticky="nsew", padx=(10, 6), pady=(0, 6))

        self.status = tk.Label(
            self,
            text="Starting…",
            font=self.f_status,
            anchor="w",
            bg=_BG,
            fg="#9a9a9a",
        )
        self.status.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 6))

    def _bind_keys(self) -> None:
        self.bind("<F11>", lambda _e: self._set_fullscreen(not self._is_fullscreen()))
        self.bind("<Escape>", lambda _e: self._set_fullscreen(False))
        self.bind("<Control-Return>", lambda _e: self._recognize_now())
        self.bind("<Control-l>", lambda _e: self.canvas.clear())

    # -- fullscreen helpers ------------------------------------------------
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

    # -- recognizer lifecycle --------------------------------------------
    def _load_recognizer_async(self) -> None:
        self._set_status(f"Loading '{self.cfg.backend}' backend…")
        self._pool.submit(self._load_recognizer)

    def _load_recognizer(self) -> None:
        try:
            recognizer = build_recognizer(self.cfg)
        except RecognitionError as exc:
            self._results.put(("fatal", str(exc)))
            return
        except Exception as exc:  # noqa: BLE001
            self._results.put(("fatal", f"{type(exc).__name__}: {exc}"))
            return
        self._recognizer = recognizer
        self._results.put(("ready", recognizer.name))

    # -- recognition -----------------------------------------------------
    def _recognize_now(self, auto: bool = False) -> None:
        self._pending_auto = None
        recognizer = self._recognizer
        if recognizer is None:
            if not auto:
                self._set_status("Backend is still loading…")
            return
        if self._busy:
            if not auto:
                self._set_status("Still working on the last one…")
            return
        image = self.canvas.ink.render(
            stroke_width=self.cfg.stroke_width, pad=self.cfg.render_pad
        )
        if image is None:
            if not auto:
                self._set_status("Nothing written yet")
            return

        self._busy = True
        self._set_status("Recognizing…")
        self._pool.submit(self._recognize_worker, recognizer, image)

    def _recognize_worker(self, recognizer: Recognizer, image: Image.Image) -> None:
        try:
            text = recognizer.recognize(image)
            self._results.put(("result", text))
        except RecognitionError as exc:
            self._results.put(("error", str(exc)))
        except Exception as exc:  # noqa: BLE001 - never let the worker die silently
            self._results.put(("error", f"{type(exc).__name__}: {exc}"))

    # -- result pump ---------------------------------------------------
    def _poll_results(self) -> None:
        try:
            while True:
                kind, payload = self._results.get_nowait()
                self._handle_result(kind, payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_results)

    def _handle_result(self, kind: str, payload: str) -> None:
        if kind == "ready":
            self.btn_recognize.config(state="normal")
            self._set_status(f"Ready · backend: {payload}")
        elif kind == "fatal":
            self._set_status("Backend failed to load")
            self.output.insert("end", f"[backend error]\n{payload}\n")
        elif kind == "result":
            self._busy = False
            self._append_recognized(payload.strip())
        elif kind == "error":
            self._busy = False
            self._set_status(f"Recognition error: {payload}")

    # -- output box helpers ------------------------------------------
    def _append_recognized(self, text: str) -> None:
        if not text:
            self._set_status("No text recognized — try writing larger")
            return
        current = self.output.get("1.0", "end-1c")
        separator = (
            self.cfg.append_separator
            if current and not current.endswith((" ", "\n"))
            else ""
        )
        self.output.insert("end", separator + text)
        self.output.see("end")
        self._set_status(f"Added: {text!r}")
        if self.cfg.clear_after_recognize:
            self.canvas.clear()

    def _edit_output(self, text: str) -> None:
        self.output.insert("end", text)
        self.output.see("end")

    def _backspace(self) -> None:
        if self.output.compare("end-1c", ">", "1.0"):
            self.output.delete("end-2c", "end-1c")

    def _clear_output(self) -> None:
        self.output.delete("1.0", "end")
        self._set_status("Text cleared")

    def _copy_all(self) -> None:
        text = self.output.get("1.0", "end-1c")
        self.clipboard_clear()
        if text:
            self.clipboard_append(text)
        self._set_status("Copied to clipboard" if text else "Nothing to copy")

    def _set_status(self, message: str) -> None:
        self.status.config(text=message)

    # -- stroke callbacks (auto-recognize) -------------------------
    def _on_stroke_start(self) -> None:
        if self._pending_auto is not None:
            self.after_cancel(self._pending_auto)
            self._pending_auto = None

    def _on_stroke_end(self) -> None:
        if self.auto_var.get():
            self._pending_auto = self.after(
                self.cfg.auto_delay_ms, lambda: self._recognize_now(auto=True)
            )

    # -- shutdown -----------------------------------------------------
    def _on_close(self) -> None:
        try:
            self._pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:  # pragma: no cover - Python < 3.9
            self._pool.shutdown(wait=False)
        self.destroy()


def main() -> None:
    config = parse_args()
    HandwritingApp(config).mainloop()


if __name__ == "__main__":
    main()
