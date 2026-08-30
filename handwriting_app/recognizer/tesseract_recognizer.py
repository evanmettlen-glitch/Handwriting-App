"""OCR backend using the system ``tesseract`` binary.

Lightweight and fully offline. Best on neat block printing; weak on cursive.
"""

from __future__ import annotations

import io
import shutil
import subprocess

from PIL import Image, ImageOps

from .base import RecognitionError, Recognizer


class TesseractRecognizer(Recognizer):
    name = "tesseract"

    def __init__(
        self,
        lang: str = "eng",
        psm: int = 7,
        whitelist: str | None = None,
        binary: str = "tesseract",
        timeout: float = 30.0,
    ) -> None:
        self.lang = lang
        self.psm = psm
        self.whitelist = whitelist
        self.binary = binary
        self.timeout = timeout
        if shutil.which(binary) is None:
            raise RecognitionError(
                f"'{binary}' was not found on PATH.\n"
                "Install it with:  sudo apt install tesseract-ocr"
            )

    def recognize(self, image: Image.Image) -> str:
        gray = ImageOps.grayscale(image)
        args = [
            self.binary,
            "-",  # read image from stdin
            "-",  # write text to stdout
            "-l",
            self.lang,
            "--psm",
            str(self.psm),
            "--oem",
            "1",
        ]
        if self.whitelist:
            args += ["-c", f"tessedit_char_whitelist={self.whitelist}"]

        buf = io.BytesIO()
        gray.save(buf, format="PNG")
        try:
            proc = subprocess.run(
                args,
                input=buf.getvalue(),
                capture_output=True,
                timeout=self.timeout,
                check=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise RecognitionError("tesseract timed out") from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode(errors="replace").strip() or "unknown error"
            raise RecognitionError(f"tesseract failed: {detail}") from exc

        return proc.stdout.decode(errors="replace").strip()
