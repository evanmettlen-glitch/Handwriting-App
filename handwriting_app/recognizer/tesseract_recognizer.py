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

    @staticmethod
    def _preprocess(image: Image.Image) -> Image.Image:
        """Make the ink look as much like clean printed text as possible.

        Tesseract's LSTM engine wants dark text on a white background at a
        generous size. Upscale small images, stretch contrast, then binarize.
        """
        gray = ImageOps.grayscale(image)
        gray = ImageOps.autocontrast(gray, cutoff=1)

        target_height = 160
        if gray.height < target_height:
            scale = target_height / gray.height
            gray = gray.resize(
                (max(1, round(gray.width * scale)), target_height), Image.LANCZOS
            )

        binarized = gray.point(lambda p: 0 if p < 175 else 255)
        return ImageOps.expand(binarized, border=28, fill=255)

    def recognize(self, image: Image.Image, *, hint: str = "line") -> str:
        gray = self._preprocess(image)
        psm = 8 if hint == "word" else self.psm  # 8 = treat image as a single word
        args = [
            self.binary,
            "-",  # read image from stdin
            "-",  # write text to stdout
            "-l",
            self.lang,
            "--psm",
            str(psm),
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
