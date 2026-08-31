"""TrOCR running directly on PyTorch — no ONNX, no optimum, no export step.

This is the low-friction path: if ``torch`` and ``transformers`` import, this
works. ``model_dir`` may be a local fine-tuned folder or a Hugging Face model id
(downloaded and cached on first use).

Slower than the ONNX backend (~3-8 s/line on a Pi 5 CPU vs ~1-5 s), but it has
no version-pinning minefield, so it is the default when no ONNX model is present.
"""

from __future__ import annotations

import os

from PIL import Image

from .base import RecognitionError, Recognizer


class TrocrTorchRecognizer(Recognizer):
    def __init__(
        self,
        model_dir: str,
        max_new_tokens: int = 48,
        num_threads: int = 4,
    ) -> None:
        self.model_dir = model_dir
        self.max_new_tokens = max_new_tokens
        self.name = f"trocr-torch:{os.path.basename(os.path.normpath(model_dir))}"

        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise RecognitionError(
                "The 'trocr' backend needs torch + transformers:\n"
                "  pip install -r requirements-trocr.txt"
            ) from exc

        self._torch = torch
        # The Pi 5 has 4 cores; leave the GUI thread some room.
        torch.set_num_threads(max(1, num_threads))

        try:
            self._processor = TrOCRProcessor.from_pretrained(model_dir, use_fast=False)
            self._model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        except Exception as exc:  # noqa: BLE001 - surface load failures to the UI
            raise RecognitionError(
                f"failed to load TrOCR model from {model_dir!r}: {exc}"
            ) from exc

        self._model.eval()

    def recognize(self, image: Image.Image, *, hint: str = "line") -> str:
        try:
            pixel_values = self._processor(
                images=image.convert("RGB"), return_tensors="pt"
            ).pixel_values
            with self._torch.inference_mode():
                generated_ids = self._model.generate(
                    pixel_values, max_new_tokens=self.max_new_tokens
                )
            text = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
        except Exception as exc:  # noqa: BLE001
            raise RecognitionError(f"TrOCR inference failed: {exc}") from exc
        return text.strip()
