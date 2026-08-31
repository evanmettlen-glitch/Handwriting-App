"""TrOCR running directly on PyTorch — no ONNX, no optimum, no export step.

This is the low-friction path: if ``torch`` and ``transformers`` import, this
works. ``model_dir`` may be a local fine-tuned folder or a Hugging Face model id
(downloaded and cached on first use).

Speed on a Pi 5 CPU is the main constraint. Three knobs matter, in order:

* ``num_beams`` — the TrOCR checkpoints ship ``num_beams=4`` in their
  generation config. Beam search runs the decoder (and the cross-attention over
  all 577 encoder patches) four times over. Greedy is the default here.
* ``quantize`` — dynamic int8 on every ``nn.Linear``, which is nearly the whole
  model. Roughly halves inference time; costs some accuracy, so measure with
  ``python -m scripts.bench_latency`` before turning it on for good.
* the model itself — ``microsoft/trocr-small-handwritten`` is ~5x less compute
  than the base checkpoint. Pass it with ``--model-dir``.

``warmup()`` runs one throwaway inference so the lazy allocation, thread-pool
spin-up, and kernel selection all happen while the UI still says "loading"
rather than on the user's first real line.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from PIL import Image

from .base import RecognitionError, Recognizer


class TrocrTorchRecognizer(Recognizer):
    def __init__(
        self,
        model_dir: str,
        max_new_tokens: int = 48,
        num_threads: int = 4,
        num_beams: int = 1,
        quantize: bool = False,
    ) -> None:
        self.model_dir = model_dir
        self.max_new_tokens = max_new_tokens
        self.num_beams = max(1, num_beams)
        self.quantized = False
        # Wall time of the last recognize() call, for the status line.
        self.last_seconds: Optional[float] = None

        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise RecognitionError(
                "The 'trocr' backend needs torch + transformers:\n"
                "  pip install -r requirements-trocr.txt"
            ) from exc

        self._torch = torch
        # The Pi 5 has 4 Cortex-A76 cores and the GUI thread is idle while a
        # recognition runs, so use all of them.
        torch.set_num_threads(max(1, num_threads))

        try:
            self._processor = TrOCRProcessor.from_pretrained(model_dir, use_fast=False)
            self._model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        except Exception as exc:  # noqa: BLE001 - surface load failures to the UI
            raise RecognitionError(
                f"failed to load TrOCR model from {model_dir!r}: {exc}"
            ) from exc

        self._model.eval()
        self._configure_generation()
        if quantize:
            self._quantize()

        self.name = self._build_name()

    # -- setup ------------------------------------------------------------
    def _build_name(self) -> str:
        base = os.path.basename(os.path.normpath(self.model_dir))
        extras = []
        if self.num_beams > 1:
            extras.append(f"beams {self.num_beams}")
        if self.quantized:
            extras.append("int8")
        suffix = f" [{', '.join(extras)}]" if extras else ""
        return f"trocr-torch:{base}{suffix}"

    def _configure_generation(self) -> None:
        """Bake the decoding strategy into the model's generation config.

        Setting it once here (rather than passing kwargs per call) keeps
        transformers from warning about beam-only options — ``early_stopping``
        and ``length_penalty`` are meaningless under greedy decoding.
        """
        config = getattr(self._model, "generation_config", None)
        if config is None:  # pragma: no cover - very old transformers
            return
        config.num_beams = self.num_beams
        config.max_new_tokens = self.max_new_tokens
        if self.num_beams == 1:
            config.early_stopping = False
            config.length_penalty = 1.0

    def _quantize(self) -> None:
        """Swap every nn.Linear for a dynamic int8 one.

        TrOCR is almost entirely Linear layers, so this covers the encoder's
        attention/MLP blocks and the decoder alike. Weights are quantized once
        here; activations are quantized per call, which is why it needs no
        calibration data.
        """
        torch = self._torch
        try:
            from torch.ao.quantization import quantize_dynamic
        except ImportError:  # pragma: no cover - torch < 1.13 layout
            from torch.quantization import quantize_dynamic
        try:
            self._model = quantize_dynamic(
                self._model, {torch.nn.Linear}, dtype=torch.qint8
            )
        except Exception as exc:  # noqa: BLE001 - never block startup on this
            raise RecognitionError(f"int8 quantization failed: {exc}") from exc
        self.quantized = True

    def warmup(self) -> float:
        """Run one throwaway inference. Returns how long it took, in seconds."""
        blank = Image.new("L", (320, 64), color=255)
        started = time.perf_counter()
        try:
            pixel_values = self._processor(
                images=blank.convert("RGB"), return_tensors="pt"
            ).pixel_values
            with self._torch.inference_mode():
                self._model.generate(pixel_values, max_new_tokens=4)
        except Exception:  # noqa: BLE001 - a failed warmup is not fatal
            pass
        return time.perf_counter() - started

    # -- inference ---------------------------------------------------------
    def recognize(self, image: Image.Image, *, hint: str = "line") -> str:
        started = time.perf_counter()
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
        self.last_seconds = time.perf_counter() - started
        return text.strip()
