"""TrOCR running directly on PyTorch — no ONNX, no optimum, no export step.

This is the low-friction path: if ``torch`` and ``transformers`` import, this
works. ``model_dir`` may be a local fine-tuned folder or a Hugging Face model id
(downloaded and cached on first use).

Speed on a Pi 5 CPU is the main constraint. Four knobs matter, in order:

* ``num_beams`` — the TrOCR checkpoints ship ``num_beams=4`` in their
  generation config. Beam search runs the decoder (and the cross-attention over
  all 577 encoder patches) four times over. Greedy is the default here.
* ``quantize`` — dynamic int8 on every ``nn.Linear``, which is nearly the whole
  model. Roughly halves inference time; costs some accuracy, so measure with
  ``python -m scripts.bench_latency`` before turning it on for good.
* ``image_size`` — the encoder's cost is fixed per image and set entirely by how
  many patches it gets. The checkpoints run at 384x384, which is 577 patches at
  a patch size of 16; 224x224 is 197, about a third of the work. The position
  embeddings have to be interpolated to do it, so it is off by default and
  measured like everything else.
* the model itself — ``microsoft/trocr-small-handwritten`` is ~5x less compute
  than the base checkpoint. Pass it with ``--model-dir``.

None of those make the wait shorter than one encoder pass plus one token at a
time, so this also *streams*: the decoder hands back each token as it is
produced, and the UI shows the line assembling instead of a dead spinner. It
costs nothing — the tokens were being generated anyway.

``warmup()`` runs one throwaway inference so the lazy allocation, thread-pool
spin-up, and kernel selection all happen while the UI still says "loading"
rather than on the user's first real line. That same pass is what probes which
optional ``generate`` kwargs this transformers build accepts, so ``name`` and
``streaming`` only read true afterwards — on a Pi an extra encoder run is
seconds, and it is not worth spending two.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

from PIL import Image

from .base import RecognitionError, Recognizer

# The size the TrOCR checkpoints were trained at. Anything else needs the
# position embeddings interpolated, which not every transformers version does.
NATIVE_IMAGE_SIZE = 384


class _Streamer:
    """The transformers streamer protocol, forwarding partial text to a callback.

    ``generate`` calls :meth:`put` with the prompt first and then with each new
    token. Decoding the whole accumulation every time is wasteful in principle
    and free in practice — the cap is 48 tokens, and the decoder step it runs
    between calls is orders of magnitude slower.
    """

    def __init__(self, processor, on_text: Callable[[str], None]) -> None:
        self._processor = processor
        self._on_text = on_text
        self._ids: list[int] = []
        self._first = True

    def put(self, value) -> None:
        ids = value.reshape(-1).tolist() if hasattr(value, "reshape") else list(value)
        self._ids.extend(int(i) for i in ids)
        if self._first:  # the decoder start token carries no text
            self._first = False
            return
        try:
            text = self._processor.batch_decode(
                [self._ids], skip_special_tokens=True
            )[0]
        except Exception:  # noqa: BLE001 - a preview is never worth an exception
            return
        self._on_text(text.strip())

    def end(self) -> None:
        pass


class TrocrTorchRecognizer(Recognizer):
    def __init__(
        self,
        model_dir: str,
        max_new_tokens: int = 48,
        num_threads: int = 4,
        num_beams: int = 1,
        quantize: bool = False,
        image_size: int = 0,
    ) -> None:
        self.model_dir = model_dir
        self.max_new_tokens = max_new_tokens
        self.num_beams = max(1, num_beams)
        self.quantized = False
        # 0 means the checkpoint's native size; set after probing support.
        self.image_size = 0
        self.streaming = False
        # Asking for 384 is asking for the default, not for a resize that could
        # then be reported as unsupported.
        self._requested_size = 0 if image_size == NATIVE_IMAGE_SIZE else image_size
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

    # -- setup ------------------------------------------------------------
    @property
    def name(self) -> str:
        """Settles once :meth:`warmup` has probed what this build supports."""
        return self._build_name()

    def _build_name(self) -> str:
        base = os.path.basename(os.path.normpath(self.model_dir))
        extras = []
        if self.num_beams > 1:
            extras.append(f"beams {self.num_beams}")
        if self.quantized:
            extras.append("int8")
        if self.image_size:
            extras.append(f"{self.image_size}px")
        elif self._requested_size:
            extras.append("resize unsupported")
        suffix = f" [{', '.join(extras)}]" if extras else ""
        return f"trocr-torch:{base}{suffix}"

    def _probe_options(self) -> None:
        """Settle which optional ``generate`` kwargs this build accepts.

        Streaming and a non-native input size are both version-dependent, and
        neither should fail on the user's first real line. Beam search also
        rules streaming out, which the probe discovers rather than assumes.

        Each probe is a real inference — the encoder runs whatever the image —
        so this doubles as the warm-up rather than costing a second pass. One
        probe answers for both options; a second runs only if the pair failed,
        because streaming is the half worth keeping.
        """
        wanted = self._requested_size
        if wanted and self._probe(size=wanted, stream=True):
            self.image_size = wanted
            self.streaming = True
            return
        self.streaming = self._probe(size=0, stream=True)

    def _probe(self, *, size: int, stream: bool) -> bool:
        blank = Image.new("L", (320, 64), color=255)
        kwargs = {}
        if size:
            kwargs["interpolate_pos_encoding"] = True
        if stream:
            kwargs["streamer"] = _Streamer(self._processor, lambda _text: None)
        try:
            with self._torch.inference_mode():
                # 4 tokens, not 1: the decode loop wants warming too.
                self._model.generate(
                    self._pixels(blank, size), max_new_tokens=4, **kwargs
                )
        except Exception:  # noqa: BLE001 - an unsupported option is not an error
            return False
        return True

    def _pixels(self, image: Image.Image, size: int = 0):
        """Preprocess to pixel values, optionally at a smaller input size."""
        kwargs = {"size": {"height": size, "width": size}} if size else {}
        return self._processor(
            images=image.convert("RGB"), return_tensors="pt", **kwargs
        ).pixel_values

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
        """Probe the optional kwargs and pay the first-inference cost, together.

        Returns the seconds spent. Both jobs need a throwaway inference, and on
        a Pi that is several seconds each, so they are deliberately the same
        pass: probing separately used to add a whole encoder run to startup.
        ``name`` only reads true once this has run.
        """
        started = time.perf_counter()
        self._probe_options()
        return time.perf_counter() - started

    def _generate_kwargs(self) -> dict:
        return {"interpolate_pos_encoding": True} if self.image_size else {}

    # -- inference ---------------------------------------------------------
    def recognize(
        self,
        image: Image.Image,
        *,
        hint: str = "line",
        on_partial: Optional[Callable[[str], None]] = None,
    ) -> str:
        started = time.perf_counter()
        kwargs = self._generate_kwargs()
        if on_partial is not None and self.streaming:
            kwargs["streamer"] = _Streamer(self._processor, on_partial)
        try:
            with self._torch.inference_mode():
                generated_ids = self._model.generate(
                    self._pixels(image, self.image_size),
                    max_new_tokens=self.max_new_tokens,
                    **kwargs,
                )
            text = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
        except Exception as exc:  # noqa: BLE001
            raise RecognitionError(f"TrOCR inference failed: {exc}") from exc
        self.last_seconds = time.perf_counter() - started
        return text.strip()
