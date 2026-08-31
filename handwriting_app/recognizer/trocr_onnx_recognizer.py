"""Neural backend: Microsoft TrOCR (handwritten) running on ONNX Runtime.

Much stronger than tesseract on real handwriting, at the cost of a large
dependency footprint and ~1-5 s per line on the Pi 5 CPU. Export the model once
with ``python -m scripts.export_trocr_onnx`` (add ``--quantize`` for int8).
"""

from __future__ import annotations

import glob
import os

from PIL import Image

from .base import RecognitionError, Recognizer

_COMPONENTS = ("encoder_model", "decoder_model", "decoder_with_past_model")


class TrocrOnnxRecognizer(Recognizer):
    def __init__(
        self,
        model_dir: str,
        max_new_tokens: int = 64,
        num_threads: int = 4,
    ) -> None:
        self.model_dir = model_dir
        self.name = f"trocr:{os.path.basename(os.path.normpath(model_dir))}"
        self.max_new_tokens = max_new_tokens
        try:
            from optimum.onnxruntime import ORTModelForVision2Seq
            from transformers import TrOCRProcessor
        except ImportError as exc:
            raise RecognitionError(
                "The 'trocr' backend needs extra packages. Install them with:\n"
                "  pip install -r requirements-trocr.txt"
            ) from exc

        if not os.path.isdir(model_dir):
            raise RecognitionError(
                f"Model directory '{model_dir}' does not exist.\n"
                "Create it once with:\n"
                "  python -m scripts.export_trocr_onnx"
            )

        file_names = self._pick_onnx_files(model_dir)
        session_options = self._session_options(num_threads)

        try:
            self._processor = TrOCRProcessor.from_pretrained(model_dir, use_fast=False)
            self._model = ORTModelForVision2Seq.from_pretrained(
                model_dir,
                use_io_binding=False,
                session_options=session_options,
                **file_names,
            )
        except Exception as exc:  # noqa: BLE001 - surface any load failure to the UI
            raise RecognitionError(f"failed to load TrOCR model: {exc}") from exc

    @staticmethod
    def _pick_onnx_files(model_dir: str) -> dict:
        """Prefer *_quantized.onnx when a full quantized set is present."""
        if all(
            glob.glob(os.path.join(model_dir, f"{name}_quantized.onnx"))
            for name in _COMPONENTS
        ):
            return {
                "encoder_file_name": "encoder_model_quantized.onnx",
                "decoder_file_name": "decoder_model_quantized.onnx",
                "decoder_with_past_file_name": "decoder_with_past_model_quantized.onnx",
            }
        return {}

    @staticmethod
    def _session_options(num_threads: int):
        try:
            from onnxruntime import GraphOptimizationLevel, SessionOptions

            options = SessionOptions()
            options.intra_op_num_threads = max(1, num_threads)
            options.graph_optimization_level = (
                GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            return options
        except Exception:  # noqa: BLE001
            return None

    def recognize(self, image: Image.Image, *, hint: str = "line") -> str:
        try:
            pixel_values = self._processor(
                images=image.convert("RGB"), return_tensors="pt"
            ).pixel_values
            generated_ids = self._model.generate(
                pixel_values=pixel_values, max_new_tokens=self.max_new_tokens
            )
            text = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
        except Exception as exc:  # noqa: BLE001
            raise RecognitionError(f"TrOCR inference failed: {exc}") from exc
        return text.strip()
