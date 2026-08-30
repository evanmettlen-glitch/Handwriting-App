from __future__ import annotations

import abc

from PIL import Image


class RecognitionError(RuntimeError):
    """Raised for any backend setup or inference failure.

    The message is shown to the user, so keep it actionable.
    """


class Recognizer(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def recognize(self, image: Image.Image, *, hint: str = "line") -> str:
        """Return the text read from a dark-ink-on-white image.

        ``hint`` is ``"line"`` or ``"word"`` — a backend may use it to tune
        segmentation (e.g. tesseract's page-segmentation mode).
        """

    def close(self) -> None:  # pragma: no cover - optional hook
        pass
