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

    def warmup(self) -> float:  # pragma: no cover - optional hook
        """Do any first-call work now, while the UI still says "loading".

        Returns the seconds spent. Backends with no lazy setup return 0.
        """
        return 0.0

    def close(self) -> None:  # pragma: no cover - optional hook
        pass
