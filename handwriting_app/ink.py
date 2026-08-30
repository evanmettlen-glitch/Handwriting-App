"""Stroke capture model and rasterization to a clean image for recognition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw

Point = Tuple[float, float]


@dataclass
class Stroke:
    """A single pen-down .. pen-up trace."""

    points: List[Point] = field(default_factory=list)

    def add(self, x: float, y: float) -> None:
        self.points.append((float(x), float(y)))

    def __len__(self) -> int:
        return len(self.points)


@dataclass
class Ink:
    """An ordered collection of strokes drawn on the canvas."""

    strokes: List[Stroke] = field(default_factory=list)

    def start_stroke(self) -> Stroke:
        stroke = Stroke()
        self.strokes.append(stroke)
        return stroke

    @property
    def is_empty(self) -> bool:
        return not any(len(s) for s in self.strokes)

    def clear(self) -> None:
        self.strokes.clear()

    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        xs = [x for s in self.strokes for x, _ in s.points]
        ys = [y for s in self.strokes for _, y in s.points]
        if not xs:
            return None
        return min(xs), min(ys), max(xs), max(ys)

    def render(
        self,
        *,
        pad: int = 32,
        stroke_width: int = 8,
        supersample: int = 2,
        max_width: int = 1600,
    ) -> Optional[Image.Image]:
        """Rasterize the strokes as dark ink on a white background.

        Returns ``None`` when there is nothing to draw. The image is cropped to
        the ink bounding box plus ``pad`` pixels and drawn at ``supersample``x
        then downscaled, which gives smooth anti-aliased strokes.
        """
        box = self.bounds()
        if box is None:
            return None

        x0, y0, x1, y1 = box
        width = max(int(x1 - x0) + 2 * pad, 8)
        height = max(int(y1 - y0) + 2 * pad, 8)

        s = max(1, int(supersample))
        canvas = Image.new("L", (width * s, height * s), color=255)
        draw = ImageDraw.Draw(canvas)
        radius = stroke_width * s / 2.0

        for stroke in self.strokes:
            if len(stroke) == 0:
                continue
            pts = [
                ((px - x0 + pad) * s, (py - y0 + pad) * s) for px, py in stroke.points
            ]
            if len(pts) == 1:
                cx, cy = pts[0]
                draw.ellipse(
                    [cx - radius, cy - radius, cx + radius, cy + radius], fill=0
                )
                continue
            draw.line(pts, fill=0, width=max(1, stroke_width * s), joint="curve")
            # Round off the stroke ends so they don't look chopped.
            for cx, cy in (pts[0], pts[-1]):
                draw.ellipse(
                    [cx - radius, cy - radius, cx + radius, cy + radius], fill=0
                )

        if s != 1:
            canvas = canvas.resize((width, height), Image.LANCZOS)
        if canvas.width > max_width:
            ratio = max_width / canvas.width
            canvas = canvas.resize(
                (max_width, max(1, round(canvas.height * ratio))), Image.LANCZOS
            )
        return canvas
