"""Load the list of words/phrases to elicit during training mode."""

from __future__ import annotations

import importlib.resources as resources
from pathlib import Path
from typing import List, Optional


def load_prompts(path: Optional[str] = None) -> List[str]:
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        resource = resources.files("handwriting_app") / "data" / "prompts.txt"
        text = resource.read_text(encoding="utf-8")

    prompts: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            prompts.append(line)
    return prompts
