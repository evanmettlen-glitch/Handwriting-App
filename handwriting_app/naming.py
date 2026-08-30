"""Shared helpers for turning a user-supplied name into a safe path component."""

from __future__ import annotations

import re

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def user_slug(name: str) -> str:
    slug = _UNSAFE.sub("_", name).strip("_")
    return slug or "user"
