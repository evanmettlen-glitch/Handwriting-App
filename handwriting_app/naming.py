"""Shared helpers for turning a user-supplied name into a safe path component."""

from __future__ import annotations

import re

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def user_slug(name: str) -> str:
    """A single safe path component for ``--user``.

    Dots are kept because they are legitimate inside a name, which means the
    filter alone still lets "." and ".." through — and those are traversal, not
    names: ``data/samples/..`` resolves outside the samples root. Anything that
    is only dots is therefore rejected outright.
    """
    slug = _UNSAFE.sub("_", name).strip("_")
    if not slug or set(slug) == {"."}:
        return "user"
    return slug
