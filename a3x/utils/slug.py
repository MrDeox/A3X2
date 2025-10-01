"""Slug generation helpers shared across the project."""

from __future__ import annotations

import re

__all__ = ["slugify"]


def slugify(value: str, *, separator: str = "_", fallback: str = "nova_skill") -> str:
    """Normalize ``value`` into a filesystem-safe slug.

    Parameters
    ----------
    value:
        Raw text that should be converted into a slug.
    separator:
        Character used to join slug fragments. Defaults to ``_``.
    fallback:
        Fallback slug returned when ``value`` does not contain any
        alphanumeric characters. Defaults to ``"nova_skill"`` to
        preserve previous behaviour within the project.
    """

    slug = re.sub(r"[^a-z0-9]+", separator, value.lower()).strip(separator)
    return slug or fallback
