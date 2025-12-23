"""Tiny wrapper around stdlib logging so internal modules can grab a
consistent logger without repeating boilerplate.
"""
from __future__ import annotations

import logging
from typing import Any

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s : %(message)s"

logging.basicConfig(format=_LOG_FORMAT, level=logging.INFO)


def get_logger(name: str | None = None, **kwargs: Any) -> logging.Logger:  # noqa: D401 – factory
    """Return a module- or user‑named :class:`logging.Logger`."""
    return logging.getLogger(name)
