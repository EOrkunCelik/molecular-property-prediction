"""Structured-ish logging setup.

Keeps logging configuration in one place instead of every module calling
`logging.basicConfig` independently (which is fragile — only the first call actually
takes effect, so ordering bugs are easy to introduce).
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Avoid duplicate handlers if configure_logging is called more than once (e.g. in tests).
    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Quiet down noisy third-party loggers at INFO level.
    logging.getLogger("uvicorn.access").setLevel("WARNING")
