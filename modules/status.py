"""Tiny shared type describing whether a component is ready to use."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelStatus:
    available: bool
    message: str
