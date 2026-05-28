"""JARVIS AI Assistant.

Top-level package. Exposes :data:`__version__` for the auto-update
checker and any UI surface that wants to display the running build.
The version is the single source of truth, kept in lock-step with
``pyproject.toml`` and the Inno Setup script.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.3"
