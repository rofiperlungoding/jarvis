"""JARVIS benchmark suite.

This package hosts long-running validation harnesses that are run as part
of release certification rather than per-PR CI (see design.md
§Wake-Word Validation and §Latency Budget). Each module exposes a
``main()`` entry point so CI workflows can invoke them with
``python -m benchmarks.<name>``.
"""

from __future__ import annotations

__all__: list[str] = []
