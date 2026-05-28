"""``python -m jarvis`` console entry point.

Forwards to :func:`jarvis.app.main`, which performs the asyncio
bootstrap, runs the three concurrent loops, and translates
``KeyboardInterrupt`` into a graceful shutdown. Keeping the module
trivially small means the console behaviour matches the
``[project.scripts] jarvis`` entry in ``pyproject.toml``: both go
through the exact same ``main`` function.
"""

from __future__ import annotations

import sys

from jarvis.app import main

if __name__ == "__main__":  # pragma: no cover - exercised via ``python -m``
    sys.exit(main())
