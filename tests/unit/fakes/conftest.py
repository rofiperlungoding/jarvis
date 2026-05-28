"""Pytest fixture re-exports for unit tests of :mod:`tests.fakes`.

The fixture itself lives next to the fake (in
:mod:`tests.fakes.fake_platform`) so the fake's public surface stays in
one place; this conftest re-exports it under the name pytest looks up so
unit tests in this package can request it as a parameter.
"""

from __future__ import annotations

from tests.fakes.fake_platform import fake_platform_adapter

__all__ = ["fake_platform_adapter"]
