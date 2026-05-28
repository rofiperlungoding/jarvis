"""Pytest fixtures for the in-memory test doubles in :mod:`tests.fakes`.

Every fake under this package gets a matching fixture here so test
modules can list the fixture name as a parameter without having to
import either the fixture or the fake explicitly. Each fixture
constructs a fresh instance per test so call recordings never leak.
"""

from __future__ import annotations

import pytest
from tests.fakes.fake_platform import FakePlatformAdapter


@pytest.fixture
def fake_platform_adapter() -> FakePlatformAdapter:
    """Return a freshly-constructed :class:`FakePlatformAdapter` for one test.

    Tests that need a different ``brightness_value`` /
    ``next_process_handle`` / ``next_script_result`` SHOULD construct
    the fake directly rather than mutating the fixture. The toggles
    that *are* expected to be flipped during a test
    (:meth:`FakePlatformAdapter.force_unsupported`,
    :meth:`FakePlatformAdapter.force_error`) operate on the fixture
    instance in-place.
    """
    return FakePlatformAdapter()
