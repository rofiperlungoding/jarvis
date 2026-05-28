"""Top-level pytest configuration shared by every test in :mod:`tests`.

This module exists for two reasons:

1. **Hypothesis profile baseline.** Task 21.1 requires that every
   property-based test in this repository run under a profile equivalent
   to ``@settings(max_examples=200, deadline=None)``. We register two
   profiles at import time:

   * ``"jarvis"`` — the production profile used by every test in CI.
     ``max_examples=200`` matches the task's baseline; ``deadline=None``
     prevents flaky timeout failures on the property tests in 21.2 ..
     21.16, which exercise the Memory_Store, Reminder_Service, and
     Dialog_Manager seams under ``hypothesis-jsonschema`` argument
     generation and can occasionally take longer than the
     library-default 200 ms per example.
   * ``"jarvis-ci"`` — a tighter variant (``max_examples=400``,
     ``HealthCheck.too_slow`` suppressed) used by the
     ``--hypothesis-profile=jarvis-ci`` invocation in
     ``.github/workflows/ci.yml``.

   Profiles are *registered* here but only *loaded* once via
   ``hypothesis.settings.load_profile``. Loading is the side effect
   that lets every ``@given`` test pick up the new defaults without
   each test having to wear an ``@settings(...)`` decorator.

2. **Cross-test fixtures.** Task 21.1 also calls out *"freezegun helpers"*
   as a candidate cross-test fixture. We expose :func:`frozen_clock` here
   so any property test that needs deterministic ``datetime.utcnow()``
   semantics — Property 5 (CP6 conversation determinism) and Property 10
   (CP13 reminder ordering) being the prime examples — can request the
   helper without re-importing :mod:`freezegun` in every test module.

The conftest deliberately stays *thin*: it neither imports the
:mod:`jarvis` package (so a totally broken install still surfaces a
collection error rather than a silent skip) nor mutates any global
state beyond Hypothesis profile registration.

The :data:`pytest_plugins` declaration registers the :mod:`tests.fakes`
modules that expose pytest fixtures (currently
:mod:`tests.fakes.fake_mcp_server` from task 22.2). Names of fixture
functions inside those modules MUST NOT start with ``pytest_`` because
pytest treats those as hook implementations.

Validates: Requirements 14.3, 14.4, 14.6
"""

from __future__ import annotations

from collections.abc import Iterator
import os
from typing import TYPE_CHECKING

from hypothesis import HealthCheck, settings
import pytest

if TYPE_CHECKING:  # pragma: no cover - import-time only
    # ``freezegun`` is a dev dependency; keep it under TYPE_CHECKING so
    # the conftest can still be imported in environments where freezegun
    # is missing (e.g., a half-installed venv) — the fixture itself
    # imports it lazily inside its body and fails with a clear pytest
    # error if it is unavailable.
    from freezegun.api import FrozenDateTimeFactory


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


# Register fake-server modules that expose pytest fixtures. Each entry is
# imported once by pytest at collection time; only fixtures explicitly
# decorated with ``@pytest.fixture`` / ``@pytest_asyncio.fixture`` become
# visible to test files.
pytest_plugins = [
    "tests.fakes.fake_mcp_server",
    "tests.fakes.fake_mistral_server",
]


# ---------------------------------------------------------------------------
# Hypothesis profile registration
# ---------------------------------------------------------------------------


# The default profile name for every property test in this repository.
# Tests pick this up automatically because we call
# ``settings.load_profile`` below.
JARVIS_PROFILE_NAME = "jarvis"

# A tighter CI-only profile, opt-in via
# ``pytest --hypothesis-profile=jarvis-ci``.
JARVIS_CI_PROFILE_NAME = "jarvis-ci"


# ``register_profile`` is idempotent in current Hypothesis versions
# (calling it twice with the same name silently overwrites), but we
# guard against an accidental double-load by checking the registry up
# front. This keeps the conftest safe to import inside test sub-packages
# (e.g., when pytest discovers tests under ``tests/property/`` first
# and then re-imports the top-level conftest for ``tests/unit/``).
def _register_profiles() -> None:
    """Register the ``jarvis`` and ``jarvis-ci`` Hypothesis profiles."""

    settings.register_profile(
        JARVIS_PROFILE_NAME,
        max_examples=200,
        deadline=None,
        # Strategy generation that goes through ``hypothesis-jsonschema``
        # is heavier than typical Hypothesis strategies; suppressing the
        # ``too_slow`` health check matches the documented escape hatch
        # for that library and is a no-op on fast machines.
        suppress_health_check=(HealthCheck.too_slow,),
    )

    settings.register_profile(
        JARVIS_CI_PROFILE_NAME,
        max_examples=400,
        deadline=None,
        suppress_health_check=(
            HealthCheck.too_slow,
            HealthCheck.data_too_large,
        ),
    )


_register_profiles()


# ``HYPOTHESIS_PROFILE`` lets CI override the active profile from the
# command line / environment without each test having to do anything
# different. Defaulting to ``jarvis`` covers the local-developer case.
_active_profile = os.environ.get("HYPOTHESIS_PROFILE", JARVIS_PROFILE_NAME)
settings.load_profile(_active_profile)


# ---------------------------------------------------------------------------
# Freezegun cross-test fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def frozen_clock() -> Iterator[FrozenDateTimeFactory]:
    """Yield a :class:`freezegun.api.FrozenDateTimeFactory` pinned to a fixed UTC instant.

    Property tests for the conversation-state determinism (CP6) and the
    reminder firing order (CP13) need the system clock to be stable
    across re-runs and across Hypothesis examples. Standardising on a
    single fixture keeps the freeze instant — ``2024-01-01T00:00:00Z`` —
    consistent across modules so failing examples are reproducible
    locally.

    ``freezegun`` is imported lazily inside the fixture body so this
    conftest can be imported in environments where freezegun is missing
    (the ``dev`` extra installs it; some sub-environments may not).
    """

    try:
        from freezegun import freeze_time  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only on broken envs
        pytest.skip(f"freezegun is required for the frozen_clock fixture: {exc}")

    # ``freeze_time`` returns a context manager whose ``__enter__``
    # produces the :class:`FrozenDateTimeFactory` (or one of its
    # ticking variants when called with ``tick=True``). The factory
    # exposes ``.tick()`` / ``.move_to(...)`` so callers can advance
    # time deterministically inside a test.
    with freeze_time("2024-01-01T00:00:00+00:00") as factory:
        yield factory  # type: ignore[misc]
