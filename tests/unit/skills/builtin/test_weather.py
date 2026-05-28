"""Unit tests for ``jarvis.skills.builtin.weather``.

Covers the design promises Requirement 7 makes for the
``WeatherSkill`` adapter:

* The schema declares ``location`` as an optional string field
  (Requirement 7.1) and the underlying provider's configured
  ``default_location`` is used when the LLM omits the field.
* Successful results carry the provider payload verbatim — the
  ``current`` conditions and the 24-hour ``forecast`` list — so the
  Dialog_Manager can pass them straight to the LLM for summarisation
  (Requirement 7.2).
* Provider failures translate into the documented Error Taxonomy
  codes (Requirement 7.7):

  * :class:`ProviderError("missing_credentials")` →
    ``missing_credentials``;
  * :class:`ProviderError("provider_unavailable")` →
    ``provider_unavailable``;
  * :class:`NetworkPolicyViolation` → ``access_denied``;
  * :class:`httpx.TimeoutException` → ``timeout``.

A fake weather client is used everywhere so the tests are hermetic
and do not exercise the underlying HTTP transport. Provider-level
coverage (allowlist enforcement, retries, audit rows) lives in
``tests/unit/automation/providers/test_weather.py``.

Validates: Requirements 7.1, 7.2, 7.7
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

import httpx
import pytest

from jarvis.automation.providers.errors import ProviderError
from jarvis.automation.providers.http import NetworkPolicyViolation
from jarvis.skills.base import (
    Skill,
    SkillContext,
    SkillManifest,
    SkillResult,
)
from jarvis.skills.builtin import weather
from jarvis.skills.builtin.weather import SKILL, WeatherSkill

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeWeatherClient:
    """Records every ``fetch`` call and replays a configured outcome.

    Mirrors :class:`WeatherClient.fetch`'s public surface (the only
    method ``WeatherSkill`` uses) while keeping the fake free of
    HTTP machinery. ``raise_exc`` short-circuits the call before any
    bookkeeping so the Skill's exception-translation paths are easy
    to exercise.
    """

    def __init__(
        self,
        *,
        payload: dict[str, Any] | None = None,
        raise_exc: BaseException | None = None,
    ) -> None:
        self.calls: list[str | None] = []
        self._payload: dict[str, Any] = dict(
            payload
            or {
                "location": "Bandung,ID",
                "current": {"main": {"temp": 25.0}},
                "forecast": [{"dt": i} for i in range(8)],
            }
        )
        self._raise: BaseException | None = raise_exc

    async def fetch(self, location: str | None = None) -> dict[str, Any]:
        self.calls.append(location)
        if self._raise is not None:
            raise self._raise
        # Echo the requested location into the payload so callers can
        # observe the resolved value without inspecting ``calls``.
        out = dict(self._payload)
        if location is not None and location.strip():
            out["location"] = location
        return out


def _make_ctx(client: _FakeWeatherClient | None = None) -> SkillContext:
    """Build a minimal :class:`SkillContext` with the weather provider."""
    providers: dict[str, Any] = {}
    if client is not None:
        providers["weather"] = client
    return SkillContext(providers=providers, run_id="weather-test")


def _run(coro: Awaitable[Any]) -> Any:
    return asyncio.run(coro)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Manifest / module surface
# ---------------------------------------------------------------------------


def test_module_exposes_skill_singleton() -> None:
    # The plugin loader reads a top-level ``SKILL`` attribute from each
    # built-in skill module; pin its presence and identity so refactors
    # cannot silently break discovery.
    assert isinstance(SKILL, WeatherSkill)
    assert weather.SKILL is SKILL


def test_skill_satisfies_skill_protocol() -> None:
    # The :class:`Skill` Protocol is runtime-checkable, so confirm the
    # singleton survives the registry's ``isinstance`` gate.
    assert isinstance(SKILL, Skill)


def test_manifest_is_non_destructive_with_expected_name() -> None:
    manifest = SKILL.manifest
    assert isinstance(manifest, SkillManifest)
    # Requirement 7.1: tool name is ``WeatherSkill``.
    assert manifest.name == "WeatherSkill"
    # Weather lookup is read-only.
    assert manifest.destructive is False
    assert manifest.source == "builtin"


def test_schema_makes_location_optional_and_constrained() -> None:
    schema = SKILL.manifest.json_schema
    # Requirement 7.1: ``location`` is the only field and it is optional.
    assert schema["required"] == []
    assert "location" in schema["properties"]
    location_schema = schema["properties"]["location"]
    assert location_schema["type"] == "string"
    assert location_schema["minLength"] == 1
    # Maximum length is bounded; the exact value is implementation
    # detail, but it must be positive.
    assert location_schema["maxLength"] > 0
    # Closed object so the LLM cannot smuggle extra fields through.
    assert schema["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_execute_passes_location_through_to_provider() -> None:
    fake = _FakeWeatherClient()
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"location": "Jakarta,ID"}, ctx))

    assert result.ok is True
    assert result.error_code is None
    assert isinstance(result.value, dict)
    # Requirement 7.2: payload carries current conditions + forecast.
    assert "current" in result.value
    assert "forecast" in result.value
    assert len(result.value["forecast"]) == 8
    # The Skill must have called the provider exactly once with the
    # supplied location verbatim.
    assert fake.calls == ["Jakarta,ID"]


def test_execute_strips_location_whitespace_before_dispatch() -> None:
    # The schema's ``minLength: 1`` accepts the string "   ", so the
    # Skill normalises by stripping; the provider sees the cleaned form.
    fake = _FakeWeatherClient()
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"location": "  London,GB  "}, ctx))

    assert result.ok is True
    assert fake.calls == ["London,GB"]


def test_execute_omits_location_when_field_missing() -> None:
    # Requirement 7.1: omitting ``location`` defers to the provider's
    # configured default. The Skill signals this by passing ``None``.
    fake = _FakeWeatherClient()
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({}, ctx))

    assert result.ok is True
    assert fake.calls == [None]


def test_execute_treats_explicit_null_location_as_omitted() -> None:
    # Some LLMs emit ``"location": null`` when they want the default;
    # the Skill normalises this to the same path as the missing field.
    fake = _FakeWeatherClient()
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"location": None}, ctx))

    assert result.ok is True
    assert fake.calls == [None]


# ---------------------------------------------------------------------------
# Argument validation (defence-in-depth beyond the JSON Schema)
# ---------------------------------------------------------------------------


def test_execute_rejects_empty_location_after_strip() -> None:
    fake = _FakeWeatherClient()
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"location": "   "}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    # The provider must not have been invoked.
    assert fake.calls == []


def test_execute_rejects_non_string_location() -> None:
    fake = _FakeWeatherClient()
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"location": 42}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.calls == []


def test_execute_rejects_oversized_location() -> None:
    # Defence against an LLM emitting an unbounded query string;
    # the cap protects the per-turn payload size.
    fake = _FakeWeatherClient()
    ctx = _make_ctx(fake)
    huge = "x" * 1000

    result = _run(SKILL.execute({"location": huge}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.calls == []


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------


def test_execute_returns_provider_unavailable_when_no_client() -> None:
    # No ``weather`` entry in providers — the Dialog_Manager will
    # render this as "weather isn't configured" rather than asking
    # the user to repeat themselves.
    ctx = _make_ctx(client=None)

    result = _run(SKILL.execute({"location": "Jakarta,ID"}, ctx))

    assert result.ok is False
    assert result.error_code == "provider_unavailable"


# ---------------------------------------------------------------------------
# Error translations (Requirement 7.7)
# ---------------------------------------------------------------------------


def test_provider_missing_credentials_maps_to_missing_credentials() -> None:
    exc = ProviderError(
        "missing_credentials",
        "credential 'weather/api_key' is not set",
        provider="openweather",
    )
    fake = _FakeWeatherClient(raise_exc=exc)
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"location": "Jakarta,ID"}, ctx))

    assert result.ok is False
    assert result.error_code == "missing_credentials"


def test_provider_unavailable_maps_to_provider_unavailable() -> None:
    exc = ProviderError(
        "provider_unavailable",
        "OpenWeather returned HTTP 503",
        provider="openweather",
    )
    fake = _FakeWeatherClient(raise_exc=exc)
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"location": "Jakarta,ID"}, ctx))

    assert result.ok is False
    assert result.error_code == "provider_unavailable"


def test_network_policy_violation_maps_to_access_denied() -> None:
    exc = NetworkPolicyViolation(
        destination="https://api.openweathermap.org",
        host="api.openweathermap.org",
    )
    fake = _FakeWeatherClient(raise_exc=exc)
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"location": "Jakarta,ID"}, ctx))

    assert result.ok is False
    assert result.error_code == "access_denied"


def test_timeout_exception_maps_to_timeout_error_code() -> None:
    # Defended at the Skill layer even though the provider already
    # converts timeouts into ProviderError("provider_unavailable") —
    # protects against future client refactors.
    exc = httpx.ReadTimeout("read timeout after 5s")
    fake = _FakeWeatherClient(raise_exc=exc)
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"location": "Jakarta,ID"}, ctx))

    assert result.ok is False
    assert result.error_code == "timeout"


# ---------------------------------------------------------------------------
# Result shape sanity
# ---------------------------------------------------------------------------


def test_skill_result_is_a_skillresult_dataclass() -> None:
    fake = _FakeWeatherClient()
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"location": "Jakarta,ID"}, ctx))

    # Defence against a refactor that returns a bare dict — the
    # registry's ``dispatch`` would otherwise have to convert it.
    assert isinstance(result, SkillResult)


def test_unexpected_provider_response_shape_maps_to_provider_unavailable() -> None:
    # A misbehaving fake / future client variant could hand back a
    # non-dict; the Skill must not crash the Dialog_Manager.
    class _BadClient:
        async def fetch(self, location: str | None = None) -> Any:
            return ["unexpected"]

    ctx = SkillContext(providers={"weather": _BadClient()}, run_id="weather-test")

    result = _run(SKILL.execute({"location": "Jakarta,ID"}, ctx))

    assert result.ok is False
    assert result.error_code == "provider_unavailable"


# Parametrised matrix covering every documented error mapping in one
# place — handy for regression-pinning future taxonomy additions.
@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (
            ProviderError("missing_credentials", "no key", provider="openweather"),
            "missing_credentials",
        ),
        (
            ProviderError(
                "provider_unavailable",
                "HTTP 500",
                provider="openweather",
            ),
            "provider_unavailable",
        ),
        (
            NetworkPolicyViolation(
                destination="https://api.openweathermap.org",
                host="api.openweathermap.org",
            ),
            "access_denied",
        ),
        (httpx.ConnectTimeout("connect timeout"), "timeout"),
        (httpx.ReadTimeout("read timeout"), "timeout"),
    ],
)
def test_error_translation_matrix(exc: BaseException, expected: str) -> None:
    fake = _FakeWeatherClient(raise_exc=exc)
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"location": "Jakarta,ID"}, ctx))

    assert result.ok is False
    assert result.error_code == expected
