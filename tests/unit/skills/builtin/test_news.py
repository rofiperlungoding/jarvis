"""Unit tests for ``jarvis.skills.builtin.news``.

Covers the design promises Requirement 7 makes for the ``NewsSkill``
adapter:

* Default ``max_items`` is 5 and the cap is 10 (Requirement 7.3). The
  registry already enforces these bounds via JSON Schema; the Skill's
  runtime clamp protects against bypasses (e.g., a hand-built dispatch).
* Successful results carry the canonical ``{"title", "source", "url",
  "published_at", "description"}`` rows plus the deduplicated
  ``cited_urls`` list and pre-formatted ``headlines`` so the
  Dialog_Manager can speak the headlines while the transcript log cites
  the source URLs (Requirement 7.4).
* Empty result sets return ``ok=True`` with an empty ``articles`` array
  and an explanatory ``headlines`` line so the Dialog_Manager can offer
  to refine the topic.
* Provider failures translate into the documented Error Taxonomy codes:
  :class:`ProviderError` codes pass through 1:1
  (``missing_credentials``, ``provider_unavailable``);
  :class:`NetworkPolicyViolation` → ``access_denied``;
  :class:`httpx.TimeoutException` → ``timeout`` (Requirements 5.6, 7.7,
  13.6).

A fake news client is used everywhere so the tests are hermetic and do
not exercise the underlying HTTP transport. Provider-level coverage
(allowlist enforcement, retries, audit rows) lives in
``tests/unit/automation/providers/test_news.py``.

Validates: Requirements 7.3, 7.4, 7.7
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

import httpx
import pytest

from jarvis.automation.providers.errors import ProviderError
from jarvis.automation.providers.http import NetworkPolicyViolation
from jarvis.skills.base import Skill, SkillContext, SkillManifest, SkillResult
from jarvis.skills.builtin import news as news_module
from jarvis.skills.builtin.news import (
    DEFAULT_MAX_ITEMS,
    MAX_ITEMS_HARD_CAP,
    SKILL,
    NewsSkill,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeNewsClient:
    """Records every ``fetch`` call and replays a configured outcome.

    Mirrors :class:`NewsClient.fetch`'s public surface (the only method
    ``NewsSkill`` uses) while keeping the fake free of HTTP machinery.
    ``raise_exc`` short-circuits the call before any bookkeeping so the
    Skill's exception-translation paths are easy to exercise.
    """

    def __init__(
        self,
        *,
        articles: list[dict[str, Any]] | None = None,
        raise_exc: BaseException | None = None,
    ) -> None:
        self.calls: list[tuple[str | None, int]] = []
        self._articles: list[dict[str, Any]] = list(articles or [])
        self._raise: BaseException | None = raise_exc

    async def fetch(
        self,
        topic: str | None = None,
        max_items: int = DEFAULT_MAX_ITEMS,
    ) -> list[dict[str, Any]]:
        self.calls.append((topic, max_items))
        if self._raise is not None:
            raise self._raise
        return list(self._articles)


def _make_ctx(client: _FakeNewsClient | None = None) -> SkillContext:
    """Build a minimal :class:`SkillContext` with the news provider wired."""
    providers: dict[str, Any] = {}
    if client is not None:
        providers["news"] = client
    return SkillContext(providers=providers, run_id="news-test")


def _run(coro: Awaitable[Any]) -> Any:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _article(
    *,
    title: str = "headline",
    source: str | None = "src",
    url: str | None = "https://example.invalid/a",
    published_at: str | None = "2024-01-01T00:00:00Z",
    description: str | None = "desc",
) -> dict[str, Any]:
    """Build a normalised NewsAPI-shaped article dict for tests."""
    return {
        "title": title,
        "source": source,
        "url": url,
        "published_at": published_at,
        "description": description,
    }


# ---------------------------------------------------------------------------
# Manifest / module surface
# ---------------------------------------------------------------------------


def test_module_exposes_skill_singleton() -> None:
    # The plugin loader reads a top-level ``SKILL`` attribute from each
    # built-in skill module; pin its presence and identity so refactors
    # cannot silently break discovery.
    assert isinstance(SKILL, NewsSkill)
    assert news_module.SKILL is SKILL


def test_skill_satisfies_skill_protocol() -> None:
    # The :class:`Skill` Protocol is runtime-checkable, so confirm the
    # singleton survives the registry's ``isinstance`` gate.
    assert isinstance(SKILL, Skill)


def test_manifest_is_non_destructive_with_expected_name() -> None:
    manifest = SKILL.manifest
    assert isinstance(manifest, SkillManifest)
    # Requirement 7.3: tool name is ``NewsSkill``.
    assert manifest.name == "NewsSkill"
    # News fetching is read-only.
    assert manifest.destructive is False
    assert manifest.source == "builtin"


def test_schema_marks_topic_optional_and_clamps_max_items() -> None:
    schema = SKILL.manifest.json_schema
    # Requirement 7.3: ``topic`` is optional, ``max_items`` is optional
    # with default 5 and cap 10.
    assert schema["required"] == []
    assert schema["properties"]["topic"]["type"] == "string"
    assert schema["properties"]["topic"]["minLength"] == 1
    assert schema["properties"]["max_items"]["type"] == "integer"
    assert schema["properties"]["max_items"]["default"] == DEFAULT_MAX_ITEMS == 5
    assert schema["properties"]["max_items"]["minimum"] == 1
    assert schema["properties"]["max_items"]["maximum"] == MAX_ITEMS_HARD_CAP == 10
    # Closed object so the LLM cannot smuggle extra fields through.
    assert schema["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_execute_returns_articles_cited_urls_and_headlines() -> None:
    rows = [
        _article(
            title="AI breakthrough",
            source="ExampleNews",
            url="https://example.invalid/1",
        ),
        _article(
            title="More AI",
            source="OtherNews",
            url="https://example.invalid/2",
        ),
    ]
    fake = _FakeNewsClient(articles=rows)
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"topic": "ai"}, ctx))

    assert result.ok is True
    assert result.error_code is None
    assert isinstance(result.value, dict)

    # Canonical shape preserved verbatim from the provider.
    assert result.value["articles"] == rows
    # Cited URLs surfaced in result order (Requirement 7.4 — the
    # transcript log can render the citations directly).
    assert result.value["cited_urls"] == [
        "https://example.invalid/1",
        "https://example.invalid/2",
    ]
    # Headline lines stitched per row with title, source, and URL so
    # the Dialog_Manager can render them straight to the log.
    headlines = result.value["headlines"]
    assert len(headlines) == 2
    assert "AI breakthrough" in headlines[0]
    assert "ExampleNews" in headlines[0]
    assert "https://example.invalid/1" in headlines[0]
    assert "More AI" in headlines[1]
    assert "https://example.invalid/2" in headlines[1]

    # The Skill must have called the provider exactly once with the
    # default ``max_items`` (5).
    assert fake.calls == [("ai", DEFAULT_MAX_ITEMS)]
    assert result.value["topic"] == "ai"
    assert result.value["max_items"] == DEFAULT_MAX_ITEMS


def test_execute_passes_caller_supplied_max_items() -> None:
    fake = _FakeNewsClient(articles=[])
    ctx = _make_ctx(fake)

    _run(SKILL.execute({"topic": "rust", "max_items": 3}, ctx))

    assert fake.calls == [("rust", 3)]


def test_execute_clamps_max_items_above_cap() -> None:
    # The schema would normally reject ``max_items=99`` but a caller
    # bypassing the registry (or a future schema evolution) must not be
    # able to defeat the design's hard cap.
    fake = _FakeNewsClient(articles=[])
    ctx = _make_ctx(fake)

    _run(SKILL.execute({"topic": "rust", "max_items": 99}, ctx))

    assert fake.calls == [("rust", MAX_ITEMS_HARD_CAP)]


def test_execute_clamps_max_items_below_floor() -> None:
    fake = _FakeNewsClient(articles=[])
    ctx = _make_ctx(fake)

    _run(SKILL.execute({"topic": "rust", "max_items": 0}, ctx))

    assert fake.calls == [("rust", 1)]


def test_execute_strips_topic_whitespace_before_dispatch() -> None:
    fake = _FakeNewsClient(articles=[])
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"topic": "  hello world  "}, ctx))

    assert result.ok is True
    assert fake.calls == [("hello world", DEFAULT_MAX_ITEMS)]
    assert result.value["topic"] == "hello world"


def test_execute_omitted_topic_is_forwarded_as_none() -> None:
    """Requirement 7.3 — ``topic`` is optional; the provider expands the
    configured default when ``None`` is passed through."""
    fake = _FakeNewsClient(articles=[])
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({}, ctx))

    assert result.ok is True
    assert fake.calls == [(None, DEFAULT_MAX_ITEMS)]
    assert result.value["topic"] is None


def test_execute_blank_topic_is_forwarded_as_none() -> None:
    # The schema rejects ``"   "`` because of ``minLength: 1``, but a
    # direct caller (or a future schema-evolution that allows
    # whitespace-only) must still see the provider's default fallback.
    fake = _FakeNewsClient(articles=[])
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"topic": "   "}, ctx))

    assert result.ok is True
    assert fake.calls == [(None, DEFAULT_MAX_ITEMS)]
    assert result.value["topic"] is None


def test_execute_deduplicates_cited_urls() -> None:
    # Two articles from the same URL must produce a single citation;
    # the Dialog_Manager renders ``cited_urls`` verbatim into the log.
    rows = [
        _article(title="A", url="https://example.com"),
        _article(title="B", url="https://example.com"),
        _article(title="C", url="https://other.example"),
    ]
    fake = _FakeNewsClient(articles=rows)
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"topic": "dup"}, ctx))

    assert result.value["cited_urls"] == [
        "https://example.com",
        "https://other.example",
    ]


def test_execute_handles_missing_fields_gracefully() -> None:
    # NewsAPI occasionally returns rows with missing source or
    # description; the Skill must still produce a usable headline line.
    rows = [_article(title="thin", source=None, description=None)]
    fake = _FakeNewsClient(articles=rows)
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"topic": "thin"}, ctx))

    assert result.ok is True
    line = result.value["headlines"][0]
    assert "thin" in line


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------


def test_execute_returns_ok_with_explanation_when_no_articles() -> None:
    fake = _FakeNewsClient(articles=[])
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"topic": "asdfghjkl"}, ctx))

    # Empty result sets stay on the success branch — the Dialog_Manager
    # offers to refine the topic, not surface an error.
    assert result.ok is True
    assert result.value["articles"] == []
    assert result.value["cited_urls"] == []
    assert len(result.value["headlines"]) == 1
    assert "asdfghjkl" in result.value["headlines"][0]


def test_execute_empty_results_with_default_topic_uses_friendly_descriptor() -> None:
    fake = _FakeNewsClient(articles=[])
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({}, ctx))

    assert result.ok is True
    assert result.value["headlines"] == [
        "No headlines were found for the configured default topic."
    ]


# ---------------------------------------------------------------------------
# Argument validation (defence-in-depth beyond the JSON Schema)
# ---------------------------------------------------------------------------


def test_execute_rejects_non_string_topic() -> None:
    fake = _FakeNewsClient(articles=[])
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"topic": 42}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.calls == []


def test_execute_rejects_bool_max_items() -> None:
    # ``bool`` is a subclass of ``int``; the JSON Schema's
    # ``"type": "integer"`` would already reject it through the
    # registry, but we defend the Skill independently so direct
    # callers cannot smuggle ``True``/``False`` past the clamp.
    fake = _FakeNewsClient(articles=[])
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"max_items": True}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.calls == []


def test_execute_rejects_non_integer_max_items() -> None:
    fake = _FakeNewsClient(articles=[])
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"max_items": "5"}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.calls == []


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------


def test_execute_returns_provider_unavailable_when_no_client() -> None:
    # No ``news`` entry in providers — the Dialog_Manager will render
    # this as "the news provider isn't configured" rather than asking
    # the user to repeat themselves.
    ctx = _make_ctx(client=None)

    result = _run(SKILL.execute({"topic": "anything"}, ctx))

    assert result.ok is False
    assert result.error_code == "provider_unavailable"


# ---------------------------------------------------------------------------
# Error translations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("missing_credentials", "missing_credentials"),
        ("provider_unavailable", "provider_unavailable"),
    ],
)
def test_provider_error_maps_to_matching_skill_error_code(code: str, expected: str) -> None:
    exc = ProviderError(code, "boom", provider="newsapi")  # type: ignore[arg-type]
    fake = _FakeNewsClient(raise_exc=exc)
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"topic": "q"}, ctx))

    assert result.ok is False
    assert result.error_code == expected


def test_network_policy_violation_maps_to_access_denied() -> None:
    exc = NetworkPolicyViolation(destination="https://newsapi.org", host="newsapi.org")
    fake = _FakeNewsClient(raise_exc=exc)
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"topic": "q"}, ctx))

    assert result.ok is False
    assert result.error_code == "access_denied"


def test_timeout_exception_maps_to_timeout_error_code() -> None:
    exc = httpx.ReadTimeout("read timeout after 5s")
    fake = _FakeNewsClient(raise_exc=exc)
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"topic": "q"}, ctx))

    assert result.ok is False
    assert result.error_code == "timeout"


def test_provider_value_error_maps_to_schema_violation() -> None:
    # If the provider rejects the arguments (e.g. unparseable max_items
    # in a future evolution), retrying with adjusted args is the right
    # Dialog_Manager response (Requirement 14.5).
    fake = _FakeNewsClient(raise_exc=ValueError("topic must be non-empty"))
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"topic": "q"}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"


# ---------------------------------------------------------------------------
# Result shape sanity
# ---------------------------------------------------------------------------


def test_skill_result_is_a_skillresult_dataclass() -> None:
    fake = _FakeNewsClient(articles=[])
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"topic": "q"}, ctx))

    # Defence against a refactor that returns a bare dict — the
    # registry's ``dispatch`` would otherwise have to convert it.
    assert isinstance(result, SkillResult)
