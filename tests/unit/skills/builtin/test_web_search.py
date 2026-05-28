"""Unit tests for ``jarvis.skills.builtin.web_search``.

Covers the four design promises Requirement 3 makes for the
``WebSearchSkill`` adapter:

* Default ``max_results`` is 5 and the cap is 10 (Requirement 3.1). The
  registry already enforces these bounds via JSON Schema; the Skill's
  runtime clamp protects against bypasses (e.g., a hand-built dispatch).
* Successful results carry the uniform ``{"title", "url", "snippet"}``
  rows plus the deduplicated ``cited_urls`` list and pre-formatted
  ``summary_lines`` so the Dialog_Manager can speak a summary while the
  transcript log cites the source URLs (Requirements 3.2, 3.3).
* Zero-result searches return ``ok=True`` with an empty ``results``
  array and an explanatory ``summary_lines`` line so the Dialog_Manager
  can offer to refine the query (Requirement 3.4).
* Non-2xx and transport-level failures translate into the documented
  Error Taxonomy codes: 401/403 → ``missing_credentials``, other
  ``WebSearchError`` → ``provider_unavailable``,
  ``NetworkPolicyViolation`` → ``access_denied``,
  ``httpx.TimeoutException`` → ``timeout``.

A fake search client is used everywhere so the tests are hermetic and do
not exercise the underlying HTTP transport. Provider-level coverage
(allowlist enforcement, retries, audit rows) lives in
``tests/unit/automation/providers/test_search.py``.

Validates: Requirements 3.1, 3.2, 3.3, 3.4
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

import httpx
import pytest

from jarvis.automation.providers.http import NetworkPolicyViolation
from jarvis.automation.providers.search import (
    DEFAULT_MAX_RESULTS,
    MAX_RESULTS_HARD_CAP,
    WebSearchError,
)
from jarvis.skills.base import Skill, SkillContext, SkillManifest, SkillResult
from jarvis.skills.builtin import web_search
from jarvis.skills.builtin.web_search import SKILL, WebSearchSkill

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeSearchClient:
    """Records every ``search`` call and replays a configured outcome.

    Mirrors :class:`WebSearchClient.search`'s public surface (the only
    method ``WebSearchSkill`` uses) while keeping the fake free of
    HTTP machinery. ``raise_exc`` short-circuits the call before any
    bookkeeping so the Skill's exception-translation paths are easy
    to exercise.
    """

    def __init__(
        self,
        *,
        results: list[dict[str, Any]] | None = None,
        raise_exc: BaseException | None = None,
    ) -> None:
        self.calls: list[tuple[str, int]] = []
        self._results: list[dict[str, Any]] = list(results or [])
        self._raise: BaseException | None = raise_exc

    async def search(
        self, query: str, max_results: int = DEFAULT_MAX_RESULTS
    ) -> list[dict[str, Any]]:
        self.calls.append((query, max_results))
        if self._raise is not None:
            raise self._raise
        return list(self._results)


def _make_ctx(client: _FakeSearchClient | None = None) -> SkillContext:
    """Build a minimal :class:`SkillContext` with the search provider wired."""
    providers: dict[str, Any] = {}
    if client is not None:
        providers["web_search"] = client
    return SkillContext(providers=providers, run_id="web-search-test")


def _run(coro: Awaitable[Any]) -> Any:
    return asyncio.run(coro)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Manifest / module surface
# ---------------------------------------------------------------------------


def test_module_exposes_skill_singleton() -> None:
    # The plugin loader reads a top-level ``SKILL`` attribute from each
    # built-in skill module; pin its presence and identity so refactors
    # cannot silently break discovery.
    assert isinstance(SKILL, WebSearchSkill)
    assert web_search.SKILL is SKILL


def test_skill_satisfies_skill_protocol() -> None:
    # The :class:`Skill` Protocol is runtime-checkable, so confirm the
    # singleton survives the registry's ``isinstance`` gate.
    assert isinstance(SKILL, Skill)


def test_manifest_is_non_destructive_with_expected_name() -> None:
    manifest = SKILL.manifest
    assert isinstance(manifest, SkillManifest)
    # Requirement 3.1: tool name is ``WebSearchSkill``.
    assert manifest.name == "WebSearchSkill"
    # Web search is read-only.
    assert manifest.destructive is False
    assert manifest.source == "builtin"


def test_schema_requires_query_and_constrains_max_results() -> None:
    schema = SKILL.manifest.json_schema
    # Requirement 3.1: ``query`` is required, ``max_results`` is optional.
    assert schema["required"] == ["query"]
    assert schema["properties"]["query"]["type"] == "string"
    assert schema["properties"]["query"]["minLength"] == 1
    # Requirement 3.1: ``max_results`` defaults to 5 and is capped at 10.
    assert schema["properties"]["max_results"]["type"] == "integer"
    assert schema["properties"]["max_results"]["default"] == DEFAULT_MAX_RESULTS == 5
    assert schema["properties"]["max_results"]["minimum"] == 1
    assert schema["properties"]["max_results"]["maximum"] == MAX_RESULTS_HARD_CAP == 10
    # Closed object so the LLM cannot smuggle extra fields through.
    assert schema["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_execute_returns_results_cited_urls_and_summary_lines() -> None:
    rows = [
        {
            "title": "Python.org",
            "url": "https://python.org",
            "snippet": "The official Python homepage.",
        },
        {
            "title": "PEP 8",
            "url": "https://peps.python.org/pep-0008/",
            "snippet": "Style guide.",
        },
    ]
    fake = _FakeSearchClient(results=rows)
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"query": "python"}, ctx))

    assert result.ok is True
    assert result.error_code is None
    assert isinstance(result.value, dict)

    # Requirement 3.2: uniform shape preserved.
    assert result.value["results"] == rows
    # Requirement 3.3: cited URLs surfaced in result order.
    assert result.value["cited_urls"] == [
        "https://python.org",
        "https://peps.python.org/pep-0008/",
    ]
    # Requirement 3.3: summary lines stitched per row with title, snippet,
    # and URL so the Dialog_Manager can render them straight to the log.
    summary = result.value["summary_lines"]
    assert len(summary) == 2
    assert "Python.org" in summary[0]
    assert "https://python.org" in summary[0]
    assert "PEP 8" in summary[1]
    assert "https://peps.python.org/pep-0008/" in summary[1]

    # The Skill must have called the provider exactly once with the
    # default ``max_results`` (5).
    assert fake.calls == [("python", DEFAULT_MAX_RESULTS)]


def test_execute_passes_caller_supplied_max_results() -> None:
    fake = _FakeSearchClient(results=[])
    ctx = _make_ctx(fake)

    _run(SKILL.execute({"query": "rust", "max_results": 3}, ctx))

    assert fake.calls == [("rust", 3)]


def test_execute_clamps_max_results_above_cap() -> None:
    # The schema would normally reject ``max_results=99`` but a caller
    # bypassing the registry (or a future schema evolution) must not be
    # able to defeat the design's hard cap.
    fake = _FakeSearchClient(results=[])
    ctx = _make_ctx(fake)

    _run(SKILL.execute({"query": "rust", "max_results": 99}, ctx))

    assert fake.calls == [("rust", MAX_RESULTS_HARD_CAP)]


def test_execute_clamps_max_results_below_floor() -> None:
    fake = _FakeSearchClient(results=[])
    ctx = _make_ctx(fake)

    _run(SKILL.execute({"query": "rust", "max_results": 0}, ctx))

    assert fake.calls == [("rust", 1)]


def test_execute_strips_query_whitespace_before_dispatch() -> None:
    # The schema's ``minLength: 1`` accepts the string "   ", so the
    # Skill normalises by stripping; the provider sees the cleaned form.
    fake = _FakeSearchClient(results=[])
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"query": "  hello world  "}, ctx))

    assert result.ok is True
    assert fake.calls == [("hello world", DEFAULT_MAX_RESULTS)]
    assert result.value["query"] == "hello world"


def test_execute_deduplicates_cited_urls() -> None:
    # Two results from the same URL must produce a single citation; the
    # Dialog_Manager renders ``cited_urls`` verbatim into the log.
    rows = [
        {"title": "A", "url": "https://example.com", "snippet": "first"},
        {"title": "B", "url": "https://example.com", "snippet": "second"},
        {"title": "C", "url": "https://other.example", "snippet": "third"},
    ]
    fake = _FakeSearchClient(results=rows)
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"query": "dup"}, ctx))

    assert result.value["cited_urls"] == [
        "https://example.com",
        "https://other.example",
    ]


def test_execute_handles_missing_fields_gracefully() -> None:
    # Provider rows occasionally come back with missing snippet / title;
    # the Skill must still produce a usable summary line.
    rows = [{"url": "https://only-url.example"}]
    fake = _FakeSearchClient(results=rows)
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"query": "thin"}, ctx))

    assert result.ok is True
    line = result.value["summary_lines"][0]
    assert "https://only-url.example" in line


# ---------------------------------------------------------------------------
# Zero results (Requirement 3.4)
# ---------------------------------------------------------------------------


def test_execute_returns_ok_with_explanation_when_no_results() -> None:
    fake = _FakeSearchClient(results=[])
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"query": "asdfghjkl"}, ctx))

    # Requirement 3.4: zero results is the success branch — the
    # Dialog_Manager offers to refine the query, not surface an error.
    assert result.ok is True
    assert result.value["results"] == []
    assert result.value["cited_urls"] == []
    assert len(result.value["summary_lines"]) == 1
    assert "asdfghjkl" in result.value["summary_lines"][0]


# ---------------------------------------------------------------------------
# Argument validation (defence-in-depth beyond the JSON Schema)
# ---------------------------------------------------------------------------


def test_execute_rejects_empty_query_after_strip() -> None:
    fake = _FakeSearchClient(results=[])
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"query": "   "}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    # The provider must not have been invoked.
    assert fake.calls == []


def test_execute_rejects_non_string_query() -> None:
    fake = _FakeSearchClient(results=[])
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"query": 42}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.calls == []


def test_execute_rejects_bool_max_results() -> None:
    # ``bool`` is a subclass of ``int``; the JSON Schema's
    # ``"type": "integer"`` would already reject it through the
    # registry, but we defend the Skill independently so direct
    # callers cannot smuggle ``True``/``False`` past the clamp.
    fake = _FakeSearchClient(results=[])
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"query": "q", "max_results": True}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.calls == []


def test_execute_rejects_non_integer_max_results() -> None:
    fake = _FakeSearchClient(results=[])
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"query": "q", "max_results": "5"}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.calls == []


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------


def test_execute_returns_provider_unavailable_when_no_client() -> None:
    # No ``web_search`` entry in providers — the Dialog_Manager will
    # render this as "search isn't configured" rather than asking the
    # user to repeat themselves.
    ctx = _make_ctx(client=None)

    result = _run(SKILL.execute({"query": "anything"}, ctx))

    assert result.ok is False
    assert result.error_code == "provider_unavailable"


# ---------------------------------------------------------------------------
# Error translations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [401, 403, 407])
def test_websearcherror_credential_status_maps_to_missing_credentials(
    status: int,
) -> None:
    exc = WebSearchError(provider="tavily", status_code=status, body="forbidden")
    fake = _FakeSearchClient(raise_exc=exc)
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"query": "q"}, ctx))

    assert result.ok is False
    assert result.error_code == "missing_credentials"


@pytest.mark.parametrize("status", [400, 404, 500, 503])
def test_websearcherror_other_statuses_map_to_provider_unavailable(
    status: int,
) -> None:
    exc = WebSearchError(provider="tavily", status_code=status, body="oops")
    fake = _FakeSearchClient(raise_exc=exc)
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"query": "q"}, ctx))

    assert result.ok is False
    assert result.error_code == "provider_unavailable"


def test_network_policy_violation_maps_to_access_denied() -> None:
    exc = NetworkPolicyViolation(
        destination="https://api.tavily.com", host="api.tavily.com"
    )
    fake = _FakeSearchClient(raise_exc=exc)
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"query": "q"}, ctx))

    assert result.ok is False
    assert result.error_code == "access_denied"


def test_timeout_exception_maps_to_timeout_error_code() -> None:
    exc = httpx.ReadTimeout("read timeout after 5s")
    fake = _FakeSearchClient(raise_exc=exc)
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"query": "q"}, ctx))

    assert result.ok is False
    assert result.error_code == "timeout"


def test_provider_value_error_maps_to_schema_violation() -> None:
    # The ``WebSearchClient`` raises ``ValueError`` for empty queries;
    # if a future evolution surfaces a different ValueError to the
    # Skill, retrying with adjusted args is the right Dialog_Manager
    # response (Requirement 14.5).
    fake = _FakeSearchClient(raise_exc=ValueError("query must be non-empty"))
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"query": "q"}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"


# ---------------------------------------------------------------------------
# Result shape sanity
# ---------------------------------------------------------------------------


def test_skill_result_is_a_skillresult_dataclass() -> None:
    fake = _FakeSearchClient(results=[])
    ctx = _make_ctx(fake)

    result = _run(SKILL.execute({"query": "q"}, ctx))

    # Defence against a refactor that returns a bare dict — the
    # registry's ``dispatch`` would otherwise have to convert it.
    assert isinstance(result, SkillResult)
