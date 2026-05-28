"""Smoke tests for :mod:`tests.strategies`.

The strategies module is consumed by the property-based tests in
``tests/property/test_property_*.py`` (tasks 21.2 .. 21.16). Those tests
quantify over interesting subsystems (Memory_Store, Reminder_Service,
Dialog_Manager) and would be hard to debug if a strategy itself were
silently broken.

This file therefore drives a small ``@given`` smoke test against every
exported strategy:

* Generated values satisfy the underlying dataclass invariants
  (:class:`Transcript`, :class:`MemoryRecord`).
* Generated tool-call argument dicts validate against the Skill's own
  :class:`jsonschema.Draft7Validator` (round-trip with the registry's
  Property-2 / CP2 gate).
* Generated PII corpus strings always contain at least one substring
  the production :class:`PIIRedactor` rewrites away.
* Generated reminder sets carry a strict total ordering on
  ``(trigger_at, seq)``.
* Generated Mistral tool payloads round-trip through
  :meth:`MistralSchemaValidator.validate` and through
  ``json.dumps`` / ``json.loads`` without information loss.

Validates: Requirements 14.3, 14.4
"""

from __future__ import annotations

from datetime import datetime
from itertools import pairwise
import json

from hypothesis import given
from jsonschema import Draft7Validator
import pytest
from tests.strategies import (
    PII_SAMPLES,
    memory_records,
    mistral_tool_payloads,
    pii_corpus,
    reminder_sets,
    tool_call_arguments,
    transcripts,
)

from jarvis.llm.mistral_schema import MistralSchemaValidator
from jarvis.memory.redactor import PIIRedactor
from jarvis.memory.store import MemoryRecord
from jarvis.skills.builtin.launch_app import LaunchAppSkill
from jarvis.skills.builtin.media_control import MediaControlSkill
from jarvis.skills.builtin.volume import VolumeSkill
from jarvis.voice.stt.base import Transcript

# ---------------------------------------------------------------------------
# Transcripts
# ---------------------------------------------------------------------------


@given(transcript=transcripts())
def test_transcripts_are_valid_transcript_instances(transcript: Transcript) -> None:
    """Every generated transcript is a fully-validated ``Transcript``."""
    assert isinstance(transcript, Transcript)
    # ``__post_init__`` already enforced the invariants below; we
    # re-check here so a regression in the strategy fails loudly with a
    # specific assertion rather than a generic constructor error.
    assert 0.0 <= transcript.confidence <= 1.0
    assert transcript.duration_ms >= 0
    assert transcript.started_at.tzinfo is not None
    assert transcript.language != ""


@given(transcript=transcripts(allow_empty=False))
def test_transcripts_with_allow_empty_false_have_text(transcript: Transcript) -> None:
    assert transcript.text != ""


@given(transcript=transcripts(min_confidence=0.5, max_confidence=1.0))
def test_transcripts_honour_confidence_bounds(transcript: Transcript) -> None:
    assert transcript.confidence >= 0.5


# ---------------------------------------------------------------------------
# Tool-call arguments
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[LaunchAppSkill, MediaControlSkill, VolumeSkill],
    ids=lambda cls: cls.__name__,
)
def skill_instance(request: pytest.FixtureRequest) -> object:
    return request.param()


def test_tool_call_arguments_validate_against_skill_schema(
    skill_instance: object,
) -> None:
    """Every generated argument dict satisfies the Skill's JSON Schema."""
    schema = skill_instance.manifest.json_schema  # type: ignore[attr-defined]
    validator = Draft7Validator(schema)

    @given(args=tool_call_arguments(skill_instance))
    def _check(args: dict[str, object]) -> None:
        # ``hypothesis_jsonschema.from_schema`` already filters by
        # validity; this assertion guards against accidental drift
        # (e.g., the strategy switching to a different schema) and
        # keeps the failure surface focussed on the strategy itself.
        assert validator.is_valid(args), list(validator.iter_errors(args))
        assert isinstance(args, dict)

    _check()


# ---------------------------------------------------------------------------
# Memory records
# ---------------------------------------------------------------------------


@given(record=memory_records())
def test_memory_records_round_trip_through_constructor(record: MemoryRecord) -> None:
    """``MemoryRecord(**fields)`` reconstructs an equal record."""
    rebuilt = MemoryRecord(
        record_id=record.record_id,
        content=record.content,
        embedding=list(record.embedding),
        timestamp=record.timestamp,
        category=record.category,
        provenance=dict(record.provenance),
        redacted=record.redacted,
    )
    assert rebuilt == record
    # Embedding length is preserved; default dimension is 8.
    assert len(record.embedding) == 8


@given(record=memory_records(redacted_only=True))
def test_memory_records_redacted_only_sets_flag(record: MemoryRecord) -> None:
    assert record.redacted is True


# ---------------------------------------------------------------------------
# Reminder sets
# ---------------------------------------------------------------------------


@given(reminders=reminder_sets())
def test_reminder_sets_have_strict_total_order(
    reminders: list[tuple[datetime, int, str]],
) -> None:
    """``(trigger_at, seq)`` is a strict total order across the generated set."""
    assert len(reminders) >= 1
    keys = [(triple[0], triple[1]) for triple in reminders]
    # Sorted (already) and strictly increasing — i.e. no two reminders
    # share both ``trigger_at`` and ``seq``.
    for prev, curr in pairwise(keys):
        assert prev < curr
    # Labels are non-empty strings (Reminder.label invariant).
    for _, _, label in reminders:
        assert isinstance(label, str)
        assert label != ""


# ---------------------------------------------------------------------------
# PII corpus
# ---------------------------------------------------------------------------


@given(text=pii_corpus())
def test_pii_corpus_contains_at_least_one_known_sample(text: str) -> None:
    """Every generated string carries a literal sample from :data:`PII_SAMPLES`."""
    assert any(sample in text for _kind, sample in PII_SAMPLES)


@given(text=pii_corpus())
def test_pii_corpus_strings_are_redacted_by_default_redactor(text: str) -> None:
    """The default :class:`PIIRedactor` strips every embedded sample."""
    redactor = PIIRedactor.with_defaults()
    redacted = redactor.redact(text)
    for _kind, sample in PII_SAMPLES:
        if sample in text:
            assert sample not in redacted, (
                f"sample {sample!r} survived redaction in {redacted!r}"
            )


# ---------------------------------------------------------------------------
# Mistral tool payloads
# ---------------------------------------------------------------------------


@given(payload=mistral_tool_payloads())
def test_mistral_tool_payloads_match_expected_shape(
    payload: dict[str, object],
) -> None:
    assert payload["type"] == "function"
    function = payload["function"]
    assert isinstance(function, dict)
    assert isinstance(function["name"], str) and function["name"]
    assert isinstance(function["description"], str)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    assert parameters["type"] == "object"


@given(payload=mistral_tool_payloads())
def test_mistral_tool_payloads_pass_subset_validator(
    payload: dict[str, object],
) -> None:
    """Generated parameters pass :class:`MistralSchemaValidator`."""
    validator = MistralSchemaValidator()
    parameters = payload["function"]["parameters"]  # type: ignore[index]
    validator.validate(parameters)


@given(payload=mistral_tool_payloads())
def test_mistral_tool_payloads_round_trip_through_json(
    payload: dict[str, object],
) -> None:
    """The payload is JSON-serialisable without information loss (CP15)."""
    encoded = json.dumps(payload, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded == payload
