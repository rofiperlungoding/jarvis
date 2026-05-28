"""Reusable Hypothesis strategies for the JARVIS property-based tests.

Task 21.1 calls out a small, well-defined catalog of strategies:

* :func:`transcripts` — :class:`~jarvis.voice.stt.base.Transcript` instances,
  including the empty-text and low-confidence corners that Property 13
  (CP — STT gating) needs to exercise.
* :func:`tool_call_arguments` — argument dicts for an arbitrary
  :class:`~jarvis.skills.base.Skill`, generated from the Skill's own
  JSON Schema via :mod:`hypothesis_jsonschema`. Feeds Property 1
  (intent round-trip) and Property 2 (schema soundness).
* :func:`memory_records` — :class:`~jarvis.memory.store.MemoryRecord`
  values that round-trip through ``MemoryRecord(...)`` construction.
  Feeds Properties 3, 4 and 15 (memory determinism / forget /
  redaction containment).
* :func:`reminder_sets` — sets of *(trigger_at, seq, label)* triples
  with strictly ordered ``(trigger_at, seq)`` keys. Feeds Property 10
  (CP13 — reminder firing order).
* :func:`pii_corpus` — input strings sprinkled with the default PII
  patterns from :class:`~jarvis.memory.redactor.PIIRedactor`. Feeds
  Property 15 (CP — redaction containment).
* :func:`mistral_tool_payloads` — Mistral-format tool definitions
  shaped to satisfy :class:`~jarvis.llm.mistral_schema.MistralSchemaValidator`.
  Feeds Property 12 (CP15 — function-definition conformance).

Design notes
------------

* Every strategy is a *plain function* that returns a Hypothesis
  ``SearchStrategy``. Tests then write ``@given(transcripts())`` rather
  than ``@given(transcripts)`` so the call-site can pass per-test
  parameters (e.g. ``transcripts(allow_empty=False)``) without each
  strategy having to be re-exported in multiple shapes.
* :func:`tool_call_arguments` accepts a :class:`Skill` *instance* (or
  any object with a ``.manifest.json_schema`` attribute) so callers can
  hand it a built-in or fake skill interchangeably.
* Strategies are deliberately conservative about edge cases: empty
  strings, naive datetimes, and out-of-range confidences are rejected
  by the underlying :class:`Transcript` / :class:`MemoryRecord`
  constructors, so we only generate values their dataclasses accept.
  Property tests that need to *exercise* the rejection path build
  malformed values directly inside the test body.

Validates: Requirements 14.3, 14.4
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from hypothesis import strategies as st
from hypothesis_jsonschema import from_schema

from jarvis.memory.redactor import DEFAULT_PATTERNS
from jarvis.memory.store import MemoryRecord
from jarvis.voice.stt.base import Transcript

__all__ = [
    "PII_SAMPLES",
    "memory_records",
    "mistral_tool_payloads",
    "pii_corpus",
    "reminder_sets",
    "tool_call_arguments",
    "transcripts",
]


# ---------------------------------------------------------------------------
# Shared building-block strategies
# ---------------------------------------------------------------------------


# A reasonable-but-bounded text strategy for free-form fields. Hypothesis'
# default ``text()`` includes surrogate / control characters that
# routinely break naïve consumers; the BMP printable range is broad
# enough to exercise interesting behaviours (Unicode normalisation,
# punctuation, mixed scripts) without inviting corner-case explosions
# in unrelated subsystems (e.g., logging handlers).
_safe_text = st.text(
    alphabet=st.characters(
        min_codepoint=0x20,
        max_codepoint=0xFFFD,
        # Exclude unicode surrogate code points (category ``Cs``); they
        # are not valid in well-formed strings and trip up downstream
        # consumers (logging handlers, JSON encoders) in ways that
        # mask the behaviour the property tests are meant to expose.
        exclude_categories=("Cs",),  # type: ignore[arg-type]
    ),
    min_size=0,
    max_size=64,
)

# A non-empty variant for fields where the empty string is rejected by
# the underlying dataclass (e.g. ``Transcript.language``,
# ``MemoryRecord.record_id``).
_non_empty_text = _safe_text.filter(bool)

# UTC-aware datetimes within a wide-enough window to round-trip through
# ISO 8601 without precision loss but narrow enough to keep Hypothesis
# from spending all its budget on year-9999 corner cases.
_utc_datetimes = st.datetimes(
    min_value=datetime(2000, 1, 1),
    max_value=datetime(2099, 12, 31),
    timezones=st.just(UTC),
)

# BCP-47 / ISO-639-1 tags. Every transcript carries one. We pick from a
# small, hand-picked set of common tags rather than generating arbitrary
# strings because the Transcript dataclass accepts the empty-string
# rejection as the only validation rule, but downstream consumers
# (faster-whisper, Mistral) have their own constraints we do not want
# to violate during composition tests.
_language_tags = st.sampled_from(
    ("en", "en-GB", "en-US", "fr", "de", "es", "ja", "zh", "ko", "pt")
)


# ---------------------------------------------------------------------------
# Transcripts
# ---------------------------------------------------------------------------


def transcripts(
    *,
    allow_empty: bool = True,
    min_confidence: float = 0.0,
    max_confidence: float = 1.0,
) -> st.SearchStrategy[Transcript]:
    """Generate :class:`~jarvis.voice.stt.base.Transcript` values.

    Default behaviour matches the Transcript invariants: confidence in
    ``[0.0, 1.0]``, timezone-aware ``started_at``, non-negative
    ``duration_ms``, non-empty ``language``. ``text`` may be empty
    unless ``allow_empty=False`` — Property 13 (STT gating) needs the
    empty-text branch in its strategy, while Properties 5 and 7 want to
    exclude it so the gating short-circuit does not cover up the
    behaviour they are checking.

    ``min_confidence`` / ``max_confidence`` let callers narrow the
    confidence range — for example, ``transcripts(min_confidence=0.4)``
    skips the low-confidence gate so Property 5 sees a real LLM call.
    """

    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError(f"min_confidence must be in [0, 1]; got {min_confidence}")
    if not 0.0 <= max_confidence <= 1.0:
        raise ValueError(f"max_confidence must be in [0, 1]; got {max_confidence}")
    if min_confidence > max_confidence:
        raise ValueError(
            f"min_confidence ({min_confidence}) > max_confidence ({max_confidence})"
        )

    text_strategy = _safe_text if allow_empty else _non_empty_text

    return st.builds(
        Transcript,
        text=text_strategy,
        confidence=st.floats(
            min_value=min_confidence,
            max_value=max_confidence,
            allow_nan=False,
            allow_infinity=False,
        ),
        started_at=_utc_datetimes,
        duration_ms=st.integers(min_value=0, max_value=60_000),
        language=_language_tags,
    )


# ---------------------------------------------------------------------------
# Tool-call arguments
# ---------------------------------------------------------------------------


def tool_call_arguments(skill: Any) -> st.SearchStrategy[dict[str, Any]]:
    """Generate JSON-Schema-valid argument dicts for ``skill``.

    ``skill`` may be either a Skill instance (anything with a
    ``.manifest.json_schema`` attribute) or a manifest-shaped object
    exposing ``.json_schema`` directly. Both forms are accepted so call
    sites can hand the strategy whichever they have to hand.

    Internally we delegate to :func:`hypothesis_jsonschema.from_schema`,
    which inspects every keyword in the schema and emits values that
    satisfy it. The strategy therefore inherits ``hypothesis_jsonschema``'s
    coverage of ``allOf`` / ``oneOf`` / ``required`` / ``enum`` /
    ``additionalProperties: false``, which is exactly what Property 2
    (CP2 schema soundness) needs.
    """

    schema = _resolve_schema(skill)
    if not isinstance(schema, Mapping):
        raise TypeError(
            "tool_call_arguments(skill): expected a Mapping json_schema, got "
            f"{type(schema).__name__}"
        )
    # ``hypothesis_jsonschema.from_schema`` requires a plain dict; a
    # ``Mapping`` (e.g., ``MappingProxyType``) would otherwise be
    # rejected. Copying is cheap and avoids forcing every caller to do
    # the conversion themselves. The library types its return as the
    # union of every JSON value; we narrow to ``dict[str, Any]`` here
    # because every Skill-manifest schema we ship is an object schema,
    # which produces dict outputs.
    return from_schema(dict(schema))  # type: ignore[return-value]


def _resolve_schema(skill: Any) -> Mapping[str, Any]:
    """Pull the JSON Schema off either a Skill or a SkillManifest-like object."""
    if hasattr(skill, "manifest"):
        manifest = skill.manifest
        if hasattr(manifest, "json_schema"):
            return manifest.json_schema  # type: ignore[no-any-return]
    if hasattr(skill, "json_schema"):
        return skill.json_schema  # type: ignore[no-any-return]
    raise AttributeError(
        "tool_call_arguments(skill): object has neither .manifest.json_schema "
        "nor .json_schema"
    )


# ---------------------------------------------------------------------------
# Memory records
# ---------------------------------------------------------------------------


# The closed set of memory categories accepted by ``MemoryRecord``.
# Mirrors the Literal in ``jarvis.memory.store.MemoryCategory`` — kept
# in sync via the unit tests in ``tests/unit/test_strategies.py``.
_MEMORY_CATEGORIES = st.sampled_from(("chat", "preference", "fact", "summary"))


def _record_ids() -> st.SearchStrategy[str]:
    """Generate record id strings."""
    # ``uuid4().hex`` matches the production producer's id shape, but
    # the dataclass itself only requires a non-empty string. Using both
    # forms gives Hypothesis room to find IDs that happen to collide in
    # ways the production producers would not.
    return st.one_of(
        st.uuids().map(lambda u: u.hex),
        st.text(alphabet=st.characters(min_codepoint=0x21, max_codepoint=0x7E),
                min_size=1, max_size=32),
    )


def _embeddings(dimension: int = 8) -> st.SearchStrategy[list[float]]:
    """Generate small fixed-size float vectors.

    The default dimension is intentionally small (eight) so generated
    records stay cheap to construct and serialise. Property tests that
    care about shape parity with the production
    :class:`~jarvis.memory.embedder.SentenceTransformerEmbedder`
    (384-dim) override this via :func:`memory_records(dimension=384)`.
    """
    if dimension <= 0:
        raise ValueError(f"dimension must be positive; got {dimension}")
    return st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=dimension,
        max_size=dimension,
    )


def memory_records(
    *,
    dimension: int = 8,
    redacted_only: bool = False,
) -> st.SearchStrategy[MemoryRecord]:
    """Generate :class:`~jarvis.memory.store.MemoryRecord` values.

    The default settings produce records that round-trip through the
    :class:`MemoryRecord` constructor (no ``__post_init__`` rejection),
    have a small embedding for cheap testing, and span every category.

    ``redacted_only=True`` forces the ``redacted`` flag on so Property
    15 (redaction containment) can quantify only over records that
    *should* have had redaction applied at write time.
    """

    return st.builds(
        MemoryRecord,
        record_id=_record_ids(),
        content=_safe_text,
        embedding=_embeddings(dimension),
        timestamp=_utc_datetimes,
        category=_MEMORY_CATEGORIES,
        provenance=st.dictionaries(
            keys=st.text(min_size=1, max_size=16),
            values=st.one_of(
                st.text(max_size=32),
                st.integers(min_value=0, max_value=1_000_000),
                st.booleans(),
            ),
            max_size=4,
        ),
        redacted=st.just(True) if redacted_only else st.booleans(),
    )


# ---------------------------------------------------------------------------
# Reminder sets
# ---------------------------------------------------------------------------


# A reminder triple as fed to :class:`ReminderService.add` /
# :class:`ReminderService.add_timer`. We keep the strategy library-free
# (no direct dependency on ``ReminderService``) so these tuples are
# usable both by the property test and by the mock implementations
# wired up in 21.11.
ReminderTriple = tuple[datetime, int, str]


def reminder_sets(
    *,
    min_size: int = 1,
    max_size: int = 8,
) -> st.SearchStrategy[list[ReminderTriple]]:
    """Generate sets of reminder triples with strictly ordered ``(trigger_at, seq)`` keys.

    Each triple is ``(trigger_at, seq, label)``. The list is sorted by
    ``(trigger_at, seq)`` ascending and is guaranteed to have a strict
    total order on those keys: every pair *(t_i, s_i)* differs from
    every other pair on at least one component. This is the precondition
    Property 10 / CP13 quantifies over.
    """

    if min_size < 1:
        raise ValueError(f"min_size must be >= 1; got {min_size}")
    if max_size < min_size:
        raise ValueError(f"max_size ({max_size}) < min_size ({min_size})")

    base_dt = datetime(2024, 1, 1, tzinfo=UTC)

    @st.composite
    def _build(draw: st.DrawFn) -> list[ReminderTriple]:
        n = draw(st.integers(min_value=min_size, max_value=max_size))
        # Generate strictly increasing offsets in milliseconds so every
        # ``trigger_at`` is unique. Strict ordering on ``trigger_at``
        # alone implies strict ordering on ``(trigger_at, seq)``
        # regardless of how ``seq`` is chosen.
        offsets_ms = draw(
            st.lists(
                st.integers(min_value=0, max_value=10_000_000),
                min_size=n,
                max_size=n,
                unique=True,
            )
        )
        # Pair each offset with a unique sequence number.
        seqs = draw(
            st.lists(
                st.integers(min_value=1, max_value=10_000),
                min_size=n,
                max_size=n,
                unique=True,
            )
        )
        labels = draw(st.lists(_non_empty_text, min_size=n, max_size=n))
        triples: list[ReminderTriple] = []
        for offset_ms, seq, label in zip(offsets_ms, seqs, labels, strict=True):
            triples.append(
                (base_dt + timedelta(milliseconds=offset_ms), seq, label)
            )
        triples.sort(key=lambda t: (t[0], t[1]))
        return triples

    return _build()


# ---------------------------------------------------------------------------
# PII corpus
# ---------------------------------------------------------------------------


# A small set of literal PII samples that match the
# :data:`jarvis.memory.redactor.DEFAULT_PATTERNS`. Hypothesis combines
# them with surrounding text so every emitted string carries at least
# one match the redactor MUST scrub. The samples are intentionally
# diverse (multiple emails, phone formats, and PAN lengths) so
# Property 15 covers the breadth of the default regex set.
PII_SAMPLES: tuple[tuple[str, str], ...] = (
    # Emails
    ("email", "alice@example.com"),
    ("email", "bob.smith+spam@subdomain.example.co.uk"),
    # North-American phone numbers (3-3-4) — the redactor's default
    # pattern accepts dash, space, or no separator.
    ("phone", "555-123-4567"),
    ("phone", "555 123 4567"),
    ("phone", "5551234567"),
    # Credit-card PANs — Visa-16, Amex-15, Diners-14.
    ("credit_card", "4111 1111 1111 1111"),
    ("credit_card", "378282246310005"),
    ("credit_card", "30569309025904"),
)


def pii_corpus() -> st.SearchStrategy[str]:
    """Generate strings containing at least one PII match.

    Each example interleaves :data:`PII_SAMPLES` with neutral free-form
    text so the redactor sees realistic-shaped inputs (PII embedded in
    a sentence, not a bare token). The neutral text comes from
    :data:`_safe_text` and is constrained to printable BMP characters
    so a logging-handler choke point is unlikely to mask a redactor
    failure during the test.
    """

    @st.composite
    def _build(draw: st.DrawFn) -> str:
        # Pick at least one sample so the string is *guaranteed* to
        # contain a PII match — Property 15 only quantifies over inputs
        # that match the redactor's patterns.
        n_samples = draw(st.integers(min_value=1, max_value=3))
        chunks: list[str] = []
        for i in range(n_samples):
            if i:
                chunks.append(draw(_safe_text))
            _kind, sample = draw(st.sampled_from(PII_SAMPLES))
            chunks.append(sample)
        chunks.append(draw(_safe_text))
        return " ".join(chunks)

    # Cross-check that PII_SAMPLES stays consistent with the redactor's
    # default kinds. If someone updates DEFAULT_PATTERNS without
    # updating this catalog, the assertion fires at strategy construction
    # rather than during a test, which is much easier to debug.
    _expected_kinds = {kind for kind, _ in DEFAULT_PATTERNS}
    _sample_kinds = {kind for kind, _ in PII_SAMPLES}
    assert _sample_kinds <= _expected_kinds, (
        "PII_SAMPLES references kinds outside the default redactor patterns "
        f"({_sample_kinds - _expected_kinds!r}); update one or the other."
    )

    return _build()


# ---------------------------------------------------------------------------
# Mistral tool payloads
# ---------------------------------------------------------------------------


def mistral_tool_payloads() -> st.SearchStrategy[dict[str, Any]]:
    """Generate Mistral function-definition dicts.

    The output shape mirrors :meth:`MistralSchemaValidator.to_mistral_tool`:

    .. code-block:: python

        {
            "type": "function",
            "function": {
                "name": <str>,
                "description": <str>,
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...],
                    "additionalProperties": false,
                },
            },
        }

    Every generated parameters block is a flat ``object`` schema using
    only Mistral-supported scalar types (``string``, ``number``,
    ``integer``, ``boolean``). This keeps the strategy aligned with
    Property 12's invariants — ``parameters.type == "object"``, no
    unsupported keywords, JSON round-trip safe — without over-constraining
    the search space.
    """

    # Function names must be non-empty identifiers. Mistral accepts the
    # JSON Schema convention (alphanumerics + underscore + hyphen),
    # which we encode as a regex-shaped string strategy.
    name_strategy = st.from_regex(r"[A-Za-z][A-Za-z0-9_]{0,31}", fullmatch=True)
    # Property names are the same shape — flat, identifier-like.
    prop_name_strategy = st.from_regex(r"[a-z][a-z0-9_]{0,15}", fullmatch=True)

    scalar_property_strategy = st.fixed_dictionaries(
        {
            "type": st.sampled_from(["string", "number", "integer", "boolean"]),
            "description": st.text(max_size=64),
        }
    )

    @st.composite
    def _build(draw: st.DrawFn) -> dict[str, Any]:
        prop_names = draw(
            st.lists(prop_name_strategy, min_size=0, max_size=4, unique=True)
        )
        properties = {name: draw(scalar_property_strategy) for name in prop_names}
        # ``required`` is a (possibly-empty) subset of the property names,
        # ordered for deterministic equality across runs.
        if prop_names:
            required_count = draw(st.integers(min_value=0, max_value=len(prop_names)))
            required = sorted(prop_names[:required_count])
        else:
            required = []
        parameters: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
        return {
            "type": "function",
            "function": {
                "name": draw(name_strategy),
                "description": draw(st.text(max_size=128)),
                "parameters": parameters,
            },
        }

    return _build()
