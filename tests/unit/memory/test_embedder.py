"""Unit and property tests for ``jarvis.memory.embedder``.

These tests exercise the public contract of the :class:`Embedder` Protocol
through the dependency-free :class:`HashEmbedder` test double, plus a
guarded smoke test for :class:`SentenceTransformerEmbedder` that auto-skips
when the heavy ``sentence-transformers`` dependency is not installed in
the active environment.

Validates: Requirements 10.3, 10.4
"""

from __future__ import annotations

import importlib.util
import math

from hypothesis import HealthCheck, given, settings, strategies as st
import pytest

from jarvis.memory.embedder import (
    DEFAULT_EMBEDDING_MODEL,
    MINI_LM_DIMENSION,
    Embedder,
    HashEmbedder,
    SentenceTransformerEmbedder,
    create_default_embedder,
)

# ---------------------------------------------------------------------------
# Module-level constants and Protocol conformance
# ---------------------------------------------------------------------------


def test_default_model_name_matches_design_doc() -> None:
    # design.md §Memory_Store and MemoryConfig.embedding_model both pin
    # the default to the all-MiniLM-L6-v2 sentence-transformers model.
    assert DEFAULT_EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"


def test_mini_lm_dimension_is_384() -> None:
    # Hard-coded as a published-model constant; HashEmbedder relies on it
    # to be shape-compatible with the production backend by default.
    assert MINI_LM_DIMENSION == 384


def test_hash_embedder_satisfies_embedder_protocol() -> None:
    # The Protocol is runtime_checkable, so isinstance is enough to
    # confirm the structural attributes/methods line up. This guards
    # against accidental signature drift in either direction.
    e = HashEmbedder()
    assert isinstance(e, Embedder)


# ---------------------------------------------------------------------------
# HashEmbedder construction
# ---------------------------------------------------------------------------


def test_hash_embedder_default_dimension_matches_mini_lm() -> None:
    e = HashEmbedder()
    assert e.dimension == MINI_LM_DIMENSION


def test_hash_embedder_custom_dimension_is_honored() -> None:
    e = HashEmbedder(dimension=16)
    assert e.dimension == 16
    assert len(e.embed("hello")) == 16


def test_hash_embedder_default_model_name_is_distinct_from_production() -> None:
    # Tests must not accidentally compare HashEmbedder vectors to
    # SentenceTransformerEmbedder vectors as if they came from the same
    # space; a clearly-distinct default name makes that mistake louder.
    e = HashEmbedder()
    assert e.model_name == "hash-embedder-v1"
    assert e.model_name != DEFAULT_EMBEDDING_MODEL


@pytest.mark.parametrize("bad", [0, -1, -384, 0.5, "12", None])
def test_hash_embedder_rejects_non_positive_or_non_int_dimension(bad: object) -> None:
    with pytest.raises(ValueError):
        HashEmbedder(dimension=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["", 0, None])
def test_hash_embedder_rejects_invalid_model_name(bad: object) -> None:
    with pytest.raises(ValueError):
        HashEmbedder(model_name=bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# HashEmbedder.embed: shape, type, and edge cases
# ---------------------------------------------------------------------------


def test_embed_returns_list_of_floats_of_correct_length() -> None:
    e = HashEmbedder(dimension=32)
    v = e.embed("the quick brown fox")
    assert isinstance(v, list)
    assert len(v) == 32
    assert all(isinstance(x, float) for x in v)


def test_embed_empty_string_is_valid_unit_vector() -> None:
    # Empty input is part of the documented contract: the embedding of ""
    # is well-defined (non-zero keystream, so non-zero norm).
    e = HashEmbedder(dimension=64)
    v = e.embed("")
    assert len(v) == 64
    norm = math.sqrt(sum(x * x for x in v))
    assert math.isclose(norm, 1.0, abs_tol=1e-9)


def test_embed_unicode_text_is_handled() -> None:
    # UTF-8 encoding inside _embed_one means non-ASCII text round-trips
    # without raising.
    e = HashEmbedder(dimension=32)
    v = e.embed("héllo, 世界 🌍")
    assert len(v) == 32


def test_embed_rejects_non_string_input() -> None:
    e = HashEmbedder(dimension=8)
    with pytest.raises(TypeError):
        e.embed(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        e.embed(b"bytes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Determinism (CP3 underpinning)
# ---------------------------------------------------------------------------


def test_embed_is_deterministic_within_an_instance() -> None:
    e = HashEmbedder(dimension=32)
    v1 = e.embed("repeat me")
    v2 = e.embed("repeat me")
    assert v1 == v2


def test_embed_is_deterministic_across_instances_with_same_config() -> None:
    # CP3 requires retrieval ordering to be stable across calls within a
    # session, which in turn requires the embedding function to be
    # stable across embedder instances created with the same config.
    a = HashEmbedder(dimension=32, model_name="fixture")
    b = HashEmbedder(dimension=32, model_name="fixture")
    assert a.embed("alpha") == b.embed("alpha")


def test_different_model_names_produce_different_vectors() -> None:
    # The seed prefix mixes in model_name; two embedders configured
    # differently must produce distinct vectors so tests can construct
    # an "unrelated" embedder when they need one.
    a = HashEmbedder(dimension=32, model_name="space-a")
    b = HashEmbedder(dimension=32, model_name="space-b")
    assert a.embed("alpha") != b.embed("alpha")


# ---------------------------------------------------------------------------
# Distinctness
# ---------------------------------------------------------------------------


def test_distinct_inputs_produce_distinct_vectors() -> None:
    e = HashEmbedder(dimension=32)
    seen: set[tuple[float, ...]] = set()
    for s in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]:
        seen.add(tuple(e.embed(s)))
    assert len(seen) == 6


# ---------------------------------------------------------------------------
# embed_batch contract
# ---------------------------------------------------------------------------


def test_embed_batch_empty_returns_empty_list() -> None:
    e = HashEmbedder(dimension=8)
    assert e.embed_batch([]) == []


def test_embed_batch_preserves_order() -> None:
    e = HashEmbedder(dimension=16)
    inputs = ["one", "two", "three", "four"]
    batched = e.embed_batch(inputs)
    assert [batched[i] for i in range(len(inputs))] == [e.embed(t) for t in inputs]


def test_embed_batch_single_matches_embed() -> None:
    # Batch / single equivalence invariant from the Protocol docstring.
    e = HashEmbedder(dimension=16)
    assert e.embed_batch(["only"])[0] == e.embed("only")


def test_embed_batch_rejects_non_list_input() -> None:
    e = HashEmbedder(dimension=8)
    with pytest.raises(TypeError):
        e.embed_batch(("a", "b"))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        e.embed_batch("ab")  # type: ignore[arg-type]


def test_embed_batch_rejects_non_string_entries() -> None:
    e = HashEmbedder(dimension=8)
    with pytest.raises(TypeError):
        e.embed_batch(["ok", 42])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------


# Reasonable text strategy: full Unicode strings of bounded length so
# Hypothesis spends its budget on structural variety rather than chasing
# pathological multi-MB inputs.
_TEXT = st.text(min_size=0, max_size=200)


@given(_TEXT)
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_embed_is_deterministic(text: str) -> None:
    """For any text, two consecutive embeds return the exact same vector."""
    e = HashEmbedder(dimension=64)
    assert e.embed(text) == e.embed(text)


@given(_TEXT)
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_embed_has_unit_norm(text: str) -> None:
    """L2 norm of every embedding equals 1.0 within IEEE-754 rounding."""
    e = HashEmbedder(dimension=64)
    v = e.embed(text)
    norm = math.sqrt(sum(x * x for x in v))
    assert math.isclose(norm, 1.0, abs_tol=1e-9)


@given(_TEXT)
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_embed_dimension_is_constant(text: str) -> None:
    """Every embedding has length exactly :attr:`Embedder.dimension`."""
    e = HashEmbedder(dimension=37)  # off-the-published-model number
    v = e.embed(text)
    assert len(v) == 37


@given(st.lists(_TEXT, min_size=0, max_size=8))
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_batch_matches_individual_embeds(texts: list[str]) -> None:
    """``embed_batch`` is element-wise equal to per-text ``embed`` calls."""
    e = HashEmbedder(dimension=32)
    assert e.embed_batch(texts) == [e.embed(t) for t in texts]


@given(st.text(min_size=1, max_size=50), st.text(min_size=1, max_size=50))
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_distinct_texts_yield_distinct_vectors(a: str, b: str) -> None:
    """Distinct inputs produce distinct vectors (no SHA-256 collision)."""
    if a == b:
        return
    e = HashEmbedder(dimension=64)
    assert e.embed(a) != e.embed(b)


# ---------------------------------------------------------------------------
# create_default_embedder
# ---------------------------------------------------------------------------


_HAS_SENTENCE_TRANSFORMERS = importlib.util.find_spec("sentence_transformers") is not None


@pytest.mark.skipif(
    not _HAS_SENTENCE_TRANSFORMERS,
    reason="sentence-transformers not installed in this environment.",
)
def test_create_default_embedder_returns_sentence_transformer_instance() -> None:
    # We only verify construction and shape; a full encode would download
    # ~80MB of weights on first run, which is excessive for unit tests.
    # If this test runs slow on a fresh CI box that is the model download,
    # not embedder logic — see the integration tests for end-to-end runs.
    e = create_default_embedder()
    assert isinstance(e, SentenceTransformerEmbedder)
    assert e.model_name == DEFAULT_EMBEDDING_MODEL
    assert e.dimension == MINI_LM_DIMENSION
    # Smoke encode to confirm the lazy import path produced a working model.
    v = e.embed("hello")
    assert len(v) == MINI_LM_DIMENSION
    norm = math.sqrt(sum(x * x for x in v))
    assert math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3)


def test_sentence_transformer_embedder_raises_runtime_error_when_dep_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``sentence_transformers`` cannot be imported, raise a clear error.

    This test simulates the missing-dependency path even when the package
    *is* installed, by interposing a failing import in ``sys.modules``.
    """
    import builtins  # noqa: PLC0415

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("simulated absence of sentence-transformers")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="sentence-transformers"):
        SentenceTransformerEmbedder()
