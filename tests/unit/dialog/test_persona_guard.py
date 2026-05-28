"""Unit tests for ``jarvis.dialog.persona_guard``.

Covers Requirement 11.5: forbidden self-references in assistant text are
either rewritten or flagged for regeneration before being forwarded to TTS.
"""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.dialog.persona_guard import PersonaGuard, PersonaLike

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakePersona:
    """Duck-typed stand-in for ``PersonaProfile`` (parallel task 13.2)."""

    name: str = "JARVIS"
    forbidden_self_refs: tuple[str, ...] = (
        "ChatGPT",
        "Claude",
        "as an AI language model",
        "as a large language model",
        "GPT-4",
    )


def _persona(**overrides: object) -> _FakePersona:
    return _FakePersona(**overrides)  # type: ignore[arg-type]


def test_fake_persona_satisfies_persona_like_protocol() -> None:
    persona: PersonaLike = _persona()
    assert isinstance(persona, PersonaLike)


# ---------------------------------------------------------------------------
# Clean text passes through unchanged
# ---------------------------------------------------------------------------


def test_clean_text_returns_unchanged_and_not_violated() -> None:
    guard = PersonaGuard()
    text = "Of course, sir. The weather in London is mild this evening."

    rewritten, violated = guard.check(text, _persona())

    assert rewritten == text
    assert violated is False


def test_empty_text_is_handled_gracefully() -> None:
    guard = PersonaGuard()

    rewritten, violated = guard.check("", _persona())

    assert rewritten == ""
    assert violated is False


def test_persona_with_no_forbidden_phrases_never_violates() -> None:
    guard = PersonaGuard()
    persona = _persona(forbidden_self_refs=())
    text = "I am ChatGPT, a large language model."

    rewritten, violated = guard.check(text, persona)

    assert rewritten == text
    assert violated is False


# ---------------------------------------------------------------------------
# Detection and rewriting
# ---------------------------------------------------------------------------


def test_detects_chatgpt_self_reference_and_rewrites_to_persona_name() -> None:
    guard = PersonaGuard()
    text = "Hello, I am ChatGPT, here to help."

    rewritten, violated = guard.check(text, _persona())

    assert violated is True
    assert "ChatGPT" not in rewritten
    assert "JARVIS" in rewritten


def test_detects_claude_self_reference() -> None:
    guard = PersonaGuard()
    text = "I'm Claude, an AI assistant made by Anthropic."

    rewritten, violated = guard.check(text, _persona())

    assert violated is True
    assert "Claude" not in rewritten
    assert "JARVIS" in rewritten


def test_detection_is_case_insensitive() -> None:
    guard = PersonaGuard()
    text = "well, chatgpt would say otherwise."

    rewritten, violated = guard.check(text, _persona())

    assert violated is True
    assert "chatgpt" not in rewritten.lower() or "JARVIS" in rewritten


def test_disclaimer_phrase_rewritten_with_as_prefix_for_grammaticality() -> None:
    guard = PersonaGuard()
    text = "As an AI language model, I cannot have opinions."

    rewritten, violated = guard.check(text, _persona())

    assert violated is True
    # The rewrite should keep the sentence grammatical: "As JARVIS, ..."
    assert rewritten.lower().startswith("as jarvis,")
    assert "language model" not in rewritten.lower()


def test_multiple_forbidden_phrases_all_rewritten_in_one_pass() -> None:
    guard = PersonaGuard()
    text = "I am ChatGPT. As a large language model, I cannot. — Claude"

    rewritten, violated = guard.check(text, _persona())

    assert violated is True
    assert "ChatGPT" not in rewritten
    assert "Claude" not in rewritten
    assert "language model" not in rewritten.lower()
    assert rewritten.count("JARVIS") >= 2


def test_repeated_phrase_replaced_everywhere() -> None:
    guard = PersonaGuard()
    text = "ChatGPT here. ChatGPT helps. ChatGPT signs off."

    rewritten, violated = guard.check(text, _persona())

    assert violated is True
    assert "ChatGPT" not in rewritten
    assert rewritten.count("JARVIS") == 3


# ---------------------------------------------------------------------------
# Word-boundary semantics: avoid false positives on partial matches
# ---------------------------------------------------------------------------


def test_phrase_inside_larger_word_does_not_trigger_alone() -> None:
    """``GPT-4`` should match, but the word ``Claudette`` should not match
    ``Claude`` because of the alphanumeric word boundary."""
    guard = PersonaGuard()
    text = "Claudette baked madeleines this afternoon."

    rewritten, violated = guard.check(text, _persona())

    assert violated is False
    assert rewritten == text


def test_punctuation_around_phrase_still_matches() -> None:
    guard = PersonaGuard()
    text = "(ChatGPT) said hello."

    rewritten, violated = guard.check(text, _persona())

    assert violated is True
    assert "ChatGPT" not in rewritten


def test_phrase_with_hyphen_is_matched_as_literal() -> None:
    guard = PersonaGuard()
    text = "Powered by GPT-4 under the hood."

    rewritten, violated = guard.check(text, _persona())

    assert violated is True
    assert "GPT-4" not in rewritten
    assert "JARVIS" in rewritten


# ---------------------------------------------------------------------------
# Persona name customisation
# ---------------------------------------------------------------------------


def test_custom_persona_name_used_in_substitution() -> None:
    guard = PersonaGuard()
    persona = _persona(name="FRIDAY")
    text = "I am ChatGPT, ready to help."

    rewritten, violated = guard.check(text, persona)

    assert violated is True
    assert "FRIDAY" in rewritten
    assert "JARVIS" not in rewritten


def test_blank_persona_name_falls_back_to_generic_label() -> None:
    """An empty persona name should not produce ``"I am , ready"``."""
    guard = PersonaGuard()
    persona = _persona(name="")
    text = "I am ChatGPT."

    rewritten, violated = guard.check(text, persona)

    assert violated is True
    # Should contain *some* substitution, not an empty hole.
    assert "ChatGPT" not in rewritten
    assert rewritten.strip() != "I am ."


# ---------------------------------------------------------------------------
# Idempotence: rewriting a rewritten text is a no-op
# ---------------------------------------------------------------------------


def test_guard_is_idempotent_on_rewritten_text() -> None:
    guard = PersonaGuard()
    persona = _persona()
    text = "I am ChatGPT. As an AI language model, I help."

    once, violated_once = guard.check(text, persona)
    twice, violated_twice = guard.check(once, persona)

    assert violated_once is True
    assert violated_twice is False
    assert twice == once
