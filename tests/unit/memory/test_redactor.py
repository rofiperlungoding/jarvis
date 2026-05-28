"""Unit tests for ``jarvis.memory.redactor``.

Covers:
    * Default-pattern coverage for emails, phones, and credit cards.
    * Replacement token format ``[REDACTED:<kind>]``.
    * Custom pattern lists and mappings.
    * Pattern application order is the declared order.
    * Empty / no-match / no-pattern fast paths.
    * ``from_config_patterns`` helper for the flat list shape used in TOML.
    * Constructor validation: bad shapes, non-string regex, empty kind.
    * Invalid regex compilation surfaces ``re.error``.

Validates: Requirements 10.8
"""

from __future__ import annotations

import re

import pytest

from jarvis.memory.redactor import DEFAULT_PATTERNS, PIIRedactor

# ---------------------------------------------------------------------------
# Default behavior
# ---------------------------------------------------------------------------


def test_default_redactor_redacts_email() -> None:
    redactor = PIIRedactor()
    out = redactor.redact("Mail me at alice.smith+tag@example.co.uk please.")
    assert "alice" not in out
    assert "example" not in out
    assert "[REDACTED:email]" in out


def test_default_redactor_redacts_phone_with_dashes() -> None:
    redactor = PIIRedactor()
    out = redactor.redact("Call me at 555-123-4567 tomorrow.")
    assert "555" not in out
    assert "[REDACTED:phone]" in out


def test_default_redactor_redacts_phone_with_spaces() -> None:
    redactor = PIIRedactor()
    out = redactor.redact("Call 555 123 4567.")
    assert "555" not in out
    assert "[REDACTED:phone]" in out


def test_default_redactor_redacts_credit_card_with_spaces() -> None:
    redactor = PIIRedactor()
    out = redactor.redact("Card: 4111 1111 1111 1111 expires soon.")
    assert "4111" not in out
    assert "[REDACTED:credit_card]" in out


def test_default_redactor_redacts_credit_card_with_dashes() -> None:
    redactor = PIIRedactor()
    out = redactor.redact("Card: 4111-1111-1111-1111.")
    assert "1111" not in out
    assert "[REDACTED:credit_card]" in out


def test_default_redactor_redacts_credit_card_continuous() -> None:
    redactor = PIIRedactor()
    out = redactor.redact("Card 4111111111111111.")
    assert "4111111111111111" not in out
    assert "[REDACTED:credit_card]" in out


def test_default_redactor_handles_multiple_kinds_in_one_string() -> None:
    redactor = PIIRedactor()
    text = (
        "Reach me at bob@x.io or 555-987-6543; "
        "the card was 4242 4242 4242 4242."
    )
    out = redactor.redact(text)
    assert "bob@x.io" not in out
    assert "555-987-6543" not in out
    assert "4242" not in out
    assert "[REDACTED:email]" in out
    assert "[REDACTED:phone]" in out
    assert "[REDACTED:credit_card]" in out


def test_default_redactor_leaves_non_pii_text_unchanged() -> None:
    redactor = PIIRedactor()
    text = "The weather is fine today, sir. Shall I open the curtains?"
    assert redactor.redact(text) == text


def test_default_redactor_returns_empty_for_empty_input() -> None:
    redactor = PIIRedactor()
    assert redactor.redact("") == ""


def test_default_patterns_constant_is_tuple_of_pairs() -> None:
    assert isinstance(DEFAULT_PATTERNS, tuple)
    for entry in DEFAULT_PATTERNS:
        assert isinstance(entry, tuple) and len(entry) == 2
        kind, regex = entry
        assert isinstance(kind, str) and kind
        assert isinstance(regex, str)
        # All defaults must compile.
        re.compile(regex)


# ---------------------------------------------------------------------------
# Custom patterns
# ---------------------------------------------------------------------------


def test_custom_patterns_as_list_of_tuples() -> None:
    redactor = PIIRedactor([("ssn", r"\b\d{3}-\d{2}-\d{4}\b")])
    out = redactor.redact("SSN: 123-45-6789.")
    assert "123-45-6789" not in out
    assert "[REDACTED:ssn]" in out


def test_custom_patterns_as_mapping() -> None:
    redactor = PIIRedactor({"badge": r"BADGE-\d+"})
    out = redactor.redact("Token BADGE-42 issued.")
    assert "BADGE-42" not in out
    assert "[REDACTED:badge]" in out


def test_replacement_format_is_exactly_redacted_kind() -> None:
    redactor = PIIRedactor([("name", r"\bAlice\b")])
    out = redactor.redact("Alice waved.")
    assert out == "[REDACTED:name] waved."


def test_patterns_are_applied_in_declared_order() -> None:
    # ``digits`` is declared before ``three_digits`` so the broader pattern
    # consumes the digit run first, leaving the second pattern with nothing
    # to match.
    redactor = PIIRedactor(
        [
            ("digits", r"\d+"),
            ("three_digits", r"\d{3}"),
        ]
    )
    out = redactor.redact("code 123456 end")
    assert out == "code [REDACTED:digits] end"
    assert "[REDACTED:three_digits]" not in out


def test_kinds_property_reports_application_order() -> None:
    redactor = PIIRedactor([("a", r"a"), ("b", r"b")])
    assert redactor.kinds == ("a", "b")


def test_len_reflects_pattern_count() -> None:
    assert len(PIIRedactor([])) == 0
    assert len(PIIRedactor([("k", r"x")])) == 1
    assert len(PIIRedactor()) == len(DEFAULT_PATTERNS)


def test_empty_pattern_list_is_pass_through() -> None:
    redactor = PIIRedactor([])
    text = "anything goes 123 here@example.com"
    assert redactor.redact(text) == text


# ---------------------------------------------------------------------------
# from_config_patterns helper
# ---------------------------------------------------------------------------


def test_from_config_patterns_synthesizes_indexed_kinds() -> None:
    redactor = PIIRedactor.from_config_patterns(
        [r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", r"\b\d{3}-\d{3}-\d{4}\b"]
    )
    assert redactor.kinds == ("pii_1", "pii_2")
    out = redactor.redact("a@b.io and 555-123-4567")
    assert "[REDACTED:pii_1]" in out
    assert "[REDACTED:pii_2]" in out


def test_from_config_patterns_custom_prefix() -> None:
    redactor = PIIRedactor.from_config_patterns([r"x"], prefix="custom")
    assert redactor.kinds == ("custom_1",)


def test_from_config_patterns_rejects_non_string_entries() -> None:
    with pytest.raises(TypeError):
        PIIRedactor.from_config_patterns([r"ok", 42])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_constructor_rejects_non_iterable() -> None:
    with pytest.raises(TypeError):
        PIIRedactor(42)  # type: ignore[arg-type]


def test_constructor_rejects_malformed_tuple() -> None:
    with pytest.raises(TypeError):
        PIIRedactor([("only-one-element",)])  # type: ignore[list-item]


def test_constructor_rejects_non_string_kind() -> None:
    with pytest.raises(TypeError):
        PIIRedactor([(123, r"x")])  # type: ignore[list-item]


def test_constructor_rejects_non_string_regex() -> None:
    with pytest.raises(TypeError):
        PIIRedactor([("k", 123)])  # type: ignore[list-item]


def test_constructor_rejects_empty_kind_label() -> None:
    with pytest.raises(ValueError):
        PIIRedactor([("", r"x")])


def test_constructor_propagates_regex_compile_error() -> None:
    with pytest.raises(re.error):
        PIIRedactor([("bad", r"(unclosed")])


def test_redact_rejects_non_string_input() -> None:
    redactor = PIIRedactor()
    with pytest.raises(TypeError):
        redactor.redact(b"bytes are not allowed")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_with_defaults_is_equivalent_to_no_arg_constructor() -> None:
    a = PIIRedactor()
    b = PIIRedactor.with_defaults()
    assert a.kinds == b.kinds
    sample = "alice@example.com 555-123-4567"
    assert a.redact(sample) == b.redact(sample)
