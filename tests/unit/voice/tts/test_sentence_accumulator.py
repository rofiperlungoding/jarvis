"""Unit tests for :class:`jarvis.voice.tts.base.SentenceAccumulator`.

The accumulator slices the LLM token stream into whole sentences so the
TTS_Engine can begin synthesis as soon as the first sentence boundary is
observed. These tests exercise:

* Basic boundary detection on ``.`` / ``?`` / ``!`` followed by whitespace.
* Abbreviation handling so titles (``Dr.``, ``Mr.``...) and Latin
  shortcuts (``e.g.``, ``i.e.``, ``etc.``, ``Ph.D.``, ``U.S.``,
  ``a.m.``, ``p.m.``) do not produce mid-sentence breaks.
* Streaming behaviour with arbitrarily small / character-by-character
  deltas — the boundary must be detected only when the terminator and
  its trailing whitespace have actually arrived.
* CJK fullwidth terminators (fullwidth period, question mark, exclamation,
  and horizontal ellipsis) which stand alone with no required trailing
  whitespace.
* :meth:`SentenceAccumulator.flush` draining the trailing partial
  sentence on stream completion.
* Empty / whitespace-only inputs being safe no-ops.
* The ``extra_abbreviations`` constructor argument extending (not
  replacing) the built-in abbreviation set.

Validates: Requirements 12.2, 19.5
"""

from __future__ import annotations

import pytest

from jarvis.voice.tts.base import SentenceAccumulator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _feed_char_by_char(acc: SentenceAccumulator, text: str) -> list[str]:
    """Feed ``text`` one character at a time and return the concatenated emissions.

    Mimics the worst-case Mistral streaming pattern where each token may
    be a single character. The accumulator must produce the exact same
    sentence list regardless of how the input is chunked.
    """
    out: list[str] = []
    for ch in text:
        out.extend(acc.feed(ch))
    return out


# ---------------------------------------------------------------------------
# Basic boundary detection
# ---------------------------------------------------------------------------


def test_simple_period_split() -> None:
    """A period followed by whitespace ends a sentence."""
    acc = SentenceAccumulator()
    sentences = acc.feed("Hello world. How are you? I am fine!")
    # The trailing "I am fine!" has no whitespace after the terminator,
    # so it stays buffered until a further feed/flush.
    assert sentences == ["Hello world.", "How are you?"]
    assert acc.pending.strip() == "I am fine!"


def test_question_mark_and_exclamation_split() -> None:
    acc = SentenceAccumulator()
    sentences = acc.feed("Really? Yes! Of course. ")
    assert sentences == ["Really?", "Yes!", "Of course."]
    assert acc.pending == ""


def test_multiple_sentences_in_single_feed() -> None:
    """All sentences fully terminated within one feed are emitted at once."""
    acc = SentenceAccumulator()
    sentences = acc.feed("One. Two. Three. Four. ")
    assert sentences == ["One.", "Two.", "Three.", "Four."]
    assert acc.pending == ""


def test_period_without_trailing_whitespace_buffers() -> None:
    """A terminator at end-of-buffer is NOT a boundary — more text may arrive."""
    acc = SentenceAccumulator()
    sentences = acc.feed("Done.")
    assert sentences == []
    assert acc.pending == "Done."


def test_leading_whitespace_is_stripped_from_emissions() -> None:
    """Whitespace between sentences is consumed, not preserved on the next one."""
    acc = SentenceAccumulator()
    sentences = acc.feed("First.    Second.\t\nThird. ")
    assert sentences == ["First.", "Second.", "Third."]


def test_only_whitespace_feed_buffers_nothing_meaningful() -> None:
    acc = SentenceAccumulator()
    assert acc.feed("   \n\t  ") == []
    # The internal buffer holds the whitespace, but flush yields nothing.
    assert acc.flush() is None


# ---------------------------------------------------------------------------
# Abbreviation handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "abbreviation, follow_up",
    [
        ("Dr.", "Smith arrived."),
        ("Mr.", "Jones is here."),
        ("Mrs.", "Doe waved."),
        ("Ms.", "Lee responded."),
        ("Prof.", "Brown lectured."),
    ],
)
def test_honorifics_do_not_split(abbreviation: str, follow_up: str) -> None:
    """Honorific titles followed by a name must NOT terminate the sentence."""
    acc = SentenceAccumulator()
    text = f"{abbreviation} {follow_up} "
    sentences = acc.feed(text)
    # Single sentence, including the abbreviation's period.
    assert sentences == [f"{abbreviation} {follow_up}"]
    assert acc.pending == ""


@pytest.mark.parametrize(
    "abbreviation",
    ["e.g.", "i.e.", "etc.", "Ph.D.", "U.S.", "a.m.", "p.m."],
)
def test_multidot_abbreviations_do_not_split(abbreviation: str) -> None:
    """Multi-period abbreviations must round-trip without producing breaks."""
    acc = SentenceAccumulator()
    sentences = acc.feed(f"See {abbreviation} the manual. ")
    assert sentences == [f"See {abbreviation} the manual."]
    assert acc.pending == ""


def test_etc_at_clause_end_does_not_split_prematurely() -> None:
    """``etc.`` mid-sentence should not split before the real boundary."""
    acc = SentenceAccumulator()
    sentences = acc.feed("Apples, oranges, etc. are fruit. ")
    assert sentences == ["Apples, oranges, etc. are fruit."]


def test_abbreviation_match_is_case_insensitive() -> None:
    acc = SentenceAccumulator()
    sentences = acc.feed("DR. Watson nodded. ")
    assert sentences == ["DR. Watson nodded."]


def test_extra_abbreviations_extend_defaults() -> None:
    """The ``extra_abbreviations`` arg adds to — does not replace — defaults."""
    acc = SentenceAccumulator(extra_abbreviations=frozenset({"approx"}))
    # Custom abbreviation is honoured ...
    custom = acc.feed("The result is approx. 3.14 today. ")
    assert custom == ["The result is approx. 3.14 today."]
    # ... and built-ins still work.
    builtin = acc.feed("Dr. Watson is here. ")
    assert builtin == ["Dr. Watson is here."]


def test_unknown_abbreviation_does_split() -> None:
    """An unrecognized token before a period followed by whitespace splits."""
    acc = SentenceAccumulator()
    sentences = acc.feed("The widget. Works fine. ")
    assert sentences == ["The widget.", "Works fine."]


# ---------------------------------------------------------------------------
# Streaming / partial deltas
# ---------------------------------------------------------------------------


def test_character_by_character_matches_bulk_feed() -> None:
    """Single-character deltas yield the same result as one big feed."""
    text = "Hello, sir. How may I help you? I am ready. "

    bulk = SentenceAccumulator()
    bulk_out = bulk.feed(text)

    streamed = SentenceAccumulator()
    stream_out = _feed_char_by_char(streamed, text)

    assert bulk_out == stream_out
    assert bulk.pending == streamed.pending == ""


def test_boundary_emerges_only_when_whitespace_arrives() -> None:
    """A pending ``.`` is buffered until a whitespace character is fed."""
    acc = SentenceAccumulator()
    assert acc.feed("Done") == []
    assert acc.feed(".") == []  # terminator alone — still need whitespace
    assert acc.pending == "Done."
    # The whitespace closes the boundary and emits the sentence.
    assert acc.feed(" ") == ["Done."]
    assert acc.pending == ""


def test_abbreviation_split_across_chunks() -> None:
    """Abbreviation detection works even when the token is split across deltas."""
    acc = SentenceAccumulator()
    # Stream "Dr. Smith arrived. " in awkward chunks.
    out: list[str] = []
    for chunk in ["D", "r", ".", " S", "mith ", "arr", "ived", ". "]:
        out.extend(acc.feed(chunk))
    assert out == ["Dr. Smith arrived."]
    assert acc.pending == ""


def test_partial_then_full_sentence() -> None:
    """A partial first feed leaves a buffered tail; the next feed completes it."""
    acc = SentenceAccumulator()
    assert acc.feed("This is a") == []
    assert acc.feed(" partial") == []
    assert acc.feed(" sentence. ") == ["This is a partial sentence."]
    assert acc.pending == ""


# ---------------------------------------------------------------------------
# Unicode / CJK terminators
# ---------------------------------------------------------------------------


def test_cjk_fullwidth_period_terminator() -> None:
    """``。`` is a standalone terminator — no trailing whitespace required."""
    acc = SentenceAccumulator()
    sentences = acc.feed("こんにちは。さようなら。")
    assert sentences == ["こんにちは。", "さようなら。"]
    assert acc.pending == ""


def test_cjk_fullwidth_question_and_exclamation() -> None:
    acc = SentenceAccumulator()
    sentences = acc.feed("元気ですか？はい！")  # noqa: RUF001
    assert sentences == ["元気ですか？", "はい！"]  # noqa: RUF001
    assert acc.pending == ""


def test_cjk_horizontal_ellipsis_terminator() -> None:
    """The horizontal ellipsis ``…`` terminates a sentence on its own."""
    acc = SentenceAccumulator()
    sentences = acc.feed("そして…次の話。")
    assert sentences == ["そして…", "次の話。"]
    assert acc.pending == ""


def test_mixed_cjk_and_ascii() -> None:
    acc = SentenceAccumulator()
    sentences = acc.feed("Hello. こんにちは。Bye! ")
    assert sentences == ["Hello.", "こんにちは。", "Bye!"]
    assert acc.pending == ""


# ---------------------------------------------------------------------------
# flush() tail handling
# ---------------------------------------------------------------------------


def test_flush_drains_unterminated_tail() -> None:
    """A tail without terminator is returned by flush()."""
    acc = SentenceAccumulator()
    acc.feed("First sentence. Second is incomplete")
    tail = acc.flush()
    assert tail == "Second is incomplete"
    # Buffer must be cleared even when content was returned.
    assert acc.pending == ""
    # A second flush on an empty buffer returns None.
    assert acc.flush() is None


def test_flush_clears_buffer_when_empty() -> None:
    """flush() on an empty / whitespace-only buffer returns None and clears it."""
    acc = SentenceAccumulator()
    assert acc.flush() is None
    assert acc.pending == ""

    acc.feed("   \n\t  ")
    assert acc.flush() is None
    assert acc.pending == ""


def test_flush_after_complete_sentences_returns_none() -> None:
    """If the buffer has been fully drained by feed(), flush yields nothing."""
    acc = SentenceAccumulator()
    assert acc.feed("All done. ") == ["All done."]
    assert acc.flush() is None


def test_flush_strips_surrounding_whitespace() -> None:
    """The flushed tail is trimmed of leading / trailing whitespace."""
    acc = SentenceAccumulator()
    acc.feed("Real sentence. \n  trailing tail   ")
    tail = acc.flush()
    assert tail == "trailing tail"
    assert acc.pending == ""


def test_flush_returns_unterminated_text_without_fabricating_terminator() -> None:
    """flush() must not append punctuation the speaker did not produce."""
    acc = SentenceAccumulator()
    acc.feed("no terminator here")
    tail = acc.flush()
    assert tail == "no terminator here"
    assert tail is not None and not tail.endswith(".")


# ---------------------------------------------------------------------------
# Empty / no-op behaviour
# ---------------------------------------------------------------------------


def test_empty_feed_is_noop() -> None:
    """feed("") returns [] and leaves the buffer untouched."""
    acc = SentenceAccumulator()
    assert acc.feed("") == []
    assert acc.pending == ""

    # And again after some buffered content — must not disturb the tail.
    acc.feed("partial")
    assert acc.feed("") == []
    assert acc.pending == "partial"


def test_pending_is_read_only_view() -> None:
    """``pending`` exposes the live tail; mutating it must not affect future feeds."""
    acc = SentenceAccumulator()
    acc.feed("hello")
    snapshot = acc.pending
    assert snapshot == "hello"
    # Re-binding the local name does not mutate the accumulator's buffer.
    snapshot = "MUTATED"
    assert acc.pending == "hello"
