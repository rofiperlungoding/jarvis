"""Base TTS interfaces and the streaming :class:`SentenceAccumulator`.

This module is the platform-neutral entry point for the Text-to-Speech layer
described in ``design.md §TTS_Engine``. It defines the structural
:class:`TTSEngine` Protocol that concrete adapters (Piper, ElevenLabs,
OpenAI, etc.) implement, together with the streaming-aware
:class:`SentenceAccumulator` used by :class:`~jarvis.dialog.manager.DialogManager`
to chop the Mistral token stream into whole sentences before handing them to
the TTS queue.

Two requirements drive the design here:

* **Requirement 12.2** — *"THE Dialog_Manager SHALL stream LLM_Backend tokens
  to the TTS_Engine as they arrive, beginning TTS synthesis as soon as the
  first sentence boundary is reached."* :class:`SentenceAccumulator.feed`
  yields finished sentences as soon as they are observed and keeps the
  trailing partial sentence buffered until the next chunk arrives.

* **Requirement 19.5** — *"THE Dialog_Manager SHALL stream responses via
  Mistral's streaming API and SHALL forward tokens to the TTS_Engine at
  sentence boundaries to enable progressive synthesis as specified in
  Requirement 12 acceptance criterion 2."* The accumulator therefore must
  not split mid-sentence on common abbreviations such as ``Dr.``, ``Mr.``,
  ``e.g.``, ``i.e.``, or ``etc.`` — doing so would emit awkward pauses in
  the rendered speech.

The Protocol is intentionally minimal: enqueue (:meth:`TTSEngine.speak`),
barge-in (:meth:`TTSEngine.stop`), liveness probe (:meth:`TTSEngine.is_playing`),
and orderly shutdown (:meth:`TTSEngine.aclose`). Concrete adapters in sibling
modules layer their backend-specific synthesis on top.

Validates: Requirements 12.2, 19.5
"""

from __future__ import annotations

from typing import Final, Protocol, runtime_checkable

__all__ = ["SentenceAccumulator", "TTSEngine"]


# ---------------------------------------------------------------------------
# TTS Engine Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TTSEngine(Protocol):
    """Structural interface for any text-to-speech backend.

    Implementations are expected to be **enqueue-and-stream**: a call to
    :meth:`speak` returns once the text has been accepted onto the playback
    queue, not once playback has finished. This matches the streaming
    contract from Requirement 12.2 / 19.5 where sentences are pushed into
    the engine as soon as they are produced by the LLM stream.

    Lifecycle:

    1. The :class:`~jarvis.dialog.manager.DialogManager` calls
       :meth:`speak` zero or more times per turn, once per finalized
       sentence emitted by :class:`SentenceAccumulator`.
    2. If the user begins speaking while playback is in progress (barge-in,
       Requirement 1.7), the audio capture loop calls :meth:`stop` to
       cancel the current utterance and drain the queue.
    3. :meth:`is_playing` is a non-blocking, synchronous probe used by the
       capture loop and Reminder_Service to decide whether the user is
       *actively* listening to the assistant.
    4. On application shutdown, :meth:`aclose` releases backend resources
       (audio device, ONNX session, HTTP client, etc.).
    """

    async def speak(self, text: str) -> None:
        """Enqueue ``text`` for synthesis and playback.

        The call MUST return promptly — typically once the text has been
        accepted onto an internal queue. It MUST NOT block until playback
        finishes; otherwise the streaming sentence-boundary contract from
        Requirement 12.2 cannot be satisfied.
        """
        ...

    async def stop(self) -> None:
        """Cancel any in-flight playback and drop pending queued text.

        Used to implement barge-in (Requirement 1.7). After ``stop``
        returns, :meth:`is_playing` MUST report ``False`` (or imminently
        report ``False`` once the audio device finishes the in-flight
        buffer, within the 150 ms barge-in budget).
        """
        ...

    def is_playing(self) -> bool:
        """Return ``True`` while the engine is actively playing audio.

        This is a synchronous, side-effect-free probe so it can be polled
        from the audio capture loop and from
        :class:`~jarvis.reminders.service.ReminderService` without
        scheduling overhead.
        """
        ...

    async def aclose(self) -> None:
        """Release backend resources.

        Implementations should be idempotent: a second call is allowed and
        MUST NOT raise. After ``aclose`` returns, no further methods on the
        engine are expected to be called.
        """
        ...


# ---------------------------------------------------------------------------
# Sentence accumulator
# ---------------------------------------------------------------------------


# Punctuation that, followed by whitespace (or end-of-input on flush),
# terminates an English sentence. The CJK fullwidth variants below are
# treated as terminators on their own — CJK text typically has no
# inter-sentence whitespace, so requiring a trailing space would suppress
# the boundary entirely.
_ASCII_TERMINATORS: Final[frozenset[str]] = frozenset(".?!")
_FULLWIDTH_TERMINATORS: Final[frozenset[str]] = frozenset("。？！…")

# Abbreviations that MUST NOT trigger a sentence break when followed by
# ``[whitespace] + capital-letter`` patterns. Stored in lowercase for
# case-insensitive matching. The set is deliberately small and English-
# centric; users with other-language workflows can extend it via the
# ``extra_abbreviations`` constructor argument.
#
# Multi-dot entries (``e.g``, ``i.e``, ``ph.d``, ``u.s``, ``u.k``, ``a.m``,
# ``p.m``) are stored *without* the trailing dot — that trailing dot is
# the candidate boundary character we are evaluating. The value before
# the boundary therefore ends one character earlier than the surface form.
_DEFAULT_ABBREVIATIONS: Final[frozenset[str]] = frozenset(
    {
        # Honorifics and titles
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "sr",
        "jr",
        "st",
        "lt",
        "capt",
        "sgt",
        "gen",
        "col",
        "rev",
        "hon",
        # Latin / scholarly
        "etc",
        "e.g",
        "i.e",
        "vs",
        "viz",
        "cf",
        "al",  # "et al."
        "ca",  # "ca." (circa)
        "ph.d",
        # Time and units
        "a.m",
        "p.m",
        "no",  # "No." (number)
        # Geographic / corporate
        "u.s",
        "u.k",
        "u.s.a",
        "inc",
        "ltd",
        "corp",
        "co",
    }
)


class SentenceAccumulator:
    """Streaming sentence boundary detector.

    The accumulator is fed arbitrary text deltas (typically Mistral
    ``content_delta`` events) and yields whole sentences once it is
    confident a boundary has been reached. The buffered tail is preserved
    across :meth:`feed` calls and can be drained on stream completion via
    :meth:`flush`.

    Boundary rules (derived from ``design.md §TTS_Engine``):

    * A sentence ends at one of ``.?!`` followed by Unicode whitespace, or
      at one of the fullwidth CJK terminators ``。？！…`` standalone.
    * A period is **not** a boundary if the trailing word (case-folded) is
      a known abbreviation such as ``Dr``, ``Mr``, ``e.g``, ``i.e``, ``etc``.
    * Question marks and exclamation marks are always boundaries — they do
      not participate in abbreviations in any common locale.
    * Whitespace immediately following the boundary punctuation is not
      preserved in the emitted sentence; the next sentence starts at the
      first non-whitespace character after the boundary.

    The implementation walks each fed chunk character-by-character so that
    arbitrary delta sizes are tolerated, including pathological one-byte
    deltas. It does not attempt to detect ellipses (``...``) — three
    sequential dots followed by whitespace will end up splitting after
    the third dot, which is acceptable rendering behaviour for TTS.
    """

    __slots__ = ("_abbreviations", "_buffer")

    def __init__(self, *, extra_abbreviations: frozenset[str] | None = None) -> None:
        """Initialize an empty accumulator.

        Parameters
        ----------
        extra_abbreviations:
            Additional case-insensitive abbreviation forms (without the
            trailing period) to recognize beyond the built-in defaults.
            Useful for domain-specific terms (e.g. ``"approx"``,
            ``"misc"``, ``"fig"``, ``"chap"``).
        """
        self._buffer: str = ""
        if extra_abbreviations is None:
            self._abbreviations: frozenset[str] = _DEFAULT_ABBREVIATIONS
        else:
            self._abbreviations = _DEFAULT_ABBREVIATIONS | frozenset(
                a.lower() for a in extra_abbreviations
            )

    # -- Public API -----------------------------------------------------------

    def feed(self, text: str) -> list[str]:
        """Append ``text`` to the buffer and return any newly-complete sentences.

        The returned list preserves order. Each entry is the trimmed
        sentence (no leading/trailing whitespace) including its terminating
        punctuation. The accumulator's internal buffer retains the
        remaining tail — typically a partially-formed next sentence —
        which will be considered on the next :meth:`feed` or :meth:`flush`.

        ``text`` may be empty; in that case an empty list is returned and
        the buffer is unchanged.
        """
        if not text:
            return []

        self._buffer += text
        sentences: list[str] = []

        # Index of the first character of the *current* sentence inside
        # ``self._buffer``. Advanced past whitespace each time a sentence
        # is emitted.
        start = self._skip_leading_whitespace(0)
        i = start
        buf = self._buffer  # local alias, hot path
        n = len(buf)

        while i < n:
            ch = buf[i]

            # Fullwidth CJK terminators stand alone — no trailing whitespace
            # is required because CJK text generally has none.
            if ch in _FULLWIDTH_TERMINATORS:
                sentences.append(buf[start : i + 1].strip())
                i += 1
                start = self._skip_leading_whitespace(i)
                i = start
                continue

            # ASCII terminator: only a boundary when followed by whitespace.
            # We deliberately do NOT treat end-of-buffer as a boundary —
            # more text may yet arrive in the next ``feed`` call. The tail
            # is drained explicitly via :meth:`flush`.
            if ch in _ASCII_TERMINATORS and i + 1 < n and buf[i + 1].isspace():
                if ch == "." and self._is_abbreviation_at(i, start):
                    i += 1
                    continue
                sentences.append(buf[start : i + 1].strip())
                i += 1
                start = self._skip_leading_whitespace(i)
                i = start
                continue

            i += 1

        self._buffer = buf[start:]
        return sentences

    def flush(self) -> str | None:
        """Return any remaining buffered tail as a final sentence.

        Called by the dialog loop once the LLM stream completes (and there
        are no further tool calls) to ensure trailing text without a
        terminator — e.g. an assistant response that does not end in
        punctuation — is still spoken. Returns ``None`` when the buffer
        is empty or contains only whitespace, and clears the buffer in
        either case.
        """
        tail = self._buffer.strip()
        self._buffer = ""
        return tail if tail else None

    @property
    def pending(self) -> str:
        """Read-only view of the currently buffered tail.

        Exposed primarily for diagnostics; production code should rely on
        :meth:`feed` and :meth:`flush` to consume buffered text.
        """
        return self._buffer

    # -- Internal -------------------------------------------------------------

    def _skip_leading_whitespace(self, idx: int) -> int:
        """Return the next index >= ``idx`` whose character is non-whitespace."""
        buf = self._buffer
        n = len(buf)
        while idx < n and buf[idx].isspace():
            idx += 1
        return idx

    def _is_abbreviation_at(self, dot_index: int, sentence_start: int) -> bool:
        """Decide whether the period at ``dot_index`` belongs to an abbreviation.

        The check walks backwards from ``dot_index - 1`` until it hits
        whitespace or the start of the current sentence, whichever comes
        first. The collected token is stripped of common leading
        punctuation (open quotes / parens) and compared case-insensitively
        against the configured abbreviation set.

        ``sentence_start`` lower-bounds the walk so we never bleed into
        a *previous* sentence's text — though in practice this only
        matters on the first sentence of a stream because the buffer is
        repeatedly trimmed after each emission.
        """
        buf = self._buffer
        # Walk backwards to find the start of the trailing token. We treat
        # whitespace as a hard boundary; everything else (letters, digits,
        # interior periods, hyphens) is part of the candidate token so
        # that ``e.g`` and ``ph.d`` round-trip correctly.
        cursor = dot_index
        while cursor > sentence_start and not buf[cursor - 1].isspace():
            cursor -= 1
        token = buf[cursor:dot_index]
        # Strip common leading punctuation that may attach to a token at
        # the start of a quoted clause: open quotes, brackets, dashes.
        token_clean = token.lstrip("\"'([{<-—‘“")
        if not token_clean:
            return False
        return token_clean.casefold() in self._abbreviations
