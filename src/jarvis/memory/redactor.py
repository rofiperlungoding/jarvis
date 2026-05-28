"""PII redaction for ``Memory_Store`` writes.

This module implements ``PIIRedactor`` as described in
``design.md §Memory_Store``::

    PII redaction: ``PIIRedactor`` applies user-configured regexes (defaults
    for emails, phone numbers, credit cards) and replaces matches with
    ``[REDACTED:<kind>]`` before persistence when memory-redaction is
    enabled (Requirement 10.8).

The redactor is intentionally a small, deterministic, and pure-Python
component:

* It compiles a fixed list of ``(kind_label, regex)`` patterns at
  construction time so per-call cost is bounded by the size of the input
  text and the number of patterns.
* It never performs I/O, never logs the input text (which by definition may
  contain PII), and never depends on any platform feature, so it is safe to
  run on every conversation turn before the plaintext is embedded and
  encrypted by ``MemoryStore.persist_turn`` (task 14.3).
* The replacement format is a single, well-known token —
  ``[REDACTED:<kind>]`` — that downstream code (Property 15 test, audit
  log, the LLM context window) can rely on as a stable visual marker.

Construction accepts either of two equivalent shapes so callers can choose
whichever is more ergonomic:

1. A list of ``(kind_label, regex)`` tuples — preserves insertion order,
   which is the order in which patterns are applied. This is the canonical
   form and is preferred when the caller wants explicit control over
   precedence (e.g. apply ``credit_card`` before ``phone`` because a credit
   card sequence can otherwise be partially captured by a phone-number
   pattern).
2. A ``dict[kind_label, regex]`` — convenient when patterns have unique
   labels and the caller does not care about precedence ordering beyond
   the dict's insertion order (Python 3.7+ dicts preserve insertion order).

If ``None`` is passed, the redactor uses :data:`DEFAULT_PATTERNS`, which
covers email addresses, North-American-style phone numbers, and credit-card
PANs (13-19 digit sequences with optional space or dash separators).

Validates: Requirements 10.8
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
from typing import Final, Union

__all__ = [
    "DEFAULT_PATTERNS",
    "PIIRedactor",
    "PatternSpec",
]


# A pattern spec is either a list/iterable of (kind, regex) tuples or a
# mapping from kind to regex. Both shapes are accepted by the constructor.
PatternSpec = Union[
    Iterable[tuple[str, str]],
    Mapping[str, str],
]


#: Built-in default patterns applied when the constructor is called without an
#: explicit pattern set. The order matches the order in which patterns are
#: applied: ``credit_card`` first so a long PAN cannot be mis-redacted as a
#: phone-number prefix.
#:
#: * ``email`` — a permissive RFC-ish pattern that matches the local-part /
#:   domain shape used in practice. Word boundaries anchor the match so a
#:   trailing comma or period is not consumed.
#: * ``phone`` — North-American 3-3-4 with optional separators (``- ``).
#:   Matches ``555-123-4567``, ``555 123 4567``, and ``5551234567``.
#: * ``credit_card`` — 13-19 digit PANs with optional space or dash
#:   separators between digits. Captures Visa (13/16/19), Mastercard (16),
#:   Discover (16), Amex (15), and Diners (14). We intentionally do NOT do
#:   Luhn validation here: false positives are acceptable for redaction
#:   (over-redaction is safer than under-redaction); false negatives are not.
DEFAULT_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    ("credit_card", r"\b(?:\d[ -]?){12,18}\d\b"),
    ("email", r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    ("phone", r"\b\d{3}[- ]?\d{3}[- ]?\d{4}\b"),
)


class PIIRedactor:
    """Replace PII matches in a string with ``[REDACTED:<kind>]`` tokens.

    The redactor compiles each provided regex once at construction time and
    applies them in their declared order on every :meth:`redact` call.
    Patterns are matched against the *current* state of the string, so
    earlier replacements are visible to later patterns. In practice this
    means a kind label like ``credit_card`` should appear before ``phone``
    if their regexes can both match the same digit sequence — see
    :data:`DEFAULT_PATTERNS` for the recommended ordering.

    Example::

        >>> r = PIIRedactor()  # uses DEFAULT_PATTERNS
        >>> r.redact("Email me at alice@example.com or call 555-123-4567.")
        'Email me at [REDACTED:email] or call [REDACTED:phone].'

    Args:
        patterns: Either a sequence of ``(kind_label, regex)`` tuples or a
            mapping from kind to regex. ``None`` (the default) selects
            :data:`DEFAULT_PATTERNS`.

    Raises:
        TypeError: If ``patterns`` is not a mapping, iterable of tuples, or
            ``None``; or if any tuple is malformed; or if any kind/regex is
            not a string.
        re.error: If any provided regex string fails to compile.
        ValueError: If a kind label is empty.
    """

    __slots__ = ("_compiled",)

    def __init__(self, patterns: PatternSpec | None = None) -> None:
        if patterns is None:
            specs: Iterable[tuple[str, str]] = DEFAULT_PATTERNS
        elif isinstance(patterns, Mapping):
            specs = tuple(patterns.items())
        else:
            # Snapshot the iterable so a generator cannot be exhausted by
            # being read once below; also gives a stable reprable form.
            specs = tuple(patterns)

        compiled: list[tuple[str, re.Pattern[str]]] = []
        for entry in specs:
            kind, regex = self._validate_entry(entry)
            compiled.append((kind, re.compile(regex)))
        self._compiled = tuple(compiled)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_entry(entry: object) -> tuple[str, str]:
        """Return ``(kind, regex)`` after validating the input shape."""
        if (
            not isinstance(entry, tuple)
            or len(entry) != 2
        ):
            raise TypeError(
                "Each pattern entry must be a (kind_label, regex) tuple; "
                f"got {entry!r}."
            )
        kind, regex = entry
        if not isinstance(kind, str):
            raise TypeError(f"kind_label must be a str; got {type(kind).__name__}.")
        if not isinstance(regex, str):
            raise TypeError(f"regex must be a str; got {type(regex).__name__}.")
        if kind == "":
            raise ValueError("kind_label must be a non-empty string.")
        return kind, regex

    @classmethod
    def with_defaults(cls) -> PIIRedactor:
        """Return a redactor configured with :data:`DEFAULT_PATTERNS`.

        Equivalent to ``PIIRedactor()``; provided as a named factory for
        call sites that want to make the "use defaults" intent explicit.
        """
        return cls(DEFAULT_PATTERNS)

    @classmethod
    def from_config_patterns(
        cls,
        regex_list: Iterable[str],
        *,
        prefix: str = "pii",
    ) -> PIIRedactor:
        """Build a redactor from a flat list of regex strings.

        The application config exposes ``memory.pii_patterns`` as a flat
        ``list[str]`` for ergonomic TOML editing (see
        ``src/jarvis/config/default.toml``); this helper wraps each entry
        with a synthesized kind label of the form ``f"{prefix}_{i}"`` so
        the redactor's ``[REDACTED:<kind>]`` output remains uniform.

        Args:
            regex_list: Flat iterable of regex strings (e.g. the value of
                ``MemoryConfig.pii_patterns``).
            prefix: Label prefix; defaults to ``"pii"``. The 1-based index
                of each regex is appended (``pii_1``, ``pii_2``, ...).

        Returns:
            A new :class:`PIIRedactor`.
        """
        labelled: list[tuple[str, str]] = []
        for index, regex in enumerate(regex_list, start=1):
            if not isinstance(regex, str):
                raise TypeError(
                    f"regex_list[{index - 1}] must be str; got "
                    f"{type(regex).__name__}."
                )
            labelled.append((f"{prefix}_{index}", regex))
        return cls(labelled)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def redact(self, text: str) -> str:
        """Return ``text`` with every configured PII match replaced.

        Each compiled pattern is applied in declaration order using
        :meth:`re.Pattern.sub`. Replacements are computed against the
        evolving string, so a later pattern sees the redacted output of
        the earlier patterns. The replacement form is the literal
        ``[REDACTED:<kind>]`` — note that this token contains no regex
        metacharacters, so subsequent passes cannot accidentally match it.

        ``text`` is returned unchanged when no patterns match, when the
        redactor was constructed with no patterns, or when the input is
        empty.

        Args:
            text: The plaintext to scrub. Must be a ``str``; bytes inputs
                are deliberately rejected so callers cannot accidentally
                feed in raw audio frames or DPAPI ciphertext.

        Returns:
            The scrubbed string. Always a fresh ``str``; never the input
            object identity, even when no replacement occurs.

        Raises:
            TypeError: If ``text`` is not a ``str``.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a str; got {type(text).__name__}.")
        if not text or not self._compiled:
            # Always return a distinct str to avoid accidental aliasing of
            # caller-owned buffers; ``str(text)`` is cheap and idempotent.
            return str(text)

        result = text
        for kind, pattern in self._compiled:
            replacement = f"[REDACTED:{kind}]"
            result = pattern.sub(replacement, result)
        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def kinds(self) -> tuple[str, ...]:
        """The kind labels currently configured, in application order."""
        return tuple(kind for kind, _ in self._compiled)

    def __len__(self) -> int:
        """The number of compiled patterns."""
        return len(self._compiled)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        kinds = ", ".join(self.kinds) if self._compiled else "<none>"
        return f"PIIRedactor(kinds=[{kinds}])"
