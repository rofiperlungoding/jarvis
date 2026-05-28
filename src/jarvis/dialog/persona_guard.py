"""Post-generation persona guard.

Implements Requirement 11.5 of the JARVIS specification: when the LLM_Backend
returns text that violates the active persona (for example, refers to itself
as ``ChatGPT`` or ``Claude``, or says ``as an AI language model``), the
:class:`DialogManager` rewrites the offending phrase or triggers a single
stricter regeneration before forwarding the response to the TTS engine.

The guard exposed here is responsible only for *detecting* forbidden
self-references and producing a *cosmetic rewrite*. The decision to
regenerate is left to :class:`DialogManager`, which uses the boolean returned
by :meth:`PersonaGuard.check` to gate at most one stricter retry.

Design notes
------------
* Detection is deterministic and case-insensitive. Each phrase from
  ``persona.forbidden_self_refs`` is matched as a literal substring with
  word-boundary anchoring on alphanumeric edges. This means ``ChatGPT`` will
  not match inside ``ChatGPT-style`` only at the prefix; but will not match
  ``ChatGPTs`` (because the trailing ``s`` is alphanumeric and breaks the
  word boundary). The intent is to flag self-identification, not incidental
  mentions.
* The rewrite substitutes the offending phrase with the persona name (e.g.,
  ``ChatGPT`` -> ``JARVIS``). For the common "AI disclaimer" idioms
  (``as an AI language model``, ``as a large language model``, etc.) the
  rewrite is normalised to ``as <persona name>`` so the surrounding sentence
  remains grammatical.
* The implementation uses only :mod:`typing.Protocol` so it does not depend
  on the concrete :class:`PersonaProfile` from ``jarvis.dialog.persona``;
  any object exposing ``name`` and ``forbidden_self_refs`` works.

References
----------
* Requirements 11.5
* Design: ``Dialog_Manager`` -> persona enforcement
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

__all__ = ["PersonaGuard", "PersonaLike"]


@runtime_checkable
class PersonaLike(Protocol):
    """Structural view of :class:`jarvis.dialog.persona.PersonaProfile`.

    Only the fields the guard reads are required, which keeps this module
    independent from the concrete dataclass and easy to test in isolation.

    Attributes are declared as ``@property`` so that callers may pass in
    either a mutable stand-in (test doubles) or a frozen dataclass like
    :class:`~jarvis.dialog.persona.PersonaProfile` — both satisfy the
    Protocol structurally because property access is read-only.
    """

    @property
    def name(self) -> str: ...

    @property
    def forbidden_self_refs(self) -> tuple[str, ...]: ...


# Phrases whose natural rewrite is "as <persona name>" rather than just
# "<persona name>". Matched case-insensitively against the *start* of each
# forbidden phrase from the persona profile.
_DISCLAIMER_PREFIXES: tuple[str, ...] = (
    "as an ai",
    "as a ai",
    "as a large language model",
    "as an llm",
    "as a language model",
)


class PersonaGuard:
    """Scan generated text for forbidden self-references and rewrite them.

    The guard is stateless; a single instance can be shared across turns.
    """

    def check(self, text: str, persona: PersonaLike) -> tuple[str, bool]:
        """Return ``(rewritten_text, was_violated)``.

        Parameters
        ----------
        text:
            The final assistant text produced by the LLM backend, after any
            tool dispatch and streaming. May be empty.
        persona:
            The active persona profile. ``persona.forbidden_self_refs`` is
            iterated in order; the first match in ``text`` is replaced first,
            then subsequent matches.

        Returns
        -------
        tuple[str, bool]
            * ``rewritten_text`` is ``text`` with every forbidden phrase
              substituted. When no phrase matches, the original ``text`` is
              returned unchanged (object identity preserved when possible).
            * ``was_violated`` is ``True`` when at least one forbidden phrase
              was found and rewritten. The DialogManager uses this flag to
              trigger one stricter regeneration; if regeneration still
              produces a violation, the rewritten text is the safe fallback.
        """
        if not text:
            return text, False

        forbidden = tuple(p for p in persona.forbidden_self_refs if p)
        if not forbidden:
            return text, False

        rewritten = text
        violated = False
        replacement_name = persona.name or "the assistant"

        for phrase in forbidden:
            pattern = _compile_phrase(phrase)
            replacement = _replacement_for(phrase, replacement_name)
            new_text = pattern.sub(replacement, rewritten)
            if new_text != rewritten:
                violated = True
                rewritten = new_text

        return rewritten, violated


def _compile_phrase(phrase: str) -> re.Pattern[str]:
    """Compile ``phrase`` as a case-insensitive literal with word boundaries.

    Word boundaries are only added on alphanumeric edges so that phrases that
    start or end with punctuation/whitespace (e.g., quoted phrases) still
    match cleanly.
    """
    escaped = re.escape(phrase)
    prefix = r"\b" if phrase[:1].isalnum() else ""
    suffix = r"\b" if phrase[-1:].isalnum() else ""
    return re.compile(prefix + escaped + suffix, re.IGNORECASE)


def _replacement_for(phrase: str, persona_name: str) -> str:
    """Compute the substitution string for a forbidden phrase.

    Disclaimer-style phrases are rewritten to ``as <persona_name>`` so the
    surrounding sentence keeps its preposition. Everything else is replaced
    by the bare persona name.
    """
    lowered = phrase.lower()
    for disclaimer in _DISCLAIMER_PREFIXES:
        if lowered.startswith(disclaimer):
            return f"as {persona_name}"
    return persona_name
