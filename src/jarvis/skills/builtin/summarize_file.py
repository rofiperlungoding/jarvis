"""Built-in ``SummarizeFileSkill``.

Implements the summarisation half of Requirement 8 (the file-reading
half lives in :mod:`jarvis.skills.builtin.read_file`). The Skill
delegates the read — and therefore the sandbox check — to
:class:`~jarvis.skills.builtin.read_file.ReadFileSkill`, then streams
the document through the active :class:`LLMBackend` to produce a
summary capped at ``max_words`` words.

Why delegate to ``ReadFileSkill``
---------------------------------

Property 9 / CP12 ("path sandbox soundness") quantifies over both
``ReadFileSkill`` and ``SummarizeFileSkill``. Re-implementing the
sandbox / size / extension checks here would risk drift between the
two Skills; instead we call ``ReadFileSkill.execute`` directly. The
``SandboxViolation`` it raises propagates unchanged through this
Skill's ``execute`` so the Skill_Registry still records exactly one
``policy_violation`` audit entry (Requirement 13.6) and surfaces
``access_denied`` to the dialog layer (Requirement 8.6). Recoverable
errors (``file_too_large``, ``not_supported``, ``internal_error``,
``schema_violation`` for malformed paths) are passed through
verbatim so the closed error taxonomy stays consistent.

LLM contract
------------

* ``ctx.llm_backend`` MUST be a :class:`~jarvis.llm.base.LLMBackend`.
  Missing backends are surfaced as ``internal_error`` so the
  Dialog_Manager can decide whether to inform the user or retry once
  the BackendSelector has stabilised.
* The streamed call uses an empty ``tools`` list because summarisation
  is a pure-text task — no Skill should be invoked recursively from
  inside a summarisation prompt.
* Tool-call events emitted by a misbehaving model are ignored; only
  ``content_delta`` text contributes to the summary.
* A focused, summariser-oriented system prompt is used in place of the
  JARVIS persona prompt. Property 11 / CP14 only constrains
  ``DialogManager.handle_turn``, which is the user-facing dialog loop;
  internal tool calls are free to use task-specific prompts so the
  model is not nudged toward dry-witty narration when the user wants a
  faithful summary.

Word-count enforcement
----------------------

Requirement 8.4: the summary "SHALL NOT exceed ``max_words`` words".
We enforce this with belt-and-braces:

1. The system prompt instructs the model to stay within the budget.
2. The accumulated output is split on whitespace and truncated to the
   first ``max_words`` tokens before being returned. This catches
   both honest overruns (the model lost count) and adversarial output
   (a misbehaving local backend that ignores instructions).

Validates: Requirements 8.3, 8.4, 8.6, 8.7
"""

from __future__ import annotations

import logging
from typing import Any, Final

from jarvis.skills.base import (
    SkillContext,
    SkillManifest,
    SkillResult,
)
from jarvis.skills.builtin.read_file import ReadFileSkill

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MAX_WORDS",
    "MAX_WORDS_HARD_CAP",
    "SCHEMA",
    "SKILL",
    "SummarizeFileSkill",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default value for ``max_words`` when the caller omits the field.
#: Requirement 8.3 anchors the value at 200.
DEFAULT_MAX_WORDS: Final[int] = 200

#: Hard ceiling on ``max_words``. The acceptance criterion does not
#: pin an explicit upper bound, but allowing arbitrarily large values
#: would let a single Tool_Call burn the entire LLM context budget.
#: 2000 words is roughly the limit of a long-form briefing and stays
#: well inside Mistral / Ollama context windows.
MAX_WORDS_HARD_CAP: Final[int] = 2000

#: Skill name surfaced to the LLM. Pinned because Requirement 8.3
#: anchors the wording ("SummarizeFileSkill") and changing it would
#: silently break configured trusted-action allowlist entries.
_SKILL_NAME: Final[str] = "SummarizeFileSkill"

_SKILL_DESCRIPTION: Final[str] = (
    "Read a file from one of the user's allowed directories and "
    "produce a concise summary using the configured LLM backend. "
    "Supports the same formats as ReadFileSkill (.txt, .md, .csv, "
    ".json, .py, .js, .ts, .pdf, .docx). Files larger than 5 MB are "
    "refused with file_too_large."
)


# ---------------------------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------------------------


SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "title": "SummarizeFile",
    "description": (
        "Summarise the contents of a file at an absolute path. The "
        "path MUST resolve inside one of the user's configured "
        "allowed directories."
    ),
    "properties": {
        "path": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Absolute filesystem path to the file. Relative paths "
                "are rejected with a schema violation; the path is "
                "canonicalised (symlinks resolved, '..' collapsed) "
                "before the sandbox check."
            ),
        },
        "max_words": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_WORDS_HARD_CAP,
            "description": (
                f"Upper bound on the number of words in the summary. "
                f"Defaults to {DEFAULT_MAX_WORDS}. Hard-capped at "
                f"{MAX_WORDS_HARD_CAP} so a single Tool_Call cannot "
                "consume the entire LLM context budget."
            ),
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_summarisation_messages(
    content: str, max_words: int
) -> list[dict[str, str]]:
    """Construct the chat messages submitted to the LLM backend.

    Kept as a free function so tests can introspect the exact prompt
    without instantiating the Skill. The prompt is intentionally
    short — Mistral and Ollama both adhere better to single-clause
    instructions than to a wall of constraints — and explicitly tells
    the model to emit only the summary text so the streamed deltas can
    be concatenated verbatim.
    """
    system_prompt = (
        "You are a precise, neutral document summariser. "
        f"Produce a single coherent summary in at most {max_words} words. "
        "Preserve key facts, names, dates, and any explicit conclusions. "
        "Do not include preamble, headings, bullet points, or commentary "
        "about the summarisation process. Output ONLY the summary text."
    )
    user_prompt = (
        "Summarise the following document. Stay strictly within the "
        f"{max_words}-word budget.\n\n"
        "<<<DOCUMENT>>>\n"
        f"{content}\n"
        "<<<END DOCUMENT>>>"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _truncate_to_words(text: str, max_words: int) -> tuple[str, int, bool]:
    """Truncate ``text`` to at most ``max_words`` whitespace-delimited tokens.

    Returns ``(summary, word_count, was_truncated)``. We split on
    arbitrary whitespace via :meth:`str.split` so the count matches
    what a human reader would call "words" (newlines, tabs, and
    multi-space runs all collapse to single delimiters). The original
    inter-token whitespace is replaced by single spaces in the
    returned string — this is the standard way to normalise a
    word-budget cut so trailing/leading whitespace does not creep in.
    """
    tokens = text.split()
    if len(tokens) <= max_words:
        return " ".join(tokens), len(tokens), False
    return " ".join(tokens[:max_words]), max_words, True


# ---------------------------------------------------------------------------
# Skill implementation
# ---------------------------------------------------------------------------


class SummarizeFileSkill:
    """Summarise a sandboxed file using the active :class:`LLMBackend`.

    Stateless — a single instance is reused across invocations. Holds a
    private :class:`ReadFileSkill` so the file-reading half of
    Requirement 8 is implemented exactly once; this Skill only owns
    the LLM orchestration and word-budget enforcement.
    """

    manifest: Final[SkillManifest] = SkillManifest(
        name=_SKILL_NAME,
        description=_SKILL_DESCRIPTION,
        json_schema=SCHEMA,
        destructive=False,
        # The LLM call is the dominant cost; allow a generous budget so
        # large documents on a slow local Ollama backend still complete
        # before the registry's wall-clock guard fires.
        timeout_seconds=120.0,
        # Summarisation is platform-agnostic (the underlying file read
        # is too); declare the full matrix so Requirement 15.4 does not
        # gate the Skill on a future build.
        platforms=("windows", "linux", "darwin"),
        source="builtin",
    )

    def __init__(self, *, read_skill: ReadFileSkill | None = None) -> None:
        # Allow tests to inject a stub reader; production code uses a
        # default :class:`ReadFileSkill` so the sandbox / size / format
        # contract is shared with the standalone Skill.
        self._read_skill: ReadFileSkill = read_skill or ReadFileSkill()

    async def execute(
        self,
        args: dict[str, Any],
        ctx: SkillContext,
    ) -> SkillResult:
        # ---- 1. Argument shape ---------------------------------------
        # The registry's draft-07 validator already enforces these
        # rules; the runtime guard catches direct callers (tests,
        # future skill chains) that bypass the registry.
        validation = self._validate_args(args)
        if isinstance(validation, SkillResult):
            return validation
        path, max_words = validation

        # ---- 2. Delegate the read -----------------------------------
        # ``ReadFileSkill`` performs:
        #   * absolute-path / non-empty-string check (schema_violation)
        #   * sandbox check (raises SandboxViolation → access_denied)
        #   * extension gate (not_supported)
        #   * 5 MB size cap (file_too_large)
        #   * format-specific parsing (.pdf, .docx, text)
        # A SandboxViolation propagates unchanged so the registry
        # records a single ``policy_violation`` audit entry per
        # Requirement 13.6 (Property 9 / CP12).
        read_result = await self._read_skill.execute({"path": path}, ctx)
        if not read_result.ok:
            # Pass error codes through verbatim so the closed error
            # taxonomy stays consistent across both Skills.
            return SkillResult.error(
                read_result.error_code,  # type: ignore[arg-type]
                read_result.error_message,
                value=read_result.value,
            )

        # ``ReadFileSkill`` always populates ``value`` on success; the
        # cast is documented by its contract.
        assert read_result.value is not None
        content: str = read_result.value["content"]
        canonical_path: str = read_result.value["path"]
        extension: str = read_result.value["extension"]
        size_bytes: int = read_result.value["size_bytes"]

        # ---- 3. LLM availability ------------------------------------
        llm_backend = ctx.llm_backend
        if llm_backend is None:
            return SkillResult.error(
                "internal_error",
                (
                    "SummarizeFileSkill requires an LLMBackend on the "
                    "SkillContext but ctx.llm_backend is None"
                ),
            )

        # ---- 4. Empty-content shortcut -------------------------------
        # Sending an empty document to the model wastes a turn and
        # often produces hallucinated summaries. Surface the empty
        # state directly with a deterministic, honest payload.
        if not content.strip():
            return SkillResult.success(
                value={
                    "path": canonical_path,
                    "extension": extension,
                    "size_bytes": size_bytes,
                    "max_words": max_words,
                    "summary": "",
                    "word_count": 0,
                    "truncated": False,
                }
            )

        # ---- 5. Stream the summary from the LLM ---------------------
        messages = _build_summarisation_messages(content, max_words)
        try:
            chunks: list[str] = []
            async with llm_backend.stream(messages, tools=[]) as stream:
                async for event in stream:
                    # The Stream contract emits two event variants:
                    # ``content_delta`` and ``tool_call``. Tool calls
                    # are ignored — summarisation is a pure-text task
                    # and a misbehaving model should not be able to
                    # invoke side-effecting Skills via this path.
                    if getattr(event, "type", None) == "content_delta":
                        text = getattr(event, "text", "")
                        if text:
                            chunks.append(text)
        except Exception as exc:
            # Backend failures (auth, rate limit, transport) bubble up
            # as exceptions; we map them to ``internal_error`` so the
            # Skill stays inside the closed error taxonomy. The
            # BackendSelector / Dialog_Manager handle the user-facing
            # recovery (retry, key prompt, fallback notice) one level
            # up; here we only need to keep the Skill contract clean.
            logger.exception("SummarizeFileSkill: LLM stream failed")
            return SkillResult.error(
                "internal_error",
                f"LLM backend failed during summarisation: {exc}",
            )

        raw_summary = "".join(chunks).strip()

        # ---- 6. Word-budget enforcement ------------------------------
        summary, word_count, truncated = _truncate_to_words(
            raw_summary, max_words
        )

        return SkillResult.success(
            value={
                "path": canonical_path,
                "extension": extension,
                "size_bytes": size_bytes,
                "max_words": max_words,
                "summary": summary,
                "word_count": word_count,
                "truncated": truncated,
            }
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_args(
        args: dict[str, Any],
    ) -> SkillResult | tuple[str, int]:
        """Return ``(path, max_words)`` or a ``SkillResult`` failure.

        The registry's JSON Schema validator already rejects malformed
        inputs, but defending the runtime path keeps the Skill correct
        when invoked directly from tests or future skill chains. We do
        not re-validate the path's absolute-ness here — that lives in
        :class:`ReadFileSkill` and the duplication would risk drift.
        """
        path = args.get("path")
        if not isinstance(path, str) or not path.strip():
            return SkillResult.error(
                "schema_violation",
                "SummarizeFileSkill 'path' must be a non-empty string",
            )

        max_words_raw = args.get("max_words", DEFAULT_MAX_WORDS)
        # Reject ``bool`` explicitly: Python booleans are instances of
        # ``int`` but the JSON Schema integer keyword does not match
        # them. The same belt-and-braces guard lives on
        # :class:`SkillManifest.timeout_seconds`.
        if isinstance(max_words_raw, bool) or not isinstance(max_words_raw, int):
            return SkillResult.error(
                "schema_violation",
                "SummarizeFileSkill 'max_words' must be an integer",
            )
        if max_words_raw < 1 or max_words_raw > MAX_WORDS_HARD_CAP:
            return SkillResult.error(
                "schema_violation",
                (
                    "SummarizeFileSkill 'max_words' must be between 1 "
                    f"and {MAX_WORDS_HARD_CAP} (got {max_words_raw})"
                ),
            )
        return path, max_words_raw


# ---------------------------------------------------------------------------
# Module-level singleton consumed by :meth:`SkillRegistry.discover`.
# ---------------------------------------------------------------------------


#: Typed as the concrete :class:`SummarizeFileSkill` rather than the
#: :class:`Skill` Protocol because the latter declares ``manifest`` as
#: a writable variable while we expose it as a :data:`Final` class
#: attribute (mirrors the convention used by :mod:`read_file`). The
#: registry's ``isinstance(obj, Skill)`` runtime check still validates
#: structural conformance at startup.
SKILL: SummarizeFileSkill = SummarizeFileSkill()
