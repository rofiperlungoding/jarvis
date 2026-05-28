"""Authorization policy and trusted-action allowlist.

The :class:`AuthorizationPolicy` is the gatekeeper between the
Dialog_Manager's tool-dispatch loop and any Skill that can mutate state
the user cares about — sending email, running a registered script,
creating a calendar event, forgetting a Memory_Record, and so on. It is
the implementation of the design's ``§Authorization_Policy`` component
and the bridge that makes Property 6 / CP9 hold "by construction":

    *For every Tool_Call C whose Skill is classified Destructive and is
    not matched by the trusted-action allowlist, the audit log SHALL
    contain a* ``confirmation_requested`` *entry whose id is strictly
    less than the corresponding* ``executed`` *or* ``denied`` *entry's
    id and whose ``skill`` and ``args_json`` match C.*

What this module exposes
------------------------

* :data:`Classification`, :data:`SAFE`, :data:`DESTRUCTIVE` — the closed
  two-element classification of a Tool_Call. Mirrors the design's
  ``Classification ∈ {Safe, Destructive}``.
* :class:`ConfirmationDialog` — a :class:`typing.Protocol` describing the
  one method :meth:`AuthorizationPolicy.confirm` needs from its caller:
  ``await dialog.ask_user(prompt) -> str``. Decoupling here lets task 11
  ship before task 13 (Dialog_Manager) and gives tests a trivial fake.
* :class:`TrustedActionAllowlist` — wraps the configured
  :class:`~jarvis.config.schema.TrustedAction` entries and answers
  "does this Tool_Call match an allowlisted entry?" with a deep
  subset-match on arguments. A match bypasses the user prompt for a
  single invocation per Requirement 16.3.
* :class:`AuthorizationPolicy` — the policy itself, with
  :meth:`classify`, :meth:`match_allowlist`, :meth:`confirm`, and the
  helper :meth:`record_executed` so the Dialog_Manager can chain audit
  writes without re-implementing argument canonicalisation.

Audit ordering invariant
------------------------

:meth:`AuthorizationPolicy.confirm` records the ``confirmation_requested``
entry **before** it speaks the prompt, and the ``denied`` entry **before**
it returns ``False`` on a refused confirmation. The Dialog_Manager
records the matching ``executed`` entry **after** the Skill_Registry
returns. Together with :class:`~jarvis.security.audit_log.AuditLog`'s
``AUTOINCREMENT`` row id and per-call lock, this gives the strict
``confirmation_requested.id < executed.id``/``denied.id`` ordering that
CP9 requires.

Allowlist-matched calls deliberately skip the prompt (Requirement 16.3)
but still emit a ``confirmation_requested`` audit entry so the audit
trail tells the same story as a non-allowlisted call. The
``executed`` outcome carries an ``allowlist_bypass`` suffix so operators
can distinguish the two flows post-hoc.

Validates: Requirements 16.1, 16.2, 16.3, 16.4, 16.5
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
import re
from typing import Any, Final, Literal, Protocol, runtime_checkable

from jarvis.config.schema import DestructiveOperation, TrustedAction
from jarvis.llm.base import ToolCall
from jarvis.security.audit_log import AuditLog
from jarvis.skills.base import SkillManifest

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_DESTRUCTIVE_SKILLS",
    "DESTRUCTIVE",
    "SAFE",
    "AuthorizationPolicy",
    "Classification",
    "ConfirmationDialog",
    "TrustedActionAllowlist",
]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

# The closed two-element classification from the design. Modelled as a
# :class:`typing.Literal` rather than a :class:`enum.Enum` to match the
# rest of the codebase (see :data:`SkillSource`, :data:`SkillErrorCode`)
# and so the spelling is JSON-stable when written into audit fixtures.
Classification = Literal["safe", "destructive"]
SAFE: Final[Classification] = "safe"
DESTRUCTIVE: Final[Classification] = "destructive"


# ---------------------------------------------------------------------------
# Destructive defaults (Requirement 16.1)
# ---------------------------------------------------------------------------

# The full hard-coded set from Requirement 16.1. Operation-level entries
# (``Skill.operation``) are encoded with a dot-separated suffix; the
# classifier interprets them as "this Skill is destructive when its
# ``operation`` argument equals the suffix". Production callers pass the
# (typically narrower) ``authorization.destructive_skills`` list from the
# user's config; this default exists so a policy constructed without an
# explicit override is never *less* strict than the requirement.
DEFAULT_DESTRUCTIVE_SKILLS: Final[tuple[str, ...]] = (
    "SendEmailSkill",
    "SendMessageSkill",
    "RunScriptSkill",
    "MemoryAdminSkill.forget",
    "CalendarSkill.create_event",
)

# Convention used when a hard-coded destructive entry has the form
# ``"SkillName.operation"`` and no explicit :class:`DestructiveOperation`
# is configured: the operation is read from this argument field. The
# built-in skills (``MemoryAdminSkill``, ``CalendarSkill``) both expose
# an ``operation`` discriminator, so this is the right default.
_DEFAULT_OPERATION_FIELD: Final[str] = "operation"


# ---------------------------------------------------------------------------
# Affirmative / negative response parsing
# ---------------------------------------------------------------------------

# We compile these once at import time. The patterns are intentionally
# conservative — anything ambiguous defaults to a "no" because
# Requirement 16.2 demands an *affirmative* response and the safety
# default for a Destructive_Action must be denial.
_AFFIRMATIVE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b("
    r"yes|yeah|yep|yup|sure|ok|okay|"
    r"affirmative|confirm(?:ed)?|proceed|go|"
    r"do(?:\s+it|\s+so)?|please(?:\s+do)?|send(?:\s+it)?"
    r")\b",
    re.IGNORECASE,
)

_NEGATIVE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b("
    r"no|nope|nah|cancel|stop|abort|halt|"
    r"don'?t|do\s+not|never\s*mind|nevermind|"
    r"negative|deny|reject|forget\s+it"
    r")\b",
    re.IGNORECASE,
)


def _is_affirmative(response: str | None) -> bool:
    """Decide whether ``response`` authorises a Destructive_Action.

    Returns ``True`` only when an affirmative token is present *and* no
    negation token precedes or accompanies it. Empty / ``None`` /
    non-string responses always evaluate as denial — this is the safety
    default required by Requirement 16.2.
    """
    if not isinstance(response, str):
        return False
    text = response.strip()
    if not text:
        return False
    # Negation wins over affirmation: a phrase like "no, do it later"
    # mixes both tokens and must not be treated as consent.
    if _NEGATIVE_PATTERN.search(text):
        return False
    return bool(_AFFIRMATIVE_PATTERN.search(text))


# ---------------------------------------------------------------------------
# ConfirmationDialog protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ConfirmationDialog(Protocol):
    """Minimal interface :meth:`AuthorizationPolicy.confirm` needs.

    The Dialog_Manager (task 13.4) implements this by composing the TTS
    engine (to *speak* the prompt) and the STT engine (to *listen* for
    the user's reply). Tests substitute a trivial fake that returns a
    canned string. We avoid importing ``DialogManager`` here both to
    skirt the circular import that would otherwise appear and to let
    this module ship before the Dialog_Manager is implemented.
    """

    async def ask_user(self, prompt: str) -> str:
        """Speak ``prompt`` to the user and return their transcribed reply."""
        ...


# ---------------------------------------------------------------------------
# TrustedActionAllowlist
# ---------------------------------------------------------------------------


def _is_subset(subset: Mapping[str, Any], full: Mapping[str, Any]) -> bool:
    """Recursive deep subset match used by the allowlist.

    ``subset`` is "contained in" ``full`` when every key in ``subset``
    appears in ``full`` and the corresponding values are equal. Nested
    mappings are matched by the same rule recursively, which lets users
    write allowlist entries like::

        {"channel": "slack", "recipient": "team-bot"}

    that match any Tool_Call with those two fields and ignore the body.
    Lists, scalars, and other non-mapping values must compare equal with
    ``==``. We deliberately do not interpret list arguments as sets so
    ordering changes do not silently match.
    """
    for key, expected in subset.items():
        if key not in full:
            return False
        actual = full[key]
        if isinstance(expected, Mapping) and isinstance(actual, Mapping):
            if not _is_subset(expected, actual):
                return False
        elif expected != actual:
            return False
    return True


class TrustedActionAllowlist:
    """Per-Skill / per-args allowlist for confirmation bypass.

    Wraps a sequence of :class:`~jarvis.config.schema.TrustedAction`
    entries (loaded from ``[authorization].trusted_action_allowlist``).
    A Tool_Call matches an entry when:

    * ``entry.skill == tool_call.skill_name``, AND
    * ``entry.args_subset`` is a deep subset of ``tool_call.arguments``
      (see :func:`_is_subset`).

    The first matching entry wins — entries earlier in the list shadow
    later ones. This lets a user write a permissive allowlist below a
    more restrictive override without having to delete the broader rule.

    Per Requirement 16.3 the bypass is single-use *per invocation*, not
    per (skill, args) pair: every fresh Tool_Call goes through
    :meth:`match` again, and an entry that matched yesterday may not
    match today if its arguments differ.
    """

    __slots__ = ("_entries",)

    def __init__(self, entries: Sequence[TrustedAction] = ()) -> None:
        # Defensive copy + tuple so callers cannot mutate the policy's
        # view of the allowlist after construction. Tuples also give us
        # cheap structural equality in tests.
        validated: list[TrustedAction] = []
        for entry in entries:
            if not isinstance(entry, TrustedAction):
                raise TypeError(
                    "TrustedActionAllowlist entries must be TrustedAction "
                    f"instances, got {type(entry).__name__}"
                )
            validated.append(entry)
        self._entries: tuple[TrustedAction, ...] = tuple(validated)

    @property
    def entries(self) -> tuple[TrustedAction, ...]:
        """The configured entries, in declaration order."""
        return self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._entries)

    def match(self, tool_call: ToolCall) -> TrustedAction | None:
        """Return the first allowlist entry that matches ``tool_call``.

        Returns ``None`` when no entry matches; callers MUST then fall
        back to the standard confirmation flow.
        """
        for entry in self._entries:
            if entry.skill != tool_call.skill_name:
                continue
            if _is_subset(entry.args_subset, tool_call.arguments):
                return entry
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DotOpEntry:
    """A hard-coded ``Skill.operation`` destructive marker.

    Built from a string like ``"MemoryAdminSkill.forget"`` at policy
    construction time so :meth:`AuthorizationPolicy.classify` does not
    re-parse the entry on every call.
    """

    skill: str
    operation: str


def _split_destructive_skills(
    entries: Sequence[str],
) -> tuple[frozenset[str], tuple[_DotOpEntry, ...]]:
    """Partition the ``hard_coded_destructive_skills`` list.

    Pure skill names (no ``.``) become the "always destructive" set.
    Entries containing a dot are parsed as ``Skill.operation`` and
    matched against ``arguments[_DEFAULT_OPERATION_FIELD]`` at classify
    time. This mirrors the convention used by the built-in
    :class:`~jarvis.skills.builtin.memory_admin.MemoryAdminSkill` and
    :class:`~jarvis.skills.builtin.calendar.CalendarSkill`, which both
    expose an ``operation`` argument as their destructive discriminator.
    """
    plain: set[str] = set()
    dot_op: list[_DotOpEntry] = []
    for raw in entries:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(
                "hard_coded_destructive_skills entries must be non-empty strings"
            )
        entry = raw.strip()
        if "." in entry:
            skill, _, op = entry.partition(".")
            if not skill or not op:
                raise ValueError(
                    f"invalid Skill.operation entry {raw!r}; expected "
                    "'SkillName.operation' with non-empty parts"
                )
            dot_op.append(_DotOpEntry(skill=skill, operation=op))
        else:
            plain.add(entry)
    return frozenset(plain), tuple(dot_op)


def _build_summary(tool_call: ToolCall, *, max_chars: int = 240) -> str:
    """Build a short, speakable summary of an intended Destructive_Action.

    The summary is read aloud by the TTS engine before the user is asked
    for confirmation (Requirement 16.2). Keeping it terse — skill name
    plus a compact JSON dump of arguments, truncated — is more useful
    than a wall of text the user has to hold in their head while
    deciding. The ``raw_arguments`` field is preferred because it
    preserves the model's original whitespace and keys.
    """
    args_repr = tool_call.raw_arguments or "{}"
    if len(args_repr) > max_chars:
        # We truncate rather than drop fields entirely so the user can
        # still hear at least the leading arguments. Truncation is
        # signalled by an ellipsis so the summary obviously did not run
        # to completion.
        args_repr = args_repr[: max_chars - 1].rstrip() + "…"
    return (
        f"I'm about to invoke {tool_call.skill_name} "
        f"with arguments {args_repr}. Shall I proceed, sir?"
    )


# ---------------------------------------------------------------------------
# AuthorizationPolicy
# ---------------------------------------------------------------------------


class AuthorizationPolicy:
    """Classify Tool_Calls and orchestrate confirmation for destructive ones.

    Construction
    ------------

    Parameters
    ----------
    allowlist:
        A :class:`TrustedActionAllowlist` (possibly empty). When a
        destructive Tool_Call matches the allowlist, :meth:`confirm`
        skips the user prompt but still records the audit trail
        (Requirement 16.3).
    audit:
        The append-only :class:`AuditLog` used to record every
        ``confirmation_requested`` / ``denied`` / ``executed`` event.
        This is the single source of truth that CP9 inspects to verify
        ordering.
    hard_coded_destructive_skills:
        Sequence of Skill names — or ``Skill.operation`` strings — that
        the policy treats as destructive regardless of the manifest.
        Defaults to :data:`DEFAULT_DESTRUCTIVE_SKILLS`. Production code
        passes ``config.authorization.destructive_skills`` so users can
        tighten or relax the policy from their TOML file.
    destructive_operations:
        Sequence of :class:`DestructiveOperation` entries. Each entry
        promotes a Tool_Call to ``Destructive`` when its ``op_field``
        argument is in ``op_values``. Used for Skills whose
        destructiveness depends on a discriminator (e.g.,
        ``CalendarSkill.operation == "create_event"``). Defaults to an
        empty tuple; the production policy is built from
        ``config.authorization.destructive_operations``.

    Three orthogonal sources can mark a Tool_Call destructive:

    1. ``manifest.destructive`` — the Skill's manifest opts in
       statically (Requirement 16.1, last clause).
    2. The Skill's name appears in the hard-coded list, either as a
       plain name (``"SendEmailSkill"``) or as a ``Skill.operation``
       reference whose operation matches the call's ``operation`` arg.
    3. A configured :class:`DestructiveOperation` matches by skill name
       *and* by the value of the configured discriminator field.

    Any single source is sufficient — they OR together — so loosening
    the manifest cannot accidentally widen the privilege envelope when
    the user's config or the hard-coded list still mark the call
    destructive.
    """

    def __init__(
        self,
        *,
        allowlist: TrustedActionAllowlist,
        audit: AuditLog,
        hard_coded_destructive_skills: Sequence[str] = DEFAULT_DESTRUCTIVE_SKILLS,
        destructive_operations: Sequence[DestructiveOperation] = (),
    ) -> None:
        if not isinstance(allowlist, TrustedActionAllowlist):
            raise TypeError(
                "allowlist must be a TrustedActionAllowlist instance, "
                f"got {type(allowlist).__name__}"
            )
        if not isinstance(audit, AuditLog):
            raise TypeError(
                f"audit must be an AuditLog instance, got {type(audit).__name__}"
            )

        plain_skills, dot_ops = _split_destructive_skills(
            hard_coded_destructive_skills
        )

        validated_ops: list[DestructiveOperation] = []
        for op in destructive_operations:
            if not isinstance(op, DestructiveOperation):
                raise TypeError(
                    "destructive_operations entries must be DestructiveOperation "
                    f"instances, got {type(op).__name__}"
                )
            validated_ops.append(op)

        self._allowlist: TrustedActionAllowlist = allowlist
        self._audit: AuditLog = audit
        self._destructive_skills: frozenset[str] = plain_skills
        self._dot_op_entries: tuple[_DotOpEntry, ...] = dot_ops
        self._destructive_operations: tuple[DestructiveOperation, ...] = tuple(
            validated_ops
        )

    # ----------------------------------------------------------------- accessors

    @property
    def allowlist(self) -> TrustedActionAllowlist:
        """The trusted-action allowlist this policy enforces."""
        return self._allowlist

    @property
    def audit(self) -> AuditLog:
        """The audit log this policy writes to."""
        return self._audit

    # -------------------------------------------------------------- classify

    def classify(
        self, tool_call: ToolCall, manifest: SkillManifest
    ) -> Classification:
        """Return :data:`SAFE` or :data:`DESTRUCTIVE` for ``tool_call``.

        The function is pure — no audit writes, no I/O — so it can be
        called freely from the Dialog_Manager's hot path. The audit
        bookkeeping happens in :meth:`confirm` and the Dialog_Manager's
        :meth:`record_executed` follow-up.
        """
        # 1. Manifest opt-in. A Skill that declares itself destructive
        #    is destructive everywhere — there is no per-args nuance to
        #    consider. This is the cheapest check and the most explicit.
        if manifest.destructive:
            return DESTRUCTIVE

        skill_name = tool_call.skill_name

        # 2. Hard-coded plain skill names.
        if skill_name in self._destructive_skills:
            return DESTRUCTIVE

        # 3. Hard-coded ``Skill.operation`` entries. We assume the
        #    discriminator is the ``operation`` argument by convention;
        #    the configured ``destructive_operations`` list is the
        #    extension point for Skills that use a different field name.
        if self._dot_op_entries:
            actual_op = tool_call.arguments.get(_DEFAULT_OPERATION_FIELD)
            for entry in self._dot_op_entries:
                if entry.skill == skill_name and entry.operation == actual_op:
                    return DESTRUCTIVE

        # 4. Configured operation-level destructive entries. The
        #    discriminator field is supplied per entry so calendars can
        #    use ``operation`` while a future Skill could use ``mode``.
        for op in self._destructive_operations:
            if op.skill != skill_name:
                continue
            actual = tool_call.arguments.get(op.op_field)
            if actual in op.op_values:
                return DESTRUCTIVE

        return SAFE

    # ------------------------------------------------------------- allowlist

    def match_allowlist(self, tool_call: ToolCall) -> TrustedAction | None:
        """Return the first allowlist entry matching ``tool_call`` or ``None``.

        Thin pass-through to :meth:`TrustedActionAllowlist.match`,
        provided so the Dialog_Manager and tests do not have to reach
        through ``policy.allowlist`` for the most common operation.
        """
        return self._allowlist.match(tool_call)

    # ---------------------------------------------------------------- confirm

    async def confirm(
        self,
        tool_call: ToolCall,
        dialog: ConfirmationDialog,
    ) -> bool:
        """Acquire user consent for a destructive ``tool_call``.

        Behaviour:

        * If the allowlist matches, we record a
          ``confirmation_requested`` audit entry — so the audit trail
          tells the same story as a non-allowlisted call — and return
          ``True`` *without* speaking the prompt or awaiting a reply.
          Requirement 16.3 mandates the bypass; the audit entry
          satisfies the "is still audited" clause.
        * Otherwise we record ``confirmation_requested`` *first*, build
          a spoken summary, and call ``dialog.ask_user(prompt)``.
          - If the response is affirmative, return ``True``. The
            Dialog_Manager is then responsible for dispatching the
            Skill and calling :meth:`record_executed` (or
            :meth:`record_error_after_confirmation`) to close the
            audit pair.
          - If the response is anything else (denial, ambiguous,
            error, empty), record a ``denied`` audit entry and return
            ``False``. Requirement 16.4 requires the cancellation to
            be audible to the caller; the Dialog_Manager surfaces
            that to the user.

        The ``confirmation_requested`` audit happens *before* the
        ``ask_user`` await point so the strict ordering invariant for
        CP9 holds even if the user's reply is delayed indefinitely or
        the dialog implementation crashes mid-await — the audit row is
        already on disk by then.
        """
        # Allowlist short-circuit (Requirement 16.3). We still emit a
        # confirmation_requested audit entry so the audit log is
        # uniform; the Dialog_Manager records the outcome with an
        # ``allowlist_bypass`` marker downstream.
        if self._allowlist.match(tool_call) is not None:
            await self._audit.record_confirmation_requested(
                skill=tool_call.skill_name,
                args_json=tool_call.arguments,
            )
            logger.debug(
                "allowlist bypass for destructive Tool_Call %s(%s)",
                tool_call.skill_name,
                tool_call.id,
            )
            return True

        # Step 1: record BEFORE asking. CP9's strict id-ordering relies
        # on this happening before any potentially long-lived await.
        await self._audit.record_confirmation_requested(
            skill=tool_call.skill_name,
            args_json=tool_call.arguments,
        )

        prompt = _build_summary(tool_call)

        # Step 2: ask the user. Any failure mode in the dialog stack
        # (TTS error, STT timeout, user hung up) is treated as denial:
        # the safety default for a Destructive_Action is to NOT take
        # the action.
        try:
            response = await dialog.ask_user(prompt)
        except Exception:
            logger.exception(
                "ConfirmationDialog raised while asking for "
                "Destructive_Action confirmation; treating as denial"
            )
            await self._audit.record_denied(
                skill=tool_call.skill_name,
                args_json=tool_call.arguments,
                outcome="dialog_error",
            )
            return False

        if not _is_affirmative(response):
            # Step 3a: explicit denial or ambiguous response.
            await self._audit.record_denied(
                skill=tool_call.skill_name,
                args_json=tool_call.arguments,
                outcome="user_denied",
            )
            return False

        # Step 3b: affirmative. The caller now owns the dispatch and
        # the matching ``executed`` audit entry.
        return True

    # ------------------------------------------------------- post-dispatch

    async def record_executed(
        self,
        tool_call: ToolCall,
        *,
        outcome: str = "ok",
        allowlist_bypass: bool = False,
    ) -> None:
        """Record the closing ``executed`` audit entry for a destructive call.

        The Dialog_Manager calls this after a successful Skill dispatch.
        It is intentionally separate from :meth:`confirm` so the audit
        write happens *after* the Skill_Registry returns — i.e., after
        the side effect actually occurred — which is exactly what CP9
        requires when comparing entry ids.

        Parameters
        ----------
        tool_call:
            The Tool_Call that was dispatched. ``args_json`` is taken
            from ``tool_call.arguments`` and canonicalised by
            :class:`AuditLog` so it byte-matches the
            ``confirmation_requested`` entry's ``args_json`` (the
            invariant CP9 checks).
        outcome:
            Short string describing the dispatch result. Defaults to
            ``"ok"``. Skills that surface non-trivial outcomes (e.g.,
            ``"sent"``, ``"created:event-id"``) should pass them here.
        allowlist_bypass:
            When ``True``, the outcome is suffixed with
            ``":allowlist_bypass"`` so operators inspecting the audit
            log can see, post-hoc, which destructive calls skipped the
            user prompt.
        """
        effective_outcome = (
            f"{outcome}:allowlist_bypass" if allowlist_bypass else outcome
        )
        await self._audit.record_executed(
            skill=tool_call.skill_name,
            args_json=tool_call.arguments,
            outcome=effective_outcome,
        )

    async def record_error_after_confirmation(
        self,
        tool_call: ToolCall,
        *,
        outcome: str,
        justification: str | None = None,
    ) -> None:
        """Record an ``error`` audit entry that pairs with a prior confirmation.

        Used when the Dialog_Manager confirmed a destructive call, the
        user said yes, but the Skill_Registry returned a structured
        error (timeout, provider_unavailable, etc.). Closing the audit
        pair with an ``error`` row keeps the ``confirmation_requested``
        entry from looking unanswered when an operator inspects the
        log later.
        """
        await self._audit.record_error(
            skill=tool_call.skill_name,
            outcome=outcome,
            justification=justification,
        )
