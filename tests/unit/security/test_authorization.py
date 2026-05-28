"""Unit tests for ``jarvis.security.authorization``.

Covers the two public types declared in the design's
``§Authorization_Policy`` section:

* :class:`AuthorizationPolicy.classify` correctly applies the closure of
  signal sources documented in Requirement 16.1 — manifest opt-in,
  hard-coded skills (plain names AND ``Skill.operation`` entries), and
  configured :class:`DestructiveOperation` discriminator entries.
* :class:`TrustedActionAllowlist.match` performs a deep subset match,
  honours first-match-wins, and returns ``None`` on miss
  (Requirement 16.3).
* :meth:`AuthorizationPolicy.confirm` records
  ``confirmation_requested`` *before* it speaks the prompt and treats a
  denied / errored / ambiguous response as cancellation
  (Requirement 16.2 / 16.4 / 16.5). The allowlist short-circuit still
  emits the audit row but skips the prompt (Requirement 16.3).
* The helper recorders :meth:`record_executed` /
  :meth:`record_error_after_confirmation` close out the audit pair so
  CP9's strict id-ordering invariant
  ``confirmation_requested.id < executed.id`` /
  ``confirmation_requested.id < denied.id`` holds.

Validates: Requirements 16.1, 16.3, 16.5
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from jarvis.config.schema import DestructiveOperation, TrustedAction
from jarvis.llm.base import ToolCall
from jarvis.security.audit_log import AuditLog
from jarvis.security.authorization import (
    DEFAULT_DESTRUCTIVE_SKILLS,
    DESTRUCTIVE,
    SAFE,
    AuthorizationPolicy,
    TrustedActionAllowlist,
)
from jarvis.skills.base import SkillManifest
from jarvis.utils.time_source import FakeTimeSource

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _run(coro: Awaitable[Any]) -> Any:
    """Synchronously execute a coroutine on a fresh event loop.

    Mirrors ``tests/unit/security/test_audit_log.py`` so the file does
    not require ``pytest-asyncio`` configuration.
    """
    return asyncio.run(coro)  # type: ignore[arg-type]


def _benign_schema() -> dict[str, Any]:
    """A minimal Mistral-compatible JSON Schema for fixture skills."""
    return {"type": "object", "properties": {}, "additionalProperties": True}


def _manifest(
    name: str,
    *,
    destructive: bool = False,
) -> SkillManifest:
    """Build a SkillManifest with reasonable defaults for tests."""
    return SkillManifest(
        name=name,
        description=f"{name} test fixture",
        json_schema=_benign_schema(),
        destructive=destructive,
    )


def _tool_call(
    skill_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    call_id: str = "call-1",
    raw_arguments: str | None = None,
) -> ToolCall:
    """Build a :class:`ToolCall` for fixture use."""
    args: dict[str, Any] = {} if arguments is None else dict(arguments)
    raw = raw_arguments if raw_arguments is not None else "{}"
    return ToolCall(
        id=call_id,
        skill_name=skill_name,
        arguments=args,
        raw_arguments=raw,
    )


class FakeConfirmationDialog:
    """A canned :class:`ConfirmationDialog` for deterministic tests.

    Records every prompt it was asked so callers can assert that
    :meth:`AuthorizationPolicy.confirm` actually invoked ``ask_user``
    (or, for the allowlist short-circuit case, did NOT invoke it).
    Optionally raises a configured exception to exercise the
    "ask_user error → treated as denial" code path.
    """

    def __init__(
        self,
        *,
        response: str = "no",
        raises: BaseException | None = None,
    ) -> None:
        self.response: str = response
        self.raises: BaseException | None = raises
        self.prompts: list[str] = []
        self.call_count: int = 0

    async def ask_user(self, prompt: str) -> str:
        self.call_count += 1
        self.prompts.append(prompt)
        if self.raises is not None:
            raise self.raises
        return self.response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def time_source() -> FakeTimeSource:
    return FakeTimeSource(now=datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC))


@pytest.fixture()
def audit_log_factory(
    tmp_path: Path, time_source: FakeTimeSource
) -> Callable[..., AuditLog]:
    """Build a fresh file-backed :class:`AuditLog` per call."""
    counter = {"n": 0}

    def _make(**kwargs: Any) -> AuditLog:
        counter["n"] += 1
        db_path = tmp_path / f"audit-{counter['n']}.sqlite"
        return AuditLog(
            db_path,
            time_source=kwargs.pop("time_source", time_source),
            run_id=kwargs.pop("run_id", "test-run"),
            **kwargs,
        )

    return _make


@pytest.fixture()
def audit_log(
    audit_log_factory: Callable[..., AuditLog],
) -> Iterator[AuditLog]:
    log = audit_log_factory()
    yield log
    log.close()


@pytest.fixture()
def policy(audit_log: AuditLog) -> AuthorizationPolicy:
    """Default policy: hard-coded destructive defaults, empty allowlist."""
    return AuthorizationPolicy(
        allowlist=TrustedActionAllowlist(),
        audit=audit_log,
    )


# ===========================================================================
# classify() — Requirement 16.1
# ===========================================================================


class TestClassifyManifestFlag:
    """``manifest.destructive=True`` → :data:`DESTRUCTIVE`."""

    def test_manifest_destructive_true_classifies_destructive(
        self, policy: AuthorizationPolicy
    ) -> None:
        # A skill not on any hard-coded list — the manifest flag alone
        # MUST flip the classification.
        call = _tool_call("BenignSkill")
        manifest = _manifest("BenignSkill", destructive=True)
        assert policy.classify(call, manifest) == DESTRUCTIVE

    def test_manifest_destructive_false_with_benign_skill_is_safe(
        self, policy: AuthorizationPolicy
    ) -> None:
        call = _tool_call("ReadFileSkill")
        manifest = _manifest("ReadFileSkill", destructive=False)
        assert policy.classify(call, manifest) == SAFE

    def test_manifest_flag_overrides_lack_of_other_signals(
        self, audit_log: AuditLog
    ) -> None:
        # No hard-coded skills at all; manifest flag is the only signal
        # left and must still classify the call as destructive.
        policy = AuthorizationPolicy(
            allowlist=TrustedActionAllowlist(),
            audit=audit_log,
            hard_coded_destructive_skills=(),
            destructive_operations=(),
        )
        call = _tool_call("UnusualSkill")
        manifest = _manifest("UnusualSkill", destructive=True)
        assert policy.classify(call, manifest) == DESTRUCTIVE


class TestClassifyHardCodedSkills:
    """Each hard-coded plain skill name from Requirement 16.1."""

    @pytest.mark.parametrize(
        "skill_name",
        ["SendEmailSkill", "SendMessageSkill", "RunScriptSkill"],
    )
    def test_hard_coded_destructive_skill_is_classified_destructive(
        self, policy: AuthorizationPolicy, skill_name: str
    ) -> None:
        call = _tool_call(skill_name, {"foo": "bar"})
        manifest = _manifest(skill_name, destructive=False)
        assert policy.classify(call, manifest) == DESTRUCTIVE

    def test_default_destructive_skills_constant_lists_required_entries(
        self,
    ) -> None:
        # Requirement 16.1 explicitly enumerates these five entries.
        assert "SendEmailSkill" in DEFAULT_DESTRUCTIVE_SKILLS
        assert "SendMessageSkill" in DEFAULT_DESTRUCTIVE_SKILLS
        assert "RunScriptSkill" in DEFAULT_DESTRUCTIVE_SKILLS
        assert "MemoryAdminSkill.forget" in DEFAULT_DESTRUCTIVE_SKILLS
        assert "CalendarSkill.create_event" in DEFAULT_DESTRUCTIVE_SKILLS


class TestClassifyDotOperationEntries:
    """Hard-coded ``Skill.operation`` entries from Requirement 16.1."""

    def test_memory_admin_forget_operation_is_destructive(
        self, policy: AuthorizationPolicy
    ) -> None:
        call = _tool_call(
            "MemoryAdminSkill", {"operation": "forget", "id": "rec-1"}
        )
        manifest = _manifest("MemoryAdminSkill", destructive=False)
        assert policy.classify(call, manifest) == DESTRUCTIVE

    def test_memory_admin_other_operation_is_safe(
        self, policy: AuthorizationPolicy
    ) -> None:
        # ``MemoryAdminSkill.forget`` is operation-scoped — listing
        # records is a benign read and MUST stay safe.
        call = _tool_call(
            "MemoryAdminSkill", {"operation": "list"}
        )
        manifest = _manifest("MemoryAdminSkill", destructive=False)
        assert policy.classify(call, manifest) == SAFE

    def test_calendar_create_event_operation_is_destructive(
        self, policy: AuthorizationPolicy
    ) -> None:
        call = _tool_call(
            "CalendarSkill",
            {"operation": "create_event", "title": "Standup"},
        )
        manifest = _manifest("CalendarSkill", destructive=False)
        assert policy.classify(call, manifest) == DESTRUCTIVE

    def test_calendar_read_only_operation_is_safe(
        self, policy: AuthorizationPolicy
    ) -> None:
        # ``CalendarSkill.next_event`` is not enumerated as destructive.
        call = _tool_call(
            "CalendarSkill", {"operation": "next_event"}
        )
        manifest = _manifest("CalendarSkill", destructive=False)
        assert policy.classify(call, manifest) == SAFE

    def test_dot_op_entry_with_missing_operation_argument_is_safe(
        self, policy: AuthorizationPolicy
    ) -> None:
        # If the model omits ``operation`` we cannot match the dot-op
        # entry and the call falls through to the SAFE default. The
        # registry's JSON Schema validation would normally reject such
        # a call upstream; the policy must not crash if it gets one.
        call = _tool_call("MemoryAdminSkill", {"id": "rec-1"})
        manifest = _manifest("MemoryAdminSkill", destructive=False)
        assert policy.classify(call, manifest) == SAFE


class TestClassifyConfiguredDestructiveOperations:
    """Configured :class:`DestructiveOperation` entries (Requirement 16.1)."""

    def test_configured_op_value_classifies_destructive(
        self, audit_log: AuditLog
    ) -> None:
        policy = AuthorizationPolicy(
            allowlist=TrustedActionAllowlist(),
            audit=audit_log,
            hard_coded_destructive_skills=(),
            destructive_operations=(
                DestructiveOperation(
                    skill="CalendarSkill",
                    op_field="operation",
                    op_values=["create_event", "delete_event"],
                ),
            ),
        )
        for op in ("create_event", "delete_event"):
            call = _tool_call("CalendarSkill", {"operation": op})
            manifest = _manifest("CalendarSkill", destructive=False)
            assert policy.classify(call, manifest) == DESTRUCTIVE

    def test_configured_op_value_not_in_set_is_safe(
        self, audit_log: AuditLog
    ) -> None:
        policy = AuthorizationPolicy(
            allowlist=TrustedActionAllowlist(),
            audit=audit_log,
            hard_coded_destructive_skills=(),
            destructive_operations=(
                DestructiveOperation(
                    skill="CalendarSkill",
                    op_field="operation",
                    op_values=["create_event"],
                ),
            ),
        )
        call = _tool_call("CalendarSkill", {"operation": "next_event"})
        manifest = _manifest("CalendarSkill", destructive=False)
        assert policy.classify(call, manifest) == SAFE

    def test_configured_op_uses_custom_field_name(
        self, audit_log: AuditLog
    ) -> None:
        # Operators may register a Skill that discriminates on a field
        # other than ``operation``. The policy must honour ``op_field``.
        policy = AuthorizationPolicy(
            allowlist=TrustedActionAllowlist(),
            audit=audit_log,
            hard_coded_destructive_skills=(),
            destructive_operations=(
                DestructiveOperation(
                    skill="DeviceControlSkill",
                    op_field="mode",
                    op_values=["shutdown"],
                ),
            ),
        )
        call = _tool_call("DeviceControlSkill", {"mode": "shutdown"})
        manifest = _manifest("DeviceControlSkill", destructive=False)
        assert policy.classify(call, manifest) == DESTRUCTIVE

        # Same skill, different (non-listed) mode: SAFE.
        safe_call = _tool_call("DeviceControlSkill", {"mode": "status"})
        assert policy.classify(safe_call, manifest) == SAFE

    def test_configured_op_for_other_skill_does_not_match(
        self, audit_log: AuditLog
    ) -> None:
        policy = AuthorizationPolicy(
            allowlist=TrustedActionAllowlist(),
            audit=audit_log,
            hard_coded_destructive_skills=(),
            destructive_operations=(
                DestructiveOperation(
                    skill="CalendarSkill",
                    op_field="operation",
                    op_values=["create_event"],
                ),
            ),
        )
        # ``operation == create_event`` only marks ``CalendarSkill`` —
        # other skills with the same arg value remain safe.
        call = _tool_call("ReadFileSkill", {"operation": "create_event"})
        manifest = _manifest("ReadFileSkill", destructive=False)
        assert policy.classify(call, manifest) == SAFE


class TestClassifyBenignSkills:
    """Benign skills with no destructive signal must classify SAFE."""

    @pytest.mark.parametrize(
        "skill_name",
        [
            "ReadFileSkill",
            "WeatherSkill",
            "NewsSkill",
            "WebSearchSkill",
            "LaunchAppSkill",
        ],
    )
    def test_benign_skill_is_safe(
        self, policy: AuthorizationPolicy, skill_name: str
    ) -> None:
        call = _tool_call(skill_name, {"path": "C:/Users/me/notes.txt"})
        manifest = _manifest(skill_name, destructive=False)
        assert policy.classify(call, manifest) == SAFE

    def test_safe_classification_is_pure_no_audit_writes(
        self, policy: AuthorizationPolicy, audit_log: AuditLog
    ) -> None:
        # ``classify`` must NOT touch the audit log; the audit row is
        # the Dialog_Manager's responsibility (after dispatch).
        call = _tool_call("ReadFileSkill", {"path": "C:/x"})
        manifest = _manifest("ReadFileSkill", destructive=False)
        policy.classify(call, manifest)
        assert audit_log.count() == 0


# ===========================================================================
# TrustedActionAllowlist.match — Requirement 16.3
# ===========================================================================


class TestAllowlistMatch:
    """Deep subset matching, first-match-wins, no-match returns None."""

    def test_empty_allowlist_returns_none(self) -> None:
        allow = TrustedActionAllowlist()
        call = _tool_call("SendEmailSkill", {"to": "alex@example.invalid"})
        assert allow.match(call) is None

    def test_no_skill_match_returns_none(self) -> None:
        allow = TrustedActionAllowlist(
            [TrustedAction(skill="SendEmailSkill", args_subset={})]
        )
        call = _tool_call("SendMessageSkill", {})
        assert allow.match(call) is None

    def test_args_mismatch_returns_none(self) -> None:
        allow = TrustedActionAllowlist(
            [
                TrustedAction(
                    skill="SendEmailSkill",
                    args_subset={"to": "alex@example.invalid"},
                )
            ]
        )
        # Different recipient — must not match.
        call = _tool_call(
            "SendEmailSkill", {"to": "bob@example.invalid"}
        )
        assert allow.match(call) is None

    def test_exact_args_match_returns_entry(self) -> None:
        entry = TrustedAction(
            skill="SendEmailSkill",
            args_subset={"to": "alex@example.invalid"},
        )
        allow = TrustedActionAllowlist([entry])
        call = _tool_call(
            "SendEmailSkill",
            {"to": "alex@example.invalid", "subject": "hi", "body": "x"},
        )
        # Subset semantics: extra fields on the actual call are fine.
        assert allow.match(call) is entry

    def test_empty_args_subset_matches_any_args(self) -> None:
        # An empty ``args_subset`` is the documented "match any args"
        # form; it should match a call whose arguments are themselves
        # arbitrary (Requirement 16.3).
        entry = TrustedAction(skill="RunScriptSkill", args_subset={})
        allow = TrustedActionAllowlist([entry])
        call = _tool_call("RunScriptSkill", {"script_id": "anything"})
        assert allow.match(call) is entry

    def test_deep_nested_subset_is_matched(self) -> None:
        entry = TrustedAction(
            skill="SendMessageSkill",
            args_subset={
                "channel": "slack",
                "metadata": {"workspace": "alex-team"},
            },
        )
        allow = TrustedActionAllowlist([entry])

        match_call = _tool_call(
            "SendMessageSkill",
            {
                "channel": "slack",
                "recipient": "alex",
                "metadata": {"workspace": "alex-team", "thread_id": "abc"},
            },
        )
        assert allow.match(match_call) is entry

        # Same skill, nested mismatch (different workspace) — must not match.
        miss_call = _tool_call(
            "SendMessageSkill",
            {
                "channel": "slack",
                "metadata": {"workspace": "other-team"},
            },
        )
        assert allow.match(miss_call) is None

    def test_first_match_wins(self) -> None:
        first = TrustedAction(
            skill="SendEmailSkill",
            args_subset={"to": "alex@example.invalid"},
        )
        second = TrustedAction(
            skill="SendEmailSkill",
            args_subset={"to": "alex@example.invalid"},
        )
        allow = TrustedActionAllowlist([first, second])
        call = _tool_call(
            "SendEmailSkill", {"to": "alex@example.invalid"}
        )
        # The first declared entry wins when both could match — this is
        # what lets users layer permissive rules above more specific
        # overrides without deleting the broader rule.
        assert allow.match(call) is first

    def test_specific_entry_can_shadow_broader_one(self) -> None:
        """Specific entries declared FIRST shadow broader ones below them."""
        specific = TrustedAction(
            skill="SendMessageSkill",
            args_subset={"channel": "slack", "recipient": "team-bot"},
        )
        broad = TrustedAction(
            skill="SendMessageSkill",
            args_subset={"channel": "slack"},
        )
        allow = TrustedActionAllowlist([specific, broad])

        # Matches both entries; first one wins.
        specific_call = _tool_call(
            "SendMessageSkill",
            {"channel": "slack", "recipient": "team-bot", "body": "hi"},
        )
        assert allow.match(specific_call) is specific

        # Only the broad entry matches; the specific one is bypassed.
        broad_call = _tool_call(
            "SendMessageSkill",
            {"channel": "slack", "recipient": "alex"},
        )
        assert allow.match(broad_call) is broad

    def test_list_arguments_must_match_exactly(self) -> None:
        """List arguments are matched by equality, not as sets."""
        entry = TrustedAction(
            skill="SendEmailSkill",
            args_subset={"to": ["alex@example.invalid"]},
        )
        allow = TrustedActionAllowlist([entry])

        exact = _tool_call(
            "SendEmailSkill", {"to": ["alex@example.invalid"]}
        )
        assert allow.match(exact) is entry

        reordered = _tool_call(
            "SendEmailSkill",
            {"to": ["bob@example.invalid", "alex@example.invalid"]},
        )
        # Different list contents — equality fails, allowlist must miss.
        assert allow.match(reordered) is None

    def test_constructor_rejects_non_trusted_action_entries(self) -> None:
        with pytest.raises(TypeError, match="TrustedAction"):
            TrustedActionAllowlist(
                [{"skill": "SendEmailSkill", "args_subset": {}}]  # type: ignore[list-item]
            )


# ===========================================================================
# AuthorizationPolicy.confirm — Requirement 16.5 audit ordering
# ===========================================================================


class TestConfirmConfirmationRequestedRecordedFirst:
    """``confirmation_requested`` is recorded BEFORE ask_user is called.

    This is the precondition for CP9: the audit row must exist on disk
    before the dialog stack starts an awaitable that may take seconds
    or never resolve.
    """

    def test_confirmation_requested_id_is_recorded_before_ask_user(
        self, policy: AuthorizationPolicy, audit_log: AuditLog
    ) -> None:
        observed_count_at_ask: list[int] = []

        class ObservingDialog:
            async def ask_user(self, prompt: str) -> str:
                # When ask_user is invoked, the confirmation_requested
                # row must already be persisted.
                observed_count_at_ask.append(audit_log.count())
                return "yes"

        call = _tool_call(
            "SendEmailSkill",
            {"to": "alex@example.invalid"},
            raw_arguments='{"to":"alex@example.invalid"}',
        )

        result = _run(policy.confirm(call, ObservingDialog()))
        assert result is True
        # confirmation_requested was already on disk when ask_user ran.
        assert observed_count_at_ask == [1]

    def test_confirmation_requested_payload_matches_tool_call(
        self, policy: AuthorizationPolicy, audit_log: AuditLog
    ) -> None:
        call = _tool_call(
            "SendEmailSkill",
            {"to": "alex@example.invalid", "subject": "hi"},
        )
        dialog = FakeConfirmationDialog(response="no")
        _run(policy.confirm(call, dialog))

        entries = audit_log.entries()
        assert len(entries) == 2
        confirm_entry, denied_entry = entries
        assert confirm_entry.kind == "confirmation_requested"
        assert confirm_entry.skill == "SendEmailSkill"
        # args_json is canonicalised by AuditLog so subject/to ordering
        # is stable regardless of dict insertion order.
        assert confirm_entry.args_json == (
            '{"subject":"hi","to":"alex@example.invalid"}'
        )
        # And the denied entry shares those fields (CP9 pairs by them).
        assert denied_entry.skill == "SendEmailSkill"
        assert denied_entry.args_json == confirm_entry.args_json


class TestConfirmDenialPath:
    """Denied / ambiguous / errored responses → ``denied`` audit + False."""

    def test_explicit_no_records_denied_and_returns_false(
        self, policy: AuthorizationPolicy, audit_log: AuditLog
    ) -> None:
        call = _tool_call(
            "SendEmailSkill", {"to": "alex@example.invalid"}
        )
        dialog = FakeConfirmationDialog(response="no")

        result = _run(policy.confirm(call, dialog))

        assert result is False
        # Dialog was actually consulted.
        assert dialog.call_count == 1
        # Audit ordering: confirmation_requested then denied.
        kinds = [e.kind for e in audit_log.entries()]
        assert kinds == ["confirmation_requested", "denied"]
        # Strict id ordering required by CP9.
        confirm_entry, denied_entry = audit_log.entries()
        assert confirm_entry.id < denied_entry.id
        # Outcome marker for analyst-readable filtering.
        assert denied_entry.outcome == "user_denied"

    def test_ambiguous_response_is_treated_as_denial(
        self, policy: AuthorizationPolicy, audit_log: AuditLog
    ) -> None:
        # The safety default for a Destructive_Action is denial, so a
        # response with no clear affirmative MUST cancel the call.
        call = _tool_call(
            "SendEmailSkill", {"to": "alex@example.invalid"}
        )
        dialog = FakeConfirmationDialog(response="hmm, maybe later")

        result = _run(policy.confirm(call, dialog))

        assert result is False
        assert [e.kind for e in audit_log.entries()] == [
            "confirmation_requested",
            "denied",
        ]

    def test_empty_response_is_treated_as_denial(
        self, policy: AuthorizationPolicy, audit_log: AuditLog
    ) -> None:
        call = _tool_call(
            "SendEmailSkill", {"to": "alex@example.invalid"}
        )
        dialog = FakeConfirmationDialog(response="")
        assert _run(policy.confirm(call, dialog)) is False
        assert [e.kind for e in audit_log.entries()] == [
            "confirmation_requested",
            "denied",
        ]

    def test_ask_user_exception_records_dialog_error_denial(
        self, policy: AuthorizationPolicy, audit_log: AuditLog
    ) -> None:
        call = _tool_call(
            "SendEmailSkill", {"to": "alex@example.invalid"}
        )
        dialog = FakeConfirmationDialog(
            raises=RuntimeError("STT timeout"),
        )

        result = _run(policy.confirm(call, dialog))

        assert result is False
        entries = audit_log.entries()
        # confirmation_requested row was written BEFORE ask_user raised,
        # so it survives even on a dialog crash — this is exactly the
        # invariant CP9 leans on.
        kinds = [e.kind for e in entries]
        assert kinds == ["confirmation_requested", "denied"]
        # Distinct outcome lets operators distinguish "user said no"
        # from "dialog stack failed".
        confirm_entry, denied_entry = entries
        assert confirm_entry.id < denied_entry.id
        assert denied_entry.outcome == "dialog_error"


class TestConfirmAffirmativePath:
    """An affirmative response returns True; caller records executed."""

    def test_affirmative_response_returns_true_and_records_only_confirmation(
        self, policy: AuthorizationPolicy, audit_log: AuditLog
    ) -> None:
        call = _tool_call(
            "SendEmailSkill", {"to": "alex@example.invalid"}
        )
        dialog = FakeConfirmationDialog(response="yes")

        result = _run(policy.confirm(call, dialog))

        assert result is True
        # Only the confirmation_requested row exists; the matching
        # ``executed`` row is the Dialog_Manager's responsibility.
        kinds = [e.kind for e in audit_log.entries()]
        assert kinds == ["confirmation_requested"]

    @pytest.mark.parametrize(
        "response",
        [
            "yes",
            "yes please",
            "Yes.",
            "sure",
            "okay",
            "go ahead",
            "do it",
            "send it",
            "proceed",
            "confirm",
            "affirmative",
        ],
    )
    def test_assorted_affirmative_phrasings_are_accepted(
        self, audit_log_factory: Callable[..., AuditLog], response: str
    ) -> None:
        # Each phrasing gets its own log so the audit assertion below
        # is over a single confirm call.
        log = audit_log_factory()
        try:
            policy = AuthorizationPolicy(
                allowlist=TrustedActionAllowlist(),
                audit=log,
            )
            call = _tool_call("SendEmailSkill", {"to": "x@y.invalid"})
            dialog = FakeConfirmationDialog(response=response)
            assert _run(policy.confirm(call, dialog)) is True
        finally:
            log.close()


class TestConfirmAuditOrdering:
    """Audit ordering required by Requirement 16.5 / CP9.

    ``confirmation_requested.id`` MUST be strictly less than the
    matching ``executed`` or ``denied`` entry's id.
    """

    def test_confirmation_id_lt_executed_id_after_record_executed(
        self, policy: AuthorizationPolicy, audit_log: AuditLog
    ) -> None:
        call = _tool_call(
            "SendEmailSkill", {"to": "alex@example.invalid"}
        )
        dialog = FakeConfirmationDialog(response="yes")

        async def driver() -> None:
            consented = await policy.confirm(call, dialog)
            assert consented is True
            # Caller (Dialog_Manager) closes the audit pair after the
            # Skill_Registry returns success.
            await policy.record_executed(call, outcome="ok")

        _run(driver())

        entries = audit_log.entries()
        assert [e.kind for e in entries] == [
            "confirmation_requested",
            "executed",
        ]
        confirm_entry, executed_entry = entries
        # Strict id ordering — CP9.
        assert confirm_entry.id < executed_entry.id
        # And both rows pair cleanly via (skill, args_json).
        assert confirm_entry.skill == executed_entry.skill
        assert confirm_entry.args_json == executed_entry.args_json

    def test_confirmation_id_lt_denied_id(
        self, policy: AuthorizationPolicy, audit_log: AuditLog
    ) -> None:
        call = _tool_call(
            "SendEmailSkill", {"to": "alex@example.invalid"}
        )
        dialog = FakeConfirmationDialog(response="no")
        _run(policy.confirm(call, dialog))

        confirm_entry, denied_entry = audit_log.entries()
        assert confirm_entry.id < denied_entry.id


class TestConfirmAllowlistBypass:
    """Allowlist match bypasses the prompt but still emits the audit row.

    Per Requirement 16.3, a matching trusted-action entry skips the
    affirmative-response requirement. Per Requirement 16.5, the audit
    log must still tell a complete story — so a
    ``confirmation_requested`` row is still emitted.
    """

    def _allowlist(self) -> TrustedActionAllowlist:
        return TrustedActionAllowlist(
            [
                TrustedAction(
                    skill="SendEmailSkill",
                    args_subset={"to": "alex@example.invalid"},
                )
            ]
        )

    def test_allowlist_bypass_skips_ask_user(
        self, audit_log: AuditLog
    ) -> None:
        policy = AuthorizationPolicy(
            allowlist=self._allowlist(), audit=audit_log
        )
        call = _tool_call(
            "SendEmailSkill",
            {"to": "alex@example.invalid", "subject": "hi"},
        )
        dialog = FakeConfirmationDialog(response="no")  # would deny

        result = _run(policy.confirm(call, dialog))

        # Returned True even though the dialog would have denied — the
        # bypass is by design.
        assert result is True
        # ``ask_user`` was NEVER called — that is the whole point.
        assert dialog.call_count == 0
        assert dialog.prompts == []

    def test_allowlist_bypass_still_records_confirmation_requested(
        self, audit_log: AuditLog
    ) -> None:
        policy = AuthorizationPolicy(
            allowlist=self._allowlist(), audit=audit_log
        )
        call = _tool_call(
            "SendEmailSkill", {"to": "alex@example.invalid"}
        )
        dialog = FakeConfirmationDialog(response="yes")

        _run(policy.confirm(call, dialog))

        entries = audit_log.entries()
        assert len(entries) == 1
        assert entries[0].kind == "confirmation_requested"
        assert entries[0].skill == "SendEmailSkill"

    def test_allowlist_bypass_audit_pairs_with_executed_under_cp9(
        self, audit_log: AuditLog
    ) -> None:
        policy = AuthorizationPolicy(
            allowlist=self._allowlist(), audit=audit_log
        )
        call = _tool_call(
            "SendEmailSkill", {"to": "alex@example.invalid"}
        )
        dialog = FakeConfirmationDialog(response="no")  # ignored

        async def driver() -> None:
            assert await policy.confirm(call, dialog) is True
            await policy.record_executed(
                call, outcome="ok", allowlist_bypass=True
            )

        _run(driver())

        entries = audit_log.entries()
        assert [e.kind for e in entries] == [
            "confirmation_requested",
            "executed",
        ]
        confirm_entry, executed_entry = entries
        # CP9 ordering still holds for allowlisted calls.
        assert confirm_entry.id < executed_entry.id
        # The ``allowlist_bypass`` marker lets operators distinguish
        # bypassed from prompted dispatches in the audit trail.
        assert executed_entry.outcome == "ok:allowlist_bypass"

    def test_allowlist_miss_falls_through_to_prompt(
        self, audit_log: AuditLog
    ) -> None:
        policy = AuthorizationPolicy(
            allowlist=self._allowlist(), audit=audit_log
        )
        # Same skill, different recipient — does NOT match the entry.
        call = _tool_call(
            "SendEmailSkill", {"to": "bob@example.invalid"}
        )
        dialog = FakeConfirmationDialog(response="yes")

        assert _run(policy.confirm(call, dialog)) is True
        # Prompt was actually issued because no allowlist match.
        assert dialog.call_count == 1


# ===========================================================================
# Helper recorders — record_executed / record_error_after_confirmation
# ===========================================================================


class TestRecordExecuted:
    """``record_executed`` closes the audit pair with the supplied outcome."""

    def test_default_outcome_is_ok(
        self, policy: AuthorizationPolicy, audit_log: AuditLog
    ) -> None:
        call = _tool_call("SendEmailSkill", {"to": "x@y.invalid"})
        _run(policy.record_executed(call))

        entries = audit_log.entries()
        assert len(entries) == 1
        assert entries[0].kind == "executed"
        assert entries[0].outcome == "ok"
        assert entries[0].skill == "SendEmailSkill"

    def test_custom_outcome_passed_through(
        self, policy: AuthorizationPolicy, audit_log: AuditLog
    ) -> None:
        call = _tool_call(
            "CalendarSkill",
            {"operation": "create_event", "title": "Standup"},
        )
        _run(policy.record_executed(call, outcome="created:event-123"))

        entry = audit_log.entries()[0]
        assert entry.outcome == "created:event-123"

    def test_allowlist_bypass_marker_is_appended(
        self, policy: AuthorizationPolicy, audit_log: AuditLog
    ) -> None:
        call = _tool_call("SendEmailSkill", {"to": "x@y.invalid"})
        _run(policy.record_executed(call, outcome="sent", allowlist_bypass=True))

        entry = audit_log.entries()[0]
        assert entry.outcome == "sent:allowlist_bypass"


class TestRecordErrorAfterConfirmation:
    """``record_error_after_confirmation`` closes pairs that errored."""

    def test_records_error_kind_with_outcome(
        self, policy: AuthorizationPolicy, audit_log: AuditLog
    ) -> None:
        call = _tool_call("SendEmailSkill", {"to": "x@y.invalid"})
        _run(
            policy.record_error_after_confirmation(
                call, outcome="provider_unavailable"
            )
        )

        entry = audit_log.entries()[0]
        assert entry.kind == "error"
        assert entry.outcome == "provider_unavailable"
        assert entry.skill == "SendEmailSkill"

    def test_optional_justification_is_persisted(
        self, policy: AuthorizationPolicy, audit_log: AuditLog
    ) -> None:
        call = _tool_call("SendEmailSkill", {"to": "x@y.invalid"})
        _run(
            policy.record_error_after_confirmation(
                call,
                outcome="internal_error:trace-7f3b",
                justification="ZeroDivisionError",
            )
        )

        entry = audit_log.entries()[0]
        assert entry.outcome == "internal_error:trace-7f3b"
        assert entry.justification == "ZeroDivisionError"


class TestPostConfirmationOrdering:
    """Full confirm → record_executed and confirm → record_error pairs."""

    def test_confirmation_then_executed_strictly_orders(
        self, policy: AuthorizationPolicy, audit_log: AuditLog
    ) -> None:
        call = _tool_call("SendEmailSkill", {"to": "x@y.invalid"})
        dialog = FakeConfirmationDialog(response="yes")

        async def driver() -> None:
            assert await policy.confirm(call, dialog) is True
            await policy.record_executed(call, outcome="ok")

        _run(driver())

        entries = audit_log.entries()
        # Two rows, in the right order, with strictly increasing ids.
        assert [e.kind for e in entries] == [
            "confirmation_requested",
            "executed",
        ]
        assert entries[0].id < entries[1].id

    def test_confirmation_then_error_strictly_orders(
        self, policy: AuthorizationPolicy, audit_log: AuditLog
    ) -> None:
        call = _tool_call("SendEmailSkill", {"to": "x@y.invalid"})
        dialog = FakeConfirmationDialog(response="yes")

        async def driver() -> None:
            assert await policy.confirm(call, dialog) is True
            await policy.record_error_after_confirmation(
                call, outcome="timeout"
            )

        _run(driver())

        entries = audit_log.entries()
        assert [e.kind for e in entries] == [
            "confirmation_requested",
            "error",
        ]
        assert entries[0].id < entries[1].id


# ===========================================================================
# Constructor validation
# ===========================================================================


class TestPolicyConstructorValidation:
    """Defensive type checks at construction time."""

    def test_rejects_non_allowlist_object(self, audit_log: AuditLog) -> None:
        with pytest.raises(TypeError, match="allowlist"):
            AuthorizationPolicy(
                allowlist=[],  # type: ignore[arg-type]
                audit=audit_log,
            )

    def test_rejects_non_audit_log_object(self) -> None:
        with pytest.raises(TypeError, match="audit"):
            AuthorizationPolicy(
                allowlist=TrustedActionAllowlist(),
                audit=object(),  # type: ignore[arg-type]
            )

    def test_rejects_blank_destructive_skill_entry(
        self, audit_log: AuditLog
    ) -> None:
        with pytest.raises(ValueError, match="non-empty strings"):
            AuthorizationPolicy(
                allowlist=TrustedActionAllowlist(),
                audit=audit_log,
                hard_coded_destructive_skills=("SendEmailSkill", ""),
            )

    def test_rejects_malformed_dot_op_entry(
        self, audit_log: AuditLog
    ) -> None:
        with pytest.raises(ValueError, match=r"Skill\.operation"):
            AuthorizationPolicy(
                allowlist=TrustedActionAllowlist(),
                audit=audit_log,
                hard_coded_destructive_skills=("CalendarSkill.",),
            )

    def test_rejects_non_destructive_operation_entries(
        self, audit_log: AuditLog
    ) -> None:
        with pytest.raises(TypeError, match="DestructiveOperation"):
            AuthorizationPolicy(
                allowlist=TrustedActionAllowlist(),
                audit=audit_log,
                destructive_operations=(
                    {"skill": "X", "op_field": "f", "op_values": ["v"]},  # type: ignore[arg-type]
                ),
            )
