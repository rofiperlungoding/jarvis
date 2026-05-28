"""Property 6 — Authorization audit precedes destructive dispatch.

From ``design.md §Correctness Properties``:

    *For any* sequence of Tool_Calls dispatched by ``DialogManager``,
    *for every* Tool_Call ``C`` whose Skill is classified
    ``Destructive`` and is not matched by the trusted-action allowlist,
    the audit log SHALL contain a ``confirmation_requested`` entry whose
    ``id`` is strictly less than the corresponding ``executed`` or
    ``denied`` entry's ``id`` and whose ``skill`` and ``args_json``
    match ``C``.

This file expresses that universal quantification with Hypothesis. The
strategy generates schema-valid arguments for a representative
destructive Skill (:class:`~jarvis.skills.builtin.send_email.SendEmailSkill`),
builds a real :class:`AuditLog` / :class:`AuthorizationPolicy` /
:class:`SkillRegistry` triple, and drives the full confirm + dispatch
flow exactly as the :class:`DialogManager` does. The assertion battery
then walks the audit log and verifies the ordering invariant directly:

* exactly two rows are written per destructive call;
* row 0 is ``confirmation_requested``, row 1 is either ``executed``
  (consent path) or ``denied`` (refusal path);
* row 0's ``id`` is strictly less than row 1's; and
* both rows share ``skill`` and ``args_json`` (the canonical JSON form
  produced by :class:`AuditLog`'s argument serialiser).

Why a recording wrapper around ``SendEmailSkill``?
--------------------------------------------------

The property is about audit ordering, not about SMTP transport. We want
the registry to use the *real* :class:`SendEmailSkill` JSON Schema (so
``tool_call_arguments`` and the registry's :class:`Draft7Validator`
agree on the input space — Property 2 / CP2's contract) but we do
**not** want the executor body to reach for an :class:`EmailClient`
provider that is not wired up in the property-test environment. The
:class:`_RecordingSkill` wrapper preserves the manifest verbatim and
substitutes a deterministic success-returning executor, the same shape
the existing Property 2 test uses.

Why ``SendEmailSkill``?
-----------------------

The task description names ``SendEmailSkill`` explicitly as a
representative destructive skill. Its schema is also the most
information-rich of the hard-coded destructive skills (three
constrained strings: recipient, subject, body), so the strategy
exercises a non-trivial ``args_json`` space rather than a single empty
object — which makes the "matching ``args_json``" half of the property
meaningful.

Validates: Requirements 16.1, 16.2, 16.3, 16.5 (CP9)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from hypothesis import HealthCheck, given, settings, strategies as st
from tests.strategies import tool_call_arguments

from jarvis.llm.base import ToolCall
from jarvis.security.audit_log import AuditLog
from jarvis.security.authorization import (
    DESTRUCTIVE,
    AuthorizationPolicy,
    ConfirmationDialog,
    TrustedActionAllowlist,
)
from jarvis.skills.base import (
    SkillContext,
    SkillManifest,
    SkillResult,
)
from jarvis.skills.builtin.send_email import SendEmailSkill
from jarvis.skills.registry import SkillRegistry

# ---------------------------------------------------------------------------
# Recording wrapper
# ---------------------------------------------------------------------------


class _RecordingSkill:
    """Wrap a real :class:`Skill`, replacing ``execute`` with a recorder.

    The wrapper exposes the wrapped Skill's :class:`SkillManifest`
    verbatim so the :class:`SkillRegistry` builds its
    :class:`Draft7Validator` from the genuine schema (and so
    ``policy.classify`` continues to see ``destructive=True``). The
    executor body is replaced by a counter / arg-recorder that always
    returns a trivial :meth:`SkillResult.success`.

    Substituting the body keeps the test focused on the audit ordering
    contract (Property 6 / CP9) and frees the harness from having to
    provide a fully wired :class:`EmailClient` provider, credential
    store, or SMTP allowlist that the production
    :class:`SendEmailSkill` would otherwise resolve at execute time.
    """

    def __init__(self, wrapped: Any) -> None:
        # Holding a reference to the wrapped Skill is unnecessary for
        # the property under test, but it makes a clearer traceback if
        # the wrong type is passed in. Typed as :class:`Any` rather
        # than :class:`Skill` because the built-in
        # :class:`SendEmailSkill` declares its ``manifest`` as
        # :class:`typing.Final` — mypy then refuses the ``Skill``
        # Protocol's "settable variable" contract even though the
        # structural shape is satisfied at runtime.
        if not isinstance(wrapped.manifest, SkillManifest):
            raise TypeError(
                "_RecordingSkill expects a Skill-like object with a "
                "SkillManifest 'manifest' attribute"
            )
        self._wrapped: Any = wrapped
        self.manifest: SkillManifest = wrapped.manifest
        self.execute_calls: int = 0
        self.last_args: dict[str, Any] | None = None

    async def execute(
        self, args: dict[str, Any], ctx: SkillContext
    ) -> SkillResult:
        del ctx  # unused — the property does not exercise context fields
        self.execute_calls += 1
        self.last_args = dict(args)
        # ``SkillResult.success`` returns ``error_code=None`` so the
        # registry's dispatch path follows the happy branch and the
        # ``executed`` audit row gets recorded by the policy below.
        return SkillResult.success(value={"sent": True})


# ---------------------------------------------------------------------------
# Confirmation-dialog fake
# ---------------------------------------------------------------------------


class _FakeDialog:
    """Canned :class:`ConfirmationDialog` whose reply is fixed per example.

    The Hypothesis strategy chooses ``"yes"`` or ``"no"`` so the
    property test exercises *both* sides of the audit-ordering
    invariant: the consent branch (``confirmation_requested`` →
    ``executed``) and the denial branch (``confirmation_requested`` →
    ``denied``). Both must satisfy CP9's strict id-ordering and
    matching ``(skill, args_json)`` clauses.
    """

    def __init__(self, *, response: str) -> None:
        self.response: str = response
        self.call_count: int = 0
        self.prompts: list[str] = []

    async def ask_user(self, prompt: str) -> str:
        self.call_count += 1
        self.prompts.append(prompt)
        return self.response


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------


def _make_environment() -> tuple[
    AuditLog,
    AuthorizationPolicy,
    SkillRegistry,
    _RecordingSkill,
]:
    """Build a fresh audit-log / policy / registry triple per example.

    Each Hypothesis example gets its own isolated environment because
    the property quantifies over a *single* destructive Tool_Call: state
    from a previous example must not pollute the next, or the "exactly
    two rows" assertion would creep upward over time. We use SQLite's
    ``":memory:"`` database so there is no temp-file cleanup to worry
    about and so each example pays only the (cheap) schema-creation
    cost.
    """
    audit = AuditLog(":memory:", run_id="prop6-test")
    policy = AuthorizationPolicy(
        # An empty allowlist forces every destructive call through the
        # confirmation prompt path. The allowlist-bypass arm of CP9
        # ("...is not matched by the trusted-action allowlist...") is
        # the precondition for the universal quantifier; covering the
        # bypass branch is left to the unit tests in
        # ``tests/unit/security/test_authorization.py``.
        allowlist=TrustedActionAllowlist(),
        audit=audit,
    )
    registry = SkillRegistry()
    recorder = _RecordingSkill(SendEmailSkill())
    registry.register(recorder)
    return audit, policy, registry, recorder


# ---------------------------------------------------------------------------
# Hypothesis strategy & property
# ---------------------------------------------------------------------------


# Sampling from a small literal set keeps the affirmative / negative
# parser deterministic. ``_is_affirmative("yes")`` returns True and
# ``_is_affirmative("no")`` returns False — both are unambiguous so the
# test does not need to second-guess the parser's tolerance for fuzzy
# replies (those code paths are exercised in the unit tests).
_RESPONSES: tuple[str, ...] = ("yes", "no")


@given(
    args=tool_call_arguments(SendEmailSkill()),
    confirm_response=st.sampled_from(_RESPONSES),
)
@settings(
    # Inherit ``max_examples=200`` / ``deadline=None`` from the
    # ``jarvis`` Hypothesis profile (see ``tests/conftest.py``). The
    # health-check suppression handles the small per-example fixed
    # overhead of opening a SQLite connection plus compiling the
    # SendEmailSkill schema, which Hypothesis would otherwise
    # occasionally classify as ``too_slow`` on slower CI runners.
    suppress_health_check=(HealthCheck.too_slow,),
)
def test_property_06_authorization_audit_precedes_destructive_dispatch(
    args: dict[str, Any],
    confirm_response: str,
) -> None:
    """``confirmation_requested.id < executed.id`` / ``denied.id`` for every destructive call.

    **Validates: Requirements 16.1, 16.2, 16.3, 16.5 (CP9)**
    """

    async def _run() -> None:
        audit, policy, registry, recorder = _make_environment()
        try:
            # Build a :class:`ToolCall` with a canonical JSON encoding
            # of ``args``. The audit log canonicalises both
            # ``confirmation_requested`` and ``executed`` rows the same
            # way (``json.dumps(..., sort_keys=True, separators=(",",":"))``)
            # so the ``args_json`` equality clause in CP9 holds
            # regardless of the original ``raw_arguments`` form.
            raw_arguments = json.dumps(
                args, sort_keys=True, separators=(",", ":")
            )
            call = ToolCall(
                id="prop6-call-1",
                skill_name="SendEmailSkill",
                arguments=dict(args),
                raw_arguments=raw_arguments,
            )

            # Precondition for CP9: ``SendEmailSkill`` MUST be
            # classified destructive. The hard-coded destructive
            # defaults include it (Requirement 16.1), so the assertion
            # is a regression guard rather than a true filter.
            assert (
                policy.classify(call, recorder.manifest) == DESTRUCTIVE
            ), "SendEmailSkill must be classified destructive (Req 16.1)"

            dialog: ConfirmationDialog = _FakeDialog(response=confirm_response)
            consented = await policy.confirm(call, dialog)
            # The two-element response set maps 1:1 onto the boolean
            # outcome of :meth:`AuthorizationPolicy.confirm`. Pinning
            # this here makes the post-conditions below unambiguous.
            assert consented is (confirm_response == "yes")

            if consented:
                # Drive dispatch through the registry, exactly as
                # :meth:`DialogManager.handle_turn` does. The registry
                # validates ``args`` against the SendEmailSkill schema
                # (and would short-circuit with ``schema_violation``
                # before invoking the recorder if the arguments were
                # malformed — they are not, since we generated them
                # via :func:`tool_call_arguments`).
                ctx = SkillContext(audit_log=audit, run_id=audit.run_id)
                result = await registry.dispatch(
                    "SendEmailSkill", dict(args), ctx
                )
                assert result.ok is True, (
                    f"recorder dispatch unexpectedly failed: "
                    f"{result.error_code}: {result.error_message}"
                )
                assert recorder.execute_calls == 1, (
                    "recorder must be invoked exactly once on the consent "
                    f"path; observed {recorder.execute_calls}"
                )
                # The Dialog_Manager writes the closing ``executed``
                # row after the registry returns. We replicate that
                # call here so the audit pair is complete (Property 6
                # quantifies over the *paired* rows).
                await policy.record_executed(call, outcome="ok")
                expected_outcome_kind = "executed"
            else:
                # On denial the policy itself writes the ``denied``
                # row before returning False; we do NOT dispatch the
                # Skill (Property 6 demands that the executor never
                # runs without consent — Property 7 / CP10 covers the
                # complementary "executor crashed" branch).
                assert recorder.execute_calls == 0, (
                    "recorder must NOT run on the denial path; observed "
                    f"{recorder.execute_calls} call(s)"
                )
                expected_outcome_kind = "denied"

            # ----- Audit ordering invariant (the property itself) -----
            entries = audit.entries()
            assert len(entries) == 2, (
                "expected exactly 2 audit rows per destructive call "
                f"({expected_outcome_kind}); got {[e.kind for e in entries]}"
            )

            confirm_entry, outcome_entry = entries

            # The first row is the confirmation request, written by
            # ``policy.confirm`` *before* the user is asked. This
            # ordering is the precondition that makes the strict
            # id-ordering hold even if the user's reply is delayed
            # indefinitely.
            assert confirm_entry.kind == "confirmation_requested", (
                f"first row must be confirmation_requested; got {confirm_entry.kind}"
            )

            # The second row is either ``executed`` (consent path) or
            # ``denied`` (refusal path), matching the branch we drove.
            assert outcome_entry.kind == expected_outcome_kind, (
                f"second row must be {expected_outcome_kind}; "
                f"got {outcome_entry.kind}"
            )

            # CP9: strict id ordering. The audit log uses SQLite's
            # ``INTEGER PRIMARY KEY AUTOINCREMENT`` plus an asyncio
            # lock around the synchronous insert, so ids are strictly
            # monotonic regardless of which coroutine wrote first.
            assert confirm_entry.id < outcome_entry.id, (
                "confirmation_requested.id must be strictly less than "
                f"{expected_outcome_kind}.id; got "
                f"{confirm_entry.id} >= {outcome_entry.id}"
            )

            # CP9: matching ``skill``.
            assert confirm_entry.skill == "SendEmailSkill"
            assert outcome_entry.skill == "SendEmailSkill"

            # CP9: matching ``args_json``. The audit log's
            # ``_serialize_args`` canonicalises both rows the same way
            # — sorted keys, no whitespace — so byte-equal comparison
            # is meaningful even when the arguments contain nested
            # dicts whose key order differs from the strategy's
            # generated insertion order.
            assert confirm_entry.args_json == outcome_entry.args_json, (
                "confirmation_requested and outcome rows must share "
                f"args_json; got {confirm_entry.args_json!r} vs "
                f"{outcome_entry.args_json!r}"
            )
            # Belt-and-braces sanity: the canonical form of our own
            # ``args`` matches the canonical form persisted by the
            # audit log. If this regresses, CP9's ``args_json``
            # equality clause silently degrades to "two empty strings
            # match", which would let serious bugs slip through.
            expected_canonical = json.dumps(
                args, sort_keys=True, separators=(",", ":")
            )
            assert confirm_entry.args_json == expected_canonical
        finally:
            audit.close()

    asyncio.run(_run())
