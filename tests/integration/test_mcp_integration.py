"""End-to-end MCP integration test.

Drives a real :class:`~jarvis.skills.registry.SkillRegistry` against the
in-process :class:`~tests.fakes.fake_mcp_server.FakeMCPServer` from task
22.2 to prove the full MCP surface area works as advertised:

* The :class:`~jarvis.skills.mcp_adapter.MCPSkillAdapter` projects each
  remote tool advertised by the MCP server into a synthetic
  :class:`~jarvis.skills.base.Skill` (Requirement 14.6, see
  ``design.md §Skill_Registry``).
* :meth:`SkillRegistry.dispatch` validates arguments, awaits the
  proxied ``ClientSession.call_tool``, and translates the
  :class:`mcp.types.CallToolResult` back into a
  :class:`~jarvis.skills.base.SkillResult`.
* The :class:`~jarvis.security.authorization.AuthorizationPolicy`
  bookends a destructive Tool_Call with ``confirmation_requested`` and
  ``executed`` audit entries that match by ``skill`` and ``args_json``
  in strict id order — the audit ordering invariant CP9 relies on.

The test deliberately wires together the **real** ``SkillRegistry``,
``MCPSkillAdapter``, ``AuthorizationPolicy``, and ``AuditLog`` so a
regression in any of them surfaces here. The only fake is the in-process
MCP server (which is a real ``mcp.server.Server`` over the SDK's
in-memory transport) and a one-line :class:`ConfirmationDialog` stub.

Validates: Requirement 14.6
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.fakes.fake_mcp_server import FakeMCPServer, fake_mcp_session

from jarvis.llm.base import ToolCall
from jarvis.security.audit_log import AuditLog
from jarvis.security.authorization import (
    DESTRUCTIVE,
    AuthorizationPolicy,
    TrustedActionAllowlist,
)
from jarvis.skills.base import SkillContext
from jarvis.skills.mcp_adapter import MCPSkillAdapter
from jarvis.skills.registry import SkillRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CannedDialog:
    """Minimal :class:`ConfirmationDialog` stub.

    Returns a single canned reply and records every prompt the
    Authorization_Policy speaks so tests can assert the user was
    actually asked before the destructive dispatch.
    """

    def __init__(self, response: str = "yes, proceed") -> None:
        self.response = response
        self.prompts: list[str] = []

    async def ask_user(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


# ---------------------------------------------------------------------------
# End-to-end destructive flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_destructive_tool_call_end_to_end(
    fake_mcp_server: FakeMCPServer, tmp_path: Path
) -> None:
    """Dispatch a destructive MCP Tool_Call through the full pipeline.

    The destructive flow is the one that exercises every audit hook:
    :meth:`AuthorizationPolicy.confirm` writes
    ``confirmation_requested`` *before* the user is prompted, the
    SkillRegistry forwards the call across the MCP transport, and
    :meth:`AuthorizationPolicy.record_executed` closes the audit pair
    with ``executed`` *after* dispatch returns. The strict id ordering
    of those two rows is the runtime invariant CP9 verifies.
    """
    audit = AuditLog(tmp_path / "audit.sqlite", run_id="mcp-integration-test")
    try:
        async with fake_mcp_session(fake_mcp_server) as (server, session):
            # 1. Wire MCPSkillAdapter into a real SkillRegistry. We
            #    deliberately leave ``server_name=None`` so the registered
            #    Skill names match the bare MCP tool names — the
            #    AuthorizationPolicy's hard-coded destructive list keys
            #    on ``ToolCall.skill_name`` and is simpler to reason
            #    about without an ``mcp.<server>.`` prefix.
            adapter = MCPSkillAdapter(session)
            registry = SkillRegistry()
            for skill in await adapter.load_skills():
                registry.register(skill)

            # The adapter registered every tool the fake advertises.
            assert "delete_widget" in registry, registry.names
            assert "echo" in registry, registry.names

            # 2. Build the destructive Tool_Call that the LLM "would
            #    have" emitted. ``raw_arguments`` is a canonical JSON
            #    encoding so we can compare it byte-for-byte with the
            #    audit log's stored ``args_json`` later.
            tool_call = ToolCall(
                id="call-001",
                skill_name="delete_widget",
                arguments={"widget_id": "widget-42"},
                raw_arguments='{"widget_id":"widget-42"}',
            )

            # 3. Construct the AuthorizationPolicy. The hard-coded
            #    destructive list pins ``delete_widget`` as a
            #    Destructive_Action even though MCP manifests are
            #    always ``destructive=False`` — a tightening the policy
            #    explicitly supports per Requirement 16.1.
            policy = AuthorizationPolicy(
                allowlist=TrustedActionAllowlist(),
                audit=audit,
                hard_coded_destructive_skills=("delete_widget",),
            )

            mcp_skill = registry.get(tool_call.skill_name)
            assert mcp_skill is not None
            assert policy.classify(tool_call, mcp_skill.manifest) == DESTRUCTIVE

            # 4. Confirm. The dialog returns an affirmative response;
            #    the policy writes ``confirmation_requested`` *before*
            #    awaiting ``ask_user`` so the audit row is on disk no
            #    matter how long the user takes to reply.
            dialog = _CannedDialog("yes, proceed")
            confirmed = await policy.confirm(tool_call, dialog)
            assert confirmed is True
            assert dialog.prompts, (
                "AuthorizationPolicy should have spoken a summary before dispatch"
            )

            # 5. Dispatch through the SkillRegistry. This is where the
            #    proxy Skill calls ``ClientSession.call_tool`` against
            #    the in-process MCP server. The registry catches any
            #    PolicyViolation and translates it to ``access_denied``;
            #    on the happy path it returns the executor's result
            #    untouched (with ``duration_ms`` back-filled).
            ctx = SkillContext(audit_log=audit, run_id="mcp-integration-test")
            result = await registry.dispatch(
                tool_call.skill_name, tool_call.arguments, ctx
            )

            # 6. Close the audit pair. The Dialog_Manager owns this call
            #    in production; we drive it directly so the integration
            #    test fully exercises the audit-write path.
            await policy.record_executed(tool_call, outcome="ok")

        # ---- Assertion 1: the MCP server received the call ----
        # The fake records every ``tools/call`` request it dispatches
        # to a handler. We assert the adapter forwarded the exact
        # arguments without mutation.
        assert len(server.calls) == 1
        assert server.calls[0].tool_name == "delete_widget"
        assert server.calls[0].arguments == {"widget_id": "widget-42"}

        # ---- Assertion 2: the result was propagated correctly ----
        assert result.ok is True
        assert result.error_code is None
        assert result.error_message is None
        assert result.value is not None
        # ``MCPProxySkill._translate_call_result`` packages the MCP
        # ``structuredContent`` dict under ``value["structured_content"]``
        # and tags the result with the originating tool name.
        assert result.value["tool"] == "delete_widget"
        assert result.value["structured_content"] == {
            "deleted": True,
            "widget_id": "widget-42",
        }
        # The ``CallToolResult`` also contained an auto-serialised
        # JSON ``TextContent`` block with the same payload; the proxy
        # surfaces that under ``value["text"]``.
        text_payload = result.value.get("text")
        assert isinstance(text_payload, str) and "widget-42" in text_payload
        # The registry back-fills wall-clock duration on a happy path.
        assert result.duration_ms >= 0

        # ---- Assertion 3: the audit log carries an executed entry ----
        entries = audit.entries()
        kinds = [e.kind for e in entries]
        # The happy path writes exactly two rows — confirmation +
        # executed — and no policy violations.
        assert "policy_violation" not in kinds, kinds
        assert "confirmation_requested" in kinds, kinds
        assert "executed" in kinds, kinds

        confirm_entry = next(
            e for e in entries if e.kind == "confirmation_requested"
        )
        executed_entry = next(e for e in entries if e.kind == "executed")

        # Skill name and args_json match across the pair (CP9 invariant)
        assert confirm_entry.skill == "delete_widget"
        assert executed_entry.skill == "delete_widget"
        assert confirm_entry.args_json == executed_entry.args_json
        # Strict id ordering: the confirmation row predates the
        # executed row even though both wrote in the same coroutine.
        assert confirm_entry.id < executed_entry.id
        assert executed_entry.outcome == "ok"
        # Both entries carry the run id we configured on the audit log.
        assert confirm_entry.run_id == "mcp-integration-test"
        assert executed_entry.run_id == "mcp-integration-test"
    finally:
        audit.close()


# ---------------------------------------------------------------------------
# Safe-path sanity check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_safe_tool_dispatch_propagates_result(
    fake_mcp_server: FakeMCPServer, tmp_path: Path
) -> None:
    """A non-destructive MCP echo tool round-trips structured content.

    No Authorization_Policy is involved on this path, so the registry's
    dispatch is the entire story. The test pins the contract that:

    * the MCP server *received* the call,
    * the SkillResult propagates ``structured_content`` and ``text``, and
    * ``SkillRegistry.dispatch`` does NOT write audit entries on a
      clean execution — only ``policy_violation`` rows appear there
      directly. ``confirmation_requested`` / ``executed`` rows are the
      Authorization_Policy's responsibility.
    """
    audit = AuditLog(tmp_path / "audit.sqlite", run_id="mcp-safe-test")
    try:
        async with fake_mcp_session(fake_mcp_server) as (server, session):
            registry = SkillRegistry()
            adapter = MCPSkillAdapter(session)
            for skill in await adapter.load_skills():
                registry.register(skill)

            ctx = SkillContext(audit_log=audit, run_id="mcp-safe-test")
            result = await registry.dispatch(
                "echo", {"text": "hello, sir"}, ctx
            )

        # The MCP server saw exactly one call with the expected payload.
        assert len(server.calls) == 1
        assert server.calls[0].tool_name == "echo"
        assert server.calls[0].arguments == {"text": "hello, sir"}

        # Result propagated correctly.
        assert result.ok is True
        assert result.value is not None
        assert result.value["tool"] == "echo"
        assert result.value["structured_content"] == {"echo": "hello, sir"}

        # Non-destructive dispatch leaves the audit log untouched.
        assert audit.count() == 0
    finally:
        audit.close()
