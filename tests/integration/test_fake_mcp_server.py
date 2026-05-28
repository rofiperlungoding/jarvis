"""Integration tests for :mod:`tests.fakes.fake_mcp_server`.

These tests verify that the in-process :class:`FakeMCPServer` plays the
MCP protocol straight enough for real consumers — both the upstream
:class:`mcp.ClientSession` (raw protocol) and JARVIS'
:class:`~jarvis.skills.mcp_adapter.MCPSkillAdapter` (the boundary that
wraps each tool as a :class:`Skill`).

The fixture :func:`tests.fakes.fake_mcp_server.fake_mcp_server_fixture`
provides a fresh :class:`FakeMCPServer` per test; the test body then
opens a session over it via :func:`fake_mcp_session`. The fixture is
synchronous on purpose — pytest-asyncio's async-generator fixtures
finalise in a different task than the one that entered them, which
clashes with the anyio task groups inside the MCP SDK's in-memory
transport.

Validates: Requirement 14.6
"""

from __future__ import annotations

from typing import Any

from mcp import types
import pytest
from tests.fakes.fake_mcp_server import (
    DEFAULT_TOOLS,
    FakeMCPServer,
    FakeTool,
    fake_mcp_session,
)

from jarvis.skills.base import SkillContext
from jarvis.skills.mcp_adapter import MCPProxySkill, MCPSkillAdapter

# ---------------------------------------------------------------------------
# Raw MCP protocol — exercise the wire directly through ClientSession
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tools_list_advertises_default_catalogue(
    fake_mcp_server: FakeMCPServer,
) -> None:
    """``tools/list`` returns the two default tools with name/desc/schema."""
    async with fake_mcp_session(fake_mcp_server) as (_server, session):
        result = await session.list_tools()

    advertised = {t.name: t for t in result.tools}
    assert set(advertised) == {"echo", "delete_widget"}

    echo = advertised["echo"]
    assert echo.description and "echo" in echo.description.lower()
    assert echo.inputSchema["type"] == "object"
    assert echo.inputSchema["required"] == ["text"]
    assert echo.inputSchema["properties"]["text"]["type"] == "string"

    # Destructive flag flows through the MCP ``annotations.destructiveHint``.
    delete_widget = advertised["delete_widget"]
    assert delete_widget.annotations is not None
    assert delete_widget.annotations.destructiveHint is True
    # Non-destructive tools omit the annotation block by default.
    assert echo.annotations is None or echo.annotations.destructiveHint is None


@pytest.mark.asyncio
async def test_tools_call_echo_returns_structured_payload(
    fake_mcp_server: FakeMCPServer,
) -> None:
    """``tools/call echo`` round-trips arguments and records the call."""
    async with fake_mcp_session(fake_mcp_server) as (server, session):
        result = await session.call_tool("echo", arguments={"text": "hello, sir"})

    assert result.isError is False
    # Structured content is the canonical machine-readable representation.
    assert result.structuredContent == {"echo": "hello, sir"}
    # The SDK auto-serialises the dict into one TextContent block.
    text_block_texts = [
        c.text for c in result.content if isinstance(c, types.TextContent)
    ]
    assert text_block_texts, "expected at least one TextContent block"
    assert any("hello, sir" in text for text in text_block_texts)

    # The fake recorded the call with exactly the supplied arguments.
    assert len(server.calls) == 1
    assert server.calls[0].tool_name == "echo"
    assert server.calls[0].arguments == {"text": "hello, sir"}


@pytest.mark.asyncio
async def test_tools_call_delete_widget_returns_destructive_payload(
    fake_mcp_server: FakeMCPServer,
) -> None:
    """The destructive tool reports the deleted widget id."""
    async with fake_mcp_session(fake_mcp_server) as (server, session):
        result = await session.call_tool(
            "delete_widget", arguments={"widget_id": "widget-42"}
        )

    assert result.isError is False
    assert result.structuredContent == {"deleted": True, "widget_id": "widget-42"}
    assert len(server.calls) == 1
    assert server.calls[0].tool_name == "delete_widget"
    assert server.calls[0].arguments == {"widget_id": "widget-42"}


@pytest.mark.asyncio
async def test_tools_call_with_invalid_arguments_returns_isError() -> None:
    """The MCP server validates arguments against ``inputSchema``.

    ``echo`` requires a ``text`` string — omitting it must produce an
    ``isError=True`` :class:`CallToolResult` rather than crashing the
    server. This anchors the contract :class:`MCPSkillAdapter` relies on
    when translating to ``SkillResult.error("internal_error", ...)``.
    """
    async with fake_mcp_session() as (server, session):
        result = await session.call_tool("echo", arguments={})

    assert result.isError is True
    assert server.calls == [], "schema-rejected calls should not reach the handler"


# ---------------------------------------------------------------------------
# Custom catalogues
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_tool_catalogue_is_advertised() -> None:
    """A test can override the catalogue with a single tool."""

    async def _ping(_args: dict[str, Any]) -> dict[str, Any]:
        return {"pong": True}

    custom = FakeTool(
        name="ping",
        description="ping/pong test tool",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_ping,
    )

    async with fake_mcp_session(tools=[custom]) as (server, session):
        listed = await session.list_tools()
        assert [t.name for t in listed.tools] == ["ping"]

        result = await session.call_tool("ping", arguments={})
        assert result.isError is False
        assert result.structuredContent == {"pong": True}
        assert server.tool_names == ("ping",)


@pytest.mark.asyncio
async def test_passing_server_and_tools_together_is_rejected() -> None:
    """Mutually-exclusive arguments raise :class:`ValueError`."""
    server = FakeMCPServer()
    with pytest.raises(ValueError, match=r"server.+tools"):
        async with fake_mcp_session(server, tools=DEFAULT_TOOLS):
            pass  # pragma: no cover - context never entered


# ---------------------------------------------------------------------------
# MCPSkillAdapter — the JARVIS-side consumer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_skill_adapter_loads_skills_from_fake_server(
    fake_mcp_server: FakeMCPServer,
) -> None:
    """The adapter wraps every tool as a :class:`Skill` with a Mistral schema."""
    async with fake_mcp_session(fake_mcp_server) as (_server, session):
        adapter = MCPSkillAdapter(session, server_name="fake")
        skills = await adapter.load_skills()

        # Two tools advertised → two synthetic Skills.
        assert len(skills) == 2
        names = {s.manifest.name for s in skills}
        assert names == {"mcp.fake.echo", "mcp.fake.delete_widget"}
        for skill in skills:
            assert isinstance(skill, MCPProxySkill)
            assert skill.manifest.source == "mcp"
            assert skill.manifest.json_schema["type"] == "object"


@pytest.mark.asyncio
async def test_mcp_skill_adapter_proxies_execute_through_call_tool() -> None:
    """``Skill.execute`` invokes the remote tool and surfaces structured data."""
    async with fake_mcp_session() as (server, session):
        adapter = MCPSkillAdapter(session)
        skills_by_name = {s.manifest.name: s for s in await adapter.load_skills()}

        echo_skill = skills_by_name["echo"]
        result = await echo_skill.execute({"text": "ping"}, SkillContext())

        assert result.ok is True
        assert result.value is not None
        assert result.value.get("structured_content") == {"echo": "ping"}
        assert result.value.get("tool") == "echo"

        # The fake recorded the call as routed through MCP.
        assert len(server.calls) == 1
        assert server.calls[0].tool_name == "echo"
        assert server.calls[0].arguments == {"text": "ping"}


@pytest.mark.asyncio
async def test_mcp_skill_adapter_translates_tool_error_to_internal_error() -> None:
    """A handler that raises is reported back as ``internal_error``.

    The MCP server-side decorator catches handler exceptions and returns
    ``isError=True``; the adapter maps that onto
    :class:`SkillResult` with ``error_code="internal_error"`` per the
    documented closed taxonomy.
    """

    async def _boom(_args: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("widget service offline")

    boom = FakeTool(
        name="boom",
        description="raises every call",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_boom,
    )

    async with fake_mcp_session(tools=[boom]) as (_server, session):
        adapter = MCPSkillAdapter(session)
        (skill,) = await adapter.load_skills()

        result = await skill.execute({}, SkillContext())

        assert result.ok is False
        assert result.error_code == "internal_error"
        assert result.error_message is not None
        assert "widget service offline" in result.error_message


@pytest.mark.asyncio
async def test_reset_calls_clears_history() -> None:
    """``FakeMCPServer.reset_calls`` empties the captured call list."""
    async with fake_mcp_session() as (server, session):
        await session.call_tool("echo", arguments={"text": "one"})
        await session.call_tool("echo", arguments={"text": "two"})
        assert len(server.calls) == 2

        server.reset_calls()
        assert server.calls == []

        await session.call_tool("echo", arguments={"text": "three"})
        assert [c.arguments for c in server.calls] == [{"text": "three"}]


def test_fixture_returns_fresh_server_with_default_tools(
    fake_mcp_server: FakeMCPServer,
) -> None:
    """The fixture surfaces a clean ``FakeMCPServer`` with the default catalogue."""
    assert isinstance(fake_mcp_server, FakeMCPServer)
    assert fake_mcp_server.tool_names == ("echo", "delete_widget")
    assert fake_mcp_server.calls == []
    assert fake_mcp_server.destructive_tools == frozenset({"delete_widget"})
