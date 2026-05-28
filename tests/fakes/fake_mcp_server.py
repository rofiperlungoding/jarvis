"""In-process fake MCP server for integration tests.

This module hosts :class:`FakeMCPServer`, a minimal but **real** Model
Context Protocol server built on the upstream :mod:`mcp` SDK. It
advertises one or two test tools and is wired to the
:class:`mcp.ClientSession` over the SDK's in-memory transport pair
(:func:`mcp.shared.memory.create_connected_server_and_client_session`),
so tests can exercise :class:`~jarvis.skills.mcp_adapter.MCPSkillAdapter`
end-to-end without spinning up a subprocess.

What lives here
---------------

* :class:`FakeMCPServer` — wraps an :class:`mcp.server.Server` instance,
  registers ``tools/list`` and ``tools/call`` handlers for the tools
  declared in :data:`DEFAULT_TOOLS`, and records every call it receives
  so tests can assert against side effects.
* :func:`fake_mcp_session` — an ``async`` context manager yielding a
  connected :class:`mcp.ClientSession` that talks to the fake server in
  the same event loop. The :class:`FakeMCPServer` reference passed in (or
  constructed by default) is reachable via the helper's return value so
  call-recording assertions are straightforward.
* :func:`fake_mcp_server_fixture` — a pytest fixture (registered through
  :mod:`pytest`'s plugin discovery as ``fake_mcp_server``) that returns a
  fresh :class:`FakeMCPServer` per test. Tests that need a connected
  :class:`mcp.ClientSession` should pair the fixture-supplied server
  with :func:`fake_mcp_session` inside the test body so the entire
  anyio task group enters and exits in a single task — avoiding the
  cross-task cancel-scope errors that surface when async-generator
  fixtures yield across pytest-asyncio's setup/teardown boundary.

Why the in-memory transport
---------------------------

The MCP SDK ships with a stdio client/server pair, but those require a
subprocess and on Windows the asyncio + ``anyio`` plumbing is fragile in
a test event loop. ``mcp.shared.memory.create_connected_server_and_client_session``
runs both ends in the same process under a shared :mod:`anyio` task
group, gives us a fully initialised :class:`ClientSession`, and tears
everything down deterministically when the context exits — which is
exactly what an integration test wants.

Tool catalogue
--------------

The fake advertises two tools by default:

* ``echo`` — returns the supplied ``text`` argument back as
  ``TextContent`` and as structured ``{"echo": text}`` content. Mistral
  schema-compatible (object root with one required ``text`` string).
* ``delete_widget`` — simulates a destructive operation. Returns a
  structured payload reporting the deleted ``widget_id``. The
  Authorization_Policy in JARVIS treats MCP tools as non-destructive by
  default, but the *FakeMCPServer*-level
  :attr:`FakeMCPServer.destructive_tools` set is exposed so tests that
  layer their own destructive-classification logic on top of the adapter
  (e.g., ``Authorization_Policy``) can opt this tool in.

Tests can override the catalogue by passing a ``tools=`` argument; see
the dataclass :class:`FakeTool` for the supported descriptor shape.

Validates: Requirement 14.6
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest

try:
    from mcp import ClientSession, types
    from mcp.server.lowlevel import Server
    from mcp.shared.memory import create_connected_server_and_client_session
except ImportError as exc:  # pragma: no cover - depends on env
    raise ImportError(
        "FakeMCPServer requires the 'mcp' package to be installed; "
        "install it via the 'jarvis' project's runtime dependencies."
    ) from exc


__all__ = [
    "DEFAULT_TOOLS",
    "FakeMCPServer",
    "FakeTool",
    "ToolCallRecord",
    "fake_mcp_server_fixture",
    "fake_mcp_session",
]


# ---------------------------------------------------------------------------
# Tool descriptors
# ---------------------------------------------------------------------------


# Handler signature: ``async def handler(args: dict[str, Any]) -> dict[str,
# Any] | tuple[Sequence[ContentBlock], dict[str, Any] | None]``. We accept
# the dict-only form because that's the most ergonomic for tests; the MCP
# SDK auto-wraps it in a ``TextContent`` block plus structured payload.
ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class FakeTool:
    """Descriptor for a tool advertised by :class:`FakeMCPServer`.

    Attributes
    ----------
    name:
        MCP tool name. Surfaces verbatim through ``tools/list``.
    description:
        Human-readable description that the LLM sees.
    input_schema:
        Mistral-compatible JSON Schema for the ``arguments`` payload.
        MUST have ``type == "object"`` for the adapter / Mistral path.
    handler:
        Async callable invoked on ``tools/call``. The returned dict is
        forwarded as structured content; the SDK additionally serialises
        it into a JSON ``TextContent`` block.
    destructive:
        Tag for the test harness to flag destructive tools. **Not** part
        of the MCP wire protocol — the upstream SDK exposes
        ``destructiveHint`` via :class:`types.ToolAnnotations` instead,
        which we set whenever this flag is ``True`` so a downstream
        Authorization_Policy can read it back.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    destructive: bool = False


@dataclass(frozen=True)
class ToolCallRecord:
    """A single ``tools/call`` event captured by :class:`FakeMCPServer`."""

    tool_name: str
    arguments: dict[str, Any]


# ---------------------------------------------------------------------------
# Default tool implementations
# ---------------------------------------------------------------------------


async def _echo_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Return ``{"echo": args["text"]}`` — used by the default ``echo`` tool."""
    text = args.get("text", "")
    return {"echo": text}


async def _delete_widget_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Pretend to delete a widget; returns ``{"deleted": True, ...}``.

    The handler is destructive in spirit only — the fake never touches
    real state. Tests can swap in a custom handler if they need richer
    side-effect simulation.
    """
    widget_id = args["widget_id"]
    return {"deleted": True, "widget_id": widget_id}


_ECHO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "description": "Text to echo back to the caller.",
        },
    },
    "required": ["text"],
    "additionalProperties": False,
}


_DELETE_WIDGET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "widget_id": {
            "type": "string",
            "description": "Identifier of the widget to delete.",
        },
    },
    "required": ["widget_id"],
    "additionalProperties": False,
}


DEFAULT_TOOLS: tuple[FakeTool, ...] = (
    FakeTool(
        name="echo",
        description="Echoes the supplied text back to the caller.",
        input_schema=_ECHO_SCHEMA,
        handler=_echo_handler,
        destructive=False,
    ),
    FakeTool(
        name="delete_widget",
        description="Permanently delete a widget by id (test-only stub).",
        input_schema=_DELETE_WIDGET_SCHEMA,
        handler=_delete_widget_handler,
        destructive=True,
    ),
)


# ---------------------------------------------------------------------------
# FakeMCPServer
# ---------------------------------------------------------------------------


@dataclass
class FakeMCPServer:
    """Minimal in-process MCP server for integration tests.

    Construct with no arguments to get the two-tool default catalogue, or
    pass ``tools=`` to publish a custom set. The server is wired to the
    :mod:`mcp` SDK's in-memory transport pair via
    :func:`fake_mcp_session`, which yields a fully initialised
    :class:`mcp.ClientSession` to the test.

    Attributes
    ----------
    name:
        MCP server name reported on ``initialize``. Defaults to
        ``"jarvis-fake-mcp-server"``.
    version:
        MCP server version reported on ``initialize``.
    tools:
        Sequence of :class:`FakeTool` descriptors advertised through
        ``tools/list`` and dispatched on ``tools/call``.
    calls:
        Append-only list of :class:`ToolCallRecord` entries — one per
        ``tools/call`` request the server handled. Tests assert against
        this list to verify the adapter routed arguments correctly.
    """

    name: str = "jarvis-fake-mcp-server"
    version: str = "0.1.0"
    tools: Sequence[FakeTool] = field(default_factory=lambda: DEFAULT_TOOLS)
    calls: list[ToolCallRecord] = field(default_factory=list)

    # ------------------------------------------------------------------ public

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Return the names of the advertised tools, in registration order."""
        return tuple(t.name for t in self.tools)

    @property
    def destructive_tools(self) -> frozenset[str]:
        """Return the names of the tools tagged as destructive."""
        return frozenset(t.name for t in self.tools if t.destructive)

    def find_tool(self, name: str) -> FakeTool:
        """Look up a registered :class:`FakeTool` by name.

        Raises:
            KeyError: if no tool with ``name`` was registered.
        """
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise KeyError(name)

    def reset_calls(self) -> None:
        """Drop the captured call history.

        Useful when a single test exercises the same fixture multiple
        times and wants to make per-phase assertions.
        """
        self.calls.clear()

    def build_server(self) -> Server[Any, Any]:
        """Create an :class:`mcp.server.Server` that surfaces the registered tools.

        Each call returns a *fresh* server so a single :class:`FakeMCPServer`
        can be re-used across consecutive sessions without leaking
        decorator state. The server is what
        :func:`mcp.shared.memory.create_connected_server_and_client_session`
        wraps.
        """
        server: Server[Any, Any] = Server(name=self.name, version=self.version)

        @server.list_tools()  # type: ignore[untyped-decorator]
        async def _handle_list_tools() -> list[types.Tool]:
            return [_to_mcp_tool(t) for t in self.tools]

        @server.call_tool()  # type: ignore[untyped-decorator]
        async def _handle_call_tool(
            tool_name: str, arguments: dict[str, Any]
        ) -> dict[str, Any]:
            self.calls.append(
                ToolCallRecord(tool_name=tool_name, arguments=dict(arguments))
            )
            tool = self.find_tool(tool_name)
            return await tool.handler(arguments)

        return server


# ---------------------------------------------------------------------------
# Wire-helpers
# ---------------------------------------------------------------------------


def _to_mcp_tool(tool: FakeTool) -> types.Tool:
    """Translate a :class:`FakeTool` into an :class:`mcp.types.Tool` descriptor."""
    annotations: types.ToolAnnotations | None = None
    if tool.destructive:
        annotations = types.ToolAnnotations(destructiveHint=True)
    return types.Tool(
        name=tool.name,
        description=tool.description,
        inputSchema=dict(tool.input_schema),
        annotations=annotations,
    )


# ---------------------------------------------------------------------------
# Async context manager — pair an MCP server with a connected client
# ---------------------------------------------------------------------------


@asynccontextmanager
async def fake_mcp_session(
    server: FakeMCPServer | None = None,
    *,
    tools: Sequence[FakeTool] | None = None,
) -> AsyncIterator[tuple[FakeMCPServer, ClientSession]]:
    """Yield a connected :class:`mcp.ClientSession` talking to the fake server.

    The helper is the canonical way to drive :class:`FakeMCPServer` from
    a test: instantiate (or pass) one, ``async with`` for the lifetime of
    the test, and use the yielded :class:`ClientSession` to exercise
    ``list_tools`` / ``call_tool`` directly **or** to construct an
    :class:`~jarvis.skills.mcp_adapter.MCPSkillAdapter` around it.

    Parameters
    ----------
    server:
        Optional pre-built :class:`FakeMCPServer`. When omitted a fresh
        server is constructed with the default tool set (or with
        ``tools`` if supplied). Passing a server lets tests inspect
        ``server.calls`` after the context exits.
    tools:
        Optional override for the tool catalogue when ``server`` is
        ``None``. Ignored when ``server`` is supplied because the server
        already owns its catalogue.

    Yields
    ------
    tuple[FakeMCPServer, ClientSession]
        The server (so call assertions are available even when one was
        not pre-built) paired with the live client session.
    """
    if server is None:
        server = FakeMCPServer() if tools is None else FakeMCPServer(tools=tuple(tools))
    elif tools is not None:
        raise ValueError(
            "pass either a pre-built `server` or a `tools` override, not both"
        )

    mcp_server = server.build_server()
    async with create_connected_server_and_client_session(mcp_server) as session:
        yield server, session


# ---------------------------------------------------------------------------
# pytest fixture
# ---------------------------------------------------------------------------


@pytest.fixture(name="fake_mcp_server")
def fake_mcp_server_fixture() -> FakeMCPServer:
    """Pytest fixture returning a fresh :class:`FakeMCPServer` per test.

    The fixture is intentionally **synchronous**: pytest-asyncio's
    async-generator fixtures finalise in a different task than the one
    that started them, which conflicts with the anyio task groups that
    :func:`mcp.shared.memory.create_connected_server_and_client_session`
    sets up internally. Tests that need a live
    :class:`mcp.ClientSession` must therefore drive
    :func:`fake_mcp_session` themselves inside the test body, e.g.::

        @pytest.mark.asyncio
        async def test_x(fake_mcp_server: FakeMCPServer) -> None:
            async with fake_mcp_session(fake_mcp_server) as (server, session):
                ...

    Returning a fresh server per test guarantees ``calls`` and any
    user-supplied tool state start empty.
    """
    return FakeMCPServer()
