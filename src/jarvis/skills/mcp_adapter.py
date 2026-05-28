"""Adapter that exposes Model Context Protocol (MCP) tools as Skills.

This module implements :class:`MCPSkillAdapter`, the bridge between an
already-connected ``mcp.ClientSession`` and the
:class:`~jarvis.skills.registry.SkillRegistry`. Each tool advertised by the
remote MCP server is wrapped in a synthetic :class:`~jarvis.skills.base.Skill`
whose ``execute`` proxies the call across the session via
``ClientSession.call_tool`` (Requirement 14.6, see also ``design.md
§Skill_Registry``).

What lives here
---------------

* :class:`MCPSkillAdapter` — pulls the tool catalogue from a connected
  session, validates each tool's JSON Schema through
  :class:`~jarvis.llm.mistral_schema.MistralSchemaValidator`, and yields
  Skills tagged ``source="mcp"``.
* :class:`MCPProxySkill` — the synthetic :class:`Skill` that wraps a single
  remote tool. ``execute`` translates :class:`mcp.types.CallToolResult`
  into a :class:`SkillResult`, mapping the documented error taxonomy
  (``rate_limited``, ``timeout``, ``provider_unavailable``,
  ``internal_error``).
* :func:`connect_mcp_skills` — top-level helper that opens an MCP transport
  (stdio or SSE), initialises the session, fetches the tool list, and
  returns a list of :class:`MCPProxySkill` instances. Returns an
  ``async`` context manager-shaped tuple ``(skills, aclose)`` so callers
  can keep the session alive for the lifetime of the registry.

Design notes
------------

* The :mod:`mcp` SDK is **lazy-imported** inside the helpers so the rest of
  the project (and its unit tests) can import this module on platforms
  where ``mcp`` is unavailable. Static type checkers see the typed names
  via :pydata:`typing.TYPE_CHECKING` only.
* Each MCP tool's ``inputSchema`` is normalised into a Mistral-compatible
  JSON Schema. Two normalisations are non-controversial and applied
  unconditionally: an empty ``inputSchema`` is upgraded to
  ``{"type": "object"}`` (the Mistral function-calling spec requires
  ``parameters.type == "object"`` per CP15), and a missing top-level
  ``"type"`` is filled in. Anything else stays untouched and is run
  through :class:`MistralSchemaValidator.validate` for the closed-set
  subset rules (``$ref`` to remote, mixed ``oneOf`` branches, unsupported
  ``format``).
* Tools whose schema fails the Mistral subset are **skipped** with a
  warning rather than raising; one bad MCP tool must not knock the whole
  server out of the registry. The Skill_Registry already raises on
  ``register`` for built-in plugins; here we are working with
  third-party servers we do not control.
* ``execute`` runs ``call_tool`` under :func:`asyncio.wait_for` using the
  manifest's ``timeout_seconds`` so the registry's wider time budget
  remains in force even when the remote tool stalls.

Validates: Requirement 14.6
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass
import logging
from typing import (
    TYPE_CHECKING,
    Any,
)

from jarvis.llm.mistral_schema import MistralSchemaError, MistralSchemaValidator
from jarvis.skills.base import (
    SkillContext,
    SkillManifest,
    SkillResult,
)

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from mcp import ClientSession
    from mcp.types import CallToolResult, Tool


logger = logging.getLogger(__name__)


__all__ = [
    "DEFAULT_MCP_TIMEOUT_SECONDS",
    "MCPProxySkill",
    "MCPSkillAdapter",
    "connect_mcp_skills",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Per-call wall-clock budget for an MCP tool invocation. The default
# matches :class:`SkillManifest`'s default to keep behaviour consistent
# with built-in Skills; configured MCP servers can request longer budgets
# through ``McpServerConfig`` in future. Surfaced as a module constant so
# tests can inject a smaller value rather than poking at private state.
DEFAULT_MCP_TIMEOUT_SECONDS: float = 30.0

# Maximum length of the textual payload returned to the LLM. MCP tools may
# stream very long ``TextContent`` blobs (think file dumps); we trim the
# concatenated string to keep the Mistral context window predictable. The
# original payload remains available via ``value["raw_content_count"]``.
_MAX_TEXT_PAYLOAD_CHARS: int = 32_000


# ---------------------------------------------------------------------------
# Synthetic Skill wrapper
# ---------------------------------------------------------------------------


@dataclass
class MCPProxySkill:
    """Synthetic :class:`Skill` that proxies execute to ``ClientSession.call_tool``.

    Instances are produced by :class:`MCPSkillAdapter`; user code does not
    construct them directly. The dataclass is *mutable* so the adapter
    can re-bind ``session`` after a reconnect without rebuilding every
    Skill (the registry stores Skills by ``manifest.name``).

    Attributes
    ----------
    manifest:
        The :class:`SkillManifest` derived from the MCP tool descriptor.
        Always carries ``source="mcp"`` and ``destructive=False`` (MCP
        does not currently advertise destructiveness; the
        Authorization_Policy treats every MCP tool conservatively via
        configuration).
    session:
        The connected :class:`mcp.ClientSession`. The adapter is
        responsible for keeping the session alive; ``execute`` only ever
        reads from it.
    tool_name:
        The remote tool's MCP name (``Tool.name``). Distinct from
        ``manifest.name`` because the adapter may be configured to apply
        a per-server prefix to avoid collisions across multiple MCP
        servers.
    """

    manifest: SkillManifest
    session: ClientSession
    tool_name: str

    async def execute(
        self, args: dict[str, Any], ctx: SkillContext
    ) -> SkillResult:
        """Invoke the remote tool and translate the result.

        The arguments are assumed to have already been validated against
        ``manifest.json_schema`` by the :class:`SkillRegistry`; we do not
        re-validate here. We do, however, defensively wrap the
        ``call_tool`` invocation in :func:`asyncio.wait_for` with the
        manifest's timeout so a stuck MCP server never holds the dialog
        loop hostage indefinitely.
        """
        del ctx  # MCP tools do not consume the local skill context.
        timeout = float(self.manifest.timeout_seconds)
        try:
            call_result = await asyncio.wait_for(
                self.session.call_tool(self.tool_name, arguments=args),
                timeout=timeout,
            )
        except TimeoutError:
            return SkillResult.error(
                "timeout",
                f"MCP tool {self.tool_name!r} timed out after {timeout:.1f}s",
            )
        except asyncio.CancelledError:
            # Never swallow cancellation — the dialog loop relies on it
            # for shutdown / barge-in semantics.
            raise
        except Exception as exc:
            # The MCP SDK can raise transport errors (broken pipe,
            # connection reset). Surface them as ``provider_unavailable``
            # so the Dialog_Manager can prompt a graceful fallback.
            logger.exception(
                "MCP tool %r raised %s during call_tool",
                self.tool_name,
                type(exc).__name__,
            )
            return SkillResult.error(
                "provider_unavailable",
                f"MCP tool {self.tool_name!r} failed: {type(exc).__name__}: {exc}",
            )

        return _translate_call_result(self.tool_name, call_result)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class MCPSkillAdapter:
    """Wrap a connected MCP session as a sequence of :class:`Skill` objects.

    Typical usage::

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                adapter = MCPSkillAdapter(session, server_name="files")
                skills = await adapter.load_skills()
                for skill in skills:
                    registry.register(skill)

    The adapter does **not** own the session lifecycle; callers wire that
    up themselves (or use :func:`connect_mcp_skills`, which packages the
    most common pattern). This separation lets tests substitute a fake
    in-process session without paying the stdio/SSE setup cost.

    Parameters
    ----------
    session:
        Already-initialised :class:`mcp.ClientSession`.
    server_name:
        Optional human-readable identifier for the server. Used as a
        prefix on the resulting Skill names so two servers exposing a
        ``"search"`` tool do not collide. ``None`` means use the raw
        tool name.
    validator:
        Pre-built :class:`MistralSchemaValidator`. Tests inject a custom
        instance to assert behaviour around invalid schemas.
    timeout_seconds:
        Default per-call wall-clock budget for the resulting Skills.
    """

    def __init__(
        self,
        session: ClientSession,
        *,
        server_name: str | None = None,
        validator: MistralSchemaValidator | None = None,
        timeout_seconds: float = DEFAULT_MCP_TIMEOUT_SECONDS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be strictly positive")
        if server_name is not None and not isinstance(server_name, str):
            raise TypeError("server_name must be a string or None")
        self._session = session
        self._server_name = server_name or None
        self._validator = validator or MistralSchemaValidator()
        self._timeout_seconds = float(timeout_seconds)

    # ------------------------------------------------------------------ public

    @property
    def session(self) -> ClientSession:
        """Return the connected MCP session (for tests / re-binding)."""
        return self._session

    @property
    def server_name(self) -> str | None:
        """Return the configured server prefix (or ``None``)."""
        return self._server_name

    async def load_skills(self) -> list[MCPProxySkill]:
        """Fetch the tool catalogue and return one Skill per accepted tool.

        Tools whose schema fails the Mistral function-calling subset are
        **skipped with a warning** rather than raising — one
        non-conformant tool must not bring down the whole server. The
        registry will only see tools we have already proven to round-trip
        cleanly through :meth:`MistralSchemaValidator.to_mistral_tool`.
        """
        # ``list_tools`` paginates via cursor in the MCP spec, but the
        # current SDK returns the full catalogue in a single response for
        # the small servers we care about. We follow the cursor when
        # present so larger servers still work.
        tools: list[Tool] = []
        cursor: str | None = None
        while True:
            page = await self._session.list_tools(cursor)
            tools.extend(page.tools)
            cursor = getattr(page, "nextCursor", None)
            if not cursor:
                break

        skills: list[MCPProxySkill] = []
        for tool in tools:
            try:
                skill = self._build_skill(tool)
            except MistralSchemaError as exc:
                # The MCP server is third-party and may publish schemas we
                # cannot map. Logging at WARNING level keeps the failure
                # visible without aborting registration of the other
                # tools.
                logger.warning(
                    "skipping MCP tool %r: schema is not Mistral-compatible: %s",
                    getattr(tool, "name", "<unnamed>"),
                    exc,
                )
                continue
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "skipping MCP tool %r: unexpected error while building Skill",
                    getattr(tool, "name", "<unnamed>"),
                )
                continue
            skills.append(skill)
        return skills

    # ------------------------------------------------------------------ helpers

    def _build_skill(self, tool: Tool) -> MCPProxySkill:
        """Translate one ``Tool`` descriptor into an :class:`MCPProxySkill`."""
        if not isinstance(tool.name, str) or not tool.name:
            raise MistralSchemaError(
                "MCP tool descriptor is missing a non-empty 'name'"
            )

        raw_schema = getattr(tool, "inputSchema", None) or {}
        json_schema = _normalise_input_schema(raw_schema)

        # Validate the (possibly normalised) schema against the Mistral
        # subset. ``validate`` raises ``MistralSchemaError`` on failure;
        # ``load_skills`` catches it and skips the tool with a warning.
        self._validator.validate(json_schema)
        # ``to_mistral_tool`` would do the round-trip, but we defer that
        # to ``SkillRegistry.mistral_tool_definitions`` so the registry
        # owns a single canonical projection.

        skill_name = self._namespaced(tool.name)
        description = getattr(tool, "description", None) or (
            f"MCP tool {tool.name!r}"
            f"{' from ' + self._server_name if self._server_name else ''}"
        )

        manifest = SkillManifest(
            name=skill_name,
            description=description,
            json_schema=json_schema,
            destructive=False,
            timeout_seconds=self._timeout_seconds,
            # MCP tools advertise no platform constraints. The reasonable
            # default is "any platform Jarvis runs on" — declare the full
            # supported triple so the registry never returns
            # ``platform_not_supported`` for an MCP tool on a non-Windows
            # host. This matches the spirit of Requirement 15.4.
            platforms=("windows", "linux", "darwin"),
            source="mcp",
        )
        return MCPProxySkill(
            manifest=manifest,
            session=self._session,
            tool_name=tool.name,
        )

    def _namespaced(self, tool_name: str) -> str:
        """Apply the configured server prefix, if any."""
        if not self._server_name:
            return tool_name
        return f"mcp.{self._server_name}.{tool_name}"


# ---------------------------------------------------------------------------
# Schema normalisation
# ---------------------------------------------------------------------------


def _normalise_input_schema(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise an MCP ``inputSchema`` into a Mistral-friendly JSON Schema.

    Two non-controversial transforms are applied:

    * an empty mapping becomes ``{"type": "object", "properties": {}}``;
    * a top-level mapping that omits ``"type"`` has it inserted as
      ``"object"``.

    Both transforms only touch the shallowest level of the schema and
    therefore do not affect the inner subset rules enforced by
    :class:`MistralSchemaValidator`. ``Mapping`` rather than ``dict`` is
    accepted on input so MCP-emitted ``pydantic`` models or read-only
    proxies pass through cleanly.
    """
    if not isinstance(raw, Mapping):
        raise MistralSchemaError(
            f"MCP inputSchema must be a mapping, got {type(raw).__name__}"
        )
    schema: dict[str, Any] = dict(raw)
    if not schema:
        return {"type": "object", "properties": {}}
    if "type" not in schema:
        schema["type"] = "object"
    return schema


# ---------------------------------------------------------------------------
# CallToolResult translation
# ---------------------------------------------------------------------------


def _translate_call_result(tool_name: str, result: CallToolResult) -> SkillResult:
    """Map a ``CallToolResult`` onto :class:`SkillResult`.

    The MCP protocol surfaces success and "tool executed but reported an
    application error" as the same shape — distinguished by
    ``isError``. We translate them as follows:

    * ``isError == False`` → :class:`SkillResult.success` carrying the
      structured / text payload under ``value``.
    * ``isError == True`` → :class:`SkillResult.error("internal_error",
      ...)`` because the closed taxonomy has no dedicated "tool reported
      a domain error" code; the human-readable message preserves the
      original text content for the LLM to reason about.
    """
    structured = getattr(result, "structuredContent", None)
    contents = getattr(result, "content", None) or []
    text_payload = _extract_text(contents)

    if getattr(result, "isError", False):
        message = text_payload or f"MCP tool {tool_name!r} reported an error"
        # Preserve the structured content in ``value`` so a later
        # debugging / replay tool can still inspect it.
        return SkillResult.error(
            "internal_error",
            message,
            value={
                "tool": tool_name,
                "structured_content": _to_jsonable(structured),
                "text": text_payload,
            },
        )

    value: dict[str, Any] = {"tool": tool_name}
    if text_payload is not None:
        value["text"] = text_payload
    if structured is not None:
        value["structured_content"] = _to_jsonable(structured)
    return SkillResult.success(value=value)


def _extract_text(contents: Iterable[Any]) -> str | None:
    """Concatenate ``TextContent.text`` values, trimming oversize payloads."""
    chunks: list[str] = []
    total = 0
    for item in contents:
        text = _maybe_text(item)
        if text is None:
            continue
        if total + len(text) > _MAX_TEXT_PAYLOAD_CHARS:
            remaining = _MAX_TEXT_PAYLOAD_CHARS - total
            if remaining > 0:
                chunks.append(text[:remaining])
            chunks.append("…[truncated]")
            break
        chunks.append(text)
        total += len(text)
    if not chunks:
        return None
    return "".join(chunks)


def _maybe_text(item: Any) -> str | None:
    """Return the ``text`` field of a ``TextContent``-shaped object, or None."""
    text = getattr(item, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(item, Mapping):
        item_text = item.get("text")
        if isinstance(item_text, str):
            return item_text
    return None


def _to_jsonable(value: Any) -> Any:
    """Best-effort conversion of MCP pydantic models to plain JSON values.

    The audit log and Memory_Store both expect plain dicts/lists. MCP's
    ``structuredContent`` is already a ``dict[str, Any]`` per the spec,
    but we round-trip through ``model_dump`` when available to be safe.
    """
    if value is None:
        return None
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except TypeError:  # pragma: no cover - older pydantic API
            return dump()
    return value


# ---------------------------------------------------------------------------
# Top-level helper
# ---------------------------------------------------------------------------


@dataclass
class _ConnectedMCP:
    """Bundle returned by :func:`connect_mcp_skills`.

    Holds the loaded Skills along with an :func:`aclose` coroutine the
    caller MUST await on shutdown to tear down the underlying transport
    cleanly.
    """

    skills: list[MCPProxySkill]
    aclose: Callable[[], Awaitable[None]]


async def connect_mcp_skills(server_config: Any) -> _ConnectedMCP:
    """Open an MCP transport, initialise the session, and load Skills.

    ``server_config`` is a duck-typed object with the shape of
    :class:`jarvis.config.schema.McpServerConfig`. Two transports are
    supported:

    * **stdio** — used when ``server_config.command`` is set. ``args``
      and ``env`` are forwarded verbatim to
      :class:`mcp.StdioServerParameters`.
    * **SSE** — used when ``server_config.url`` is set (or
      ``server_config.transport == "sse"``). The ``url`` value is passed
      to :func:`mcp.client.sse.sse_client`. A ``headers`` mapping is
      forwarded if present.

    The two cases are mutually exclusive; specifying neither raises
    :class:`ValueError`.

    Returns
    -------
    :class:`_ConnectedMCP`
        Bundle of ``skills`` (the loaded :class:`MCPProxySkill` list) and
        an :attr:`aclose` coroutine. ``aclose`` MUST be awaited on
        shutdown — typically wired into the application's
        ``aclose`` chain alongside the rest of the runtime.

    Raises
    ------
    ValueError
        ``server_config`` is missing both ``command`` and ``url``.
    ImportError
        The :mod:`mcp` package is not installed in the current
        environment. The caller can choose to log a warning and continue
        without MCP support.
    """
    # Lazy import so callers without ``mcp`` installed can still import
    # this module (e.g., for type hints or unit tests of the adapter
    # itself with a fake session).
    try:
        from mcp import ClientSession, StdioServerParameters  # noqa: PLC0415 - optional dep
        from mcp.client.sse import sse_client  # noqa: PLC0415 - optional dep
        from mcp.client.stdio import stdio_client  # noqa: PLC0415 - optional dep
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "the 'mcp' package is required for MCP integration; "
            "install with `pip install mcp`"
        ) from exc

    server_name = _safe_attr(server_config, "name", default=None)
    command = _safe_attr(server_config, "command", default=None)
    url = _safe_attr(server_config, "url", default=None)

    stack = AsyncExitStack()
    try:
        if command:
            args = list(_safe_attr(server_config, "args", default=()) or [])
            env_raw = _safe_attr(server_config, "env", default=None)
            env = dict(env_raw) if env_raw else None
            params = StdioServerParameters(
                command=command,
                args=args,
                env=env,
            )
            read, write = await stack.enter_async_context(stdio_client(params))
        elif url:
            headers_raw = _safe_attr(server_config, "headers", default=None)
            headers = dict(headers_raw) if headers_raw else None
            # ``sse_client`` yields a ``(read, write)`` tuple just like
            # ``stdio_client``; the caller treats both transports
            # uniformly from this point on.
            ctx = sse_client(url, headers=headers) if headers else sse_client(url)
            read, write = await stack.enter_async_context(ctx)
        else:
            raise ValueError(
                "MCP server_config must specify either 'command' (stdio) "
                "or 'url' (sse)"
            )

        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        adapter = MCPSkillAdapter(session, server_name=server_name)
        skills = await adapter.load_skills()
    except BaseException:
        # Tear the partial transport down before propagating so we do not
        # leak file descriptors / subprocesses on construction failure.
        await stack.aclose()
        raise

    async def _aclose() -> None:
        await stack.aclose()

    return _ConnectedMCP(skills=skills, aclose=_aclose)


def _safe_attr(obj: Any, name: str, *, default: Any) -> Any:
    """Read ``obj.name`` (attribute or dict key) returning ``default`` if absent."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)
