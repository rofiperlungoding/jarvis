"""Built-in :class:`DesktopAutomationSkill` — scripted UI automation.

Wraps :meth:`jarvis.automation.platform.PlatformAdapter.click`,
:meth:`~jarvis.automation.platform.PlatformAdapter.type_text`,
:meth:`~jarvis.automation.platform.PlatformAdapter.hotkey`, and
:meth:`~jarvis.automation.platform.PlatformAdapter.focus_window` behind a
single Skill that the LLM can invoke through Mistral function calling.
The acceptance criteria (Requirement 9.6) fix the user-facing action
vocabulary at four values — ``click``, ``type``, ``hotkey``,
``focus_window`` — and require a *typed payload* for each. Requirement
9.7 mandates the action be performed via ``pyautogui`` / ``pywinauto``,
which is exactly what :class:`~jarvis.automation.windows_adapter.WindowsAdapter`
does on Windows; the Skill itself stays platform-agnostic by talking to
the :class:`PlatformAdapter` Protocol.

Argument schema
---------------

The schema declares one required field ``action`` whose enum is
:data:`DESKTOP_AUTOMATION_ACTIONS`. The remaining fields are optional at
the property-level so the model can omit irrelevant ones, with a per-action
``allOf`` / ``if`` / ``then`` block enforcing exactly which payload fields
are required for each action:

* ``click``        → ``x`` and ``y`` are required (integers); ``button``
  defaults to ``"left"`` and is constrained to
  :data:`MOUSE_BUTTONS`.
* ``type``         → ``text`` is required (string).
* ``hotkey``       → ``keys`` is required (non-empty list of non-empty
  strings).
* ``focus_window`` → ``title_pattern`` is required (non-empty string).

``additionalProperties: false`` is the gate that lets the Skill_Registry
return ``schema_violation`` when a Tool_Call carries fields outside this
set (Property 2 / CP2). The conditional ``allOf`` block lets the same
schema layer reject, e.g., a ``click`` call missing coordinates without
the executor ever running.

Action → adapter call mapping
-----------------------------

* ``click``        → :meth:`PlatformAdapter.click(x, y, button)`.
* ``type``         → :meth:`PlatformAdapter.type_text(text)`. The
  adapter's :class:`InputSanitizer` scrubs ``text`` of unsafe control
  characters before forwarding it to ``pyautogui``.
* ``hotkey``       → :meth:`PlatformAdapter.hotkey(*keys)`. The adapter
  rejects keys containing control characters or longer than 32 chars.
* ``focus_window`` → :meth:`PlatformAdapter.focus_window(title_pattern)`.
  The adapter sanitises the pattern before forwarding it to
  ``pywinauto``.

Failure mapping
---------------

* :class:`PlatformNotSupportedError` from the adapter is converted to
  ``SkillResult.error("platform_not_supported", ...)``. This mirrors
  the contract used by :mod:`jarvis.skills.builtin.media_control` and
  :mod:`jarvis.skills.builtin.volume` and matches the error-taxonomy
  entry in :mod:`jarvis.skills.base`.
* ``ctx.platform_adapter`` missing or not satisfying the
  :class:`PlatformAdapter` Protocol → ``internal_error`` so the
  Dialog_Manager can surface the misconfiguration without crashing.
* ``ValueError`` from the adapter (empty title pattern after
  sanitisation, unsafe hotkey key, unsupported mouse button) → mapped
  to ``schema_violation`` with the offending field surfaced in
  ``value``. The Dialog_Manager can then trigger the standard LLM
  retry loop (Requirement 14.5) instead of bailing out.
* Any other adapter exception bubbles up to the registry, which converts
  it to ``SkillResult.error("internal_error", ...)`` — the Skill itself
  does not pre-empt that handling because doing so would swallow
  tracebacks that are useful for forensics (Property 7 / CP10).

Why ``destructive=False``
-------------------------

A single click / keystroke / window focus has no permanent side effect
in isolation, so :attr:`SkillManifest.destructive` is ``False`` and the
Authorization_Policy will not request user confirmation before dispatch
(Requirement 16.1). The combination of input sanitisation in the
adapter, the closed action vocabulary, and the requirement that the
LLM emit explicit coordinates and key chords keeps the surface area
small enough that the Authorization_Policy treats it the same as media
keys and brightness.

Validates: Requirements 9.6, 9.7, 15.4
"""

from __future__ import annotations

import logging
from typing import Any, Final

from jarvis.automation.platform import (
    PLATFORM_NOT_SUPPORTED,
    MouseButton,
    PlatformAdapter,
    PlatformNotSupportedError,
)
from jarvis.skills.base import (
    Skill,
    SkillContext,
    SkillManifest,
    SkillResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MOUSE_BUTTON",
    "DESKTOP_AUTOMATION_ACTIONS",
    "MOUSE_BUTTONS",
    "SKILL",
    "DesktopAutomationSkill",
]


# ---------------------------------------------------------------------------
# Action vocabulary and defaults
# ---------------------------------------------------------------------------

#: Closed set of user-facing actions accepted by the Skill. Mirrors
#: Requirement 9.6 exactly. Exposed at module scope so tests and the
#: registry property tests can introspect the set without parsing the
#: schema dict.
DESKTOP_AUTOMATION_ACTIONS: Final[tuple[str, ...]] = (
    "click",
    "type",
    "hotkey",
    "focus_window",
)

#: Mouse buttons accepted by the ``click`` action. Mirrors the
#: :data:`jarvis.automation.platform.MouseButton` literal so the schema
#: enum and the adapter contract stay in sync. Exposed at module scope
#: for the same reason as :data:`DESKTOP_AUTOMATION_ACTIONS`.
MOUSE_BUTTONS: Final[tuple[str, ...]] = ("left", "right", "middle")

#: Default mouse button applied when the LLM omits ``button`` from a
#: ``click`` payload. ``"left"`` is the overwhelmingly-common choice,
#: matches ``pyautogui.click``'s own default, and keeps the LLM's
#: function-calling output terse.
DEFAULT_MOUSE_BUTTON: Final[MouseButton] = "left"


# JSON Schema accepted by the LLM. Kept as a plain dict so it round-trips
# unchanged through ``json.dumps``/``json.loads`` (Property 12 / CP15).
#
# Per-action requirements are encoded inside the schema using the
# ``allOf`` / ``if`` / ``then`` pattern (the same trick used by
# :mod:`jarvis.skills.builtin.volume` for its ``set`` operation). This
# lets the :class:`SkillRegistry` surface "click without coordinates" as
# ``schema_violation`` (Property 2 / CP2) and trigger the standard LLM
# retry loop (Requirement 14.5) instead of having the executor diagnose
# the missing field.
#
# ``additionalProperties: false`` is critical: it gates extraneous
# arguments the Skill does not understand. The properties block lists
# every field that any action might use; the conditional ``required``
# blocks fix per-action validity.
_JSON_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "title": "DesktopAutomationSkillArguments",
    "description": (
        "Arguments for a single UI automation action. The 'action' "
        "field selects which adapter method is invoked; the remaining "
        "fields carry the typed payload for that action."
    ),
    "properties": {
        "action": {
            "type": "string",
            "enum": list(DESKTOP_AUTOMATION_ACTIONS),
            "description": (
                "UI action to perform. 'click' presses a mouse button "
                "at absolute screen coordinates; 'type' enters text "
                "into the focused control; 'hotkey' presses a chord of "
                "keys; 'focus_window' brings a window matching a "
                "regex/substring pattern to the foreground."
            ),
        },
        "x": {
            "type": "integer",
            "description": (
                "Absolute screen X coordinate for 'click' (pixels)."
            ),
        },
        "y": {
            "type": "integer",
            "description": (
                "Absolute screen Y coordinate for 'click' (pixels)."
            ),
        },
        "button": {
            "type": "string",
            "enum": list(MOUSE_BUTTONS),
            "description": (
                "Mouse button for 'click'. Defaults to 'left' when "
                "omitted; 'right' opens context menus and 'middle' "
                "is forwarded as-is."
            ),
        },
        "text": {
            "type": "string",
            "description": (
                "Text to type for 'type'. Control characters other "
                "than tab/newline/return are stripped by the adapter "
                "before being forwarded to pyautogui."
            ),
        },
        "keys": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
            "description": (
                "Ordered list of keys forming the chord for 'hotkey'. "
                "Each key is a pyautogui key name such as 'ctrl', "
                "'shift', 'c', or 'enter'. The adapter rejects keys "
                "containing control characters or longer than 32 "
                "characters."
            ),
        },
        "title_pattern": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Regex / substring pattern used by 'focus_window' to "
                "match the target window title. The adapter sanitises "
                "the pattern before forwarding it to pywinauto."
            ),
        },
    },
    "required": ["action"],
    "additionalProperties": False,
    "allOf": [
        {
            "if": {
                "properties": {"action": {"const": "click"}},
                "required": ["action"],
            },
            "then": {"required": ["x", "y"]},
        },
        {
            "if": {
                "properties": {"action": {"const": "type"}},
                "required": ["action"],
            },
            "then": {"required": ["text"]},
        },
        {
            "if": {
                "properties": {"action": {"const": "hotkey"}},
                "required": ["action"],
            },
            "then": {"required": ["keys"]},
        },
        {
            "if": {
                "properties": {"action": {"const": "focus_window"}},
                "required": ["action"],
            },
            "then": {"required": ["title_pattern"]},
        },
    ],
}


_MANIFEST: Final[SkillManifest] = SkillManifest(
    name="desktop_automation",
    description=(
        "Perform a single scripted UI action via pyautogui / pywinauto. "
        "Use 'click' with 'x', 'y', and optional 'button' to press a "
        "mouse button at screen coordinates; 'type' with 'text' to "
        "enter text into the focused control; 'hotkey' with 'keys' to "
        "press a key chord (e.g. ['ctrl', 'c']); 'focus_window' with "
        "'title_pattern' to bring a window to the foreground."
    ),
    json_schema=_JSON_SCHEMA,
    destructive=False,
    timeout_seconds=10.0,
    platforms=("windows",),
    source="builtin",
)


# ---------------------------------------------------------------------------
# Skill implementation
# ---------------------------------------------------------------------------


class DesktopAutomationSkill:
    """Skill exposing scripted UI automation (click / type / hotkey / focus).

    The Skill is a thin dispatcher: argument validation is owned by the
    Skill_Registry through the JSON Schema (Property 2 / CP2), so the
    executor only needs to (a) verify the platform adapter is wired up,
    (b) route the action to the correct adapter method, and (c)
    translate :class:`PlatformNotSupportedError` and adapter-level
    :class:`ValueError` into the Skill error taxonomy.

    All free-form text destined for ``pyautogui`` / ``pywinauto`` is
    funneled through the adapter's :class:`InputSanitizer` (see
    ``design.md §Automation_Service``); the Skill therefore does not
    need to scrub the payloads itself.
    """

    manifest: SkillManifest = _MANIFEST

    async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        """Dispatch ``args`` to the appropriate :class:`PlatformAdapter` call.

        ``args`` has already been validated against :data:`_JSON_SCHEMA`
        by the :class:`SkillRegistry` (Property 2 / CP2), so we can
        assume:

        * ``"action"`` is present and one of
          :data:`DESKTOP_AUTOMATION_ACTIONS`;
        * the typed payload required by the chosen action is present
          and well-formed (the conditional ``allOf`` clause guarantees
          this);
        * any ``"button"`` value is one of :data:`MOUSE_BUTTONS`;
        * any ``"keys"`` value is a non-empty list of non-empty
          strings.

        The executor still defends against context-level
        misconfiguration (no platform adapter, smuggled-in non-Protocol
        objects) and against drift between the schema and the
        :data:`DESKTOP_AUTOMATION_ACTIONS` literal by returning
        ``internal_error`` with a clear message.
        """
        adapter = ctx.platform_adapter
        adapter_error = self._validate_adapter(adapter)
        if adapter_error is not None:
            return adapter_error
        # ``adapter`` is now guaranteed to satisfy the PlatformAdapter
        # Protocol (the validator above re-checks).
        assert isinstance(adapter, PlatformAdapter)

        action = args["action"]
        try:
            return await self._dispatch(adapter, action, args)
        except PlatformNotSupportedError as exc:
            # Mirror the platform error code into the SkillResult error
            # taxonomy. The message preserves the capability and any
            # adapter-supplied detail so the Dialog_Manager can speak a
            # useful explanation back to the user (Requirement 15.4).
            logger.info(
                "desktop_automation: action %r unsupported on platform %r: %s",
                action,
                exc.platform,
                exc.detail,
            )
            return SkillResult.error(
                PLATFORM_NOT_SUPPORTED,
                str(exc),
            )
        except ValueError as exc:
            # The adapter raises ``ValueError`` for payload-level
            # constraint violations that the JSON Schema cannot express
            # (e.g. an empty title pattern after sanitisation, an unsafe
            # hotkey key with embedded control characters). Surface as
            # ``schema_violation`` so the Dialog_Manager triggers the
            # standard LLM retry loop (Requirement 14.5) rather than
            # bailing out with an opaque ``internal_error``.
            logger.info(
                "desktop_automation: action %r rejected by adapter: %s",
                action,
                exc,
            )
            return SkillResult.error(
                "schema_violation",
                str(exc),
                value={"action": action},
            )

    @staticmethod
    async def _dispatch(
        adapter: PlatformAdapter, action: str, args: dict[str, Any]
    ) -> SkillResult:
        """Route ``action`` to the corresponding adapter call.

        Pulled out as a helper so the public :meth:`execute` keeps a
        single ``try``/``except`` for :class:`PlatformNotSupportedError`
        and :class:`ValueError` and stays under the per-function
        return-statement budget.
        """
        if action == "click":
            # Schema guarantees ``x`` and ``y`` are present and integers
            # for ``click``. ``button`` is optional and constrained to
            # the :data:`MOUSE_BUTTONS` enum; default to ``"left"``.
            x = int(args["x"])
            y = int(args["y"])
            button: MouseButton = args.get("button", DEFAULT_MOUSE_BUTTON)
            await adapter.click(x, y, button)
            return SkillResult.success(
                value={"action": action, "x": x, "y": y, "button": button},
            )

        if action == "type":
            # Schema guarantees ``text`` is present and a string for
            # ``type``. The adapter is responsible for sanitising
            # control characters and capping length before forwarding
            # the payload to pyautogui.
            text = args["text"]
            await adapter.type_text(text)
            return SkillResult.success(
                value={"action": action, "length": len(text)},
            )

        if action == "hotkey":
            # Schema guarantees ``keys`` is a non-empty list of
            # non-empty strings. We materialise it as a tuple to match
            # the adapter's ``*keys`` signature and to keep the value
            # echoed in the result hashable.
            keys = tuple(str(k) for k in args["keys"])
            await adapter.hotkey(*keys)
            return SkillResult.success(
                value={"action": action, "keys": list(keys)},
            )

        if action == "focus_window":
            # Schema guarantees ``title_pattern`` is a non-empty string
            # for ``focus_window``. The adapter sanitises the pattern
            # and may raise :class:`ValueError` if nothing remains
            # after sanitisation; the caller maps that to
            # ``schema_violation``.
            title_pattern = args["title_pattern"]
            await adapter.focus_window(title_pattern)
            return SkillResult.success(
                value={"action": action, "title_pattern": title_pattern},
            )

        # Defence in depth: drift between
        # :data:`DESKTOP_AUTOMATION_ACTIONS` and the schema enum would
        # surface as an unknown action here. The schema enum prevents
        # this in practice.
        return SkillResult.error(  # pragma: no cover - schema enforces
            "internal_error",
            f"unknown desktop automation action {action!r}; expected "
            f"one of {list(DESKTOP_AUTOMATION_ACTIONS)!r}",
        )

    @staticmethod
    def _validate_adapter(adapter: Any) -> SkillResult | None:
        """Return an ``internal_error`` result if ``adapter`` is unusable.

        The :class:`SkillContext` field is typed ``Any`` to avoid an
        import cycle with :mod:`jarvis.automation.platform`, so a
        misconfigured context can smuggle in either ``None`` or an
        unrelated object. Both cases are wiring bugs, not user-facing
        limitations.
        """
        if adapter is None:
            return SkillResult.error(
                "internal_error",
                "DesktopAutomationSkill requires ctx.platform_adapter",
            )
        if not isinstance(adapter, PlatformAdapter):
            return SkillResult.error(
                "internal_error",
                "ctx.platform_adapter does not satisfy the PlatformAdapter "
                f"protocol (got {type(adapter).__name__})",
            )
        return None


#: Top-level export consumed by :meth:`SkillRegistry.discover`. Plugin
#: discovery looks for an attribute named exactly ``SKILL`` on the
#: loaded module, so we expose a single shared instance. Built-in
#: Skills are also registered explicitly during application bootstrap;
#: exposing ``SKILL`` here keeps the discovery contract uniform between
#: built-in and user-supplied modules.
SKILL: Skill = DesktopAutomationSkill()
