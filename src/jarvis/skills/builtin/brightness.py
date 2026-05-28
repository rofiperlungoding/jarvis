"""``BrightnessSkill`` — adjust primary display brightness via the platform adapter.

Implements the ``BrightnessSkill`` tool advertised in
``design.md §Built-in Skills`` and required by Requirements 4.6, 4.7,
and 4.8:

* The Skill argument schema constrains ``operation`` to
  ``{"set", "increase", "decrease"}`` (Requirement 4.6) and accepts an
  optional integer ``level`` in ``[0, 100]``. With ``operation="set"``
  the ``level`` is required (the operation is meaningless without a
  target); with ``increase`` / ``decrease`` it is the delta in
  percentage points and defaults to 10 percent — mirroring the
  ``VolumeSkill`` convention from the task list ("default delta 10").
* The Skill delegates to :meth:`PlatformAdapter.get_brightness` and
  :meth:`PlatformAdapter.set_brightness` for the actual WMI roundtrip
  (Requirement 4.7). The Skill never imports ``wmi`` directly: that
  detail belongs to ``WindowsAdapter``.
* When the active display does not support programmatic brightness
  control — surfaced as :class:`PlatformNotSupportedError` from the
  adapter's ``WmiMonitorBrightnessMethods`` call — the Skill returns
  ``SkillResult.error("not_supported", ...)`` per Requirement 4.8.

Why ``not_supported`` rather than ``platform_not_supported``
-----------------------------------------------------------

The platform layer raises :class:`PlatformNotSupportedError` whose
``error_code`` is ``"platform_not_supported"``; that code is reserved
for capabilities the *whole adapter* does not implement (Requirement
15.4). Brightness is more nuanced: WMI is implemented on Windows but
the *active monitor* may not honour ``WmiMonitorBrightnessMethods`` —
Requirement 4.8 calls this case ``"not_supported"``. We therefore
translate the platform-level exception into the Skill-level
``not_supported`` error code rather than forwarding
``platform_not_supported``, keeping the user-facing message faithful to
the requirement ("THE Dialog_Manager SHALL inform the user of the
limitation").

Validates: Requirements 4.6, 4.7, 4.8
"""

from __future__ import annotations

import logging
from typing import Any, Final

from jarvis.automation.platform import (
    PlatformAdapter,
    PlatformNotSupportedError,
)
from jarvis.skills.base import Skill, SkillContext, SkillManifest, SkillResult

logger = logging.getLogger(__name__)

__all__ = [
    "BRIGHTNESS_OPERATIONS",
    "DEFAULT_DELTA_PCT",
    "SKILL",
    "BrightnessSkill",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Closed set of operations accepted by this Skill, mirroring Requirement
#: 4.6. Exposed at module scope so the registry property tests can pin
#: the vocabulary without importing the schema dict.
BRIGHTNESS_OPERATIONS: Final[tuple[str, ...]] = ("set", "increase", "decrease")

#: Default delta (in percentage points) applied to ``increase`` /
#: ``decrease`` when ``level`` is omitted. Mirrors the ``VolumeSkill``
#: precedent ("default delta 10") from the task list.
DEFAULT_DELTA_PCT: Final[int] = 10


# ---------------------------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------------------------

# JSON-Schema draft-07 conforming to the Mistral function-calling subset
# enforced by ``MistralSchemaValidator``: no ``$ref``, no scalar/object
# ``oneOf`` mixing, no unsupported ``format`` keyword. ``level`` is bounded
# to ``[0, 100]`` per Requirement 4.6 and described for the LLM so it knows
# the field is a percentage. ``additionalProperties: false`` is critical:
# it is the gate that lets the registry return ``schema_violation`` for a
# Tool_Call carrying arguments the Skill does not understand.
_JSON_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "enum": list(BRIGHTNESS_OPERATIONS),
            "description": (
                "Brightness operation to perform. 'set' assigns an "
                "absolute level; 'increase' / 'decrease' adjust the "
                "current level by 'level' percent (default 10)."
            ),
        },
        "level": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": (
                "Percentage in [0, 100]. Required for 'set'; optional "
                "delta for 'increase'/'decrease' (default 10)."
            ),
        },
    },
    "required": ["operation"],
    "additionalProperties": False,
}


_MANIFEST: Final[SkillManifest] = SkillManifest(
    name="brightness",
    description=(
        "Adjust the primary display brightness. operation is one of "
        "'set' (absolute), 'increase', or 'decrease'. level is an "
        "integer percentage in [0, 100]; required for 'set' and "
        "optional (default 10) for 'increase'/'decrease'."
    ),
    json_schema=_JSON_SCHEMA,
    destructive=False,
    timeout_seconds=5.0,
    platforms=("windows",),
    source="builtin",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp_pct(value: int) -> int:
    """Clamp ``value`` into ``[0, 100]`` without raising on out-of-range input.

    The Skill registry has already validated the schema so ``level``
    arguments are within range, but the *result* of an ``increase`` /
    ``decrease`` may overflow either bound (e.g. ``decrease`` when
    current brightness is 5 with delta 10). Clamping here keeps the
    platform call well-formed and matches the "clamped to ``[0, 100]``"
    contract documented on :meth:`PlatformAdapter.set_brightness`.
    """
    return max(0, min(100, int(value)))


# ---------------------------------------------------------------------------
# Skill implementation
# ---------------------------------------------------------------------------


class BrightnessSkill:
    """Adjust the primary display brightness via the platform adapter.

    The Skill is non-destructive: changing brightness is reversible and
    has no permanent side effect, so :attr:`SkillManifest.destructive`
    is ``False`` and the Authorization_Policy will not request user
    confirmation before dispatch (Requirement 16.1).

    ``platforms`` is restricted to ``("windows",)`` because the only
    adapter implementing brightness today is
    :class:`~jarvis.automation.windows_adapter.WindowsAdapter` (see
    Requirement 4.7). Future macOS / Linux adapters that gain brightness
    support can re-register the Skill with a broader manifest — the
    executor itself is platform-agnostic; it talks only to the Protocol.
    """

    manifest: SkillManifest = _MANIFEST

    async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        """Apply the requested ``operation`` to the active display.

        ``args`` has already been validated against :data:`_JSON_SCHEMA`
        by the :class:`SkillRegistry` (Property 2 / CP2), so we can
        assume ``"operation"`` is present and one of the three accepted
        values, and ``"level"`` (if present) is an integer in
        ``[0, 100]``.
        """
        # Pre-flight checks. The dataclass default allows ``ctx`` to omit
        # the platform adapter for test-injection convenience, but the
        # Skill cannot fulfil its contract without an OS-level side
        # effect, so we surface those misconfigurations as
        # ``internal_error`` rather than the broader
        # ``platform_not_supported``.
        adapter = ctx.platform_adapter
        adapter_error = self._validate_adapter(adapter)
        if adapter_error is not None:
            return adapter_error

        operation = args["operation"]
        level = args.get("level")

        # Requirement 4.6 makes ``level`` optional in the *schema*, but
        # the ``set`` operation is meaningless without a target. Surface
        # the missing-field case as ``schema_violation`` so the LLM
        # retries with a corrected payload (Requirement 14.5) rather
        # than the broader ``internal_error``.
        if operation == "set" and level is None:
            return SkillResult.error(
                "schema_violation",
                "'level' is required when operation == 'set'",
                value={"missing": "level"},
            )

        try:
            return await self._apply(adapter, operation, level)
        except PlatformNotSupportedError as exc:
            # Requirement 4.8: the active display does not support
            # programmatic brightness control. Map to the dedicated
            # ``not_supported`` error code (rather than the broader
            # ``platform_not_supported``) so the Dialog_Manager can
            # surface the right message — "this display doesn't support
            # that, sir" — instead of the misleading "brightness isn't
            # available on this platform".
            return self._not_supported(operation, exc)

    @staticmethod
    def _validate_adapter(adapter: Any) -> SkillResult | None:
        """Return an ``internal_error`` result if ``adapter`` is unusable."""
        if adapter is None:
            return SkillResult.error(
                "internal_error",
                "BrightnessSkill requires ctx.platform_adapter",
            )
        # Cheap belt-and-braces guard: the SkillContext field is typed
        # ``Any`` to avoid an import cycle, so misconfigured contexts
        # could smuggle in an unrelated object.
        if not isinstance(adapter, PlatformAdapter):
            return SkillResult.error(
                "internal_error",
                "ctx.platform_adapter does not satisfy the PlatformAdapter "
                f"protocol (got {type(adapter).__name__})",
            )
        return None

    @staticmethod
    async def _apply(
        adapter: PlatformAdapter, operation: str, level: int | None
    ) -> SkillResult:
        """Execute the platform call(s) for ``operation`` and build the result.

        Splitting this off the main ``execute`` keeps the top-level
        coroutine's return-statement count under the linter's
        ``PLR0911`` budget while preserving a single ``except
        PlatformNotSupportedError`` site in the caller.
        """
        if operation == "set":
            # ``level`` is non-None here: the caller has already
            # short-circuited the missing-level case via
            # ``schema_violation``.
            assert level is not None
            target = _clamp_pct(level)
            await adapter.set_brightness(target)
            return SkillResult.success(value={"operation": "set", "level": target})

        # ``increase`` / ``decrease`` need to read the current value
        # before computing the new absolute target. Both
        # ``get_brightness`` and ``set_brightness`` may raise
        # :class:`PlatformNotSupportedError`; the caller's ``except``
        # funnels both into the ``not_supported`` branch.
        delta = DEFAULT_DELTA_PCT if level is None else int(level)
        current = int(await adapter.get_brightness())
        signed = delta if operation == "increase" else -delta
        target = _clamp_pct(current + signed)
        await adapter.set_brightness(target)
        return SkillResult.success(
            value={
                "operation": operation,
                "level": target,
                "previous_level": current,
                "delta": delta,
            }
        )

    @staticmethod
    def _not_supported(operation: str, exc: PlatformNotSupportedError) -> SkillResult:
        """Translate :class:`PlatformNotSupportedError` into ``not_supported``."""
        logger.info(
            "brightness operation %r not supported on this display "
            "(capability=%s, platform=%s, detail=%s)",
            operation,
            exc.capability,
            exc.platform,
            exc.detail,
        )
        detail = exc.detail or str(exc)
        return SkillResult.error(
            "not_supported",
            (
                "this display does not support programmatic "
                f"brightness control ({detail})"
            ),
            value={
                "capability": exc.capability,
                "platform": exc.platform,
                "detail": exc.detail,
            },
        )


#: Top-level export consumed by :meth:`SkillRegistry.discover`. Plugin
#: discovery looks for an attribute named exactly ``SKILL`` on the loaded
#: module, so we expose a single shared instance.
SKILL: Skill = BrightnessSkill()
