# Authoring JARVIS Skill Plugins

This guide walks through writing your own Skill, dropping it into the plugin
directory, and (optionally) hooking up an external MCP server so its tools
appear alongside the built-ins. By the end you will have:

- A working `WeatherEcho` plugin that JARVIS discovers at startup, exposes to
  the Mistral LLM, and dispatches with full schema validation.
- A clear picture of which JSON Schema constructs the registry accepts
  (Mistral function-calling subset).
- A configured `[skills].mcp_servers` entry that contributes additional tools
  through `MCPSkillAdapter`.

> Read [`docs/setup.md`](setup.md) first — it covers the installation,
> credential store, and config layering this guide assumes.

---

## 1. How discovery works

At startup the `SkillRegistry` scans every directory listed under
`[app].plugin_dirs` for `*.py` files (excluding names beginning with `_`) and
imports each one in sorted order. For every imported module it looks for a
single top-level attribute named `SKILL` whose value satisfies the `Skill`
Protocol; anything else is skipped with a warning.

The contract is intentionally minimal so plugin authors do not need to
subclass anything — a plain object with the right shape is enough. The
registry then:

1. Runs `Draft7Validator.check_schema(manifest.json_schema)` to confirm the
   schema document is itself a valid JSON Schema.
2. Runs `MistralSchemaValidator.validate(...)` to confirm the schema sits
   inside the Mistral function-calling subset (see [§4](#4-json-schema-the-mistral-subset)).
3. Compiles a `Draft7Validator` once and caches it; every Tool_Call's
   arguments are validated through this cached validator before `execute` is
   ever called (Requirement 14.4 / CP2).

If any of those steps fail, the plugin is rejected — the rest of the
directory keeps loading.

---

## 2. The `Skill` Protocol

Every plugin exposes one object that satisfies `jarvis.skills.base.Skill`:

```python
from collections.abc import Awaitable
from typing import Any, Protocol, runtime_checkable

from jarvis.skills.base import SkillContext, SkillManifest, SkillResult


@runtime_checkable
class Skill(Protocol):
    manifest: SkillManifest

    def execute(
        self, args: dict[str, Any], ctx: SkillContext
    ) -> Awaitable[SkillResult]: ...
```

Two attributes, that is the whole interface:

- **`manifest`** — a `SkillManifest` value (frozen dataclass) describing the
  Skill's identity, its argument schema, and its execution policy.
- **`execute`** — a coroutine that runs the Skill against pre-validated
  arguments and returns a `SkillResult`. The signature is typed as returning
  an `Awaitable[SkillResult]` rather than `async def` so implementations may
  use either form, but `async def execute(...)` is the common case.

A few invariants the registry relies on, stated as guarantees you can lean on
when writing `execute`:

- `args` has already been validated against `manifest.json_schema`. You do
  not need to re-check structural shape — defend instead against *semantic*
  failures (missing credentials, paths outside the sandbox, unreachable
  providers).
- Exceptions raised from `execute` are caught and turned into
  `SkillResult.error("internal_error", ...)`. Prefer returning an explicit
  `SkillResult.error(...)` so the error code is precise; the catch-all is
  a safety net, not a primary control-flow path.
- Returning anything other than a `SkillResult` is an error and is rewritten
  to `internal_error`. Always go through `SkillResult.success(...)` /
  `SkillResult.error(...)`.

### 2.1 `SkillResult`

`SkillResult` is the structured return type. Use the convenience
constructors:

```python
SkillResult.success(value={"forecast": "sunny"})
SkillResult.error("missing_credentials", "weather/api_key not configured")
```

The `error_code` MUST be one of the eleven values in the closed taxonomy:

| Code | Use it when… |
|---|---|
| `schema_violation` | A semantic precondition turns out to violate the JSON Schema invariant the registry could not see (rare; usually returned by the registry itself). |
| `missing_credentials` | A provider key is not in `CredentialStore`. |
| `not_supported` | The operation is conceptually fine but unavailable on this device (e.g., monitor without WMI brightness control). |
| `access_denied` | Sandbox / network / policy boundary breached. Prefer raising `PolicyViolation` so the audit log is written automatically. |
| `file_too_large` | File-touching Skills exceeding the size cap. |
| `script_not_found` | A configured script is missing. |
| `timeout` | Skill exceeded `manifest.timeout_seconds`. |
| `provider_unavailable` | Upstream HTTP service returned 5xx / network error. |
| `internal_error` | Any other unexpected failure. The registry correlates these via a short `traceback_id`. |
| `platform_not_supported` | Skill ran on a platform tag outside `manifest.platforms`. |
| `rate_limited` | Upstream returned 429. |

Returning any other string fails the `SkillResult` post-init check.

---

## 3. `SkillManifest`

The manifest is what gets serialised into Mistral's `function` definition,
what the Authorization_Policy reads to decide on confirmation, and what the
registry uses to gate platforms.

```python
from typing import Any, Mapping

from jarvis.skills.base import SkillManifest


SkillManifest(
    name="WeatherEcho",
    description="Echo back a fake forecast for a city. Demo plugin.",
    json_schema={...},                 # Mapping[str, Any] — see §4
    destructive=False,                 # default; True triggers confirmation
    timeout_seconds=30.0,              # default 30 s wall-clock budget
    platforms=("windows", "linux", "darwin"),
    source="user",                     # "builtin" | "user" | "mcp"
)
```

| Field | Notes |
|---|---|
| `name` | Stable identifier shared with the LLM (`function.name` in the Mistral payload). Non-empty string. The registry refuses duplicate names. |
| `description` | Model-facing description. Surfaces as `function.description` and to the user in confirmation prompts. |
| `json_schema` | JSON Schema (draft-07) describing the `arguments` object. Validated against the meta-schema and the Mistral subset on registration. |
| `destructive` | `True` makes every invocation a Destructive_Action — the Dialog_Manager produces a spoken summary and waits for an affirmative response (Requirement 16.1). Use it for anything that mutates remote state, sends messages, or runs scripts. |
| `timeout_seconds` | Wall-clock budget the registry enforces around `execute`. Strictly positive; default 30 s. |
| `platforms` | Tuple of platform tags (`"windows"`, `"linux"`, `"darwin"`) the Skill works on. The registry returns `platform_not_supported` on platforms outside the set (Requirement 15.4). Non-empty. |
| `source` | Provenance: `"builtin"` for shipped Skills, `"user"` for plugin-authored Skills, `"mcp"` for MCP-bridged tools. User plugins should set `"user"` so the audit log records the right origin. |

The dataclass is frozen — its `__post_init__` performs cheap shape checks and
raises `ValueError` / `TypeError` early if you pass the wrong kind of value.

---

## 4. JSON Schema (the Mistral subset)

`MistralSchemaValidator` walks the entire schema tree and refuses three
classes of constructs that Mistral's function-calling endpoint cannot
reliably parse:

1. **Remote `$ref`** — every `$ref` value must be a string starting with
   `"#"` (a local definition). Remote references like `https://...`,
   `file://...`, or `schema.json#/foo` are rejected so the LLM never has to
   fetch external documents.
2. **Mixed-type `oneOf`** — every branch of a `oneOf` must be the same
   "shape": all scalar (`string`, `number`, `integer`, `boolean`, `null`)
   or all non-scalar (`object`, `array`). Mixing the two confuses Mistral's
   tool parser into emitting malformed arguments.
3. **Disallowed `format`** — the only `format` keyword on the allow-list is
   `"date-time"`. Anything else (`email`, `uri`, `uuid`, …) is rejected so
   you see the failure at registration time rather than at runtime through
   silent argument coercion.

Everything else in draft-07 is allowed: `properties`, `required`, `items`,
`enum`, `additionalProperties`, `pattern`, `minLength` / `maxLength`,
`minimum` / `maximum`, `allOf`, `anyOf`, descriptions, defaults, and so on.
The walker recurses through `properties`, `patternProperties`,
`additionalProperties`, `items`, `definitions`, and the composition keywords
so a violation buried deep in a sub-schema is still caught.

Two practical conventions to keep schemas LLM-friendly:

- Set the root `type` to `"object"` and list your inputs under `properties`.
  The Mistral endpoint requires `parameters.type == "object"`; the registry
  produces a clearer error when this is true at the manifest level too.
- Set `additionalProperties: false` so the model gets a clear signal about
  unknown fields. Otherwise hallucinated keys silently pass validation and
  reach your executor.

---

## 5. `SkillContext`

The registry constructs one `SkillContext` per Tool_Call and passes it as the
second argument to `execute`. All fields are optional — tests routinely
build minimal contexts that exercise a single dependency — but production
runs typically populate the whole bundle.

| Field | Purpose |
|---|---|
| `audit_log` | Append-only `AuditLog` for `policy_violation`, `network_egress`, and `error` records. The registry writes the policy-violation entry for you when `execute` raises `PolicyViolation`. Provider clients call `record_network_egress` directly. |
| `time_source` | Injectable clock. Use `ctx.time_source.now()` instead of `datetime.utcnow()` so tests stay deterministic (Property 5 / CP6). |
| `platform_adapter` | OS-level side effects (launch processes, brightness, media keys, notifications, scripted UI). |
| `credential_store` | DPAPI-backed `CredentialStore` for reading secrets. Look up by the credential name your config references — never hard-code keys. |
| `llm_backend` | Active `LLMBackend` (Mistral primary, Ollama fallback). Used by Skills that themselves call the model — `SummarizeFileSkill` is the canonical example. |
| `providers` | `Mapping[str, Any]` keyed by provider name (`"weather"`, `"news"`, `"email"`, `"calendar"`, `"web_search"`). Skills that need a typed adapter pull it from here. |
| `allowed_directories` | Tuple of paths the file-reading Skills may access (Requirements 8.2, 8.6). Anything outside is rejected upstream. |
| `incognito` | `True` while the user is in incognito mode (Requirement 13.3). Skills that persist anything beyond the audit log MUST honour this flag. |
| `run_id` | Stable identifier of the current process run, propagated to audit entries. |
| `extras` | Open-ended `Mapping[str, Any]` for MCP-injected dependencies and test-injected fakes. Use it for one-off integrations; for built-ins prefer adding a typed field. |

If a dependency you require is not on the context, return the appropriate
error code (`missing_credentials`, `provider_unavailable`, `internal_error`)
rather than raising — the Dialog_Manager will surface a useful message to
the user.

---

## 6. The `SKILL` convention

The registry's discovery loop looks for `module.SKILL` exactly:

```python
# my_plugin.py
from jarvis.skills.base import Skill

class MyPluginSkill: ...

SKILL: Skill = MyPluginSkill()
```

Conventions every built-in Skill follows, and that you should follow too:

- Bind the value at module top level — discovery does not introspect classes
  or call factories.
- Bind a *single instance*, not a class. The Skill is a value object; one
  instance is reused across Tool_Calls.
- Use the `Skill` type annotation (or your concrete class) so static
  type-checkers verify the Protocol structurally before runtime.
- Keep import-time work side-effect-free. Module import is part of
  discovery; an import-time exception kills only that plugin, but it still
  shows up in logs as a startup failure.

Files whose names start with `_` (e.g., `_helpers.py`) are skipped, so put
private helpers there.

---

## 7. `plugin_dirs` configuration

Add user plugin directories to `[app].plugin_dirs` in your override config
(`%APPDATA%\Jarvis\config.toml`):

```toml
[app]
data_dir = "%LOCALAPPDATA%/Jarvis"
plugin_dirs = [
  "%APPDATA%/Jarvis/plugins",          # default user plugin dir
  "%USERPROFILE%/Documents/JarvisExtras",
]
```

Tokens (`%APPDATA%`, `%LOCALAPPDATA%`, `%USERPROFILE%`, `%USERNAME%`) are
expanded by the loader. Directories that do not exist are silently skipped,
so it is safe to list optional paths.

The defaults already include `%APPDATA%/Jarvis/plugins` — for most users
that is the only directory you ever need. Drop your `.py` file in there and
restart `jarvis`.

---

## 8. MCP server configuration

External MCP servers contribute Skills via `MCPSkillAdapter` (Requirement
14.6). Each entry under `[skills].mcp_servers` becomes a child process that
JARVIS speaks the Model Context Protocol with; every tool the server
advertises is wrapped as a synthetic `Skill` whose `manifest.source` is
`"mcp"` and whose `execute` proxies to the server's `call_tool` method.

```toml
[skills]
registry_meta_schema = "draft-07"

[[skills.mcp_servers]]
name = "filesystem"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "%USERPROFILE%/Documents"]
env = { NODE_NO_WARNINGS = "1" }

[[skills.mcp_servers]]
name = "git"
command = "uvx"
args = ["mcp-server-git", "--repository", "%USERPROFILE%/Documents/Codes/jarvis"]
```

| Field | Purpose |
|---|---|
| `name` | Stable identifier used as a prefix in audit entries and log lines. Must be unique. |
| `command` | Executable launched as the MCP server (`npx`, `uvx`, `python`, an absolute path). |
| `args` | Argument list passed to the executable. Path tokens are expanded before launch. |
| `env` | Extra environment variables merged into the child process's environment. Secrets should still come from `CredentialStore`; treat this for non-secret tuning only. |

Tools published by an MCP server inherit the same JSON-Schema validation and
Mistral subset rules as built-in Skills. If a server publishes a schema that
fails the subset checker (e.g., uses `format: "uri"`), the offending tool is
dropped from the registry with a logged warning — the rest of the server's
tools keep working.

---

## 9. Worked example: `WeatherEcho`

A self-contained plugin that takes a city name, an optional unit selector,
and an optional ISO-8601 timestamp; returns a fake forecast. It exercises
every concept above without needing a real provider.

Save the file as `%APPDATA%\Jarvis\plugins\weather_echo.py`:

```python
"""WeatherEcho — a worked-example JARVIS plugin.

Returns a synthetic forecast for the requested city. Demonstrates:

* declaring a SkillManifest with a Mistral-compatible JSON Schema,
* using SkillContext for the time source and incognito flag,
* returning structured SkillResult values for both success and error,
* exposing the module-level SKILL attribute the registry looks for.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Final

from jarvis.skills.base import (
    Skill,
    SkillContext,
    SkillManifest,
    SkillResult,
)


# ---------------------------------------------------------------------------
# JSON Schema (draft-07, Mistral subset)
# ---------------------------------------------------------------------------

_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["city"],
    "properties": {
        "city": {
            "type": "string",
            "minLength": 1,
            "maxLength": 80,
            "description": "City name, e.g. 'Bandung' or 'San Francisco'.",
        },
        "units": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "default": "celsius",
            "description": "Temperature unit for the returned forecast.",
        },
        "at": {
            "type": "string",
            "format": "date-time",
            "description": (
                "Optional ISO-8601 instant the forecast should describe. "
                "Defaults to 'now' as observed by SkillContext.time_source."
            ),
        },
    },
}


# ---------------------------------------------------------------------------
# Skill implementation
# ---------------------------------------------------------------------------


class WeatherEchoSkill:
    """Echo back a fake forecast. Read-only; safe by design."""

    manifest: Final[SkillManifest] = SkillManifest(
        name="WeatherEcho",
        description=(
            "Return a synthetic short-form forecast for a given city. "
            "Useful as a JARVIS plugin authoring example; does not call "
            "any external service."
        ),
        json_schema=_SCHEMA,
        destructive=False,           # read-only fake; no confirmation
        timeout_seconds=5.0,         # synthetic work, finishes instantly
        platforms=("windows", "linux", "darwin"),
        source="user",
    )

    async def execute(
        self,
        args: dict[str, Any],
        ctx: SkillContext,
    ) -> SkillResult:
        # The registry has already validated `args` against `_SCHEMA`,
        # so structural defences are not required — but we still guard
        # against *semantic* anomalies the schema cannot express.
        city = args["city"].strip()
        if not city:
            return SkillResult.error(
                "schema_violation",
                "city must contain non-whitespace characters",
            )

        units = args.get("units", "celsius")

        # Prefer the injected time source so tests are deterministic.
        if "at" in args:
            try:
                at = datetime.fromisoformat(args["at"])
            except ValueError as exc:
                return SkillResult.error(
                    "schema_violation",
                    f"could not parse 'at' as ISO-8601: {exc}",
                )
        elif ctx.time_source is not None:
            at = ctx.time_source.now()
        else:
            at = datetime.now(tz=timezone.utc)

        # Honour incognito: a real provider call would skip caching here.
        cached = not ctx.incognito

        temp = 22 if units == "celsius" else 72
        return SkillResult.success(
            value={
                "city": city,
                "units": units,
                "temperature": temp,
                "summary": f"{temp}\u00b0 and clear in {city}",
                "observed_at": at.isoformat(),
                "cached": cached,
            }
        )


# Discovery hook: the registry imports this module and registers whatever
# object is bound to `SKILL` at module top level.
SKILL: Skill = WeatherEchoSkill()
```

### 9.1 Verifying the plugin loads

Restart `jarvis` (or run any entrypoint that builds a `SkillRegistry`) and
watch the debug log:

```powershell
jarvis --log-level DEBUG
```

A successful load produces:

```
DEBUG jarvis.skills.registry: registered skill 'WeatherEcho' (source=user)
```

If you see a `SkillRegistrationError` instead, the message tells you which
rule failed:

| Symptom | Likely cause |
|---|---|
| `has an invalid JSON Schema: ...` | The schema document itself is malformed. Check `type`, `required`, and `properties` shapes. |
| `JSON Schema is not Mistral-compatible: ...format = 'email' is not in the Mistral-supported set` | A `format` keyword other than `date-time` snuck in. Drop it or replace with a `pattern` regex. |
| `JSON Schema is not Mistral-compatible: ...$ref must reference a local definition` | A `$ref` points outside the document. Inline the referenced schema or move it under `definitions` and reference with `#/definitions/...`. |
| `JSON Schema is not Mistral-compatible: ...oneOf mixes scalar and non-scalar branches` | Split the `oneOf` into homogeneous branches, or model it with `anyOf` over a single shape. |
| `a skill named 'WeatherEcho' is already registered` | You loaded the file twice (e.g., it lives in two `plugin_dirs` entries) or the name collides with a built-in. Rename it. |

### 9.2 Smoke-testing without launching the full app

The registry is straightforward to drive directly from a Python REPL or a
test:

```python
import asyncio
from jarvis.skills.base import SkillContext
from jarvis.skills.registry import SkillRegistry

# Import the plugin module the same way the registry would.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "weather_echo", r"%APPDATA%\Jarvis\plugins\weather_echo.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

reg = SkillRegistry()
reg.register(module.SKILL)

result = asyncio.run(
    reg.dispatch("WeatherEcho", {"city": "Bandung"}, SkillContext())
)
assert result.ok, result
print(result.value)
# {'city': 'Bandung', 'units': 'celsius', 'temperature': 22, ...}
```

Invalid arguments short-circuit with `schema_violation` and never invoke the
executor:

```python
result = asyncio.run(
    reg.dispatch("WeatherEcho", {"city": ""}, SkillContext())
)
assert not result.ok and result.error_code == "schema_violation"
```

---

## 10. Patterns worth copying

A few habits from the built-in Skills you should mirror in plugins:

- **Pull dependencies from `ctx`, not from globals.** It keeps tests fast
  and lets the registry inject fakes (Property 5 / CP6).
- **Surface every failure as a `SkillResult.error(...)`.** The registry's
  exception catch is a last-resort safety net; explicit error codes give
  the user a much clearer message.
- **Use `PolicyViolation` for sandbox / network breaches.** Raising
  `PolicyViolation("path outside allowed directories", justification=...)`
  causes the registry to write the audit entry for you and return
  `access_denied` automatically.
- **Pin `additionalProperties: false` and `required: [...]`.** Loose
  schemas are easy for the LLM to drift on; tight schemas mean
  hallucinated fields fail fast at the validator instead of confusing your
  executor.
- **Mark anything mutating `destructive=True`.** Sending a message,
  deleting a file, running a script, hitting a webhook — anything you
  cannot transparently undo — should require confirmation.

---

_Validates: Requirements 14.1, 14.2, 14.3, 14.6._
