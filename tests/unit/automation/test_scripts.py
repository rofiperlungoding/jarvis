"""Unit tests for :mod:`jarvis.automation.scripts`.

Covers the lookup-only execution contract (Requirement 9.5), the
``script_not_found`` error path (Requirement 9.4), the 60 s default
timeout (Requirement 9.8), and the basic catalog read API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jarvis.automation.platform import (
    BasePlatformAdapter,
    MouseButton,
    PlatformAdapter,
    ProcessHandle,
    ScriptInterpreter,
    ScriptResult,
)
from jarvis.automation.scripts import (
    DEFAULT_SCRIPT_TIMEOUT_SECONDS,
    ScriptCatalog,
)
from jarvis.config.schema import ScriptCatalogEntry

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _RecordingAdapter(BasePlatformAdapter):
    """Platform adapter that records ``run_script`` calls for assertions.

    Inherits :class:`BasePlatformAdapter` so every other capability still
    raises ``PlatformNotSupportedError``; only ``run_script`` is
    overridden, which keeps the test focused on the runner's contract.
    """

    platform_tag = "test"

    def __init__(
        self,
        result: ScriptResult | None = None,
        *,
        raise_exc: BaseException | None = None,
    ) -> None:
        self.calls: list[tuple[ScriptInterpreter, Path, float]] = []
        self._result = result or ScriptResult(
            exit_code=0, stdout="ok\n", stderr="", duration_ms=12
        )
        self._raise_exc = raise_exc

    async def run_script(
        self,
        interpreter: ScriptInterpreter,
        script_path: Path,
        timeout_s: float,
    ) -> ScriptResult:
        self.calls.append((interpreter, script_path, timeout_s))
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._result


def _entry(
    interpreter: ScriptInterpreter = "powershell",
    path: str = "C:/scripts/sample.ps1",
    description: str = "",
) -> ScriptCatalogEntry:
    return ScriptCatalogEntry(
        interpreter=interpreter, path=path, description=description
    )


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_default_timeout_is_60_seconds() -> None:
    """Requirement 9.8 — the default budget matches the requirement floor."""
    assert DEFAULT_SCRIPT_TIMEOUT_SECONDS == 60.0


def test_construction_with_valid_entries() -> None:
    adapter = _RecordingAdapter()
    catalog = ScriptCatalog(
        {
            "backup": _entry(description="Daily backup"),
            "deploy": _entry(interpreter="python", path="C:/scripts/deploy.py"),
        },
        adapter,
    )
    assert catalog.list_ids() == ["backup", "deploy"]


def test_entries_snapshot_isolated_from_caller_mutation() -> None:
    """Mutating the source dict must not change the catalog."""
    adapter = _RecordingAdapter()
    src: dict[str, ScriptCatalogEntry] = {"backup": _entry()}
    catalog = ScriptCatalog(src, adapter)
    src["deploy"] = _entry(interpreter="python", path="C:/scripts/deploy.py")
    assert catalog.list_ids() == ["backup"]


def test_construction_rejects_non_mapping_entries() -> None:
    adapter = _RecordingAdapter()
    with pytest.raises(TypeError):
        ScriptCatalog(["not", "a", "mapping"], adapter)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_id", ["", 123, None])
def test_construction_rejects_bad_id(bad_id: Any) -> None:
    adapter = _RecordingAdapter()
    with pytest.raises((TypeError, ValueError)):
        ScriptCatalog({bad_id: _entry()}, adapter)


def test_construction_rejects_non_entry_value() -> None:
    adapter = _RecordingAdapter()
    with pytest.raises(TypeError):
        ScriptCatalog(
            {"backup": {"interpreter": "powershell", "path": "x"}},  # type: ignore[dict-item]
            adapter,
        )


def test_construction_rejects_non_protocol_adapter() -> None:
    class NotAnAdapter:
        pass

    with pytest.raises(TypeError):
        ScriptCatalog({"backup": _entry()}, NotAnAdapter())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Read-only catalog access
# ---------------------------------------------------------------------------


def test_list_ids_returns_fresh_copy() -> None:
    adapter = _RecordingAdapter()
    catalog = ScriptCatalog({"a": _entry(), "b": _entry()}, adapter)
    ids = catalog.list_ids()
    ids.append("c")
    assert catalog.list_ids() == ["a", "b"]


def test_get_returns_entry() -> None:
    adapter = _RecordingAdapter()
    entry = _entry(description="Daily backup")
    catalog = ScriptCatalog({"backup": entry}, adapter)
    assert catalog.get("backup") is entry


def test_get_returns_none_for_unknown_id() -> None:
    adapter = _RecordingAdapter()
    catalog = ScriptCatalog({"backup": _entry()}, adapter)
    assert catalog.get("nope") is None


def test_get_rejects_non_string_id() -> None:
    adapter = _RecordingAdapter()
    catalog = ScriptCatalog({"backup": _entry()}, adapter)
    with pytest.raises(TypeError):
        catalog.get(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# run() — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_forwards_to_adapter_with_default_timeout() -> None:
    """Requirements 9.1 / 9.3 — successful lookup invokes the adapter."""
    adapter = _RecordingAdapter()
    catalog = ScriptCatalog(
        {"backup": _entry(path="C:/scripts/backup.ps1")},
        adapter,
    )
    result = await catalog.run("backup")
    assert isinstance(result, ScriptResult)
    assert result.exit_code == 0
    assert adapter.calls == [
        ("powershell", Path("C:/scripts/backup.ps1"), 60.0),
    ]


@pytest.mark.asyncio
async def test_run_forwards_explicit_timeout() -> None:
    adapter = _RecordingAdapter()
    catalog = ScriptCatalog(
        {"deploy": _entry(interpreter="python", path="C:/scripts/deploy.py")},
        adapter,
    )
    await catalog.run("deploy", timeout_seconds=10.5)
    assert adapter.calls[0][2] == pytest.approx(10.5)


@pytest.mark.asyncio
async def test_run_propagates_timed_out_result() -> None:
    """Requirement 9.8 — the runner does not re-classify a timeout result."""
    adapter = _RecordingAdapter(
        result=ScriptResult(
            exit_code=-1,
            stdout="",
            stderr="killed",
            duration_ms=60_000,
            timed_out=True,
        )
    )
    catalog = ScriptCatalog({"slow": _entry()}, adapter)
    result = await catalog.run("slow")
    assert result.timed_out is True


@pytest.mark.parametrize(
    "interpreter",
    ["powershell", "python", "batch"],
)
@pytest.mark.asyncio
async def test_run_supports_every_interpreter(
    interpreter: ScriptInterpreter,
) -> None:
    """Requirement 9.3 — all three interpreters are forwarded verbatim."""
    adapter = _RecordingAdapter()
    catalog = ScriptCatalog(
        {"job": _entry(interpreter=interpreter, path="C:/scripts/job")},
        adapter,
    )
    await catalog.run("job")
    assert adapter.calls[0][0] == interpreter


# ---------------------------------------------------------------------------
# run() — error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_unknown_id_raises_key_error() -> None:
    """Requirement 9.4 — unknown ids surface as KeyError."""
    adapter = _RecordingAdapter()
    catalog = ScriptCatalog({"backup": _entry()}, adapter)
    with pytest.raises(KeyError) as excinfo:
        await catalog.run("nope")
    # The missing id is the only argument so the Skill can quote it back
    # to the user verbatim.
    assert excinfo.value.args == ("nope",)
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_run_does_not_accept_arbitrary_script_text() -> None:
    """Requirement 9.5 — only registered ids are accepted.

    The runner never parses ``script_id`` as a path or as inline script
    text, so even a value that *looks* like a script body is rejected
    with ``KeyError`` rather than being executed.
    """
    adapter = _RecordingAdapter()
    catalog = ScriptCatalog({"backup": _entry()}, adapter)
    with pytest.raises(KeyError):
        await catalog.run("Write-Host 'pwned'")
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_run_rejects_non_string_id() -> None:
    adapter = _RecordingAdapter()
    catalog = ScriptCatalog({"backup": _entry()}, adapter)
    with pytest.raises(TypeError):
        await catalog.run(123)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_run_rejects_empty_id() -> None:
    adapter = _RecordingAdapter()
    catalog = ScriptCatalog({"backup": _entry()}, adapter)
    with pytest.raises(ValueError):
        await catalog.run("")


@pytest.mark.parametrize("bad_timeout", [0, -1, -0.5])
@pytest.mark.asyncio
async def test_run_rejects_non_positive_timeout(bad_timeout: float) -> None:
    adapter = _RecordingAdapter()
    catalog = ScriptCatalog({"backup": _entry()}, adapter)
    with pytest.raises(ValueError):
        await catalog.run("backup", timeout_seconds=bad_timeout)


@pytest.mark.asyncio
async def test_run_rejects_non_finite_timeout() -> None:
    adapter = _RecordingAdapter()
    catalog = ScriptCatalog({"backup": _entry()}, adapter)
    with pytest.raises(ValueError):
        await catalog.run("backup", timeout_seconds=float("nan"))


@pytest.mark.asyncio
async def test_run_rejects_bool_timeout() -> None:
    adapter = _RecordingAdapter()
    catalog = ScriptCatalog({"backup": _entry()}, adapter)
    with pytest.raises(TypeError):
        await catalog.run("backup", timeout_seconds=True)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_run_rejects_non_numeric_timeout() -> None:
    adapter = _RecordingAdapter()
    catalog = ScriptCatalog({"backup": _entry()}, adapter)
    with pytest.raises(TypeError):
        await catalog.run("backup", timeout_seconds="60")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Adapter exception passthrough
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_propagates_adapter_exception() -> None:
    """Adapter-level OS errors propagate so the registry can classify them."""
    adapter = _RecordingAdapter(raise_exc=OSError("simulated"))
    catalog = ScriptCatalog({"backup": _entry()}, adapter)
    with pytest.raises(OSError, match="simulated"):
        await catalog.run("backup")


# ---------------------------------------------------------------------------
# Sanity: ensure the test adapter does not accidentally type-mismatch
# ---------------------------------------------------------------------------


def test_recording_adapter_is_a_platform_adapter() -> None:
    """``runtime_checkable`` Protocol should accept the test stub."""
    adapter = _RecordingAdapter()
    assert isinstance(adapter, PlatformAdapter)
    # Touching MouseButton / ProcessHandle prevents lint from flagging the
    # imports as unused while keeping the file's value-type provenance
    # explicit alongside ScriptResult.
    assert MouseButton == MouseButton  # noqa: PLR0124
    assert ProcessHandle.__name__ == "ProcessHandle"
