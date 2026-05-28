"""Unit tests for :mod:`jarvis.diagnostics`.

These tests exercise :class:`jarvis.diagnostics.DiagnosticsOfferFlow` —
the user-facing diagnostics offer that runs on the next launch after a
JARVIS process crashed (Requirement 17.4). Each test focuses on a
single observable behaviour:

* The flow asks the user via the consent prompt; the prompt receives
  the canonical wording from :data:`DEFAULT_CONSENT_PROMPT`.
* On consent, a redacted JSON report is written under
  ``${app.data_dir}/diagnostics/<timestamp>.json`` and the audit
  ``crash`` row references the report path.
* On decline, no report is written but the audit row is still emitted.
* When no consent prompt is wired, the flow records ``no_prompt`` and
  skips the report write.
* Recent audit entries from the prior run are embedded in the report
  with credential and PII redactors applied first.
* Filenames carry microseconds so two reports per second do not
  collide.

Validates: Requirement 17.4
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any

import pytest

from jarvis.diagnostics import (
    DEFAULT_CONSENT_PROMPT,
    ConsentPrompt,
    DiagnosticsOfferFlow,
    DiagnosticsOfferOutcome,
    DiagnosticsReport,
)
from jarvis.memory.redactor import PIIRedactor
from jarvis.security.audit_log import AuditEntry, AuditLog
from jarvis.security.log_redaction import LogRedactionFilter
from jarvis.utils.time_source import FakeTimeSource

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeConsentPrompt:
    """Consent prompt that returns a canned answer.

    Records every prompt text it receives so tests can assert on the
    exact wording the flow chose.
    """

    def __init__(
        self,
        *,
        consent: bool = False,
        raises: BaseException | None = None,
    ) -> None:
        self._consent = consent
        self._raises = raises
        self.prompts: list[str] = []

    async def ask_consent(self, prompt: str) -> bool:
        self.prompts.append(prompt)
        if self._raises is not None:
            raise self._raises
        return self._consent


# Verify the fake satisfies the runtime-checked Protocol so a future
# refactor of the Protocol surfaces immediately.
def test_fake_prompt_satisfies_protocol() -> None:
    assert isinstance(_FakeConsentPrompt(), ConsentPrompt)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def audit_log_factory(tmp_path: Path) -> Callable[..., AuditLog]:
    """Open a fresh AuditLog under ``tmp_path``.

    Returned as a callable so individual tests can decide which run id
    to use without sharing state across tests.
    """
    counter = [0]

    def _make(
        *,
        run_id: str = "current-run",
        time_source: FakeTimeSource | None = None,
    ) -> AuditLog:
        counter[0] += 1
        path = tmp_path / f"audit_{counter[0]}.sqlite"
        return AuditLog(
            path,
            time_source=time_source or FakeTimeSource(),
            run_id=run_id,
        )

    return _make


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


def _run(coro: Any) -> Any:
    """Tiny ``asyncio.run`` helper used by the synchronous tests."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


def test_construction_rejects_non_audit_log(data_dir: Path) -> None:
    with pytest.raises(TypeError):
        DiagnosticsOfferFlow(audit_log=object(), data_dir=data_dir)  # type: ignore[arg-type]


def test_construction_rejects_negative_tail_limit(
    audit_log_factory: Callable[..., AuditLog], data_dir: Path
) -> None:
    log = audit_log_factory()
    try:
        with pytest.raises(ValueError):
            DiagnosticsOfferFlow(
                audit_log=log,
                data_dir=data_dir,
                audit_tail_limit=-1,
            )
    finally:
        log.close()


def test_diagnostics_dir_is_under_data_dir(
    audit_log_factory: Callable[..., AuditLog], data_dir: Path
) -> None:
    log = audit_log_factory()
    try:
        flow = DiagnosticsOfferFlow(audit_log=log, data_dir=data_dir)
        assert flow.diagnostics_dir == data_dir / "diagnostics"
    finally:
        log.close()


# ---------------------------------------------------------------------------
# Consent paths
# ---------------------------------------------------------------------------


def test_run_with_no_prompt_records_no_prompt_outcome_and_writes_no_report(
    audit_log_factory: Callable[..., AuditLog], data_dir: Path
) -> None:
    """No consent prompt → audit row exists, but nothing is written to disk."""
    log = audit_log_factory()
    try:
        flow = DiagnosticsOfferFlow(audit_log=log, data_dir=data_dir)

        outcome = _run(
            flow.run(
                prior_run_id="prev-run",
                prior_started_at=datetime(2024, 1, 1, tzinfo=UTC),
                last_bootstrap_step="discover_skills",
            )
        )
    finally:
        log.close()

    assert outcome.consented is False
    assert outcome.report_path is None
    assert outcome.outcome.endswith(":no_prompt")
    # Audit row must still exist.
    assert outcome.audit_entry.kind == "crash"
    # And the diagnostics dir must NOT contain any report.
    assert not (data_dir / "diagnostics").exists() or not list(
        (data_dir / "diagnostics").iterdir()
    )


def test_run_on_decline_records_decline_outcome_and_writes_no_report(
    audit_log_factory: Callable[..., AuditLog], data_dir: Path
) -> None:
    log = audit_log_factory()
    try:
        prompt = _FakeConsentPrompt(consent=False)
        flow = DiagnosticsOfferFlow(
            audit_log=log,
            data_dir=data_dir,
            consent_prompt=prompt,
        )

        outcome = _run(
            flow.run(
                prior_run_id="prev-run",
                prior_started_at=datetime(2024, 1, 1, tzinfo=UTC),
                last_bootstrap_step="init_voice_pipeline",
            )
        )
    finally:
        log.close()

    assert prompt.prompts == [DEFAULT_CONSENT_PROMPT]
    assert outcome.consented is False
    assert outcome.report_path is None
    assert outcome.outcome.endswith(":declined")
    assert outcome.audit_entry.kind == "crash"
    assert not (data_dir / "diagnostics").exists() or not list(
        (data_dir / "diagnostics").iterdir()
    )


def test_run_on_consent_writes_redacted_report_and_audits_path(
    audit_log_factory: Callable[..., AuditLog], data_dir: Path
) -> None:
    """Consented offer writes a JSON report; audit references its path."""
    time_source = FakeTimeSource()
    log = audit_log_factory(time_source=time_source)
    try:
        prompt = _FakeConsentPrompt(consent=True)
        flow = DiagnosticsOfferFlow(
            audit_log=log,
            data_dir=data_dir,
            consent_prompt=prompt,
        )

        outcome = _run(
            flow.run(
                prior_run_id="prev-run",
                prior_started_at=datetime(2024, 1, 1, tzinfo=UTC),
                last_bootstrap_step="init_dialog_manager",
                now=datetime(2024, 6, 1, 9, 0, 0, tzinfo=UTC),
            )
        )
    finally:
        log.close()

    assert prompt.prompts == [DEFAULT_CONSENT_PROMPT]
    assert outcome.consented is True
    assert outcome.outcome.endswith(":report_written")
    assert outcome.report_path is not None
    assert outcome.report_path.exists()
    assert outcome.report_path.parent == data_dir / "diagnostics"
    assert outcome.report_path.suffix == ".json"

    # Audit justification references the report path so an operator
    # can find it via the audit log alone.
    assert outcome.audit_entry.kind == "crash"
    assert outcome.audit_entry.justification is not None
    assert str(outcome.report_path) in outcome.audit_entry.justification

    # Report payload is well-formed JSON with the expected fields.
    payload = json.loads(outcome.report_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "prev-run"
    assert payload["last_bootstrap_step"] == "init_dialog_manager"
    assert payload["started_at"] == "2024-01-01T00:00:00+00:00"
    assert payload["generated_at"] == "2024-06-01T09:00:00+00:00"
    assert isinstance(payload["audit_tail"], list)


def test_run_when_prompt_raises_records_prompt_failed_outcome(
    audit_log_factory: Callable[..., AuditLog], data_dir: Path
) -> None:
    log = audit_log_factory()
    try:
        prompt = _FakeConsentPrompt(raises=RuntimeError("user walked away"))
        flow = DiagnosticsOfferFlow(
            audit_log=log,
            data_dir=data_dir,
            consent_prompt=prompt,
        )

        outcome = _run(
            flow.run(
                prior_run_id="prev-run",
                prior_started_at=datetime(2024, 1, 1, tzinfo=UTC),
                last_bootstrap_step="init_dialog_manager",
            )
        )
    finally:
        log.close()

    assert outcome.consented is False
    assert outcome.report_path is None
    assert outcome.outcome.endswith(":prompt_failed")
    assert outcome.audit_entry.kind == "crash"


# ---------------------------------------------------------------------------
# Audit tail collection + redaction
# ---------------------------------------------------------------------------


def test_audit_tail_filters_to_prior_run_id(
    audit_log_factory: Callable[..., AuditLog], data_dir: Path
) -> None:
    """When ``prior_run_id`` is known, the tail only includes that run."""
    log = audit_log_factory(run_id="this-run")
    try:
        # Seed: 3 entries from "prev-run" and 2 entries from "other-run".
        async def seed() -> None:
            for i in range(3):
                await log.record_executed(
                    skill="WeatherSkill",
                    args_json={"i": i},
                    outcome="ok",
                    run_id="prev-run",
                )
            for i in range(2):
                await log.record_executed(
                    skill="NewsSkill",
                    args_json={"i": i},
                    outcome="ok",
                    run_id="other-run",
                )

        _run(seed())

        prompt = _FakeConsentPrompt(consent=True)
        flow = DiagnosticsOfferFlow(
            audit_log=log,
            data_dir=data_dir,
            consent_prompt=prompt,
        )
        outcome = _run(
            flow.run(
                prior_run_id="prev-run",
                prior_started_at=datetime(2024, 1, 1, tzinfo=UTC),
                last_bootstrap_step="init_dialog_manager",
            )
        )
    finally:
        log.close()

    assert outcome.report_path is not None
    payload = json.loads(outcome.report_path.read_text(encoding="utf-8"))
    tail_run_ids = [e["run_id"] for e in payload["audit_tail"]]
    # Only prev-run entries appear; the new ``crash`` row written
    # *during* this offer flow belongs to ``this-run`` and is not
    # included.
    assert set(tail_run_ids) == {"prev-run"}
    assert len(payload["audit_tail"]) == 3


def test_audit_tail_respects_limit(
    audit_log_factory: Callable[..., AuditLog], data_dir: Path
) -> None:
    log = audit_log_factory()
    try:

        async def seed() -> None:
            for i in range(20):
                await log.record_executed(
                    skill="WeatherSkill",
                    args_json={"i": i},
                    outcome="ok",
                    run_id="prev-run",
                )

        _run(seed())

        prompt = _FakeConsentPrompt(consent=True)
        flow = DiagnosticsOfferFlow(
            audit_log=log,
            data_dir=data_dir,
            consent_prompt=prompt,
            audit_tail_limit=5,
        )
        outcome = _run(
            flow.run(
                prior_run_id="prev-run",
                prior_started_at=datetime(2024, 1, 1, tzinfo=UTC),
                last_bootstrap_step="init_dialog_manager",
            )
        )
    finally:
        log.close()

    assert outcome.report_path is not None
    payload = json.loads(outcome.report_path.read_text(encoding="utf-8"))
    # Only the most recent 5 entries appear.
    assert len(payload["audit_tail"]) == 5
    indices = [json.loads(e["args_json"])["i"] for e in payload["audit_tail"]]
    assert indices == [15, 16, 17, 18, 19]


def test_report_scrubs_registered_credential_values(
    audit_log_factory: Callable[..., AuditLog], data_dir: Path
) -> None:
    """A registered secret value is scrubbed from the audit tail."""
    log = audit_log_factory(run_id="prev-run")
    try:

        async def seed() -> None:
            await log.record_error(
                skill="MistralBackend",
                outcome="auth_failed",
                justification="API key sk-SUPERSECRET123 is rejected",
                run_id="prev-run",
            )

        _run(seed())

        redactor = LogRedactionFilter(secrets=["sk-SUPERSECRET123"])
        prompt = _FakeConsentPrompt(consent=True)
        flow = DiagnosticsOfferFlow(
            audit_log=log,
            data_dir=data_dir,
            consent_prompt=prompt,
            log_redaction_filter=redactor,
        )
        outcome = _run(
            flow.run(
                prior_run_id="prev-run",
                prior_started_at=datetime(2024, 1, 1, tzinfo=UTC),
                last_bootstrap_step="init_dialog_manager",
            )
        )
    finally:
        log.close()

    assert outcome.report_path is not None
    text = outcome.report_path.read_text(encoding="utf-8")
    assert "sk-SUPERSECRET123" not in text
    assert "[REDACTED]" in text


def test_report_scrubs_pii_via_pii_redactor(
    audit_log_factory: Callable[..., AuditLog], data_dir: Path
) -> None:
    """An email in audit text is replaced by ``[REDACTED:email]``."""
    log = audit_log_factory(run_id="prev-run")
    try:

        async def seed() -> None:
            await log.record_executed(
                skill="SendEmailSkill",
                args_json={"recipient": "alice@example.com", "body": "hi"},
                outcome="ok",
                run_id="prev-run",
            )

        _run(seed())

        prompt = _FakeConsentPrompt(consent=True)
        flow = DiagnosticsOfferFlow(
            audit_log=log,
            data_dir=data_dir,
            consent_prompt=prompt,
            pii_redactor=PIIRedactor.with_defaults(),
        )
        outcome = _run(
            flow.run(
                prior_run_id="prev-run",
                prior_started_at=datetime(2024, 1, 1, tzinfo=UTC),
                last_bootstrap_step="init_dialog_manager",
            )
        )
    finally:
        log.close()

    assert outcome.report_path is not None
    text = outcome.report_path.read_text(encoding="utf-8")
    assert "alice@example.com" not in text
    # Default PII redactor uses the ``email`` kind label.
    assert "[REDACTED:email]" in text


# ---------------------------------------------------------------------------
# Filename uniqueness
# ---------------------------------------------------------------------------


def test_consent_twice_within_a_second_does_not_collide(
    audit_log_factory: Callable[..., AuditLog], data_dir: Path
) -> None:
    """Two reports written in the same second land in distinct files."""
    log = audit_log_factory()
    try:
        prompt = _FakeConsentPrompt(consent=True)
        flow = DiagnosticsOfferFlow(
            audit_log=log,
            data_dir=data_dir,
            consent_prompt=prompt,
        )

        out1 = _run(
            flow.run(
                prior_run_id="prev-run",
                prior_started_at=datetime(2024, 1, 1, tzinfo=UTC),
                last_bootstrap_step="init_dialog_manager",
                now=datetime(2024, 6, 1, 9, 0, 0, 100, tzinfo=UTC),
            )
        )
        out2 = _run(
            flow.run(
                prior_run_id="prev-run",
                prior_started_at=datetime(2024, 1, 1, tzinfo=UTC),
                last_bootstrap_step="init_dialog_manager",
                now=datetime(2024, 6, 1, 9, 0, 0, 200, tzinfo=UTC),
            )
        )
    finally:
        log.close()

    assert out1.report_path is not None
    assert out2.report_path is not None
    assert out1.report_path != out2.report_path
    assert out1.report_path.exists()
    assert out2.report_path.exists()


def test_filename_format_is_filesystem_safe(
    audit_log_factory: Callable[..., AuditLog], data_dir: Path
) -> None:
    """Generated filenames contain no characters Windows forbids."""
    log = audit_log_factory()
    try:
        prompt = _FakeConsentPrompt(consent=True)
        flow = DiagnosticsOfferFlow(
            audit_log=log,
            data_dir=data_dir,
            consent_prompt=prompt,
        )
        outcome = _run(
            flow.run(
                prior_run_id="prev-run",
                prior_started_at=datetime(2024, 1, 1, tzinfo=UTC),
                last_bootstrap_step="init_dialog_manager",
                now=datetime(2024, 6, 1, 9, 0, 0, 123_456, tzinfo=UTC),
            )
        )
    finally:
        log.close()

    assert outcome.report_path is not None
    # Windows-forbidden characters: < > : " / \ | ? *
    assert not re.search(r'[<>:"/\\|?*]', outcome.report_path.name)


# ---------------------------------------------------------------------------
# DiagnosticsReport
# ---------------------------------------------------------------------------


def test_diagnostics_report_to_json_payload_round_trips_via_json() -> None:
    """A :class:`DiagnosticsReport` round-trips through ``json``."""
    report = DiagnosticsReport(
        run_id="r1",
        started_at=datetime(2024, 1, 1, tzinfo=UTC),
        last_bootstrap_step="init_dialog_manager",
        audit_tail=(
            {
                "id": 1,
                "ts": "2024-01-01T00:00:00+00:00",
                "kind": "executed",
                "skill": "WeatherSkill",
                "args_json": "{}",
                "outcome": "ok",
                "destination": None,
                "justification": None,
                "run_id": "r1",
            },
        ),
        generated_at=datetime(2024, 6, 1, 9, 0, 0, tzinfo=UTC),
    )
    payload = report.to_json_payload()
    encoded = json.dumps(payload, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["run_id"] == "r1"
    assert decoded["last_bootstrap_step"] == "init_dialog_manager"
    assert len(decoded["audit_tail"]) == 1


def test_diagnostics_report_payload_includes_sentinel_extras() -> None:
    """User-supplied sentinel extras are preserved into the report."""
    report = DiagnosticsReport(
        run_id="r1",
        started_at=None,
        last_bootstrap_step=None,
        audit_tail=(),
        generated_at=datetime(2024, 6, 1, tzinfo=UTC),
        sentinel_extras={"hostname": "host-1", "version": "0.1.0"},
    )
    payload = report.to_json_payload()
    assert payload["sentinel_extras"] == {
        "hostname": "host-1",
        "version": "0.1.0",
    }


# ---------------------------------------------------------------------------
# Outcome dataclass
# ---------------------------------------------------------------------------


def test_outcome_dataclass_is_hashable_and_repr_safe(
    audit_log_factory: Callable[..., AuditLog], data_dir: Path
) -> None:
    log = audit_log_factory()
    try:
        prompt = _FakeConsentPrompt(consent=True)
        flow = DiagnosticsOfferFlow(
            audit_log=log,
            data_dir=data_dir,
            consent_prompt=prompt,
        )
        outcome = _run(
            flow.run(
                prior_run_id="prev-run",
                prior_started_at=datetime(2024, 1, 1, tzinfo=UTC),
                last_bootstrap_step="init_dialog_manager",
            )
        )
    finally:
        log.close()

    # ``frozen=True`` dataclasses are hashable; we use this to make sure
    # downstream code can put outcomes into sets / dict keys.
    assert hash(outcome) is not None
    assert "DiagnosticsOfferOutcome" in repr(outcome)
    assert isinstance(outcome, DiagnosticsOfferOutcome)


# ---------------------------------------------------------------------------
# Failure-tolerance
# ---------------------------------------------------------------------------


def test_run_when_audit_entries_read_fails_still_writes_report(
    audit_log_factory: Callable[..., AuditLog],
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken ``entries()`` call falls through to an empty tail."""
    log = audit_log_factory()
    try:

        def _boom() -> list[AuditEntry]:
            raise RuntimeError("DB unavailable")

        monkeypatch.setattr(log, "entries", _boom)

        prompt = _FakeConsentPrompt(consent=True)
        flow = DiagnosticsOfferFlow(
            audit_log=log,
            data_dir=data_dir,
            consent_prompt=prompt,
        )
        outcome = _run(
            flow.run(
                prior_run_id="prev-run",
                prior_started_at=datetime(2024, 1, 1, tzinfo=UTC),
                last_bootstrap_step="init_dialog_manager",
            )
        )
    finally:
        log.close()

    assert outcome.consented is True
    assert outcome.report_path is not None
    payload = json.loads(outcome.report_path.read_text(encoding="utf-8"))
    assert payload["audit_tail"] == []
