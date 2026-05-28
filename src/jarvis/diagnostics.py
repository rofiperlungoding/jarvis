"""Crash detection and diagnostics offer flow.

Implements the user-facing half of Requirement 17.4 ("WHEN the JARVIS
application crashes, THE next launch SHALL detect the prior crash and
SHALL offer to submit an anonymized diagnostic report"). The
infrastructure that *detects* the crash — the
``${app.data_dir}/last_run.json`` sentinel and the ``record_crash``
audit row — already lives in :mod:`jarvis.app` and
:class:`jarvis.security.audit_log.AuditLog`. This module layers the
*offer* on top of that infrastructure:

1. A :class:`ConsentPrompt` :class:`typing.Protocol` describing the
   one method the flow needs from its caller — ``ask_consent(prompt)
   -> bool``. The Dialog_Manager (when it lives) implements this by
   composing the TTS engine with the STT engine; tests substitute a
   fake that returns a canned answer. Decoupling here lets the flow
   ship without a hard dependency on the Dialog_Manager and gives
   tests a trivial injection point.
2. A :class:`DiagnosticsReport` dataclass — the redacted bundle that
   would be uploaded to a future telemetry endpoint. Today the bundle
   is *only ever written to disk* under
   ``${app.data_dir}/diagnostics/<timestamp>.json``; no automatic
   upload happens, matching the "anonymized" + "offer" wording in
   Requirement 17.4 and the privacy posture the design takes
   throughout.
3. :class:`DiagnosticsOfferFlow` — the orchestrator. Given a stale
   prior :class:`~jarvis.app.LastRunSentinel`, the audit log, the
   data directory, and a consent prompt:
     * Gathers a redacted report (run id, started_at, the last
       bootstrap step the prior run reached, and a tail of recent
       audit entries with secrets / PII removed),
     * Asks the user via the consent prompt: *"Sir, my last run
       terminated abruptly. Would you like me to compose a
       diagnostics report?"*,
     * On consent, writes the redacted report to disk and records a
       ``crash`` audit entry referencing the report path,
     * On decline (or on a missing prompt), records the decline
       outcome on the same ``crash`` audit row.

Audit semantics
---------------

The ``crash`` audit row is always written — that is the unconditional
half of Requirement 17.4. The *content* of the row distinguishes the
three possible outcomes via the ``outcome`` field:

* ``"prior_run_did_not_shut_down_cleanly:report_written"`` — consent
  was granted and the redacted report exists at
  :attr:`DiagnosticsOfferOutcome.report_path`.
* ``"prior_run_did_not_shut_down_cleanly:declined"`` — the user
  explicitly declined the offer.
* ``"prior_run_did_not_shut_down_cleanly:no_prompt"`` — no consent
  prompt was wired (e.g. headless / first-bootstrap before the TTS
  engine exists). The flow writes the audit row and *does not* write
  a report.

Redaction
---------

The diagnostics report is post-processed by two redactors before it
hits disk:

* :class:`~jarvis.security.log_redaction.LogRedactionFilter` — scrubs
  any registered credential value (Mistral API key, OAuth tokens, ...).
* :class:`~jarvis.memory.redactor.PIIRedactor` — scrubs emails, phone
  numbers, credit-card numbers, and any user-configured pattern.

Both are optional dependencies on the flow constructor; tests can
pass ``None`` for both to inspect the raw shape, but the production
wiring in :mod:`jarvis.app` always supplies them.

Validates: Requirement 17.4
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from jarvis.security.audit_log import AuditEntry, AuditLog
from jarvis.security.log_redaction import LogRedactionFilter

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CONSENT_PROMPT",
    "ConsentPrompt",
    "DiagnosticsOfferFlow",
    "DiagnosticsOfferOutcome",
    "DiagnosticsReport",
]


#: The exact text the design specifies for the consent prompt. Kept as
#: a module-level constant so tests can match on it without hard-coding
#: a copy of the prompt (which would silently drift from the design).
DEFAULT_CONSENT_PROMPT: Final[str] = (
    "Sir, my last run terminated abruptly. "
    "Would you like me to compose a diagnostics report?"
)

#: Maximum number of audit entries embedded in the report. Bounded so
#: a long-running prior session does not produce an unbounded
#: report; 50 is enough to cover the last few turns of activity but
#: small enough to keep the report under a few KB after redaction.
_AUDIT_TAIL_LIMIT: Final[int] = 50

#: Outcome strings written to the ``crash`` audit row. Centralised so
#: tests can match on them without hard-coding the literal in two
#: places.
_OUTCOME_REPORT_WRITTEN: Final[str] = (
    "prior_run_did_not_shut_down_cleanly:report_written"
)
_OUTCOME_DECLINED: Final[str] = "prior_run_did_not_shut_down_cleanly:declined"
_OUTCOME_NO_PROMPT: Final[str] = "prior_run_did_not_shut_down_cleanly:no_prompt"
_OUTCOME_PROMPT_FAILED: Final[str] = "prior_run_did_not_shut_down_cleanly:prompt_failed"


# ---------------------------------------------------------------------------
# Consent prompt
# ---------------------------------------------------------------------------


@runtime_checkable
class ConsentPrompt(Protocol):
    """Minimal interface :class:`DiagnosticsOfferFlow` needs.

    Implementers:

    * The production Dialog_Manager-backed prompt composes TTS (to
      *speak* the question) and STT (to *listen* for the user's
      reply), then maps the reply onto a ``bool``.
    * A notification-only fallback (no audio yet) can speak the
      prompt via the system toast and return ``False`` if no
      response arrives within a short window.
    * Tests substitute a class that returns a canned answer.

    The contract is intentionally narrow: a single async method that
    takes the prompt text and returns a boolean. Anything richer
    (multiple choice, free-text reply) belongs in a higher-level
    abstraction.
    """

    async def ask_consent(self, prompt: str) -> bool:
        """Speak ``prompt`` to the user; return ``True`` for an affirmative reply."""
        ...


# ---------------------------------------------------------------------------
# DiagnosticsReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnosticsReport:
    """Redacted bundle written to disk on user consent.

    Attributes
    ----------
    run_id:
        The run id recorded by the prior run, or ``None`` when the
        sentinel did not include one.
    started_at:
        Wall-clock launch timestamp the prior run wrote, or ``None``.
    last_bootstrap_step:
        The last bootstrap step the prior run successfully recorded,
        or ``None`` when the sentinel was missing or did not include
        one. Useful for distinguishing "the previous run crashed
        before it finished initialising" from "the previous run
        crashed mid-conversation".
    audit_tail:
        A list of redacted audit entries (as JSON-friendly dicts)
        from the prior run, capped at :data:`_AUDIT_TAIL_LIMIT`.
    generated_at:
        Wall-clock timestamp the *current* run produced the report.
    """

    run_id: str | None
    started_at: datetime | None
    last_bootstrap_step: str | None
    audit_tail: tuple[Mapping[str, Any], ...]
    generated_at: datetime
    sentinel_extras: Mapping[str, Any] = field(default_factory=dict)

    def to_json_payload(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        return {
            "run_id": self.run_id,
            "started_at": (self.started_at.isoformat() if self.started_at else None),
            "last_bootstrap_step": self.last_bootstrap_step,
            "generated_at": self.generated_at.isoformat(),
            "sentinel_extras": dict(self.sentinel_extras),
            "audit_tail": [dict(entry) for entry in self.audit_tail],
        }


# ---------------------------------------------------------------------------
# DiagnosticsOfferOutcome
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnosticsOfferOutcome:
    """Result of running :meth:`DiagnosticsOfferFlow.run`.

    Attributes
    ----------
    consented:
        ``True`` when the user explicitly agreed to compose the
        report. ``False`` when the user declined, when no consent
        prompt was wired, or when the consent prompt itself raised an
        exception.
    report_path:
        Filesystem path to the written report, or ``None`` if no
        report was written. Always relative to
        ``${app.data_dir}/diagnostics``.
    audit_entry:
        The :class:`AuditEntry` recorded for this offer flow. Always
        present so callers can correlate the offer with the audit log.
    outcome:
        The exact ``outcome`` string written to the ``crash`` audit
        row. One of the ``_OUTCOME_*`` constants.
    """

    consented: bool
    report_path: Path | None
    audit_entry: AuditEntry
    outcome: str


# ---------------------------------------------------------------------------
# DiagnosticsOfferFlow
# ---------------------------------------------------------------------------


class DiagnosticsOfferFlow:
    """User-facing diagnostics offer for a stale prior-run sentinel.

    Construction is cheap; all real work happens in :meth:`run`.

    Parameters
    ----------
    audit_log:
        The audit log that will receive the ``crash`` row. Required.
    data_dir:
        The application data directory; the flow writes reports
        under ``data_dir / "diagnostics"``. Required.
    consent_prompt:
        The user-facing prompt. Optional — when ``None``, the flow
        records ``no_prompt`` and skips the report write. The
        production wiring in :mod:`jarvis.app` supplies a real
        prompt only after the TTS engine and Dialog_Manager exist;
        early-bootstrap callers (or headless test runs) can omit it.
    log_redaction_filter:
        Scrubs registered credential values from the report.
        Optional but strongly recommended.
    pii_redactor:
        Scrubs configured PII patterns from the report. Optional but
        strongly recommended. The flow only requires that the object
        expose a ``redact(text: str) -> str`` method.
    prompt_text:
        Override the prompt text. Defaults to
        :data:`DEFAULT_CONSENT_PROMPT`.
    audit_tail_limit:
        Maximum number of audit entries embedded in the report.
        Defaults to :data:`_AUDIT_TAIL_LIMIT`.
    """

    def __init__(
        self,
        *,
        audit_log: AuditLog,
        data_dir: Path,
        consent_prompt: ConsentPrompt | None = None,
        log_redaction_filter: LogRedactionFilter | None = None,
        pii_redactor: Any | None = None,
        prompt_text: str = DEFAULT_CONSENT_PROMPT,
        audit_tail_limit: int = _AUDIT_TAIL_LIMIT,
    ) -> None:
        if not isinstance(audit_log, AuditLog):
            raise TypeError("audit_log must be an AuditLog instance")
        if audit_tail_limit < 0:
            raise ValueError("audit_tail_limit must be non-negative")

        self._audit_log: AuditLog = audit_log
        self._data_dir: Path = Path(data_dir)
        self._consent_prompt: ConsentPrompt | None = consent_prompt
        self._log_redaction_filter: LogRedactionFilter | None = log_redaction_filter
        self._pii_redactor: Any | None = pii_redactor
        self._prompt_text: str = prompt_text
        self._audit_tail_limit: int = audit_tail_limit

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def diagnostics_dir(self) -> Path:
        """Directory where reports are written."""
        return self._data_dir / "diagnostics"

    async def run(
        self,
        *,
        prior_run_id: str | None,
        prior_started_at: datetime | None,
        last_bootstrap_step: str | None,
        sentinel_extras: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> DiagnosticsOfferOutcome:
        """Drive the consent prompt and (on consent) write a report.

        Parameters mirror the fields of
        :class:`~jarvis.app.LastRunSentinel`. The orchestrator in
        :mod:`jarvis.app` extracts those fields from the
        already-parsed sentinel and passes them in here so this
        module does not have to import :class:`LastRunSentinel`
        and risk a circular import.

        Returns a :class:`DiagnosticsOfferOutcome` describing what
        happened. The outcome is *also* recorded as a ``crash``
        audit row, so callers that only need the audit-log side of
        Requirement 17.4 can ignore the return value.
        """
        consented = False
        report_path: Path | None = None
        outcome_str: str

        # 1. Ask for consent. A missing prompt or a raised exception
        # both fall through to a recorded-but-unwritten outcome so
        # the audit-log half of Requirement 17.4 is never skipped.
        if self._consent_prompt is None:
            outcome_str = _OUTCOME_NO_PROMPT
            logger.info(
                "diagnostics offer: no consent prompt wired; " "skipping report write"
            )
        else:
            try:
                consented = bool(
                    await self._consent_prompt.ask_consent(self._prompt_text)
                )
            except Exception:
                logger.exception(
                    "diagnostics offer: consent prompt raised; " "treating as decline"
                )
                outcome_str = _OUTCOME_PROMPT_FAILED
            else:
                outcome_str = (
                    _OUTCOME_REPORT_WRITTEN if consented else _OUTCOME_DECLINED
                )

        # 2. On consent, gather the report and write it. Failure to
        # write does not propagate — we still want the audit row.
        if consented:
            try:
                report = self._build_report(
                    prior_run_id=prior_run_id,
                    prior_started_at=prior_started_at,
                    last_bootstrap_step=last_bootstrap_step,
                    sentinel_extras=sentinel_extras,
                    now=now,
                )
                report_path = self._write_report(report)
            except Exception:
                logger.exception(
                    "diagnostics offer: failed to write report; "
                    "audit row will still be emitted"
                )
                outcome_str = _OUTCOME_PROMPT_FAILED
                report_path = None

        # 3. Record the ``crash`` audit row. The ``outcome`` field
        # encodes the three branches above; the ``justification``
        # carries the path to the written report (or a short reason
        # string for the other branches) so an operator reviewing
        # the audit log can find the report directly.
        justification = self._build_justification(
            outcome=outcome_str,
            report_path=report_path,
            prior_run_id=prior_run_id,
            prior_started_at=prior_started_at,
            last_bootstrap_step=last_bootstrap_step,
        )
        audit_entry = await self._audit_log.record_crash(
            outcome=outcome_str,
            justification=justification,
        )

        return DiagnosticsOfferOutcome(
            consented=consented,
            report_path=report_path,
            audit_entry=audit_entry,
            outcome=outcome_str,
        )

    # ------------------------------------------------------------------
    # Report building
    # ------------------------------------------------------------------

    def _build_report(
        self,
        *,
        prior_run_id: str | None,
        prior_started_at: datetime | None,
        last_bootstrap_step: str | None,
        sentinel_extras: Mapping[str, Any] | None,
        now: datetime | None,
    ) -> DiagnosticsReport:
        """Assemble a redacted :class:`DiagnosticsReport`."""
        tail = self._collect_recent_audit_entries(prior_run_id)
        generated_at = now or datetime.now(tz=UTC)
        extras = dict(sentinel_extras) if sentinel_extras else {}
        return DiagnosticsReport(
            run_id=prior_run_id,
            started_at=prior_started_at,
            last_bootstrap_step=last_bootstrap_step,
            audit_tail=tuple(tail),
            generated_at=generated_at,
            sentinel_extras=extras,
        )

    def _collect_recent_audit_entries(
        self, prior_run_id: str | None
    ) -> Sequence[Mapping[str, Any]]:
        """Return up to ``audit_tail_limit`` redacted entries from the prior run.

        We filter by ``run_id`` when possible so the report focuses on
        the prior run's activity. When the prior run did not record a
        run id (sentinel was missing or corrupt), we fall back to the
        most recent entries regardless of run id — those are still
        useful for diagnostics and Requirement 17.4 does not specify
        which run's events the report should describe.
        """
        try:
            entries = self._audit_log.entries()
        except Exception:
            logger.exception(
                "diagnostics offer: could not read audit entries; " "tail will be empty"
            )
            return []

        if prior_run_id is not None:
            filtered = [e for e in entries if e.run_id == prior_run_id]
            # If the prior run produced no audit rows, fall back to
            # the global tail rather than a fully empty list — there
            # may still be operator-relevant context (e.g. a
            # ``record_crash`` row from an even earlier run).
            relevant = filtered or entries
        else:
            relevant = entries

        tail = relevant[-self._audit_tail_limit :]
        return [self._redact_entry(e) for e in tail]

    def _redact_entry(self, entry: AuditEntry) -> Mapping[str, Any]:
        """Convert an :class:`AuditEntry` to a redacted JSON-friendly dict."""
        return {
            "id": entry.id,
            "ts": entry.ts.isoformat() if entry.ts else None,
            "kind": entry.kind,
            "skill": entry.skill,
            "args_json": self._redact_text(entry.args_json),
            "outcome": self._redact_text(entry.outcome),
            "destination": self._redact_text(entry.destination),
            "justification": self._redact_text(entry.justification),
            "run_id": entry.run_id,
        }

    def _redact_text(self, text: str | None) -> str | None:
        """Apply both redactors to ``text`` if configured."""
        if text is None:
            return None
        scrubbed = text
        # Order matters: scrub literal credential values FIRST so the
        # PII redactor never sees an unredacted secret. The credential
        # values are typically long random strings; even though the
        # PII redactor would not match them, defence in depth is
        # cheap.
        if self._log_redaction_filter is not None:
            try:
                scrubbed = self._log_redaction_filter._scrub(scrubbed)
            except Exception:
                logger.exception(
                    "diagnostics offer: log redaction filter raised; "
                    "leaving text unscrubbed"
                )
        if self._pii_redactor is not None:
            try:
                scrubbed = self._pii_redactor.redact(scrubbed)
            except Exception:
                logger.exception(
                    "diagnostics offer: PII redactor raised; " "leaving text unscrubbed"
                )
        return scrubbed

    # ------------------------------------------------------------------
    # Disk I/O
    # ------------------------------------------------------------------

    def _write_report(self, report: DiagnosticsReport) -> Path:
        """Write ``report`` to ``data_dir/diagnostics/<timestamp>.json``.

        The filename is derived from :attr:`DiagnosticsReport.generated_at`
        so two reports written within the same run are easy to
        correlate with the run's wall-clock timeline. Microseconds are
        included to avoid filename collisions when consent is granted
        twice within the same second (e.g. from a stuck-loop test).
        """
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        # Use a filesystem-safe ISO-like timestamp: replace ``:`` with
        # ``-`` and strip the timezone suffix because Windows filenames
        # forbid ``:``. Keep microseconds for collision avoidance.
        filename_stamp = report.generated_at.astimezone(UTC).strftime(
            "%Y%m%dT%H%M%S_%f"
        )
        path = self.diagnostics_dir / f"{filename_stamp}.json"
        payload = report.to_json_payload()
        # ``sort_keys`` keeps the file diff-friendly for operators who
        # want to compare two reports side by side.
        path.write_text(
            json.dumps(payload, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _build_justification(
        *,
        outcome: str,
        report_path: Path | None,
        prior_run_id: str | None,
        prior_started_at: datetime | None,
        last_bootstrap_step: str | None,
    ) -> str:
        """Build the ``justification`` field for the ``crash`` audit row.

        Operators typically read the audit log first, so we make sure
        this field is human-readable and points at the report (when
        one exists).
        """
        parts = [
            f"prior run_id={prior_run_id!r}",
            f"started_at={prior_started_at!s}",
            f"last_bootstrap_step={last_bootstrap_step!r}",
            f"outcome={outcome}",
        ]
        if report_path is not None:
            parts.append(f"report_path={report_path!s}")
        return "; ".join(parts)
