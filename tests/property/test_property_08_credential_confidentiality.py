"""Property 8 — Credential confidentiality.

From ``design.md §Correctness Properties``:

    *For any* secret value ``V`` written via ``CredentialStore.set(name, V)``,
    *for every* file ``F`` under the application data directory other than
    the credential blob, and *for every* line in the audit log, the byte
    sequence ``V.encode("utf-8")`` SHALL NOT appear as a substring.

This file expresses that universal quantification with Hypothesis. The
strategy generates plausible API-key-shaped secret values, stores each
one through a real :class:`CredentialStore` (backed by :class:`NullDPAPI`
so the test works on non-Windows CI runners), wires up a real
:class:`LogRedactionFilter` against a :class:`logging.FileHandler` rooted
inside the data directory, exercises every distinct audit-emitting flow
on a real :class:`AuditLog` SQLite file, then walks both the on-disk
file tree and the persisted audit rows asserting that the secret's
UTF-8 byte sequence never appears as a substring.

Three system surfaces are exercised per example:

* **Credential storage.** :meth:`CredentialStore.set` is the primary
  write path. With :class:`NullDPAPI` the on-disk blob is XOR-obfuscated
  rather than encrypted, but the obfuscation is sufficient that the
  literal plaintext does not appear in the blob (the unit tests in
  ``tests/unit/security/test_credential_store.py`` carry the matching
  per-example assertion). The credential blob is the *only* permitted
  location of ``V`` per CP11; the property explicitly excludes it from
  the file scan.

* **Logging.** A :class:`LogRedactionFilter` is registered with the
  generated secret and attached to a :class:`logging.FileHandler` whose
  log file lives directly under the data directory. Several log
  records — including a :meth:`logging.Logger.exception` call whose
  traceback embeds the secret — are emitted to verify the filter
  scrubs both ``record.msg`` and ``record.exc_text`` before the
  handler writes the line out.

* **Audit log.** Every public ``record_*`` method on :class:`AuditLog`
  is invoked with realistic non-secret arguments. The persisted rows
  are then read back via :meth:`AuditLog.entries` (the synchronous
  iteration helper called out in the task description) and each row is
  serialised to a single JSON line for the substring check. The audit
  log MUST NOT contain ``V`` because the production code never passes
  secrets into ``args_json`` / ``outcome`` / ``destination`` /
  ``justification`` fields.

Why a distinctive secret prefix?
--------------------------------

The audit log writes structured fields (``"executed"``, ``"ok"``,
``"prop8-test"``, ``"WeatherSkill"``, ...) whose byte content is
fixed across examples. A secret generated as a short alphanumeric
string can — by sheer chance — be a substring of one of those literal
fields, producing a Hypothesis falsifying example that has nothing to
do with a real confidentiality regression. We prefix every generated
secret with ``jarvis-prop8-secret-`` so coincidental matches are
impossible.

Validates: Requirements 13.1, 19.3 (CP11)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Final
from uuid import uuid4

from hypothesis import HealthCheck, given, settings, strategies as st
import pytest

from jarvis.security.audit_log import AuditLog
from jarvis.security.credential_store import CredentialStore
from jarvis.security.dpapi import NullDPAPI
from jarvis.security.log_redaction import LogRedactionFilter

# ---------------------------------------------------------------------------
# Secret strategy
# ---------------------------------------------------------------------------


# Distinctive prefix that cannot occur inside any structured audit field
# emitted by the test below. The prefix is long enough (20 bytes) that
# accidental collision with a path component or a SQLite header byte is
# vanishingly unlikely. Every generated secret therefore satisfies the
# per-example precondition "the secret's bytes are not a substring of
# the audit log's *invariant* content" — leaving Hypothesis free to
# search the space of *system-induced* leaks.
_SECRET_PREFIX: Final[str] = "jarvis-prop8-secret-"


@st.composite
def _secret_values(draw: st.DrawFn) -> str:
    """Generate API-key-shaped secret strings.

    The body uses printable ASCII excluding the JSON-special characters
    ``"`` and ``\\``. JSON-escaping ``"`` to ``\\"`` would still leave
    the original ``"`` byte in the encoded string (so a substring
    search would still match), but excluding it keeps the equivalence
    between the secret's UTF-8 bytes and the persisted form simpler to
    reason about — when the assertion fires, the failing example is
    immediately readable rather than obscured by escape sequences.
    """
    body = draw(
        st.text(
            alphabet=st.characters(
                # Printable ASCII range: ``!`` (0x21) through ``~`` (0x7E).
                min_codepoint=0x21,
                max_codepoint=0x7E,
                # Exclude JSON-escapable characters and the path
                # separator. None of these would actually break the
                # property — just the failing-example readability.
                exclude_characters='"\\',  # type: ignore[arg-type]
            ),
            min_size=8,
            max_size=48,
        )
    )
    return _SECRET_PREFIX + body


# Closed set of representative credential names from
# ``design.md §Credential_Store``. Sampling lets Hypothesis exercise the
# percent-encoded-filename path (``mistral/api_key`` →
# ``mistral%2Fapi_key.bin``) and the multi-segment naming convention.
_CREDENTIAL_NAMES: Final[tuple[str, ...]] = (
    "mistral/api_key",
    "weather/api_key",
    "news/api_key",
    "email/smtp_password",
    "calendar/google/refresh_token",
)


# ---------------------------------------------------------------------------
# Audit-emitting flow
# ---------------------------------------------------------------------------


async def _exercise_audit_flows(audit_path: object) -> None:
    """Emit one row of every public :class:`AuditLog` ``record_*`` kind.

    The argument shapes mirror what the production code paths actually
    pass in (e.g., :class:`SendEmailSkill` arguments,
    :class:`WeatherSkill` network destinations). Critically, none of
    the calls receive the generated secret — the property test verifies
    that legitimate audit traffic never carries the secret, while the
    earlier logger emissions verify the redaction filter holds even
    when caller code accidentally tries to log the secret.
    """
    audit = AuditLog(audit_path, run_id="prop8-test-run")  # type: ignore[arg-type]
    try:
        await audit.record_confirmation_requested(
            skill="SendEmailSkill",
            args_json={
                "recipient": "alice@example.com",
                "subject": "Status update",
                "body": "Hello",
            },
        )
        await audit.record_executed(
            skill="SendEmailSkill",
            args_json={
                "recipient": "alice@example.com",
                "subject": "Status update",
                "body": "Hello",
            },
            outcome="ok",
        )
        await audit.record_denied(
            skill="RunScriptSkill",
            args_json={"script_id": "deploy"},
            outcome="user-denied",
        )
        await audit.record_network_egress(
            destination="api.openweathermap.org",
            justification="WeatherSkill request",
            skill="WeatherSkill",
            outcome="http-200",
        )
        await audit.record_policy_violation(
            skill="ReadFileSkill",
            justification="path outside sandbox",
            args_json={"path": "/etc/passwd"},
        )
        await audit.record_error(
            skill="MistralBackend",
            outcome="rate_limited",
            justification="HTTP 429 from upstream",
        )
        await audit.record_crash(
            outcome="last_run_stale",
            justification="sentinel timestamp was older than grace window",
        )
    finally:
        audit.close()


def _make_logger_with_file_handler(
    *,
    log_path: object,
    redactor: LogRedactionFilter,
) -> tuple[logging.Logger, logging.FileHandler]:
    """Return a private logger writing redacted records to ``log_path``.

    The logger name is suffixed with a fresh UUID per call so concurrent
    Hypothesis examples cannot share state through the global logger
    registry. ``propagate=False`` keeps the records from bubbling up to
    pytest's capture handler, which would otherwise see the un-redacted
    pre-filter view of the records.
    """
    logger_name = f"jarvis.tests.prop8.{uuid4().hex}"
    log = logging.getLogger(logger_name)
    log.propagate = False
    log.setLevel(logging.DEBUG)
    # Defensive cleanup: a previous example's handlers (if any) are
    # closed and removed so they cannot intercept our records.
    for stale in list(log.handlers):
        try:
            stale.close()
        finally:
            log.removeHandler(stale)
    handler = logging.FileHandler(str(log_path), encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.addFilter(redactor)
    log.addHandler(handler)
    return log, handler


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@given(
    secret=_secret_values(),
    cred_name=st.sampled_from(_CREDENTIAL_NAMES),
)
@settings(
    # Inherits ``max_examples=200`` / ``deadline=None`` from the
    # ``jarvis`` Hypothesis profile (see ``tests/conftest.py``). The
    # function-scoped-fixture suppression mirrors Property 9 / 7 — we
    # use ``tmp_path_factory.mktemp`` to give each example a unique
    # directory, which is safe to call repeatedly inside a Hypothesis
    # generated body.
    suppress_health_check=(
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ),
)
def test_property_08_credential_confidentiality(
    tmp_path_factory: pytest.TempPathFactory,
    secret: str,
    cred_name: str,
) -> None:
    """``V.encode("utf-8")`` does not appear in any non-credential file or audit row.

    **Validates: Requirements 13.1, 19.3 (CP11)**
    """

    # ---- Per-example data directory -----------------------------------
    data_dir = tmp_path_factory.mktemp("prop8-data-")
    secrets_dir = data_dir / "secrets"
    audit_path = data_dir / "audit.sqlite"
    log_path = data_dir / "jarvis.log"

    # ---- 1. Store the secret via CredentialStore ----------------------
    # ``NullDPAPI`` keeps the storage path identical to production while
    # avoiding the Windows-only ``win32crypt`` dependency in CI. Its
    # XOR keystream guarantees the literal plaintext does not appear
    # inside the on-disk blob (verified separately by the unit tests in
    # ``tests/unit/security/test_credential_store.py``); we still
    # *exclude* the credential blob from the file scan below to honour
    # the design's wording (CP11 quantifies over files "other than the
    # credential blob").
    dpapi = NullDPAPI(suppress_warning=True)
    cred_store = CredentialStore(secrets_dir, dpapi)
    cred_store.set(cred_name, secret)
    # Regression guard: the round-trip works (and therefore the test
    # is checking confidentiality of a real, recoverable secret rather
    # than a value that silently failed to persist).
    assert cred_store.get(cred_name) == secret, (
        "CredentialStore did not round-trip the generated secret; "
        "the rest of the property is vacuous."
    )

    # ---- 2. Wire the redaction filter into a real FileHandler ---------
    redactor = LogRedactionFilter()
    redactor.register_secret(secret)
    log, handler = _make_logger_with_file_handler(
        log_path=log_path, redactor=redactor
    )

    # ---- 3. Emit log records that *would* leak the secret without --
    # the filter. Every record below mentions the secret somewhere; the
    # filter must scrub it before the FileHandler renders the line to
    # disk. The :meth:`logger.exception` call is the most stringent
    # case because the traceback string is rendered separately and is
    # only redacted when :class:`LogRedactionFilter.filter` populates
    # ``record.exc_text`` ahead of the formatter (see
    # ``log_redaction.py``).
    try:
        log.info("Provider authenticated with token: %s", secret)
        log.warning("Authorization header: Bearer %s", secret)
        log.debug("Raw config payload: %s", {"api_key": secret})
        try:
            raise RuntimeError(f"upstream rejected key {secret!r}")
        except RuntimeError:
            log.exception("upstream call failed; investigate")
    finally:
        # Ensure every byte hits disk before we walk the file tree, and
        # that the handler releases its file lock so the inode can be
        # removed cleanly on Windows when ``tmp_path_factory`` cleans up.
        handler.flush()
        handler.close()
        log.removeHandler(handler)

    # ---- 4. Exercise the audit-emitting flows -------------------------
    asyncio.run(_exercise_audit_flows(audit_path))

    # ---- 5. Assertion 1: secret is not in any non-credential file ----
    secret_bytes = secret.encode("utf-8")
    canonical_secrets_dir = secrets_dir.resolve()
    for entry in data_dir.rglob("*"):
        if not entry.is_file():
            continue
        # Exclude the credential blob (the sole permitted location of
        # ``V`` per CP11 / Requirement 13.1). Any ``.bin`` file directly
        # inside ``secrets/`` is a credential blob produced by
        # :class:`CredentialStore.set` — that is the production layout.
        try:
            is_credential_blob = (
                entry.parent.resolve() == canonical_secrets_dir
                and entry.suffix == ".bin"
            )
        except OSError:
            # A vanished file in the middle of an iteration is harmless
            # — there is nothing to read, so nothing can leak.
            continue
        if is_credential_blob:
            continue
        try:
            body = entry.read_bytes()
        except OSError:
            continue
        assert secret_bytes not in body, (
            f"secret leaked into non-credential file {entry}: "
            f"file_size={len(body)}, secret_prefix={secret[:24]!r}"
        )

    # ---- 6. Assertion 2: secret is not in any audit log row ---------
    # Re-open the audit log read-only-style. The synchronous
    # :meth:`AuditLog.entries` iterator returns every persisted row
    # in strict id order (Property 6 / CP9 holds across this iteration
    # as well, but is not the property under test here).
    audit = AuditLog(audit_path, run_id="prop8-readback-run")
    try:
        rows = audit.entries()
        # Defensive sanity: we just wrote seven rows. If this drops to
        # zero, the audit-emission path silently failed and the
        # substring assertion would pass vacuously — exactly the
        # failure mode CP9's adjacent property test guards against.
        assert len(rows) >= 7, (
            f"audit log must record every emitted row; got {len(rows)} "
            f"row(s) of kinds {[e.kind for e in rows]}"
        )
        for row in rows:
            # Render every persisted field into a single canonical line.
            # ``ensure_ascii=False`` means UTF-8 bytes round-trip
            # verbatim — a non-ASCII byte sequence in the secret would
            # otherwise be JSON-escaped to ``\uXXXX`` and the substring
            # search would miss the leak.
            line = json.dumps(
                {
                    "id": row.id,
                    "ts": row.ts.isoformat(),
                    "kind": row.kind,
                    "skill": row.skill,
                    "args_json": row.args_json,
                    "outcome": row.outcome,
                    "destination": row.destination,
                    "justification": row.justification,
                    "run_id": row.run_id,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            line_bytes = line.encode("utf-8")
            assert secret_bytes not in line_bytes, (
                f"secret leaked into audit row id={row.id} "
                f"kind={row.kind!r}: {line!r}"
            )
    finally:
        audit.close()
