"""Unit tests for ``jarvis.security.log_redaction``.

Covers:
    * Substring scrubbing for plain messages and %-formatted records.
    * Multiple secrets, longest-first replacement (CP11 defence-in-depth:
      a short prefix must not leak the suffix of a longer secret).
    * Empty / non-string registration is rejected or ignored cleanly.
    * Duplicate registration is idempotent.
    * Unregister removes a secret and stops scrubbing it on subsequent records.
    * ``exc_text`` and ``stack_info`` are scrubbed on the record.
    * ``record.args`` is cleared after scrubbing so downstream formatters
      cannot reintroduce the original interpolation.
    * Concurrent registration from multiple threads does not corrupt the
      internal ordering and never leaks a secret on records emitted after
      the registration has returned.
    * ``install_log_redaction_filter`` attaches the filter to both the target
      logger and its existing handlers, and is idempotent under re-install.
    * The filter never suppresses records (always returns truthy).

Validates: Requirements 13.1, 19.3
"""

from __future__ import annotations

from collections.abc import Iterator
import io
import logging
import threading

import pytest

from jarvis.security.log_redaction import (
    DEFAULT_REPLACEMENT,
    LogRedactionFilter,
    install_log_redaction_filter,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_logger_with_buffer(
    name: str, *, redactor: LogRedactionFilter | None = None
) -> tuple[logging.Logger, io.StringIO]:
    """Create an isolated logger writing into an in-memory buffer.

    Each test gets a uniquely-named logger so tests don't interfere with
    one another and so no records leak to the root logger / pytest's
    capture (which would fight with our own assertions).
    """
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    # Bare formatter — we only assert on substring presence/absence, so the
    # default ``%(message)s`` keeps assertions concrete.
    handler.setFormatter(logging.Formatter("%(message)s"))
    log = logging.getLogger(name)
    log.handlers.clear()
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    log.propagate = False  # keep records out of the root logger
    if redactor is not None:
        log.addFilter(redactor)
        handler.addFilter(redactor)
    return log, buf


@pytest.fixture()
def filter_() -> LogRedactionFilter:
    return LogRedactionFilter()


@pytest.fixture()
def unique_logger_name(request: pytest.FixtureRequest) -> Iterator[str]:
    """Generate a unique logger name per test and clean it up afterwards."""
    name = f"jarvis.tests.{request.node.name}"
    yield name
    # Reset to avoid leaking handlers/filters into other tests.
    log = logging.getLogger(name)
    log.handlers.clear()
    log.filters.clear()


# ---------------------------------------------------------------------------
# Basic scrubbing
# ---------------------------------------------------------------------------


def test_default_replacement_is_redacted_marker(filter_: LogRedactionFilter) -> None:
    assert filter_.replacement == DEFAULT_REPLACEMENT
    assert "REDACTED" in DEFAULT_REPLACEMENT


def test_filter_scrubs_registered_secret_substring(
    filter_: LogRedactionFilter, unique_logger_name: str
) -> None:
    secret = "sk-mistral-AAAAAAAAAAAA-1234"
    filter_.register_secret(secret)
    log, buf = _make_logger_with_buffer(unique_logger_name, redactor=filter_)

    log.info("Authorization: Bearer %s for endpoint %s", secret, "https://api.mistral.ai")

    output = buf.getvalue()
    assert secret not in output
    assert filter_.replacement in output
    assert "https://api.mistral.ai" in output  # non-secret args remain intact


def test_filter_does_nothing_when_no_secrets_registered(
    filter_: LogRedactionFilter, unique_logger_name: str
) -> None:
    log, buf = _make_logger_with_buffer(unique_logger_name, redactor=filter_)
    log.info("plain message with no secrets")
    assert "plain message with no secrets" in buf.getvalue()


def test_filter_scrubs_plain_string_message(
    filter_: LogRedactionFilter, unique_logger_name: str
) -> None:
    secret = "abcdef-1234567890"
    filter_.register_secret(secret)
    log, buf = _make_logger_with_buffer(unique_logger_name, redactor=filter_)

    log.warning(f"raw message containing {secret} mid-sentence")

    out = buf.getvalue()
    assert secret not in out
    assert filter_.replacement in out


def test_filter_scrubs_multiple_occurrences(
    filter_: LogRedactionFilter, unique_logger_name: str
) -> None:
    secret = "TOKEN-XYZ"
    filter_.register_secret(secret)
    log, buf = _make_logger_with_buffer(unique_logger_name, redactor=filter_)

    log.info("first %s, second %s, third %s", secret, secret, secret)

    out = buf.getvalue()
    assert secret not in out
    assert out.count(filter_.replacement) == 3


def test_filter_returns_true_so_records_are_emitted(filter_: LogRedactionFilter) -> None:
    record = logging.LogRecord(
        name="x", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hello", args=(), exc_info=None,
    )
    assert filter_.filter(record) is True


def test_args_are_cleared_after_scrub_to_prevent_reformatting(
    filter_: LogRedactionFilter,
) -> None:
    secret = "very-secret-value"
    filter_.register_secret(secret)
    record = logging.LogRecord(
        name="x", level=logging.INFO, pathname=__file__, lineno=1,
        msg="key=%s", args=(secret,), exc_info=None,
    )
    filter_.filter(record)
    # After filtering the record, getMessage() must NOT return the original
    # secret regardless of how many times it is called.
    assert secret not in record.getMessage()
    assert record.args in ((), None)
    # Idempotent: a second call to getMessage() must also produce a sanitised
    # output (this is the property that broke if `args` weren't cleared).
    assert secret not in record.getMessage()


# ---------------------------------------------------------------------------
# Longest-first replacement and overlap semantics
# ---------------------------------------------------------------------------


def test_longer_secret_redacted_before_shorter_substring(
    unique_logger_name: str,
) -> None:
    """If a short secret is a prefix of a longer one, the longer one wins.

    Without longest-first replacement, registering "sk" would shadow
    "sk-LIVE-ABC", leaving "sk-LIVE-ABC" partially un-redacted.
    """
    short = "sk"
    long = "sk-LIVE-ABCDEFG-XYZ"
    redactor = LogRedactionFilter()
    redactor.register_secret(short)
    redactor.register_secret(long)
    log, buf = _make_logger_with_buffer(unique_logger_name, redactor=redactor)

    log.info("token=%s", long)

    out = buf.getvalue()
    assert long not in out
    # The dangerous tail "LIVE-ABCDEFG-XYZ" must never leak.
    assert "LIVE" not in out
    assert "ABCDEFG" not in out


def test_secret_inserted_inside_url_query_is_scrubbed(
    unique_logger_name: str,
) -> None:
    redactor = LogRedactionFilter()
    secret = "tok_abcdefghijklmno"
    redactor.register_secret(secret)
    log, buf = _make_logger_with_buffer(unique_logger_name, redactor=redactor)

    log.info("GET https://example.invalid/api?token=%s&q=hi", secret)

    out = buf.getvalue()
    assert secret not in out
    assert "https://example.invalid/api?token=" in out
    assert "&q=hi" in out


# ---------------------------------------------------------------------------
# Registration API
# ---------------------------------------------------------------------------


def test_empty_string_secret_is_ignored(filter_: LogRedactionFilter) -> None:
    assert filter_.register_secret("") is False
    assert filter_.registered_secret_count() == 0


def test_duplicate_registration_is_idempotent(filter_: LogRedactionFilter) -> None:
    assert filter_.register_secret("abc") is True
    assert filter_.register_secret("abc") is False
    assert filter_.registered_secret_count() == 1


def test_register_secrets_bulk_counts_only_new(filter_: LogRedactionFilter) -> None:
    added = filter_.register_secrets(["a-secret", "b-secret", "a-secret", "", "c-secret"])
    assert added == 3
    assert filter_.registered_secret_count() == 3


def test_register_non_string_raises(filter_: LogRedactionFilter) -> None:
    with pytest.raises(TypeError):
        filter_.register_secret(12345)  # type: ignore[arg-type]


def test_unregister_secret_stops_scrubbing(unique_logger_name: str) -> None:
    redactor = LogRedactionFilter()
    secret = "rotated-secret-1"
    redactor.register_secret(secret)
    log, buf = _make_logger_with_buffer(unique_logger_name, redactor=redactor)

    log.info("first emit: %s", secret)
    assert secret not in buf.getvalue()

    assert redactor.unregister_secret(secret) is True
    # Re-using a logger that has already buffered output: clear the buffer.
    buf.truncate(0)
    buf.seek(0)
    log.info("second emit: %s", secret)
    # The secret has been rotated/removed; subsequent records contain it
    # verbatim. The caller is responsible for deciding when this is safe.
    assert secret in buf.getvalue()


def test_unregister_unknown_secret_returns_false(filter_: LogRedactionFilter) -> None:
    assert filter_.unregister_secret("never-registered") is False


def test_clear_removes_all_secrets(filter_: LogRedactionFilter) -> None:
    filter_.register_secrets(["one", "two", "three"])
    assert filter_.registered_secret_count() == 3
    filter_.clear()
    assert filter_.registered_secret_count() == 0


def test_constructor_accepts_initial_secrets(unique_logger_name: str) -> None:
    redactor = LogRedactionFilter(secrets=["alpha-secret", "beta-secret"])
    assert redactor.registered_secret_count() == 2
    log, buf = _make_logger_with_buffer(unique_logger_name, redactor=redactor)
    log.info("payload=%s", "alpha-secret")
    assert "alpha-secret" not in buf.getvalue()


def test_constructor_rejects_non_string_replacement() -> None:
    with pytest.raises(TypeError):
        LogRedactionFilter(replacement=123)  # type: ignore[arg-type]


def test_custom_replacement_is_used(unique_logger_name: str) -> None:
    redactor = LogRedactionFilter(replacement="<<HIDDEN>>")
    redactor.register_secret("the-secret")
    log, buf = _make_logger_with_buffer(unique_logger_name, redactor=redactor)
    log.info("here it is: the-secret")
    assert "the-secret" not in buf.getvalue()
    assert "<<HIDDEN>>" in buf.getvalue()


# ---------------------------------------------------------------------------
# Exception text / stack info
# ---------------------------------------------------------------------------


def test_exc_text_is_scrubbed(filter_: LogRedactionFilter) -> None:
    secret = "leaked-traceback-token"
    filter_.register_secret(secret)
    record = logging.LogRecord(
        name="x", level=logging.ERROR, pathname=__file__, lineno=1,
        msg="boom", args=(), exc_info=None,
    )
    record.exc_text = (
        "Traceback (most recent call last):\n"
        f"  File ..., line ...: KeyError: '{secret}'\n"
    )
    filter_.filter(record)
    assert secret not in (record.exc_text or "")
    assert filter_.replacement in (record.exc_text or "")


def test_stack_info_is_scrubbed(filter_: LogRedactionFilter) -> None:
    secret = "frame-locals-leak"
    filter_.register_secret(secret)
    record = logging.LogRecord(
        name="x", level=logging.WARNING, pathname=__file__, lineno=1,
        msg="msg", args=(), exc_info=None,
    )
    record.stack_info = f"Stack (most recent call last):\n  ... value={secret} ...\n"
    filter_.filter(record)
    assert secret not in (record.stack_info or "")


def test_logger_exception_path_emits_redacted_traceback(
    unique_logger_name: str,
) -> None:
    secret = "sk-exception-AAAAAAAAAA"
    redactor = LogRedactionFilter()
    redactor.register_secret(secret)
    log, buf = _make_logger_with_buffer(unique_logger_name, redactor=redactor)

    try:
        raise ValueError(f"the api key was {secret}")
    except ValueError:
        log.exception("request failed")

    output = buf.getvalue()
    assert secret not in output
    assert "request failed" in output


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_filter_survives_bad_format_args(filter_: LogRedactionFilter) -> None:
    """A misformatted record (e.g. wrong number of %-args) must not crash."""
    filter_.register_secret("doesnt-matter")
    record = logging.LogRecord(
        name="x", level=logging.INFO, pathname=__file__, lineno=1,
        msg="this %s requires one arg", args=(), exc_info=None,
    )
    # Filter should return True without raising; downstream formatters might
    # still complain, but the redactor's job is to be best-effort and never
    # the proximate cause of a logging failure.
    assert filter_.filter(record) is True


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_registration_does_not_corrupt_ordering() -> None:
    """Concurrently registering and reading from the filter is safe.

    The contract: any secret returned from a successful ``register_secret``
    call is redacted on every record whose ``filter`` call begins
    *after* the register call returns. Records in flight at the moment of
    registration may not yet see the new secret — that race is documented.
    """
    redactor = LogRedactionFilter()
    secrets = [f"secret-{i:04d}-payload" for i in range(200)]

    def producer(values: list[str]) -> None:
        for v in values:
            redactor.register_secret(v)

    # Two producer threads register disjoint halves concurrently.
    half = len(secrets) // 2
    t1 = threading.Thread(target=producer, args=(secrets[:half],))
    t2 = threading.Thread(target=producer, args=(secrets[half:],))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert redactor.registered_secret_count() == len(secrets)

    # After both threads have joined, every secret must be redacted.
    for s in secrets:
        record = logging.LogRecord(
            name="x", level=logging.INFO, pathname=__file__, lineno=1,
            msg="value=%s", args=(s,), exc_info=None,
        )
        redactor.filter(record)
        assert s not in record.getMessage()


def test_concurrent_filter_calls_do_not_interleave_writes() -> None:
    redactor = LogRedactionFilter()
    secret = "shared-secret-VALUE-XYZ"
    redactor.register_secret(secret)

    errors: list[str] = []

    def emitter() -> None:
        for _ in range(50):
            record = logging.LogRecord(
                name="x", level=logging.INFO, pathname=__file__, lineno=1,
                msg="payload contains %s and %s", args=(secret, secret),
                exc_info=None,
            )
            redactor.filter(record)
            msg = record.getMessage()
            if secret in msg:
                errors.append(msg)

    threads = [threading.Thread(target=emitter) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []


# ---------------------------------------------------------------------------
# install_log_redaction_filter
# ---------------------------------------------------------------------------


def test_install_attaches_filter_to_logger_and_handlers(
    unique_logger_name: str,
) -> None:
    log = logging.getLogger(unique_logger_name)
    handler = logging.StreamHandler(io.StringIO())
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    log.propagate = False

    redactor = install_log_redaction_filter(logger_name=unique_logger_name)

    assert redactor in log.filters
    assert redactor in handler.filters


def test_install_creates_filter_when_none_supplied() -> None:
    redactor = install_log_redaction_filter(logger_name="jarvis.tests.install_default")
    try:
        assert isinstance(redactor, LogRedactionFilter)
    finally:
        log = logging.getLogger("jarvis.tests.install_default")
        log.removeFilter(redactor)


def test_install_reuses_supplied_filter(unique_logger_name: str) -> None:
    pre_existing = LogRedactionFilter()
    pre_existing.register_secret("alpha")
    returned = install_log_redaction_filter(pre_existing, logger_name=unique_logger_name)
    assert returned is pre_existing


def test_install_is_idempotent(unique_logger_name: str) -> None:
    redactor = LogRedactionFilter()
    log = logging.getLogger(unique_logger_name)
    log.propagate = False

    install_log_redaction_filter(redactor, logger_name=unique_logger_name)
    install_log_redaction_filter(redactor, logger_name=unique_logger_name)

    assert log.filters.count(redactor) == 1


def test_install_redacts_subsequently_registered_secret(
    unique_logger_name: str,
) -> None:
    """The end-to-end happy path: install → register → emit → no leak."""
    log = logging.getLogger(unique_logger_name)
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    log.propagate = False

    redactor = install_log_redaction_filter(logger_name=unique_logger_name)
    secret = "post-install-token-Z"
    redactor.register_secret(secret)

    log.info("authorization=%s", secret)
    out = buf.getvalue()
    assert secret not in out
    assert redactor.replacement in out
