"""Logging filter that scrubs registered secret substrings before emission.

This module implements the redacting log filter described in
``design.md §Credential_Store``::

    A redacting log filter installs a ``logging.Filter`` that scrubs any
    string equal to a known credential value before it is emitted, defending
    CP11 across the entire process.

The filter intentionally redacts *substrings*, not just whole-string matches,
so that messages such as ``"Mistral request failed: Authorization: Bearer
sk-XXXX..."`` are sanitised before reaching any handler — formatter, stream,
file, syslog, or HTTP shipping. Both the message itself and any cached
exception traceback (``record.exc_text``) and stack info (``record.stack_info``)
are scrubbed.

Typical wiring (performed by ``app.py`` at startup, task 19.1)::

    redactor = install_log_redaction_filter()
    # ... after CredentialStore is unlocked ...
    redactor.register_secret(credential_store.get("mistral/api_key"))

Validates: Requirements 13.1, 19.3
"""

from __future__ import annotations

from collections.abc import Iterable
import logging
import threading
from typing import Final

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_REPLACEMENT",
    "LogRedactionFilter",
    "install_log_redaction_filter",
]


#: Default placeholder substituted in place of a redacted secret. Chosen to be
#: long, distinctive, and unlikely to collide with legitimate log content.
DEFAULT_REPLACEMENT: Final[str] = "[REDACTED]"


class LogRedactionFilter(logging.Filter):
    """A :class:`logging.Filter` that scrubs registered secret substrings.

    The filter holds a set of secret string values. Before each
    :class:`logging.LogRecord` is emitted, the filter:

    1. Resolves the record's message via :meth:`logging.LogRecord.getMessage`,
       so ``%``-style ``args`` are interpolated into the final string.
    2. Replaces every occurrence of any registered secret with
       :attr:`replacement`, processing longer secrets first so that a short
       secret which is a prefix or substring of a longer one cannot leak the
       remaining tail of the longer secret.
    3. Writes the sanitised message back into ``record.msg`` and clears
       ``record.args`` so downstream formatters emit the redacted text
       verbatim instead of re-applying the original ``%`` substitution.
    4. Scrubs ``record.exc_text`` and ``record.stack_info`` if those
       attributes are populated, so secrets that appear inside tracebacks
       (e.g. raised by an HTTP client that echoes a request header) cannot
       leak through ``logger.exception(...)`` calls.

    Concurrency:
        :meth:`register_secret`, :meth:`register_secrets`,
        :meth:`unregister_secret`, :meth:`clear`, and the per-record scrub
        loop are guarded by a :class:`threading.RLock`. Secrets may be
        registered or unregistered from any thread while logging is in
        progress; the worst-case race is that a secret added *after* the
        scrub-loop snapshot will not be redacted on the in-flight record,
        but will be redacted on every subsequent record.

    Caller contract:
        The caller is responsible for the values it registers. Registering
        a very short or very common substring (e.g. ``"a"``) will redact
        far more text than intended. In practice the registered values are
        long-form secrets such as Mistral API keys and OAuth tokens, sourced
        from :class:`~jarvis.security.credential_store.CredentialStore`.

    Args:
        replacement: The placeholder substituted in place of each secret.
            Defaults to :data:`DEFAULT_REPLACEMENT`.
        secrets: Optional iterable of secrets to register at construction.
        name: Optional logger-name filter passed through to
            :class:`logging.Filter`. The default empty string matches every
            logger, which is what we want for a process-wide redactor.
    """

    def __init__(
        self,
        *,
        replacement: str = DEFAULT_REPLACEMENT,
        secrets: Iterable[str] | None = None,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        if not isinstance(replacement, str):
            raise TypeError("replacement must be a str")
        self._replacement = replacement
        # Held as a list sorted by length descending so replacement is
        # deterministic and longest-first. Set semantics are enforced by
        # `register_secret`.
        self._secrets: list[str] = []
        self._lock = threading.RLock()
        if secrets is not None:
            self.register_secrets(secrets)

    # ------------------------------------------------------------------
    # Registration API
    # ------------------------------------------------------------------

    def register_secret(self, value: str) -> bool:
        """Register a single secret value.

        Empty strings are silently ignored (returning ``False``) because they
        would otherwise cause the scrub loop to insert the replacement string
        between every character of every log message.

        Returns:
            ``True`` if the secret was newly added; ``False`` if it was empty
            or already registered.
        """
        if not isinstance(value, str):
            raise TypeError("secret must be a str")
        if value == "":
            return False
        with self._lock:
            if value in self._secrets:
                return False
            self._secrets.append(value)
            # `sort` is stable and `reverse=True` orders by length descending.
            self._secrets.sort(key=len, reverse=True)
            return True

    def register_secrets(self, values: Iterable[str]) -> int:
        """Register many secrets at once.

        Returns:
            The number of secrets newly added (excluding empties and
            duplicates).
        """
        added = 0
        for value in values:
            if self.register_secret(value):
                added += 1
        return added

    def unregister_secret(self, value: str) -> bool:
        """Remove a previously-registered secret.

        Returns:
            ``True`` if the secret was registered and has been removed;
            ``False`` if it was not registered.
        """
        if not isinstance(value, str):
            raise TypeError("secret must be a str")
        with self._lock:
            try:
                self._secrets.remove(value)
            except ValueError:
                return False
            return True

    def clear(self) -> None:
        """Remove every registered secret."""
        with self._lock:
            self._secrets.clear()

    @property
    def replacement(self) -> str:
        """The placeholder string substituted in place of each secret."""
        return self._replacement

    def registered_secret_count(self) -> int:
        """Return the count of currently registered secrets.

        Provided for diagnostics and tests. Does NOT expose the secrets
        themselves — there is no public accessor for the set of registered
        values, by design.
        """
        with self._lock:
            return len(self._secrets)

    # ------------------------------------------------------------------
    # logging.Filter implementation
    # ------------------------------------------------------------------

    def _scrub(self, text: str) -> str:
        """Return ``text`` with every registered secret replaced."""
        if not text:
            return text
        # Snapshot under the lock so concurrent registration during a long
        # scrub cannot interleave a partial sort. The snapshot is a tuple of
        # already-sorted (longest-first) secrets, so iteration order is
        # well-defined.
        with self._lock:
            secrets = tuple(self._secrets)
        for secret in secrets:
            # Re-checked because secrets may have been registered with
            # `register_secret` from another thread; an empty string would
            # never be present (filtered above) but defence-in-depth is cheap.
            if secret and secret in text:
                text = text.replace(secret, self._replacement)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        # Resolve %-format args first so embedded secrets in argument values
        # are scrubbed alongside literal-message secrets. After scrubbing we
        # write the result back as a plain msg with no args so downstream
        # formatters do not re-introduce the original substitutions.
        try:
            formatted = record.getMessage()
        except Exception:
            # If formatting fails (e.g. mismatched %-args), fall back to the
            # raw msg coerced to str so we still scrub something reasonable
            # rather than dropping the record outright. Logging is best-effort.
            formatted = str(record.msg)

        scrubbed = self._scrub(formatted)
        if scrubbed != formatted:
            record.msg = scrubbed
            record.args = ()

        # If the record carries unrendered exception info (the common
        # ``logger.exception(...)`` path), pre-render the traceback here so
        # the cached ``exc_text`` is populated and can be scrubbed BEFORE any
        # downstream :class:`logging.Formatter` would otherwise render it
        # itself. Without this, the formatter calls
        # :meth:`logging.Formatter.formatException` after our filter has run,
        # producing an un-scrubbed traceback string.
        if record.exc_info and not record.exc_text:
            try:
                record.exc_text = logging.Formatter().formatException(record.exc_info)
            except Exception:
                # Defensive: never let traceback formatting failures suppress
                # the underlying log record. The original ``exc_info`` remains
                # available for downstream handlers if they choose to retry.
                record.exc_text = None

        # Cached exception traceback text — either populated above or by an
        # earlier filter / formatter / caller that pre-rendered exceptions.
        # Scrub so every subsequent handler sees the redacted version.
        exc_text = getattr(record, "exc_text", None)
        if isinstance(exc_text, str) and exc_text:
            record.exc_text = self._scrub(exc_text)

        # stack_info, if attached, is a multi-line string from
        # traceback.print_stack — scrub it for the same reason.
        stack_info = getattr(record, "stack_info", None)
        if isinstance(stack_info, str) and stack_info:
            record.stack_info = self._scrub(stack_info)

        # logging.Filter convention: returning truthy permits the record to
        # be emitted. We never suppress records — only sanitise them.
        return True


def install_log_redaction_filter(
    filter_: LogRedactionFilter | None = None,
    *,
    logger_name: str | None = None,
) -> LogRedactionFilter:
    """Attach a :class:`LogRedactionFilter` to a logger and its handlers.

    Intended to be called once during application startup (see ``app.py``
    bootstrap, task 19.1) before any third-party library can produce log
    output, so every emitted record is scrubbed regardless of which logger
    created it.

    Implementation note:
        Python's logging documentation specifies that filters attached to a
        :class:`logging.Logger` only apply to records originated at that
        logger; records propagated up from child loggers do *not* re-run
        the ancestor's logger-level filters. To cover both the
        originated-here case and the propagated-here case, we add the
        filter to both the target logger AND every handler currently
        attached to it. Handlers added after this call are the
        responsibility of the code that adds them; ``app.py`` wraps that.

    Args:
        filter_: An existing :class:`LogRedactionFilter` to install. If
            ``None``, a fresh one is created and returned.
        logger_name: Logger to install on. ``None`` (the default) targets
            the root logger.

    Returns:
        The installed :class:`LogRedactionFilter` so callers can register
        secrets on it later (e.g. once
        :class:`~jarvis.security.credential_store.CredentialStore` is unlocked).
    """
    target = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    redactor = filter_ if filter_ is not None else LogRedactionFilter()
    if redactor not in target.filters:
        target.addFilter(redactor)
    for handler in target.handlers:
        if redactor not in handler.filters:
            handler.addFilter(redactor)
    return redactor
