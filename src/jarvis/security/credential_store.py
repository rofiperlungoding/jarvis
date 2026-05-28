"""Credential store for JARVIS secrets.

This module implements the Credential_Store described in
``design.md §Credential_Store``: a per-user, DPAPI-encrypted file store for
third-party secrets such as ``mistral/api_key``, ``weather/api_key``, and
``email/smtp_password`` (Requirements 5.6, 13.1, 13.5, 19.3, 19.7).

Two backends are provided:

* :class:`CredentialStore` — the default DPAPI-backed file store. Each
  credential is persisted as a separate file under ``${root}/`` whose
  contents are the output of :meth:`DPAPI.protect`. The file name is derived
  from the credential name by URL-encoding (``urllib.parse.quote`` with
  ``safe=""``) so the on-disk layout is safe on every supported file system
  — in particular, the slash in names like ``mistral/api_key`` is encoded as
  ``%2F`` and the layout stays flat.

* :class:`KeyringBackend` — an optional adapter that delegates each operation
  to the third-party ``keyring`` library so the OS Credential Manager (or
  any other keyring backend) can host individual entries. Most ``keyring``
  backends do not expose an enumeration API, so this adapter maintains a
  small JSON index file that tracks which names have been written. The
  index records names only — never the secret values — so it is safe to
  read by anyone who can read the application's configuration directory.

Both backends share a common :class:`CredentialBackend` Protocol so the
Dialog_Manager and provider clients can be wired against either one without
changes.

Confidentiality (Property 8 / CP11):
    The DPAPI-backed store binds each ciphertext to its credential name via
    the ``entropy`` parameter. Renaming or moving a file on disk causes
    decryption to fail rather than silently returning the wrong secret. The
    plaintext value is never written anywhere outside the protected blob;
    the file name carries only the (non-secret) credential identifier.

Validates: Requirements 5.6, 13.1, 13.5, 19.3, 19.7
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Final, Protocol, runtime_checkable
from urllib.parse import quote, unquote

from jarvis.security.dpapi import DPAPI

logger = logging.getLogger(__name__)

__all__ = [
    "CredentialBackend",
    "CredentialStore",
    "KeyringBackend",
]


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------


#: File suffix used for every persisted credential blob. Chosen so the files
#: are unmistakably binary and so :meth:`CredentialStore.list_names` can
#: ignore unrelated files that may share the directory (e.g., README notes).
_CREDENTIAL_FILE_SUFFIX: Final[str] = ".bin"

#: Prefix used when generating a temp file inside the credentials directory
#: during an atomic write. The leading dot keeps the file out of typical
#: GUI listings; :meth:`CredentialStore.list_names` and :meth:`wipe` skip
#: anything starting with this prefix so an interrupted write cannot
#: masquerade as a credential.
_TMP_FILE_PREFIX: Final[str] = ".tmp-"

#: Domain-separation prefix mixed into DPAPI ``entropy`` for every credential
#: blob. The full entropy is ``_ENTROPY_PREFIX + name.encode("utf-8")``,
#: which binds each ciphertext to its credential name. This is a defence in
#: depth: if an attacker swaps files between names on disk, decryption fails
#: rather than yielding the wrong secret.
_ENTROPY_PREFIX: Final[bytes] = b"jarvis/credential_store/"


def _validate_name(name: str) -> None:
    """Reject obviously invalid credential names early.

    Names must be non-empty strings without NUL bytes. The shape (e.g.,
    ``provider/<id>``) is documented in ``design.md`` but not enforced here
    so users remain free to organise their own namespaces.
    """
    if not isinstance(name, str):
        raise TypeError("credential name must be a str")
    if not name:
        raise ValueError("credential name must be non-empty")
    if "\x00" in name:
        raise ValueError("credential name must not contain NUL bytes")


def _encode_name(name: str) -> str:
    """Encode a credential name into a filesystem-safe identifier.

    Uses :func:`urllib.parse.quote` with ``safe=""`` so every reserved
    character — including ``/``, ``\\``, ``:`` and the percent sign itself
    — is percent-encoded. The result is reversible via
    :func:`urllib.parse.unquote`.
    """
    return quote(name, safe="")


def _decode_name(encoded: str) -> str:
    """Reverse :func:`_encode_name`. Returns the original credential name."""
    return unquote(encoded)


def _entropy_for(name: str) -> bytes:
    """Return the per-credential DPAPI ``entropy`` value bound to ``name``."""
    return _ENTROPY_PREFIX + name.encode("utf-8")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CredentialBackend(Protocol):
    """Structural interface shared by every credential storage backend.

    Implementations MUST behave idempotently for :meth:`delete` and
    :meth:`wipe` (deleting a missing entry is a no-op), MUST round-trip
    ``set`` / ``get`` for any UTF-8 string value, and MUST return ``None``
    from :meth:`get` when the requested name has never been ``set`` or has
    since been ``delete`` d.
    """

    def set(self, name: str, value: str) -> None:
        """Persist ``value`` under ``name``, replacing any existing entry."""
        ...

    def get(self, name: str) -> str | None:
        """Return the persisted value for ``name`` or ``None`` if absent."""
        ...

    def delete(self, name: str) -> None:
        """Remove the entry for ``name``. Idempotent: missing names are OK."""
        ...

    def list_names(self) -> list[str]:
        """Return the sorted list of credential names currently stored."""
        ...

    def wipe(self) -> None:
        """Remove every stored credential. Idempotent on an empty store."""
        ...


# ---------------------------------------------------------------------------
# DPAPI-backed file store
# ---------------------------------------------------------------------------


class CredentialStore:
    """DPAPI-backed file store for JARVIS secrets.

    Each credential is written as a separate file inside ``root``; the file
    body is the opaque blob produced by :meth:`DPAPI.protect`, with
    per-credential ``entropy`` derived from the credential name. The file
    name is the URL-encoded credential name with the ``.bin`` suffix. For
    example, the credential ``mistral/api_key`` is persisted at
    ``<root>/mistral%2Fapi_key.bin``.

    Atomicity:
        :meth:`set` writes via a sibling temp file followed by
        :func:`os.replace` so a process that crashes mid-write cannot leave
        a partially-encrypted blob in place. On Windows ``os.replace`` is
        atomic; on POSIX it is atomic when the source and destination live
        on the same filesystem, which is true by construction here.

    Concurrency:
        The store is safe for use from a single asyncio loop: every method
        is synchronous and short, and the underlying filesystem operations
        are serialised by the OS. Callers running multiple processes
        against the same directory should arrange their own coordination.

    Args:
        root: Directory under which credential files live. Created on demand
            (recursively) so callers do not need to ``mkdir -p`` first.
        dpapi: The :class:`DPAPI` envelope. In production this is
            :class:`~jarvis.security.dpapi.WindowsDPAPI`; tests typically
            pass :class:`~jarvis.security.dpapi.NullDPAPI`.
    """

    def __init__(self, root: Path, dpapi: DPAPI) -> None:
        if not isinstance(root, Path):
            # Accept :class:`str` for ergonomic call sites; coerce to Path so
            # the rest of the implementation can rely on the typed API.
            root = Path(root)  # type: ignore[unreachable]
        self._root: Path = root
        self._dpapi: DPAPI = dpapi
        self._root.mkdir(parents=True, exist_ok=True)
        if not dpapi.is_genuine:
            # Audit-trail breadcrumb for non-Windows / test environments. The
            # message itself is harmless because the Null backend self-
            # identifies via the ``NOT-ENCRYPTED`` magic header on disk.
            logger.warning(
                "CredentialStore initialised with a non-genuine DPAPI backend; "
                "stored credentials are NOT cryptographically protected."
            )

    # ------------------------------------------------------------------
    # Read-only accessors (mostly for tests and diagnostics)
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        """Root directory containing the credential files."""
        return self._root

    @property
    def dpapi(self) -> DPAPI:
        """The DPAPI envelope used to protect every blob."""
        return self._dpapi

    def _path_for(self, name: str) -> Path:
        """Return the absolute path to the on-disk blob for ``name``."""
        return self._root / (_encode_name(name) + _CREDENTIAL_FILE_SUFFIX)

    # ------------------------------------------------------------------
    # CredentialBackend interface
    # ------------------------------------------------------------------

    def set(self, name: str, value: str) -> None:
        """Persist ``value`` under ``name``, replacing any existing entry.

        ``value`` is encoded as UTF-8 and protected via :meth:`DPAPI.protect`
        with name-bound entropy. The resulting blob is written via a
        temp-file-and-rename sequence so partial writes are impossible.
        """
        _validate_name(name)
        if not isinstance(value, str):
            raise TypeError("credential value must be a str")

        plaintext = value.encode("utf-8")
        blob = self._dpapi.protect(plaintext, entropy=_entropy_for(name))
        target = self._path_for(name)

        # ``mkstemp`` opens the temp file with mode 0o600 on POSIX and the
        # equivalent locked-down ACL on Windows, matching the protections
        # we want for the final credential blob.
        fd, tmp_path_str = tempfile.mkstemp(
            prefix=_TMP_FILE_PREFIX,
            suffix=_CREDENTIAL_FILE_SUFFIX,
            dir=self._root,
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(blob)
                # ``flush`` + ``fsync`` would be the most paranoid choice but
                # the cost on every credential write is steep and the rename
                # below is already atomic w.r.t. observers. We accept that a
                # post-power-loss recovery may need to redo the write.
            os.replace(tmp_path, target)
        except Exception:
            # Best-effort cleanup of the temp file; never mask the original
            # exception, which carries the diagnostic the caller actually
            # cares about.
            with contextlib.suppress(OSError):
                tmp_path.unlink()
            raise

    def get(self, name: str) -> str | None:
        """Return the persisted value for ``name``, or ``None`` if absent.

        Raises:
            ValueError: If the on-disk blob exists but cannot be decrypted
                with the expected per-name entropy. This typically indicates
                tampering, a different Windows user account, or a database
                migration that did not move the credentials directory along
                with the data root.
        """
        _validate_name(name)
        path = self._path_for(name)
        try:
            blob = path.read_bytes()
        except FileNotFoundError:
            return None
        plaintext = self._dpapi.unprotect(blob, entropy=_entropy_for(name))
        return plaintext.decode("utf-8")

    def delete(self, name: str) -> None:
        """Remove the credential entry for ``name``.

        Idempotent: deleting a name that is not present is a no-op.
        """
        _validate_name(name)
        path = self._path_for(name)
        try:
            path.unlink()
        except FileNotFoundError:
            return

    def list_names(self) -> list[str]:
        """Return the sorted list of credential names currently stored.

        Files that do not match the ``<encoded>.bin`` pattern, in-flight
        temp files (prefix ``.tmp-``), and subdirectories are ignored so the
        method is robust against accidental clutter.
        """
        if not self._root.exists():
            return []
        names: list[str] = []
        for entry in self._root.iterdir():
            if not entry.is_file():
                continue
            entry_name = entry.name
            if entry_name.startswith(_TMP_FILE_PREFIX):
                continue
            if not entry_name.endswith(_CREDENTIAL_FILE_SUFFIX):
                continue
            encoded = entry_name[: -len(_CREDENTIAL_FILE_SUFFIX)]
            try:
                decoded = _decode_name(encoded)
            except ValueError:
                # Filename whose encoded form is not legal percent-encoding;
                # almost certainly placed there by a third party. Skip rather
                # than crash the listing.
                continue
            if not decoded:
                # Defensive: an empty name should never be persisted via
                # :meth:`set`, but an external tool might have created one.
                continue
            names.append(decoded)
        names.sort()
        return names

    def wipe(self) -> None:
        """Remove every stored credential and any leftover temp files.

        Implements the credential-store half of Requirement 13.5: a "wipe-all"
        request must erase every persisted credential. The directory itself
        is preserved so subsequent :meth:`set` calls do not need to recreate
        it.
        """
        if not self._root.exists():
            return
        # Delete known credentials first so :meth:`list_names` reports an
        # empty store immediately.
        for name in self.list_names():
            self.delete(name)
        # Then sweep any temp files that an interrupted :meth:`set` may have
        # left behind. We deliberately do NOT touch unrelated files (e.g.,
        # subdirectories or non-``.bin`` artefacts) so users who repurpose
        # the directory for ancillary state do not lose unrelated data.
        for entry in self._root.iterdir():
            if entry.is_file() and entry.name.startswith(_TMP_FILE_PREFIX):
                try:
                    entry.unlink()
                except OSError:
                    # The leftover temp file is unreadable or in use; log
                    # and continue rather than aborting the wipe.
                    logger.exception(
                        "failed to remove temp credential file during wipe: %s",
                        entry,
                    )


# ---------------------------------------------------------------------------
# Optional Keyring adapter
# ---------------------------------------------------------------------------


#: Default service name used for keyring entries. Chosen to namespace JARVIS
#: secrets within the OS Credential Manager so they cannot collide with
#: entries created by other applications using the same backend.
_DEFAULT_KEYRING_SERVICE: Final[str] = "jarvis"

#: Index file name. Stored in the same directory as the DPAPI-backed store
#: when the two backends share a root, but does NOT contain any secret
#: material — only the (non-secret) list of credential names that have been
#: written through the keyring backend.
_KEYRING_INDEX_FILENAME: Final[str] = "keyring_index.json"


class KeyringBackend:
    """Adapter that stores credentials in the OS keyring via ``keyring``.

    Most ``keyring`` backends (Windows Credential Manager, macOS Keychain,
    libsecret) do not expose an enumeration API, so this adapter persists
    a small JSON index of names to make :meth:`list_names` and :meth:`wipe`
    O(1) per entry instead of relying on the OS keyring's introspection
    capabilities. **The index never contains secret values.** It is written
    via the same temp-file-and-rename pattern used by :class:`CredentialStore`.

    Args:
        index_path: Path to the JSON index file. Parent directories are
            created on demand. Pass any path under ``${app.data_dir}`` —
            the index is not sensitive on its own.
        service: ``service`` argument forwarded to every ``keyring`` call.
            Defaults to ``"jarvis"`` so JARVIS entries are namespaced within
            the OS keyring.
        keyring_module: Optional dependency-injection seam used by tests.
            When ``None`` the real ``keyring`` package is imported lazily so
            installations that never enable the adapter avoid loading the
            module at all.
    """

    DEFAULT_SERVICE: Final[str] = _DEFAULT_KEYRING_SERVICE

    def __init__(
        self,
        index_path: Path,
        *,
        service: str = _DEFAULT_KEYRING_SERVICE,
        keyring_module: Any | None = None,
    ) -> None:
        if not isinstance(index_path, Path):
            index_path = Path(index_path)  # type: ignore[unreachable]
        if not isinstance(service, str) or not service:
            raise ValueError("service must be a non-empty str")

        self._index_path: Path = index_path
        self._service: str = service
        self._keyring = keyring_module if keyring_module is not None else _import_keyring()

        # Ensure the directory exists so :meth:`set` can write the index.
        self._index_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def index_path(self) -> Path:
        """Path to the JSON index of stored credential names."""
        return self._index_path

    @property
    def service(self) -> str:
        """Service name passed to every underlying ``keyring`` call."""
        return self._service

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def _read_index(self) -> set[str]:
        """Return the current index as a set of names; ``{}`` if missing."""
        try:
            raw = self._index_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return set()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # A malformed index is treated as empty so a corrupted file does
            # not permanently lock the user out of :meth:`set`. The next
            # successful write will rebuild the index from the names that
            # were persisted in it.
            logger.warning(
                "keyring index at %s is malformed; treating as empty",
                self._index_path,
            )
            return set()
        if not isinstance(data, list):
            logger.warning(
                "keyring index at %s has unexpected shape; treating as empty",
                self._index_path,
            )
            return set()
        return {item for item in data if isinstance(item, str) and item}

    def _write_index(self, names: set[str]) -> None:
        """Persist ``names`` to the index file via temp-file-and-rename."""
        ordered = sorted(names)
        directory = self._index_path.parent
        directory.mkdir(parents=True, exist_ok=True)
        fd, tmp_path_str = tempfile.mkstemp(
            prefix=_TMP_FILE_PREFIX,
            suffix=".json",
            dir=directory,
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(ordered, f, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp_path, self._index_path)
        except Exception:
            with contextlib.suppress(OSError):
                tmp_path.unlink()
            raise

    # ------------------------------------------------------------------
    # CredentialBackend interface
    # ------------------------------------------------------------------

    def set(self, name: str, value: str) -> None:
        _validate_name(name)
        if not isinstance(value, str):
            raise TypeError("credential value must be a str")
        # ``set_password`` overwrites any existing entry, matching the
        # documented :class:`CredentialBackend` contract. Names are encoded
        # the same way as in the file store so a single index can refer to
        # both backends if a future migration ever needs it.
        encoded_name = _encode_name(name)
        self._keyring.set_password(self._service, encoded_name, value)
        # Update the index AFTER the keyring write so an OS-level failure
        # does not leave a phantom name in the index.
        index = self._read_index()
        if name not in index:
            index.add(name)
            self._write_index(index)

    def get(self, name: str) -> str | None:
        _validate_name(name)
        encoded_name = _encode_name(name)
        value = self._keyring.get_password(self._service, encoded_name)
        if value is None:
            return None
        if not isinstance(value, str):
            # ``keyring`` is documented to return ``str | None``; coerce
            # defensively in case a custom backend returns bytes.
            return str(value)
        return value

    def delete(self, name: str) -> None:
        _validate_name(name)
        encoded_name = _encode_name(name)
        try:
            self._keyring.delete_password(self._service, encoded_name)
        except Exception:
            # The third-party ``keyring`` package raises a backend-specific
            # ``PasswordDeleteError`` when the entry does not exist. We treat
            # that as a no-op to keep the documented idempotence contract,
            # but only if the entry truly is not present in our index.
            # Anything else (e.g. permission errors) re-raises.
            if name in self._read_index():
                raise
        index = self._read_index()
        if name in index:
            index.remove(name)
            self._write_index(index)

    def list_names(self) -> list[str]:
        return sorted(self._read_index())

    def wipe(self) -> None:
        """Remove every entry tracked in the index, then clear the index.

        Errors raised by ``keyring.delete_password`` for individual entries
        are logged and skipped so a single broken entry cannot prevent the
        wipe from completing — Requirement 13.5 mandates that the operation
        finishes within five seconds.
        """
        for name in self.list_names():
            encoded_name = _encode_name(name)
            try:
                self._keyring.delete_password(self._service, encoded_name)
            except Exception:
                logger.exception(
                    "keyring.delete_password failed for name=%r during wipe",
                    name,
                )
        # Clear the index unconditionally so subsequent :meth:`list_names`
        # calls return an empty list even if a few keyring deletions failed
        # above. Operators can fix lingering OS-keyring entries by hand.
        try:
            self._index_path.unlink()
        except FileNotFoundError:
            return


def _import_keyring() -> Any:
    """Lazily import the third-party ``keyring`` package.

    Centralised so the import error surfaces with an actionable message and
    so unit tests can replace the function via ``keyring_module=`` rather
    than monkey-patching ``sys.modules``.
    """
    try:
        import keyring  # noqa: PLC0415 - optional dep
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise RuntimeError(
            "KeyringBackend requires the `keyring` package. Install it with "
            "`pip install keyring` or use CredentialStore directly."
        ) from exc
    return keyring
