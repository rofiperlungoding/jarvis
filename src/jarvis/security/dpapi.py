"""DPAPI (Windows Data Protection API) envelope used by Credential_Store and Memory_Store.

This module implements the DPAPI interface described in
``design.md §Credential_Store`` and ships two concrete backends:

* :class:`WindowsDPAPI` — the real, user-scoped encryption backend, backed by
  ``win32crypt.CryptProtectData`` / ``win32crypt.CryptUnprotectData`` with
  ``CRYPTPROTECT_LOCAL_MACHINE`` cleared so produced blobs are decryptable
  only by the current Windows user account on the current machine
  (Requirement 13.1).

* :class:`NullDPAPI` — a non-encrypting test double for non-Windows CI runners
  (Linux/macOS). It round-trips bytes via a deterministic XOR keystream so
  property tests like CP11 (credential confidentiality) still hold
  byte-for-byte (the literal plaintext does not appear on disk), and prefixes
  every blob with a ``JARVIS-NULL-DPAPI-NOT-ENCRYPTED`` magic header so the
  artefact is unmistakably recognisable as non-secret. **Never** use it to
  protect real secrets.

The ``DPAPI`` Protocol is structurally typed so :class:`WindowsDPAPI`,
:class:`NullDPAPI`, and any future backend (e.g. macOS Keychain or Linux
``libsecret``) can be supplied to consumers such as ``CredentialStore`` and
``MemoryStore`` without changes to their type signatures.

Validates: Requirements 10.7, 13.1
"""

from __future__ import annotations

import hashlib
import logging
import sys
from typing import Final, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "DPAPI",
    "NullDPAPI",
    "WindowsDPAPI",
    "create_default_dpapi",
]


@runtime_checkable
class DPAPI(Protocol):
    """Structural interface for a per-user data-protection envelope.

    Implementations encrypt (or, for the test double, obfuscate) ``plaintext``
    bytes such that ``unprotect(protect(p, entropy=e), entropy=e) == p`` for
    matching ``entropy`` values, and raise / return ciphertext that fails to
    round-trip when the entropy does not match.

    Attributes:
        is_genuine: ``True`` when the implementation provides real
            cryptographic confidentiality (e.g. :class:`WindowsDPAPI`).
            ``False`` for test doubles like :class:`NullDPAPI`. Callers that
            store production secrets should refuse to start when this flag is
            ``False`` outside of an explicit test/CI environment.
    """

    @property
    def is_genuine(self) -> bool: ...

    def protect(self, plaintext: bytes, *, entropy: bytes = b"jarvis") -> bytes:
        """Return an opaque blob that protects ``plaintext`` for the current user."""
        ...

    def unprotect(self, blob: bytes, *, entropy: bytes = b"jarvis") -> bytes:
        """Reverse :meth:`protect`. Raises on entropy mismatch or tampering."""
        ...


# ---------------------------------------------------------------------------
# Windows backend
# ---------------------------------------------------------------------------


class WindowsDPAPI:
    """Real DPAPI envelope backed by ``pywin32``'s ``win32crypt`` module.

    Uses ``CryptProtectData`` / ``CryptUnprotectData`` with ``Flags=0`` so
    ``CRYPTPROTECT_LOCAL_MACHINE`` is *not* set. The resulting blob is
    decryptable only by the current user on the current machine
    (Requirement 13.1).
    """

    is_genuine: Final[bool] = True

    def __init__(self) -> None:
        try:
            # Import lazily so the module is importable on non-Windows runners
            # for tests that only exercise NullDPAPI.
            import win32crypt  # noqa: PLC0415 - Windows-only optional dep
        except ImportError as exc:  # pragma: no cover - exercised on non-Windows
            raise RuntimeError(
                "WindowsDPAPI requires the `pywin32` package (win32crypt). "
                "Install pywin32 on Windows, or use NullDPAPI for non-Windows tests."
            ) from exc
        self._win32crypt = win32crypt

    def protect(self, plaintext: bytes, *, entropy: bytes = b"jarvis") -> bytes:
        if not isinstance(plaintext, (bytes, bytearray)):
            raise TypeError("plaintext must be bytes")
        if not isinstance(entropy, (bytes, bytearray)):
            raise TypeError("entropy must be bytes")
        # CryptProtectData(DataIn, DataDescr, OptionalEntropy, Reserved,
        #                  PromptStruct, Flags) -> bytes
        # Flags=0 => CRYPTPROTECT_LOCAL_MACHINE is FALSE; secret is bound to user.
        return self._win32crypt.CryptProtectData(  # type: ignore[no-any-return]
            bytes(plaintext),
            None,  # DataDescr — deliberately omitted to avoid leaking metadata.
            bytes(entropy),  # OptionalEntropy
            None,  # Reserved
            None,  # PromptStruct
            0,  # Flags — CRYPTPROTECT_LOCAL_MACHINE = False
        )

    def unprotect(self, blob: bytes, *, entropy: bytes = b"jarvis") -> bytes:
        if not isinstance(blob, (bytes, bytearray)):
            raise TypeError("blob must be bytes")
        if not isinstance(entropy, (bytes, bytearray)):
            raise TypeError("entropy must be bytes")
        # CryptUnprotectData(DataIn, OptionalEntropy, Reserved, PromptStruct,
        #                    Flags) -> (DataDescr, DataOut)
        _description, data = self._win32crypt.CryptUnprotectData(
            bytes(blob),
            bytes(entropy),
            None,
            None,
            0,
        )
        return bytes(data)


# ---------------------------------------------------------------------------
# Test double for non-Windows CI
# ---------------------------------------------------------------------------

# A fixed, recognisable header that makes Null payloads unmistakable in audit
# trails or accidental leaks. Embedding the literal phrase
# "NOT-ENCRYPTED" is intentional: an operator who finds this on disk should
# realise immediately that the artefact is not a real secret blob.
_NULL_MAGIC: Final[bytes] = b"JARVIS-NULL-DPAPI-NOT-ENCRYPTED\x00"
_NULL_KEY_DOMAIN: Final[bytes] = b"jarvis-null-dpapi-keystream-v1"
_ENTROPY_FP_LEN: Final[int] = 32  # SHA-256 digest length
_LENGTH_FIELD: Final[int] = 8


class NullDPAPI:
    """Non-encrypting in-process test double for non-Windows CI.

    The payload format is::

        | _NULL_MAGIC | sha256(entropy) | uint64 BE length | XOR(plaintext, ks) |

    where ``ks`` is a deterministic keystream derived from the entropy via
    repeated SHA-256 hashing. This is **not** cryptographically secure; the
    keystream is fully reproducible by anyone who knows the entropy. The
    obfuscation exists purely so:

    * The literal plaintext does not appear on disk verbatim, allowing
      Property 8 (credential confidentiality, CP11) to be exercised
      meaningfully in non-Windows CI.
    * Entropy mismatches are detected and reported, mirroring DPAPI's
      ``CryptUnprotectData`` behaviour.
    * Truncated or malformed blobs are rejected, so consumers can rely on the
      same error surface as the real backend.

    Use exclusively in tests or local development environments.
    """

    is_genuine: Final[bool] = False

    def __init__(self, *, suppress_warning: bool = False) -> None:
        if not suppress_warning:
            logger.warning(
                "NullDPAPI in use: payloads are obfuscated, NOT encrypted. "
                "Do not use to protect real secrets."
            )

    @staticmethod
    def _derive_keystream(entropy: bytes, length: int) -> bytes:
        """Derive a ``length``-byte keystream deterministically from ``entropy``."""
        out = bytearray()
        counter = 0
        while len(out) < length:
            block = hashlib.sha256(
                _NULL_KEY_DOMAIN + entropy + counter.to_bytes(8, "big")
            ).digest()
            out.extend(block)
            counter += 1
        return bytes(out[:length])

    def protect(self, plaintext: bytes, *, entropy: bytes = b"jarvis") -> bytes:
        if not isinstance(plaintext, (bytes, bytearray)):
            raise TypeError("plaintext must be bytes")
        if not isinstance(entropy, (bytes, bytearray)):
            raise TypeError("entropy must be bytes")
        plaintext = bytes(plaintext)
        entropy = bytes(entropy)
        entropy_fp = hashlib.sha256(entropy).digest()
        keystream = self._derive_keystream(entropy, len(plaintext))
        ciphertext = bytes(p ^ k for p, k in zip(plaintext, keystream, strict=True))
        length_prefix = len(plaintext).to_bytes(_LENGTH_FIELD, "big")
        return _NULL_MAGIC + entropy_fp + length_prefix + ciphertext

    def unprotect(self, blob: bytes, *, entropy: bytes = b"jarvis") -> bytes:
        if not isinstance(blob, (bytes, bytearray)):
            raise TypeError("blob must be bytes")
        if not isinstance(entropy, (bytes, bytearray)):
            raise TypeError("entropy must be bytes")
        blob = bytes(blob)
        entropy = bytes(entropy)

        header_len = len(_NULL_MAGIC)
        min_len = header_len + _ENTROPY_FP_LEN + _LENGTH_FIELD
        if len(blob) < min_len:
            raise ValueError("NullDPAPI blob is truncated (smaller than header)")
        if not blob.startswith(_NULL_MAGIC):
            raise ValueError("NullDPAPI blob has invalid magic header")

        offset = header_len
        entropy_fp = blob[offset : offset + _ENTROPY_FP_LEN]
        if entropy_fp != hashlib.sha256(entropy).digest():
            raise ValueError("NullDPAPI entropy mismatch")
        offset += _ENTROPY_FP_LEN

        length = int.from_bytes(blob[offset : offset + _LENGTH_FIELD], "big")
        offset += _LENGTH_FIELD

        ciphertext = blob[offset : offset + length]
        if len(ciphertext) != length:
            raise ValueError("NullDPAPI blob is truncated (ciphertext shorter than declared length)")

        keystream = self._derive_keystream(entropy, length)
        return bytes(c ^ k for c, k in zip(ciphertext, keystream, strict=True))


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_default_dpapi() -> DPAPI:
    """Pick the best available DPAPI backend for the current host.

    On Windows with ``pywin32`` installed, returns a :class:`WindowsDPAPI`
    instance — real, user-scoped encryption. On any other platform (or on
    Windows hosts without ``pywin32``, which should never occur in supported
    deployments but does happen on bare-bones CI), logs a warning and returns
    a :class:`NullDPAPI`.

    Production startup code should additionally refuse to proceed when the
    returned backend has ``is_genuine == False``, unless a documented
    test/CI override is set.
    """
    if sys.platform == "win32":
        try:
            return WindowsDPAPI()
        except RuntimeError:
            logger.warning(
                "pywin32 (win32crypt) not available on Windows; falling back to "
                "NullDPAPI. Install pywin32 to enable real DPAPI encryption."
            )
    return NullDPAPI()
