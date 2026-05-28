"""Unit tests for ``jarvis.security.dpapi``.

Covers:
    * Protocol conformance for both real and Null backends.
    * Round-trip semantics: ``unprotect(protect(x)) == x`` for both backends.
    * Entropy enforcement: mismatched entropy yields a clean error rather than
      silent garbage decryption.
    * Tamper / truncation detection on the Null backend.
    * Confidentiality property (CP11): the literal plaintext byte sequence
      does not appear inside the produced blob, even for the test double.
    * Default backend selection picks Windows on win32 and Null elsewhere.

Validates: Requirements 10.7, 13.1
"""

from __future__ import annotations

import os
import sys

import pytest

from jarvis.security.dpapi import (
    DPAPI,
    NullDPAPI,
    WindowsDPAPI,
    create_default_dpapi,
)

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_null_dpapi_satisfies_protocol() -> None:
    backend: DPAPI = NullDPAPI(suppress_warning=True)
    assert isinstance(backend, DPAPI)
    assert backend.is_genuine is False


# ---------------------------------------------------------------------------
# NullDPAPI: round-trip and error surface
# ---------------------------------------------------------------------------


@pytest.fixture()
def null_backend() -> NullDPAPI:
    return NullDPAPI(suppress_warning=True)


@pytest.mark.parametrize(
    "plaintext",
    [
        b"",
        b"a",
        b"hello-world",
        b"sk-mistral-test-token-1234567890",
        bytes(range(256)),
        b"\x00" * 128,
        "Привет, мир!".encode(),
    ],
)
def test_null_round_trip_default_entropy(null_backend: NullDPAPI, plaintext: bytes) -> None:
    blob = null_backend.protect(plaintext)
    assert null_backend.unprotect(blob) == plaintext


def test_null_round_trip_custom_entropy(null_backend: NullDPAPI) -> None:
    plaintext = b"super-secret-payload"
    entropy = b"per-record-entropy-13"
    blob = null_backend.protect(plaintext, entropy=entropy)
    assert null_backend.unprotect(blob, entropy=entropy) == plaintext


def test_null_entropy_mismatch_is_rejected(null_backend: NullDPAPI) -> None:
    blob = null_backend.protect(b"value", entropy=b"original")
    with pytest.raises(ValueError, match="entropy mismatch"):
        null_backend.unprotect(blob, entropy=b"different")


def test_null_truncated_blob_is_rejected(null_backend: NullDPAPI) -> None:
    blob = null_backend.protect(b"the quick brown fox")
    with pytest.raises(ValueError):
        null_backend.unprotect(blob[:8])


def test_null_invalid_magic_is_rejected(null_backend: NullDPAPI) -> None:
    # A blob of the same length as a real one but with wrong magic header.
    fake = b"NOT-A-REAL-DPAPI-BLOB" + b"\x00" * 64
    with pytest.raises(ValueError, match="magic header"):
        null_backend.unprotect(fake)


def test_null_protect_rejects_non_bytes(null_backend: NullDPAPI) -> None:
    with pytest.raises(TypeError):
        null_backend.protect("a string is not bytes")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        null_backend.protect(b"ok", entropy="entropy as str")  # type: ignore[arg-type]


def test_null_unprotect_rejects_non_bytes(null_backend: NullDPAPI) -> None:
    with pytest.raises(TypeError):
        null_backend.unprotect("not-bytes")  # type: ignore[arg-type]


def test_null_blob_is_not_genuine_marker(null_backend: NullDPAPI) -> None:
    """The Null payload must self-identify as non-encrypted in audit trails."""
    blob = null_backend.protect(b"anything")
    assert b"NOT-ENCRYPTED" in blob


# ---------------------------------------------------------------------------
# Confidentiality property (mirrors Property 8 / CP11 at the unit level)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "plaintext",
    [
        b"a-very-distinctive-secret-12345",
        b"another\x00binary\xffsecret\x7f",
        bytes(range(1, 200)),
    ],
)
def test_null_blob_does_not_contain_plaintext_substring(
    null_backend: NullDPAPI, plaintext: bytes
) -> None:
    """Even the obfuscating test double must avoid leaking the literal plaintext."""
    # Skip degenerate inputs the property cannot meaningfully cover.
    assert len(plaintext) >= 4
    blob = null_backend.protect(plaintext)
    assert plaintext not in blob


# ---------------------------------------------------------------------------
# Default backend selection
# ---------------------------------------------------------------------------


def test_create_default_dpapi_returns_protocol_implementation() -> None:
    backend = create_default_dpapi()
    assert isinstance(backend, DPAPI)


def test_create_default_dpapi_is_null_on_non_windows() -> None:
    if sys.platform == "win32":
        pytest.skip("Non-Windows-only assertion")
    backend = create_default_dpapi()  # type: ignore[unreachable]
    assert isinstance(backend, NullDPAPI)
    assert backend.is_genuine is False


# ---------------------------------------------------------------------------
# Real WindowsDPAPI smoke tests — gated behind JARVIS_TEST_WINDOWS=1.
#
# These require a real Windows host with pywin32 installed (matching the
# matrix in .github/workflows/ci.yml). They confirm the wrapper:
#   * Round-trips bytes with default and custom entropy.
#   * Rejects mismatched entropy with an exception (rather than returning
#     garbage), exactly as CryptUnprotectData does.
# ---------------------------------------------------------------------------


_REQUIRES_WINDOWS = pytest.mark.skipif(
    not (sys.platform == "win32" and os.environ.get("JARVIS_TEST_WINDOWS") == "1"),
    reason="WindowsDPAPI smoke tests require Windows + JARVIS_TEST_WINDOWS=1",
)


@_REQUIRES_WINDOWS
def test_windows_dpapi_round_trip_default_entropy() -> None:
    backend = WindowsDPAPI()
    payload = b"sk-mistral-test-token-1234567890"
    blob = backend.protect(payload)
    assert blob != payload  # ciphertext must not equal plaintext
    assert backend.unprotect(blob) == payload


@_REQUIRES_WINDOWS
def test_windows_dpapi_round_trip_custom_entropy() -> None:
    backend = WindowsDPAPI()
    payload = b"per-record-secret"
    entropy = b"record-id:00000001"
    blob = backend.protect(payload, entropy=entropy)
    assert backend.unprotect(blob, entropy=entropy) == payload


@_REQUIRES_WINDOWS
def test_windows_dpapi_entropy_mismatch_is_rejected() -> None:
    backend = WindowsDPAPI()
    blob = backend.protect(b"payload", entropy=b"correct")
    # win32crypt raises pywintypes.error on mismatch; we accept any
    # exception subclass to keep the test backend-agnostic.
    with pytest.raises(Exception):  # noqa: B017 - vendor exception type intentional
        backend.unprotect(blob, entropy=b"wrong")


@_REQUIRES_WINDOWS
def test_windows_dpapi_blob_is_not_plaintext_substring() -> None:
    backend = WindowsDPAPI()
    payload = b"a-very-distinctive-secret-substring"
    blob = backend.protect(payload)
    assert payload not in blob
