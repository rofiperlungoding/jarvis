"""Unit tests for ``jarvis.security.credential_store``.

Covers:
    * Round-trip ``set`` / ``get`` for a variety of UTF-8 secret values.
    * ``delete`` is idempotent (deleting a missing name is a no-op).
    * ``list_names`` reports the set of names currently persisted, sorted,
      and ignores temp / unrelated files in the credentials directory.
    * ``wipe`` clears every persisted credential and any orphaned temp
      file but preserves the credentials directory itself.
    * Encryption-at-rest is verifiable: the literal plaintext substring
      MUST NOT appear inside the on-disk blob (CP11 / Property 8 at the
      unit level), even when the test uses :class:`NullDPAPI`.
    * Per-name entropy binding: renaming the on-disk blob to a different
      credential name causes :meth:`get` to fail decryption rather than
      silently returning the secret stored under the original name.
    * ``KeyringBackend`` adapter behaviour — round-trip, idempotent delete,
      ``list_names`` via the JSON index, and ``wipe`` clears both the OS
      keyring and the index file.

Validates: Requirements 5.6, 13.1, 13.5, 19.3, 19.7
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote

import pytest

from jarvis.security.credential_store import (
    CredentialBackend,
    CredentialStore,
    KeyringBackend,
)
from jarvis.security.dpapi import NullDPAPI

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(root: Path) -> CredentialStore:
    """Construct a :class:`CredentialStore` over the test double DPAPI.

    Tests on non-Windows runners use :class:`NullDPAPI`, which obfuscates
    rather than encrypts. The CP11-style assertions in this file only
    require that the literal plaintext does not appear in the produced
    blob — a property the Null backend already provides via its XOR
    keystream — so the assertions remain meaningful in CI.
    """
    dpapi = NullDPAPI(suppress_warning=True)
    return CredentialStore(root, dpapi)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_constructor_creates_root_directory(tmp_path: Path) -> None:
    target = tmp_path / "secrets"
    assert not target.exists()
    store = _make_store(target)
    assert store.root.is_dir()
    assert store.list_names() == []


def test_constructor_accepts_string_path(tmp_path: Path) -> None:
    # The constructor coerces ``str`` to ``Path`` defensively; mypy's
    # narrowing thinks the path is unreachable when the param is typed
    # as Path, but the runtime branch is the contract under test.
    store = CredentialStore(str(tmp_path / "store"), NullDPAPI(suppress_warning=True))  # type: ignore[arg-type]
    assert isinstance(store.root, Path)
    assert store.root.is_dir()


def test_satisfies_credential_backend_protocol(tmp_path: Path) -> None:
    store = _make_store(tmp_path / "store")
    backend: CredentialBackend = store
    assert isinstance(backend, CredentialBackend)


# ---------------------------------------------------------------------------
# Round-trip set/get
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("mistral/api_key", "sk-this-is-a-mistral-key-1234567890"),
        ("weather/api_key", "ow_abcdef1234567890"),
        ("email/smtp_password", "p@ssw0rd!#$%"),
        ("user/profile", "name with spaces and unicode: Привет, мир!"),
        ("multi/segment/name", ""),  # empty string is a valid secret
    ],
)
def test_set_get_round_trip(tmp_path: Path, name: str, value: str) -> None:
    store = _make_store(tmp_path / "store")
    store.set(name, value)
    assert store.get(name) == value


def test_set_overwrites_existing_value(tmp_path: Path) -> None:
    store = _make_store(tmp_path / "store")
    store.set("mistral/api_key", "v1")
    store.set("mistral/api_key", "v2-rotated")
    assert store.get("mistral/api_key") == "v2-rotated"


def test_get_returns_none_for_missing_name(tmp_path: Path) -> None:
    store = _make_store(tmp_path / "store")
    assert store.get("never/written") is None


def test_set_rejects_empty_name(tmp_path: Path) -> None:
    store = _make_store(tmp_path / "store")
    with pytest.raises(ValueError):
        store.set("", "anything")


def test_set_rejects_non_string_name(tmp_path: Path) -> None:
    store = _make_store(tmp_path / "store")
    with pytest.raises(TypeError):
        store.set(123, "anything")  # type: ignore[arg-type]


def test_set_rejects_non_string_value(tmp_path: Path) -> None:
    store = _make_store(tmp_path / "store")
    with pytest.raises(TypeError):
        store.set("mistral/api_key", 123)  # type: ignore[arg-type]


def test_set_rejects_nul_byte_in_name(tmp_path: Path) -> None:
    store = _make_store(tmp_path / "store")
    with pytest.raises(ValueError):
        store.set("bad\x00name", "v")


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def test_delete_removes_persisted_credential(tmp_path: Path) -> None:
    store = _make_store(tmp_path / "store")
    store.set("mistral/api_key", "v")
    store.delete("mistral/api_key")
    assert store.get("mistral/api_key") is None
    assert "mistral/api_key" not in store.list_names()


def test_delete_is_idempotent(tmp_path: Path) -> None:
    store = _make_store(tmp_path / "store")
    # Deleting a name that has never been written is a no-op.
    store.delete("never/written")
    # Deleting again right after a successful delete is also a no-op.
    store.set("mistral/api_key", "v")
    store.delete("mistral/api_key")
    store.delete("mistral/api_key")  # must not raise
    assert store.get("mistral/api_key") is None


# ---------------------------------------------------------------------------
# list_names
# ---------------------------------------------------------------------------


def test_list_names_is_sorted_and_round_trips_encoded_filenames(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path / "store")
    names = ["zeta/key", "alpha/key", "mistral/api_key", "weather/api_key"]
    for n in names:
        store.set(n, f"value-of-{n}")
    listed = store.list_names()
    assert listed == sorted(names)


def test_list_names_ignores_unrelated_files(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = _make_store(root)
    store.set("mistral/api_key", "v")

    # Drop a few decoy files that should be ignored by ``list_names``.
    (root / "README.md").write_text("not a credential", encoding="utf-8")
    (root / "garbage.txt").write_text("nope", encoding="utf-8")
    (root / "subdir").mkdir()
    (root / "subdir" / "still_not.bin").write_bytes(b"\x00")
    # In-flight temp file (matches the ``.tmp-`` prefix used by the store).
    (root / ".tmp-orphaned.bin").write_bytes(b"\x00")
    # A ``.bin`` file whose stem decodes to an empty string is skipped.
    (root / ".bin").write_bytes(b"\x00")

    assert store.list_names() == ["mistral/api_key"]


def test_list_names_skips_corrupt_percent_encoded_filenames(
    tmp_path: Path,
) -> None:
    """A ``.bin`` file whose stem cannot be decoded must not crash listing."""
    root = tmp_path / "store"
    store = _make_store(root)
    store.set("mistral/api_key", "v")
    # ``%ZZ`` is not a valid percent escape but ``urllib.parse.unquote`` is
    # tolerant and returns the literal text. We just ensure listing is
    # robust regardless: it should at least not raise.
    (root / "garbage%ZZ.bin").write_bytes(b"\x00")
    listed = store.list_names()
    assert "mistral/api_key" in listed


def test_list_names_when_root_missing(tmp_path: Path) -> None:
    store = _make_store(tmp_path / "store")
    # Remove the directory after construction to simulate a wiped data dir.
    store.set("mistral/api_key", "v")
    store.delete("mistral/api_key")
    # The directory itself is preserved by design; remove it to verify the
    # graceful fallback in :meth:`list_names`.
    os.rmdir(store.root)
    assert store.list_names() == []


# ---------------------------------------------------------------------------
# Wipe
# ---------------------------------------------------------------------------


def test_wipe_removes_all_credentials_and_temp_files(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = _make_store(root)
    for n in ("mistral/api_key", "weather/api_key", "email/smtp_password"):
        store.set(n, f"value-of-{n}")
    # Drop an orphaned temp file like a crashed write would.
    orphan = root / ".tmp-orphan.bin"
    orphan.write_bytes(b"\x00")
    # Drop a non-credential file that must be preserved.
    keep = root / "README.md"
    keep.write_text("operator notes", encoding="utf-8")

    store.wipe()

    assert store.list_names() == []
    assert not orphan.exists()
    assert keep.exists()  # unrelated files survive the wipe
    assert root.is_dir()  # directory itself is preserved


def test_wipe_on_missing_directory_is_a_no_op(tmp_path: Path) -> None:
    store = _make_store(tmp_path / "store")
    os.rmdir(store.root)
    store.wipe()  # must not raise


def test_set_after_wipe_works_again(tmp_path: Path) -> None:
    store = _make_store(tmp_path / "store")
    store.set("mistral/api_key", "v1")
    store.wipe()
    store.set("mistral/api_key", "v2")
    assert store.get("mistral/api_key") == "v2"


# ---------------------------------------------------------------------------
# Encryption-at-rest verification (CP11 at the unit level)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "sk-very-distinctive-secret-XXYYZZ",
        "ow_abcdef1234567890",
        "p@ssw0rd!#$%",
        "Привет, мир — это секрет!",
    ],
)
def test_plaintext_is_not_present_in_on_disk_blob(
    tmp_path: Path, value: str
) -> None:
    """The literal secret bytes MUST NOT appear inside the persisted file.

    Mirrors Property 8 / CP11 at the unit-test level. With
    :class:`NullDPAPI` as the test double, the obfuscation is not
    cryptographically secure — but the property under test is a hard
    requirement that "no leaked plaintext substring" holds even for the
    test-only backend, so it is exercised here as well.
    """
    store = _make_store(tmp_path / "store")
    name = "mistral/api_key"
    store.set(name, value)

    # Read the on-disk blob directly, not through ``get``.
    expected_path = tmp_path / "store" / (quote(name, safe="") + ".bin")
    assert expected_path.is_file()

    blob = expected_path.read_bytes()
    assert value.encode("utf-8") not in blob


def test_credential_filename_uses_url_encoding(tmp_path: Path) -> None:
    """Slashes in the name MUST be encoded so the layout stays flat."""
    store = _make_store(tmp_path / "store")
    store.set("mistral/api_key", "v")
    files = sorted(p.name for p in (tmp_path / "store").iterdir() if p.is_file())
    assert files == ["mistral%2Fapi_key.bin"]


# ---------------------------------------------------------------------------
# Per-name entropy binding
# ---------------------------------------------------------------------------


def test_renaming_blob_breaks_decryption(tmp_path: Path) -> None:
    """Per-name entropy makes a renamed blob undecryptable.

    The store binds each ciphertext to its credential name via the DPAPI
    ``entropy`` parameter. Renaming a blob on disk must therefore cause
    :meth:`get` to fail rather than silently return the wrong secret.
    """
    store = _make_store(tmp_path / "store")
    store.set("source/name", "the-source-secret")
    store.set("victim/name", "the-victim-secret")  # placeholder so destination doesn't 404

    src = tmp_path / "store" / (quote("source/name", safe="") + ".bin")
    dst = tmp_path / "store" / (quote("victim/name", safe="") + ".bin")

    # Overwrite the victim's blob with the source blob under the victim's name.
    dst.write_bytes(src.read_bytes())

    # Reading "victim/name" now must NOT return "the-source-secret".
    # The Null backend raises ValueError on entropy mismatch; the real
    # Windows backend raises a pywin32 error. Either way, we forbid silent
    # success — so we accept any non-trivial exception subclass.
    with pytest.raises(Exception):  # noqa: B017
        store.get("victim/name")

    # And the original (correctly-named) blob still decrypts.
    assert store.get("source/name") == "the-source-secret"


def test_two_different_names_produce_different_blobs_for_same_value(
    tmp_path: Path,
) -> None:
    """Same secret stored under two names yields different ciphertexts.

    Because the entropy is name-bound, two credentials with identical
    plaintext must NOT produce identical on-disk blobs — otherwise an
    attacker who saw both could conclude the values are equal.
    """
    store = _make_store(tmp_path / "store")
    secret = "shared-value-not-a-real-key"
    store.set("first/name", secret)
    store.set("second/name", secret)

    blob_a = (tmp_path / "store" / (quote("first/name", safe="") + ".bin")).read_bytes()
    blob_b = (
        tmp_path / "store" / (quote("second/name", safe="") + ".bin")
    ).read_bytes()
    assert blob_a != blob_b


# ---------------------------------------------------------------------------
# Atomic write semantics
# ---------------------------------------------------------------------------


def test_set_does_not_leave_orphan_temp_file_on_success(tmp_path: Path) -> None:
    store = _make_store(tmp_path / "store")
    store.set("mistral/api_key", "v")
    leftover = [
        p
        for p in (tmp_path / "store").iterdir()
        if p.is_file() and p.name.startswith(".tmp-")
    ]
    assert leftover == []


# ---------------------------------------------------------------------------
# KeyringBackend
# ---------------------------------------------------------------------------


class _InMemoryKeyring:
    """Minimal stand-in for the ``keyring`` module.

    Implements only the three functions :class:`KeyringBackend` calls.
    Tracks every call so tests can assert on the backend's interaction
    pattern (e.g., ``set_password`` overwrite semantics).
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}
        self.calls: list[tuple[str, str, str | None]] = []

    def set_password(self, service: str, username: str, password: str) -> None:
        self._store[(service, username)] = password
        self.calls.append(("set", service, username))

    def get_password(self, service: str, username: str) -> str | None:
        self.calls.append(("get", service, username))
        return self._store.get((service, username))

    def delete_password(self, service: str, username: str) -> None:
        try:
            del self._store[(service, username)]
        except KeyError:
            raise _PasswordDeleteError("not found") from None
        self.calls.append(("delete", service, username))


class _PasswordDeleteError(Exception):
    """Stand-in for :class:`keyring.errors.PasswordDeleteError`."""


def test_keyring_backend_round_trip(tmp_path: Path) -> None:
    fake_keyring = _InMemoryKeyring()
    backend = KeyringBackend(
        tmp_path / "keyring_index.json", keyring_module=fake_keyring
    )
    backend.set("mistral/api_key", "sk-keyring-XYZ")
    assert backend.get("mistral/api_key") == "sk-keyring-XYZ"


def test_keyring_backend_get_returns_none_for_missing(tmp_path: Path) -> None:
    backend = KeyringBackend(
        tmp_path / "keyring_index.json", keyring_module=_InMemoryKeyring()
    )
    assert backend.get("never/written") is None


def test_keyring_backend_list_names_after_set(tmp_path: Path) -> None:
    backend = KeyringBackend(
        tmp_path / "keyring_index.json", keyring_module=_InMemoryKeyring()
    )
    backend.set("alpha/key", "a")
    backend.set("zeta/key", "z")
    backend.set("mistral/api_key", "m")
    assert backend.list_names() == ["alpha/key", "mistral/api_key", "zeta/key"]


def test_keyring_backend_set_overwrite_does_not_duplicate_index(
    tmp_path: Path,
) -> None:
    backend = KeyringBackend(
        tmp_path / "keyring_index.json", keyring_module=_InMemoryKeyring()
    )
    backend.set("mistral/api_key", "v1")
    backend.set("mistral/api_key", "v2")
    assert backend.list_names() == ["mistral/api_key"]


def test_keyring_backend_delete_is_idempotent(tmp_path: Path) -> None:
    backend = KeyringBackend(
        tmp_path / "keyring_index.json", keyring_module=_InMemoryKeyring()
    )
    backend.set("mistral/api_key", "v")
    backend.delete("mistral/api_key")
    backend.delete("mistral/api_key")  # must not raise
    backend.delete("never/written")  # must not raise either
    assert backend.list_names() == []


def test_keyring_backend_wipe_clears_index_and_keyring(tmp_path: Path) -> None:
    fake_keyring = _InMemoryKeyring()
    backend = KeyringBackend(
        tmp_path / "keyring_index.json", keyring_module=fake_keyring
    )
    backend.set("mistral/api_key", "v1")
    backend.set("weather/api_key", "v2")
    backend.wipe()

    assert backend.list_names() == []
    assert not (tmp_path / "keyring_index.json").exists()
    # The underlying keyring is empty too.
    assert backend.get("mistral/api_key") is None
    assert backend.get("weather/api_key") is None


def test_keyring_backend_handles_malformed_index(tmp_path: Path) -> None:
    """A malformed index is treated as empty rather than raising."""
    index = tmp_path / "keyring_index.json"
    index.write_text("this is not json", encoding="utf-8")
    backend = KeyringBackend(index, keyring_module=_InMemoryKeyring())
    assert backend.list_names() == []
    # And subsequent writes succeed and rebuild the index.
    backend.set("mistral/api_key", "v")
    assert backend.list_names() == ["mistral/api_key"]


def test_keyring_backend_index_is_valid_json(tmp_path: Path) -> None:
    """The index format is documented to be a JSON list of names."""
    backend = KeyringBackend(
        tmp_path / "keyring_index.json", keyring_module=_InMemoryKeyring()
    )
    backend.set("alpha/key", "a")
    backend.set("zeta/key", "z")
    payload = json.loads((tmp_path / "keyring_index.json").read_text(encoding="utf-8"))
    assert payload == ["alpha/key", "zeta/key"]


def test_keyring_backend_does_not_persist_secret_values(tmp_path: Path) -> None:
    """The on-disk index MUST NOT contain any secret value."""
    backend = KeyringBackend(
        tmp_path / "keyring_index.json", keyring_module=_InMemoryKeyring()
    )
    secret = "do-not-write-me-to-disk"
    backend.set("mistral/api_key", secret)
    raw = (tmp_path / "keyring_index.json").read_bytes()
    assert secret.encode("utf-8") not in raw


def test_keyring_backend_rejects_invalid_service(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        KeyringBackend(
            tmp_path / "keyring_index.json",
            service="",
            keyring_module=_InMemoryKeyring(),
        )


def test_keyring_backend_passes_through_service_name(tmp_path: Path) -> None:
    fake_keyring = _InMemoryKeyring()
    backend = KeyringBackend(
        tmp_path / "keyring_index.json",
        service="jarvis-test",
        keyring_module=fake_keyring,
    )
    backend.set("mistral/api_key", "v")
    backend.get("mistral/api_key")
    services = {service for _kind, service, _user in fake_keyring.calls}
    assert services == {"jarvis-test"}


def test_keyring_backend_uses_encoded_username(tmp_path: Path) -> None:
    fake_keyring = _InMemoryKeyring()
    backend = KeyringBackend(
        tmp_path / "keyring_index.json", keyring_module=fake_keyring
    )
    backend.set("mistral/api_key", "v")
    # The slash in the credential name must be percent-encoded so the OS
    # keyring treats it as a single opaque identifier.
    usernames = {user for _kind, _service, user in fake_keyring.calls}
    assert usernames == {"mistral%2Fapi_key"}
