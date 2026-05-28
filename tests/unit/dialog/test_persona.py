"""Unit tests for ``jarvis.dialog.persona``.

Covers the :class:`PersonaProfile` data model and the :func:`load_persona`
config-driven loader.

Validates Requirements 11.1, 11.2, 11.3, 11.4.
"""

from __future__ import annotations

import pytest

from jarvis.config.schema import Config
from jarvis.dialog.persona import (
    BUILTIN_PERSONAS,
    DEFAULT_FORBIDDEN_SELF_REFS,
    PersonaProfile,
    default_jarvis_persona,
    load_persona,
    register_persona,
    unregister_persona,
)
from jarvis.dialog.persona_guard import PersonaGuard, PersonaLike

# ---------------------------------------------------------------------------
# PersonaProfile dataclass invariants
# ---------------------------------------------------------------------------


def test_persona_profile_basic_construction() -> None:
    profile = PersonaProfile(
        name="JARVIS",
        honorific="sir",
        system_prompt="You are JARVIS.",
        tts_voice="en_GB-alan-medium",
        forbidden_self_refs=("ChatGPT",),
    )
    assert profile.name == "JARVIS"
    assert profile.honorific == "sir"
    assert profile.tts_voice == "en_GB-alan-medium"
    assert profile.forbidden_self_refs == ("ChatGPT",)


def test_persona_profile_is_frozen() -> None:
    """Requirement 11.3 — persona is immutable across turns."""
    profile = default_jarvis_persona()
    with pytest.raises((AttributeError, TypeError)):
        profile.name = "FRIDAY"  # type: ignore[misc]


def test_persona_profile_is_hashable() -> None:
    """Two equal profiles must hash identically (frozen dataclass guarantee)."""
    a = default_jarvis_persona()
    b = default_jarvis_persona()
    assert a == b
    assert hash(a) == hash(b)
    assert {a, b} == {a}


def test_persona_profile_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        PersonaProfile(
            name="",
            honorific="sir",
            system_prompt="prompt",
            tts_voice="voice",
            forbidden_self_refs=(),
        )


def test_persona_profile_rejects_empty_system_prompt() -> None:
    with pytest.raises(ValueError, match="system_prompt"):
        PersonaProfile(
            name="JARVIS",
            honorific="sir",
            system_prompt="",
            tts_voice="voice",
            forbidden_self_refs=(),
        )


def test_persona_profile_rejects_empty_tts_voice() -> None:
    with pytest.raises(ValueError, match="tts_voice"):
        PersonaProfile(
            name="JARVIS",
            honorific="sir",
            system_prompt="prompt",
            tts_voice="",
            forbidden_self_refs=(),
        )


def test_persona_profile_rejects_non_tuple_forbidden_self_refs() -> None:
    with pytest.raises(TypeError, match="tuple"):
        PersonaProfile(
            name="JARVIS",
            honorific="sir",
            system_prompt="prompt",
            tts_voice="voice",
            forbidden_self_refs=["ChatGPT"],  # type: ignore[arg-type]
        )


def test_persona_profile_rejects_empty_forbidden_phrase() -> None:
    with pytest.raises(ValueError, match="forbidden_self_refs"):
        PersonaProfile(
            name="JARVIS",
            honorific="sir",
            system_prompt="prompt",
            tts_voice="voice",
            forbidden_self_refs=("ChatGPT", ""),
        )


# ---------------------------------------------------------------------------
# Default JARVIS persona (Requirements 11.1, 11.2)
# ---------------------------------------------------------------------------


def test_default_jarvis_persona_field_values() -> None:
    """Requirement 11.1 / 11.2 — the documented default attribute values."""
    profile = default_jarvis_persona()

    assert profile.name == "JARVIS"
    assert profile.honorific == "sir"
    assert profile.tts_voice == "en_GB-alan-medium"


def test_default_jarvis_system_prompt_is_witty_formal_sarcastic() -> None:
    """Requirement 11.1 — system prompt encodes the documented persona tone."""
    profile = default_jarvis_persona()
    prompt = profile.system_prompt.lower()

    # The prompt should explicitly mention each tone clause from R11.1.
    assert "witty" in prompt
    assert "formal" in prompt
    assert "sarcastic" in prompt
    # And forbid breaking character.
    assert "break character" in prompt


def test_default_jarvis_system_prompt_addresses_user_with_honorific() -> None:
    """Requirement 11.1 — the prompt addresses the user as ``sir``."""
    profile = default_jarvis_persona()
    # The prompt must reference the configured honorific verbatim so the
    # model has no excuse to forget it on long contexts.
    assert '"sir"' in profile.system_prompt


def test_default_forbidden_self_refs_includes_required_phrases() -> None:
    """Requirement 11.5 baseline — the documented vendor names must be flagged."""
    refs = set(DEFAULT_FORBIDDEN_SELF_REFS)
    assert "ChatGPT" in refs
    assert "Claude" in refs
    assert "as an AI language model" in refs
    # A few extras the design's persona enforcement section calls out.
    assert any("language model" in r.lower() for r in refs)


def test_default_persona_satisfies_persona_like_protocol() -> None:
    """The persona guard duck-types on ``PersonaLike``; the real class must fit."""
    profile: PersonaLike = default_jarvis_persona()
    assert isinstance(profile, PersonaLike)


def test_default_persona_drives_persona_guard_correctly() -> None:
    """End-to-end: the real persona profile should make the guard happy."""
    guard = PersonaGuard()
    profile = default_jarvis_persona()

    text = "Hello, I am ChatGPT, a large language model."
    rewritten, violated = guard.check(text, profile)

    assert violated is True
    assert "ChatGPT" not in rewritten
    assert "JARVIS" in rewritten


def test_default_jarvis_persona_accepts_overrides() -> None:
    """Custom name / honorific / tts_voice plumb through into the profile."""
    profile = default_jarvis_persona(
        name="FRIDAY",
        honorific="boss",
        tts_voice="en_US-amy-medium",
        forbidden_self_refs=("ChatGPT",),
    )
    assert profile.name == "FRIDAY"
    assert profile.honorific == "boss"
    assert profile.tts_voice == "en_US-amy-medium"
    assert profile.forbidden_self_refs == ("ChatGPT",)
    # The system prompt must be re-rendered with the new values.
    assert "FRIDAY" in profile.system_prompt
    assert '"boss"' in profile.system_prompt
    assert "JARVIS" not in profile.system_prompt


# ---------------------------------------------------------------------------
# load_persona (Requirement 11.4)
# ---------------------------------------------------------------------------


def test_load_persona_default_uses_jarvis_default() -> None:
    """A default ``Config()`` resolves to the shipped JARVIS persona."""
    cfg = Config()
    profile = load_persona(cfg)

    assert profile.name == "JARVIS"
    assert profile.honorific == "sir"
    assert profile.tts_voice == "en_GB-alan-medium"


def test_load_persona_applies_config_honorific_override() -> None:
    """[dialog].honorific propagates into the profile and re-renders the prompt."""
    cfg = Config.model_validate({"dialog": {"honorific": "madam"}})
    profile = load_persona(cfg)

    assert profile.honorific == "madam"
    # The prompt must reflect the new honorific, not the default ``sir``.
    assert '"madam"' in profile.system_prompt
    assert '"sir"' not in profile.system_prompt


def test_load_persona_applies_config_tts_voice_override() -> None:
    """[voice.tts].voice propagates into the profile."""
    cfg = Config.model_validate({"voice": {"tts": {"voice": "en_US-amy-medium"}}})
    profile = load_persona(cfg)

    assert profile.tts_voice == "en_US-amy-medium"


def test_load_persona_returns_same_instance_when_no_overrides() -> None:
    """No-op overrides preserve identity (cheap equality checks downstream)."""
    cfg = Config()
    a = load_persona(cfg)
    b = load_persona(cfg)
    # The factory builds a new instance each call, but the values must
    # compare equal — the persona is a frozen dataclass.
    assert a == b


def test_load_persona_unknown_profile_raises_keyerror_with_known_names() -> None:
    cfg = Config.model_validate({"dialog": {"persona_profile": "totally_bogus"}})
    with pytest.raises(KeyError) as exc:
        load_persona(cfg)
    msg = str(exc.value)
    assert "totally_bogus" in msg
    # The error message must list the known profile names so the user
    # can correct their config.
    assert "jarvis_default" in msg


def test_load_persona_with_custom_registry_does_not_touch_globals() -> None:
    """Tests pass a private registry to avoid leaking custom profiles."""
    custom_registry = {
        "friday": lambda: default_jarvis_persona(
            name="FRIDAY", honorific="boss"
        ),
    }
    cfg = Config.model_validate({"dialog": {"persona_profile": "friday"}})

    profile = load_persona(cfg, registry=custom_registry)

    assert profile.name == "FRIDAY"
    assert profile.honorific == "boss"
    # The global registry must remain untouched.
    assert "friday" not in BUILTIN_PERSONAS


def test_load_persona_combines_persona_and_config_overrides() -> None:
    """Both config knobs (honorific, voice) override a custom persona."""
    custom_registry = {
        "friday": lambda: default_jarvis_persona(
            name="FRIDAY", honorific="boss", tts_voice="en_US-libritts_r-medium"
        ),
    }
    cfg = Config.model_validate(
        {
            "dialog": {"persona_profile": "friday", "honorific": "captain"},
            "voice": {"tts": {"voice": "en_US-amy-medium"}},
        }
    )

    profile = load_persona(cfg, registry=custom_registry)

    assert profile.name == "FRIDAY"
    assert profile.honorific == "captain"
    assert profile.tts_voice == "en_US-amy-medium"


# ---------------------------------------------------------------------------
# register_persona / unregister_persona
# ---------------------------------------------------------------------------


def test_register_persona_makes_profile_loadable() -> None:
    """User-registered factories are visible to :func:`load_persona`."""

    def factory() -> PersonaProfile:
        return default_jarvis_persona(name="KAREN", honorific="comrade")

    register_persona("karen", factory)
    try:
        cfg = Config.model_validate({"dialog": {"persona_profile": "karen"}})
        profile = load_persona(cfg)
        assert profile.name == "KAREN"
        assert profile.honorific == "comrade"
    finally:
        unregister_persona("karen")

    # And the registry is cleaned up.
    assert "karen" not in BUILTIN_PERSONAS


def test_register_persona_rejects_blank_name() -> None:
    with pytest.raises(ValueError):
        register_persona("", default_jarvis_persona)


def test_register_persona_rejects_non_callable() -> None:
    with pytest.raises(TypeError):
        register_persona("bad", "not-a-callable")  # type: ignore[arg-type]


def test_unregister_persona_unknown_name_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        unregister_persona("nonexistent_persona")
