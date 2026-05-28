"""Optional cloud Speech-to-Text engine stub.

This module hosts :class:`CloudSTT`, the design's placeholder for a
cloud-hosted transcription backend (e.g., OpenAI Whisper API). The
JARVIS voice pipeline is local-first by default: :class:`CloudSTT` is
*not* on the wake-to-response path unless the user explicitly opts in
via ``[voice.stt] local_only = false`` and ``engine = "cloud"`` in the
TOML config.

The single behavioral guarantee this stub must hold today is the
privacy gate: when ``voice.stt.local_only`` is true, instantiating
:class:`CloudSTT` MUST fail at construction time so no audio is ever
shipped off-device. That gate is the runtime mirror of the static
config-validation rule already enforced by
:meth:`jarvis.config.schema.VoiceSttConfig._local_only_blocks_cloud_stt`
and is what implements Requirement 13.2 ("the system MUST process voice
locally and MUST NOT transmit raw audio to any cloud service") at the
engine boundary.

The actual network transcription call is intentionally left as a
:class:`NotImplementedError`; wiring a concrete provider is a follow-up
task and out of scope for the stub. This keeps the import graph clean
(no optional cloud SDK dependency required) while still letting the
bootstrap layer resolve a "cloud" engine alias to a real class.

Requirement IDs referenced below map to ``requirements.md``:

* Requirement 13.2 — privacy mode: local processing only; cloud STT
  must be refused while ``voice.stt.local_only=true``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from jarvis.voice.stt.base import AudioBuffer, STTEngine, Transcript

if TYPE_CHECKING:
    # Imported only for type-checking to keep the runtime import graph of
    # this stub free of the full pydantic config module. Bootstrappers in
    # ``app.py`` already import the config eagerly, so the forward ref is
    # never an issue at instantiation time.
    from jarvis.config.schema import VoiceSttConfig


__all__ = [
    "CloudSTT",
    "CloudSTTDisabledError",
]


# Sentinel used for the ``local_only`` keyword so we can distinguish
# "caller passed local_only=False explicitly" from "caller did not pass
# local_only and expects us to read it from config". A plain ``None``
# default would conflate the two and silently ignore an explicit False
# from the caller.
_UNSET: Final[object] = object()


class CloudSTTDisabledError(RuntimeError):
    """Raised when :class:`CloudSTT` is constructed under privacy mode.

    The error is a :class:`RuntimeError` subclass (not a
    :class:`ValueError`) because the failure is a *policy* refusal at
    runtime — the caller's arguments are well-formed; the privacy
    setting forbids the engine from existing at all. Bootstrap code in
    ``app.py`` can catch this specifically to surface a clear "cloud STT
    is blocked by voice.stt.local_only=true" message and fall back to
    the local engine.
    """


class CloudSTT(STTEngine):
    """Optional cloud-backed STT engine stub.

    The constructor enforces Requirement 13.2: if ``local_only`` is
    truthy (either passed directly or read from ``config.local_only``),
    a :class:`CloudSTTDisabledError` is raised before any provider state
    is initialized and before any network resource is reserved. This
    guarantees no audio buffer can ever reach this class while privacy
    mode is on, even if a future bug allowed the static config validator
    to be bypassed.

    The class implements :class:`STTEngine` structurally (the protocol
    is :func:`typing.runtime_checkable`); :meth:`transcribe` is a stub
    that raises :class:`NotImplementedError` until a concrete cloud
    provider is wired in. That keeps the import surface and dependency
    set minimal — picking a provider (OpenAI Whisper, Azure Speech,
    Deepgram, etc.) is a separate task with its own SDK pulls.

    Example::

        from jarvis.config.schema import VoiceSttConfig
        from jarvis.voice.stt.cloud import CloudSTT, CloudSTTDisabledError

        cfg = VoiceSttConfig(local_only=False, engine="cloud")
        engine = CloudSTT(cfg)            # ok

        cfg = VoiceSttConfig(local_only=True)  # privacy mode
        try:
            CloudSTT(cfg)
        except CloudSTTDisabledError:
            ...                           # fall back to FasterWhisperSTT

    Args:
        config: Optional :class:`~jarvis.config.schema.VoiceSttConfig`.
            When provided, its ``local_only`` attribute is consulted as
            the default privacy gate value. The full config is also
            stashed for later use by a concrete implementation (model
            choice, language, etc.).
        local_only: Optional explicit privacy override. When supplied,
            it takes precedence over ``config.local_only``. This lets
            callers gate the engine without constructing a full config
            object (useful in unit tests). Passing both is allowed; the
            keyword wins, mirroring how CLI flags override config files.

    Raises:
        CloudSTTDisabledError: When the resolved ``local_only`` value is
            truthy. Requirement 13.2.
        TypeError: When neither ``config`` nor ``local_only`` is
            supplied; the constructor needs at least one source for the
            privacy gate decision so it cannot silently default to "off"
            and let cloud audio leave the device.
    """

    def __init__(
        self,
        config: VoiceSttConfig | None = None,
        *,
        local_only: bool | object = _UNSET,
    ) -> None:
        # Resolve the privacy gate. The keyword overrides the config so
        # tests and bootstrap shims can force a known value without
        # constructing a full pydantic model.
        if local_only is _UNSET:
            if config is None:
                raise TypeError(
                    "CloudSTT requires either a `config` argument or an "
                    "explicit `local_only` keyword so the privacy gate "
                    "(Requirement 13.2) can be evaluated."
                )
            resolved_local_only = bool(config.local_only)
        else:
            # ``local_only`` is either a real bool from a caller or our
            # sentinel; the sentinel branch is handled above.
            resolved_local_only = bool(local_only)

        if resolved_local_only:
            # Requirement 13.2: never instantiate a cloud STT engine
            # while privacy mode is enabled. Raising here, before any
            # provider client or audio buffer is bound, guarantees the
            # engine cannot transmit raw audio off-device.
            raise CloudSTTDisabledError(
                "CloudSTT cannot be instantiated while "
                "voice.stt.local_only=true (Requirement 13.2). "
                "Set voice.stt.local_only=false in your config to opt "
                "into a cloud STT engine, or use FasterWhisperSTT."
            )

        # Stash the config for a future concrete implementation. The
        # stub does not introspect it further; the assignment exists so
        # the eventual real CloudSTT can pick up model/language/etc.
        # without an ABI change.
        self._config = config

    async def transcribe(
        self,
        audio: AudioBuffer,
        language: str,
    ) -> Transcript:
        """Transcribe a captured utterance via a cloud provider.

        The stub deliberately refuses to perform any work: raising
        :class:`NotImplementedError` makes accidental use during the
        local-first MVP loud and immediate, rather than silently
        producing an empty transcript that the Dialog_Manager would
        gate on as low-confidence (Requirement 1.8).

        A concrete implementation will replace this body with a call to
        the configured cloud provider. The signature is fixed by the
        :class:`STTEngine` Protocol and must not change.
        """
        raise NotImplementedError(
            "CloudSTT is a stub; no cloud STT provider has been wired "
            "yet. Use FasterWhisperSTT for local transcription."
        )
