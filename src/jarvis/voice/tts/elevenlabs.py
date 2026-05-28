"""ElevenLabs cloud TTS adapter.

This module implements :class:`ElevenLabsTTS`, an optional cloud-backed
:class:`~jarvis.voice.tts.base.TTSEngine` that synthesizes assistant
responses through the ElevenLabs HTTP API. It is selected via the
``[voice.tts]`` configuration section when ``engine = "elevenlabs"``
(see ``src/jarvis/config/schema.py::VoiceTtsConfig``) and is therefore
an *optional* backend — the local Piper engine remains the default per
Requirement 11.2.

Design notes
------------

* The class extends :class:`~jarvis.voice.tts._cloud_base._CloudTTSEngine`
  so the queue-and-worker scaffolding (and barge-in semantics required
  by Requirement 1.7) is shared with the OpenAI adapter. Only the
  provider-specific HTTP request, response parsing, and audio format
  decisions live in this file.
* Networking uses :mod:`httpx` (already a project runtime dependency).
  The async client is constructed lazily on first synthesis so simply
  *importing* this module — e.g. for type checks — never opens a socket.
* The streaming endpoint
  ``POST /v1/text-to-speech/{voice_id}/stream?output_format=pcm_<rate>``
  returns raw little-endian signed 16-bit mono PCM. The ``pcm_*`` family
  is the only response format that needs no decoder; we request 16 kHz
  by default to match the rest of the voice pipeline (Wake_Word_Detector
  / VAD / STT all run at 16 kHz, so the audio device is already opened
  at that rate when the user is talking to the assistant). The sample
  rate is configurable for installations that prefer the cleaner
  high-rate voices.
* Errors raised during synthesis are logged by the base class and the
  worker advances to the next sentence; the Dialog_Manager surfaces
  user-visible diagnostics when persistent failures warrant.

Validates: Requirement 11.2
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Final

from jarvis.voice.audio_io import AudioFormat
from jarvis.voice.tts._cloud_base import _CloudTTSEngine

if TYPE_CHECKING:  # pragma: no cover - typing only
    import httpx

logger = logging.getLogger(__name__)

__all__ = ["ElevenLabsTTS"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_DEFAULT_BASE_URL: Final[str] = "https://api.elevenlabs.io"

# Sample rates ElevenLabs exposes through the ``output_format=pcm_<rate>``
# parameter. The list is the documented enum at the time of writing and
# is enforced locally so a typo in config produces a clear error before
# the request is sent.
_SUPPORTED_PCM_RATES: Final[frozenset[int]] = frozenset(
    {16000, 22050, 24000, 44100}
)

# Default voice. ``Rachel`` is the long-standing public sample voice;
# users will almost always override this in config to a voice ID that
# matches the configured persona honorific (Requirement 11.2). The
# constant exists only to give the class a sensible default that does
# not require config to instantiate.
_DEFAULT_VOICE_ID: Final[str] = "21m00Tcm4TlvDq8ikWAM"

# Default model. ``eleven_multilingual_v2`` produces the most natural
# British-accented voices for the JARVIS persona; ``eleven_turbo_v2``
# is faster but lower fidelity and is left to be opted into via config.
_DEFAULT_MODEL_ID: Final[str] = "eleven_multilingual_v2"

# Per-request timeout. The ElevenLabs streaming endpoint typically begins
# returning audio within a second; 30 s is a generous ceiling that still
# trips well before pipelines stall indefinitely on a wedged connection.
_DEFAULT_REQUEST_TIMEOUT_S: Final[float] = 30.0


# ---------------------------------------------------------------------------
# ElevenLabsTTS
# ---------------------------------------------------------------------------


class ElevenLabsTTS(_CloudTTSEngine):
    """Cloud TTS adapter speaking through the ElevenLabs API.

    Selected by setting ``[voice.tts] engine = "elevenlabs"`` in
    ``config.toml``; the API key is resolved through the
    :class:`~jarvis.security.credential_store.CredentialStore` at
    application startup and passed as ``api_key`` here. The class never
    persists or logs the key.

    Parameters
    ----------
    api_key:
        ElevenLabs API key (``xi-api-key`` header value). Pulled from
        the Credential_Store by ``app.py`` and never written to disk
        from this module.
    voice_id:
        ElevenLabs voice identifier. Maps to the ``voice`` field of
        ``[voice.tts]`` in config.
    model_id:
        ElevenLabs model identifier (e.g.
        ``eleven_multilingual_v2``).
    sample_rate_hz:
        PCM sample rate to request. Must be one of the rates the
        provider supports. Defaults to ``16000`` so the audio device
        configuration matches the rest of the voice pipeline; users on
        higher-quality speakers may prefer ``24000``.
    speaking_rate:
        Multiplier applied to ``voice_settings.style`` / ``stability``
        is *not* a thing in ElevenLabs; the parameter is accepted for
        configuration symmetry but currently no-ops. Documented here
        explicitly so users do not silently expect rate adjustment.
    base_url:
        Override for the API host. Defaults to the public endpoint;
        primarily useful in tests with a fake server.
    request_timeout_seconds:
        Per-synthesis HTTP timeout. The streaming endpoint must produce
        the first PCM byte before this elapses or the request is
        aborted.
    device, queue_depth:
        Forwarded to the cloud engine base class.
    """

    _provider_name = "elevenlabs"

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str = _DEFAULT_VOICE_ID,
        model_id: str = _DEFAULT_MODEL_ID,
        sample_rate_hz: int = 16000,
        speaking_rate: float = 1.0,
        base_url: str = _DEFAULT_BASE_URL,
        request_timeout_seconds: float = _DEFAULT_REQUEST_TIMEOUT_S,
        device: int | str | None = None,
        queue_depth: int = 64,
    ) -> None:
        if not api_key:
            raise ValueError("ElevenLabsTTS requires a non-empty api_key")
        if not voice_id:
            raise ValueError("ElevenLabsTTS requires a non-empty voice_id")
        if not model_id:
            raise ValueError("ElevenLabsTTS requires a non-empty model_id")
        if sample_rate_hz not in _SUPPORTED_PCM_RATES:
            raise ValueError(
                "ElevenLabsTTS sample_rate_hz must be one of "
                f"{sorted(_SUPPORTED_PCM_RATES)}; got {sample_rate_hz!r}"
            )
        if speaking_rate <= 0.0:
            raise ValueError("speaking_rate must be positive")
        if request_timeout_seconds <= 0.0:
            raise ValueError("request_timeout_seconds must be positive")

        # ElevenLabs PCM responses are 16-bit mono at the requested rate.
        # ``frame_samples`` controls how the AudioPlayer slices writes to
        # the device; 20 ms (sample_rate_hz / 50) is a reasonable hop —
        # large enough to amortize PortAudio write overhead, small
        # enough to keep the barge-in cancellation responsive (the
        # player additionally yields between writes).
        audio_format = AudioFormat(
            sample_rate_hz=sample_rate_hz,
            frame_samples=max(1, sample_rate_hz // 50),
            channels=1,
            sample_width=2,
        )
        super().__init__(
            audio_format=audio_format,
            device=device,
            queue_depth=queue_depth,
        )

        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._sample_rate_hz = sample_rate_hz
        self._speaking_rate = speaking_rate
        self._base_url = base_url.rstrip("/")
        self._request_timeout_seconds = request_timeout_seconds
        self._client: httpx.AsyncClient | None = None

    # -- provider client lifecycle -------------------------------------------

    def _ensure_client(self) -> httpx.AsyncClient:
        """Lazily construct the shared async HTTP client.

        ``httpx`` is imported at function scope so this module remains
        importable on hosts where the optional cloud-tts extras have
        been pruned to avoid pulling the audio stack — the class is
        only *used* when the engine is selected in config, at which
        point ``httpx`` is already a runtime dependency.
        """
        client = self._client
        if client is not None:
            return client
        # Lazy import keeps module import cheap and avoids forcing
        # ``httpx`` to be present for callers that only reference the
        # class for typing purposes.
        import httpx  # noqa: PLC0415 - intentional lazy import

        # Setting the API key as a default header lets us avoid
        # reconstructing the dict on every request, while keeping it out
        # of the per-request kwargs that ultimately end up in logs.
        client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._request_timeout_seconds,
            headers={
                "xi-api-key": self._api_key,
                "Accept": "audio/pcm",
                "Content-Type": "application/json",
            },
        )
        self._client = client
        return client

    async def _close_provider(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()

    # -- synthesis ------------------------------------------------------------

    async def _synthesize(self, text: str) -> bytes:
        """Render ``text`` to raw 16-bit mono PCM via the ElevenLabs API.

        Uses the streaming endpoint so audio bytes arrive incrementally
        and we hand off the assembled buffer to the player as soon as
        the response completes. The ``output_format=pcm_<rate>`` query
        parameter selects raw PCM (no container, no decoder needed).
        """
        client = self._ensure_client()
        body: dict[str, Any] = {
            "text": text,
            "model_id": self._model_id,
        }
        params = {"output_format": f"pcm_{self._sample_rate_hz}"}
        url = f"/v1/text-to-speech/{self._voice_id}/stream"

        # ``stream`` keeps memory bounded for long sentences and lets
        # us surface non-2xx responses with their error body before
        # consuming gigabytes of bytes. We collect the chunks into a
        # single buffer because the base class plays one sentence at
        # a time; the ElevenLabs streaming endpoint typically returns
        # the entire utterance within a few hundred milliseconds.
        chunks: list[bytes] = []
        async with client.stream("POST", url, json=body, params=params) as response:
            if response.status_code >= 400:
                # Read the error body so the log line is actionable.
                # ``aread`` decodes any ``transfer-encoding: chunked``
                # framing into a single bytes payload.
                error_body = await response.aread()
                raise RuntimeError(
                    f"ElevenLabs synthesis failed: HTTP "
                    f"{response.status_code}: {error_body.decode('utf-8', 'replace')[:512]}"
                )
            async for chunk in response.aiter_bytes():
                if chunk:
                    chunks.append(chunk)
        return b"".join(chunks)
