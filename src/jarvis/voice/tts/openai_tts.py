"""OpenAI cloud TTS adapter.

This module implements :class:`OpenAITTS`, an optional cloud-backed
:class:`~jarvis.voice.tts.base.TTSEngine` that synthesizes assistant
responses through the OpenAI Audio Speech API. It is selected via the
``[voice.tts]`` configuration section when ``engine = "openai"`` (see
``src/jarvis/config/schema.py::VoiceTtsConfig``).

Design notes
------------

* The class extends :class:`~jarvis.voice.tts._cloud_base._CloudTTSEngine`
  so the queue + worker scaffolding (and barge-in semantics required by
  Requirement 1.7) is shared with :class:`ElevenLabsTTS`. Only the
  provider-specific synthesis call lives in this file.
* The :mod:`openai` SDK is **lazy-imported** inside
  :meth:`_synthesize` / :meth:`_ensure_client`. The package is declared
  as an *optional* runtime dependency under the ``cloud-tts`` extra
  (see ``pyproject.toml``); installations that never select the OpenAI
  TTS engine should not be required to install ``openai`` just to
  import :mod:`jarvis.voice.tts`.
* The OpenAI Speech API supports ``response_format="pcm"`` which
  returns raw 24 kHz signed-16-bit mono PCM. Using ``pcm`` avoids the
  decode step that ``mp3`` / ``opus`` / ``aac`` would require, which
  matches the project's "no extra audio dependencies" philosophy. The
  sample rate is fixed at 24 kHz by the OpenAI API for the ``pcm``
  format, so the parameter is not exposed to callers.
* The OpenAI SDK exposes ``client.audio.speech.with_streaming_response``
  which returns an async context manager. We use the streaming variant
  so we can iterate over response bytes — even though we currently
  buffer one full sentence before handing it to the player, this keeps
  the door open to chunked playback without changing the wire-level
  call site.

Validates: Requirement 11.2
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Final

from jarvis.voice.audio_io import AudioFormat
from jarvis.voice.tts._cloud_base import _CloudTTSEngine

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

__all__ = ["OpenAITTS"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# OpenAI's ``response_format="pcm"`` is documented as raw 24 kHz signed
# 16-bit little-endian mono PCM. Anything else would require a decoder,
# which is outside the project's runtime dependency list.
_OPENAI_PCM_SAMPLE_RATE_HZ: Final[int] = 24000

# Default model. ``tts-1`` is the lower-latency variant; ``tts-1-hd`` is
# higher fidelity but slower. The project favours conversational latency
# (Requirement 12.1, 800 ms wake-to-response budget) so the default is
# ``tts-1``.
_DEFAULT_MODEL: Final[str] = "tts-1"

# Default voice. ``onyx`` is the closest match to the JARVIS persona's
# mature, calm tone (Requirement 11.2). Users will typically override
# via the ``voice`` field of ``[voice.tts]``.
_DEFAULT_VOICE: Final[str] = "onyx"

# Per-request timeout. Speech endpoints typically return within a couple
# of seconds; 30 s is a generous ceiling that still trips well before a
# wedged connection stalls the dialog loop indefinitely.
_DEFAULT_REQUEST_TIMEOUT_S: Final[float] = 30.0


# ---------------------------------------------------------------------------
# OpenAITTS
# ---------------------------------------------------------------------------


class OpenAITTS(_CloudTTSEngine):
    """Cloud TTS adapter speaking through the OpenAI Audio Speech API.

    Selected by setting ``[voice.tts] engine = "openai"`` in
    ``config.toml``; the API key is resolved through the
    :class:`~jarvis.security.credential_store.CredentialStore` at
    application startup and passed as ``api_key`` here. This class
    never persists or logs the key.

    Parameters
    ----------
    api_key:
        OpenAI API key. Pulled from the Credential_Store by ``app.py``.
    voice:
        OpenAI voice identifier (``alloy``, ``echo``, ``fable``,
        ``onyx``, ``nova``, ``shimmer`` at the time of writing). Maps
        to the ``voice`` field of ``[voice.tts]`` in config.
    model:
        OpenAI model identifier. ``tts-1`` (low latency) or
        ``tts-1-hd`` (higher fidelity).
    speaking_rate:
        Multiplier passed to the API as ``speed``. The OpenAI API
        accepts values in ``[0.25, 4.0]``; we clamp at construction.
    base_url:
        Override for the API host. Defaults to the OpenAI public API;
        primarily useful in tests with a fake server (or for
        OpenAI-compatible deployments).
    organization:
        Optional ``OpenAI-Organization`` header value.
    request_timeout_seconds:
        Per-synthesis HTTP timeout.
    device, queue_depth:
        Forwarded to the cloud engine base class.
    """

    _provider_name = "openai-tts"

    def __init__(
        self,
        *,
        api_key: str,
        voice: str = _DEFAULT_VOICE,
        model: str = _DEFAULT_MODEL,
        speaking_rate: float = 1.0,
        base_url: str | None = None,
        organization: str | None = None,
        request_timeout_seconds: float = _DEFAULT_REQUEST_TIMEOUT_S,
        device: int | str | None = None,
        queue_depth: int = 64,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAITTS requires a non-empty api_key")
        if not voice:
            raise ValueError("OpenAITTS requires a non-empty voice")
        if not model:
            raise ValueError("OpenAITTS requires a non-empty model")
        if speaking_rate <= 0.0:
            raise ValueError("speaking_rate must be positive")
        if request_timeout_seconds <= 0.0:
            raise ValueError("request_timeout_seconds must be positive")

        # OpenAI's PCM response is 24 kHz / 16-bit / mono. We do not
        # expose sample_rate as a parameter because the API does not
        # let us select it for ``response_format="pcm"``; using any
        # other rate would require a resampler (out of scope and
        # latency-prohibitive).
        audio_format = AudioFormat(
            sample_rate_hz=_OPENAI_PCM_SAMPLE_RATE_HZ,
            # 20 ms hop — see ElevenLabsTTS for rationale.
            frame_samples=_OPENAI_PCM_SAMPLE_RATE_HZ // 50,
            channels=1,
            sample_width=2,
        )
        super().__init__(
            audio_format=audio_format,
            device=device,
            queue_depth=queue_depth,
        )

        self._api_key = api_key
        self._voice = voice
        self._model = model
        # The OpenAI documented ``speed`` range is [0.25, 4.0]. Clamp
        # rather than reject so users do not get a startup error from
        # a slightly out-of-range value pasted from another tool.
        self._speed = max(0.25, min(4.0, speaking_rate))
        self._base_url = base_url
        self._organization = organization
        self._request_timeout_seconds = request_timeout_seconds
        self._client: AsyncOpenAI | None = None

    # -- provider client lifecycle -------------------------------------------

    def _ensure_client(self) -> AsyncOpenAI:
        """Lazily construct the shared :class:`openai.AsyncOpenAI` client.

        Imported at function scope so installations without the
        ``cloud-tts`` extra can still import :mod:`jarvis.voice.tts`
        without paying the ``openai`` import cost or pulling in a
        package they will never use.
        """
        client = self._client
        if client is not None:
            return client
        try:
            from openai import AsyncOpenAI  # noqa: PLC0415 - lazy import
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "OpenAITTS requires the optional 'openai' package; install "
                "the project with the 'cloud-tts' extra to use this engine."
            ) from exc

        # ``base_url=None`` preserves the SDK default; passing ``None``
        # explicitly through is harmless and avoids branching.
        client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            organization=self._organization,
            timeout=self._request_timeout_seconds,
        )
        self._client = client
        return client

    async def _close_provider(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            # ``AsyncOpenAI`` exposes ``close`` (it manages an internal
            # ``httpx.AsyncClient``); guarding with ``getattr`` keeps
            # us forward-compatible if the SDK renames the method.
            close = getattr(client, "close", None)
            if callable(close):
                await close()

    # -- synthesis ------------------------------------------------------------

    async def _synthesize(self, text: str) -> bytes:
        """Render ``text`` to raw 24 kHz / 16-bit / mono PCM via OpenAI.

        Uses ``response_format="pcm"`` which the OpenAI API documents as
        24 kHz signed 16-bit little-endian mono — matching the
        :class:`AudioFormat` configured at construction. The streaming
        response variant is used so we can iterate bytes without
        materializing the entire payload twice; we still buffer one
        full sentence before handing off to the player because the
        base class plays one sentence at a time.
        """
        client = self._ensure_client()

        # ``with_streaming_response.create`` returns an async context
        # manager whose response object exposes ``iter_bytes`` for
        # incremental consumption.
        speech_kwargs: dict[str, Any] = {
            "model": self._model,
            "voice": self._voice,
            "input": text,
            "response_format": "pcm",
            "speed": self._speed,
        }

        chunks: list[bytes] = []
        async with client.audio.speech.with_streaming_response.create(
            **speech_kwargs,
        ) as response:
            async for chunk in response.iter_bytes():
                if chunk:
                    chunks.append(chunk)
        return b"".join(chunks)
