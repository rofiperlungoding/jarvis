"""Spoken notice when :class:`BackendSelector` opens its circuit.

This small helper bridges the synchronous, zero-arg ``on_flip`` callback
exposed by :class:`jarvis.llm.selector.BackendSelector` to the async
:class:`jarvis.voice.tts.base.TTSEngine`. The Dialog_Manager layer
imports it during application wiring (task 19.x) to satisfy
Requirement 12.4: when the cloud Mistral backend is judged unhealthy
and the selector flips to the local Ollama fallback, the user hears a
brief in-character notice rather than wondering why the next response
is taking longer than usual.

Why this lives here rather than inside :class:`BackendSelector`
---------------------------------------------------------------

The selector module deliberately knows nothing about TTS or the persona —
it lives in :mod:`jarvis.llm`, one layer below voice. Embedding a
``TTSEngine`` reference there would invert the dependency graph the
design pins (LLM → Dialog → Voice). Instead, the selector carries a
generic ``Callable[[], None]`` and the dialog layer composes the TTS
notice on top.

Behavioural contract
--------------------

The :class:`BackendSelector` already guarantees the **one-shot**
property: ``on_flip`` fires exactly once per circuit-open transition
(see ``jarvis.llm.selector._trip``). The bridge therefore does not
need its own deduplication — it only has to make sure each flip
results in *at most* one spoken notice, even when the speak hits an
error.

The bridge MUST be:

* **Non-blocking from the LLM call path.** The selector invokes
  ``on_flip`` synchronously from inside its ``_trip`` method, which
  runs on the dialog loop. If we awaited ``tts.speak`` here, the
  dialog turn would block on the audio queue before the fallback
  backend even gets to start streaming — defeating the point of the
  fallback. We schedule the speak as a fire-and-forget task on the
  running event loop instead.

* **Exception isolated.** A failing audio device or a misbehaving TTS
  adapter MUST NOT abort the dialog flow. Exceptions are caught and
  logged at every stage: when scheduling the task (e.g., no running
  loop), and inside the task body around the actual ``speak`` call.

* **Persona-aware.** The notice addresses the user with the persona's
  configured honorific (``"sir"`` by default) so the cinematic JARVIS
  voice stays in character on a fallback.

Validates: Requirement 12.4
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import logging

from jarvis.voice.tts.base import TTSEngine

logger = logging.getLogger(__name__)

# Strong references to in-flight notice tasks. Without this set the GC
# may collect the Task before the speak coroutine completes — Python's
# event loop only keeps weak references to scheduled tasks, so a
# fire-and-forget pattern needs an external pin (see ``asyncio``
# documentation for ``loop.create_task``). We discard each task from
# the set in its done-callback so the set stays bounded by the number
# of *concurrent* notices, which in practice is at most one.
_PENDING_NOTICE_TASKS: set[asyncio.Task[None]] = set()

__all__ = [
    "DEFAULT_FALLBACK_NOTICE_TEMPLATE",
    "build_backend_fallback_notice",
    "format_fallback_notice",
]


#: Default in-character notice template. The ``{honorific}`` slot is
#: filled with the persona's address form (``"sir"`` for the default
#: JARVIS persona). The phrasing matches the design's
#: "Mistral → Local Fallback Flow" diagram verbatim.
DEFAULT_FALLBACK_NOTICE_TEMPLATE: str = (
    "The cloud is being slow, {honorific}. Switching to local."
)


def format_fallback_notice(honorific: str, *, template: str | None = None) -> str:
    """Render the spoken notice for a given honorific.

    Split out so callers (and tests) can render the exact string the
    user will hear without instantiating a callback. The template
    defaults to :data:`DEFAULT_FALLBACK_NOTICE_TEMPLATE`; pass an
    override to localise or customise the wording. The template MUST
    contain the literal token ``{honorific}`` — no other placeholders
    are recognised — so we keep formatting simple and surprise-free.
    """
    effective_template = (
        template if template is not None else DEFAULT_FALLBACK_NOTICE_TEMPLATE
    )
    # ``str.format`` raises KeyError on unknown placeholders and
    # IndexError on positional refs; both are clear signals that the
    # template is malformed. We let those propagate at construction
    # time rather than swallowing them silently.
    return effective_template.format(honorific=honorific)


def build_backend_fallback_notice(
    tts: TTSEngine,
    *,
    honorific: str = "sir",
    template: str | None = None,
) -> Callable[[], None]:
    """Build the ``on_flip`` callback that speaks the fallback notice.

    The returned callable is suitable for
    :class:`jarvis.llm.selector.BackendSelector`'s ``on_flip``
    parameter. Each invocation schedules a single
    :meth:`TTSEngine.speak` on the running event loop and returns
    immediately.

    Parameters
    ----------
    tts:
        The active :class:`TTSEngine` whose ``speak`` queue receives
        the notice. The same engine the :class:`DialogManager` uses
        for ordinary assistant utterances — using a separate channel
        would re-order with respect to the user's current turn.
    honorific:
        Form of address rendered into the notice. Defaults to
        ``"sir"`` to match the default JARVIS persona; pass the
        active :attr:`PersonaProfile.honorific` from
        ``app.py`` to honour user overrides.
    template:
        Optional override for the notice template. Must contain a
        literal ``{honorific}`` placeholder. Defaults to
        :data:`DEFAULT_FALLBACK_NOTICE_TEMPLATE`.

    Returns
    -------
    Callable[[], None]
        A zero-argument callback. Invoking it schedules one TTS speak
        and returns synchronously. Exceptions from scheduling or the
        TTS call itself are logged and swallowed.

    Notes
    -----
    Per-flip one-shot semantics are inherited from
    :class:`BackendSelector`: the selector calls ``on_flip`` exactly
    once on each *transition* into the open state (and not on every
    subsequent request while the circuit remains open). The bridge
    does not add a second layer of deduplication on top of that.
    """
    notice_text = format_fallback_notice(honorific, template=template)

    def _on_flip() -> None:
        # ``asyncio.get_running_loop`` raises ``RuntimeError`` when no
        # event loop is currently running on this thread. The
        # ``BackendSelector`` only calls ``on_flip`` from inside its
        # async ``_stream``, so a missing loop here would indicate the
        # bridge was wired up incorrectly (e.g., invoked from a sync
        # test harness). Treat it as a non-fatal misconfiguration: log
        # and skip rather than tearing the dialog loop down.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(
                "build_backend_fallback_notice: no running event loop; "
                "fallback TTS notice skipped"
            )
            return

        # ``loop.create_task`` schedules the coroutine on the running
        # loop. We hold a strong reference in
        # :data:`_PENDING_NOTICE_TASKS` until the task completes — the
        # event loop itself only keeps weak references, so without an
        # external pin the GC could collect the Task before the speak
        # coroutine finishes. The done-callback removes the task again
        # so the set never grows beyond the (small) number of
        # concurrent notices.
        try:
            task = loop.create_task(
                _speak_notice(tts, notice_text), name="backend-fallback-notice"
            )
        except Exception:
            # Defensive: ``create_task`` itself can raise
            # ``RuntimeError`` if the loop is closing. The dialog flow
            # outranks the notification.
            logger.exception(
                "Failed to schedule backend fallback TTS notice"
            )
            return

        _PENDING_NOTICE_TASKS.add(task)
        task.add_done_callback(_PENDING_NOTICE_TASKS.discard)

    return _on_flip


async def _speak_notice(tts: TTSEngine, text: str) -> None:
    """Speak ``text`` through ``tts``, swallowing any exception.

    Any failure inside ``tts.speak`` (audio device gone, ONNX session
    error, cancelled queue, etc.) is logged and discarded so that a
    misbehaving TTS adapter cannot abort the dialog flow. We
    explicitly let :class:`asyncio.CancelledError` propagate so a
    cooperative shutdown of the dialog loop still works as expected.
    """
    try:
        await tts.speak(text)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception(
            "TTSEngine.speak raised while emitting backend fallback notice"
        )
