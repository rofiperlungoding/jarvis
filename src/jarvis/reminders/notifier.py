"""Toast / TTS notifier for the Reminder_Service.

This module implements :class:`ToastNotifier`, the concrete
notification adapter the :class:`~jarvis.reminders.service.ReminderService`
calls when a reminder, alarm, or timer fires (Requirement 6.5). It
satisfies the structural ``ToastNotifier`` Protocol declared in
:mod:`jarvis.reminders.service` while keeping the reminders package
free of any direct ``win10toast`` import — toast delivery itself is
delegated to :class:`~jarvis.automation.platform.PlatformAdapter.notify`,
which on Windows is wired to ``win10toast.ToastNotifier.show_toast``
(``design.md §Reminder_Service > Notification delivery``).

Why a separate adapter?
-----------------------

``ReminderService`` is platform-neutral persistence + scheduling logic;
it does not — and must not — know how to speak text or how to draw a
toast. This wrapper bridges the two responsibilities:

* **Always** dispatch a system toast through the platform adapter so the
  user gets a visual alert even if they have walked away from the
  microphone.
* **Conditionally** speak the reminder body via the TTS engine *when
  the user is currently engaged in, or has just completed, a
  conversation* (Requirement 6.5 / ``design.md §Reminder_Service``).
  Talking aloud while the user has clearly walked away would be
  startling and would step on whatever is happening in the foreground.

"Currently engaged or just completed a conversation" is determined from
three independent signals, any of which is sufficient:

1. An optional ``conversation_active_callback`` supplied by the
   :class:`~jarvis.dialog.manager.DialogManager` / Voice_Pipeline
   wiring layer. Returning ``True`` means a turn is in progress
   (e.g. STT is streaming, the LLM is composing, the TTS queue is
   non-empty for the current turn).
2. A 30 s "recently active" window driven by
   :meth:`mark_conversation_active`, which the dialog loop calls
   whenever a turn starts or finishes. This is the
   *just-completed-a-conversation* leg from Requirement 6.5.
3. :meth:`TTSEngine.is_playing` reporting ``True`` — the assistant is
   mid-utterance, so by definition the user is in a conversation.

The window length is 30 seconds, matching the
``recently-active conversation (within 30 s)`` clause in
``design.md §Reminder_Service``. It is configurable for tests and for
future tuning, but never accepts a negative value.

The ``mark_conversation_active`` cursor is taken from the injected
:class:`~jarvis.utils.time_source.TimeSource`'s :meth:`monotonic`
clock rather than its wall clock so an NTP correction does not
spuriously open or close the recency window — this is exactly the
"measure elapsed durations" guidance in the ``TimeSource`` docstring.

Validates: Requirement 6.5
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Final

from jarvis.automation.platform import PlatformAdapter
from jarvis.utils.time_source import SystemTimeSource, TimeSource
from jarvis.voice.tts.base import TTSEngine

logger = logging.getLogger(__name__)

__all__ = ["ToastNotifier"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


#: Window, in seconds, during which a recently-finished conversation is
#: still considered "active" for the purposes of Requirement 6.5. Mirrors
#: the ``within 30 s`` clause in ``design.md §Reminder_Service`` and the
#: matching task spec for 15.2.
_RECENTLY_ACTIVE_WINDOW_SECONDS: Final[float] = 30.0


# ---------------------------------------------------------------------------
# ToastNotifier
# ---------------------------------------------------------------------------


class ToastNotifier:
    """Toast + optional-TTS notifier used by :class:`ReminderService`.

    Implements the structural ``ToastNotifier`` Protocol declared in
    :mod:`jarvis.reminders.service`. Instances are cheap to construct;
    no system resources are acquired until :meth:`notify` is called.

    Parameters
    ----------
    platform_adapter:
        Platform abstraction whose :meth:`PlatformAdapter.notify` is
        used to surface the visible toast. On Windows this fans out to
        ``win10toast`` (``design.md §Reminder_Service``); on
        unsupported platforms it raises
        :class:`~jarvis.automation.platform.PlatformNotSupportedError`,
        which we log and swallow rather than propagating into the
        scheduler tick.
    tts_engine:
        Voice engine used to speak the reminder body when the user is
        engaged. Only :meth:`TTSEngine.speak` and
        :meth:`TTSEngine.is_playing` are exercised here.
    time_source:
        Injectable clock; defaults to :class:`SystemTimeSource`. The
        recency window measurement uses
        :meth:`TimeSource.monotonic` so a wall-clock NTP correction
        never spuriously opens or closes the window.
    conversation_active_callback:
        Optional probe supplied by the Voice_Pipeline /
        :class:`DialogManager`. ``True`` means a turn is currently in
        progress and the reminder body SHOULD be spoken.
    recently_active_window_seconds:
        Length of the post-conversation grace window (Requirement 6.5).
        Defaults to 30 seconds; rejected if negative. Tests may shrink
        this to zero to assert the boundary behaviour.

    Notes
    -----
    The notifier is intentionally permissive about exceptions raised by
    its collaborators: a flaky toast subsystem (e.g. a wonky shell
    integration) MUST NOT prevent the spoken announcement, and a
    transient TTS failure MUST NOT prevent the visual toast. Both
    failures are logged at ``exception`` level and swallowed so the
    APScheduler tick can complete without re-firing the job under
    APScheduler's misfire policy.
    """

    def __init__(
        self,
        platform_adapter: PlatformAdapter,
        tts_engine: TTSEngine,
        time_source: TimeSource | None = None,
        conversation_active_callback: Callable[[], bool] | None = None,
        *,
        recently_active_window_seconds: float = _RECENTLY_ACTIVE_WINDOW_SECONDS,
    ) -> None:
        if not isinstance(recently_active_window_seconds, (int, float)) or isinstance(
            recently_active_window_seconds, bool
        ):
            raise TypeError(
                "recently_active_window_seconds must be a real number; got "
                f"{type(recently_active_window_seconds).__name__}"
            )
        if recently_active_window_seconds < 0:
            raise ValueError(
                "recently_active_window_seconds must be non-negative; got "
                f"{recently_active_window_seconds!r}"
            )
        if conversation_active_callback is not None and not callable(
            conversation_active_callback
        ):
            raise TypeError(
                "conversation_active_callback must be callable or None; got "
                f"{type(conversation_active_callback).__name__}"
            )

        self._platform_adapter: PlatformAdapter = platform_adapter
        self._tts_engine: TTSEngine = tts_engine
        self._time_source: TimeSource = time_source or SystemTimeSource()
        self._conversation_active_callback: Callable[[], bool] | None = (
            conversation_active_callback
        )
        self._window_seconds: float = float(recently_active_window_seconds)

        # Monotonic timestamp of the most recent
        # ``mark_conversation_active`` call. ``None`` means no
        # conversation has been observed yet; in that case only the
        # callback / ``is_playing`` signals can mark the conversation
        # as active.
        self._last_active_monotonic: float | None = None

    # ------------------------------------------------------------------ public

    def mark_conversation_active(self) -> None:
        """Stamp "now" as the most recent conversation activity.

        Called by the Dialog_Manager whenever a turn boundary is
        observed (utterance captured, response started, response
        finished). This is what feeds the
        ``recently_active_window_seconds`` leg of Requirement 6.5: a
        reminder firing within ``recently_active_window_seconds`` of
        the last call here will be spoken via TTS in addition to the
        visual toast.
        """

        self._last_active_monotonic = self._time_source.monotonic()

    async def notify(self, title: str, body: str) -> None:
        """Deliver a reminder notification.

        Always dispatches the visual toast through
        :meth:`PlatformAdapter.notify`. Additionally speaks ``body``
        via the TTS engine if any of the conversation-active signals
        is currently truthy (see the module docstring for the
        three-signal definition).

        Parameters
        ----------
        title:
            Toast title. Forwarded verbatim to the platform adapter;
            empty strings are accepted because some platforms render an
            untitled toast by design.
        body:
            Toast body. Used both as the toast text *and* — when
            non-empty — as the text spoken by the TTS engine. An empty
            body skips the TTS call (there is nothing to say) but
            still fires the toast for parity with the design.

        Raises
        ------
        TypeError
            If ``title`` or ``body`` is not a string. The reminder
            service hands us validated metadata so this is purely a
            defensive check; surfacing the error early is preferable
            to letting it land inside the platform adapter.
        """

        if not isinstance(title, str):
            raise TypeError(f"title must be str; got {type(title).__name__}")
        if not isinstance(body, str):
            raise TypeError(f"body must be str; got {type(body).__name__}")

        # 1. Visual toast — always attempted (Requirement 6.5: "SHALL emit
        #    a Windows toast notification"). The PlatformAdapter raises
        #    PlatformNotSupportedError on hosts where toast delivery is
        #    not implemented; that is a known operational state on
        #    non-Windows hosts (the Mac/Linux adapters land in a later
        #    task) and must not prevent the TTS announcement.
        try:
            await self._platform_adapter.notify(title, body)
        except Exception:
            logger.exception(
                "platform_adapter.notify failed for reminder title=%r; "
                "continuing to TTS path so the user is still notified.",
                title,
            )

        # 2. Spoken announcement — gated on the conversation signals.
        if not body:
            # No text to speak. The visual toast above already covered
            # the user notification; bail out early without consulting
            # the TTS engine to avoid a no-op ``speak("")`` that some
            # backends reject.
            return

        if not self._is_conversation_active():
            return

        try:
            await self._tts_engine.speak(body)
        except Exception:
            logger.exception(
                "tts_engine.speak failed for reminder title=%r; "
                "visual toast already delivered.",
                title,
            )

    # ---------------------------------------------------------------- helpers

    def _is_conversation_active(self) -> bool:
        """Return ``True`` when the user is engaged or just-finished engaging.

        Combines the three independent signals listed in the module
        docstring with short-circuit evaluation: as soon as any one
        signal indicates activity, the rest are skipped. Each signal
        is wrapped in a defensive ``try/except`` so a buggy collaborator
        cannot prevent the others from being consulted — the worst-case
        behaviour for any single failing signal is that we fall through
        to the next one.
        """

        # Signal 1 — explicit callback from the Voice_Pipeline /
        # DialogManager, when wired. This is the authoritative signal:
        # if the dialog layer says "the user is mid-turn", we believe it
        # without further questions.
        callback = self._conversation_active_callback
        if callback is not None:
            try:
                if callback():
                    return True
            except Exception:
                logger.exception(
                    "conversation_active_callback raised; "
                    "treating as not-active for this signal and "
                    "falling through to other signals."
                )

        # Signal 2 — the assistant is currently speaking. By definition,
        # if TTS is mid-utterance the conversation is active even if
        # nobody bothered to register a callback. ``is_playing`` is
        # documented as a synchronous, side-effect-free probe so we can
        # call it freely on the scheduler tick.
        try:
            if self._tts_engine.is_playing():
                return True
        except Exception:
            logger.exception(
                "tts_engine.is_playing raised; "
                "treating as not-playing for this signal."
            )

        # Signal 3 — within the recency window of the last marked
        # turn boundary. Use the monotonic clock so the window is
        # immune to wall-clock jumps (NTP correction, DST), per the
        # TimeSource design guidance.
        last = self._last_active_monotonic
        if last is None:
            return False
        try:
            elapsed = self._time_source.monotonic() - last
        except Exception:
            # An exotic TimeSource implementation could in principle
            # raise; tolerate it the same way as the other signals
            # rather than letting a clock fault drop the toast tick.
            logger.exception(
                "time_source.monotonic raised; "
                "treating recency window as expired."
            )
            return False
        # ``elapsed`` is non-negative for any well-behaved monotonic
        # clock; the explicit lower bound defends against a misbehaving
        # fake that returns a smaller value than the one we stored.
        return 0.0 <= elapsed <= self._window_seconds
