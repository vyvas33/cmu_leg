"""Operator intent: the two toggles a human flips to move between modes.

The baseline exposes exactly two, as `Constant` blocks in Get Prosthesis State
that an operator changes live from the Simulink Real-Time host:
`standing_mode_toggle` and `zero_torque_toggle` (ours is `hold_toggle`,
named for what the drives actually do -- see types.commands.Mode). They are
LEVEL-triggered --
the toggle only expresses intent, and the mode FSM's guards decide whether the
transition is allowed. That is why a raw Constant with no debounce is enough,
and why standing_toggle going false does not immediately start walking.

Everything here is a source of that same pair. The mode FSM consumes
OperatorInput and cannot tell them apart, so swapping sources never touches
control logic:

  TimedOperator     scripted, a pure function of t. Deterministic, so this is
                    what replay harnesses and automated tests use. Also the
                    drop-in for the current `state.standing = t > warmup`.
  TerminalOperator  keypresses, for live bench work.

`seq` increments on every read so the consumer can detect a source that has
stopped updating. A stale operator must fall to hold -- never to
walking, never "hold last".

`swing_seq` is the one edge-triggered signal here, and it is carried as a
counter rather than a bool so no press is lost between control ticks and no
debounce is needed. It exists for a leg hanging in a harness, where the foot
never loads, contact never asserts and the gait FSM therefore never produces a
toe-off of its own.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorInput:
    """One sample of operator intent.

    Defaults match the baseline's Constant blocks: `standing_mode_toggle = 1`,
    `zero_torque_toggle = 0`. So standing is REQUESTED from the start and the
    operator flips it OFF to walk -- the toggle is "stay standing", not "start
    standing". The mode FSM still holds in INITIALIZING until the drives report
    control_enabled, which is the other half of the baseline's
    `GetInitialState -> StandingMode [control_enabled && standing_mode_toggle]`.

    That means the logistic wait lives entirely in low_warmup_s (when the drives
    energize); standing follows as soon as they do, then ramps to pose and holds
    until someone asks for walking.
    """
    standing_toggle: bool = True
    hold_toggle: bool = False
    seq: int = 0

    # Bench swing trigger. `swing_seq` increments once per request; a consumer
    # fires on the increment. `swing_duration` is the swing length asked for,
    # in seconds, before the [swing_duration_min, swing_duration_max] clamp.
    swing_seq: int = 0
    swing_duration: float = 0.6


class OperatorSource:
    """Base class. `read(t)` is called once per control tick and must not block."""

    def read(self, t: float) -> OperatorInput:
        raise NotImplementedError

    def close(self) -> None:
        pass


class TimedOperator(OperatorSource):
    """Toggles on a fixed schedule. Deterministic and headless.

    Bare `TimedOperator()` is the default posture: stand as soon as the drives
    allow, never walk, never hold -- which is what an unattended run wants.

    Args:
        stand_at: request standing from this time on [s]. Defaults to 0 because
            standing is the resting request; the drives, not this, decide when
            it can actually begin.
        walk_at: drop standing_toggle at this time, arming standing -> walking
            (which still needs the leg unloaded). None = never walk.
        hold_at: raise hold_toggle at this time. None = never.
        swing_at: times to request a swing. This is the scripted stand-in for
            the space key, and it is what a spawned process has to use --
            multiprocessing replaces a child's stdin with devnull, so
            TerminalOperator can never come up there.
        swing_duration: seconds asked for on each of those, before the clamp.
    """

    def __init__(self, stand_at: float = 0.0, walk_at: float | None = None,
                 hold_at: float | None = None,
                 swing_at: tuple[float, ...] = (),
                 swing_duration: float = 0.6):
        self._stand_at = stand_at
        self._walk_at = walk_at
        self._hold_at = hold_at
        self._swing_at = tuple(sorted(swing_at))
        self._swing_duration = swing_duration
        self._seq = 0

    def read(self, t: float) -> OperatorInput:
        self._seq += 1
        standing = t >= self._stand_at
        if self._walk_at is not None and t >= self._walk_at:
            standing = False
        hold = self._hold_at is not None and t >= self._hold_at
        # Counting elapsed entries keeps swing_seq monotonic without carrying
        # state, so a missed tick cannot lose a request.
        swing_seq = sum(1 for s in self._swing_at if t >= s)
        return OperatorInput(standing, hold, self._seq,
                             swing_seq, self._swing_duration)


class TerminalOperator(OperatorSource):
    """Keypresses from the controlling terminal, read on a daemon thread.

    Keys:
        s     request standing   (standing_toggle = True)
        w     request walking    (standing_toggle = False; still needs ~contact)
        z     toggle hold
        q     hold, latched -- the panic key
        space fire one swing, at the current duration
        [ ]   shorten / lengthen that duration by 0.1 s

    The read blocks in its own thread rather than polling, so it costs the
    control loop nothing: a blocking stdin read releases the GIL and only wakes
    on input.

    Needs a tty. If stdin is not one -- piped, or a child process that did not
    inherit it -- construction degrades to a source that never changes state
    and says so, rather than failing at some later moment. In the two-process
    runtime that means this belongs wherever stdin actually lives; check
    `.available` after constructing.
    """

    def __init__(self, stdin=None, swing_duration: float = 0.6):
        import sys
        self._in = stdin or sys.stdin
        self._standing = True     # baseline default: standing requested
        self._hold = False
        self._seq = 0
        self._swing_seq = 0
        self._swing_duration = swing_duration
        self._stop = False
        self.available = False
        self._restore = None

        try:
            if not self._in.isatty():
                raise OSError("stdin is not a tty")
            import termios, tty, threading
            self._fd = self._in.fileno()
            self._termios = termios
            self._restore = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            self.available = True
            print("[operator] keys: s=stand  w=walk  z=hold  q=panic  "
                  "space=swing  []=duration")
        except Exception as exc:      # noqa: BLE001 - any failure means no tty
            print(f"[operator] terminal input unavailable ({exc}); "
                  f"toggles will stay at their defaults")

    def _loop(self) -> None:
        while not self._stop:
            ch = self._in.read(1)
            if not ch:
                break
            c = ch.lower()
            if c == "s":
                self._standing, self._hold = True, False
            elif c == "w":
                self._standing = False
            elif c == "z":
                self._hold = not self._hold
            elif c == "q":
                self._hold = True         # latched: only 's' clears it
            elif c == " ":
                self._swing_seq += 1
                print(f"[operator] swing {self._swing_seq} "
                      f"({self._swing_duration:.2f} s)")
            elif c in "[]":
                step = -0.1 if c == "[" else 0.1
                self._swing_duration = round(self._swing_duration + step, 2)
                print(f"[operator] swing duration {self._swing_duration:.2f} s")

    def read(self, t: float) -> OperatorInput:
        self._seq += 1
        return OperatorInput(self._standing, self._hold, self._seq,
                             self._swing_seq, self._swing_duration)

    def close(self) -> None:
        self._stop = True
        if self._restore is not None:
            try:
                self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN,
                                        self._restore)
            except Exception:         # noqa: BLE001 - best effort on teardown
                pass
