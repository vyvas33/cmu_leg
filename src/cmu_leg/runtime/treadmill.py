"""
Treadmill control for the runtime, in two pieces:

  BeltSupervisor  the belts driven FROM A RUN, stepped by the parent in
                  launch.py. This is what a walking test uses.
  main()          a standalone script that runs the belts on their own, with
                  no leg involved. The bring-up tool and the manual fallback.

--- the script ---------------------------------------------------------------

Run the Bertec treadmill at a set speed until Ctrl+C, then ramp it to a stop.
The belts start START_IN_S seconds after launch, to get on the treadmill.

  uv run python -m cmu_leg.runtime.treadmill                     # 0.8 m/s, 0.5 m/s^2
  uv run python -m cmu_leg.runtime.treadmill --speed 1.0 --acc 0.3
  uv run python -m cmu_leg.runtime.treadmill --host <PC_IP>      # default params.TREADMILL.host

Both belts run at the same speed; positive = person walking forward. --acc is
used for both the ramp up and the ramp down. To change speed, Ctrl+C and
re-run. Ctrl+C during the countdown aborts without sending anything.

Control Panel setup (Ready for Remote Control, etc.) is in
hardware/treadmill/bertec_treadmill.py. Closing the connection does NOT stop
the belts, so this script always sends a stop on the way out; if it is killed
hard (kill -9, power loss, Wi-Fi drop) the belts keep running. Keep the E-stop
in hand.
"""
import argparse
import math
import time
from enum import IntEnum

from cmu_leg.hardware.treadmill.bertec_treadmill import BertecTreadmill
from cmu_leg.config.params import TREADMILL, RUNTIME, STANDING
from cmu_leg.utility.recorder import Recorder

START_IN_S = 10.0          # s from launch to the belts starting (script only)
PRINT_PERIOD_S = 0.5


class BeltState(IntEnum):
    """Where the supervisor is. DONE is terminal on purpose: belts that restart
    themselves after a fault are a surprise, and a surprise is the one thing a
    moving floor must never be. Re-run to get them back."""
    DISABLED = 0     # not opted in; the socket is never opened
    WAITING = 1      # connected, waiting for the leg to settle in standing
    COUNTDOWN = 2    # settled; announcing, belts about to move
    RUNNING = 3      # commanded to speed
    STOPPING = 4     # stop sent, waiting for the belts to read zero
    DONE = 5         # stopped, or faulted; terminal


class BeltSupervisor:
    """The belts, slaved to a run, owned by the PARENT process.

    Not in the control process: `connect()` blocks for up to TREADMILL.timeout_s
    and a send is a syscall with no latency bound, which a pinned 500 Hz
    SCHED_FIFO loop cannot absorb. The parent already runs a 0.5 s monitor loop,
    already holds b1/b3, and is the only process with a tty -- so it is the only
    place where "stop the belts" can be made to happen on every exit path.

    Reads the leg over b3 (mode, and `seq` as a liveness signal that tells a
    HUNG control process from a dead one -- `Process.is_alive()` cannot) and b1
    (contact, only if require_contact). Writes nothing back: the belts never
    command the leg.

    The gate is two-sided. TREADMILL.enabled is the deliberate opt-in and
    RUNTIME.arm_motion is the veto; a run with the leg unarmed can never move
    the belts.

    Start: `mode.actuate` held CONTINUOUSLY for STANDING.duration -- the standing
    pose ramp, so "the leg has finished standing up" -- then a countdown. The
    dwell is continuous rather than latched so a drop to HOLD restarts the
    countdown instead of letting the belts fire the moment the leg recovers.

    Stop: mode leaving STANDING/WALKING, b3 going stale, or shutdown(). Running
    through STANDING<->WALKING is deliberate: you need the belt moving BEFORE you
    can take the first step, so standing on a moving belt is a normal part of the
    sequence, and ending a trial is a deliberate act (hold, or Ctrl-C).
    """

    def __init__(self, cfg=TREADMILL, armed: bool | None = None,
                 run_dir=None):
        self._cfg = cfg
        armed = RUNTIME.arm_motion if armed is None else armed
        self._state = (BeltState.WAITING if (cfg.enabled and armed)
                       else BeltState.DISABLED)
        self._settle = STANDING.duration
        self._run_dir = run_dir
        self._tm = None
        self._log = Recorder("treadmill")
        self._actuate_for = 0.0    # continuous seconds of a belt-safe leg state
        self._t_prev = None
        self._t_mark = 0.0         # countdown / stop start
        self._last_seq = None
        self._stale = 0
        self._next_tick = 0.0      # next whole second to print during countdown
        self._why = ""

    # ------------------------------------------------------------- lifecycle

    def connect(self) -> None:
        """Open the socket BEFORE anything is spawned, so a wrong host or a
        Control Panel that is not listening aborts the run while the leg is
        still cold -- not once the drives are live and someone is on the belt.
        No-op when not opted in."""
        if self._state is BeltState.DISABLED:
            why = ("TREADMILL.enabled is False" if not self._cfg.enabled
                   else "RUNTIME.arm_motion is False")
            print(f"[belt] disabled ({why}); the belts will not move")
            return

        if abs(self._cfg.speed) > self._cfg.max_speed:
            raise ValueError(f"TREADMILL.speed {self._cfg.speed} exceeds "
                             f"max_speed {self._cfg.max_speed}")
        # The leg must be WILLING to walk before the belts move, or the first
        # thing that happens is a straight, stiff knee being dragged backwards.
        walk = RUNTIME.walk_after_standing_s
        if walk is None:
            raise ValueError("TREADMILL.enabled with walk_after_standing_s=None: "
                             "the belts would move under a leg that can never walk")
        if walk >= self._settle + self._cfg.countdown_s:
            raise ValueError(
                f"walk_after_standing_s {walk} s must fall before the belts move "
                f"at STANDING+{self._settle + self._cfg.countdown_s} s "
                f"(STANDING.duration + TREADMILL.countdown_s)")

        self._tm = BertecTreadmill()
        self._tm.connect()
        print(f"[belt] connected to {self._tm.host}:{self._tm.port}; "
              f"{self._cfg.speed:.2f} m/s at {self._cfg.acc:.2f} m/s^2 once the "
              f"leg has stood for {self._settle:.1f} s "
              f"+ {self._cfg.countdown_s:.0f} s countdown")

    def shutdown(self) -> None:
        """Stop the belts and wait for zero. BLOCKING, and safe in any state --
        this is what the parent's terminate() calls, so it runs on a normal exit,
        a fault, Ctrl-C and SIGTERM alike. ExIntGuard has already set SIGINT to
        SIG_IGN by then, so a second Ctrl-C cannot interrupt the stop."""
        if self._tm is None:
            return
        try:
            if self._state in (BeltState.COUNTDOWN, BeltState.RUNNING,
                               BeltState.STOPPING):
                print(f"\n[belt] stopping at {self._cfg.stop_acc:.2f} m/s^2 ...")
                self._tm.stop(acc=self._cfg.stop_acc)
                deadline = time.monotonic() + self._cfg.stop_timeout_s
                while time.monotonic() < deadline:
                    time.sleep(PRINT_PERIOD_S)
                    if self._zeroed(self._tm.read()):
                        break
                else:
                    print("[belt] belts did not read zero -- USE THE E-STOP")
                print("[belt] stopped.")
            self._state = BeltState.DONE
        except OSError as e:
            print(f"[belt] LINK LOST during stop ({e}); the belts may still be "
                  f"running -- USE THE E-STOP")
        finally:
            self._tm.close()
            if self._run_dir is not None:
                self._log.save(self._run_dir)

    # ------------------------------------------------------------------ step

    def step(self, t: float, cmd, st=None) -> None:
        """One pass, from the parent's monitor loop. Never blocks.

        Args:
            t: seconds from the run's synced origin (control's start_time), so
                the belt log joins the others on `t`.
            cmd: b3.latest() -- HighLevelCommand or None.
            st: b1.latest() -- StampedState or None; only read if require_contact.
        """
        if self._state in (BeltState.DISABLED, BeltState.DONE):
            return

        dt = 0.0 if self._t_prev is None else max(t - self._t_prev, 0.0)
        self._t_prev = t

        try:
            self._log.append(t, self._tm.read())
            ok = self._ready(cmd, st)
            self._actuate_for = (self._actuate_for + dt) if ok else 0.0
            self._advance(t, ok)
        except OSError as e:
            print(f"\n[belt] LINK LOST ({e}); cannot command the belts -- "
                  f"USE THE E-STOP IF THEY ARE MOVING")
            self._state = BeltState.DONE

    def _advance(self, t: float, ok: bool) -> None:
        if self._state is BeltState.WAITING:
            if self._actuate_for >= self._settle:
                self._state = BeltState.COUNTDOWN
                self._t_mark = t
                self._next_tick = 0.0
                print(f"\n[belt] leg has finished standing; belts in "
                      f"{self._cfg.countdown_s:.0f} s (Ctrl-C aborts)")

        elif self._state is BeltState.COUNTDOWN:
            if not ok:
                self._state = BeltState.WAITING
                print(f"\n[belt] countdown cancelled: {self._why}")
            elif t - self._t_mark >= self._cfg.countdown_s:
                self._tm.set_speeds(self._cfg.speed, self._cfg.speed,
                                    acc_r=self._cfg.acc, acc_l=self._cfg.acc)
                self._state = BeltState.RUNNING
                print(f"\n[belt] running at {self._cfg.speed:.2f} m/s -- "
                      f"step in as the belt picks up")
            else:
                left = math.ceil(self._cfg.countdown_s - (t - self._t_mark))
                if left != self._next_tick:
                    self._next_tick = left
                    print(f"[belt] {left:2d} s", end="\r", flush=True)

        elif self._state is BeltState.RUNNING:
            if not ok:
                print(f"\n[belt] FAULT: {self._why}; stopping at "
                      f"{self._cfg.stop_acc:.2f} m/s^2")
                self._tm.stop(acc=self._cfg.stop_acc)
                self._state = BeltState.STOPPING
                self._t_mark = t

        elif self._state is BeltState.STOPPING:
            if (self._zeroed(self._tm.read())
                    or t - self._t_mark > self._cfg.stop_timeout_s):
                self._state = BeltState.DONE
                print("[belt] stopped.")

    def _ready(self, cmd, st) -> bool:
        """May the belts run under the leg as it stands right now? Sets _why."""
        if cmd is None:
            self._why = "control has not published a command yet"
            return False

        # b3.seq is the liveness signal: a control process stuck in a syscall is
        # still `is_alive()`, but its seq stops. At 500 Hz it should advance by
        # hundreds between our passes, so no advance at all means hung.
        if cmd.seq == self._last_seq:
            self._stale += 1
            if self._stale >= self._cfg.max_stale_passes:
                self._why = "control stopped publishing (b3.seq stalled)"
                return False
        else:
            self._stale = 0
            self._last_seq = cmd.seq

        if not cmd.mode.actuate:
            self._why = f"mode is {cmd.mode.name}"
            return False

        if self._cfg.require_contact and not (st is not None and st.state.contact):
            self._why = "prosthetic foot not loaded"
            return False
        return True

    @staticmethod
    def _zeroed(f) -> bool:
        return f.fresh and f.right_speed == 0.0 and f.left_speed == 0.0



def _show(f) -> None:
    print(f"right {f.right_speed:+.3f} m/s | left {f.left_speed:+.3f} m/s", end="\r")


def _countdown(seconds: float) -> None:
    end = time.monotonic() + seconds
    while (left := end - time.monotonic()) > 0:
        print(f"Starting in {math.ceil(left):2d} s  (Ctrl+C to abort)", end="\r")
        time.sleep(min(0.1, left))
    print()


def main():
    parser = argparse.ArgumentParser(description="Run the treadmill at a set speed until Ctrl+C.")
    parser.add_argument("--speed", type=float, default=0.8, help="belt speed [m/s], both belts")
    parser.add_argument("--acc", type=float, default=0.5, help="accel/decel [m/s^2]")
    parser.add_argument("--host", default=None, help="treadmill PC IP (default params.TREADMILL.host)")
    args = parser.parse_args()
    if abs(args.speed) > TREADMILL.max_speed:
        parser.error(f"|speed| must be <= {TREADMILL.max_speed} m/s "
                     f"(TREADMILL.max_speed)")
    if not 0.0 < args.acc <= 1.5:
        parser.error("acc must be in (0, 1.5] m/s^2")

    tm = BertecTreadmill(host=args.host)
    if not tm.host:
        parser.error("no treadmill host: pass --host or set params.TREADMILL.host")

    try:
        tm.connect()
    except OSError as e:
        parser.exit(1, f"Could not connect to {tm.host}:{tm.port} ({e}). Is the Control Panel "
                       "running with Remote TCP/IP Control enabled?\n")

    try:
        print(f"Connected to {tm.host}:{tm.port}. Will ramp to {args.speed:.2f} m/s "
              f"at {args.acc:.2f} m/s^2.")
        try:
            _countdown(START_IN_S)
        except KeyboardInterrupt:
            print("\nAborted before start; nothing sent.")
            return

        print("Starting. Ctrl+C to stop.")
        try:
            tm.set_speeds(args.speed, args.speed, acc_r=args.acc, acc_l=args.acc)
            while True:
                time.sleep(PRINT_PERIOD_S)
                _show(tm.read())
        except KeyboardInterrupt:
            pass
        finally:
            print(f"\nStopping at {args.acc:.2f} m/s^2 ...")
            tm.stop(acc=args.acc)

        # Wait for the belts to actually reach zero, so the operator sees it.
        # A second Ctrl+C here is safe: the stop command has already been sent.
        deadline = time.monotonic() + TREADMILL.stop_timeout_s
        f = tm.read()
        while time.monotonic() < deadline:
            time.sleep(PRINT_PERIOD_S)
            f = tm.read()
            _show(f)
            if f.fresh and f.right_speed == 0.0 and f.left_speed == 0.0:
                break
        print("\nStopped." if f.right_speed == 0.0 and f.left_speed == 0.0
              else "\nBelts did not read zero in time -- check the Control Panel.")
    finally:
        tm.close()


if __name__ == "__main__":
    main()
