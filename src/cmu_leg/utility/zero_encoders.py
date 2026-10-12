"""
Zero Encoders

Drives both motors to raw absolute count 0 in CSP (the mechanical zero) and
records the median raw counts of every encoder there -> new params.ENCODERS
offsets. Run with the leg hanging FREE: any spring load at the zero pose bakes
a torque bias into the joint offsets.

Usage:
    sudo .venv/bin/python -m cmu_leg.utility.zero_encoders
"""

import time
import gc
from statistics import median

# hardware boundary
from cmu_leg.hardware.ethercat_io import EtherCATInterface
from cmu_leg.hardware.elmos.state_machine import drive_state, DriveState

# control
from cmu_leg.control.high_level.minjerk import min_jerk
from cmu_leg.control.utils import clamp
from cmu_leg.estimation.encoders import _wrap_counts

# data types
from cmu_leg.structures.commands import PositionCommand
from cmu_leg.config.params import COMMON, ENCODERS, ENCODER_CORRECTION

# logging output dir (post-loop dump)
from cmu_leg.utility.utils import make_run_dir
from cmu_leg.utility.recorder import Recorder

PERIOD = COMMON.sample_time
WARM_UP_TIME = 5.0 #secs
ENABLE_TIME = 5.0 #secs
RAMP_TIME = 5.0 #secs
HOLD_TIME = 10.0 #secs
HOLD_SETTLE = 1.0 #secs, dropped from the start of the hold window
SHUTDOWN_TIME = 1.0 #secs

FOLLOW_ERR_LIMIT = 10_000_000   # counts (~3.4 deg); leg blocked or spring loaded
KNEE_START_LIMIT = 300_000_000  # counts (~100 deg); implausible latch -> bad frame
ANKLE_START_LIMIT = 100_000_000 # counts (~33 deg)
STEP_LIMIT = 150_000            # counts/tick (~0.9 rad/s); min-jerk peaks ~100k

def main():

    # hardware bringup
    ec = EtherCATInterface("eth0")
    ec.connect()

    frame_log, cmd_log = Recorder("frame"), Recorder("pos_cmd")

    t0 = time.perf_counter()
    clk = t0
    frame = ec.read()

    knee_start = ankle_start = t0_ramp = None   # latched on the enabled rising edge
    prev_knee_cmd = prev_ankle_cmd = None       # last emitted targets, for the rate limit
    abort = None                                # reason string; once set, drop enable and only track

    def emit(knee_target: int, ankle_target: int) -> PositionCommand:
        """Rate-limit the outgoing command so no single tick can jump."""
        nonlocal prev_knee_cmd, prev_ankle_cmd
        if prev_knee_cmd is not None:
            knee_target = prev_knee_cmd + clamp(knee_target - prev_knee_cmd, -STEP_LIMIT, STEP_LIMIT)
            ankle_target = prev_ankle_cmd + clamp(ankle_target - prev_ankle_cmd, -STEP_LIMIT, STEP_LIMIT)
        prev_knee_cmd, prev_ankle_cmd = knee_target, ankle_target
        return PositionCommand(knee_target, ankle_target)

    def get_pos_cmd(t: float) -> PositionCommand:
        """Tracks the current pose until the enabled edge (and after an abort),
        min-jerks to raw 0 afterwards."""
        nonlocal knee_start, ankle_start, t0_ramp, abort

        # pre-latch faults are left to the CiA-402 FSM (it sends CW_FAULT_RESET
        # once enable is true); only a fault mid-motion, while still commanding,
        # aborts -- not during the post-run shutdown ticks
        if abort is None and ec.enable and t0_ramp is not None and (
                drive_state(frame.knee_status_word) == DriveState.FAULT
                or drive_state(frame.ankle_status_word) == DriveState.FAULT):
            abort = "drive fault"

        if abort is not None:
            return emit(frame.knee_motor_counts, frame.ankle_motor_counts)

        if t0_ramp is None:
            if ec.enable and ec.control_enabled():
                if (abs(frame.knee_motor_counts) > KNEE_START_LIMIT
                        or abs(frame.ankle_motor_counts) > ANKLE_START_LIMIT):
                    abort = f"implausible start pose (bad frame?): knee={frame.knee_motor_counts} ankle={frame.ankle_motor_counts}"
                    return emit(frame.knee_motor_counts, frame.ankle_motor_counts)
                knee_start, ankle_start = frame.knee_motor_counts, frame.ankle_motor_counts
                t0_ramp = t
                print(f"[{t:6.3f}s] drives enabled, ramping to zero from knee={knee_start} ankle={ankle_start}")
            return emit(frame.knee_motor_counts, frame.ankle_motor_counts)

        knee_target = round(min_jerk(knee_start, 0.0, t0_ramp, RAMP_TIME, t))
        ankle_target = round(min_jerk(ankle_start, 0.0, t0_ramp, RAMP_TIME, t))
        pos_cmd = emit(knee_target, ankle_target)

        # following error is emitted command vs actual (what the drive really sees)
        if (abs(pos_cmd.knee_position_counts - frame.knee_motor_counts) > FOLLOW_ERR_LIMIT
                or abs(pos_cmd.ankle_position_counts - frame.ankle_motor_counts) > FOLLOW_ERR_LIMIT):
            abort = "following error (leg blocked or spring loaded?)"

        return pos_cmd

    def run(duration: float, until=None) -> None:
        """Tick at 1 kHz for `duration`, or until `until()` is true."""
        nonlocal clk, frame
        phase_start = time.perf_counter() - t0
        while (t := time.perf_counter() - t0) - phase_start < duration:
            clk += PERIOD
            pos_cmd = get_pos_cmd(t)
            if abort is not None:
                ec.enable = False
            frame = ec.write_pos(pos_cmd)
            idle = clk - time.perf_counter()

            frame_log.append(t, frame); cmd_log.append(t, pos_cmd)

            if until is not None and until():
                return
            if idle > 0:
                time.sleep(idle)

    gc.collect()
    gc.freeze()
    gc.disable()
    try:
        # enable is False
        print("Warming up......")
        run(WARM_UP_TIME)

        # Set enable to true; ramp starts on the enabled edge
        ec.enable = True
        run(ENABLE_TIME, until=lambda: t0_ramp is not None or abort is not None)

        if t0_ramp is None:
            print(f"WARNING: failed to reach OPERATION_ENABLED ({abort or 'timeout'}); no zeros recorded")
        else:
            run(RAMP_TIME + HOLD_TIME)

        if abort is not None:
            print(f"ABORTED: {abort}")

    except KeyboardInterrupt:
        print("terminated early")

    finally:
        gc.enable()
        ec.enable = False
        try:
            run(SHUTDOWN_TIME)
        finally:
            ec.close()
            run_dir = make_run_dir("zero_encoders")
            for lg in (frame_log, cmd_log):
                lg.save(run_dir)

    report(frame_log, t0_ramp, abort)


def report(frame_log: Recorder, t0_ramp, abort) -> None:
    """Median raw counts over the settled hold window vs current params.ENCODERS."""
    if t0_ramp is None or abort is not None:
        print("no valid hold window; skipping offset report")
        return

    ts = frame_log.column("t")
    hold_start = t0_ramp + RAMP_TIME + HOLD_SETTLE
    idx = [i for i, t in enumerate(ts) if t >= hold_start]
    if len(idx) < 1000:
        print(f"only {len(idx)} hold samples (>= {hold_start:.1f}s); not enough for medians")
        return

    def med(col, wrap=False):
        vals = [frame_log.column(col)[i] for i in idx]
        if wrap:
            vals = [_wrap_counts(v, ENCODER_CORRECTION.ankle_total_counts) for v in vals]
        return round(median(vals))

    new = {
        "knee_abs_counts_offset": med("knee_joint_counts"),
        "ankle_abs_counts_offset": med("ankle_joint_counts", wrap=True),
        "knee_motor_abs_counts_offset": med("knee_motor_counts"),
        "ankle_motor_abs_counts_offset": med("ankle_motor_counts"),
        "hall_0_offset": med("hall_0"),
        "hall_1_offset": med("hall_1"),
    }

    print(f"\nzero offsets from {len(idx)} hold samples (current params.py value -> measured):")
    for name, val in new.items():
        print(f"  {name:32s} {getattr(ENCODERS, name):>12} -> {val:>12}")

    print("\nparams.py paste block:")
    for name, val in new.items():
        print(f"    {name}={val:_},")


if __name__ == "__main__":
    main()
