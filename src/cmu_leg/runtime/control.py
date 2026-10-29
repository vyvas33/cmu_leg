"""
Control process body (P2). Runs as a child process.

Owns the two USB IMUs, the ESKF, AND the high-level controller -- the estimator
and high-level stages, MERGED into one process (co-located, hgeyer-aligned). Each
tick: reads the IMUs, pulls the latest full ProsthesisState off the shared B1
StateBuffer, runs the ESKF to overlay hip_*, decides the mode (standing/walking),
runs the active behavioral policy, and publishes the desired joint torques + mode
flag straight to B3 (CommandBuffer, down to the low-level process).

WHY MERGED: running the ESKF in its own process put an extra IPC hop (and a
500 Hz vs 1 kHz rate mismatch) between the sensor and the controller, so the
standing impedance's velocity-damping term acted on ~3-4 ms-stale velocity and
drove an 11 Hz limit cycle. Co-locating estimation + control halves that transport
delay (2 hops -> 1). See the multiprocess design notes. The fast, delay-sensitive
velocity still lives at the sensor (low-level, 1 kHz); if standing still oscillates,
the damping term moves DOWN into the low-level -- it can't be fixed up here.

Symmetric to runtime/low_level.py: a run() Process target, hardware initialized
AFTER spawn (USB handles can't be pickled), a completion_stage-guarded teardown,
and an integer-cycle soft-realtime loop. There is no start Barrier anymore: this
is the only child that stamps start_time (low-level heartbeats the bus until it
becomes nonzero), so it stamps directly after IMU init.
"""

import gc
import time
from dataclasses import dataclass
from math import floor, nan
from time import perf_counter, sleep
from multiprocessing import current_process

from cmu_leg.hardware.imu_io import IMUInterface
from cmu_leg.estimation.eskf.eskf import ESKF
from cmu_leg.control.high_level_control import ProsthesisControl
from cmu_leg.control.high_level.nm_npc import NMNPC
from cmu_leg.control.high_level.nm_minjerk import NMMinJerk
from cmu_leg.control.utils import BackwardDifference
from cmu_leg.structures.state import ProsthesisState
from cmu_leg.structures.commands import Mode
from cmu_leg.runtime.operator import TimedOperator
from cmu_leg.runtime.buffer import StateBuffer, CommandBuffer
from cmu_leg.runtime.guard import ExIntGuard
from cmu_leg.utility.recorder import Recorder
from cmu_leg.utility.rtutils import pin_to_core
import cmu_leg.config.params as params

# Standing is the resting request (see runtime.operator); the drives gate it
# in run()). TODO: replace the time schedule with the real mode FSM / supervisor.


# Per-thread CPU clock. `perf_counter` measures WALL time, which on a saturated
# core conflates "this stage did work" with "this stage was descheduled or sat
# waiting on the GIL". CLOCK_THREAD_CPUTIME_ID counts only cycles this thread
# actually retired, so (wall - cpu) over the same span IS the stolen time. That
# is the whole point of the instrumentation: the two candidate explanations for
# the 500 Hz overrun -- an expensive ESKF vs. the two IMU parser threads
# contending for this core -- are indistinguishable in wall time alone.
try:
    time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)

    def _thread_cpu() -> float:
        return time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)
except (AttributeError, OSError):        # not Linux/macOS -- degrade, don't fail
    def _thread_cpu() -> float:
        return 0.0


@dataclass
class LoopStats:
    """Per-tick timing trace for the online idle-margin / loop-period plots.
    Recorder logs dataclasses, so the scalars need this wrapper.

    `idle` / `dt` are the original two: scheduling margin and measured period.

    The `t_*` fields are a per-stage WALL breakdown of the tick body, and
    `cpu_body` is the control thread's own CPU time over that same body. They
    answer different questions and are only useful together:

        sum(t_*)            ~= t_body           (by construction)
        t_body - cpu_body    = time STOLEN from this thread

    A large `t_eskf` with `t_body ~= cpu_body` means the ESKF is genuinely
    expensive and decimating or relocating it is the fix. A large `t_eskf` with
    `t_body >> cpu_body` means the stage is not slow, it is being interrupted --
    and the fix is the IMU stream rate / parser threads, not the filter.

    `t_log` is the previous tick's logging cost, carried one tick forward: the
    Recorder appends necessarily happen after this record is built, so a tick
    cannot measure its own. Statistics over a run are unaffected by the shift.
    """
    idle: float
    dt: float
    t_imu: float
    t_buf: float
    t_eskf: float
    t_ctrl: float
    t_pub: float
    t_tail: float
    t_log: float
    t_body: float
    cpu_body: float


@dataclass
class GaitTrace:
    """Per-tick trace of the gait FSM, logged beside its own input.

    `contact` is the raw thresholded measurement off the bus and `in_stance` is
    what the FSM made of it after ContactParams.min_stance / min_swing. Logging
    them together is the point: the refractory hold is the only thing between
    them, so a run where contact chatters but stance does not is the hold doing
    its job, and one where both chatter means the threshold needs work. Before
    this, only `contact` was recorded and the FSM output existed nowhere but the
    terminal prints.

    `mode` is here because in_stance is only meaningful while WALKING -- in any
    other mode the FSM is not ticked and simply holds its last value.

    `stance_duration` is the last COMPLETED stance (what sizes the next swing),
    NaN until one finishes, so it reads as a gap rather than a fake zero.
    """
    mode: int
    contact: bool
    in_stance: bool
    stance_duration: float


def run(dt, tmax, buffer, cmd_buf, stop_event, start_time,
        run_dir, is_verbose=False, core=None, rt_priority=None):
    """Control child process entry point (the Process target).

    Args:
        dt (float): Nominal loop period (s) -- steps the sleep-to-clock ONLY.
            The controller runs at THIS rate (~500 Hz); the ESKF integration and
            every BackwardDifference use the MEASURED dt instead.
        tmax (float): Max elapsed time (s); fixes the integer cycle count.
        buffer (StateBuffer): B1, low-level -> control. Sole reader; carries the
            full proprioceptive ProsthesisState (hip_* zero until we overlay it).
        cmd_buf (CommandBuffer): B3, control -> low-level. Sole writer; desired
            joint torques + the standing mode flag.
        stop_event (multiprocessing.Event): Graceful-shutdown request.
        start_time (multiprocessing.Value): Shared 'd' synced perf_counter origin.
            Control is the sole stamper (no barrier); low-level heartbeats the bus
            until this becomes nonzero, then both run from the common origin.
        run_dir (str | Path): Log directory, created ONCE by the parent so both
            children's CSVs land together and can be joined on `t`.
        is_verbose (bool): Verbose prints.

    No init_evnt here: nothing downstream is gated on control coming up.
    """
    pname = current_process().name
    # Real-time scheduling, applied by the child itself: sched_setscheduler
    # acts on the calling process, and doing it here removes the race with
    # the first ticks that setting affinity from the parent after spawn left.
    if core is not None:
        pin_to_core(core, rt_priority if rt_priority is not None else 80)
    rt = params.RUNTIME           # runtime knobs

    completion_stage = 0
    # Loop-entry baselines for the teardown CPU accounting (set just before the
    # loop). Process CPU counts EVERY thread; thread CPU counts only this one.
    # The difference over the run is what the two IMU parser threads burned --
    # the one number that says whether this core is short on time because the
    # control loop is expensive or because something else is eating it.
    cpu_base = {"wall": 0.0, "proc": 0.0, "thread": 0.0}

    def terminate_gracefully():
        gc.enable()  # undo the RT gc freeze unconditionally
        if completion_stage >= 1:
            if cpu_base["wall"]:
                wall = perf_counter() - cpu_base["wall"]
                proc = time.process_time() - cpu_base["proc"]
                thrd = _thread_cpu() - cpu_base["thread"]
                other = proc - thrd
                print(f"[{pname}] CPU over {wall:.1f} s of loop: "
                      f"control thread {thrd:.2f} s ({thrd/wall*100:.1f}% of a core), "
                      f"other threads {other:.2f} s ({other/wall*100:.1f}%), "
                      f"process total {proc/wall*100:.1f}%")
            for lg in (state_log, torque_log, timing_log, gait_log):
                lg.save(run_dir)
            imu.close()
            # OperatorSource teardown; a no-op for the timed source.
            operator.close()
        if is_verbose:
            print(f"{pname}: terminated gracefully.")

    # A SIGTERM/SIGINT/exception sets stop_event and still runs teardown.
    with ExIntGuard(pname, stop_event, terminate_gracefully):
        # --- stage 1: init hardware AFTER spawn ------------------------------
        imu = IMUInterface()   # ports/baud default to params.IMU
        imu.connect()
        eskf = ESKF()
        hip_accel = BackwardDifference()
        # Neuromuscular stance + neural-predictive knee swing. Min-jerk still
        # owns the ankle, and the knee too until the NN arms.
        pc = ProsthesisControl(walking=NMMinJerk())
        # Operator intent. This process is spawned, so its stdin is devnull and
        # a keypress source could never come up here -- the schedule is the only
        # way to drive a run, and it is also the only deterministic one.
        operator = TimedOperator(walk_after_standing=rt.walk_after_standing_s,
                                 swing_at=rt.bench_swing_at,
                                 swing_duration=rt.bench_swing_duration)
        print(f"[{pname}] operator: timed  "
              f"walk@STANDING+{rt.walk_after_standing_s}s  "
              f"swings@{list(rt.bench_swing_at)}s "
              f"({rt.bench_swing_duration}s each)")
        state = ProsthesisState()   # this process's OWN local bus

        # tmax counts from the first ACTUATING tick, not from t0, so warmup
        # does not eat the run. Both processes detect that moment independently
        # -- control from its own FSM, low-level from the mode on the command it
        # receives -- which differ by one IPC hop (sub-ms) and need no extra
        # shared state. `max_arm_wait_s` bounds a run that never arms.
        num_cycles = floor((rt.max_arm_wait_s + tmax) / dt) + 1
        t_armed = None            # t of the first actuating tick
        state_log = Recorder("control_prosthesis_state")  # full ProsthesisState (incl. hip_*)
        torque_log = Recorder("control_torque")   # TorqueDesired each tick
        timing_log = Recorder("control_idle")     # LoopStats (idle margin + measured dt)
        gait_log = Recorder("control_gait")       # GaitTrace (contact -> stance, + mode)

        completion_stage = 1
        # restore the terminal on the way out if we grabbed it

        # --- RT gc hygiene (mirrors main.py) ---------------------------------
        gc.collect()
        gc.freeze()
        gc.disable()

        # --- stamp the shared start origin (no barrier; sole stamper) --------
        # Low-level is spinning on `while start_time == 0`, heartbeating the
        # EtherCAT bus to hold the drives in OP. Stamping here -- after the slow
        # IMU init -- releases it, and both loops run from this common origin.
        with start_time.get_lock():
            start_time.value = perf_counter()
        t0 = start_time.value

        # --- soft-realtime loop ---------------------------------------------
        clock = t0
        t_prev = t0 - dt
        cycle = 0
        t_log_prev = 0.0      # previous tick's logging cost; see LoopStats
        cpu_base["wall"] = perf_counter()
        cpu_base["proc"] = time.process_time()
        cpu_base["thread"] = _thread_cpu()
        while cycle < num_cycles:
            now = perf_counter()
            cpu0 = _thread_cpu()
            t = now - t0
            dt_meas = now - t_prev
            t_prev = now

            imu_frame = imu.read()
            m1 = perf_counter()

            # adopt the full proprioceptive state off the buffer as this tick's
            # base, BEFORE the ESKF (its loop closures read joint velocities off
            # the bus). hip_* arrive as zero and are overlaid by eskf.update below;
            # every other physical field (motor, spring, torques, GRF) carries
            # through to the controller + the log. .seq is available for staleness.
            ms = buffer.latest()
            if ms is not None:
                state = ms.state
            m2 = perf_counter()

            # --- inline ESKF step (was ProsthesisStateEstimator._apply_eskf) ---
            # pass this loop's own clock t (shares the IMU timebase); dt_meas
            # drives the filter's integration + the hip_accel diff.
            eskf.update(imu_frame, state, t, dt_meas)
            if eskf.initialized:
                state.hip_accel = hip_accel.compute(state.hip_velocity, dt_meas)
            m3 = perf_counter()

            # --- mode FSM + high-level policy (co-located, no extra hop) -------
            # control OWNS the mode; the low level derives `actuate` and the
            # knee extension-stop suppression from it.
            # pc.mode is last tick's (the FSM advances inside compute below),
            # which is what the operator's standing-relative schedule wants.
            op = operator.read(t, pc.mode)
            desired = pc.compute(state, t, op, dt_meas)   # TorqueDesired
            state.standing = pc.mode is Mode.STANDING     # for logging/consumers
            m4 = perf_counter()

            cmd_buf.publish(desired, pc.mode)             # -> B3, down to low-level
            m5 = perf_counter()

            # arm the run window on the first actuating tick, then end tmax
            # later. Checked after logging (below) so the last tick is recorded,
            # matching low-level.
            if t_armed is None and pc.mode.actuate:
                t_armed = t
                print(f"[{pname}] armed at t={t:.2f}s ({pc.mode.name}); "
                      f"running {tmax}s")
            armed_expired = t_armed is not None and t - t_armed >= tmax

            if stop_event.is_set():
                return

            # `t_tail` covers the arming checks and stop_event.is_set(). That
            # last one is NOT a flag read: multiprocessing.Event.is_set()
            # acquires its Condition, try-acquires the semaphore and releases
            # it -- three semaphore ops per tick, on an RT thread. It sat
            # outside every stage timer in the first version of this
            # instrumentation, which is why the stage totals did not reconcile
            # with the measured margin.
            m6 = perf_counter()
            cpu1 = _thread_cpu()

            clock += dt
            idle = clock - perf_counter()
            log0 = perf_counter()
            state_log.append(t, state)
            torque_log.append(t, desired)
            timing_log.append(t, LoopStats(
                idle, dt_meas,
                t_imu=m1 - now, t_buf=m2 - m1, t_eskf=m3 - m2,
                t_ctrl=m4 - m3, t_pub=m5 - m4, t_tail=m6 - m5,
                t_log=t_log_prev,
                t_body=m6 - now, cpu_body=cpu1 - cpu0,
            ))
            gait_log.append(t, GaitTrace(
                mode=int(pc.mode),
                contact=state.contact,
                in_stance=pc.in_stance,
                stance_duration=(nan if pc.last_stance_duration is None
                                 else pc.last_stance_duration),
            ))
            t_log_prev = perf_counter() - log0
            sleep(idle) if idle >= 0.0 else sleep(0.0)

            if armed_expired:
                break

            cycle += 1
    # (ExIntGuard runs terminate_gracefully on exit -- no finally needed)
