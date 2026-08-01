"""
Split-aware per-stage timing: single-process (main) vs the 3-process split.

Companion to prof_control.py. Where that one times main's serial per-tick path,
this one additionally:
  - breaks the estimator into its two halves (proprioceptive vs ESKF), since the
    ESKF is the split point and the dominant cost,
  - times the buffer publish/latest calls (the IPC copy overhead), and
  - sums each RUNTIME PROCESS's per-tick budget, so you can see the effective
    cost of the split -- which is the SLOWEST process, since the three run in
    parallel, not the sum.

The point it makes: the split does not make the ESKF faster; it isolates it, so
the 1 kHz torque-critical low-level loop no longer inherits the ESKF's median
latency or its millisecond jitter tail.

No hardware required, nothing is energized -- pure CPU timing on synthetic input.

  uv run python -m cmu_leg.tests.prof_multiprocess

Absolute numbers are host-relative (a Mac dev box is ~3-5x faster than the
Jetson); the per-stage breakdown and the per-process budgets are what transfer.
"""
import time
import math
import statistics

from cmu_leg.estimation.proprioceptive_estimator import ProprioceptiveEstimator
from cmu_leg.estimation.eskf.eskf import ESKF
from cmu_leg.control.high_level_control import ProsthesisControl
from cmu_leg.control.low_level_control import LowLevelControl
from cmu_leg.types.sensors import RawEtherCATFrame, IMUFrame
from cmu_leg.runtime.buffer import MedullaStateBuffer, StateBuffer, CommandBuffer

N_WARM = 2000
N = 30000


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def _report(name, xs):
    us = [x * 1e6 for x in xs]
    print(f"  {name:22s} med={statistics.median(us):7.2f}  mean={statistics.fmean(us):7.2f}"
          f"  p99={_pct(us, 99):7.2f}  max={max(us):8.2f}  (us)")


def _make_frame(k):
    # slowly-varying counts so nothing short-circuits on a lucky zero; this path's
    # timing is essentially input-independent (filters/clips do fixed work).
    w = math.sin(k * 0.01)
    return RawEtherCATFrame(
        knee_motor_counts=int(100000 + 5000 * w),
        knee_joint_counts=int(20000 + 2000 * w),
        ankle_motor_counts=int(80000 + 4000 * w),
        ankle_joint_counts=int(15000 + 1500 * w),
        knee_motor_torque_raw=int(500 * w),
        ankle_motor_torque_raw=int(400 * w),
    )


def main():
    proprio = ProprioceptiveEstimator()
    eskf = ESKF()
    pc = ProsthesisControl()
    llc = LowLevelControl()
    b1, b2, b3 = MedullaStateBuffer(), StateBuffer(), CommandBuffer()
    imu = IMUFrame()
    state = proprio.state   # proprio writes here; eskf reads/writes the same bus

    keys = ("proprio", "eskf", "high", "low", "drive",
            "b1pub", "b1lat", "b2pub", "b2lat", "b3pub", "b3lat")
    T = {k: [] for k in keys}

    for k in range(N_WARM + N):
        frame = _make_frame(k)
        t = k * 0.001
        dt = 0.001
        rec = k >= N_WARM

        c = [time.perf_counter()]
        proprio.update(frame, dt);                                       c.append(time.perf_counter())
        eskf.update(imu, state, t, dt);                                  c.append(time.perf_counter())
        state.standing = True
        desired = pc.compute(state, t);                                  c.append(time.perf_counter())
        llo = llc.compute(state, desired_torque=desired,
                          control_enabled=True, dt=dt);                  c.append(time.perf_counter())
        llc.to_drive_command(llo);                                       c.append(time.perf_counter())
        b1.publish(state.knee_angle, state.ankle_angle, state.knee_velocity,
                   state.ankle_velocity, state.toe_grf, state.heel_grf, t); c.append(time.perf_counter())
        b1.latest();                                                     c.append(time.perf_counter())
        b2.publish(state);                                               c.append(time.perf_counter())
        b2.latest();                                                     c.append(time.perf_counter())
        b3.publish(desired, True);                                       c.append(time.perf_counter())
        b3.latest();                                                     c.append(time.perf_counter())

        if rec:
            for i, key in enumerate(keys):
                T[key].append(c[i + 1] - c[i])

    print(f"\n== per-stage compute, N={N} ticks ==")
    for k in ("proprio", "eskf", "high", "low", "drive"):
        _report(k, T[k])

    print("\n== buffer ops (IPC copy overhead) ==")
    for k in ("b1pub", "b1lat", "b2pub", "b2lat", "b3pub", "b3lat"):
        _report(k, T[k])

    med = lambda k: statistics.median(T[k]) * 1e6
    single = sum(med(k) for k in ("proprio", "eskf", "high", "low", "drive"))
    # each runtime process's per-tick budget (compute + its buffer ops)
    low_p = med("proprio") + med("low") + med("drive") + med("b1pub") + med("b3lat")
    est_p = med("eskf") + med("b1lat") + med("b2pub")
    high_p = med("high") + med("b2lat") + med("b3pub")
    crit = max(low_p, est_p, high_p)

    print("\n== single-process (main): all stages serial in one tick ==")
    print(f"  TOTAL (sum of stages)   = {single:7.1f} us / tick")
    print("\n== 3-process split: parallel, so effective cost = slowest process ==")
    print(f"  low-level process       = {low_p:7.1f} us / tick")
    print(f"  estimator process       = {est_p:7.1f} us / tick")
    print(f"  high-level process      = {high_p:7.1f} us / tick")
    print(f"  critical path (max)     = {crit:7.1f} us / tick")
    print(f"\n  The RT low-level loop is decoupled from the ESKF's {med('eskf'):.0f} us"
          f" median + its ms-scale spikes (see eskf max above).\n")


if __name__ == "__main__":
    main()
