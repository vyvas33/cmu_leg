"""
Per-stage timing of the standing control path (off-hardware).

Purpose: measure how long one tick of the control compute takes -- estimator ->
high-level -> low-level -> to_drive_command -- with a synthetic frame, no
EtherCAT/IMU. Use it to confirm the compute fits the 1 kHz budget (1000 us minus
the EtherCAT exchange) and to prove optimizations (e.g. the biquad Lowpass) help.

No hardware required, nothing is energized -- pure CPU timing.

  uv run python -m cmu_leg.tests.prof_control

  OR, pinned to the isolated core + RT priority to mimic the real loop
  (needs `sysctl kernel.sched_rt_runtime_us=-1` if RT_GROUP_SCHED blocks it):

  sudo chrt -f 90 taskset -c 3 /home/cmuleg/cmu_leg/.venv/bin/python \
      -m cmu_leg.tests.prof_control

Absolute numbers are host-relative; the per-stage *breakdown* and before/after
*deltas* are what transfer across machines.
"""
import time
import math
import statistics

from cmu_leg.estimation.state_estimator import ProsthesisStateEstimator
from cmu_leg.control.high_level_control import ProsthesisControl
from cmu_leg.runtime.operator import OperatorInput
from cmu_leg.types.commands import Mode
from cmu_leg.config.params import MODE
from cmu_leg.control.low_level_control import LowLevelControl
from cmu_leg.types.sensors import RawEtherCATFrame, IMUFrame

N_WARM = 2000
N = 30000


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def _report(name, xs):
    us = [x * 1e6 for x in xs]
    print(f"{name:24s} med={statistics.median(us):7.2f}  mean={statistics.fmean(us):7.2f}"
          f"  p95={_pct(us, 95):7.2f}  p99={_pct(us, 99):7.2f}  max={max(us):8.2f}  (us)")


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
    se = ProsthesisStateEstimator()
    pc = ProsthesisControl()
    llc = LowLevelControl()
    imu = IMUFrame()

    t_est, t_hi, t_lo, t_drv, t_all = [], [], [], [], []

    for k in range(N_WARM + N):
        frame = _make_frame(k)
        rec = k >= N_WARM

        a0 = time.perf_counter()
        state = se.get_state(frame, imu, k * 0.001, 0.001)
        # Drive the mode FSM into STANDING: it needs the drives enabled, a
        # standing request, and enable_dwell elapsed. N_WARM must cover the
        # dwell or every recorded tick profiles INITIALIZING (which returns
        # early with zero torque) and the numbers mean nothing.
        state.control_enabled = True
        op = OperatorInput(standing_toggle=True, hold_toggle=False, seq=k)
        a1 = time.perf_counter()
        desired = pc.compute(state, k * 0.001, op, 0.001)
        a2 = time.perf_counter()
        llo = llc.compute(state, desired_torque=desired, control_enabled=True,
                          dt=0.001, actuate=pc.mode.actuate)
        a3 = time.perf_counter()
        llc.to_drive_command(llo)
        a4 = time.perf_counter()

        assert not rec or pc.mode is Mode.STANDING, (
            f"profiling {pc.mode.name}, not STANDING -- raise N_WARM above "
            f"MODE.enable_dwell ({MODE.enable_dwell}s = {int(MODE.enable_dwell*1000)} ticks)")
        if rec:
            t_est.append(a1 - a0)
            t_hi.append(a2 - a1)
            t_lo.append(a3 - a2)
            t_drv.append(a4 - a3)
            t_all.append(a4 - a0)

    print(f"\nstanding control path, N={N} ticks\n")
    _report("estimator.get_state", t_est)
    _report("high_level.compute", t_hi)
    _report("low_level.compute", t_lo)
    _report("to_drive_command", t_drv)
    print("-" * 92)
    _report("TOTAL compute", t_all)
    print(f"\nmedian total = {statistics.median(t_all) * 1e6:.1f} us "
          f"(budget 1000 us minus EtherCAT exchange)\n")


if __name__ == "__main__":
    main()
