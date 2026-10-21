"""Replay MinJerkSwing against the baseline's own commanded swing.

The trial ran the NPC controller with min-jerk swing live, and the `walk` scope
group logs that block's internals, so the setpoints it commanded are available
as a reference rather than just as scale context.

Compared, kept separate on purpose:

1. Trajectory, both joints, against walk p1 (knee) and p4 (ankle).

2. Torque, against the baseline's pid-only output. Its knee torque is
   `ff*1 + pid` (Torque Control/Sum3) and the feedforward is logged separately,
   so walk[:,0] - walk[:,4] isolates the pid. The gains and setpoint match and
   replay feeds the same measured angle, so that comparison exercises the
   discrete pid. The ankle feedforward gain is 0, so its torque compares
   directly. walk[:,4] is what the dropped knee feedforward was contributing.

   The measured angle here came from a leg that had the feedforward, so this
   bounds the torque difference, not the tracking that would result without it.

Inputs come from the baseline's own latches so the comparison isolates the
solver:
  base[:,40] `from stance to swing/time`     -- t_toeoff, latched
  base[:,41] `Triggered Subsystem/time`      -- t_heelstrike, latched
which reproduces swing_duration exactly instead of re-deriving it from our
contact detection. base[:,39] is the latched toe-off knee angle, a cross-check
that we sample the same instant.

Initial conditions go through the baseline's input filters (2nd-order
Butterworth at 1 kHz: knee velocity 25 Hz, ankle velocity 20 Hz, both
accelerations 10 Hz), which live inside `minjerk swing Control` and feed the
trajectory initial conditions.

Runs at the log's 1 kHz rather than the 500 Hz control rate. The trajectory is a
function of time and does not care, but the filters and the pid's discrete
states do.

The walking window opens mid-swing. That first truncated swing has no toe-off to
latch, so it is skipped; initiating from the window edge instead compares a
fresh trajectory against one already half done.

walk column map (0-based into scope_data.walk.data), from signalNames:
    0  Torque Control/Sum3 ............ knee torque (ff + pid)
    1  Torque Control/Sum2 ............ ankle torque
    2  evaluate polynomials/p1 ........ knee angle setpoint
    3  evaluate polynomials/p4 ........ ankle angle setpoint
    4  Calculate FF Torque/p1 ......... tau_knee_ff
    6  minjerk swing Control/Sum1 ..... stance duration
    7  evaluate polynomials/p2 ........ knee velocity setpoint
    8  evaluate polynomials/p3 ........ knee acceleration setpoint

Data + base column map: see tests/impedance_stance_replay.py.

Run:  uv run python -m cmu_leg.tests.minjerk_swing_replay
Writes minjerk_swing_replay.png next to this file.
"""
from dataclasses import replace
from math import radians
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cmu_leg.control.high_level.walking.minjerk_swing import MinJerkSwing
from cmu_leg.control.utils import Lowpass, clamp
from cmu_leg.structures.state import ProsthesisState
from cmu_leg.config.params import MJSWING, COMMON

HERE = Path(__file__).parent
MAT = HERE / "data" / "gaitData20251015_test7_NN_vicon.mat"
PNG = HERE / "minjerk_swing_replay.png"

# The `Ankle Angle` subsystem emits six signals, giving two plausible
# (angle, vel, accel) triples in the log:
#   15/16/17  Convert to radians2/Add   + Discrete Derivative /Diff, 2/Diff
#   18/19/20  Convert to radians2/Bias1 + Discrete Derivative1/Diff, 4/Diff
# 18/19/20 is the one the swing block consumes, established by substitution
# rather than by reading the model: it reproduces the commanded ankle trajectory
# to 0.0000 deg, where 15/16/17 gives 3.90 deg max and 0.60 RMS. The two differ
# by a small state-dependent amount, mean -0.17 deg and up to -2.9 deg, which
# looks like ankle spring deflection rather than a fixed offset. Which of them
# ProsthesisState.ankle_angle carries is unsettled, so this exercises the solver
# rather than the plumbing.
BASE = dict(knee_ang=7, knee_vel=8, knee_acc=9,
            ankle_ang=18, ankle_vel=19, ankle_acc=20, stance=29,
            toeoff_knee=39, t_toeoff=40, t_heelstrike=41, time=107)
WALK = dict(knee_tq=0, ankle_tq=1, knee_traj=2, ankle_traj=3, knee_ff=4,
            stance_dur=6, knee_vel_traj=7, knee_acc_traj=8)

WALK_START_S = 16.5
LOG_RATE = 1000.0


def load():
    sd = sio.loadmat(str(MAT), squeeze_me=True,
                     struct_as_record=False)["scope_data"]
    t = sd.base.data[:, BASE["time"]]
    w = t >= WALK_START_S
    sig = {k: sd.base.data[w, c] for k, c in BASE.items()}
    sig.update({k: sd.walk.data[w, c] for k, c in WALK.items()})
    return sig


def replay(sig):
    """Drive MinJerkSwing over the trial, resetting on each toe-off edge."""
    # Run the pid at the log's rate, which is also the nominal period it seeds
    # _t_prev from.
    # Pin the trajectory targets and pid gains to the values the baseline log was
    # generated with. MJSWING is tuned for the bench and has already moved away
    # from them -- knee_extend_angle off KNEE.angle_min_soft to stop the parked
    # leg chattering on the extension stop, knee_kd down for phase margin at
    # 500 Hz -- and this has to keep comparing like with like, or a deliberate
    # tuning change reads as a regression.
    cfg = replace(MJSWING,
                  knee_flex_angle=radians(60.0), knee_extend_angle=radians(5.0),
                  ankle_flex_angle=radians(5.0), ankle_extend_angle=0.0,
                  knee_kp=30.0, knee_ki=145.0, knee_kd=12.5,
                  ankle_kp=125.0, ankle_ki=125.0, ankle_kd=10.0,
                  deriv_filter_n=100.0)
    ctrl = MinJerkSwing(cfg=cfg,
                        common=replace(COMMON, control_sample_rate=LOG_RATE))
    ctrl._verified = True

    # Velocity and acceleration are filtered before they become trajectory
    # initial conditions; angle stays raw.
    vel_lp = Lowpass(cutoff_freq=25.0, sample_rate=LOG_RATE, order=2)
    acc_lp = Lowpass(cutoff_freq=10.0, sample_rate=LOG_RATE, order=2)
    a_vel_lp = Lowpass(cutoff_freq=20.0, sample_rate=LOG_RATE, order=2)
    a_acc_lp = Lowpass(cutoff_freq=10.0, sample_rate=LOG_RATE, order=2)
    vel_lp.reset(sig["knee_vel"][0])
    acc_lp.reset(sig["knee_acc"][0])
    a_vel_lp.reset(sig["ankle_vel"][0])
    a_acc_lp.reset(sig["ankle_acc"][0])

    n = len(sig["time"])
    traj = np.full(n, np.nan)
    torque = np.full(n, np.nan)
    a_traj = np.full(n, np.nan)
    a_torque = np.full(n, np.nan)
    swing = sig["stance"] < 0.5
    events = []                      # (t_toeoff, T_ours, T_base, ic_err)

    for i in range(n):
        t = sig["time"][i]
        vel = vel_lp.compute(sig["knee_vel"][i])
        acc = acc_lp.compute(sig["knee_acc"][i])
        st = ProsthesisState(knee_angle=sig["knee_ang"][i],
                             knee_velocity=vel, knee_accel=acc,
                             ankle_angle=sig["ankle_ang"][i],
                             ankle_velocity=a_vel_lp.compute(sig["ankle_vel"][i]),
                             ankle_accel=a_acc_lp.compute(sig["ankle_acc"][i]))

        # Genuine toe-off edges only; see the note on the truncated first swing.
        if i > 0 and swing[i] and not swing[i - 1]:
            # Toe-off. Take the stance duration from the baseline's own latches,
            # read a few ticks in so the triggered latch has certainly updated.
            j = min(i + 5, n - 1)
            stance_dur = sig["t_toeoff"][j] - sig["t_heelstrike"][j]
            ctrl.reset(st, t, prev_stance_duration=stance_dur)
            events.append((t, ctrl._T,
                           clamp(cfg.swing_to_stance_ratio * stance_dur,
                                 cfg.swing_duration_min, cfg.swing_duration_max),
                           sig["knee_ang"][i] - sig["toeoff_knee"][j]))

        if ctrl.t0 is not None and swing[i]:
            traj[i] = ctrl.knee.setpoint(t - ctrl.t0)[0]
            a_traj[i] = ctrl.ankle.setpoint(t - ctrl.t0)[0]
            out = ctrl.compute(st, t)
            torque[i] = out.knee_desired
            a_torque[i] = out.ankle_desired

    return traj, torque, a_traj, a_torque, np.array(events), swing


def _stats(ours, ref, m):
    e = ours[m] - ref[m]
    return (np.corrcoef(ours[m], ref[m])[0, 1],
            np.sqrt((e ** 2).mean()), np.abs(e).max())


def main():
    sig = load()
    traj, torque, a_traj, a_torque, events, swing = replay(sig)
    m = swing & np.isfinite(traj)

    ref_traj = sig["knee_traj"]
    ff = sig["knee_ff"]
    ref_total = sig["knee_tq"]           # FF*1 + PID
    ref_pid = ref_total - ff             # the baseline's PID alone

    c, rms, mx = _stats(traj, ref_traj, m)
    print("TRAJECTORY  ours vs walk p1")
    print(f"  swings {len(events)}   samples {m.sum()}")
    print(f"  corr {c:.6f}   RMS {np.degrees(rms):.4f} deg   "
          f"max {np.degrees(mx):.4f} deg")

    c, rms, mx = _stats(torque, ref_pid, m)
    print("\nTORQUE  ours vs baseline PID-only (walk Sum3 - FF)")
    print(f"  corr {c:.6f}   RMS {rms:.4f} Nm   max {mx:.4f} Nm")

    c, rms, mx = _stats(torque, ref_total, m)
    print("\nTORQUE  ours vs baseline TOTAL (what the leg actually got)")
    print(f"  corr {c:.6f}   RMS {rms:.4f} Nm   max {mx:.4f} Nm")

    c, rms, mx = _stats(a_traj, sig["ankle_traj"], m)
    print("\nANKLE TRAJECTORY  ours (jerk-optimal knot accel) vs walk p4")
    print(f"  corr {c:.6f}   RMS {np.degrees(rms):.4f} deg   "
          f"max {np.degrees(mx):.4f} deg")
    print(f"  ankle range: ours {np.degrees(a_traj[m].min()):.2f}..."
          f"{np.degrees(a_traj[m].max()):.2f}  baseline "
          f"{np.degrees(sig['ankle_traj'][m].min()):.2f}..."
          f"{np.degrees(sig['ankle_traj'][m].max()):.2f} deg")

    c, rms, mx = _stats(a_torque, sig["ankle_tq"], m)
    print("\nANKLE TORQUE  ours vs baseline (ankle FF gain is 0, so comparable)")
    print(f"  corr {c:.6f}   RMS {rms:.4f} Nm   max {mx:.4f} Nm")

    print("\nWhat the dropped feedforward was contributing")
    print(f"  FF     RMS {np.sqrt((ff[m]**2).mean()):.4f} Nm   "
          f"peak {np.abs(ff[m]).max():.4f} Nm")
    print(f"  PID    RMS {np.sqrt((ref_pid[m]**2).mean()):.4f} Nm   "
          f"peak {np.abs(ref_pid[m]).max():.4f} Nm")
    print(f"  total  RMS {np.sqrt((ref_total[m]**2).mean()):.4f} Nm   "
          f"peak {np.abs(ref_total[m]).max():.4f} Nm")
    print(f"  FF share of total RMS: "
          f"{np.sqrt((ff[m]**2).mean()) / np.sqrt((ref_total[m]**2).mean()):.1%}")

    print("\nper-swing swing_duration (ours vs baseline latches):")
    for k, (t0, T, T_base, ic) in enumerate(events):
        print(f"  {k:2d}  t={t0:7.3f}  T={T:.4f}  base_T={T_base:.4f}  "
              f"IC angle delta={np.degrees(ic):+.3f} deg")

    t = sig["time"]
    fig, ax = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

    ax[0].plot(t, np.degrees(ref_traj), lw=1.6, label="baseline commanded")
    ax[0].plot(t, np.degrees(traj), lw=1.1, ls="--", label="ours")
    ax[0].plot(t, np.degrees(sig["knee_ang"]), lw=0.7, alpha=0.45,
               label="measured knee angle")
    ax[0].set_ylabel("knee angle [deg]")
    ax[0].set_title("Min-jerk knee swing vs baseline "
                    f"(trajectory corr {_stats(traj, ref_traj, m)[0]:.6f})")
    ax[0].legend(loc="upper right", fontsize=8)

    ax[1].plot(t, ref_pid, lw=1.6, label="baseline PID only (Sum3 - FF)")
    ax[1].plot(t, torque, lw=1.1, ls="--", label="ours (PID only)")
    ax[1].set_ylabel("knee torque [Nm]")
    ax[1].legend(loc="upper right", fontsize=8)

    ax[2].plot(t, ref_total, lw=1.4, color="k", label="baseline TOTAL (PID+FF)")
    ax[2].plot(t, torque, lw=1.1, ls="--", label="ours (no FF)")
    ax[2].plot(t, ff, lw=1.0, alpha=0.7, color="tab:orange",
               label="FF we dropped")
    ax[2].set_ylabel("knee torque [Nm]")
    ax[2].legend(loc="upper right", fontsize=8)

    ax[3].fill_between(t, 0, swing.astype(float), step="mid", alpha=0.3)
    ax[3].set_ylabel("swing")
    ax[3].set_xlabel("time [s]")
    for a in ax:
        a.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PNG, dpi=130)
    print(f"\nwrote {PNG}")


if __name__ == "__main__":
    main()
