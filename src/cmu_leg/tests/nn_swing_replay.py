"""End-to-end replay of the NN swing knee path against the logged knee torque.

Everything downstream of the bus runs: predictor, phase chart, all three phase
laws, the clamp and the pid. The reference is `nntorque` col 0, the baseline's
own `nn knee torque`.

Two variants, so a torque error can be attributed:

    logged nn_traj   the predictor is replaced by the trial's own trajectories,
                     leaving chart + phases + clamp + pid under test
    our predictor    the whole chain, including the network

The pid is never reset at swing entry: the baseline's reset port is fed by
difference blocks inside the swing-enabled subsystem, so they see no edge and
it resumes with the previous swing's state. Resetting costs 2.2 Nm rms vs 0.6.

State comes off the log rather than through our estimator, and the gait flags
are the logged `stance` / `standing` rather than our contact detection, so a
disagreement here is the swing controller and nothing upstream of it.

Run:  uv run python -m cmu_leg.tests.nn_swing_replay
Writes nn_swing_replay.png next to this file.

Authors: vkalva, Claude
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cmu_leg.config.params import COMMON, NNSWING
from cmu_leg.control.utils import Lowpass, PID, clamp
from cmu_leg.control.high_level.walking.nn_predictor import (NeuralPredictor,
                                                             SPLINE_FUTURE)
from cmu_leg.control.high_level.walking.nn_swing import (Phase, PhaseChart,
                                                         Phase1, Phase2, Phase3)
from cmu_leg.control.high_level.walking import retargeting
from cmu_leg.control.utils import BackwardDifference
from cmu_leg.structures.state import ProsthesisState
from cmu_leg.tests.nn_phase2_replay import CFG, HIP_ANGLE_CUTOFF, HIP_VEL_CUTOFF
from cmu_leg.tests.nn_phase_chart_replay import (HEEL_ANGLE, HEEL_RADIUS,
                                                 THIGH_LENGTH)

HERE = Path(__file__).parent
MAT = HERE / "data" / "gaitData20251015_test7_NN_vicon.mat"
PNG = HERE / "nn_swing_replay.png"

BASE = dict(hip_angle=0, knee_angle=7, hip_x=58, hip_z=60, knee_x=61,
            knee_z=63, hip_z_comp=86, stance=29, standing=30, hip_vel=1)
LOG_RATE = 1000.0
WALK_START_S = 16.5


def load():
    sd = sio.loadmat(str(MAT), squeeze_me=True,
                     struct_as_record=False)["scope_data"]
    base, nt, nn = sd.base.data, sd.nntorque.data, sd.nn.data
    w = nt[:, 25] >= WALK_START_S

    sig = {k: base[w, c].astype(float) for k, c in BASE.items()}
    sig["time"] = nt[w, 25].astype(float)
    sig["torque"] = nt[w, 0].astype(float)
    sig["knee_desired"] = nt[w, 23].astype(float)
    sig["armed"] = nn[w, 0].astype(float)
    sig["traj"] = nn[w, 2:43].astype(float)
    sig["phase"] = np.where(nt[w, 2] != 0, 1,
                            np.where(nt[w, 3] != 0, 2,
                                     np.where(nt[w, 4] != 0, 3, 0)))

    for name, cut in (("hip_angle_f", HIP_ANGLE_CUTOFF),
                      ("hip_vel_f", HIP_VEL_CUTOFF)):
        src = base[:, BASE["hip_angle" if "angle" in name else "hip_vel"]]
        lp = Lowpass(cutoff_freq=cut, sample_rate=LOG_RATE, order=2)
        lp.reset(float(src[0]))
        sig[name] = np.array([lp.compute(float(x)) for x in src])[w]
    return sig


class LoggedPredictor:
    """Stands in for NeuralPredictor, serving the trial's own trajectories."""

    def __init__(self, sig):
        self._sig = sig
        self.k = 0

    ready = True

    def update(self, *_):
        pass

    def predict(self):
        return SPLINE_FUTURE @ self._sig["traj"][self.k]


def replay(sig, predictor):
    """Run the swing chain over the trial. Returns torque and phase per tick."""
    dt = 1.0 / LOG_RATE
    common = replace(COMMON, control_sample_rate=LOG_RATE)
    chart = PhaseChart(CFG, hip_to_knee=THIGH_LENGTH,
                       heel_radius=HEEL_RADIUS, heel_angle=HEEL_ANGLE)
    p1, p2, p3 = Phase1(CFG, common), Phase2(CFG, common), Phase3(CFG, common)
    pid = PID(CFG.knee_kp, CFG.knee_ki, CFG.knee_kd, CFG.deriv_filter_n)
    d_vel, d_acc = BackwardDifference(), BackwardDifference()

    st = ProsthesisState()
    knee_cmd = sig["knee_desired"][0]
    vel = acc = 0.0
    torque = np.zeros(len(sig["time"]))
    phases = np.zeros(len(sig["time"]), int)

    for k, t in enumerate(sig["time"]):
        if hasattr(predictor, "k"):
            predictor.k = k
        st.hip_angle = sig["hip_angle"][k]
        st.hip_pos_x, st.hip_pos_z = sig["hip_x"][k], sig["hip_z"][k]

        in_stance = sig["stance"][k] != 0
        if in_stance:
            knee_cmd = sig["knee_angle"][k]
        predictor.update(st, knee_cmd, t)

        if in_stance or not sig["armed"][k]:
            chart.reset()
            d_vel.reset(knee_cmd)
            d_acc.reset(0.0)
            vel = acc = 0.0
            continue

        phase = chart.update(True, sig["standing"][k] != 0, knee_cmd,
                             sig["hip_angle_f"][k], sig["hip_x"][k],
                             p1.descending)
        phases[k] = int(phase)
        if phase is Phase.IDLE:
            continue

        if chart.entered:
            if phase is Phase.ONE:
                p1.reset(knee_cmd, t)
            elif phase is Phase.TWO:
                p2.reset(knee_cmd, vel, sig["hip_vel_f"][k], t)
            else:
                p3.reset(knee_cmd, vel, acc, t)

        if phase is Phase.ONE:
            if p1.needs_trajectory(t):
                p1.set_trajectory(predictor.predict())
            knee_cmd = p1.compute(t)
        elif phase is Phase.TWO:
            knee_cmd = p2.compute(t, sig["hip_angle_f"][k], sig["hip_vel_f"][k])
        else:
            knee_cmd = p3.compute(t, sig["hip_angle_f"][k])

        vel = d_vel.compute(knee_cmd, dt)
        acc = d_acc.compute(vel, dt)
        error = clamp(knee_cmd, CFG.knee_min, CFG.knee_max) - sig["knee_angle"][k]
        torque[k] = pid.compute(error, dt)

    return torque, phases


def summarize(name, ours, sig, m):
    err = np.abs(ours[m] - sig["torque"][m])
    corr = np.corrcoef(ours[m], sig["torque"][m])[0, 1]
    print(f"   {name:<18} corr {corr:.6f}   rms {err.std():6.3f} Nm"
          f"   median {np.median(err):6.3f}   p99 {np.percentile(err, 99):7.3f}"
          f"   max {err.max():7.3f}")


def main():
    sig = load()
    retargeting.L_THIGH = THIGH_LENGTH
    retargeting.L_HUMAN = 0.5 * (THIGH_LENGTH + retargeting.L_SHANK)

    m = sig["phase"] != 0
    print(f"{m.sum()} swing-phase ticks, baseline torque "
          f"{sig['torque'][m].min():.1f} .. {sig['torque'][m].max():.1f} Nm\n")

    t_log, ph_log = replay(sig, LoggedPredictor(sig))
    t_ours, ph_ours = replay(sig, NeuralPredictor(
        replace(COMMON, control_sample_rate=LOG_RATE)))

    summarize("logged nn_traj", t_log, sig, m)
    summarize("our predictor", t_ours, sig, m)
    print(f"\n   phase agreement: logged-traj {100*(ph_log == sig['phase']).mean():.3f}%"
          f"   ours {100*(ph_ours == sig['phase']).mean():.3f}%")

    plot(sig, t_log, t_ours, m)


def plot(sig, t_log, t_ours, m):
    t = sig["time"]
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    y = np.where(m, sig["torque"], np.nan)
    ax[0].plot(t, y, "k", lw=1.2, label="baseline")
    ax[0].plot(t, np.where(m, t_log, np.nan), "C0", lw=0.9, label="logged nn_traj")
    ax[0].plot(t, np.where(m, t_ours, np.nan), "C3--", lw=0.9, label="our predictor")
    ax[0].set_ylabel("knee torque [Nm]")
    ax[0].legend(loc="upper right")
    ax[1].plot(t, np.where(m, np.abs(t_log - sig["torque"]), np.nan), "C0", lw=0.8)
    ax[1].plot(t, np.where(m, np.abs(t_ours - sig["torque"]), np.nan), "C3", lw=0.8)
    ax[1].set_ylabel("|error| [Nm]")
    ax[1].set_xlabel("trial time [s]")
    ax[1].set_yscale("log")
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"\nwrote {PNG}")


if __name__ == "__main__":
    main()
