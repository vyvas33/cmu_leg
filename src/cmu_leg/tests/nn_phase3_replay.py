"""Replay Phase3 against the baseline's logged phase 3 command.

Phase 3 fired in only 2 of the 6 logged swings -- its entry needs the knee to
cross below `hip_angle - offset`, and usually heel strike comes first -- but it
ran for 311 and 501 ms in those two, a third to a half of the swing.

Entry velocity and acceleration are pinned from the log, as the first two
discrete derivatives of `knee_desired` on the tick before entry, since that
latch belongs to the enclosing controller. Hip angle is filtered at 50 Hz, and
the 26 deg clamp is off for the same reason as in the phase 2 replay.

References, all `nntorque`:
    col 20  phase 3 knee command
    col 21  phase 3 knee velocity
    col 22  gamma, the weight still on the entry velocity

Run:  uv run python -m cmu_leg.tests.nn_phase3_replay

Authors: vkalva, Claude
"""
from dataclasses import replace

import numpy as np
import scipy.io as sio

from cmu_leg.config.params import COMMON
from cmu_leg.control.high_level.walking.nn_swing import Phase3
from cmu_leg.tests.nn_phase2_replay import CFG, LOG_RATE, MAT, WALK_START_S, load

NT = dict(phase3=4, knee=20, knee_vel=21, gamma=22, time=25)


def episodes(flag):
    d = np.diff(np.r_[0, flag != 0, 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def main():
    sig = load()
    sd = sio.loadmat(str(MAT), squeeze_me=True,
                     struct_as_record=False)["scope_data"]
    nt = sd.nntorque.data
    w = nt[:, NT["time"]] >= WALK_START_S
    ref = {k: nt[w, c].astype(float) for k, c in NT.items()}

    t, kd, dt = sig["time"], sig["knee_desired"], 1.0 / LOG_RATE
    eps = episodes(ref["phase3"])
    print(f"{len(eps)} phase 3 episodes\n")
    print("   start     ticks   entry vel   max|err|      rms     gamma (ours/base)")

    worst = 0.0
    for s, e in eps:
        p = Phase3(cfg=CFG, common=replace(COMMON, control_sample_rate=LOG_RATE))
        vel = (kd[s - 1] - kd[s - 2]) / dt
        accel = (kd[s - 1] - 2 * kd[s - 2] + kd[s - 3]) / dt ** 2
        p.reset(kd[s - 1], vel, accel, t[s])

        err = np.array([abs(p.compute(t[k], sig["hip_angle"][k]) - ref["knee"][k])
                        for k in range(s, e)])
        worst = max(worst, err.max())
        print(f"   {t[s]:7.3f}  {e - s:5d}  {vel:+9.3f}  {np.degrees(err.max()):9.4f} deg"
              f"  {np.degrees(err.std()):7.4f}   {p.gamma:.4f}/{ref['gamma'][e - 1]:.4f}")

    print(f"\n   worst over all ticks: {worst:.3e} rad = {np.degrees(worst):.3e} deg")


if __name__ == "__main__":
    main()
