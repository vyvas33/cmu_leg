"""Replay PhaseChart against the baseline's logged phase flags.

The chart lives inside `NN swing torque Control`, which is enabled by
`computation_signal AND nn_enable`, so it only runs while the NN path is armed.
Ungated, it reports phase 1 through every swing of the trial and agreement
falls to 89%.

The leg constants are pinned to the trial's, from params.m: thigh 0.71
("0.71 for Russell"), heel radius 0.147, heel angle atan(0.09/0.096). They have
to be taken as a set -- our heel radius is 0.1316, and pairing it with a swept
thigh makes 0.68 look right when it is not.

References:
    nntorque cols 2, 3, 4   the three phase flags
    nn       col 0          computation_signal, the arming gate
    base     29, 30         stance, standing
    base     58             hip x
    base     0              hip angle, filtered at 50 Hz as the chart sees it

Run:  uv run python -m cmu_leg.tests.nn_phase_chart_replay

Authors: vkalva, Claude
"""
import numpy as np
import scipy.io as sio

from cmu_leg.control.utils import Lowpass
from cmu_leg.control.high_level.walking.nn_swing import PhaseChart
from cmu_leg.tests.nn_phase2_replay import (HIP_ANGLE_CUTOFF, LOG_RATE, MAT,
                                            WALK_START_S)

# The trial's constants, taken as a SET from params.m. Mixing them with ours
# is what made an earlier pass read the thigh as 0.68: our heel radius is
# hypot(0.09, 0.096) = 0.1316, while the baseline carries a separately measured
# 0.147. With all four from one source the chart matches exactly.
THIGH_LENGTH = 0.71
HEEL_RADIUS = 0.147
HEEL_ANGLE = np.arctan(0.09 / 0.096)


def main():
    sd = sio.loadmat(str(MAT), squeeze_me=True,
                     struct_as_record=False)["scope_data"]
    base, nt, nn = sd.base.data, sd.nntorque.data, sd.nn.data

    lp = Lowpass(cutoff_freq=HIP_ANGLE_CUTOFF, sample_rate=LOG_RATE, order=2)
    raw = base[:, 0].astype(float)
    lp.reset(raw[0])
    hip = np.array([lp.compute(x) for x in raw])

    w = nt[:, 25] >= WALK_START_S
    t, stance, standing, hip_x = nt[w, 25], base[w, 29], base[w, 30], base[w, 58]
    hip, knee, neg, armed = hip[w], nt[w, 23], nt[w, 24], nn[w, 0]
    ref = np.where(nt[w, 2] != 0, 1,
                   np.where(nt[w, 3] != 0, 2, np.where(nt[w, 4] != 0, 3, 0)))

    chart = PhaseChart(hip_to_knee=THIGH_LENGTH, heel_radius=HEEL_RADIUS,
                       heel_angle=HEEL_ANGLE)
    ours = np.empty(len(ref), int)
    for k in range(len(ref)):
        if armed[k]:
            ours[k] = int(chart.update(stance[k] == 0, standing[k] != 0,
                                       knee[k], hip[k], hip_x[k], neg[k] != 0))
        else:
            chart.reset()
            ours[k] = 0

    ok = ours == ref
    print(f"agreement {100 * ok.mean():.3f}%   "
          f"({(~ok).sum()} of {len(ref)} ticks differ)\n")
    for p, name in enumerate(("idle", "phase 1", "phase 2", "phase 3")):
        m = ref == p
        print(f"   {name:<8} {m.sum():6d} ticks   match {100 * ok[m].mean():7.3f}%")

    d = np.where(~ok)[0]
    if len(d):
        print("\n   disagreements (t, ours, logged):")
        for i in d:
            print(f"     {t[i]:.3f}   {ours[i]}   {ref[i]}")


if __name__ == "__main__":
    main()
