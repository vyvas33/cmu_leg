"""Convert the baseline's NN swing weights into one .npz for the runtime.

    uv run python tools/build_nn_weights.py

Reads two .mat files from the NPC repo and writes
config/data/nn_swing_weights.npz. Re-run it if the trained network changes, or
if PLAYBACK_DT changes -- the spline matrix is the only rate-dependent thing in
the file, and it is baked at build time so the runtime never calls scipy.

What ends up in the .npz
  A1..A4, b1..b4   MLP weights. 126 -> 128 -> 64 -> 64 -> 41, ReLU between,
                   linear out. b are stored as column vectors, ready to add.
  mean, std        (6,) each, per input channel, in CHANNELS order. The .mat
                   stores these interleaved as a flat 12-vector; unpacking here
                   means the runtime never does index arithmetic on it.
  knee_mean,       scalars, de-standardize the 41 outputs. Same numbers as
  knee_std         mean[1]/std[1] -- the network predicts the channel it also
                   consumes -- but named separately because they are used for a
                   different purpose and a future net might not share them.
  spline_future    (n_future, 41). Resamples the 41-point prediction onto
                   PLAYBACK_DT and keeps only the future half, in one matmul.
                   See below.

The spline
  tune_NN_traj interpolates the 41 points (5 ms apart, spanning -100..+100 ms)
  onto 1 ms with MATLAB's spline(), then keeps samples 102:201 -- the future
  100 ms. MATLAB's spline() is a not-a-knot cubic, which scipy matches with
  CubicSpline(..., bc_type="not-a-knot").

  Knots and query points are both fixed, so that whole operation is linear in
  the 41 samples: resample the 41 unit basis vectors once, stack the columns,
  and interpolation becomes a matrix multiply. Measured 44 us for the scipy
  call vs 1.2 us for the matmul, exact to 2e-15, and it keeps scipy out of the
  control loop entirely.

Not in the .npz, deliberately
  The geometry transforms (human <-> leg) need LEG, which is ours and subject
  specific, so they live in the predictor. The structural constants (21 input
  samples, 5 ms spacing, the 1.4 height ratio) are properties of the training
  set: they belong beside the loader as constants, not in a tunable config.
"""
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.interpolate import CubicSpline

NPC = Path("/Users/vyvaswath/Documents/Personal/Research/CMU_Leg"
           "/prosthesis_control_NPC/Learning_in_matlab")
OUT = Path(__file__).resolve().parents[1] / "src/cmu_leg/config/data"

WEIGHTS_MAT = NPC / "param_net_50ms.mat"
NORMS_MAT = NPC / "all_mean_std_50ms.mat"

# Input channel order, which is also the order mean/std are stored in. The
# network's 126-vector is these six interleaved per timestep.
CHANNELS = ("hip_angle", "knee_angle", "hip_x", "hip_z", "knee_x", "knee_z")

N_OUT = 41              # network output points
OUT_DT = 0.005          # spacing of those points [s]
PLAYBACK_DT = 0.001     # spacing we resample onto [s]


def spline_matrix(n_src: int, src_dt: float, dst_dt: float):
    """Linear operator for not-a-knot resampling, future half only.

    Returns (M, n_future) with M of shape (n_future, n_src): M @ y is the same
    as splining y onto dst_dt and keeping the samples strictly after the
    midpoint. The 41 points straddle 'now', so the first half is the network
    restating the past and only the second half is a prediction.
    """
    span = (n_src - 1) * src_dt                  # 0.200 s
    n_dst = round(span / dst_dt) + 1             # 201
    src = np.arange(n_src) * src_dt
    dst = np.linspace(0.0, span, n_dst)

    n_past = n_dst // 2 + 1                      # index of 'now' (101), dropped
    m = np.empty((n_dst - n_past, n_src))
    for i in range(n_src):
        e = np.zeros(n_src)
        e[i] = 1.0
        m[:, i] = CubicSpline(src, e, bc_type="not-a-knot")(dst)[n_past:]
    return np.ascontiguousarray(m), n_dst - n_past


def main():
    w = sio.loadmat(str(WEIGHTS_MAT))
    norms = sio.loadmat(str(NORMS_MAT))["all_mean_std"].ravel().astype(float)
    if norms.size != 2 * len(CHANNELS):
        raise ValueError(f"expected {2 * len(CHANNELS)} norm constants, "
                         f"got {norms.size}")

    layers = {}
    prev = None
    for i in (1, 2, 3, 4):
        a = np.ascontiguousarray(w[f"A{i}"], dtype=float)
        b = np.ascontiguousarray(w[f"b{i}"], dtype=float).reshape(-1, 1)
        if a.shape[0] != b.shape[0] or (prev is not None and a.shape[1] != prev):
            raise ValueError(f"layer {i} shape mismatch: A{a.shape} b{b.shape}")
        prev = a.shape[0]
        layers[f"A{i}"], layers[f"b{i}"] = a, b

    n_in = layers["A1"].shape[1]
    if n_in % len(CHANNELS):
        raise ValueError(f"input width {n_in} is not a multiple of "
                         f"{len(CHANNELS)} channels")
    if prev != N_OUT:
        raise ValueError(f"output width {prev} != {N_OUT}")

    # the flat 12-vector alternates mean, std per channel
    mean, std = norms[0::2], norms[1::2]

    m, n_future = spline_matrix(N_OUT, OUT_DT, PLAYBACK_DT)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "nn_swing_weights.npz"
    np.savez(path, **layers, mean=mean, std=std,
             knee_mean=mean[CHANNELS.index("knee_angle")],
             knee_std=std[CHANNELS.index("knee_angle")],
             spline_future=m,
             channels=np.array(CHANNELS),
             n_hist=n_in // len(CHANNELS), hist_dt=OUT_DT,
             playback_dt=PLAYBACK_DT)

    print(f"wrote {path}")
    print(f"  mlp {' -> '.join(str(layers[f'A{i}'].shape[1]) for i in (1, 2, 3, 4))}"
          f" -> {prev}   ({sum(v.size for v in layers.values())} params)")
    print(f"  inputs {n_in} = {len(CHANNELS)} channels x {n_in // len(CHANNELS)}"
          f" samples @ {OUT_DT * 1e3:g} ms")
    print(f"  spline_future {m.shape}  ({n_future} pts @ {PLAYBACK_DT * 1e3:g} ms"
          f" = {n_future * PLAYBACK_DT * 1e3:g} ms horizon)")
    for c, mu, sd in zip(CHANNELS, mean, std):
        print(f"  {c:<11} mean {mu:>10.4f}   std {sd:>9.4f}")


if __name__ == "__main__":
    main()
