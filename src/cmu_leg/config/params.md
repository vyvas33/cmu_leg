# Controller parameters

The reasoning behind `config/params.py`. The file itself keeps one line per
field; everything that needed a paragraph lives here.

Ported from `baseline_controller/params.m` and its `.mat` files (`sys_id_*`,
`ankle_stiffness_params`, `spring_zero_positions`, `encoder_counts_to_rad_polys`,
`ankle_angle_offset`, `dynamic_params`).

---

## CommonParams

**`control_sample_rate`** — the merged ESKF + high-level process cannot hold
1 kHz on the Jetson, so it runs at 500 Hz. Fixed-coefficient filters ticking in
that process (the standing velocity lowpasses) must be *designed* at this rate,
not at `sample_rate`, or their effective cutoff is wrong. `launch.py` derives
`DT_CONTROL` from it.

**`dt_min` / `dt_max`** — the measured-dt clamp for every differentiator and
integrator. `dt_max` stops a scheduler-stall tick (~21 ms observed) integrating
as one step; `dt_min` bounds the 1/dt quantization-noise gain on the velocity
derivatives.

**`startup_mute_ticks`** — post-enable ticks the pipeline runs muted while
filters settle. The baseline's LLC "1 s Delay" = 1000 ticks at 1 kHz.

**`lowpass_cutoff` / `lowpass_order`** — default torque-conditioning lowpass,
unused until those blocks take parameters.

---

## RuntimeParams

Multiprocess runtime (`runtime/launch.py`, `low_level.py`, `control.py`). Loop
periods derive from the rates in `CommonParams`; these are the orchestration and
safety knobs that used to be scattered across those files.

**`tmax`** — counted from the **first actuating tick**, the moment the mode FSM
first reaches STANDING or WALKING, not from the synced origin. Warmup no longer
eats into the run: a fixed window from t0 would give an arbitrary amount of
usable data depending on how long the drives took to enable.

**`max_arm_wait_s`** — ceiling on the wait for that first actuating tick. If the
operator never asks, or the drives never enable, the run ends here instead of
spinning.

**`rt_priority`** — SCHED_FIFO priorities for (low-level, control), applied by
each child itself. Low-level outranks control: it holds the 1 kHz EtherCAT
cycle, and a stall past the drives' sync-manager watchdog (measured between 2.5
and 2.8 ms) quick-stops them. Both sit below 99 so kernel threads landing on the
core can still pre-empt. Needs root or `CAP_SYS_NICE`; without it the child warns
and continues at default priority.

**`low_warmup_s`** — when `ec.enable` is commanded, measured from the **synced
origin**: the moment control finishes IMU bring-up and stamps `start_time`, not
from launch. Wall-clock is therefore (EtherCAT init + IMU init + this). It no
longer has to clear anything on the control side, because the mode FSM waits for
`state.control_enabled` — the drives actually reaching OPERATION_ENABLED — so an
enable walk that runs long delays standing instead of being ramped into.

> `control_warmup_s` was removed: standing is now the resting operator request,
> matching the baseline's `standing_mode_toggle` constant of 1, so there is
> nothing left to schedule.

**`allow_motion`** — a bring-up gate on **motion**, not on commanding. False
still energizes the drives, still walks CiA-402 to OPERATION_ENABLED, and still
runs the whole mode FSM and low-level pipeline; it just substitutes a zero
`DriveCommand`, which in CSV is target-velocity-zero, i.e. the drives hold. The
leg is stiff, not limp, and nothing moves. (Was `command_motors`, which read as
"do not command the motors" when we very much still do.)

**`drive_knee` / `drive_ankle`** — per-joint gate applied at the wire in
`low_level.run()`. False sends a zero `DriveCommand` for that joint, so the drive
holds it rather than leaving it free. Everything upstream still runs and still
logs, so the joint's commanded torque stays visible.

The ankle is off by choice, not because it misbehaves. The 42 Hz standing
oscillation was a coupled knee+ankle+rig mode, not an ankle torque-loop limit
cycle, and the ankle has since run fine on hardware. It stays gated only
because the controllers being brought up are knee-only, and a policy that
commands `ankle_desired=0.0` would leave the foot limp rather than held.

**`walk_after_standing_s`** — when `standing_toggle` drops, measured as a
**delta from the mode FSM reaching STANDING**, not from the synced origin. The
control process is spawned, and multiprocessing replaces a child's stdin with
devnull, so no keypress source can come up there; the schedule is the only way
to drive a run.

It replaced an absolute `bench_walk_at`, which had two failure modes a delta
cannot have. A slow CiA-402 enable walk ate the standing window instead of
sliding the whole sequence. And an absolute time that landed before standing
dropped `standing_toggle` while the FSM was still in INITIALIZING — whose guard
needs it **true** — parking the run there for good.

It is a **permission, not a trigger**: STANDING → WALKING still needs the leg
unloaded, so nothing happens until you take a step. Being early therefore costs
nothing; being late means the leg will not swing when you do. Set it just past
`STANDING.duration` (the pose ramp), and well before the treadmill belts move if
they are in play — see `runtime/README.md`. `None` = never walk.

**`bench_swing_at`** — the scripted stand-in for the space key, for a leg in a
harness where the foot never loads and no toe-off is ever produced. Entering
walking already fires one swing of its own, so the first entry is spaced clear of
it. `tmax` runs from the first actuating tick, so the last entry has to leave
room for its own swing to finish.

---

## TreadmillParams

The Bertec belts, driven from a run by `BeltSupervisor` in the parent process.
`host` / `port` / `timeout_s` are the Control Panel socket; the rest govern belt
motion.

**`enabled`** — the deliberate per-session opt-in. Belts move only with this
**and** `RUNTIME.arm_motion`: arming the leg is the *veto*, not the permission,
because `arm_motion` is true on every walking run including bench and overground
ones. With `require_contact` false there is no physical precondition left, so
this flag is the only thing between a bench run and the belts spinning up behind
you. Put it back to `False` afterwards.

**`speed` / `acc`** — belt speed and the planned ramp. `max_speed` refuses a
configured `speed` above it, to catch a typo before it reaches the hardware.

**`stop_acc`** — fault stops, deliberately harder than `acc`: 0.8 m/s dies in
~0.5 s instead of 1.6.

**`countdown_s`** — from "the leg has finished standing" (`STANDING.duration`
after the mode FSM reaches STANDING) to the belts moving.

**`require_contact`** — also require the prosthetic foot loaded before starting.
False for treadmill work, where you are certainly standing on the belt; turn it
on for overground, and bench-check that contact asserts when you stand the way
you stand.

**`max_stale_passes`** — monitor-loop passes with no advance in `b3.seq` before
the control process is called hung and the belts stop. `seq` is used rather than
`Process.is_alive()`, which stays true for a process stuck in a syscall.

The full sequence, the stop conditions and what the software cannot cover are in
`runtime/README.md`.

---

## KneeParams / AnkleParams

**`flex_ext` / `extend_ext`** — per-stop sign multipliers (+1 upper, −1 lower).
Wiring into `KneeTorqueConditioning` is pending.

**`up_limit`** — alias for the conditioning block's `knee_up_limit`, which is
`angle_max_soft`.

**Ankle `motor_eff`** — efficiency compensation is disabled (see `arch.md`).

**Ankle `stiffness_poly`** — nonlinear Nm/rad fit, not yet used.

**Ankle `angle_offset`** — 0, to match NPC `params.m`, where the
`ankle_angle_offset.mat` load is commented out and the Sep-2025 zeroing assumes
no extra offset. The old value was 0.0444.

**Ankle `torque_ctrl_spring_vel_lowpass_*`** — the ankle torque controller's
spring-velocity lowpass. The spring-acceleration branch is unported.

**Knee `vel_max`** — was 5000 before the current derate.

---

## EncoderParams

From `spring_zero_positions.mat`: linear scales, zeros, hall offsets. Motor
counts-to-rad lives on the joint params. Used by Get Prosthesis State. Values
updated July 2026.

`knee_abs_counts_to_rad` is empirical and sign-flipped — it is **not** 2π/2¹⁷.
`ankle_abs_counts_to_rad` is the 17-bit 2π/2¹⁷.

---

## IMUParams

Two WitMotion HWT906, thigh and foot, over USB-serial. Set the port paths from
`imu_io.discover_imu_ports()` (which CH340 is which IMU) — they can swap across
reboots and replugs, so update after discovery.

acc/gyro ranges match the baseline Simulink Read IMU block (±16 g, ±2000 dps).
Baud 921600 is reliable on Linux. Keep `lpf_cutoff_hz` ≤ 0.4 × `stream_rate_hz`.

---

## EncoderCorrection

From `encoder_counts_to_rad_polys.mat` (Feb 2024) — the set paired with the
Sep-2025 zero offsets. Knee angle and ankle spring angle each add a normalized
polynomial correction: normalize counts `((x - mean)/std)`, evaluate `*_err_poly`,
de-normalize `(* err_std + err_mean)`. Polynomials are highest-power-first, in
`polyval` order.

Polys, means and stds must always come from the same `.mat` as each other.

---

## StandingParams

Baseline "Standing Mode Control". A min-jerk ramp from the latched start pose to
a fixed pose over `duration`, then per-joint PD:

```
tau = kp*(theta_traj - theta) - kd*theta_dot_filt
```

`theta_dot_filt` is the joint velocity through a Butterworth lowpass; the angle
stays raw.

---

## ImpedanceParams

Three-state finite-state impedance stance (`ImpedanceStance`). Each phase applies
`tau = -k*(angle - off) - b*angle_dot` for knee and ankle. Phase advance:
Stance1 → Stance2 when `knee_angle > stance_12_threshold`, Stance2 → Stance3 when
`ankle_angle > stance_23_threshold`.

Source: `impedance_control_basic/ParameterLibrary/impedance_control/optimized_gains_linear_v2.mat`
— a 20-vector demuxed in the baseline's "Split up parameters" block. Index order
verified against that demux (2026-08-10): 1–12 k/b interleaved per stance, 13–18
offsets, 19–20 thresholds. v2 differs from `optimized_gains_linear.mat` only in
`stance_12_threshold` (0.18 vs 0.3649); every other element is identical.

Units: k [Nm/rad], b [Nm/(rad/s)], offsets and thresholds [rad].

**Clamped damping.** Three b values were *negative* in the `.mat` (optimizer
output). The baseline floors all six at `damping_lower_lim` on load, so the joints
never see them; the clamped values are stored here for the same reason — what is
in this file is what runs. `ImpedanceStance` must **not** re-apply the floor.
`damping_lower_lim` is provenance for anyone reloading a different `.mat`.

`stance2.ankle_k` is negative stiffness; the baseline floors b only, not k.

---

## ContactParams

Gait phase detection (`WalkingControl`). Hall GRF proxies are conditioned in
`ProprioceptiveEstimator`; the threshold and the FSM live in the controller,
because `stance` never goes on the bus (see `structures/state.py`).

The baseline (`Get Prosthesis State/ini SM` + `SM`) auto-tares the halls by
averaging them **while unloaded** and comparing against that live zero, which
absorbs hall drift between sessions. Thresholds are in hall-count units and
`abs()` is applied, so the sign of the hall wiring does not matter.

**`baseline_tau`** — live-zero tracking, a first-order average of the GRF while
unloaded only. A long tau means the zero follows drift but not steps.

**`min_stance` / `min_swing`** — ours, not the baseline's. The baseline
de-chatters its FSM with a leg-angle guard (enter stance only if
`leg_angle < 85°`), which needs the IMU. Until that exists we use a refractory
hold: once a phase is entered it cannot be left for this long. Chosen over a
symmetric debounce because a debounce delays the heel-strike edge too, clipping
stance duty; a hold only suppresses spurious re-triggers.

On the Oct-2025 trial, `min_stance = 0.30` takes heel strikes from 16 (chatter)
to 7 (correct), and agreement with the baseline stance flag from 0.940 to 0.965.

> **Caution:** 0.30 s is ~35% of the stance duration in that trial (0.85–1.47 s).
> At a markedly faster cadence it would start truncating real stances. Revisit if
> walking speed goes up, and delete it once the leg-angle guard is available.

`min_swing` measured no benefit; kept as a knob.

---

## MJSwingParams

Min-jerk swing (`MinJerkSwing`), from `minjerk swing Control/Min Jerk Swing` in
`nm_nn_swing`. Values come from `nm_nn.m` section 2.5 and `params.m` — **not**
from `minjerk_params.m` in the same folder, which is an unused older script
holding different values.

Per joint the trajectory is two quintics in swing-relative time, solved once at
toe-off from the measured toe-off state:

```
segment 1  [0, flex_frac*T]  (theta_0, dtheta_0, ddtheta_0) -> (flex, 0, a_knot)
segment 2  [flex_frac*T, T]  (flex, 0, a_knot)              -> (extend, 0, 0)
```

Past `T` a clamp holds it at `(extend, 0, 0)`, which the segment-2 end constraints
already pin.

The knee pins its knot acceleration to a cadence-scaled Winter value,
`a_knot = winter_flex_knee_accel / (T / winter_swing_duration)**2`, which leaves
both knee segments fully determined. The ankle constrains only continuity of the
knot acceleration, leaving one free degree of freedom that the jerk cost resolves.

**Swing duration** adapts to cadence off the previous stance:
`T = clamp(swing_to_stance_ratio * prev_stance_duration, min, max)`.

**`winter_flex_knee_accel`** — from Winter's Appendix A4 relative joint angular
kinematics: the mean of the two samples straddling peak knee flexion, over a
swing of `winter_swing_duration`. Time-scaled to T by the 1/t² law. Negative
because it is decelerating flexion.

**Tracking gains.** A discrete PID on the error `e = theta_traj - theta`, forward
Euler on both integrator and derivative filter, no output saturation and no
anti-windup — the low-level torque conditioning bounds the command. Gains are
absolute Nm/rad, since the baseline's stance/swing mixing scales only the stance
branch by body weight.

The d term acts on the **error**, not on the measured velocity as in
`StandingController`. Swing `theta_dot_traj` peaks near 6.5 rad/s, where damping
the measurement would brake the commanded flexion instead of correcting lag.

**`knee_flex_vel` / `ankle_flex_vel`** — pinned by the formulation, not tuning
knobs: the knot is the flexion peak, so velocity there is zero by definition on
both sides.

`ankle_flex_frac` is earlier than the knee's, for clearance.

---

## NNSwingParams

NN swing (`control/high_level/walking/nn_swing.py`). The network's own constants
are not here — they are fixed by training and live with the weights.

**`trajectory_period`** — how much of a trajectory gets played before the next
replaces it. The baseline trips a 0.0505 s comparator and swaps on the following
tick, which is 52 ms at its 1 kHz rate.

**`knee_min` / `knee_max`** — applied to the summed phase output before the PID,
not to the stored command, so the data store can and does exceed these.

**`descend_hold`** — a trajectory that ends below where it starts means the knee
is turning back toward extension, which hands phase 1 over. It has to have been
playing this long first, so one descending prediction mid-flexion cannot end the
phase on its own.

**Phase 2**, the landing preparation, from Xing's eq 4.3–4.4. The knee is driven
to `hip_angle - hip_knee_offset`, which holds the shank at a fixed forward lean
for touchdown. Its velocity is `slope * hip_vel`, and slope ramps from whatever
phase 1 left to `slope_target`. `slope_step` is per baseline tick; `baseline_dt`
converts it to the per-second rate the port ramps at.

- `hip_vel_target` — floor under the hip velocity, ramped to over the same
  interval. Stops the knee freezing when the hip stalls; it does **not** track
  cadence.
- `hip_angle_min` — lower clamp on hip angle, so the target never drops below
  `hip_angle_min - hip_knee_offset`.
- `decay_exponent` — exponent on the normalized error once the ramp ends. Above
  1, so the knee approaches the target asymptotically instead of crossing it.

**Phase 3**, Xing's eq 4.5 reached through the eq 4.6 blend. `gain_vel` and
`gain_accel` are decay rates per baseline tick: velocity fades onto the
target-tracking command over ~1/`gain_vel` ticks, the entry acceleration over
~1/`gain_accel`.

---

## ModeParams

Mode FSM (`ProsthesisControl`). Mirrors the baseline's
`Get Prosthesis State/State Machine`, minus the terms that need the IMU.

Baseline guards, for reference:

```
StandingMode -> Swing   [~contact && ~toggle && hip_velocity > 0
                         && leg_angle > 90 deg]
Stance -> StandingMode  [toggle && |hip_velocity| < 0.1
                         && |leg_angle - pi/2| < 10 deg]
```

Of those, only `~contact` is **safety-critical** — it is what guarantees walking
is entered unloaded, so the knee stop can never fire against a straight standing
knee. The `hip_velocity` / `leg_angle` terms are **intent** detection (is the
wearer actually trying to walk). Without the ESKF we keep the guarantee and lose
the intent, which means the operator has to be deliberate about when they flip
the toggle, but nothing unsafe can happen.

**`still_joint_velocity`** — "nearly still", the encoder-only stand-in for
`|hip_velocity| < 0.1`. Joint velocities are all we have without the IMU.

**`enable_dwell`** — how long the drives must hold OPERATION_ENABLED before a
policy may run. Through this window the mode is still INITIALIZING, so
`mode.actuate` is False and the drives receive real zero commands: their internal
loops settle on a zero-velocity target, and a bouncing enable shows up here
instead of underneath a trajectory. Standing's min-jerk clock only starts
afterwards, so the whole ramp is actually delivered.

**`operator_timeout`** — an operator source that stops updating for this long is
treated as gone and the FSM drops to HOLD. Mirrors `RuntimeParams.max_stale_ticks`
on the command buffer.

---

## DYNPARS

`dynamic_params.mat` → `dynpars` (9×1), for the high-level controller (not yet
ported). MATLAB order preserved; the semantic mapping is unconfirmed — label
before use.

---

## ESKFConfig

`estimation/eskf`. Values from `params_new.m`, the set `nm_nn.m` actually loads;
`params.m` has an older set.

Q sigmas follow the baseline convention: each 30-dim Q diagonal entry is
`(sigma * dt)^2`, per-body order `[dtheta, gyro bias, pos (0), vel, accel bias]`.
`sigma_dtheta`/`sigma_dv` drive the attitude and velocity random walks (IMU white
noise); `sigma_gyr`/`sigma_acc` drive the bias random walks.

The white terms and bias walks come from Allan-deviation characterization of each
HWT906 (10 min stationary, worst axis), scaled `sigma = C * sqrt(f_loop)` with
`f_loop = 500 Hz`. **`sigma_dv` is left inflated at 1.0** — the ~0.021 Allan floor
is the stationary sensor VRW only; a swinging leg needs velocity distrusted well
above that. Tune `dv` on the replay harness, not from Allan.

**`dt_min_factor` / `dt_max_factor`** — dead. The caller now passes a dt already
clamped to `COMMON.dt_min`/`dt_max`, so the ESKF no longer clamps. Kept only so
the literal still constructs; delete both once nothing references them.

**`imu_stale_timeout`** — an IMU sample older than this (host clock, same timebase
as `t`) means the USB stream stalled. Zero-order hold would silently integrate
frozen data, so the filter reports itself unhealthy instead. The IMUs stream at
~200 Hz (5 ms), so 50 ms is 10 missed samples.

**`p0_*`** — initial error covariance, per-body diagonal *variances* (baseline
`Delay_P` init; both bodies identical). `p0_pos = 0` because the start pose
defines the nav origin.

**`comp_beta`** — comp-filter pull rate [rad/s]; must exceed worst tilt divided by
`init_duration`.

**`dynamic_process_noise`** — inflate Q's dtheta/dv terms with the measured
integration error (change in corrected acc/gyro between ticks) instead of white
noise alone.

**`correct_thigh_attitude`** — correct thigh *attitude* from the velocity loop
closure (the `+R_th @ hat(u)` Jacobian block the draft dropped). Off = baseline
behaviour, where thigh pitch rides gyro integration only and drifts; on = pitch is
observed during swing.

**`treadmill_speed`** — belt speed [m/s], positive forward: the number on the
treadmill console. 0 overground, where every term below vanishes identically.

It enters in two places with **opposite signs**, which is the whole reason it is
one scalar and not two:

- *ZUPT*: the stance foot is not still in the lab — the belt carries it backward,
  so the pseudo-measurement is `v_foot = -treadmill_speed` in x, not zero. Getting
  this wrong asserts a false velocity once per stride and leaks through P into foot
  tilt and accel bias.
- *published hip/knee x*: `+ treadmill_speed * t`, turning the lab frame (where the
  walker stays put) into the virtual overground frame the NN predictor was trained
  in.

Assumes nav +x points along the belt, i.e. the leg faced forward when the filter
initialized — yaw is unobservable, so nothing enforces that.

---

## LegGeometry

Leg geometry for the ESKF kinematic chain (`estimation/eskf`). Decoded from the
baseline ESKF transforms (`prosthesis_control_NPC/ESKF/T_O*_func_eskf_new.m`)
evaluated at identity attitude, with lengths from `params_new.m`. Body frame
convention (see `eskf/kinematics.py`): +x forward, +y left, +z up.

**`shank_length`** — `params_new.m length_shank`; the baseline's
`trans_knee_ankle = [0;0;-length_shank]` confirms the offset is purely along the
link.

**`knee_to_thigh_imu`** — from `T_OK` at identity, the knee sits at
`[-0.0846, -0.0550, -0.1040]` in the thigh IMU frame, so the IMU is that far the
other way. **Not** a pure offset along the link: the IMU is mounted 5.5 cm off the
sagittal plane and 8.5 cm fore-aft, only 10.4 cm above the knee. The baseline's
alpha (0.67° thigh-IMU pitch mounting offset) is *already baked in* — do not apply
it a second time.

**`foot_imu_to_ankle`** — confirmed zero on our hardware (Jul 2026): the foot IMU
sits on the ankle joint itself. Matches the baseline, which never models this
offset — its loop closure walks thigh → knee → ankle and compares that straight to
the foot position.

**`hip_to_knee`** — `params_new.m length_thigh`, subject specific. The file lists
0.55 to 0.71 for different subjects; 0.66 is "John", 0.71 is "Russell" (the
subject of the Oct-2025 NN trial). Not used by the loop closures; needed for hip
position and by the NN swing path.

**`knee_angle_offset`** — baseline gamma = 6°, added to the knee angle before the
shank rotation. Kept for reference **only** — see `thigh_imu_pitch` for why we do
not apply it as a knee offset. Unused.

**`thigh_imu_pitch`** — fixed pitch of the thigh IMU mount about the body +y
(sagittal) axis. Positive means the sensor's +x (forward) axis tilts **up**
relative to the thigh link, so the sensor→body map is
`rot_y(-thigh_imu_pitch) @ <axis permutation>` (`eskf.py`), and
`knee_to_thigh_imu`, which is expressed in the *sensor* frame, gets the same
`rot_y(-pitch)` applied (`measurements.py`).

Both effects **must stay coupled**: one physical tilt rotates the frame the chain
descends in *and* re-expresses the knee→IMU vector. This is the correct home for
the constant angular offset — putting it in `THETA0_KNEE` instead would get the
attitude sum right but leave the knee position wrong by
`|knee_to_thigh_imu| * pitch` (~14 mm at 6°).

Confirmed Jul 2026 that the thigh IMU sits 6° negative about body +y. This is
*our* mount, established on our hardware — it is not the baseline's gamma (6° at
the knee slot), which stays unused; the numeric match is a coincidence of
magnitude, not the same physical quantity. The literal is currently 7°, was −6.

---

## Attachment

One muscle's attachment at one joint, from the human muscle geometry. The moment
arm falls from `r_max` at `phi_max` to `r_min` at `phi_min`:

```
r(phi)  = r_max * cos(a * (phi - phi_max))
dl(phi) = (rho*r_max/a) * [sin(a*(phi-phi_max)) - sin(a*(phi_ref-phi_max))]
```

with `a = angle_scale = acos(r_min/r_max)/(phi_min - phi_max)`, which is
`angle_scale_*` in `nm_nn.m`.

Note `r` is **not** `d(dl)/d(phi)`: `rho` scales the length change but not the
moment arm. That asymmetry is in the source model deliberately — keep it. The sign
of `a` does not matter (`cos` and `sin(ax)/a` are both even in `a`).

`constant_arm` is the hip-style case: `r = r_max`, `dl` linear in phi.

---

## MuscleParams

**`F_max` is per-kilogram** (N/kg). `nm_nn.m` writes these as e.g. `7400/80`; the
baseline multiplies stance torque by `stance_body_weight` in NM stance Control
(`Gain`/`Gain1`), so keep the normalisation here and scale at the torque.

**`attachments`** — joint name → `Attachment`. Keys must match the joints the
controller passes in (`"knee"`, `"ankle"`); a missing joint means the muscle does
not span it, so its moment arm there is zero.

**`v_max`** is in `l_opt`/s — multiply by `l_opt` for m/s. **`eps_ref`** is the SE
reference strain; the ankle plantarflexors differ (`eref_ankle_pf = 0.09`).

---

## NeuromuscularStanceParams

Source: `prosthesis_control_NPC/nm_nn.m` (section 2) plus the tuned 7-vector in
`ParameterLibrary/nm_gains_new/nm_gains_1.mat`, which overrides the three `F_max`
values:

```
Gains = [fmax_vas, fmax_gas, fmax_sol, st_fgain_vas, st_lgain_vas,
         st_fgain_gas, st_fgain_sol] = [92.5, 16.875, 37.5, 1.2, 10, 1.67, 1.25]
```

Angles are the **human anatomical convention** the muscle tables are written in,
not hardware angles — `neuromuscular_stance` converts at the boundary.

**Shared muscle constants.** `W` is the CE force-length width; `C_RESID` is the
remaining force at ±width, and the CE gain is `log(C_RESID)`; `N` is eccentric
force enhancement; `K_HILL` the force-velocity shape factor; `TAU` the
excitation-contraction coupling; `DT` the muscle integration substep. `EPS_BE`,
the BE reference strain, is literally `W/2` in the baseline block.

**Delays** are a pathway property, not a muscle one — the reflex arcs pick which
one they read at. `LONG` is the distal/ankle group, `MEDIUM` the knee, `SHORT` the
hip.

**Stance reflex arcs** (Muscle Stimulation/Stance Phase), verified against the
block diagram 2026-08-10:

```
stim_vas = prestim_vas + FGAIN_VAS*(F_vas(t-d)/F_max_vas)
                       + clamp(LGAIN_VAS*(l_ce_vas(t-d)/l_opt_vas - LCEOFF_VAS), -1, 0)
stim_gas = prestim_gas + FGAIN_GAS*(F_gas(t-d)/F_max_gas)
stim_sol = prestim_sol + FGAIN_SOL*(F_sol(t-d)/F_max_sol)
```

then all clamped to `[STIM_MIN, STIM_MAX]`.

**VAS reads both its afferents at `SHORT` (5 ms)** in the baseline — not `MEDIUM`,
despite being a knee muscle. GAS and SOL read at `LONG` (20 ms).

`FGAIN_GAS = 1.67` comes from `nm_gains_1.mat`; the `nm_nn.m` literal is 1.6693.
`SOL.F_max = 37.5` likewise; the `nm_nn.m` literal is `2500/80 = 31.25`.

**Late-stance VAS suppression** (Stimulation Mixing Prosthesis). Once the thigh
passes behind `LATE_STANCE_HIP` and GAS force falls off its peak for this stance,
VAS stim is scaled by `(F_gas / max F_gas)^VAS_SUPPRESS_EXP`, clamped to [0,1].
This is the swing-initiation unlock: it lets the knee break into flexion.

> **Needs hip angle**, so it is gated off until the ESKF provides one — see
> `NeuromuscularStance`.

**Knee-only joint stop and damper**, on the *real* leg angle and velocity.
Verified against Muscle Models 2026-08-10:

```
damper = -KD_DAMPER*knee_vel                     if knee_vel < 0
stop   = KP_STOP*(STOP_ANGLE - knee) - KD_STOP*knee_vel
                                                 if knee < STOP_ANGLE
                                                 and STOP_DELAY elapsed
```

The stop is a buckling guard: it pushes the knee back toward flexion when it gets
straighter than `STOP_ANGLE`. Both are per-kilogram like the muscle forces. The
baseline sums them **inside** Muscle Models, so they also pass through the
human→leg Jacobian — arguably wrong for a hardware stop, but we match it so the
replay is comparable.

`STOP_ANGLE` is Muscle Models/Constant1; `STOP_DELAY` is a Delay block with
`NumDelays = 200` at 1 kHz.

**Muscle signs.** VAS knee `sign=+1`: a knee extensor lengthens with flexion. GAS
knee `sign=-1`: a knee flexor shortens with flexion. GAS and SOL ankle `sign=+1`:
plantarflexors lengthen with dorsiflexion.
