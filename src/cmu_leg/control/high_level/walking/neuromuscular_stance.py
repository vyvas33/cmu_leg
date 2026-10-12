"""
Neuromuscular Stance Control
"""

from cmu_leg.control.high_level.base_controller import TorqueController
from cmu_leg.control.utils import clamp
from cmu_leg.control.high_level.walking import retargeting
import cmu_leg.config.params as params
from cmu_leg.config.params import (CommonParams, NeuromuscularStanceParams,
                                   MuscleParams, NM_STANCE, LEG)
from cmu_leg.structures.state import ProsthesisState
from cmu_leg.structures.commands import TorqueDesired
import collections
import math
from typing import NamedTuple

import numpy as np



# Leg -> human retargeting.
#
# The muscle tables are human: their moment arms and reference angles assume a
# leg with roughly equal segments. Ours is 0.66 thigh / 0.215 shank, so the
# baseline runs the muscles on a VIRTUAL leg with both segments equal to
# (thigh+shank)/2 and maps back. Two quantities are held equal, and they are
# the two the stance model is a function of:
#
#   1. |hip -> ankle|             the leg-spring length (what VAS resists)
#   2. t2 - ankle_angle           how the foot sits relative to the leg axis
#                                 (what SOL/GAS act on), t2 = shank-to-chord
#
# Everything else differs -- the virtual knee is ~22 cm from the real one and
# the foot is not in the same place. That is fine: a muscle only knows its own
# length, which is a function of joint angles alone.
#
# Both legs use the same joint convention (knee 0 = straight, flexion +;
# ankle 0 = neutral, dorsiflexion +), so there is NO unit or offset conversion
# here -- only the kinematic remap. Ankle sign confirmed from the walking log:
# the ankle reads positive through midstance (tibia over the foot) and drives
# negative at toe-off (push-off plantarflexion).
# The knee half of the remap is shared with the NN swing predictor, which needs
# the same virtual leg for the same reason -- see walking/retargeting.py.
_L_THIGH, _L_SHANK, _L_HUMAN = (retargeting.L_THIGH, retargeting.L_SHANK,
                                retargeting.L_HUMAN)

# Torque map. The baseline applies M = d(human)/d(leg) directly to the torque
# vector; power consistency wants M^T. They differ in which joint the cross
# term lands on, and the baseline's choice is not power conserving (~4.6% at a
# representative pose). Default matches the baseline so its tuned reflex gains
# mean what they meant; flip to use the physically consistent map.
_TORQUE_MAP_TRANSPOSE = False


class Afferents(NamedTuple):
    """Delayed feedback signals available to the reflex law this substep."""
    f_vas: float
    f_gas: float
    f_sol: float
    lce_vas: float


class NeuromuscularStance(TorqueController):
    _verified = True

    def __init__(self, nm = NM_STANCE, common: CommonParams = params.COMMON):
        self._nm = nm
        self._dt = nm.DT

        # muscle states integrate at nm.DT; _compute is called at the control
        # rate, so each call runs a fixed number of substeps. Fixed, not
        # measured-dt: the delay buffers encode their delay as a slot count,
        # so a varying dt would silently change every reflex delay.
        self._control_rate = common.control_sample_rate
        self._n_substeps = round((1.0/common.control_sample_rate) / nm.DT)
        # seconds of stance elapsed; arms the knee stop after STOP_DELAY
        self._t_since_reset = 0.0
        self._body_weight = common.stance_body_weight
        # late-stance VAS suppression state, reset each stance
        self._hip_angle = 0.0
        self._f_gas_prev = 0.0
        self._f_gas_max = 0.0

        self.VAS = Muscle(nm.VAS)
        self.GAS = Muscle(nm.GAS)
        self.SOL = Muscle(nm.SOL)            

        # create muscles list 
        self.muscles = [self.VAS, self.GAS, self.SOL]   

        for muscle in self.muscles:
            muscle.reset()

        # Delays come from the reflex arcs, not the muscles. Both VAS
        # afferents share DELAY_VAS -- same nerve, same conduction time.
        self._F_VAS_buf   = DelayBuffer(delay_t=nm.DELAY_VAS, dt=nm.DT)
        self._F_GAS_buf   = DelayBuffer(delay_t=nm.DELAY_GAS, dt=nm.DT)
        self._F_SOL_buf   = DelayBuffer(delay_t=nm.DELAY_SOL, dt=nm.DT)
        self._lce_VAS_buf = DelayBuffer(delay_t=nm.DELAY_VAS, dt=nm.DT)

        self._buffers = (self._F_VAS_buf, self._F_GAS_buf,
                         self._F_SOL_buf, self._lce_VAS_buf)

    def _pull_afferents(self) -> Afferents:
        """Every delayed signal for this substep, read as one group.

        Must be called before _push_afferents in the same substep -- the
        buffers encode their delay in their depth, so a push first makes
        every read one substep short.
        """
        return Afferents(
            f_vas   = self._F_VAS_buf.get_delayed(),
            f_gas   = self._F_GAS_buf.get_delayed(),
            f_sol   = self._F_SOL_buf.get_delayed(),
            lce_vas = self._lce_VAS_buf.get_delayed(),
        )

    def _push_afferents(self, f_vas, f_gas, f_sol, lce_vas) -> None:
        """Advance every buffer by one substep. All four, unconditionally.

        Values must all be from the same instant t: forces computed from the
        current l_ce, and lce_vas read before the l_ce integration step.
        Skipping any one buffer desyncs that channel's delay permanently.
        """
        self._F_VAS_buf.push(f_vas)
        self._F_GAS_buf.push(f_gas)
        self._F_SOL_buf.push(f_sol)
        self._lce_VAS_buf.push(lce_vas)

    def _compute(self, state, time):
        # Joint angles are held constant across the substeps (the control loop
        # samples them once), so geometry is resolved once per call.
        self._hip_angle = state.hip_angle
        phi = self._to_human_angles(state.knee_angle, state.ankle_angle)
        for muscle in self.muscles:
            muscle.l_m = self._get_muscle_length(muscle, phi)

        f = {}
        for _ in range(self._n_substeps):
            aff = self._pull_afferents()

            # captured at t, before any muscle integrates its l_ce below
            lce_vas = self.VAS.l_ce

            for m in self.muscles:
                stim = self._reflex_stim(m, aff)
                m.act += m.get_activation_derivative(m.act, stim) * self._dt

                # f_m is evaluated at the pre-integration l_ce, so it is the
                # force at t and is safe to push after l_ce advances
                f_m, v_ce = m.get_muscle_dynamics(m.l_ce, m.act, m.l_m)
                f[m.name] = f_m
                m.l_ce += v_ce * self._dt

            self._f_gas_prev = f["GAS"]
            self._f_gas_max = max(self._f_gas_max, f["GAS"])
            self._push_afferents(f["VAS"], f["GAS"], f["SOL"], lce_vas)

        # f now holds the last substep's forces -- the ones the joints see
        self._t_since_reset += 1.0/self._control_rate
        return self._get_joint_torques(f, state)

    def _reflex_stim(self, muscle, aff: Afferents) -> float:
        """Stimulation for one muscle from the delayed afferents.

        The reflex topology -- which muscle reads whose delayed signal, with
        what gain. Lives here rather than on Muscle so the plant stays
        independent of the reflex set. Verified block-by-block against
        Muscle Stimulation/Stance Phase, 2026-08-10:

            stim_vas = prestim + FGAIN_VAS*(F_vas/F_max_vas)
                               + clamp(LGAIN_VAS*(l_ce_vas/l_opt_vas - LCEOFF_VAS), -1, 0)
            stim_gas = prestim + FGAIN_GAS*(F_gas/F_max_gas)
            stim_sol = prestim + FGAIN_SOL*(F_sol/F_max_sol)

        Force feedback is on NORMALISED force (the baseline divides by F_max,
        so the per-kg F_max cancels and this is body-weight independent). The
        VAS length term is inhibitory only: the gain is applied BEFORE the
        [-1, 0] clamp, so it can subtract at most 1.0 stim unit however short
        the fibre gets. That is the knee-overextension inhibition, expressed
        through VAS fibre length rather than knee angle.
        """
        nm = self._nm
        if muscle is self.VAS:
            stim = muscle.stim0 + nm.FGAIN_VAS * (aff.f_vas / muscle.F_max)
            stim += clamp(nm.LGAIN_VAS * (aff.lce_vas / muscle.l_opt - nm.LCEOFF_VAS),
                          -1.0, 0.0)
            stim *= self._vas_late_stance_scale()
        elif muscle is self.GAS:
            stim = muscle.stim0 + nm.FGAIN_GAS * (aff.f_gas / muscle.F_max)
        elif muscle is self.SOL:
            stim = muscle.stim0 + nm.FGAIN_SOL * (aff.f_sol / muscle.F_max)
        else:
            raise KeyError(f"no reflex arc defined for muscle {muscle.name!r}")

        return clamp(stim, nm.STIM_MIN, nm.STIM_MAX)

    def _vas_late_stance_scale(self) -> float:
        """Late-stance VAS suppression: the swing-initiation unlock.

        Once the thigh is behind LATE_STANCE_HIP and GAS force has fallen from
        its peak for this stance, VAS is scaled by (F_gas/F_gas_max)^3 -- a
        sharp rolloff that lets the knee break into flexion for swing. Returns
        1.0 (no suppression) outside late stance.

        Uses the previous substep's GAS force, because VAS is evaluated before
        GAS within a substep; at 0.2 ms the lag is immaterial.

        Requires hip angle. Without it (hip_angle left at its default) the gate
        never fires and VAS stays unsuppressed -- which costs ~0.09 on VAS
        activation and ~4 Nm RMS on knee torque against the baseline.
        """
        nm = self._nm
        if self._hip_angle >= nm.LATE_STANCE_HIP or self._f_gas_max <= 0.0:
            return 1.0
        ratio = self._f_gas_prev / self._f_gas_max
        return clamp(ratio ** nm.VAS_SUPPRESS_EXP, 0.0, 1.0)

    @staticmethod
    def _moment_arm(a, phi_j) -> float:
        """r_{m,j}(phi) for a single attachment, m."""
        if a.constant_arm:
            return a.r_max
        return a.r_max * math.cos(a.angle_scale * (phi_j - a.phi_max))

    @staticmethod
    def _delta_l(a, phi_j) -> float:
        """Per-joint MTU length deviation from rest, signed per the muscle's action."""
        if a.constant_arm:
            return a.sign * a.rho * a.r_max * (phi_j - a.phi_ref)
        k = a.angle_scale
        return a.sign * a.rho * a.r_max / k * (
            math.sin(k * (phi_j - a.phi_max)) - math.sin(k * (a.phi_ref - a.phi_max))
        )

    @staticmethod
    def _to_human_angles(knee_angle, ankle_angle) -> dict:
        """Leg joint angles -> the virtual human leg's joint angles.

        Holds |hip->ankle| and (t2 - ankle) equal; see the module header.
        `t1`/`t2` are the triangle's interior angles at hip/ankle, t1+t2 = knee.
        """
        ls = _L_SHANK
        if knee_angle == 0.0:
            return {"knee": 0.0, "ankle": ankle_angle}

        l3 = retargeting.chord(knee_angle)      # hip->ankle on the real leg
        knee_h = retargeting.knee_leg_to_human(knee_angle)
        # t1 sits at the HIP, so the side opposite it is the SHANK
        t1 = math.asin(clamp(ls*math.sin(knee_angle)/l3, -1.0, 1.0))
        # t2 - ankle preserved; virtual leg is isoceles so t2_h = knee_h/2
        return {"knee": knee_h,
                "ankle": ankle_angle + 0.5*knee_h - knee_angle + t1}

    @staticmethod
    def _human_leg_jacobian(knee_angle, ankle_angle):
        """M = d(human angles)/d(leg angles), as (m11, m12, m21, m22).

        Lower triangular: the human ankle depends on both leg joints, the
        human knee only on the leg knee. Analytic -- the baseline builds the
        same matrix as pinv(J_human) @ J_leg, but that product is identically
        this derivative, so the two Jacobians and the pseudo-inverse are not
        needed (and neither is their singularity at a straight knee).
        """
        lt, ls, lh = _L_THIGH, _L_SHANK, _L_HUMAN
        if knee_angle == 0.0:
            return 1.0, 0.0, 0.0, 1.0

        l3 = math.sqrt(lt*lt + ls*ls + 2.0*lt*ls*math.cos(knee_angle))
        dl3 = -lt*ls*math.sin(knee_angle) / l3

        ck = (2.0*lh*lh - l3*l3) / (2.0*lh*lh)
        # d(knee_h)/d(knee) = -d(acos(ck))/d(knee) = (dck/dl3 * dl3) / sqrt(1-ck^2)
        dkh = (-l3/(lh*lh) * dl3) / math.sqrt(max(1.0 - ck*ck, 1e-12))

        u = ls*math.sin(knee_angle)/l3
        du = ls*(math.cos(knee_angle)*l3 - math.sin(knee_angle)*dl3) / (l3*l3)
        dt1 = du / math.sqrt(max(1.0 - u*u, 1e-12))

        # ankle_h = ankle + knee_h/2 - knee + t1
        return dkh, 0.0, 0.5*dkh - 1.0 + dt1, 1.0

    def _get_moment_arm(self, muscle, phi):
        """(r_knee, r_ankle) as human-frame torque coefficients, at human angles.

        A muscle pulls to shorten, so it torques in the direction that
        decreases its own length: tau_j = -sign * r_j * F. Zero on joints the
        muscle does not span. `rho` is deliberately absent -- it scales length
        change but not the moment arm.
        """
        arms = []
        for j in ("knee", "ankle"):
            a = muscle.attachments.get(j)
            arms.append(0.0 if a is None
                        else -a.sign * self._moment_arm(a, phi[j]))
        return tuple(arms)

    def _get_muscle_length(self, muscle, phi):
        """MTU length for one muscle, m: rest length plus each joint's deviation.

        `phi` is the HUMAN joint angles, as returned by _to_human_angles.
        """
        d_total = 0.0
        for joint, a in muscle.attachments.items():
            if joint not in phi:
                raise KeyError(f"no angle for joint {joint!r} of muscle {muscle.name!r}")
            d_total += self._delta_l(a, phi[joint])
        return muscle.l_rest + d_total

    def _stop_and_damper(self, knee_angle, knee_velocity) -> float:
        """Knee-only stop + damper torque, per kilogram, on the REAL leg state.

        A buckling guard, not an end-stop: when the knee is straighter than
        STOP_ANGLE it is pushed back toward flexion, with its own damping. Armed
        only after STOP_DELAY of stance has elapsed, so it stays out of the way
        through loading response. The plain damper is separate and acts only
        while the knee is extending.
        """
        nm = self._nm
        tau = 0.0
        if knee_velocity < 0.0:
            tau -= nm.KD_DAMPER * knee_velocity
        armed = self._t_since_reset >= nm.STOP_DELAY
        if armed and knee_angle < nm.STOP_ANGLE:
            tau += nm.KP_STOP * (nm.STOP_ANGLE - knee_angle)
            tau -= nm.KD_STOP * knee_velocity
        return tau

    def _get_joint_torques(self, f, state) -> TorqueDesired:
        """Muscle forces -> leg joint torques.

        Moment arms in the HUMAN frame give human joint torques, the
        retargeting Jacobian maps those onto the leg, and body weight turns the
        per-kilogram result into Nm. The stop/damper is added on the human side
        because that is where the baseline sums it -- so it goes through the
        Jacobian too. See the note in NeuromuscularStanceParams.
        """
        knee_angle, ankle_angle = state.knee_angle, state.ankle_angle
        phi = self._to_human_angles(knee_angle, ankle_angle)
        tau_knee_h = self._stop_and_damper(knee_angle, state.knee_velocity)
        tau_ankle_h = 0.0
        for m in self.muscles:
            r_knee, r_ankle = self._get_moment_arm(m, phi)
            tau_knee_h  += r_knee  * f[m.name]
            tau_ankle_h += r_ankle * f[m.name]

        m11, m12, m21, m22 = self._human_leg_jacobian(knee_angle, ankle_angle)
        if _TORQUE_MAP_TRANSPOSE:
            knee  = m11*tau_knee_h + m21*tau_ankle_h
            ankle = m12*tau_knee_h + m22*tau_ankle_h
        else:
            knee  = m11*tau_knee_h + m12*tau_ankle_h
            ankle = m21*tau_knee_h + m22*tau_ankle_h

        return TorqueDesired(knee * self._body_weight, ankle * self._body_weight)

    def reset(self, state, time):
        self._t_since_reset = 0.0
        self._hip_angle = state.hip_angle
        self._f_gas_prev = 0.0
        self._f_gas_max = 0.0
        phi = self._to_human_angles(state.knee_angle, state.ankle_angle)
        for m in self.muscles:
            m.l_m = self._get_muscle_length(m, phi)
            m.reset()
            m.settle(m.l_m)

        # Prefill every slot with the settled value, so the first stance
        # substep reads a real afferent instead of raising IndexError (and so
        # the reflex does not ramp up over the first delay window).
        f0 = {m.name: m.get_muscle_force(m.l_ce, m.act, m.l_m) for m in self.muscles}
        self._F_VAS_buf.fill(f0["VAS"])
        self._F_GAS_buf.fill(f0["GAS"])
        self._F_SOL_buf.fill(f0["SOL"])
        self._lce_VAS_buf.fill(self.VAS.l_ce)


class Muscle:
    def __init__(self, params: MuscleParams, nm = NM_STANCE):
        self.name = params.name
        self.F_max      = params.F_max
        self.l_opt      = params.l_opt
        self.l_slack    = params.l_slack
        # nm_nn.m stores v_max in l_opt/s; the baseline's `vmax*lopt` gain
        # block converts it. Do the same once here so v_ce comes out in m/s.
        self.v_max      = params.v_max * params.l_opt
        self.eps_ref    = params.eps_ref     # ankle plantarflexors use eref_ankle_pf
        self.stim0      = params.prestim
        self.attachments = params.attachments
        # The MTU sits at l_opt + l_slack when every joint it spans is at
        # that attachment's phi_ref.
        self.l_rest     = params.l_opt + params.l_slack

        self.w          = nm.W
        self.N          = nm.N
        self.K          = nm.K_HILL
        self.tau        = nm.TAU
        self.eps_be     = nm.EPS_BE
        self.c_resid    = nm.C_RESID

        self.l_ce       = self.l_opt
        self.l_m = None
        self.act        = self.stim0

        
    def force_length_relationship_CE(self, l_ce):
        w = self.w
        l_opt = self.l_opt
        c = math.log(self.c_resid) 

        f_l = math.exp(c*abs((l_ce - l_opt) / (l_opt*w))**3)     
        return f_l

    # def force_velocity_relationship_CE(self, v_ce):
    #     v_max = self.v_max
    #     K = self.K
    #     N = self.N

    #     if v_ce <= 0:
    #         f_v = (v_max + v_ce) / (v_max - K * v_ce)   

    #     else:
    #         f_v = N  + (N-1)*(( - v_max + v_ce)/(7.56*K*v_ce + v_max))

    #     return f_v
    
    def force_length_relationship_SEE(self, l_see):
        l_slack = self.l_slack
        eps = (l_see - l_slack)/l_slack

        eps_ref = self.eps_ref 

        if eps > 0:
            f_see = (eps/eps_ref)**2
        else:
            f_see = 0

        return f_see
    
    #this corresponds to F_pe* 
    def force_length_relationship_PE(self, l_ce):
        l_opt = self.l_opt

        if l_ce > l_opt:
            f_pe = ((l_ce/l_opt - 1) / self.w)**2
        else:
            f_pe = 0
        return f_pe

    def force_length_relationship_be(self, l_ce):
        l_opt = self.l_opt
        w = self.w
        eps_be = self.eps_be
        l_min = l_opt - w*l_opt

        if l_ce < l_min:
            f_be = ((l_min - l_ce)/(eps_be*l_opt))**2
        else:
            f_be = 0
        return f_be
    
    def inverse_force_velocity_CE(self, f_v):
        v_max = self.v_max
        K = self.K
        N = self.N

        if f_v <= 1:
            v_ce = v_max * (f_v - 1) / (1 + K * f_v)    
        elif (f_v > 1)   and (f_v <= N):
            v_ce = (((f_v - 1) * v_max) / (N - 1- 7.56*K*(f_v - N)))
        else:
            v_ce = v_max*(1 + 0.01*(f_v - N))
        return v_ce
    
    def get_activation_derivative(self, act, stim):
        return (stim - act) / self.tau
    
    def get_muscle_dynamics(self, l_ce, act, l_m):        
        """
        Computes the muscle force based on current muscle state and activation.
        This function calculates derivatives without mutating state.

        f_m = f_see_norm * self.F_max_iso
        f_ce = f_see_norm - f_pe_norm

        we calculate v_ce here and return f_see and v_ce. f_see is the muscle force (or basically f_m).
        """
        l_see = l_m - l_ce
        f_see_norm = self.force_length_relationship_SEE(l_see)
        f_pe_norm = self.force_length_relationship_PE(l_ce)

        f_l_ce_norm = self.force_length_relationship_CE(l_ce)
        f_be_norm = self.force_length_relationship_be(l_ce)

        f_ce = (f_see_norm + f_be_norm)

        if f_ce < 0:
            f_ce = 0

        denom = act * f_l_ce_norm + f_pe_norm
        if denom < 1e-6:
            f_v = 1 
        else:
            f_v = f_ce / denom 
        v_ce = self.inverse_force_velocity_CE(f_v)
        f_m = f_see_norm * self.F_max

        return f_m, v_ce 

    def get_muscle_force(self, l_ce, act, l_m):
        f, _ = self.get_muscle_dynamics(l_ce, act, l_m)
        return f

    def reset(self):
        self.l_ce       = self.l_opt
        self.act        = self.stim0

    def settle(self, l_m):
        """Seed l_ce at the length where v_ce = 0 for this l_m and activation.

        Starting from l_opt leaves the tendon off its equilibrium, and l_ce is a
        slow integrator with nothing to pull it back, so that offset persists
        through the whole stance. Solving it instead takes VAS's l_ce from 0.53
        to 0.99 correlation against the baseline, and the knee torque from 0.974
        to 0.985, without touching GAS or SOL.

        Bisection on f_see + f_be - (act*f_l + f_pe), which is monotone in l_ce:
        lengthening the fibre shortens the tendon, so the left side falls and
        the right side rises. Falls back to l_opt if the bracket has no root
        (an unloaded MTU shorter than slack).
        """
        lo, hi = 1e-4, l_m - 1e-4
        if hi <= lo:
            self.l_ce = self.l_opt
            return

        def residual(l_ce):
            return ((self.force_length_relationship_SEE(l_m - l_ce)
                     + self.force_length_relationship_be(l_ce))
                    - (self.act * self.force_length_relationship_CE(l_ce)
                       + self.force_length_relationship_PE(l_ce)))

        if residual(lo) * residual(hi) > 0.0:
            self.l_ce = self.l_opt
            return
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if residual(lo) * residual(mid) <= 0.0:
                hi = mid
            else:
                lo = mid
        self.l_ce = 0.5 * (lo + hi)


class DelayBuffer:
    """Fixed-depth afferent delay line, ticked once per muscle substep.

    Depth encodes the nominal delay: with maxlen = D, the buffer holds
    x[t-D] ... x[t-1], so get_delayed() returns the sample from D substeps
    ago. That only holds if the caller reads BEFORE pushing -- push first
    and every read is one substep short.
    """

    def __init__(self, delay_t: float, dt: float):
        self.delay_t = delay_t
        self.dt = dt
        self.delay_steps = round(delay_t/dt)
        if self.delay_steps < 1:
            raise ValueError(f"delay {delay_t} s is under one substep of {dt} s")
        self._buffer = collections.deque(maxlen=self.delay_steps)

    def push(self, value):
        self._buffer.append(value)

    def get_delayed(self, delay_t: float = None):
        """Sample from delay_t seconds ago; the full buffer depth if omitted."""
        if delay_t is None:
            steps = self.delay_steps
        else:
            steps = round(delay_t/self.dt)
            if not 1 <= steps <= self.delay_steps:
                raise ValueError(
                    f"delay {delay_t} s is {steps} substeps, buffer holds {self.delay_steps}"
                )
        return self._buffer[-steps]

    def fill(self, value):
        """Prefill every slot; call on reset so the first read is not an IndexError."""
        self._buffer.extend([value] * self.delay_steps)
