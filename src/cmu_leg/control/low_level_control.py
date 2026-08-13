# Low Level Control orchestrator. Owns the per-joint conditioning -> controller
# -> saturation pipelines and is responsible for:
#   - Detecting the control_enabled rising/falling edge.
#   - Resetting every stateful sub-block on the rising edge
#     (matches Simulink "StatesWhenEnabling: reset" on the Low Level Control
#     enable port).
#   - Holding the drive outputs at zero for STARTUP_MUTE_TICKS after enable
#     (matches the 1 s Delay on the drive-output switches in Simulink).
#   - Running the pipeline only while control_enabled is true.
#
# Differs from Simulink on disable behavior (intentionally more conservative):
# Simulink holds the last drive-bound outputs ("OutputWhenDisabled: held") and
# trusts the Elmo CiA 402 state machine to ignore them. This port emits zeros
# on every drive-facing field the moment control_enabled drops. See docs/arch.md.

from cmu_leg.control.low_level.knee import (
    KneeTorqueConditioning,
    KneeTorqueController,
    KneeVelocitySaturation,
)
from cmu_leg.control.low_level.ankle import (
    AnkleTorqueConditioning,
    AnkleTorqueController,
    AnkleVelocitySaturation,
)
from cmu_leg.types.state import ProsthesisState
from cmu_leg.types.commands import TorqueDesired, LowLevelOutput, DriveCommand
import cmu_leg.config.params as params 


class LowLevelControl:
    def __init__(self):
        self.startup_mute_ticks = params.COMMON.startup_mute_ticks

        # knee pipeline: conditioning -> controller -> saturation
        self.knee_torque_conditioning = KneeTorqueConditioning()
        self.knee_torque_controller = KneeTorqueController()
        self.knee_velocity_saturation = KneeVelocitySaturation()

        # ankle pipeline: conditioning -> controller -> saturation
        self.ankle_torque_conditioning = AnkleTorqueConditioning()
        self.ankle_torque_controller = AnkleTorqueController()
        self.ankle_velocity_saturation = AnkleVelocitySaturation()

        # edge detection + post-enable mute counter
        self._prev_active = False
        self._prev_control_enabled = False
        self._post_enable_ticks = 0

        # change this to true once everything is verified as safe to use.
        self._verified = True

        knee = params.KNEE
        ankle = params.ANKLE
        # drive command gains 
         # rad/s -> counts/s   (1 / motor_abs_counts_to_rad)
        self.knee_velocity_gain = 1.0 / knee.motor_abs_counts_to_rad
        self.ankle_velocity_gain = 1.0 / ankle.motor_abs_counts_to_rad

        # Nm -> per-mille of rated torque
        # rated torque at gearbox output = rated_current * torque_constant * gear_ratio
        self.knee_torque_gain = 1000.0 / (knee.rated_current
                                          * knee.torque_constant
                                          * knee.gear_ratio)
        self.ankle_torque_gain = 1000.0 / (ankle.rated_current
                                           * ankle.torque_constant
                                           * ankle.gear_ratio)

    # Reset every stateful sub-block. Called on the rising edge of
    # control_enabled to mirror Simulink "StatesWhenEnabling: reset".
    def _reset_sub_blocks(self) -> None:
        self.knee_torque_conditioning.reset()
        self.knee_torque_controller.reset()
        self.knee_velocity_saturation.reset()
        self.ankle_torque_conditioning.reset()
        self.ankle_torque_controller.reset()
        self.ankle_velocity_saturation.reset()

    def to_drive_command(self, low_level_output: LowLevelOutput) -> DriveCommand:
        return DriveCommand(
                    knee_velocity_counts=int(round(low_level_output.knee_velocity_commanded * self.knee_velocity_gain)),
                    knee_torque_per_mille=int(round(low_level_output.knee_motor_torque_ff * self.knee_torque_gain)),
                    ankle_velocity_counts=int(round(low_level_output.ankle_velocity_commanded * self.ankle_velocity_gain)),
                    ankle_torque_per_mille=int(round(low_level_output.ankle_motor_torque_ff * self.ankle_torque_gain)),
                )

    def compute(self, state: ProsthesisState, desired_torque: TorqueDesired,
                control_enabled: bool, dt: float,
                actuate: bool = True) -> LowLevelOutput:

        if not self._verified:
            raise NotImplementedError("Low Level Control not verified yet!")

        # edge detection. control_enabled is a hardware fact (the Elmos reaching
        # OPERATION_ENABLED), passed in by the loop from ec.control_enabled() —
        # not a supervisor flag on the state bus.
        # `actuate` joins it: a mode that is not driving must not run the
        # pipeline either. Otherwise the conditioning lowpass keeps integrating
        # a joint-stop torque whose output is thrown away, and hands over a
        # wound-up value the instant actuation begins. Concretely: with the leg
        # straight the knee extension stop demands ~14 Nm (knee_angle_min_soft
        # is 5 deg, so 0 deg is past it), and `not standing` makes that a hard
        # floor -- across the 3 s between drive-enable and standing the filter
        # converges there, then steps into the joint. Gating the pipeline on
        # `active` removes the windup AND re-arms the startup mute, so control
        # engages from a settled state.
        active = control_enabled and actuate
        active_edge = active and not self._prev_active
        enable_edge = control_enabled and not self._prev_control_enabled
        self._prev_active = active
        self._prev_control_enabled = control_enabled

        # RESET on the actuate edge -- that is what clears the wound-up
        # conditioning lowpass, so control always engages from zero.
        if active_edge:
            self._reset_sub_blocks()
        # MUTE only on the hardware enable edge, NOT on the actuate edge. The
        # mute zeroes drive-facing output for a second; re-arming it when a
        # policy starts would swallow the first half of standing's 2 s min-jerk
        # ramp and the leg would snap to setpoint when it lifted. The dwell in
        # ModeParams.enable_dwell is what covers the post-enable settle instead.
        if enable_edge:
            self._post_enable_ticks = 0
        # Counts ticks since the HARDWARE enable, not ticks of a running
        # pipeline -- so it advances through the INITIALIZING dwell too and is
        # already spent by the time a policy starts. Incremented before the
        # early return below for exactly that reason: leaving it after meant the
        # mute was still fully in force at the first actuating tick and ate the
        # front of standing's ramp.
        self._post_enable_ticks += 1

        # not driving: do NOT run the pipeline, emit zeros to every field
        # (more conservative than Simulink, which holds the last value). Zeros
        # mean target-velocity-zero at the drives, i.e. HOLD -- see Mode.
        if not active:
            return LowLevelOutput()

        # enabled: run the full pipeline
        # knee pipeline
        knee_torque_commanded = self.knee_torque_conditioning.compute(state, desired_torque)
        knee_velocity_desired, knee_motor_torque_ff_desired = self.knee_torque_controller.compute(state, knee_torque_commanded, dt)
        knee_velocity_commanded, knee_motor_torque_ff = self.knee_velocity_saturation.compute(state, knee_velocity_desired, knee_motor_torque_ff_desired)

        # ankle pipeline
        ankle_torque_commanded = self.ankle_torque_conditioning.compute(state, desired_torque)
        ankle_velocity_desired, ankle_motor_torque_ff_desired = self.ankle_torque_controller.compute(state, ankle_torque_commanded, dt)
        ankle_velocity_commanded, ankle_motor_torque_ff = self.ankle_velocity_saturation.compute(state, ankle_velocity_desired, ankle_motor_torque_ff_desired)

        # startup mute window: pipeline state advances (filters settle, BDs
        # build history) but drive-facing fields are forced to zero. Logging
        # fields (commanded torque, desired velocity) carry the live values
        # so the user can see what the controller would have asked for.
        if self._post_enable_ticks <= self.startup_mute_ticks:
            return LowLevelOutput(
                knee_torque_commanded=knee_torque_commanded,
                knee_velocity_desired=knee_velocity_desired,
                knee_velocity_commanded=0.0,
                knee_motor_torque_ff=0.0,
                ankle_torque_commanded=ankle_torque_commanded,
                ankle_velocity_desired=ankle_velocity_desired,
                ankle_velocity_commanded=0.0,
                ankle_motor_torque_ff=0.0,
            )

        return LowLevelOutput(
            knee_torque_commanded=knee_torque_commanded,
            knee_velocity_desired=knee_velocity_desired,
            knee_velocity_commanded=knee_velocity_commanded,
            knee_motor_torque_ff=knee_motor_torque_ff,
            ankle_torque_commanded=ankle_torque_commanded,
            ankle_velocity_desired=ankle_velocity_desired,
            ankle_velocity_commanded=ankle_velocity_commanded,
            ankle_motor_torque_ff=ankle_motor_torque_ff,
        )
