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
                control_enabled: bool) -> LowLevelOutput:

        if not self._verified:
            raise NotImplementedError("Low Level Control not verified yet!")

        # edge detection. control_enabled is a hardware fact (the Elmos reaching
        # OPERATION_ENABLED), passed in by the loop from ec.control_enabled() —
        # not a supervisor flag on the state bus.
        rising_edge = control_enabled and not self._prev_control_enabled
        self._prev_control_enabled = control_enabled

        if rising_edge:
            self._reset_sub_blocks()
            self._post_enable_ticks = 0

        # disabled: do NOT run the pipeline, emit zeros to every field
        # (more conservative than Simulink, which holds the last value)
        if not control_enabled:
            return LowLevelOutput()

        # enabled: run the full pipeline
        # knee pipeline
        knee_torque_commanded = self.knee_torque_conditioning.compute(state, desired_torque)
        knee_velocity_desired, knee_motor_torque_ff_desired = self.knee_torque_controller.compute(state, knee_torque_commanded)
        knee_velocity_commanded, knee_motor_torque_ff = self.knee_velocity_saturation.compute(state, knee_velocity_desired, knee_motor_torque_ff_desired)

        # ankle pipeline
        ankle_torque_commanded = self.ankle_torque_conditioning.compute(state, desired_torque)
        ankle_velocity_desired, ankle_motor_torque_ff_desired = self.ankle_torque_controller.compute(state, ankle_torque_commanded)
        ankle_velocity_commanded, ankle_motor_torque_ff = self.ankle_velocity_saturation.compute(state, ankle_velocity_desired, ankle_motor_torque_ff_desired)

        self._post_enable_ticks += 1

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
