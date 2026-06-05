# Drive Output. Converts the Low Level Control output (SI units: rad/s, Nm)
# into the integer values the Elmo drive PDOs expect (encoder counts/s and
# per-mille of rated torque). Mirrors the Simulink blocks "To Knee Elmo Drive"
# and "To Ankle Elmo Drive".
#
# Conversion formulas (from params.m, baseline_impedance_min_jerk):
#   velocity_counts   = velocity_rad_s / motor_abs_counts_to_rad
#   torque_per_mille  = torque_Nm * 1000 / (rated_current * torque_constant * gear_ratio)
#
# Pure stateless math — no filters, no derivatives, no enable gating. The
# orchestrator (and ultimately the hardware layer) is responsible for deciding
# whether the resulting DriveCommand actually gets written to the bus.

from cmu_leg.types.commands import LowLevelOutput, DriveCommand
import cmu_leg.config.params as params


class DriveOutput:
    def __init__(self):
        # rad/s -> counts/s   (1 / motor_abs_counts_to_rad)
        self.knee_velocity_gain = 1.0 / params.KNEE_MOTOR_ABS_COUNTS_TO_RAD
        self.ankle_velocity_gain = 1.0 / params.ANKLE_MOTOR_ABS_COUNTS_TO_RAD

        # Nm -> per-mille of rated torque
        # rated torque at gearbox output = rated_current * torque_constant * gear_ratio
        self.knee_torque_gain = 1000.0 / (params.KNEE_RATED_CURRENT
                                          * params.KNEE_TORQUE_CONSTANT
                                          * params.KNEE_GEAR_RATIO)
        self.ankle_torque_gain = 1000.0 / (params.ANKLE_RATED_CURRENT
                                           * params.ANKLE_TORQUE_CONSTANT
                                           * params.ANKLE_GEAR_RATIO)

        # change this to true once everything is verified as safe to use.
        self._verified = False

    def compute(self, low_level_output: LowLevelOutput) -> DriveCommand:
        if not self._verified:
            raise NotImplementedError("Drive Output not verified yet!")

        return DriveCommand(
            knee_velocity_counts=int(round(low_level_output.knee_velocity_commanded * self.knee_velocity_gain)),
            knee_torque_per_mille=int(round(low_level_output.knee_motor_torque_ff * self.knee_torque_gain)),
            ankle_velocity_counts=int(round(low_level_output.ankle_velocity_commanded * self.ankle_velocity_gain)),
            ankle_torque_per_mille=int(round(low_level_output.ankle_motor_torque_ff * self.ankle_torque_gain)),
        )
