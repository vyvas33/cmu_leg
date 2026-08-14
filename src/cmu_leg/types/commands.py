from dataclasses import dataclass
from enum import IntEnum


class Mode(IntEnum):
    """Which high-level policy is live. Crosses to the low level on the command
    channel, which is why it lives here and not in the controller.

    It replaces the old `standing` bool, which had come to mean two things --
    "run the standing policy" AND "actuation is permitted". Those come apart
    once walking exists: WALKING must actuate without being standing, and HOLD
    must not actuate without being standing either. The low level derives both
    from this:
        actuate                 = mode in (STANDING, WALKING)
        suppress extension stop = mode is STANDING
    IntEnum so it survives the float-array shared buffer intact.

    HOLD is named for what the hardware actually does, not what we wish it did.
    The drives run Cyclic Synchronous Velocity (`_pack_drive` fixes mode-of-
    operation byte 20 to 9), so an all-zero DriveCommand is target velocity
    zero -- the drive closes its own velocity loop and actively resists motion.
    That is a HOLD, not a free joint. Calling it ZERO_TORQUE would be a lie:
    getting a genuinely free leg needs CST (mode 10) with zero target torque,
    or dropping out of OPERATION_ENABLED. Hold is also the safer failure for a
    worn leg -- a free knee collapses under someone standing on it -- which is
    why the stale-command watchdog falls here too.
    """
    INITIALIZING = 0
    STANDING = 1
    WALKING = 2
    HOLD = 3

    @property
    def actuate(self) -> bool:
        return self in (Mode.STANDING, Mode.WALKING)

#the output of the low level controller 
@dataclass
class LowLevelOutput:
    # knee
    knee_torque_commanded: float = 0.0
    knee_velocity_desired: float = 0.0
    knee_velocity_commanded: float = 0.0
    knee_motor_torque_ff: float = 0.0
    # ankle
    ankle_torque_commanded: float = 0.0
    ankle_velocity_desired: float = 0.0
    ankle_velocity_commanded: float = 0.0
    ankle_motor_torque_ff: float = 0.0

#the command that goes to the motors. Hardware-ready values in the units the
#Elmo PDOs natively consume:
#  velocity_counts:   encoder counts per second (Target Velocity, 0x60FF, int32)
#  torque_per_mille:  per-mille of rated torque (Target Torque,   0x6071, int16)
@dataclass
class DriveCommand:
    # knee
    knee_velocity_counts: int = 0
    knee_torque_per_mille: int = 0
    # ankle
    ankle_velocity_counts: int = 0
    ankle_torque_per_mille: int = 0

@dataclass
class PositionCommand:
    # knee
    knee_position_counts: int
    # ankle
    ankle_position_counts: int

#the output of the high level controller
@dataclass
class TorqueDesired:
    knee_desired: float = 0.0
    ankle_desired: float = 0.0
