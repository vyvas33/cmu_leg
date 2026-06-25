"""
The ELMO CiA 402 state machine script.

Please read README.md 

vkalva, Jun 2026
"""
from enum import Enum, auto

_LIMIT_ACTIVE_BIT = 1 << 11  # status-word bit 11: internal limit active

# CiA-402 control-word commands.
CW_DISABLE_VOLTAGE = 0x00
CW_SHUTDOWN = 0x06          # -> Switch On Disabled / Ready to Switch On
CW_SWITCH_ON = 0x07         # -> Switched On
CW_ENABLE_OPERATION = 0x0F  # -> Operation Enabled
CW_FAULT_RESET = 0xC0       # clears a fault

class DriveState(Enum):
    NOT_READY_TO_SWITCH_ON = auto()
    SWITCH_ON_DISABLED = auto()
    READY_TO_SWITCH_ON = auto()
    SWITCHED_ON = auto()
    OPERATION_ENABLED = auto()
    QUICK_STOP_ACTIVE = auto()
    FAULT_REACTION_ACTIVE = auto()
    FAULT = auto()
    UNKNOWN = auto()

# CiA-402 status-word fingerprints, (mask, value) -> state
_STATE_FINGERPRINTS = [
    (0x4F, 0x00, DriveState.NOT_READY_TO_SWITCH_ON),
    (0x4F, 0x40, DriveState.SWITCH_ON_DISABLED),
    (0x6F, 0x21, DriveState.READY_TO_SWITCH_ON),
    (0x6F, 0x23, DriveState.SWITCHED_ON),
    (0x6F, 0x27, DriveState.OPERATION_ENABLED),
    (0x6F, 0x07, DriveState.QUICK_STOP_ACTIVE),
    (0x4F, 0x0F, DriveState.FAULT_REACTION_ACTIVE),
    (0x4F, 0x08, DriveState.FAULT),
]

def drive_state(status_word: int) -> DriveState:
    """Decode the CiA-402 state the drive is in, from its status word."""
    for mask, value, state in _STATE_FINGERPRINTS:
        if (status_word & mask) == value:
            return state
    return DriveState.UNKNOWN

# Determine state (from status word), transition to next state 
# with the goal of reaching Operation Enabled
_CONTROL_WORD = {
    DriveState.NOT_READY_TO_SWITCH_ON: CW_DISABLE_VOLTAGE,
    DriveState.SWITCH_ON_DISABLED:     CW_SHUTDOWN,         # -> Ready to Switch On
    DriveState.READY_TO_SWITCH_ON:     CW_SWITCH_ON,        # -> Switched On
    DriveState.SWITCHED_ON:            CW_ENABLE_OPERATION, # -> Operation Enabled
    DriveState.OPERATION_ENABLED:      CW_ENABLE_OPERATION, # hold enabled
    DriveState.QUICK_STOP_ACTIVE:      CW_ENABLE_OPERATION, # recover to enabled
    DriveState.FAULT_REACTION_ACTIVE:  CW_ENABLE_OPERATION,
    DriveState.FAULT:                  CW_FAULT_RESET,
}

def _control_word_for(state: DriveState) -> int:
    """Control word to command for the current state, climbing toward
    OPERATION_ENABLED."""
    return _CONTROL_WORD.get(state, CW_DISABLE_VOLTAGE)

def control_word(status_word: int, enable: bool) -> int:
    """Control word to command this tick, from the raw status word and the master
    enable."""
    if not enable or (status_word & _LIMIT_ACTIVE_BIT):
        return CW_SHUTDOWN
    return _control_word_for(drive_state(status_word))
