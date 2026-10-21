"""
One-directional shared buffers between the runtime processes.

Each buffer is one lock-guarded multiprocessing.Array('d'): some value slots
followed by a monotonic `seq` counter (0 = never published, bumped per publish so
the reader tells a fresh sample from a held repeat). One writer, one reader, one
direction. Built in the PARENT before spawn and handed to both children; the
Array survives spawn (unlike the pysoem master or USB handles -- hence hardware
is initialized AFTER spawn).

Rule for all of them: never compute under the lock -- copy in/out only.

Buffers:
  - StateBuffer: the full physical ProsthesisState. Used for BOTH hops that carry
    state -- B1 (low-level -> estimator) and B2 (estimator -> high-level) are each
    a StateBuffer. Low-level publishes the complete proprioceptive state on B1;
    the estimator seeds its own state from it, overlays hip_* (ESKF), and
    republishes the now-complete state on B2. One payload type, no field-picking.
  - CommandBuffer (B3): high-level -> low-level. Desired joint torques + the
    standing mode flag -- high-level emits a COMMAND, not a state, so this is
    small and separate.

_SharedRecord holds the array/lock/seq mechanics; subclasses own the
field<->slot mapping.
"""

from dataclasses import fields as dataclass_fields
from multiprocessing import Array
from typing import NamedTuple

from cmu_leg.structures.state import ProsthesisState
from cmu_leg.structures.commands import TorqueDesired, Mode


class _SharedRecord:
    """Base for the one-directional latest-value buffers.

    Owns a multiprocessing.Array('d') of `n_payload` value slots + 1 seq slot,
    the lock, and the two locked critical sections (_publish, _snapshot).
    Subclasses map their fields onto the value slots.

    Args:
        n_payload (int): number of value slots (seq is added as slot n_payload).
    """

    def __init__(self, n_payload: int):
        self._seq = n_payload                 # seq lives in the last slot
        self._arr = Array("d", n_payload + 1)

    def _publish(self, values) -> None:
        """Store the payload and bump seq under the lock.

        Args:
            values (sequence[float]): exactly n_payload floats, in slot order.
        """
        with self._arr.get_lock():
            buf = self._arr.get_obj()
            buf[0:self._seq] = values
            buf[self._seq] += 1.0

    def _snapshot(self) -> list[float]:
        """Copy all slots out in one shot under the lock (the only reader-side
        critical section); the typed payload is built by the caller, lock-free."""
        with self._arr.get_lock():
            return self._arr.get_obj()[:]


# ====================================================================
# StateBuffer: full physical ProsthesisState -- used for B1 and B2
# ====================================================================

_STATE_FIELDS = [f.name for f in dataclass_fields(ProsthesisState)]
_STATE_N = len(_STATE_FIELDS)
# Bools ride the 'd' array as 1.0/0.0 and have to be decoded back on read.
# DERIVED, not hardcoded: this used to name `standing` explicitly, and both
# `contact` and `control_enabled` were then added and came back as floats.
# Truthiness happened to work, so nothing broke loudly -- exactly the kind of
# thing that bites later. Adding a bool to ProsthesisState now just works.
_BOOL_FIELDS = [n for n in _STATE_FIELDS
                if isinstance(getattr(ProsthesisState(), n), bool)]


class StampedState(NamedTuple):
    """A ProsthesisState snapshot plus the buffer's seq (for freshness /
    staleness checks). ProsthesisState itself stays seq-free."""
    state: ProsthesisState
    seq: int


class StateBuffer(_SharedRecord):
    """A one-directional latest-value channel carrying the full ProsthesisState.

    Used for both state hops:
      - B1 (low-level -> estimator): low-level publishes its complete
        proprioceptive state (hip_* still zero -- it runs no ESKF). The estimator
        seeds its own state from this, then overlays hip_*.
      - B2 (estimator -> high-level): the estimator republishes the now-complete
        state (proprioceptive + hip_*).

    Writer ownership: whoever publishes owns every value, so a field is only
    authoritative on the hop where its writer produces it. `standing` is a mode
    field neither low-level nor the estimator authors (high-level owns it and
    sends it DOWN on B3), so it is inert on both B1 and B2. It rides along only
    because it is part of the dataclass.

    Bool fields are stored as 1.0/0.0 in the 'd' array and decoded back to bool
    on read; the set is derived from the dataclass, so it tracks new fields.
    """

    def __init__(self):
        super().__init__(_STATE_N)

    def publish(self, state: ProsthesisState) -> None:
        """Writer (estimator). Serialize every field in dataclass order."""
        self._publish([float(getattr(state, name)) for name in _STATE_FIELDS])

    def latest(self) -> StampedState | None:
        """Reader (high-level). None until published once; else state + seq."""
        snap = self._snapshot()
        seq = int(snap[self._seq])
        if seq == 0:
            return None
        vals = snap[0:self._seq]
        kwargs = {name: vals[i] for i, name in enumerate(_STATE_FIELDS)}
        for name in _BOOL_FIELDS:
            kwargs[name] = bool(round(kwargs[name]))
        return StampedState(ProsthesisState(**kwargs), seq)


# ====================================================================
# B3: high-level -> low-level (command: desired torques + mode flag)
# ====================================================================

class HighLevelCommand(NamedTuple):
    """One high-level command snapshot. `mode` tells the low level whether to
    actuate at all and whether to suppress the knee extension stop; `seq` drives
    the low-level staleness watchdog."""
    knee_desired: float
    ankle_desired: float
    mode: Mode
    seq: int


_CMD_N = 3   # knee_desired, ankle_desired, mode (as float)


class CommandBuffer(_SharedRecord):
    """High-level -> low-level channel. Desired joint torques + the standing mode
    -- NOT a full state (high-level emits a command, and low-level already
    holds the freshest physical state).

    SAFETY: the low-level process MUST watchdog `seq`. If it has not advanced in
    N ticks the high-level process is hung/dead -- zero the torque, never hold a
    stale command.
    """

    def __init__(self):
        super().__init__(_CMD_N)

    def publish(self, torque: TorqueDesired, mode: Mode) -> None:
        """Writer (high-level)."""
        self._publish((torque.knee_desired, torque.ankle_desired, float(int(mode))))

    def latest(self) -> HighLevelCommand | None:
        """Reader (low-level). None until published once."""
        snap = self._snapshot()
        seq = int(snap[self._seq])
        if seq == 0:
            return None
        knee, ankle, mode = snap[0:self._seq]
        return HighLevelCommand(knee, ankle, Mode(int(round(mode))), seq)
