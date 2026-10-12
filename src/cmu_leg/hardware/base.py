"""
Hardware interface contracts.

Defines the structural contract that every hardware interface satisfies, so
the main loop can bring up, read, and tear down any sensor source uniformly,
and so test fakes (replay/virtual) can stand in for real hardware without
inheriting anything.

These are typing.Protocols: a class counts as a SensorInterface purely by
having the right methods -- no base class, no import, no registration. The
existing EtherCATInterface and IMUInterface already satisfy SensorInterface as
written; declaring conformance is optional (see notes at the bottom).

vkalva, June 2026
"""
from typing import Protocol, TypeVar, runtime_checkable

from cmu_leg.structures.commands import DriveCommand

# The per-interface frame type (RawEtherCATFrame, IMUFrame, ...). Covariant so a
# SensorInterface[RawEtherCATFrame] is usable where SensorInterface[Frame] is
# expected. read() returns this type, so a single interface keeps its concrete
# frame type; for a heterogeneous list use SensorInterface[Any].
FrameT = TypeVar("FrameT", covariant=True)


@runtime_checkable
class SensorInterface(Protocol[FrameT]):
    """Common lifecycle of any hardware source: open it, pull one frame per
    call, close it. Implemented by EtherCATInterface and IMUInterface.

    read() is non-blocking and returns the interface's own frame type every
    call (a stale/empty frame when no new data is available), so the loop runs
    at a fixed rate regardless of per-device arrival timing.
    """

    def connect(self) -> None:
        """Bring the link up. Raise on failure; leave nothing half-open."""
        ...

    def read(self) -> FrameT:
        """Exchange once and return the latest frame. Non-blocking."""
        ...

    def close(self) -> None:
        """Release the link. Safe to call after a failed connect()."""
        ...


@runtime_checkable
class DriveInterface(SensorInterface[FrameT], Protocol[FrameT]):
    """A SensorInterface that also drives actuators -- adds command output and
    a readiness gate on top of the read lifecycle. Implemented by
    EtherCATInterface (the IMUs are read-only and do NOT satisfy this).

    `enable` is the master switch the main loop sets; write() derives the
    drives' control words from it each cycle.
    """

    enable: bool

    def write(self, cmd: DriveCommand) -> FrameT:
        """Send one command cycle and return the fresh input frame."""
        ...

    def control_enabled(self) -> bool:
        """True when the drives are enabled and the bus is live -- the main
        loop's gate for whether it may run the controllers this tick."""
        ...


# --------------------------------------------------------------------
# Declaring conformance (optional):
#
#   Because these are structural Protocols, EtherCATInterface and IMUInterface
#   already match without any change. You do NOT need to subclass these.
#
#   To make the intent explicit and let the type checker verify a class on the
#   spot, annotate a throwaway at module load:
#
#       _: DriveInterface = EtherCATInterface("eth0")   # checked, not run
#
#   or annotate the consumer instead (preferred -- this is where the value is):
#
#       def bring_up(sources: list[SensorInterface]) -> None:
#           for s in sources:
#               s.connect()
#
#   A fake needs only the same methods, no inheritance:
#
#       class ReplayEtherCAT:
#           def connect(self): ...
#           def read(self): return next(self._frames)
#           def close(self): ...
