"""
Context manager for safe handling of system interrupts and exceptions in a
process. Adapted from hgeyer's guard.py.

On entering, installs a SIGTERM handler that raises SystemExit so a parent's
Process.terminate() unwinds through normal exception handling. On exiting (for
ANY reason -- normal return, exception, or signal) it sets the shared stop_event
so every process shuts down, runs the cleanup function, and restores the original
signal handlers. Ctrl-C and ControlError are consumed (graceful shutdown); other
exceptions re-raise after cleanup.
"""

from signal import signal, getsignal, SIGTERM, SIGINT, SIG_IGN


class ControlError(Exception):
    """Raise for a prosthesis control fault that should shut everything down
    gracefully (consumed by the guard, not re-raised)."""


class ExIntGuard:
    """`with` guard that guarantees cleanup on exceptions or system interrupts.

    Args:
        name (str): Identifier for prints (which process tripped).
        stop_event (multiprocessing.Event): Shared shutdown flag; set on any exit
            with an exception so every process stops.
        cleanup_fcn (callable): Runs on exit whether or not an exception occurred.
        is_main_process (bool): True in the parent (restores the SIGINT handler
            it owns); children leave SIGINT alone.
    """

    def __init__(self, name, stop_event, cleanup_fcn, is_main_process=False):
        self.name = name
        self.stop_event = stop_event
        self.cleanup_fcn = cleanup_fcn
        self.orig_sigterm = getsignal(SIGTERM)
        signal(SIGTERM, self._on_sigterm)
        self.orig_sigint = getsignal(SIGINT) if is_main_process else None

    def _on_sigterm(self, signum, frame):
        # args unused but required for a signal handler
        raise SystemExit(f"\n{self.name}: ExIntGuard: SIGTERM received.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is not None:
                self.stop_event.set()

                if exc_type is KeyboardInterrupt:
                    signal(SIGINT, SIG_IGN)   # ignore further Ctrl-C during shutdown
                    print(f"\n{self.name}: ExIntGuard: Ctrl-C interrupt by user.")
                    return True               # consume it

                if exc_type is ControlError:
                    print(exc_val)
                    print("\nSHUTTING DOWN CONTROL.")
                    return True               # consume it

                print(f"\n{self.name}: ExIntGuard: error: "
                      f"{exc_type.__name__}: {exc_val}")
                return False                  # re-raise anything else
        finally:
            signal(SIGTERM, self.orig_sigterm)
            self.cleanup_fcn()
            if self.orig_sigint is not None:
                signal(SIGINT, self.orig_sigint)

        return False
