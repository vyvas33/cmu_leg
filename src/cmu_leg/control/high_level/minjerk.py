# min jerk trajectory, can be used within standing and swing controllers
def min_jerk_factor(tau: float) -> float:
    """Quintic min-jerk blend s(tau) for tau in [0, 1].

    s = 10*tau^3 - 15*tau^4 + 6*tau^5, with s(0)=0, s(1)=1 and zero velocity
    and acceleration at both ends.
    """
    return tau * tau * tau * (10.0 - 15.0 * tau + 6.0 * tau * tau)


def min_jerk(start: float, end: float, t0: float, duration: float, time: float) -> float:
    """Min-jerk position from `start` to `end` over `duration`, starting at `t0`.

    Holds at `start` before t0 and at `end` after t0 + duration (tau clamps).
    """
    tau = (time - t0) / duration  # [0, 1]
    tau = 0.0 if tau < 0.0 else 1.0 if tau > 1.0 else tau
    return start + (end - start) * min_jerk_factor(tau)
