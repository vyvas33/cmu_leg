import os
import sys


def _rtprio_limit():
    """RLIMIT_RTPRIO soft limit, or a note if it cannot be read. A non-root user
    cannot exceed this even with the right cgroup."""
    try:
        import resource
        soft, _ = resource.getrlimit(resource.RLIMIT_RTPRIO)
        return soft
    except (ImportError, AttributeError, ValueError, OSError):
        return "unknown"


def pin_to_core(cpu_core: int, rt_priority: int = 80) -> bool:
    """Pin the calling process to one core and put it on SCHED_FIFO.

    Affinity alone is not enough: it says which core we run on, not that we run
    when we want to. At ordinary CFS priority the kernel will deschedule the
    loop for milliseconds at a time, which on a 1 kHz EtherCAT cycle is long
    enough for the Elmo sync-manager watchdog to quick-stop the drives.

    Parameters
    ----------
    cpu_core    : core to run on. It should be isolated from the general
                  scheduler (isolcpus / cpuset), or other work still lands here
                  and the priority only helps once we are already running.
    rt_priority : SCHED_FIFO priority, 1 (lowest) to 99 (highest). 80 sits above
                  typical background threads and leaves headroom for kernel
                  threads that transiently land on the core.

    Returns True if the real-time policy was applied. Never raises: a bench
    machine without privileges, or a non-Linux host, should still run -- just
    with the timing that implies, and having said so.
    """
    ok = True
    try:
        os.sched_setaffinity(0, {cpu_core})
    except (AttributeError, OSError) as e:
        print(f"RT setup: could not pin to core {cpu_core}: {e}")
        ok = False

    if not sys.platform.startswith("linux"):
        print("RT setup: SCHED_FIFO is Linux-only; running at default priority")
        return False
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(rt_priority))
    except (AttributeError, OSError, PermissionError) as e:
        print(f"RT setup: SCHED_FIFO priority {rt_priority} refused ({e}); "
              f"running at default priority -- expect scheduler overruns")
        print(f"          euid={os.geteuid()}  rlimit_rtprio={_rtprio_limit()}")
        if os.geteuid() != 0:
            print("          not root: re-run under sudo, and check that sudo "
                  "reaches the child (uv/venv wrappers can drop it)")
        else:
            print("          root but still refused, so it is the cgroup, not "
                  "the user: with CONFIG_RT_GROUP_SCHED=y systemd gives the "
                  "session cgroup zero rt bandwidth. Check "
                  "`zcat /proc/config.gz | grep RT_GROUP_SCHED` and run under "
                  "`systemd-run --scope`.")
        return False

    print(f"RT setup: core {cpu_core}, SCHED_FIFO priority {rt_priority}")
    return ok
