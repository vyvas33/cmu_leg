import os


def pin_to_core(cpu_core: int, rt_priority: int = 80):
    """
    Pins process which executes this function to an isolated CPU core and
    switches it to SCHED_FIFO (first-in, first-out real-time scheduling).

    Parameters
    ----------
    cpu_core    : isolated core to run on — must be in the set defined
                  by isolcpus in cmdline.txt (cores 1, 2, or 3)
    rt_priority : SCHED_FIFO priority, 1 (lowest) to 99 (highest).
                  80 is a reasonable default — it sits above typical
                  background threads but leaves headroom for any kernel
                  threads that transiently land on the core.
    """
    os.sched_setaffinity(0, {cpu_core})
    os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(rt_priority))
    print(f"RT setup: core {cpu_core}, SCHED_FIFO priority {rt_priority}")
