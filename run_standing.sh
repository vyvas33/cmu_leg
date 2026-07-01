#!/usr/bin/env bash
#
# Launch the standing control loop with real-time settings.
#
# EtherCAT needs a raw socket (root) and the loop needs SCHED_FIFO, so this must
# run as root. Just call it -- it re-execs itself under sudo if needed:
#
#     ./run_standing.sh
#
# It also sets the Jetson performance preconditions on every launch so you can't
# forget them. Adjust CORE if you isolate a different core (see isolcpus).

set -euo pipefail

REPO=/home/cmuleg/cmu_leg
PY="$REPO/.venv/bin/python"
CORE=3            # control core to pin to (must exist; keep off core 0)
PRIO=90           # SCHED_FIFO priority

# --- must be root: EtherCAT raw socket + RT scheduling ---
if [[ $EUID -ne 0 ]]; then
    exec sudo "$0" "$@"
fi

# --- performance preconditions (idempotent, safe to repeat) ---
nvpmodel -m 0            >/dev/null 2>&1 || echo "warn: nvpmodel failed (not a Jetson?)"
jetson_clocks           >/dev/null 2>&1 || echo "warn: jetson_clocks failed"
sysctl -w kernel.sched_rt_runtime_us=-1 >/dev/null   # allow SCHED_FIFO past RT_GROUP_SCHED

# --- run: SCHED_FIFO priority, pinned to the control core ---
cd "$REPO"
echo "launching standing on core $CORE at SCHED_FIFO prio $PRIO ..."
chrt -f "$PRIO" taskset -c "$CORE" "$PY" -m cmu_leg.main "$@"

# --- give the log files back to the invoking user (else they're root-owned) ---
if [[ -n "${SUDO_USER:-}" ]]; then
    chown -R "$SUDO_USER:$SUDO_USER" "$REPO/output" 2>/dev/null || true
fi
