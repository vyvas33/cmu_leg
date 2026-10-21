# `runtime/` — multiprocess control stack

This package splits the control stack across two async processes so that
estimation and the (swappable) high-level controller stop eating into the 1 kHz
real-time budget. It is the multiprocess counterpart to the single-process
`main.py`, which still exists and stays the byte-identical verification
reference.

```
  ┌───────────────── low-level process ─────────────────┐
  │  EtherCAT (ec.read / ec.write)                      │
  │      │                                               │
  │      ▼                                               │
  │  ProprioceptiveEstimator.update(frame, dt_meas)     │
  │      │                                               │
  │      ├──► StateBuffer.publish(state)   ──── B1 ──┐  │
  │      │                                           │  │
  │  CommandBuffer.latest()  ◄──────────── B3 ──────┐│  │
  │      │  (watchdog: stale → zero torque)         ││  │
  │      ▼                                          ││  │
  │  LowLevelControl.compute → DriveCommand → ec.write  │
  └───────────────────────────────────────────────┼─┼──┘
                                                   │ │ full proprioceptive
        ┌──────────── control process ──────────┐  │ ▼ state (hip_* = 0)
        │  StateBuffer.latest() ◄────────────────┼──┼─┘
        │      │                                 │     │
        │      ▼                                 │     │
        │  ESKF.update(imu, state, ...) → hip_*  │     │
        │      │  (full physical state assembled)│     │
        │      ▼   (mode FSM sets `standing`)    │     │
        │  ProsthesisControl.compute → TorqueDesired    │
        │      └──► CommandBuffer.publish(...) ──┼─ B3 ─┘
        └────────────────────────────────────── ┘  back down to low-level
```

Both processes run **async**, each free-running its own loop (low-level ~1 kHz,
control ~500 Hz). The buffers decouple them — nobody blocks on anybody.

## Why two processes (and not three)

Estimation (ESKF) and the high-level controller used to be **two separate
processes** with a `StateBuffer` B2 between them. That put an extra IPC hop — and
a 500 Hz vs 1 kHz rate mismatch — between the ESKF and the controller, so the
standing impedance's velocity-damping term acted on ~3–4 ms-stale velocity and
drove an **11 Hz limit cycle**. Merging the ESKF and the controller into one
`control` process (hgeyer-aligned co-location) halves that transport delay
(2 hops → 1). The fast, delay-sensitive velocity still lives at the sensor
(low-level, 1 kHz); if standing still oscillates, the damping term moves *down*
into the low-level — it cannot be fixed from the control process.

## The two processes

- **Low-level** — owns the EtherCAT master. Reads the frame, runs
  `ProprioceptiveEstimator` (raw frame → physical state), publishes it (**B1**),
  consumes the control command (**B3**) to compute the low-level `DriveCommand`,
  and `ec.write`s it (one PDO exchange per tick, which also reads the next frame).
- **Control** — owns the two IMUs. Reads B1, runs the `ESKF` to add `hip_*`
  (completing the full physical `ProsthesisState`), decides the mode, runs the
  active behavioral policy (`ProsthesisControl`), and publishes `TorqueDesired` +
  the mode flag (**B3**). This is also the **swap point**: impedance today, a
  neuromuscular / NN controller later, isolated so a heavier controller can't
  constrain the RT loop.

The estimation code is split to match: `ProprioceptiveEstimator` (low-level half)
and `ESKF` (control half) are separate classes so each process constructs only
what it uses. `ProsthesisStateEstimator` composes both for the single-process
`main.py` path.

## Buffers

Each is a lock-guarded `multiprocessing.Array('d')` with a monotonic `seq`
counter (0 = never published). One writer, one reader, one direction. `seq` lets
the reader tell a fresh sample from a held repeat. Common mechanics live in the
`_SharedRecord` base.

| Buffer | Class | Direction | Payload |
|---|---|---|---|
| B1 | `StateBuffer` | low-level → control | full proprioceptive `ProsthesisState` (`hip_*` = 0) |
| B3 | `CommandBuffer` | control → low-level | `TorqueDesired` (knee/ankle) + `standing` |

(`StateBuffer` is still generic — it carried B2 before the merge — but only B1
uses it now.)

Two rules:

- **Writer ownership.** Whoever publishes a buffer owns every value in it, so a
  field is authoritative only on the hop where its writer produces it. `standing`
  is set by the control process's mode FSM and is meaningful only on B3 (on B1 the
  low-level's copy is inert).
- **Velocities/accelerations are computed once, in the low-level process, and
  shared — never re-differentiated downstream.** A backward difference amplifies
  dt jitter, and a held-sample read would alias it. The 1 kHz low-level loop is
  the most deterministic clock, so it owns every derivative. `hip_accel` is the
  lone exception (its source `hip_velocity` is born in the ESKF).

## Why a command, not a state, on B3

Control's output is `TorqueDesired` — two floats — not a state. Low-level already
holds the freshest physical state locally, so B3 carries only the command (desired
torques + `standing`). Sending a full `ProsthesisState` back down would be
wasteful and backwards. A held command is at most ~1 low-level tick stale, so raw
torque on B3 is fine; if a genuinely slow controller is ever loaded, revisit —
ship gains/setpoints and let low-level close the loop at 1 kHz.

### B3 safety watchdog

Low-level tracks B3's `seq`. If it hasn't advanced in `MAX_STALE_TICKS` ticks, the
control process is assumed hung/dead and low-level forces a **safe zero** (zero
torque, `standing = False`) rather than holding the last, possibly nonzero,
command. This is safety-critical.

## Startup handshake

1. The launcher spawns **low-level**. It inits EtherCAT (after spawn — the pysoem
   master can't be pickled) and sets `init_evnt`.
2. The launcher blocks on `init_evnt` and only then spawns **control** — the
   consumer never comes up before the hardware is confirmed live.
3. **Control** inits its IMUs (~seconds) and then stamps a shared `start_time`
   (`perf_counter`). Low-level is **not** blocked on it: it heartbeats the
   EtherCAT bus (holding the drives in OP — a silent bus trips the Elmo
   sync-manager watchdog and drops them out of OP) and reads `start_time` once it
   becomes nonzero. Both loops then count `t` from that one origin, so both logs
   share a timeline. No `Barrier`: control is the sole stamper now that it's the
   only other process.

## Timing detail: two `dt`s

Each loop keeps **two** deltas: the nominal `dt` (a fixed param) steps the
sleep-to-clock, while the *measured* `dt_meas = now - t_prev` drives every
backward-difference velocity and the ESKF integration. hgeyer needs only one
because his velocities come from motor firmware; ours are computed in-loop, so the
measured dt is load-bearing.

## Logging

Each process logs its **own** trace with its own `Recorder`s (memory-only append
in the loop, disk write at teardown), into a single `run_dir` the parent creates
once and passes to both — so the CSVs land together and join on `t`. The parent
never logs the per-tick trace; it only monitors and tears down. Log names follow
`<process>_<payload>`: `low_prosthesis_state` / `low_frame` / `low_output` /
`low_drive` / `low_idle`, and `control_prosthesis_state` / `control_torque` /
`control_idle`.

## Files

| File | What it is |
|---|---|
| `buffer.py` | `_SharedRecord` base + `StateBuffer` (B1) + `CommandBuffer` (B3) |
| `low_level.py` | low-level process body (`run()` = the `Process` target) |
| `control.py` | control process body — ESKF + high-level controller, merged |
| `guard.py` | `ExIntGuard` — SIGTERM/Ctrl-C/exception → set `stop_event`, run cleanup |
| `launch.py` | the parent: builds IPC, spawns + gates + monitors + terminates |

## Running

```
uv run python -m cmu_leg.runtime.launch
```

## Status

The IPC plumbing is validated under `spawn`: the loop low → B1 → control → B3 →
low carries a command all the way around, and both processes agree on the stamped
`start_time`. Ran end-to-end on the Jetson + leg (EtherCAT + both IMUs + ESKF +
controller, all CSVs logged).

**`standing` is driven on a schedule, by design** — `standing = t > WARMUP_S` in
`control.py`. We set the mode ourselves for now (like `main.py`'s phased run); an
autonomous mode FSM is future work, not a gap. Adjust `WARMUP_S` / add phases to
change the schedule.

**Re-tune after the merge:** the controller now runs at ~500 Hz (was 1 kHz in the
old high-level process). Standing's velocity lowpass / dt-dependent coefficients
in `standing.py` may need re-tuning at the new rate.

Knobs to set for hardware: `CORES` in `launch.py` (isolated Jetson cores),
`DT_CONTROL`, and `MAX_STALE_TICKS` (the B3 watchdog threshold).
