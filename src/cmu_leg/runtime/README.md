# `runtime/` — multiprocess control stack

This package splits the control stack across three async processes so that
estimation and the (swappable) high-level controller stop eating into the 1 kHz
real-time budget. It is the multiprocess counterpart to the single-process
`main.py`, which still exists and stays the byte-identical verification
reference.

```
  ┌───────────────── low-level process ─────────────────┐
  │  EtherCAT (ec.read)                                  │
  │      │                                               │
  │      ▼                                               │
  │  ProprioceptiveEstimator.update(frame, dt_meas)     │
  │      │                                               │
  │      ├──► StateBuffer.publish(state)   ──── B1 ──┐  │
  │      │                                           │  │
  │  CommandBuffer.latest()  ◄──────────── B3 ──────┐│  │
  │      │  (watchdog: stale → zero torque)         ││  │
  │      ▼                                          ││  │
  │  LowLevelControl.compute → DriveCommand         ││  │
  │  (ec.write DEFERRED — logged, not actuated yet) ││  │
  └───────────────────────────────────────────────┼─┼──┘
                                                   │ │ full proprioceptive
        ┌────────── estimator process ──────────┐  │ ▼ state (hip_* = 0)
        │  StateBuffer.latest() ◄────────────────┼──┼─┘
        │      │                                 │     │
        │      ▼                                 │     │
        │  ESKF.update(imu, state, ...) → hip_*  │     │
        │      │  (full physical state assembled)│     │
        │      └──► StateBuffer.publish(state) ──┼─ B2 ─┐
        └─────────────────────────────────────── ┘     │
        ┌────────── high-level process ─────────┐       │ full ProsthesisState
        │  StateBuffer.latest()  ◄──────────────┼───────┘
        │      │                                │
        │      ▼   (mode FSM sets `standing`)   │
        │  ProsthesisControl.compute → TorqueDesired
        │      └──► CommandBuffer.publish(...) ──┘  B3, back down to low-level
        └───────────────────────────────────────┘
```

All three processes run **async**, each free-running its own ~1 kHz loop. The
buffers decouple them — nobody blocks on anybody. The only synchronization is a
one-time start barrier.

## The three processes

- **Low-level** — owns the EtherCAT master. Reads the frame, runs
  `ProprioceptiveEstimator` (raw frame → physical state), publishes the
  medulla subset (**B1**), and consumes the high-level command (**B3**) to
  compute the low-level `DriveCommand`. Eventually does `ec.write`.
- **Estimator** — owns the two IMUs. Reads B1, runs the `ESKF` to add `hip_*`,
  which completes the full physical `ProsthesisState`, and publishes it (**B2**).
- **High-level** — the **swap point**: impedance today, a neuromuscular / NN
  controller later, isolated so a heavier controller can't constrain the
  estimator or the RT loop. Reads B2, decides the mode, runs the active policy,
  and publishes `TorqueDesired` + the mode flag (**B3**).

The estimation code is split to match: `ProprioceptiveEstimator` (low-level
half) and `ESKF` (estimator half) are separate classes so each process
constructs only what it uses. `ProsthesisStateEstimator` composes both for the
single-process `main.py` path.

## Buffers

Each is a lock-guarded `multiprocessing.Array('d')` with a monotonic `seq`
counter (0 = never published). One writer, one reader, one direction. `seq` lets
the reader tell a fresh sample from a held repeat. Common mechanics live in the
`_SharedRecord` base.

| Buffer | Class | Direction | Payload |
|---|---|---|---|
| B1 | `StateBuffer` | low-level → estimator | full proprioceptive `ProsthesisState` (`hip_*` = 0) |
| B2 | `StateBuffer` | estimator → high-level | full `ProsthesisState` (proprioceptive + `hip_*`) |
| B3 | `CommandBuffer` | high-level → low-level | `TorqueDesired` (knee/ankle) + `standing` |

Two rules:

- **Writer ownership.** Whoever publishes a buffer owns every value in it, so a
  field is authoritative only on the hop where its writer produces it. `standing`
  is set by high-level's mode FSM and is meaningful only on B3 (on B2 the
  estimator's copy is inert).
- **Velocities/accelerations are computed once, in the low-level process, and
  shared — never re-differentiated downstream.** A backward difference amplifies
  dt jitter, and a held-sample read would alias it. The 1 kHz low-level loop is
  the most deterministic clock, so it owns every derivative. `hip_accel` is the
  lone exception (its source `hip_velocity` is born in the ESKF).

## Why a command, not a state, on B3

High-level's output is `TorqueDesired` — two floats — not a state. Low-level
already holds the freshest physical state locally, so B3 carries only the command
(desired torques + `standing`). Sending a full `ProsthesisState` back down would
be wasteful and backwards.

Because all three run at ~1 kHz, raw torque on B3 is fine (no rate gap; a held
command is at most ~1 tick stale). If a genuinely slow controller is ever loaded,
revisit — ship gains/setpoints and let low-level close the loop at 1 kHz.

### B3 safety watchdog

Low-level tracks B3's `seq`. If it hasn't advanced in `MAX_STALE_TICKS` ticks,
the high-level process is assumed hung/dead and low-level forces a **safe zero**
(zero torque, `standing = False`) rather than holding the last, possibly nonzero,
command. This is safety-critical.

## Startup handshake

1. The launcher spawns **low-level**. It inits EtherCAT (after spawn — the
   pysoem master can't be pickled) and sets `init_evnt`.
2. The launcher blocks on `init_evnt` and only then spawns **estimator** and
   **high-level** — the consumers never come up before the hardware is confirmed
   live.
3. All three hit a shared `Barrier(3)`. The first to trip it stamps a shared
   `start_time` (`perf_counter`); a second `wait()` guarantees the others read it
   only after. Every loop then counts `t` from that one origin, so all three logs
   share a timeline. (Two-phase barrier rather than a `Barrier(action=...)`,
   which is awkward to pickle to spawned children.)

## Timing detail: two `dt`s

Each loop keeps **two** deltas: the nominal `dt` (a fixed param) steps the
sleep-to-clock, while the *measured* `dt_meas = now - t_prev` drives every
backward-difference velocity and the ESKF integration. hgeyer needs only one
because his velocities come from motor firmware; ours are computed in-loop, so
the measured dt is load-bearing.

## Logging

Each process logs its **own** trace with its own `Recorder`s (memory-only append
in the loop, disk write at teardown), into a single `run_dir` the parent creates
once and passes to all three — so the CSVs land together and join on `t`. The
parent never logs the per-tick trace; it only monitors and tears down.

## Files

| File | What it is |
|---|---|
| `buffer.py` | `_SharedRecord` base + `StateBuffer` (B1 + B2) + `CommandBuffer` (B3) |
| `low_level.py` | low-level process body (`run()` = the `Process` target) |
| `estimator.py` | estimator process body |
| `high_level.py` | high-level process body (the swappable controller) |
| `guard.py` | `ExIntGuard` — SIGTERM/Ctrl-C/exception → set `stop_event`, run cleanup |
| `launch.py` | the parent: builds IPC, spawns + gates + monitors + terminates |

## Running

```
uv run python -m cmu_leg.runtime.launch
```

## Status

The IPC plumbing is validated under `spawn`: the full three-hop chain
(low → B1 → estimator → B2 → high-level → B3 → low) carries a command all the way
around, and all three processes agree on the stamped `start_time`.

**Not yet done — needs the Jetson + leg:**

- **Actuation is deferred.** Low-level computes and logs the `DriveCommand` but
  does not `ec.write` yet (still read-only `ec.read()` each tick). To actuate,
  switch to the `main.py` exchange pattern — read once before the loop and
  `frame = ec.write(drive_cmd)` per tick (one PDO exchange per tick) — and manage
  `ec.enable`. See the `TODO(bench)` in `low_level.py`.
- **`standing` is driven on a schedule, by design** — `standing = t > WARMUP_S`
  in `high_level.py`. We set the mode ourselves for now (like `main.py`'s phased
  run); an autonomous mode FSM is future work, not a gap. Adjust `WARMUP_S` / add
  phases to change the schedule.
- Real EtherCAT/IMU bring-up, the `init_evnt` handshake, and guarded teardown
  saving all CSVs are untested on hardware.

Knobs to set for hardware: `CORES` in `launch.py` (isolated Jetson cores),
`DT_EST`/`DT_HIGH` (at 1 kHz for bring-up), and `MAX_STALE_TICKS` (the B3
watchdog threshold).
