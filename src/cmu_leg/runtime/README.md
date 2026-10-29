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

The parent logs exactly one thing, `treadmill` (belt speeds at 2 Hz), because it
is the one device it owns. Belt speed is the biggest single confound when reading
a walking trial and it used to be recorded nowhere.

## Files

| File | What it is |
|---|---|
| `buffer.py` | `_SharedRecord` base + `StateBuffer` (B1) + `CommandBuffer` (B3) |
| `low_level.py` | low-level process body (`run()` = the `Process` target) |
| `control.py` | control process body — ESKF + high-level controller, merged |
| `guard.py` | `ExIntGuard` — SIGTERM/Ctrl-C/exception → set `stop_event`, run cleanup |
| `operator.py` | operator intent (`TimedOperator` scripted, `TerminalOperator` keys) |
| `treadmill.py` | `BeltSupervisor` (belts slaved to a run) + a standalone belt script |
| `launch.py` | the parent: builds IPC, spawns + gates + monitors + terminates |


## Treadmill

The Bertec belts can be driven **from a run**, so a walking test does not need a
second person at the Control Panel. `BeltSupervisor` (in `treadmill.py`) lives in
the **parent** and is stepped once per pass of the monitor loop.

It is not in the control process on purpose. `connect()` blocks for up to
`TREADMILL.timeout_s` and a send is a syscall with no latency bound — a pinned
500 Hz SCHED_FIFO loop cannot absorb either. The parent, by contrast, already
runs a 0.5 s loop, already holds **b1** and **b3**, and is the only process with
a tty, so it is the only place where "stop the belts" can be made to happen on
every exit path.

It needs no new IPC. The mode arrives on **b3** (`CommandBuffer.latest()`), and
that record's `seq` is the liveness signal that distinguishes a *hung* control
process from a dead one — `Process.is_alive()` stays True for a process stuck in
a syscall, `seq` does not. `contact` comes off **b1** when `require_contact` is
set. The belts never command the leg; the coupling is one-way.

### Parameters (`TREADMILL`, in `config/params.py`)

| Param | Default | What it does |
|---|---|---|
| `enabled` | `False` | the deliberate per-session opt-in |
| `speed` / `acc` | 0.8 m/s, 0.5 m/s² | belt speed and the planned ramp |
| `stop_acc` | 1.5 m/s² | fault stops; 0.8 m/s dies in ~0.5 s instead of 1.6 |
| `max_speed` | 1.5 m/s | refuses a configured `speed` above this, to catch a typo |
| `countdown_s` | 5.0 s | from "the leg has finished standing" to the belts moving |
| `require_contact` | `False` | also require the prosthetic foot loaded before starting |
| `max_stale_passes` | 2 | monitor passes with no `b3.seq` before calling control hung |
| `stop_timeout_s` | 15.0 s | longest wait for the belts to read zero after a stop |
| `host` / `port` / `timeout_s` | — | the Control Panel socket |

The gate is **two-sided**: belts move only with `TREADMILL.enabled` **and**
`RUNTIME.arm_motion`. Arming the leg is the veto, not the permission —
`arm_motion` is true on every walking run, including bench and overground ones
where belts spinning up behind you would be a nasty surprise.

With `require_contact` false there is no *physical* precondition left, so
`enabled` is the only thing between a bench run and the belts moving. **Put it
back to `False` when you are done.**

### The sequence

Everything is keyed off one observable event — the mode FSM reaching STANDING —
which both processes see on the same tick, so the ordering is arithmetic rather
than two absolute times that have to be hand-aligned.

```
T          mode FSM reaches STANDING          (parent sees it on b3)
T + 2.0    STANDING.duration — the pose ramp is finished, the leg is holding
           -> countdown begins, "belts in 5 s"
T + 3.0    RUNTIME.walk_after_standing_s -> standing_toggle drops.
           The leg is now WILLING to walk. Belts still stationary.
T + 7.0    countdown expires -> belts commanded to speed
T + 8.6    belts at 0.8 m/s -> you step in; the first unloaded tick takes the
           mode FSM STANDING -> WALKING
```

Two things about that order matter:

**Permission comes before motion, not after.** `standing_toggle` is a permission,
not a trigger — STANDING → WALKING still needs `not contact`, i.e. *you* taking a
step. If the toggle dropped after the belts were already moving, the gap would be
spent with a straight, stiff knee (`StandingController` holds it at 5° with
`kp=150`) being dragged backwards with no swing available. So
`walk_after_standing_s` must fall **before** `STANDING.duration + countdown_s`;
`connect()` refuses to start otherwise, and refuses `walk_after_standing_s=None`
outright.

**The dwell is on *continuous* readiness**, not a latch — same pattern as
`_enabled_for` in the mode FSM. Drop to HOLD during the countdown and it cancels
and starts over on recovery, rather than firing the instant the leg comes back.

Events land on the monitor loop's 0.5 s grid, so the belts move up to half a
second after the arithmetic says.

### Stopping

The belts run while `mode.actuate` — STANDING **or** WALKING. Running through
STANDING ↔ WALKING is deliberate: the belt has to be moving before you can take
the first step, so standing on a moving belt is a normal part of the sequence,
and ending a trial stays a deliberate act (hold, or Ctrl-C).

They stop at `stop_acc` on: the mode leaving STANDING/WALKING, `b3.seq` stalling,
a lost socket, `stop_event` (a fault or torque trip anywhere), a dead child, the
liveness deadline, Ctrl-C and SIGTERM. `terminate()` calls `belt.shutdown()`
**first**, before joining the children — those joins take up to 5 s each and the
belts should not be running through them. `ExIntGuard` has already set SIGINT to
`SIG_IGN` by that point, so a second Ctrl-C cannot interrupt the stop.

`BeltState.DONE` is terminal. Belts that restart themselves after a fault are a
surprise, and a moving floor is the last place for one — re-run instead.

What the software **cannot** cover: `kill -9`, Jetson power loss, or the link
dropping. The belts keep running. **Keep the physical E-stop in reach**, and note
that with no second person it is the person on the belt who has to reach it.

### Changing it

- **Belt speed** — `TREADMILL.speed`. Above `max_speed` it refuses; raise that
  deliberately, in params, not at the call site.
- **More or less time before the belts move** — `countdown_s`. If you push it
  past a few seconds, raise `walk_after_standing_s` to match, keeping it under
  `STANDING.duration + countdown_s`.
- **When the leg becomes willing to walk** — `RUNTIME.walk_after_standing_s`.
  Being early costs nothing (it is a permission); being late means the leg will
  not swing when you step.
- **Walking overground instead** — set `require_contact=True` so the belts need
  the foot loaded as well, and bench-check that contact actually asserts when you
  stand the way you stand.
- **Belts off entirely** — `enabled=False`. The supervisor then never opens the
  socket and `step()` is a no-op; nothing else about the run changes.
- **No leg at all** — `uv run python -m cmu_leg.runtime.treadmill` still runs the
  belts standalone, with its own countdown and `--speed` / `--acc` / `--host`.
  That path is untouched by any of the above and stays the bring-up tool.

Control Panel setup (Remote TCP/IP Control, *Listen on 127.0.0.1 Only* unchecked,
and the **Ready for Remote Control** button — without it commands are silently
ignored) is documented in `hardware/treadmill/bertec_treadmill.py`.

## Running

```
uv run python -m cmu_leg.runtime.launch
```

## Status

The IPC plumbing is validated under `spawn`: the loop low → B1 → control → B3 →
low carries a command all the way around, and both processes agree on the stamped
`start_time`. Ran end-to-end on the Jetson + leg (EtherCAT + both IMUs + ESKF +
controller, all CSVs logged).

**Re-tune after the merge:** the controller now runs at ~500 Hz (was 1 kHz in the
old high-level process). Standing's velocity lowpass / dt-dependent coefficients
in `standing.py` may need re-tuning at the new rate.

Knobs to set for hardware: `CORES` in `launch.py` (isolated Jetson cores),
`DT_CONTROL`, and `MAX_STALE_TICKS` (the B3 watchdog threshold).
