- Hardware chain order: Medulla → Knee Elmo → Ankle Elmo
- Cycle rate: 1ms
- Slave indices: Medulla=0, Knee=1, Ankle=2
- PDO offsets: (from our XML extraction)
- Low level control pipeline: Conditioning → Controller → Saturation
- Knee Kp: 0.09, Ankle Kp: 0.08
- Knee efficiency compensation: active
- Ankle efficiency compensation: disabled
- Parameters needed: knee_motor_damping, knee_motor_inertia, 
                     knee_stiffness, knee_motor_eff, 
                     ankle equivalents
- Still need: parameter .m file values from MATLAB project

## Low Level Control enable behavior

- `control_enabled` is the AND of (knee Elmo CiA 402 status word == 15,
  ankle Elmo status word == 15, medulla counter has ticked at least once).
  It comes in as a field on `ProsthesisState` — the orchestrator does not
  recompute it.

- Rising edge (False → True): all stateful sub-blocks are reset
  (conditioning LPFs, controller backward-differences and spring-velocity LPF,
  saturation LPFs). Matches Simulink "StatesWhenEnabling: reset" on the
  Low Level Control enable port.

- Startup mute window: for `STARTUP_MUTE_TICKS` (= 1000 ticks = 1.0 s)
  after the rising edge, the full pipeline runs every tick — so filters
  settle and backward-difference history populates — but the four
  drive-facing fields are forced to zero
  (`knee_velocity_commanded`, `knee_motor_torque_ff`,
   `ankle_velocity_commanded`, `ankle_motor_torque_ff`).
  The four "logging" fields (`*_torque_commanded`, `*_velocity_desired`)
  carry the live values so you can see what the controller would have asked
  for. Matches the Simulink 1-second `Delay(sample_freq * 1)` on the
  drive-output switches.

- Falling edge / disabled: pipeline is **not run**, and every field of
  `LowLevelOutput` is set to zero — including the logging fields.
  This is intentionally more conservative than Simulink, which has
  `OutputWhenDisabled = "held"` on the eight outports and trusts the Elmo
  state machine to ignore stale commands. We emit zeros in software so
  there is no possible window where a stale `LowLevelOutput` could reach
  the drives if the EtherCAT interlocks are misconfigured.
