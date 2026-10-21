# Architecture

The stack runs as **two processes**. The low-level process talks to the hardware. It
reads the EtherCAT (a communication protocol) frame at 1 kHz, derives everything that
comes straight from the encoders and hall sensors, runs the joint torque loops, and
writes the drive command.

The control process runs at 500 Hz: it performs the estimation by fusing the IMUs and
produces desired joint torques from the current `ProsthesisState`. The two processes
communicate through a small array of shared memory, written once per tick in each
direction.

Data flows one way. Sensors fill a `ProsthesisState`, the state bus that every
consumer reads and only the estimator writes. The high level uses that state and writes
`TorqueDesired`. The low level turns that into a `DriveCommand` in the units the
drives natively consume. Each stage is a plain class with a `compute()` method.

## Package layout

```
src/cmu_leg/
  main.py          
  config/          
  structures/      
  hardware/        
  estimation/      
  control/
    high_level/    
    low_level/     
  runtime/         
  utility/         
  tests/          
```

## Covered in the following sections

- **Structures** — `ProsthesisState`, `TorqueDesired`, `DriveCommand`, and who is
  allowed to write what.
- **Hardware** — the EtherCAT frame, the CiA-402 drive state machine, the IMUs.
- **Estimation** — encoders and velocities, contact detection, and the ESKF.
- **High-level control** — the mode FSM, the gait FSM, and the behaviours they select.
- **Low-level control** — conditioning, the torque controller, and saturation, per joint.
- **Runtime** — how the two processes are launched, how they exchange data, and how
  they fail safe.
- **Conventions** — the cross-cutting rules that hold across all of the above.

!!! note "This page is cited from the code"
    Keep the filename `docs/arch.md`. Two comments point here by path:
    `control/low_level_control.py` (why disable behaviour deviates from the Simulink
    baseline) and `config/params.py` (why the ankle motor-efficiency compensation is
    disabled). Both still need writing.
