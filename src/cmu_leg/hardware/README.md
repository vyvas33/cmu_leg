# Hardware Interfaces

Live hardware boundary for the CMU leg. Two interfaces, both `connect()` / `read()` / `close()`, each returning a raw (pre-interpretation) frame:

- **`EtherCATInterface`** (`ethercat_io.py`) — joints + drives over EtherCAT → `RawEtherCATFrame` (raw integer counts) / `DriveCommand`.
- **`IMUInterface`** (`imu_io.py`) — two WitMotion HWT906 IMUs (thigh + foot) over USB serial (CH340) → `IMUFrame` (SI units, **each sensor's own coordinate frame**).

## IMU coordinate frames

The IMUs stream accel/gyro in their **own body frame**, and `imu.read()` passes them through unchanged (SI units, sensor frame). Conversion into the EKF frame is **deferred to the ESKF layer**.

**EKF frame:** `x = forward`, `y = left`, `z = up` (right-handed).

**Sensor mountings** — physical, from the baseline Simulink `Read IMU` annotations. Re-derive if an IMU is remounted:

| IMU | sensor +x | sensor +y | sensor +z |
|---|---|---|---|
| Thigh | backward (−x = forward) | up | left |
| Foot | left | backward | up |

**Resulting remap** (`v_ekf = R · v_sensor`; applies identically to accel and gyro):

| IMU | EKF x (fwd) | EKF y (left) | EKF z (up) |
|---|---|---|---|
| Thigh | −sensor_x | sensor_z | sensor_y |
| Foot | −sensor_y | sensor_x | sensor_z |

Both are proper rotations (axis relabel + sign, det = +1).

**Verify against hardware before trusting this table.** It documents the *intent* from the Simulink annotations; the exact Selector index mapping in the block was not byte-verified. Ground truth is the gravity check: hold each IMU still and confirm the post-remap EKF **z** reads ≈ +9.8 m/s² (support-force convention — the up-axis reads +g, not −g).

## EtherCAT

- Chain order (SOEM slave indices): **Knee Elmo (0) → Ankle Elmo (1) → Medulla (2)**, 1 ms cycle, DC SYNC0 = 1 ms.
- Source of truth for all layouts below: `Hardware/knee_ankle_prosthesis.xml` (ENI).
- Elmo mode: startup SDO `0x6060 = 0x0A` (CST, the ENI default); runtime mode is **Cyclic Sync Velocity (CSV, `9`)** commanded each cycle via output PDO byte 20 — velocity command + torque feedforward. `Target Torque`/`Target Position` are unused.

### Input PDOs (slave → master) → `RawEtherCATFrame`

#### Knee / Ankle Elmo — 14 bytes (`0x1a00 / 0x1a13 / 0x1a1d`)
| Byte | struct | Object | Field |
|---|---|---|---|
| 0 | `<i` | `0x6064` Position actual value | `*_motor_counts` |
| 4 | `<I` | `0x60fd` Digital Inputs | `*_digital_inputs` |
| 8 | `<H` | `0x6041` Status word (CiA-402; `0x0F` = enabled) | `*_status_word` |
| 10 | `<h` | `0x6077` Torque actual value | `*_motor_torque_raw` |
| 12 | `<h` | `0x2205:01` Analog Input 1 (PT1000 temp) | `*_analog` |

#### Medulla 1.5 — 23 bytes (`0x1a00..0x1a05`)
| Byte | struct | Object | Field |
|---|---|---|---|
| 0 | `<B` | `0x0005:02` ID | `medulla_id` |
| 1 | `<B` | `0x0005:03` State | `medulla_state` |
| 2 | `<B` | `0x0005:04` Counter | `medulla_counter` |
| 3 | `<B` | `0x0005:05` Error Flags | `medulla_error_flags` |
| 4 | `<B` | `0x0005:06` Limit Switch | `medulla_limit_switch` |
| 5 | `<I` | `0x0007:01` Knee Encoder (high-res abs) | `knee_joint_counts` |
| 9 | `<H` | `0x0006:03` Knee Encoder Timestamp | `knee_joint_timestamp` |
| 11 | `<I` | `0x0007:02` Ankle Encoder (17-bit, pre-wrap) | `ankle_joint_counts` |
| 15 | `<H` | `0x0006:04` Ankle Encoder Timestamp | `ankle_joint_timestamp` |
| 17 | `<H` | `0x0006:06` Hall 0 (toe GRF) | `hall_0` |
| 19 | `<H` | `0x0006:07` Hall 1 (heel GRF) | `hall_1` |
| 21 | `<H` | `0x0006:08` Logic Voltage | `medulla_logic_voltage` |

### Output PDOs (master → slave) → `DriveCommand`

#### Knee / Ankle Elmo — 24 bytes (`0x1606 / 0x160b / 0x160c`)
| Byte | struct | Object | Use |
|---|---|---|---|
| 0 | `<i` | `0x607a` Target Position | unused |
| 4 | `<I` | `0x60fe:01` Digital Outputs | unused |
| 8 | `<i` | `0x60ff` Target Velocity | **velocity command** (counts/s) |
| 12 | `<i` | `0x60b1` Velocity Offset | unused |
| 16 | `<h` | `0x60b2` Torque Offset | **torque feedforward** (per-mille of rated) |
| 18 | `<H` | `0x6040` Control word | CiA-402 enable sequence |
| 20 | `<b` | `0x6060` Mode of operation | `0x09` (CSV) |
| 21 | — | pad | — |
| 22 | `<h` | `0x6071` Target Torque | unused |

#### Medulla — 3 bytes (`0x1600`)
| Byte | struct | Object | Use |
|---|---|---|---|
| 0 | `<B` | `0x0005:01` Command | — |
| 1 | `<H` | `0x0006:01` Counter | — |

### Drive output scaling (from Simulink `To {Knee,Ankle} Elmo Drive`)
- Target Velocity = `commanded_velocity_rad_s / motor_abs_counts_to_rad` (rad/s → counts/s), int32.
- Torque Offset  = `1000 * torque_ff_Nm / (rated_current * torque_constant * gear_ratio)` (Nm → per-mille rated), int16.
