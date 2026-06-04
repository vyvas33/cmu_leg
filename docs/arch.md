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
