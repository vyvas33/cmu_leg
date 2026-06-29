# Low Level Control

Low level control is the layer between a desired joint torque and the numbers the Elmo drives actually take. The high level controller decides how much torque it wants at the knee and the ankle; this layer turns each of those into a motor velocity command and a torque feedforward, keeps both inside the limits the hardware can survive, and hands them off in the units the drives consume. It runs once per control tick at 1 kHz, and it is deliberately dumb about intent: it never decides *what* the leg should do, only how to ask the motors for it safely.

Each joint is a three-stage pipeline — conditioning, then the torque controller, then saturation — and the knee and ankle each get their own copy because the two joints are not symmetric. The stages live in `knee.py` and `ankle.py`, one class per stage, and `low_level_control.py` one level up is the orchestrator that runs them in order.

## The orchestrator

`LowLevelControl` owns the pipeline and everything around it that has to happen once per tick rather than inside a single stage. It watches the enable signal, and on the edge where control turns on it resets every stage so no filter or derivative carries stale history into the first live command. While control is off it does not run the pipeline at all and emits zero on every drive-facing field; this is stricter than the bench controller, which holds its last output and trusts the drive state machine to ignore it, but holding zero is the safer default for a fresh port. For the first second after enable it keeps advancing the pipeline — letting the filters settle and the derivatives build history — while still forcing the drive outputs to zero, so the leg does not lurch the instant it is switched on.

The last thing it does is convert units. The stages all speak SI — radians per second and newton-metres — and the drives speak encoder counts per second and per-mille of rated torque, so the orchestrator applies the per-joint scaling and produces the `DriveCommand` the EtherCAT interface writes. Everything upstream of that conversion stays in physical units, which keeps the stages readable and testable without thinking about the bus.

## Conditioning

Conditioning takes the raw desired torque and makes it safe to ask for near the ends of travel. Each joint has soft limits short of its hard limits, and as the joint pushes past a soft limit a backstop torque ramps in to oppose the motion — stiffer the further it goes, and stronger the faster it is moving into the stop, so the leg is eased away from its mechanical ends rather than slammed there. There is a backstop for flexion and one for extension, and they are mirror images: flexion engages at the upper soft limit and pushes back, extension engages at the lower soft limit and pushes the other way.

This is the stage where the two joints differ most. The knee sums its backstop torques into the desired torque and then clamps the result, adds a small velocity-proportional damping term for stability, and checks its flexion stop against the joint angle while checking extension against the motor angle. It also suppresses the extension backstop while the leg is standing, since standing deliberately holds the knee near its extension limit and the backstop would fight it. The ankle is simpler: it clamps the bare desired torque against backstop-derived bounds without summing them in, has no damping term, and checks both stops against the motor angle through a single filtered motor velocity. Both joints run the conditioned torque through a lowpass before passing it on, so the clamp's corners do not show up as steps downstream.

## Torque controller

The torque controller is the inverse of the actuator: given the torque we want at the joint, it works out the motor velocity to command and the torque to feed forward. The velocity command is the velocity the joint is already moving at, plus a term that accounts for how fast the commanded torque is changing through the transmission stiffness, plus a proportional correction on the error between commanded and measured torque. For the knee the "velocity the joint is moving at" is the knee velocity; for the ankle it is the filtered spring velocity.

The feedforward is the commanded torque plus the motor's own dynamics — what it costs to accelerate the rotor and overcome its damping as the commanded torque changes — expressed through the transmission. The knee divides this by the drivetrain efficiency; the ankle does not, because its efficiency compensation is left off. A few terms that exist in the bench design carry zero gain and are simply dropped here, which is why the controller is shorter than its block diagram.

## Saturation

Saturation is the last line before the drive and it does not reason, it only clips. Each joint has hard limits, and the moment the joint is past one the command in that direction is forced to zero — both the velocity and the torque feedforward — so the controller can never drive further into a stop it has already hit. Within the limits the commands pass through untouched. As with conditioning, the knee checks flexion against the joint angle and extension against the motor angle, while the ankle checks both against the motor angle. The clipped signals then go through a smoothing lowpass so the zeroing happens cleanly rather than as a hard discontinuity arriving at the motor.

## What it doesn't do, and the safety gate

This layer takes an estimated state and a desired torque and returns a `LowLevelOutput`; it does not estimate anything, does not decide modes, and does not touch the bus. State comes from the estimator, the desired torque from the high level controller, and the conversion to a wire-ready `DriveCommand` is the orchestrator's final step.

Every stage carries a `_verified` flag that defaults to false and raises if its `compute` is called. This is a hardware safety interlock, not a code-quality marker: it stays false until the stage has been checked against the bench behaviour and someone has decided it is safe to command a real motor, and flipping it is a deliberate human act at the bench, not something the code does on its own. The same edge that resets the stages is where these gates matter most, since that is the moment the leg first moves under its own command.
