"""
Main loop for Prosthesis Control

Usage:
    uv run python -m cmu_leg.main
"""

import time
import gc

# hardware boundary
from cmu_leg.hardware.ethercat_io import EtherCATInterface
from cmu_leg.hardware.imu_io import IMUInterface

# control stages (run in order each tick)
from cmu_leg.estimation.state_estimator import ProsthesisStateEstimator
from cmu_leg.control.high_level_control import ProsthesisControl
from cmu_leg.control.low_level_control import LowLevelControl

# data types
from cmu_leg.structures.state import ProsthesisState
from cmu_leg.structures.sensors import IMUFrame   # standing is encoder-only; pass an empty frame
from cmu_leg.structures.commands import DriveCommand, Mode
from cmu_leg.runtime.operator import OperatorInput
from cmu_leg.config.params import COMMON, STANDING

# logging output dir (post-loop dump)
from cmu_leg.utility.utils import make_run_dir
from cmu_leg.utility.recorder import Recorder

PERIOD = COMMON.sample_time
WARM_UP_TIME = 5.0 #secs
ENABLE_TIME = 5.0 #secs
STANDING_DURATION = STANDING.duration + 2.0
SHUTDOWN_TIME = 1.0 #secs

# Ankle disabled for this run
DISABLE_ANKLE = False

def main():

    # hardware bringup
    ec = EtherCATInterface("eth0")
    imu = IMUInterface()
    imu.connect()
    ec.connect()

    # control
    se = ProsthesisStateEstimator()
    pc = ProsthesisControl()
    llc = LowLevelControl()
    
    state_log, torque_log, out_log, drive_log, idle_time_log, frame_log= (
    Recorder("state"), Recorder("torque"), Recorder("lowlevel"), Recorder("drive"), Recorder("idle_time"), Recorder("frame"))

    t0 = time.perf_counter()
    clk = t0
    reached_enabled = False
    frame = ec.read()
    t_prev = - PERIOD # so the first tick's dt is ~ PERIOD
    cycle = 0         # OperatorInput.seq -- must advance or the FSM's staleness
                      # watchdog treats the operator as dead and drops to HOLD

    def run(duration: float, standing=False) -> None:
        """Tick at 1 kHz for `duration`.

        `standing` here is the single-process stand-in for the mode FSM's
        operator input: True asks to stand, False asks for nothing. The FSM
        still gates on control_enabled + enable_dwell, so torque only flows once
        the drives are genuinely up.
        """
        nonlocal clk, reached_enabled, frame, t_prev, cycle
        phase_start = time.perf_counter() - t0
        while (t := time.perf_counter() - t0) - phase_start < duration:
            dt = t - t_prev
            t_prev = t
            cycle += 1
            clk += PERIOD
            imu_frame = imu.read()
            state = se.get_state(frame, imu_frame, t, dt)
            frame_log.append(t, frame)  
            state.control_enabled = ec.control_enabled()
            op = OperatorInput(standing_toggle=standing, hold_toggle=False,
                               seq=cycle)
            desired_torque = pc.compute(state, t, op, dt)
            state.standing = pc.mode is Mode.STANDING
            llo = llc.compute(state, desired_torque=desired_torque,
                              control_enabled=ec.control_enabled(), dt=dt,
                              actuate=pc.mode.actuate)
            # only send non-zero commands when the MODE actuates, else zeros --
            # even a small bias in measured torque can poison the low level.
            if pc.mode.actuate:
                drive_cmd = llc.to_drive_command(llo)
                if DISABLE_ANKLE:
                    drive_cmd.ankle_velocity_counts = 0
                    drive_cmd.ankle_torque_per_mille = 0
            else:
                drive_cmd = DriveCommand()
            frame = ec.write(drive_cmd)
            idle = clk - time.perf_counter()

            state_log.append(t, state); torque_log.append(t, desired_torque); out_log.append(t, llo); drive_log.append(t, drive_cmd); idle_time_log.append(t, idle)
    
            if not reached_enabled and ec.control_enabled():
                reached_enabled = True
                print(f"[{t:6.3f}s] control_enabled")


            if idle > 0:
                time.sleep(idle)

    gc.collect()
    gc.freeze()
    gc.disable()
    try:
        # enable is False
        print("Warming up......")
        run(WARM_UP_TIME)

        # Set enable to true 
        ec.enable = True
        run(ENABLE_TIME)

        if not reached_enabled:
            print("WARNING: failed to reach OPERATION_ENABLED")

        # set standing to True 
        run(STANDING_DURATION, standing=False)

    except KeyboardInterrupt:
        print("terminated early")

    finally:
        gc.enable()
        ec.enable = False
        try:
            run(SHUTDOWN_TIME)
        finally:
            ec.close()
            run_dir = make_run_dir("standing")
            for lg in (state_log, torque_log, out_log, drive_log, idle_time_log, frame_log):
                lg.save(run_dir)

        

if __name__ == "__main__":
    main()





