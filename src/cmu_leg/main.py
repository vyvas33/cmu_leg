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
from cmu_leg.estimation.state_estimator import StateEstimator
from cmu_leg.control.high_level_control import ProsthesisControl
from cmu_leg.control.low_level_control import LowLevelControl

# data types
from cmu_leg.types.state import ProsthesisState
from cmu_leg.types.sensors import IMUFrame   # standing is encoder-only; pass an empty frame

from cmu_leg.config.params import COMMON, STANDING

# logging output dir (post-loop dump)
from cmu_leg.utility.utils import make_run_dir
from cmu_leg.utility.recorder import Recorder

PERIOD = COMMON.sample_time
WARM_UP_TIME = 5.0 #secs
ENABLE_TIME = 5.0 #secs
STANDING_DURATION = STANDING.duration + 2.0
SHUTDOWN_TIME = 1.0 #secs

def main():

    # hardware bringup
    ec = EtherCATInterface("eth0")
    ec.connect()

    # control
    se = StateEstimator()
    pc = ProsthesisControl()
    llc = LowLevelControl()
    
    state_log, torque_log, out_log, drive_log, idle_time_log= (
    Recorder("state"), Recorder("torque"), Recorder("lowlevel"), Recorder("drive"), Recorder("idle_time"))

    t0 = time.perf_counter()
    clk = t0
    reached_enabled = False
    frame = ec.read()

    def run(duration: float, standing=False) -> None:
        """Tick at 1 kHz for `duration`"""
        nonlocal clk, reached_enabled, frame
        phase_start = time.perf_counter() - t0
        while (t := time.perf_counter() - t0) - phase_start < duration:
            clk += PERIOD
            t_ctrl = clk - t0   # ideal uniform time, no OS jitter
            state = se.get_state(frame, IMUFrame(), t_ctrl)
            state.standing = standing
            desired_torque = pc.compute(state, t_ctrl)
            llo = llc.compute(state, desired_torque=desired_torque,control_enabled=ec.control_enabled())
            drive_cmd = llc.to_drive_command(llo)
            frame = ec.write(drive_cmd)                       
            idle = clk - time.perf_counter()

            state_log.append(t, state); torque_log.append(t, desired_torque); out_log.append(t, llo); drive_log.append(t, drive_cmd)
    
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
        run(STANDING_DURATION, standing=True)

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
            for lg in (state_log, torque_log, out_log, drive_log, idle_time_log):
                lg.save(run_dir)

        

if __name__ == "__main__":
    main()





