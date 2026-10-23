"""
Run the Bertec treadmill at a set speed until Ctrl+C, then ramp it to a stop.

  uv run python -m cmu_leg.runtime.treadmill                     # 0.8 m/s, 0.5 m/s^2
  uv run python -m cmu_leg.runtime.treadmill --speed 1.0 --acc 0.3
  uv run python -m cmu_leg.runtime.treadmill --host <PC_IP>      # default params.TREADMILL.host

Both belts run at the same speed; positive = person walking forward. --acc is
used for both the ramp up and the ramp down. Re-run with a new --speed to
change speed: the new command takes over from the current belt speed.

Control Panel setup (Ready for Remote Control, etc.) is in
hardware/treadmill/bertec_treadmill.py. Closing the connection does NOT stop
the belts, so this script always sends a stop on the way out; if it is killed
hard (kill -9, power loss, Wi-Fi drop) the belts keep running. Keep the E-stop
in hand.
"""
import argparse
import time

from cmu_leg.hardware.treadmill.bertec_treadmill import BertecTreadmill

PRINT_PERIOD_S = 0.5
STOP_TIMEOUT_S = 15.0      # longest wait for the belts to read zero after stop
MAX_SPEED = 1.5            # m/s, refuse anything faster (catches typos); raise deliberately


def _show(f) -> None:
    print(f"right {f.right_speed:+.3f} m/s | left {f.left_speed:+.3f} m/s", end="\r")


def main():
    parser = argparse.ArgumentParser(description="Run the treadmill at a set speed until Ctrl+C.")
    parser.add_argument("--speed", type=float, default=0.8, help="belt speed [m/s], both belts")
    parser.add_argument("--acc", type=float, default=0.5, help="accel/decel [m/s^2]")
    parser.add_argument("--host", default=None, help="treadmill PC IP (default params.TREADMILL.host)")
    args = parser.parse_args()
    if abs(args.speed) > MAX_SPEED:
        parser.error(f"|speed| must be <= {MAX_SPEED} m/s (MAX_SPEED in this file)")
    if not 0.0 < args.acc <= 1.5:
        parser.error("acc must be in (0, 1.5] m/s^2")

    tm = BertecTreadmill(host=args.host)
    if not tm.host:
        parser.error("no treadmill host: pass --host or set params.TREADMILL.host")

    try:
        tm.connect()
    except OSError as e:
        parser.exit(1, f"Could not connect to {tm.host}:{tm.port} ({e}). Is the Control Panel "
                       "running with Remote TCP/IP Control enabled?\n")

    try:
        print(f"Connected to {tm.host}:{tm.port}. Ramping to {args.speed:.2f} m/s "
              f"at {args.acc:.2f} m/s^2. Ctrl+C to stop.")
        try:
            tm.set_speeds(args.speed, args.speed, acc_r=args.acc, acc_l=args.acc)
            while True:
                time.sleep(PRINT_PERIOD_S)
                _show(tm.read())
        except KeyboardInterrupt:
            pass
        finally:
            print(f"\nStopping at {args.acc:.2f} m/s^2 ...")
            tm.stop(acc=args.acc)

        # Wait for the belts to actually reach zero, so the operator sees it.
        # A second Ctrl+C here is safe: the stop command has already been sent.
        deadline = time.monotonic() + STOP_TIMEOUT_S
        f = tm.read()
        while time.monotonic() < deadline:
            time.sleep(PRINT_PERIOD_S)
            f = tm.read()
            _show(f)
            if f.fresh and f.right_speed == 0.0 and f.left_speed == 0.0:
                break
        print("\nStopped." if f.right_speed == 0.0 and f.left_speed == 0.0
              else "\nBelts did not read zero in time -- check the Control Panel.")
    finally:
        tm.close()


if __name__ == "__main__":
    main()
