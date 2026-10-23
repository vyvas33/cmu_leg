"""
Bring-up test for bertec_treadmill.py, run from the Jetson.

  uv run python -m cmu_leg.hardware.treadmill.bertec_test [HOST]           # read-only
  uv run python -m cmu_leg.hardware.treadmill.bertec_test [HOST] --move    # + 0.2 m/s for 3 s

HOST defaults to params.TREADMILL.host.

Stage 1 (read-only) never sends a command. Belts at 0 read 0 under any byte
order, so to actually confirm the status layout, start the belts BY HAND on
the Control Panel (e.g. 0.5 m/s) and check the printout matches.

Only use --move with the belts clear and the E-stop pendant in your hand.
"""
import sys
import time

from cmu_leg.hardware.treadmill.bertec_treadmill import BertecTreadmill


def _show(frame) -> str:
    return (f"right {frame.right_speed:+.3f} m/s | left {frame.left_speed:+.3f} m/s | "
            f"incline {frame.incline:.2f} deg | fresh {frame.fresh}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    host = args[0] if args else None
    do_move = "--move" in sys.argv

    tm = BertecTreadmill(host=host)
    if not tm.host:
        print(__doc__)
        sys.exit(1)

    print(f"Connecting to {tm.host or '<unset>'}:{tm.port} ...")
    with tm:
        print("Connected.")

        # The Control Panel streams status packets; give it a moment to send one.
        frame = tm.read()
        deadline = time.monotonic() + 2.0
        while not frame.fresh and time.monotonic() < deadline:
            time.sleep(0.1)
            frame = tm.read()

        if not frame.fresh:
            print("Connected but no status packet received. Check that the Control Panel's "
                  "TCP/IP remote control is enabled/armed. Not sending anything.")
            return

        print(_show(frame))
        print("These should match the Control Panel display. Belts idle read 0 under any "
              "byte order; run the belts by hand to confirm the layout.")

        if not do_move:
            print("Read-only test done. Re-run with --move for the gentle speed test.")
            return

        if input("Belts clear, E-stop in hand? Type YES to run 0.2 m/s for 3 s: ") != "YES":
            print("Aborted.")
            return

        try:
            tm.set_speeds(0.2, 0.2, acc_r=0.3, acc_l=0.3)
            for _ in range(6):
                time.sleep(0.5)
                print(_show(tm.read()))
        finally:
            tm.stop()
            time.sleep(1.5)
            print("Stopped.", _show(tm.read()))


if __name__ == "__main__":
    main()
