# CMU Leg Control

## Installation

```bash
git clone https://github.com/vyvas33/cmu_leg.git
cd cmu_leg
# if uv is not installed on your computer
curl -LsSf https://astral.sh/uv/install.sh | sh 
uv sync
```

## Running 
### Connecting

- Connect the Jetson (and the treadmill PC, if using the treadmill) to the mobile hotspot.
- To sync outputs to your laptop run: 
```bash 
    rsync -avz cmuleg@172.20.10.4:~/cmu_leg/output/ ./output/ 
```

### Control

- In `src/cmu_leg/config/params.py`, set `RUNTIME.arm_motion = True` to walk with the leg on. With it `False` the drives enable and hold, but nothing moves, so you can put on the leg.
- From the repo root (needs sudo for EtherCAT and real-time priority):
```bash
    sudo .venv/bin/python -m cmu_leg.runtime.launch
```
- The leg stands once the drives are enabled. `RUNTIME.walk_after_standing_s` (3 s after standing) then arms walking, which begins as soon as the foot is unloaded. Ctrl+C to stop. Logs go to `output/`.
- To drive the Bertec belts from the run itself, see the treadmill section of `src/cmu_leg/runtime/README.md`.

### Treadmill

- On the treadmill PC, open the Treadmill Control Panel. In Settings, enable Remote TCP/IP Control and uncheck "Listen on 127.0.0.1 Only". Hit "Ready for Remote Control" on the dashboard.
- On the Jetson:
```bash
    uv run python -m cmu_leg.runtime.treadmill --host <PC_IP> --speed 0.8 --acc 0.5
```
- The belts start after 10 s. Ctrl+C to stop.
