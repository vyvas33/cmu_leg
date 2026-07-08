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
### Connecting to Jetson 

- Power on the Jetson
- You should now see a WiFi network named CMULeg. Connect to it. The password is pasted on the Jetson.
- SSH into Jetson on Code/Terminal using this command, password is same as the WiFi.
    ```bash
    ssh -v cmuleg@10.42.0.1
    ```
- You are in
