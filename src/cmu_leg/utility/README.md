# Utility Functions 

## 1 Read Encoders 
Usage: uv run python -m cmu_leg.utility.read_encoders

This script reads the input PDOs from EtherCAT, specifically the motor and joint encoders, converts them into SI units and plots them. A good chunk of the code involves processing that converts the raw int counts to SI units. This function also includes a polynomial correction term for some readings. Note that this script has to be updated whenever there is any change in how the inputs are processed, as it is self-contained.  

## 2 Read IMU 
Usage: uv run python -m cmu_leg.utility.read_imu

This script reads the IMU stream over USB. The IMU connected ports are pulled from the global config. The script auto discovers the baud rate the sensors are currently configured to and then plots the readings. There is not much processing involved here as the lab's custom witmotion API outputs in SI units directly. 

The reading script only plots and saves fresh IMU data. `imu.read()` returns if a value is fresh. The script oversamples the data collection (2x). It also plots the idle time to observe the performance.


