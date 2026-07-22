"""
Custom WitMotion HWT906 Python Driver Package.

This package provides a Python interface for the WitMotion HWT906
Inertial Measurement Unit (IMU). It handles hardware configuration, asynchronous
data streaming, and software PLL-based timestamping.

Key Files:
- hwt906.py: The core driver containing the WitMotionSensor class.
- settings.py: Script for configuring sensor parameters and persisting them to flash.
- recovery.py: Utility to auto-discover baud rates and recover "lost" sensors.
- example.py: Starter templates for real-time polling and gapless logging.
- README.md: Documentation of the driver and sensor architecture.

To get started, see README.md and the example.py script.

hgeyer, Jun 2026
"""

from .hwt906 import WitMotionSensor, discover_baud, discover_port
