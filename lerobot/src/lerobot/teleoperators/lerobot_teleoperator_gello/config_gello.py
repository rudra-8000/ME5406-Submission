# """Configuration dataclass for the GELLO teleoperator plugin.

# Defines GelloConfig with serial port settings, calibration position, joint signs,
# and optional smoothing/async parameters.
# """

# from dataclasses import dataclass, field

# from lerobot.teleoperators.config import TeleoperatorConfig

# @TeleoperatorConfig.register_subclass("gello")
# @dataclass
# class GelloConfig(TeleoperatorConfig):
#     # Port to connect to the arm
#     port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAO528D-if00-port0"
#     baudrate: int = 57_600
#     calibration_position: list[float] = field(default_factory=lambda: [0, -1.57, 1.57, -1.57, -1.57, -1.57])
#     joint_signs: list[int] = field(default_factory=lambda: [1, 1, -1, 1, 1, 1])
#     gripper_travel_counts: int = 575

#     # Smoothing factor for Exponential Moving Average (EMA).
#     # Range [0, 1]. 1 means no smoothing (instant update), 0 means no update (freeze).
#     # Lower values smooth out jitter but add latency.
#     smoothing: float = 0.85
#     # Whether to run device reading in a background thread.
#     # This helps when USB communication is slow (e.g. long cables).
#     use_async: bool = True


from dataclasses import dataclass, field
from lerobot.teleoperators.config import TeleoperatorConfig
import numpy as np

@TeleoperatorConfig.register_subclass("gello")
@dataclass
class GelloConfig(TeleoperatorConfig):
    port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAO528D-if00-port0"
    baudrate: int = 1_000_000

    # Joint IDs 1-6 for arm, 7 for gripper
    joint_ids: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])

    # From gello_agent.py — validated on hardware
    joint_offsets: list[float] = field(
        default_factory=lambda: [
            2*np.pi/2, 2*np.pi/2, 2*np.pi/2, 2*np.pi/2, 2*np.pi/2, 1*np.pi/2
        ]
    )

    # ← fixed to match gello_agent.py
    joint_signs: list[int] = field(default_factory=lambda: [1, 1, -1, 1, 1, 1])

    # Gripper Dynamixel ID=7, open=192 ticks, close=150 ticks
    gripper_id: int = 7
    gripper_open: int = 192
    gripper_close: int = 150
    gripper_travel_counts: int = 575

    calibration_position: list[float] = field(
        default_factory=lambda: [0, -1.57, 1.57, -1.57, -1.57, 1.57]
    )

    smoothing: float = 0.85
    use_async: bool = True
