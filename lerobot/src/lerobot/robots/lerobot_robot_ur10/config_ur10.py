"""Configuration dataclass for the UR10 CB3 robot with PincOpen gripper."""

from dataclasses import dataclass, field
from lerobot.cameras.configs import ColorMode, CameraConfig
from lerobot.robots import RobotConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig


@RobotConfig.register_subclass("ur10")
@dataclass
class UR10Config(RobotConfig):
    ip: str = "192.168.100.3"

    # gripper_port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAO4W75-if00-port0"
    gripper_port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAO51RF-if00-port0"
    gripper_baud: int = 1_000_000
    gripper_dxl_id: int = 0
    gripper_open_angle: float  = 285.0
    gripper_close_angle: float = 166.0
    gripper_default_speed: float  = 1.0
    # gripper_default_torque: float = 0.5
    gripper_default_torque: float = 0.2

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "wrist": RealSenseCameraConfig(
                serial_number_or_name="923322071837",  # D435I
                fps=30,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
                warmup_s=2.0,
            ),
            "top": RealSenseCameraConfig(
                serial_number_or_name="204322061013",  # D415
                fps=30,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
                warmup_s=2.0,
            ),
        }
    )
