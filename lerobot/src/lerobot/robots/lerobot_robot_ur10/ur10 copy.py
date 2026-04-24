"""UR10 robot interface using RTDE protocol.

Implements the LeRobot Robot interface for Universal Robots UR10 CB3 with
PincOpen gripper (Dynamixel XM430-W350-T). Uses servoJ for smooth real-time
joint control.
"""

import time
from typing import Any, Optional

import numpy as np
import rtde_control
import rtde_receive

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots import Robot
from lerobot.utils.errors import DeviceNotConnectedError

from .config_ur10 import UR10Config
from .pincopen_gripper import GripperController


class UR10(Robot):
    config_class = UR10Config
    name = "ur10"

    def __init__(self, config: UR10Config):
        super().__init__(config)

        # Cameras are injected by example scripts (e.g. recording), so `UR10Config`
        # may not include a `cameras` field.
        cameras_cfg = getattr(config, "cameras", None)
        self.cameras = make_cameras_from_configs(cameras_cfg) if cameras_cfg else {}

        self.robot_ip = config.ip

        self.rtde_ctrl: Optional[rtde_control.RTDEControlInterface] = None
        self.rtde_rec:  Optional[rtde_receive.RTDEReceiveInterface]  = None

        # servoJ — aligned with gello_software/gello/robots/ur.py (lookahead/gain).
        # CB3 controller is 125 Hz: use servoj_t=1/125. For UR e-series streaming at
        # 500 Hz, set servoj_t=1/500 to match gello's dt.
        self.acc = 0.5
        self.speed = 0.5
        self.servoj_t = 1.0 / 500
        self.servoj_lookahead = 0.2
        self.servoj_gain = 100

        # Gripper throttling — non-blocking commands (see set_pos_normalized_non_blocking);
        # match gello URRobot new-gripper defaults (eps / interval).
        self._last_gripper_cmd: float = -1.0
        self._gripper_min_delta: float = 0.002
        self._gripper_min_period_s: float = 0.1  # modified by gu from 0.008 to 0.1

        self._last_gripper_cmd_time: float = 0.0

        # Gripper — connection handled inside GripperController.__init__
        self.gripper = GripperController(
            port=config.gripper_port,
            baud=config.gripper_baud,
            dxl_id=config.gripper_dxl_id,
            open_angle=config.gripper_open_angle,
            close_angle=config.gripper_close_angle,
            default_speed=config.gripper_default_speed,
            default_torque=config.gripper_default_torque,
        )

    # ── Feature descriptors ────────────────────────────────────────────────

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            "joint_0": float,
            "joint_1": float,
            "joint_2": float,
            "joint_3": float,
            "joint_4": float,
            "joint_5": float,
            "gripper": float,
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            name: (cam.height, cam.width, 3)
            for name, cam in self.cameras.items()
        }

    @property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict:
        return self._motors_ft

    # ── Connection lifecycle ───────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return (
            self.rtde_ctrl is not None
            and self.rtde_rec is not None
            and self.rtde_ctrl.isConnected()
            and self.rtde_rec.isConnected()
            and self.gripper._connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return

        # RTDE — raises if robot is unreachable
        self.rtde_ctrl = rtde_control.RTDEControlInterface(self.robot_ip)
        self.rtde_rec  = rtde_receive.RTDEReceiveInterface(self.robot_ip)

        # Gripper already connected in __init__; nothing extra needed here.
        # If you want lazy gripper connection, move GripperController() here.

        for cam in self.cameras.values():
            cam.connect()

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        if self.rtde_ctrl is not None:
            self.rtde_ctrl.servoStop()   # cleanly exit servoJ mode first
            self.rtde_ctrl.disconnect()
            self.rtde_ctrl = None

        if self.rtde_rec is not None:
            self.rtde_rec.disconnect()
            self.rtde_rec = None

        self.gripper.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()

    # ── Calibration (no-op for UR10 — uses factory joint offsets) ─────────

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    # ── Data I/O ───────────────────────────────────────────────────────────

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        joint_positions = self.rtde_rec.getActualQ()   # list of 6 floats, radians
        gripper_pos     = self.gripper.get_pos_normalized()  # [0.0=open, 1.0=closed]

        obs = {f"joint_{i}": float(v) for i, v in enumerate(joint_positions)}
        obs["gripper"] = float(gripper_pos)

        for name, cam in self.cameras.items():
            obs[name] = cam.async_read()

        return obs

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not all(k in self.action_features for k in action):
            raise ValueError(f"Invalid action keys: {set(action) - set(self.action_features)}")

        # ── Arm + gripper (RTDE period sync like gello URRobot.command_joint_state)
        assert self.rtde_ctrl is not None
        t_start = self.rtde_ctrl.initPeriod()

        goal_joints = [action[f"joint_{i}"] for i in range(6)]
        self.rtde_ctrl.servoJ(
            goal_joints,
            self.speed,
            self.acc,
            self.servoj_t,
            self.servoj_lookahead,
            self.servoj_gain,
        )

        gripper_cmd = float(np.clip(action["gripper"], 0.0, 1.0))
        now = time.monotonic()
        delta = abs(gripper_cmd - self._last_gripper_cmd)

        if (
            delta >= self._gripper_min_delta
            and now - self._last_gripper_cmd_time >= self._gripper_min_period_s
        ):
            # self.gripper.set_pos_normalized_non_blocking(gripper_cmd)
            self.gripper.set_pos_normalized_async(gripper_cmd)
            self._last_gripper_cmd = gripper_cmd
            self._last_gripper_cmd_time = now

        self.rtde_ctrl.waitPeriod(t_start)
        return action
