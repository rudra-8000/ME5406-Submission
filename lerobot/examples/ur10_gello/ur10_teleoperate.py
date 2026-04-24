"""Direct teleoperation script for UR10 with GELLO."""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import numpy as np

from lerobot.processor import RobotObservation, make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.utils import init_logging

# Importing these registers them automatically via decorators
from lerobot.teleoperators.lerobot_teleoperator_gello import GelloConfig
from lerobot.robots.lerobot_robot_ur10 import UR10Config


def default_ur_home_action() -> dict[str, float]:
    """Same preset as gello_software/experiments/launch_nodes.py (_default_ur_home_rad for 6-DOF arm).

    Degrees: (0, -90, 90, -90, -90, 0); gripper open (normalized 0 = open in UR10 driver).
    """
    arm_rad = np.deg2rad([0.0, -90.0, 90.0, -90.0, -90.0, 90.0])
    action: dict[str, float] = {f"joint_{i}": float(arm_rad[i]) for i in range(6)}
    action["gripper"] = 0.0
    return action


def smooth_move_to_home(
    robot: Any,
    home_action: dict[str, float],
    *,
    max_step_rad: float = 0.01,
    max_steps: int = 100,
    min_steps: int = 2,
    step_sleep_s: float = 0.002,
    settle_s: float = 1.0,
) -> None:
    """Interpolate from current pose to home using repeated send_action (mirrors launch_nodes._smooth_move_to_home)."""
    obs = robot.get_observation()
    curr = np.array(
        [float(obs[f"joint_{i}"]) for i in range(6)] + [float(obs["gripper"])],
        dtype=np.float64,
    )
    home_vec = np.array(
        [home_action[f"joint_{i}"] for i in range(6)] + [float(home_action["gripper"])],
        dtype=np.float64,
    )
    max_delta = float(np.abs(curr - home_vec).max())
    steps = min(int(max_delta / max_step_rad), max_steps)
    steps = max(steps, min_steps)
    logging.info("Moving arm to home (%d interpolation steps)...", steps)
    for jnt_g in np.linspace(curr, home_vec, steps):
        step_action: dict[str, float] = {f"joint_{i}": float(jnt_g[i]) for i in range(6)}
        step_action["gripper"] = float(jnt_g[6])
        robot.send_action(step_action)
        time.sleep(step_sleep_s)
    time.sleep(settle_s)
    logging.info("Home pose reached.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UR10 + GELLO teleoperation")
    p.add_argument("--ur-ip", type=str, default="172.17.0.2", help="UR controller IP (RTDE)")
    p.add_argument(
        "--teleop-port",
        type=str,
        default="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAO528D-if00-port0",
    )
    p.add_argument("--teleop-id", type=str, default="gello_teleop")
    p.add_argument(
        "--skip-move-home",
        action="store_true",
        help="Do not move to the gello default home before teleoperation",
    )
    p.add_argument("--loop-hz", type=float, default=50.0, help="Main teleop loop rate")
    return p.parse_args()


def main() -> None:
    init_logging()
    args = parse_args()
    logging.info("Starting UR10 <-> GELLO teleoperation")

    robot_cfg = UR10Config(ip=args.ur_ip.strip())
    teleop_cfg = GelloConfig(port=args.teleop_port, id=args.teleop_id)

    teleop = make_teleoperator_from_config(teleop_cfg)
    robot = make_robot_from_config(robot_cfg)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    teleop.connect()
    try:
        robot.connect()
    except Exception as exc:
        logging.warning("Robot connection failed: %s", exc)

    if robot.is_connected and not args.skip_move_home:
        try:
            smooth_move_to_home(robot, default_ur_home_action())
        except Exception as exc:
            logging.exception("move-to-home failed: %s", exc)

    loop_hz = max(args.loop_hz, 1.0)
    loop_period = 1.0 / loop_hz

    try:
        while True:
            loop_start = time.perf_counter()

            obs: RobotObservation = {}
            if robot.is_connected:
                try:
                    obs = robot.get_observation()
                except DeviceNotConnectedError:
                    logging.warning("Robot disconnected while reading observation")
                    obs = {}

            raw_action = teleop.get_action()
            teleop_action = teleop_action_processor((raw_action, obs))
            robot_action = robot_action_processor((teleop_action, obs))

            if robot.is_connected:
                robot.send_action(robot_action)

            elapsed = time.perf_counter() - loop_start
            if elapsed < loop_period:
                time.sleep(loop_period - elapsed)

    except KeyboardInterrupt:
        logging.info("Teleoperation interrupted by user")
    finally:
        teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
