#!/usr/bin/env python
#
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local UR10 client: stream observations to a remote LeRobot policy server and execute returned actions.

Run the server (on the GPU machine), then on the robot workstation:

  pip install websockets msgpack-numpy
  python examples/ur10_gello/client_ur10_control.py \\
    --server-host 10.245.91.19 --server-port 8765

Observation keys must match the dataset used for training (same layout as record.py / robot.get_observation):
state components as joint_0..joint_5, gripper; images under camera names matching dataset (e.g. cam_high).
The server builds observation.state and observation.images.* via dataset metadata.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot.robots.lerobot_robot_ur10 import UR10Config
from lerobot.utils.utils import init_logging


def _default_camera_configs(fps: int) -> dict[str, RealSenseCameraConfig]:
    # Same defaults as examples/ur10_gello/record.py — edit serials to match your hardware.
    return {
        "cam_high": RealSenseCameraConfig(
            serial_number_or_name="204322061013",
            fps=fps,
            width=640,
            height=480,
            color_mode=ColorMode.RGB,
        ),
        "cam_right_wrist": RealSenseCameraConfig(
            serial_number_or_name="923322071837",
            fps=fps,
            width=640,
            height=480,
            color_mode=ColorMode.RGB,
        ),
    }


def _observation_to_msgpack_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert robot observation dict to numpy-friendly values for msgpack_numpy."""
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, np.ndarray):
            arr = np.ascontiguousarray(v)
            if arr.dtype == np.uint8:
                out[k] = arr
            else:
                out[k] = arr.astype(np.float32, copy=False)
        elif isinstance(v, (float, int)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _build_robot(args: argparse.Namespace):
    robot_cfg = UR10Config(ip=args.ur_ip)
    if args.gripper_port:
        robot_cfg.gripper_port = args.gripper_port
    robot = make_robot_from_config(robot_cfg)
    # Same pattern as record.py: cameras are attached after construction.
    robot.cameras = make_cameras_from_configs(_default_camera_configs(args.fps))
    return robot


async def _control_loop(args: argparse.Namespace) -> None:
    try:
        import msgpack_numpy
        from websockets.asyncio.client import connect
    except ImportError as e:
        raise ImportError(
            "Install client deps: pip install websockets msgpack-numpy"
        ) from e

    uri = f"ws://{args.server_host}:{args.server_port}/"
    packer = msgpack_numpy.Packer()
    robot = _build_robot(args)

    robot.connect()
    logging.info("Robot connected; cameras: %s", list(robot.cameras.keys()))

    try:
        async with connect(
            uri,
            max_size=None,
            compression=None,
        ) as websocket:
            meta_raw = await websocket.recv()
            metadata = msgpack_numpy.unpackb(meta_raw)
            logging.info("Server metadata: protocol=%s policy_type=%s", metadata.get("protocol"), metadata.get("policy_type"))
            if metadata.get("protocol") != "lerobot_policy_v1":
                logging.warning("Unexpected protocol %r; continuing anyway.", metadata.get("protocol"))

            if args.reset_policy:
                await websocket.send(packer.pack({"__ctrl__": "reset"}))
                ack = msgpack_numpy.unpackb(await websocket.recv())
                logging.info("Policy reset ack: %s", ack)

            period = 1.0 / max(args.fps, 1)
            step = 0
            loop = asyncio.get_running_loop()
            while True:
                t0 = loop.time()

                raw_obs = await asyncio.to_thread(robot.get_observation)
                payload = _observation_to_msgpack_payload(raw_obs)
                msg: dict[str, Any] = {"observation": payload}
                if args.task:
                    msg["task"] = args.task

                await websocket.send(packer.pack(msg))
                resp_raw = await websocket.recv()
                resp = msgpack_numpy.unpackb(resp_raw)

                if not isinstance(resp, dict):
                    raise RuntimeError(f"Bad response type: {type(resp)}")
                if "error" in resp:
                    raise RuntimeError(f"Server error:\n{resp['error']}")
                if resp.get("ok") and resp.get("reset"):
                    continue

                action = resp.get("action")
                if not isinstance(action, dict):
                    raise RuntimeError(f"Missing or invalid action in response: {resp.keys()}")

                await asyncio.to_thread(robot.send_action, action)

                step += 1
                if args.log_every > 0 and step % args.log_every == 0:
                    st = resp.get("server_timing") or {}
                    logging.info(
                        "step=%d infer_ms=%.2f prev_total_ms=%s",
                        step,
                        st.get("infer_ms", -1.0),
                        st.get("prev_total_ms", "n/a"),
                    )

                elapsed = loop.time() - t0
                if elapsed < period:
                    await asyncio.sleep(period - elapsed)

    finally:
        if getattr(robot, "is_connected", False):
            robot.disconnect()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UR10 WebSocket client for remote policy inference.")
    p.add_argument("--server-host", type=str, default="10.245.91.19", help="Policy server hostname or IP")
    p.add_argument("--server-port", type=int, default=8765, help="Policy server WebSocket port")
    p.add_argument("--ur-ip", type=str, default="172.17.0.2", help="UR10 controller IP (RTDE)")
    p.add_argument(
        "--gripper-port",
        type=str,
        default="",
        help="Override gripper serial device (default: UR10Config default)",
    )
    p.add_argument("--fps", type=int, default=30, help="Control loop rate (Hz)")
    p.add_argument("--task", type=str, default="", help="Optional language task string for the policy")
    p.add_argument(
        "--reset-policy",
        action="store_true",
        help="Send policy reset control once after connecting (clears ACT queues, etc.)",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=30,
        help="Log timing every N steps (0 to disable)",
    )
    return p.parse_args()


def main() -> None:
    init_logging()
    args = parse_args()
    try:
        asyncio.run(_control_loop(args))
    except KeyboardInterrupt:
        logging.info("Client stopped by user")


if __name__ == "__main__":
    main()
