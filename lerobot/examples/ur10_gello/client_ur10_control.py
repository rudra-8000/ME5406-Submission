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
  python examples/ur10_gello/client_ur10_control.py \
    --server-host 10.245.91.19 --server-port 8765 \
    --ur-ip 192.168.100.3


python examples/ur10_gello/client_ur10_control.py \
--server-host 10.245.91.19 --server-port 8765 \
--control-dt 0.033 \
--max-steps 300 \
--num-rollouts 2 \
--video-dir /path/to/videos \
--reset-between-rollouts \
--reset-policy


python examples/ur10_gello/client_ur10_control.py \
--server-host 10.245.91.19 --server-port 8765 \
--control-dt 0.033 \
--max-steps 300 \
--num-rollouts 2 \
--video-dir /path/to/videos \
--reset-between-rollouts \
--reset-policy

Optional: ``--control-dt`` (seconds per step), ``--max-steps`` / ``--num-rollouts`` for bounded tests,
``--video-dir`` to save side-by-side camera mp4s per rollout, ``--reset-between-rollouts`` for ACT-style resets.

Observation keys must match the dataset used for training (same layout as record.py / robot.get_observation):
state components as joint_0..joint_5, gripper; images under camera names matching dataset (e.g. cam_high).
The server builds observation.state and observation.images.* via dataset metadata.
"""

from __future__ import annotations
import time
import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np
import sys
from lerobot.cameras import make_cameras_from_configs
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot.robots.lerobot_robot_ur10 import UR10Config
from lerobot.utils.utils import init_logging


# Same directory as this script (for sibling import of ur10_teleoperate).
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ur10_teleoperate import default_ur_home_action, smooth_move_to_home  # noqa: E402


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


def _control_dt_seconds(args: argparse.Namespace) -> float:
    """Duration of one control iteration (sleep target after each step)."""
    if args.control_dt is not None:
        return float(args.control_dt)
    return 1.0 / max(args.fps, 1)


def _composite_rgb_frame(raw_obs: dict[str, Any], camera_names: list[str]) -> np.ndarray | None:
    """Horizontally stack camera RGB images (sorted by name)."""
    try:
        import cv2
    except ImportError as e:
        raise ImportError("Saving video requires opencv: pip install opencv-python-headless") from e

    pieces: list[np.ndarray] = []
    for name in camera_names:
        img = raw_obs.get(name)
        if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
            continue
        pieces.append(np.ascontiguousarray(img))
    if not pieces:
        return None
    target_h = max(p.shape[0] for p in pieces)
    resized: list[np.ndarray] = []
    for p in pieces:
        if p.shape[0] != target_h:
            scale = target_h / p.shape[0]
            w = max(1, int(round(p.shape[1] * scale)))
            p = cv2.resize(p, (w, target_h), interpolation=cv2.INTER_AREA)
        resized.append(p)
    return np.concatenate(resized, axis=1)


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
    camera_names = sorted(robot.cameras.keys())
    logging.info("Robot connected; cameras: %s", camera_names)

    video_dir_str = (args.video_dir or "").strip()
    video_dir: Path | None = Path(video_dir_str).expanduser().resolve() if video_dir_str else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Videos will be saved under %s", video_dir)

    num_rollouts = max(1, args.num_rollouts)
    if num_rollouts > 1 and args.max_steps <= 0:
        raise ValueError("--num-rollouts > 1 requires --max-steps > 0 (each rollout must end)")

    control_dt = _control_dt_seconds(args)
    logging.info("Control timestep: %.4f s (~%.2f Hz)", control_dt, 1.0 / control_dt)

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

            loop = asyncio.get_running_loop()
            writer: Any = None
            video_path: Path | None = None

            for rollout_idx in range(num_rollouts):
                home_action = default_ur_home_action()
                smooth_move_to_home(robot, home_action)
                time.sleep(5.0)  # let things settle
                if rollout_idx > 0 and args.reset_between_rollouts:
                    await websocket.send(packer.pack({"__ctrl__": "reset"}))
                    ack = msgpack_numpy.unpackb(await websocket.recv())
                    logging.info("Rollout %d: policy reset ack: %s", rollout_idx, ack)

                if video_dir is not None:
                    video_path = video_dir / f"rollout_{rollout_idx:04d}.mp4"
                    writer = None

                step = 0
                logging.info("Starting rollout %d / %d", rollout_idx + 1, num_rollouts)

                while True:
                    if args.max_steps > 0 and step >= args.max_steps:
                        break

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

                    if video_dir is not None and video_path is not None:
                        import cv2

                        composite = _composite_rgb_frame(raw_obs, camera_names)
                        if composite is not None:
                            if writer is None:
                                h, w = composite.shape[:2]
                                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                                fps_out = min(120.0, max(1.0, 1.0 / control_dt))
                                writer = cv2.VideoWriter(str(video_path), fourcc, fps_out, (w, h))
                                if not writer.isOpened():
                                    raise RuntimeError(f"Failed to open VideoWriter for {video_path}")
                                logging.info("Recording %s at %dx%d %.2f fps", video_path.name, w, h, fps_out)
                            bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
                            writer.write(bgr)

                    await asyncio.to_thread(robot.send_action, action)

                    step += 1
                    if args.log_every > 0 and step % args.log_every == 0:
                        st = resp.get("server_timing") or {}
                        logging.info(
                            "rollout=%d step=%d infer_ms=%.2f prev_total_ms=%s",
                            rollout_idx,
                            step,
                            st.get("infer_ms", -1.0),
                            st.get("prev_total_ms", "n/a"),
                        )

                    elapsed = loop.time() - t0
                    sleep_time = control_dt - elapsed
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)

                if writer is not None:
                    writer.release()
                    writer = None
                    logging.info("Finished rollout %d (%d steps); video: %s", rollout_idx + 1, step, video_path)
                elif video_dir is not None and video_path is not None:
                    logging.warning(
                        "Rollout %d: no video written (no valid camera frames in observations)",
                        rollout_idx + 1,
                    )

    finally:
        if getattr(robot, "is_connected", False):
            robot.disconnect()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UR10 WebSocket client for remote policy inference.")
    p.add_argument("--server-host", type=str, default="10.245.91.19", help="Policy server hostname or IP")
    p.add_argument("--server-port", type=int, default=8765, help="Policy server WebSocket port")
    p.add_argument("--ur-ip", type=str, default="192.168.100.3", help="UR10 controller IP (RTDE)")
    p.add_argument(
        "--gripper-port",
        type=str,
        default="",
        help="Override gripper serial device (default: UR10Config default)",
    )
    p.add_argument("--fps", type=int, default=30, help="Nominal control rate (Hz); used when --control-dt is unset")
    p.add_argument(
        "--control-dt",
        type=float,
        default=None,
        help="Seconds per control step (timestep). If unset, uses 1/--fps",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop each rollout after this many control steps (0 = run until Ctrl+C)",
    )
    p.add_argument(
        "--num-rollouts",
        type=int,
        default=1,
        help="Number of test rollouts to run (requires --max-steps > 0 when > 1)",
    )
    p.add_argument(
        "--video-dir",
        type=str,
        default="",
        help="If set, save a side-by-side RGB video per rollout under this directory (mp4)",
    )
    p.add_argument(
        "--reset-between-rollouts",
        action="store_true",
        help="Send policy reset between rollouts (useful for ACT / temporal policies)",
    )
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
    if args.control_dt is not None and args.control_dt <= 0:
        raise ValueError("--control-dt must be positive")
    try:
        asyncio.run(_control_loop(args))
    except KeyboardInterrupt:
        logging.info("Client stopped by user")


if __name__ == "__main__":
    main()
