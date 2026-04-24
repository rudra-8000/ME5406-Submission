#!/usr/bin/env python

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
# See the License for the specific permissions and limitations under the License.

"""UR10 + GELLO multi-episode recording (same control flow as examples/aloha/record.py).

Before each episode: move to the gello default home pose (see ur10_teleoperate.py), then close
the follower gripper at home. Then wait until the GELLO leader gripper is squeezed closed to
start teleop recording for that episode.
"""

from __future__ import annotations

import logging
import pickle
import re
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import shutil

from lerobot.cameras import make_cameras_from_configs
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.processor import make_default_processors
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.lerobot_teleoperator_gello import GelloConfig
from lerobot.robots import make_robot_from_config
from lerobot.robots.lerobot_robot_ur10 import UR10Config
from lerobot.utils.control_utils import init_keyboard_listener, sanity_check_dataset_robot_compatibility
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say

# Same directory as this script (for sibling import of ur10_teleoperate).
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ur10_teleoperate import default_ur_home_action, smooth_move_to_home  # noqa: E402

NUM_EPISODES = 1
FPS = 30
EPISODE_TIME_SEC = 60

TASK_DESCRIPTION = "My task description"
HF_REPO_ID = "gnq/ur10_gello"

UR10_IP = "172.17.0.2"  # 172.17.0.2   192.168.100.3
TELEOP_PORT = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAO528D-if00-port0"
TELEOP_ID = "gello_teleop"

DISPLAY_DATA = False
PUSH_TO_HUB = False

# Per-phase timing inside lerobot_record.record_loop (avg ms per frame, logged to root logger).
RECORD_PHASE_TIMINGS = True
RECORD_PHASE_LOG_INTERVAL = 150

NUM_IMAGE_WRITER_THREADS_PER_CAMERA = 4
DATASET_ROOT = "/mnt/robot/guningquan/dataset/testlerobot_tactile"
TACTILE_ENABLED = True
TACTILE_HOST = "127.0.0.1"
TACTILE_PORT = 19090
TACTILE_CONNECT_TIMEOUT_S = 5.0
TACTILE_RECV_TIMEOUT_S = 0.2
# First packet after connect can take long: server runs cold Sensor.create + selectSensorInfo for each sensor.
TACTILE_FIRST_FRAME_TIMEOUT_S = 120.0
TACTILE_REQUIRED = True

# GELLO gripper is normalized to [0, 1] in gello.py (0 = open, 1 = closed). Tune if your hardware
# never reaches 1.0 when squeezed.
GELLO_GRIPPER_START_THRESHOLD = 0.72


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed while receiving tactile data.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def decode_tactile_value(item: dict[str, object]) -> object:
    kind = item["kind"]
    if kind == "none":
        return None
    if kind == "scalar":
        return item["value"]
    if kind == "ndarray":
        dtype = np.dtype(item["dtype"])
        shape = tuple(item["shape"])
        array = np.frombuffer(item["data"], dtype=dtype)
        return array.reshape(shape).copy()
    raise ValueError(f"Unknown tactile payload kind: {kind}")


def normalize_tactile_message(message: dict[str, object]) -> dict[str, dict[str, object]]:
    if "sensors" in message:
        return {
            serial: {name: decode_tactile_value(enc) for name, enc in block["items"].items()}
            for serial, block in message["sensors"].items()
        }
    if "items" in message:
        return {"default": {name: decode_tactile_value(enc) for name, enc in message["items"].items()}}
    raise ValueError("Tactile frame has neither 'sensors' nor 'items'.")


def sanitize_feature_token(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", text)


def flatten_tactile_frame(frame: dict[str, dict[str, object]]) -> dict[str, np.ndarray]:
    flat: dict[str, np.ndarray] = {}
    for serial, items in frame.items():
        serial_token = sanitize_feature_token(serial)
        for name, value in items.items():
            name_token = sanitize_feature_token(name)
            feature_key = f"{OBS_STR}.tactile.{serial_token}.{name_token}"
            if value is None:
                flat[feature_key] = np.array([0.0], dtype=np.float32)
            elif isinstance(value, np.ndarray):
                flat[feature_key] = value.copy()
            elif isinstance(value, np.generic):
                flat[feature_key] = np.array([value.item()], dtype=np.float64)
            elif np.isscalar(value):
                flat[feature_key] = np.array([value], dtype=np.float64)
            else:
                raise TypeError(f"Unsupported tactile value type for {name}: {type(value)}")
    return flat


def tactile_features_from_frame(tactile_frame: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {
        key: {
            "dtype": str(value.dtype),
            "shape": tuple(value.shape),
        }
        for key, value in tactile_frame.items()
    }


class TactileBridgeClient:
    def __init__(self, host: str, port: int, timeout_s: float) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.sock: socket.socket | None = None

    def connect(self, *, timeout_s: float | None = None) -> None:
        timeout = timeout_s if timeout_s is not None else self.timeout_s
        self.sock = socket.create_connection((self.host, self.port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(self.timeout_s)

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def recv_frame(self, *, recv_timeout_s: float | None = None) -> dict[str, dict[str, object]]:
        if self.sock is None:
            raise ConnectionError("Tactile socket is not connected.")
        old_timeout = self.sock.gettimeout()
        if recv_timeout_s is not None:
            self.sock.settimeout(recv_timeout_s)
        try:
            header = recv_exact(self.sock, 4)
            payload_size = struct.unpack("!I", header)[0]
            payload = recv_exact(self.sock, payload_size)
            message = pickle.loads(payload)
            return normalize_tactile_message(message)
        finally:
            self.sock.settimeout(old_timeout)


def poll_tactile_frame(
    tactile_client: TactileBridgeClient | None,
    cached_tactile_frame: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    if tactile_client is None:
        return {}
    if cached_tactile_frame is None:
        raise RuntimeError("Tactile cache is not initialized.")
    try:
        latest = tactile_client.recv_frame()
        return flatten_tactile_frame(latest)
    except (socket.timeout, TimeoutError):
        return cached_tactile_frame
    except (ConnectionError, BrokenPipeError, EOFError, OSError) as exc:
        logging.warning("Tactile stream disconnected, trying reconnect: %s", exc)
        try:
            tactile_client.close()
            tactile_client.connect()
            latest = tactile_client.recv_frame(recv_timeout_s=TACTILE_FIRST_FRAME_TIMEOUT_S)
            return flatten_tactile_frame(latest)
        except Exception as reconnect_exc:
            logging.warning("Tactile reconnect failed, using cached frame: %s", reconnect_exc)
            return cached_tactile_frame
    except Exception as exc:
        logging.warning("Failed to receive tactile frame, using cached frame: %s", exc)
        return cached_tactile_frame


@safe_stop_image_writer
def record_loop(
    robot,
    teleop,
    dataset,
    events,
    fps,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    control_time_s,
    single_task=None,
    save_frames=True,
    tactile_client: TactileBridgeClient | None = None,
    tactile_cache: dict[str, np.ndarray] | None = None,
):
    timestamp = 0.0
    start_t = time.perf_counter()

    frame_count = 0
    fps_window_start = time.perf_counter()
    fps_window_frames = 0
    fps_log_interval = 1.0

    while timestamp < control_time_s:
        loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        tactile_cache = poll_tactile_frame(tactile_client, tactile_cache)

        raw_action = teleop.get_action()
        teleop_action = teleop_action_processor((raw_action, obs_processed))
        robot_action = robot_action_processor((teleop_action, obs_processed))
        robot.send_action(robot_action)

        if save_frames and dataset is not None:
            obs_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
            if tactile_cache:
                obs_frame.update({k: v.copy() for k, v in tactile_cache.items()})
            act_frame = build_dataset_frame(dataset.features, teleop_action, prefix=ACTION)
            dataset.add_frame({**obs_frame, **act_frame, "task": single_task})

        frame_count += 1
        fps_window_frames += 1

        now = time.perf_counter()
        elapsed_window = now - fps_window_start
        if elapsed_window >= fps_log_interval:
            achieved_fps = fps_window_frames / elapsed_window
            total_elapsed = now - start_t
            logging.info(
                "FPS: %.1f (target %d) | frames: %d | elapsed: %.1fs",
                achieved_fps,
                fps,
                frame_count,
                total_elapsed,
            )
            fps_window_start = now
            fps_window_frames = 0

        dt = time.perf_counter() - loop_t
        if dt < 1 / fps:
            remaining = 1 / fps - dt
            if remaining > 0:
                time.sleep(remaining)
        elif dt > 2.5 / fps:
            logging.warning("Slow loop: %.3fs (target %.3fs)", dt, 1 / fps)

        timestamp = time.perf_counter() - start_t

    total_s = time.perf_counter() - start_t
    if total_s > 0 and frame_count > 0:
        avg_fps = frame_count / total_s
        logging.info(
            "Loop complete - avg FPS: %.1f | total frames: %d | duration: %.1fs",
            avg_fps,
            frame_count,
            total_s,
        )


def smooth_close_gripper_at_current_pose(
    robot,
    *,
    target_closed: float = 1.0,
    steps: int = 15,
    step_sleep_s: float = 0.03,
) -> None:
    """Close gripper while holding current arm joints (after arm has reached home)."""
    if not getattr(robot, "is_connected", False):
        return
    obs = robot.get_observation()
    g0 = float(obs["gripper"])
    if abs(g0 - target_closed) < 1e-3:
        return
    for g in np.linspace(g0, target_closed, max(2, steps)):
        obs = robot.get_observation()
        action = {f"joint_{i}": float(obs[f"joint_{i}"]) for i in range(6)}
        action["gripper"] = float(g)
        robot.send_action(action)
        time.sleep(step_sleep_s)


def prep_robot_for_episode(robot) -> None:
    """Move to ur10_teleoperate home joints, then close gripper (recording-ready pose)."""
    if not getattr(robot, "is_connected", False):
        print("Robot not connected; skipping move-to-home.")
        return
    print("\n" + "=" * 60)
    print("PREPARING ROBOT — moving to home (ur10_teleoperate preset)...")
    print("=" * 60)
    # Same arm pose as teleop script; interpolation ends with gripper open, then we close.
    home_arm = default_ur_home_action()
    smooth_move_to_home(robot, home_arm)
    print("Closing gripper at home...")
    smooth_close_gripper_at_current_pose(robot, target_closed=1.0)
    print("Robot at home with gripper closed.")
    print("=" * 60 + "\n")


def wait_for_gello_gripper_to_start_episode(teleop: Any, events: dict[str, Any]) -> None:
    """Block until the GELLO leader gripper reads as closed, or ESC sets stop_recording."""
    print("\nClose the GELLO gripper firmly to start recording this episode (ESC to quit).")
    period = 1.0 / max(FPS, 1)
    while not events["stop_recording"]:
        t0 = time.perf_counter()
        try:
            action = teleop.get_action()
            g = float(action.get("gripper", 0.0))
        except Exception:
            logging.exception("Failed to read GELLO action while waiting for gripper close")
            g = 0.0
        if g >= GELLO_GRIPPER_START_THRESHOLD:
            print("GELLO gripper closed — starting teleoperation.")
            return
        elapsed = time.perf_counter() - t0
        precise_sleep(max(period - elapsed, 0.0))


def main() -> None:
    init_logging()

    robot_cfg = UR10Config(ip=UR10_IP)

    teleop_cfg = GelloConfig(
        port=TELEOP_PORT,
        id=TELEOP_ID,
    )

    robot = make_robot_from_config(robot_cfg)

    # Camera configuration is defined here (not in UR10Config) to keep the
    # robot config reusable for different camera setups.
    robot.cameras = make_cameras_from_configs(
        {
            "cam_high": RealSenseCameraConfig(
                serial_number_or_name="204322061013",
                fps=FPS,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
            ),
            "cam_right_wrist": RealSenseCameraConfig(
                serial_number_or_name="923322071837",
                fps=FPS,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
            ),
        }
    )

    teleop = make_teleoperator_from_config(teleop_cfg)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    tactile_client: TactileBridgeClient | None = None
    tactile_cache: dict[str, np.ndarray] = {}
    tactile_features: dict[str, dict[str, object]] = {}
    if TACTILE_ENABLED:
        tactile_client = TactileBridgeClient(TACTILE_HOST, TACTILE_PORT, TACTILE_RECV_TIMEOUT_S)
        try:
            tactile_client.connect(timeout_s=TACTILE_CONNECT_TIMEOUT_S)
            tactile_initial = tactile_client.recv_frame(recv_timeout_s=TACTILE_FIRST_FRAME_TIMEOUT_S)
            tactile_cache = flatten_tactile_frame(tactile_initial)
            tactile_features = tactile_features_from_frame(tactile_cache)
            logging.info(
                "Connected tactile bridge %s:%d with %d features",
                TACTILE_HOST,
                TACTILE_PORT,
                len(tactile_features),
            )
        except Exception as exc:
            tactile_client.close()
            tactile_client = None
            tactile_cache = {}
            tactile_features = {}
            if TACTILE_REQUIRED:
                raise RuntimeError(
                    f"Failed to connect tactile bridge at {TACTILE_HOST}:{TACTILE_PORT}. "
                    "Start examples/tactile_bridge/tactile_bridge_server.py first."
                ) from exc
            logging.warning("Tactile disabled because bridge connection failed: %s", exc)

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
        tactile_features,
    )

    image_writer_threads = NUM_IMAGE_WRITER_THREADS_PER_CAMERA * len(robot.cameras)

    # Same pattern as examples/aloha/record.py: probe for an existing dataset on disk.
    dataset_path = Path(DATASET_ROOT) if DATASET_ROOT else None
    dataset_exists = False
    existing_episodes = 0
    dataset: LeRobotDataset | None = None

    if dataset_path and dataset_path.exists():
        try:
            dataset = LeRobotDataset(
                repo_id=HF_REPO_ID,
                root=DATASET_ROOT,
                download_videos=False,
            )
            existing_episodes = dataset.num_episodes
            dataset_exists = True
            print(f"\n{'=' * 60}")
            print("EXISTING DATASET FOUND!")
            print(f"{'=' * 60}")
            print(f"Dataset location: {dataset.root}")
            print(f"Already recorded episodes: {existing_episodes}")
            print(f"Total frames: {dataset.num_frames}")
            print(f"{'=' * 60}\n")
        except Exception as e:
            print(f"Warning: Could not load existing dataset: {e}")
            print("Creating new dataset...")
            dataset_exists = False
            dataset = None

    if dataset_exists and dataset is not None:
        dataset.meta.load_metadata()
        existing_episodes = dataset.meta.total_episodes
        existing_keys = set(dataset.features.keys())
        expected_keys = set(dataset_features.keys())
        if existing_keys != expected_keys:
            logging.warning("Existing dataset features mismatch current config; creating a new dataset.")
            dataset_exists = False
            dataset = None
            existing_episodes = 0

    if dataset_exists and dataset is not None:
        print(f"Continuing recording from episode {existing_episodes + 1}")
        if image_writer_threads:
            dataset.start_image_writer(0, image_writer_threads)
        sanity_check_dataset_robot_compatibility(dataset, robot, FPS, dataset_features)
    else:
        try:
            if DATASET_ROOT and Path(DATASET_ROOT).exists():
                shutil.rmtree(DATASET_ROOT)
        except Exception as e:
            print(f"Warning: Could not remove existing dataset: {e} before creating new one")
        dataset = LeRobotDataset.create(
            repo_id=HF_REPO_ID,
            fps=FPS,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=image_writer_threads,
            root=DATASET_ROOT,
        )

    teleop.connect()
    try:
        print("Connecting to robot")
        robot.connect()
    except Exception as exc:
        log_say(f"Robot connection failed: {exc}")

    listener, events = init_keyboard_listener()

    print("\n" + "=" * 60)
    print("RECORDING SETUP COMPLETE")
    print("=" * 60)
    print(f"Episodes to record in this session: {NUM_EPISODES}")
    if dataset_exists:
        print(f"Already recorded episodes: {existing_episodes}")
        print(f"Total episodes after this session: {existing_episodes + NUM_EPISODES}")
    print(f"Episode duration: {EPISODE_TIME_SEC} seconds")
    print(f"Task: {TASK_DESCRIPTION}")
    print("\nKeyboard controls:")
    print("  -> (Right arrow): End current episode early")
    print("  <- (Left arrow): End current episode and re-record it")
    print("  ESC: Stop recording completely")
    print("=" * 60 + "\n")

    recorded_episodes = 0
    try:
        while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:

            prep_robot_for_episode(robot)  

            current_episode_num = existing_episodes + recorded_episodes + 1
            print(f"\n{'=' * 60}")
            print(f"EPISODE {recorded_episodes + 1} of {NUM_EPISODES} (dataset total: {current_episode_num})")
            print(f"{'=' * 60}")
            log_say(f"Recording episode {current_episode_num}")

            events["exit_early"] = False
            wait_for_gello_gripper_to_start_episode(teleop, events)
            if events["stop_recording"]:
                break

            record_loop(
                robot=robot,
                teleop=teleop,
                dataset=dataset,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                save_frames=True,
                tactile_client=tactile_client,
                tactile_cache=tactile_cache,
            )

            if events["rerecord_episode"]:
                log_say("Re-record episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            if not events["stop_recording"]:
                dataset.save_episode()
                recorded_episodes += 1
            else:
                dataset.clear_episode_buffer()
                print("Recording stopped. Current episode discarded.")

    finally:
        log_say("Stop recording")

        if listener is not None:
            listener.stop()

        teleop.disconnect()
        if getattr(robot, "is_connected", False):
            robot.disconnect()
        if tactile_client is not None:
            tactile_client.close()

        if dataset is not None:
            dataset.finalize()
            if PUSH_TO_HUB:
                dataset.push_to_hub()


if __name__ == "__main__":
    main()
