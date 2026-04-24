"""Dataset recording script for UR10 + GELLO teleoperation.

Usage:
    python scripts/record.py \
        --repo-id rudra-8000/ur10_pick_place_v1 \
        --single-task "Pick the cube and place it in the box" \
        --num-episodes 50 \
        --episode-time-s 30 \
        --reset-time-s 15

Keyboard controls during recording:
    Right arrow  →  exit current episode early (saves it)
    Left arrow   ←  discard and re-record episode
    Escape       →  stop recording entirely
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
# from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.feature_utils import combine_feature_dicts
from lerobot.datasets.feature_utils import build_dataset_frame

from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.processor import RobotObservation, make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
# from lerobot.utils.robot_utils import busy_wait
# from lerobot.utils.control_utils import busy_wait
import sys

from lerobot.utils.utils import init_logging, log_say

from lerobot.robots.lerobot_robot_ur10 import UR10Config


# from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.lerobot_teleoperator_gello import GelloConfig
logger = logging.getLogger(__name__)



from ur10_teleoperate import default_ur_home_action, smooth_move_to_home  # noqa: E402


ROBOT_IP   = "192.168.100.3"
GELLO_PORT = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAO528D-if00-port0"


@safe_stop_image_writer
def record_loop(
    robot, teleop, dataset, events, fps,
    teleop_action_processor, robot_action_processor, robot_observation_processor,
    control_time_s, single_task=None, save_frames=True,
):
    timestamp = 0.0
    start_t = time.perf_counter()

    frame_count = 0
    _fps_window_start = time.perf_counter()
    _fps_window_frames = 0
    _fps_log_interval = 1.0

    while timestamp < control_time_s:
        loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)

        raw_action = teleop.get_action()
        teleop_action = teleop_action_processor((raw_action, obs_processed))
        robot_action = robot_action_processor((teleop_action, obs_processed))
        robot.send_action(robot_action)

        if save_frames and dataset is not None:
            obs_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
            act_frame = build_dataset_frame(dataset.features, teleop_action, prefix=ACTION)
            dataset.add_frame({**obs_frame, **act_frame, "task": single_task})

        # Always count frames for FPS reporting (even during reset/no-save loops)
        frame_count += 1
        _fps_window_frames += 1

        now = time.perf_counter()
        elapsed_window = now - _fps_window_start
        if elapsed_window >= _fps_log_interval:
            achieved_fps = _fps_window_frames / elapsed_window
            total_elapsed = now - start_t
            logger.info(
                f"  FPS: {achieved_fps:.1f} (target {fps}) | "
                f"frames: {frame_count} | elapsed: {total_elapsed:.1f}s"
            )
            _fps_window_start = now
            _fps_window_frames = 0

        dt = time.perf_counter() - loop_t
        if dt < 1 / fps:
            # busy_wait(1 / fps - dt)
            remaining = 1 / fps - dt
            if remaining > 0:
                time.sleep(remaining)
        elif dt > 2.5 / fps:
            logger.warning(f"Slow loop: {dt:.3f}s (target {1/fps:.3f}s)")

        timestamp = time.perf_counter() - start_t
    total_s = time.perf_counter() - start_t
    if total_s > 0 and frame_count > 0:
        avg_fps = frame_count / total_s
        logger.info(
            f"  Loop complete — avg FPS: {avg_fps:.1f} | "
            f"total frames: {frame_count} | duration: {total_s:.1f}s"
        )


def main(
    repo_id="rudra-8000/ur10_demo_v1",
    single_task="Pick the cube and place it in the box",
    root=None,
    fps=30,
    episode_time_s=30.0,
    reset_time_s=15.0,
    num_episodes=50,
    video=True,
    push_to_hub=False,
    private=True,
    num_image_writer_processes=0,
    num_image_writer_threads_per_camera=4,
    video_encoding_batch_size=20,
    resume=False,
):
    init_logging()

    robot_cfg  = UR10Config(ip=ROBOT_IP)
    teleop_cfg = GelloConfig(port=GELLO_PORT, id="gello_teleop")

    robot  = make_robot_from_config(robot_cfg)
    teleop = make_teleoperator_from_config(teleop_cfg)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=video,
        ),
    )

    num_threads = num_image_writer_threads_per_camera * len(robot_cfg.cameras)

    if resume:
        dataset = LeRobotDataset(repo_id, root=root, batch_encoding_size=video_encoding_batch_size)
        dataset.start_image_writer(num_processes=num_image_writer_processes, num_threads=num_threads)
        sanity_check_dataset_robot_compatibility(dataset, robot, fps, dataset_features)
    else:
        sanity_check_dataset_name(repo_id, policy_cfg=None)
        # sanity_check_dataset_name(repo_id)
        dataset = LeRobotDataset.create(
            repo_id, fps, root=root, robot_type=robot.name,
            features=dataset_features, use_videos=video,
            image_writer_processes=num_image_writer_processes,
            image_writer_threads=num_threads,
            batch_encoding_size=video_encoding_batch_size,
        )

    robot.connect()
    teleop.connect()  # triggers calibration prompt if no calibration file

    listener, events = init_keyboard_listener()
    logger.info("\nControls:\n  → right arrow : finish episode early\n  ← left arrow  : re-record episode\n  Esc           : stop\n")

    try:
        with VideoEncodingManager(dataset):
            recorded = 0
            
            while recorded < num_episodes and not events["stop_recording"]:
                logger.info(f"Target FPS: {fps} | FPS reported every 5s")
                logger.info(f"--- Episode {dataset.num_episodes + 1} / {num_episodes} ---")

                record_loop(
                    robot=robot, teleop=teleop, dataset=dataset, events=events,
                    fps=fps, teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    control_time_s=episode_time_s, single_task=single_task, save_frames=True,
                )

                if events["rerecord_episode"]:
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    logger.info("Episode discarded, re-recording...")
                    continue

                if events["stop_recording"]:
                    break

                dataset.save_episode()
                recorded += 1
                logger.info(f"Episode {recorded} saved.")

                if recorded < num_episodes and not events["stop_recording"]:
                    logger.info(f"Reset environment ({reset_time_s}s)...")
                    record_loop(
                        robot=robot, teleop=teleop, dataset=dataset, events=events,
                        fps=fps, teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        control_time_s=reset_time_s, single_task=single_task,
                        save_frames=False,  # ← don't record during reset
                    )
    finally:
        robot.disconnect()
        teleop.disconnect()
        if not is_headless() and listener is not None:
            listener.stop()

    if push_to_hub:
        dataset.push_to_hub(private=private)

    logger.info(f"Done. {recorded} episodes at {dataset.root}")
    return dataset


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id",        default="rudra-8000/ur10_demo_v1")
    p.add_argument("--single-task",    default="Pick the cube and place it in the box")
    p.add_argument("--root",           default=None)
    p.add_argument("--fps",            type=int,   default=30)
    p.add_argument("--episode-time-s", type=float, default=60.0)
    p.add_argument("--reset-time-s",   type=float, default=15.0)
    p.add_argument("--num-episodes",   type=int,   default=5)
    p.add_argument("--no-video",       action="store_false", dest="video")
    p.add_argument("--push-to-hub",    action="store_true")
    p.add_argument("--resume",         action="store_true")
    p.add_argument("--batch-encode",   type=int, default=20, dest="video_encoding_batch_size")
    args = p.parse_args()
    main(**{k: v for k, v in vars(args).items()})
