#!/usr/bin/env python3
"""
serl_client_ur10.py — runs on the robot PC.

Connects to the SERL training server on the GPU machine.
Streams observations, executes actions, handles resets.

Usage:
  python examples/ur10_gello/serl_client_ur10.py \
    --server-host 10.245.91.19 --server-port 8766 \
    --control-dt 0.033 \
    --episode-steps 150

python examples/ur10_gello/serl_client_ur10.py \
    --server-host 10.245.123.173 --server-port 8766 \
    --control-dt 0.033 \
    --episode-steps 150
"""
import argparse
import asyncio
import logging
import time
from pathlib import Path

import numpy as np
import msgpack_numpy
from websockets.asyncio.client import connect

from lerobot.cameras import make_cameras_from_configs
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot.robots.lerobot_robot_ur10 import UR10Config

import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from ur10_teleoperate import default_ur_home_action, smooth_move_to_home


def _default_camera_configs(fps=30):
    return {
        "cam_high": RealSenseCameraConfig(
            serial_number_or_name="204322061013",
            fps=fps, width=640, height=480, color_mode=ColorMode.RGB,
        ),
        "cam_right_wrist": RealSenseCameraConfig(
            serial_number_or_name="923322071837",
            fps=fps, width=640, height=480, color_mode=ColorMode.RGB,
        ),
    }

def _obs_to_payload(raw: dict, robot) -> dict:
    """
    Convert robot.get_observation() dict to the ACT observation format the
    SERL server expects:
      observation.state          → (7,) float32  [joint_0..5, gripper]
      observation.images.cam_high        → (H,W,3) uint8
      observation.images.cam_right_wrist → (H,W,3) uint8
      tcp_pose                   → (3,) float32
    """
    out = {}

    # Joint state vector: [joint_0, joint_1, ..., joint_5, gripper]
    state = np.array(
        [raw[f"joint_{i}"] for i in range(6)] + [raw["gripper"]],
        dtype=np.float32,
    )
    out["observation.state"] = state

    # Images — already (H,W,3) uint8 from camera.read_latest()
    for cam_name in ("cam_high", "cam_right_wrist"):
        if cam_name in raw:
            out[f"observation.images.{cam_name}"] = np.ascontiguousarray(raw[cam_name])

    # TCP pose from RTDE
    try:
        tcp = np.array(robot.rtde_rec.getActualTCPPose()[:3], dtype=np.float32)
    except Exception:
        tcp = np.zeros(3, dtype=np.float32)
    out["tcp_pose"] = tcp

    return out

def _action_to_robot(action: np.ndarray) -> dict:
    """
    Convert (7,) action array from SERL server into the dict format
    robot.send_action() expects: {joint_0..5: float, gripper: float}
    Action is already in real joint-space units (radians + gripper [0,1]).
    """
    action = np.asarray(action, dtype=np.float64).flatten()
    return {
        "joint_0": float(action[0]),
        "joint_1": float(action[1]),
        "joint_2": float(action[2]),
        "joint_3": float(action[3]),
        "joint_4": float(action[4]),
        "joint_5": float(action[5]),
        "gripper":  float(np.clip(action[6], 0.0, 1.0)),
    }

# def _obs_to_payload(raw: dict, robot) -> dict:
#     out = {}
#     for k, v in raw.items():
#         if isinstance(v, np.ndarray):
#             arr = np.ascontiguousarray(v)
#             out[k] = arr.astype(np.float32) if arr.dtype != np.uint8 else arr
#         else:
#             out[k] = float(v)

#     # Append TCP XYZ from robot RTDE — shape (3,)
#     try:
#         tcp = np.array(robot.robot.get_tcp_pose()[:3], dtype=np.float32)  # UR RTDEControl
#     except Exception:
#         try:
#             tcp = np.array(robot.robot.get_actual_tcp_pose()[:3], dtype=np.float32)
#         except Exception:
#             tcp = np.zeros(3, dtype=np.float32)
#     out["tcp_pose"] = tcp
#     return out

# def _obs_to_payload(raw: dict) -> dict:
#     out = {}
#     for k, v in raw.items():
#         if isinstance(v, np.ndarray):
#             arr = np.ascontiguousarray(v)
#             out[k] = arr.astype(np.float32) if arr.dtype != np.uint8 else arr
#         else:
#             out[k] = float(v)
#     return out


# async def _run(args):
#     robot_cfg = UR10Config(ip=args.ur_ip)
#     robot = make_robot_from_config(robot_cfg)
#     robot.cameras = make_cameras_from_configs(_default_camera_configs(args.fps))
#     robot.connect()
#     logging.info("Robot connected.")

#     uri = f"ws://{args.server_host}:{args.server_port}/"
#     packer = msgpack_numpy.Packer()

#     try:
#         async with connect(uri, max_size=None, compression=None) as ws:
#             logging.info("Connected to SERL server at %s", uri)
            
#             while True:  # outer loop: one episode per iteration
#                 ctrl_raw = await ws.recv()
#                 ctrl = msgpack_numpy.unpackb(ctrl_raw)

#                 if ctrl.get("__ctrl__") == "reset":
#                     logging.info("Resetting robot to home...")
#                     home_action = default_ur_home_action()
#                     smooth_move_to_home(robot, home_action)
#                     time.sleep(1.5)
#                     raw_obs = await asyncio.to_thread(robot.get_observation)
#                     await ws.send(packer.pack({
#                         "type": "reset_done",
#                         "observation": _obs_to_payload(raw_obs, robot),
#                     }))
#                     logging.info("Reset done, sent initial obs.")

#                 elif ctrl.get("__ctrl__") == "shutdown":
#                     logging.info("Server requested shutdown.")
#                     break

#                 elif "action" in ctrl:
#                     # Server sent first action of new episode directly after reset_done
#                     # (no second reset ctrl) — handle it as step 0
#                     logging.info("Received first action directly, starting episode.")
#                     action = ctrl["action"]
#                     await asyncio.to_thread(robot.send_action, _action_to_robot(action))
#                     raw_obs = await asyncio.to_thread(robot.get_observation)
#                     await ws.send(packer.pack({
#                         "type": "step_result",
#                         "observation": _obs_to_payload(raw_obs, robot),
#                     }))
#                     # Now fall into the step loop
#                     step = 1
#                     episode_done = False
#                     while not episode_done:
#                         # ... (existing step loop body, starting from msg_raw = await ws.recv())

#             # while True:  # outer loop: one episode per iteration
#             #     # ── wait for server's episode start signal ──
#             #     ctrl_raw = await ws.recv()
#             #     ctrl = msgpack_numpy.unpackb(ctrl_raw)

#             #     if ctrl.get("__ctrl__") == "reset":
#             #         logging.info("Resetting robot to home...")
#             #         home_action = default_ur_home_action()
#             #         smooth_move_to_home(robot, home_action)
#             #         time.sleep(3.0)
#             #         raw_obs = await asyncio.to_thread(robot.get_observation)
#             #         await ws.send(packer.pack({
#             #             "type": "reset_done",
#             #             "observation": _obs_to_payload(raw_obs, robot),
#             #         }))
#             #         logging.info("Reset done, sent initial obs.")

#             #     elif ctrl.get("__ctrl__") == "shutdown":
#             #         logging.info("Server requested shutdown.")
#             #         break

#             #     else:
#             #         logging.warning("Unexpected ctrl message: %s", ctrl)
#             #         continue

#             #     # ── step loop for one episode ──
#             #     step = 0
#             #     episode_done = False
#             #     while not episode_done:
#                     # Receive action from server
#                     msg_raw = await ws.recv()
#                     msg = msgpack_numpy.unpackb(msg_raw)

#                     if msg.get("__ctrl__") == "reset":
#                         # Server ended episode early — break to outer loop
#                         logging.info("Server ended episode at step %d", step)
#                         home_action = default_ur_home_action()
#                         smooth_move_to_home(robot, home_action)
#                         time.sleep(3.0)
#                         raw_obs = await asyncio.to_thread(robot.get_observation)
#                         await ws.send(packer.pack({
#                             "type": "reset_done",
#                             "observation": _obs_to_payload(raw_obs, robot),
#                         }))
#                         break

#                     action = msg.get("action")
#                     print(action)
#                     if action is None:
#                         logging.warning("No action in message, skipping step.")
#                         continue

#                     t0 = asyncio.get_event_loop().time()

#                     # Execute action on robot
#                     # await asyncio.to_thread(robot.send_action, action)
#                     await asyncio.to_thread(robot.send_action, _action_to_robot(action))


#                     # Get observation
#                     raw_obs = await asyncio.to_thread(robot.get_observation)

#                     # Send obs back
#                     await ws.send(packer.pack({
#                         "type": "step_result",
#                         "observation": _obs_to_payload(raw_obs, robot),
#                     }))

#                     step += 1

#                     # Timing
#                     elapsed = asyncio.get_event_loop().time() - t0
#                     sleep_t = args.control_dt - elapsed
#                     if sleep_t > 0:
#                         await asyncio.sleep(sleep_t)

#     finally:
#         robot.disconnect()
#         logging.info("Robot disconnected.")

# async def run_episode(ws, robot, packer, args):
#     step = 0
#     while True:
#         msg_raw = await ws.recv()
#         msg = msgpack_numpy.unpackb(msg_raw)

#         if msg.get("__ctrl__") == "reset":
#             logging.info("Reset received at step %d, going home.", step)
#             smooth_move_to_home(robot, default_ur_home_action())
            
#             # Wait for arm to fully settle at home
#             await asyncio.sleep(1.5)
            
#             # Now capture obs — this is from the settled home position
#             raw_obs = await asyncio.to_thread(robot.get_observation)
#             await ws.send(packer.pack({
#                 "type": "reset_done",
#                 "observation": _obs_to_payload(raw_obs, robot),
#             }))
#             return

#         elif msg.get("__ctrl__") == "shutdown":
#             raise StopAsyncIteration

#         elif "action" in msg:
#             t0 = asyncio.get_event_loop().time()
#             await asyncio.to_thread(robot.send_action, _action_to_robot(msg["action"]))
#             raw_obs = await asyncio.to_thread(robot.get_observation)
#             await ws.send(packer.pack({
#                 "type": "step_result",
#                 "observation": _obs_to_payload(raw_obs, robot),
#             }))
#             step += 1
#             elapsed = asyncio.get_event_loop().time() - t0
#             sleep_t = args.control_dt - elapsed
#             if sleep_t > 0:
#                 await asyncio.sleep(sleep_t)

#         else:
#             logging.warning("Unknown message: %s", msg)

# async def run_episode(ws, robot, packer, args):
#     step = 0
#     while True:
#         msg_raw = await ws.recv()
#         msg = msgpack_numpy.unpackb(msg_raw)

#         if msg.get("__ctrl__") == "reset":
#             logging.info("Reset received at step %d, going home.", step)

#             # Run blocking smooth_move_to_home off the event loop thread
#             # so it doesn't corrupt WebSocket timing
#             await asyncio.to_thread(
#                 smooth_move_to_home,
#                 robot,
#                 default_ur_home_action(),
#             )

#             # Extra settle — smooth_move_to_home ends with settle_s=1.0 already,
#             # but the UR controller may still be finishing its last servo command
#             await asyncio.sleep(1.0)

#             # Capture obs only after full settle
#             raw_obs = await asyncio.to_thread(robot.get_observation)
#             await ws.send(packer.pack({
#                 "type": "reset_done",
#                 "observation": _obs_to_payload(raw_obs, robot),
#             }))
#             logging.info("Reset done. Home joints: %s",
#                          [raw_obs.get(f"joint_{i}", 0) for i in range(6)])
#             return

#         elif msg.get("__ctrl__") == "shutdown":
#             raise StopAsyncIteration

#         elif "action" in msg:
#             t0 = asyncio.get_event_loop().time()
#             await asyncio.to_thread(robot.send_action, _action_to_robot(msg["action"]))
#             raw_obs = await asyncio.to_thread(robot.get_observation)
#             await ws.send(packer.pack({
#                 "type": "step_result",
#                 "observation": _obs_to_payload(raw_obs, robot),
#             }))
#             step += 1
#             elapsed = asyncio.get_event_loop().time() - t0
#             sleep_t = args.control_dt - elapsed
#             if sleep_t > 0:
#                 await asyncio.sleep(sleep_t)

#         else:
#             logging.warning("Unknown message: %s", msg)

async def run_episode(ws, robot, packer, args, episode_idx: int = 0):
    """
    Run one episode. Records a side-by-side MP4 of cam_high + cam_right_wrist
    for every episode unless --no-video is set.
    Saves to: <video_dir>/ep_<NNNN>_<timestamp>.mp4
    """
    import cv2
    import datetime

    record = not args.no_video
    frames = []          # list of (H, W*2, 3) uint8 BGR frames
    step = 0

    while True:
        msg_raw = await ws.recv()
        msg = msgpack_numpy.unpackb(msg_raw)

        if msg.get("__ctrl__") == "reset":
            logging.info("Reset received at step %d, going home.", step)

            # ── Save video before resetting ───────────────────────────────
            if record and frames:
                Path(args.video_dir).mkdir(parents=True, exist_ok=True)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = str(Path(args.video_dir) / f"ep_{episode_idx:04d}_{ts}.mp4")
                h, w = frames[0].shape[:2]
                writer = cv2.VideoWriter(
                    out_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    int(1.0 / args.control_dt),   # fps matches control rate
                    (w, h),
                )
                for f in frames:
                    writer.write(f)
                writer.release()
                logging.info("Saved rollout video (%d frames) → %s", len(frames), out_path)
            frames = []

            # ── Gripper reboot if latched hardware error ──────────────────
            gripper = getattr(robot, "gripper", None)
            if gripper is not None and hasattr(gripper, "reboot_if_error"):
                rebooted = await asyncio.to_thread(gripper.reboot_if_error)
                if rebooted:
                    logging.info("Gripper rebooted due to hardware error.")
                    await asyncio.to_thread(gripper.open)

            # ── Move arm home ─────────────────────────────────────────────
            await asyncio.to_thread(
                smooth_move_to_home,
                robot,
                default_ur_home_action(),
            )
            await asyncio.sleep(1.0)

            raw_obs = await asyncio.to_thread(robot.get_observation)
            await ws.send(packer.pack({
                "type": "reset_done",
                "observation": _obs_to_payload(raw_obs, robot),
            }))
            logging.info("Reset done. Home joints: %s",
                         [raw_obs.get(f"joint_{i}", 0) for i in range(6)])
            return

        elif msg.get("__ctrl__") == "shutdown":
            # Save final episode video before exiting
            if record and frames:
                import cv2, datetime
                Path(args.video_dir).mkdir(parents=True, exist_ok=True)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = str(Path(args.video_dir) / f"ep_{episode_idx:04d}_{ts}.mp4")
                h, w = frames[0].shape[:2]
                writer = cv2.VideoWriter(
                    out_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    int(1.0 / args.control_dt),
                    (w, h),
                )
                for f in frames:
                    writer.write(f)
                writer.release()
                logging.info("Saved final rollout video → %s", out_path)
            raise StopAsyncIteration

        elif "action" in msg:
            t0 = asyncio.get_event_loop().time()
            await asyncio.to_thread(robot.send_action, _action_to_robot(msg["action"]))
            raw_obs = await asyncio.to_thread(robot.get_observation)

            # ── Grab frames for recording ─────────────────────────────────
            if record:
                import cv2
                imgs = []
                for cam in ("cam_high", "cam_right_wrist"):
                    img = raw_obs.get(cam)
                    if img is not None:
                        # obs images are RGB uint8 (H,W,3) — convert to BGR for cv2
                        imgs.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if imgs:
                    # Pad to same height if cameras differ, then concat side-by-side
                    max_h = max(im.shape[0] for im in imgs)
                    padded = []
                    for im in imgs:
                        if im.shape[0] < max_h:
                            pad = np.zeros((max_h - im.shape[0], im.shape[1], 3), dtype=np.uint8)
                            im = np.vstack([im, pad])
                        padded.append(im)
                    frame = np.hstack(padded)

                    # Overlay step counter
                    cv2.putText(frame, f"ep {episode_idx:04d}  step {step:04d}",
                                (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    frames.append(frame)

            await ws.send(packer.pack({
                "type": "step_result",
                "observation": _obs_to_payload(raw_obs, robot),
            }))
            step += 1
            elapsed = asyncio.get_event_loop().time() - t0
            sleep_t = args.control_dt - elapsed
            if sleep_t > 0:
                await asyncio.sleep(sleep_t)

        else:
            logging.warning("Unknown message: %s", msg)

# async def run_episode(ws, robot, packer, args):
#     """Run one episode. Returns when episode ends or reset is received."""
#     step = 0
#     while True:
#         msg_raw = await ws.recv()
#         msg = msgpack_numpy.unpackb(msg_raw)

#         if msg.get("__ctrl__") == "reset":
#             logging.info("Reset received at step %d, going home.", step)
#             smooth_move_to_home(robot, default_ur_home_action())
#             time.sleep(1.5)
#             raw_obs = await asyncio.to_thread(robot.get_observation)
#             await ws.send(packer.pack({
#                 "type": "reset_done",
#                 "observation": _obs_to_payload(raw_obs, robot),
#             }))
#             return  # caller starts a new episode

#         elif msg.get("__ctrl__") == "shutdown":
#             raise StopAsyncIteration

#         elif "action" in msg:
#             t0 = asyncio.get_event_loop().time()
#             await asyncio.to_thread(robot.send_action, _action_to_robot(msg["action"]))
#             raw_obs = await asyncio.to_thread(robot.get_observation)
#             await ws.send(packer.pack({
#                 "type": "step_result",
#                 "observation": _obs_to_payload(raw_obs, robot),
#             }))
#             step += 1
#             elapsed = asyncio.get_event_loop().time() - t0
#             sleep_t = args.control_dt - elapsed
#             if sleep_t > 0:
#                 await asyncio.sleep(sleep_t)

#         else:
#             logging.warning("Unknown message: %s", msg)


async def _run(args):
    gripper = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAO51RF-if00-port0"
    robot_cfg = UR10Config(ip=args.ur_ip, gripper_port=gripper)
    robot = make_robot_from_config(robot_cfg)
    robot.cameras = make_cameras_from_configs(_default_camera_configs(args.fps))
    robot.connect()
    logging.info("Robot connected.")

    uri = f"ws://{args.server_host}:{args.server_port}/"
    packer = msgpack_numpy.Packer()

    try:
        async with connect(uri, max_size=None, compression=None) as ws:
            logging.info("Connected to SERL server at %s", uri)
            episode_idx = 0
            while True:
                try:
                    await run_episode(ws, robot, packer, args, episode_idx)
                    episode_idx += 1
                except StopAsyncIteration:
                    logging.info("Server requested shutdown.")
                    break
    finally:
        robot.disconnect()
        logging.info("Robot disconnected.")

<<<<<<< HEAD
=======

>>>>>>> 86760d5fb4b2b916c91d8e3736abf60f4663a89b
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--server-host", default="10.245.91.19")
    p.add_argument("--server-port", type=int, default=8766)
    p.add_argument("--ur-ip", default="192.168.100.3")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--control-dt", type=float, default=0.033)
    p.add_argument("--video-dir", type=str, default="rollout_videos",
                   help="Directory to save per-episode MP4 recordings.")
    p.add_argument("--no-video", action="store_true",
                   help="Disable video recording.")
    return p.parse_args()

# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--server-host", default="10.245.91.19")
#     p.add_argument("--server-port", type=int, default=8766)
#     p.add_argument("--ur-ip", default="192.168.100.3")
#     p.add_argument("--fps", type=int, default=30)
#     p.add_argument("--control-dt", type=float, default=0.033)
#     return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_run(parse_args()))