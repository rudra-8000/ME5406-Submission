#!/usr/bin/env python3
"""
save_classifier_frames.py

Runs your ACT policy for N episodes on the real UR10 and saves every frame
from both cameras to disk for classifier retraining.

After collecting, manually sort frames into:
  <out_dir>/wrist/success/   ← gripper closed on object
  <out_dir>/wrist/failure/   ← gripper empty / near-miss
  <out_dir>/top/success/     ← object in box / goal state
  <out_dir>/top/failure/     ← object displaced / knocked out

Usage:
  python examples/ur10_gello/save_classifier_frames.py \
      --checkpoint_path /home_local/rudra_1/rudra/act_4/checkpoints/mod/ \
      --robot-ws-port 8766 \
      --num_episodes 15 \
      --max_episode_steps 300 \
      --out_dir ./classifier_data \
      --save_every_n_steps 5
"""

import argparse
import copy
import sys
import signal
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from safetensors.torch import load_file as load_safetensors

# Reuse the WebSocket robot interface exactly as in serl_finetune_act.py
import asyncio, threading, websockets, msgpack_numpy

# ---------------------------------------------------------------------------
# Copy of UR10RobotInterface (identical to serl_finetune_act.py)
# ---------------------------------------------------------------------------
class UR10RobotInterface:
    def __init__(self, host="0.0.0.0", port=8766):
        self.host = host
        self.port = port
        self._packer = msgpack_numpy.Packer()
        self._ws = None
        self._loop = None
        self._obs_queue = None
        self._last_raw_obs = {}
        self._server_thread = threading.Thread(target=self._start_server, daemon=True)
        self._server_thread.start()
        self._wait_for_client()

    def _start_server(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._obs_queue = asyncio.Queue()
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        async with websockets.serve(self._handler, self.host, self.port,
                                    max_size=None, compression=None) as server:
            await server.wait_closed()

    async def _handler(self, websocket):
        self._ws = websocket
        async for raw in websocket:
            await self._obs_queue.put(msgpack_numpy.unpackb(raw))

    def _wait_for_client(self):
        print(f"[FrameSaver] Waiting for robot client on port {self.port}...")
        while self._ws is None:
            time.sleep(0.2)
        print("[FrameSaver] Robot client connected!")

    def _send_sync(self, data):
        future = asyncio.run_coroutine_threadsafe(
            self._ws.send(self._packer.pack(data)), self._loop)
        future.result(timeout=10.0)

    def _recv_sync(self, timeout=15.0):
        future = asyncio.run_coroutine_threadsafe(self._obs_queue.get(), self._loop)
        return future.result(timeout=timeout)

    def reset(self):
        self._send_sync({"__ctrl__": "reset"})
        msg = self._recv_sync(timeout=30.0)
        assert msg["type"] == "reset_done"
        self._last_raw_obs = msg.get("observation", {})
        return self._raw_to_obs(self._last_raw_obs)

    def step(self, action):
        self._send_sync({"action": action})
        msg = self._recv_sync(timeout=10.0)
        assert msg["type"] == "step_result"
        self._last_raw_obs = msg.get("observation", {})
        return self._raw_to_obs(self._last_raw_obs), False, {}

    def get_tcp_pose(self):
        tcp = self._last_raw_obs.get("tcp_pose", None)
        if tcp is None:
            return np.zeros(3)
        return np.array(tcp, dtype=np.float32)

    def _raw_to_obs(self, raw):
        if "observation.state" in raw:
            state = np.array(raw["observation.state"], dtype=np.float32)
        else:
            state = np.array([raw.get(f"joint_{i}", 0.0) for i in range(6)]
                             + [raw.get("gripper", 0.0)], dtype=np.float32)
        obs = {"observation.state": state}
        for cam in ("cam_high", "cam_right_wrist"):
            for key in (cam, f"observation.images.{cam}"):
                if key in raw:
                    obs[f"observation.images.{cam}"] = raw[key]
                    break
        return obs


# ---------------------------------------------------------------------------
# Frame saver
# ---------------------------------------------------------------------------

def save_frame(img_hwc_uint8: np.ndarray, out_path: Path):
    """Save (H,W,3) uint8 RGB image as JPEG. Converts to BGR for OpenCV."""
    bgr = cv2.cvtColor(img_hwc_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

def check_tcp_z(robot: UR10RobotInterface, tcp_z_min: float) -> bool:
    tcp = robot.get_tcp_pose()
    if tcp is None:
        if not hasattr(check_tcp_z, "_warned"):
            print("[Safety] WARNING: tcp_pose not in observation — "
                  "tcp_z_min check is DISABLED. Update serl_client_ur10.py "
                  "to send tcp_pose in every observation.")
            check_tcp_z._warned = True
        return True   # can't check, allow to continue

    z = float(tcp[2])
    if z < tcp_z_min:
        print(f"[Safety] ⚠  TCP Z = {z:.4f} m  <  tcp_z_min = {tcp_z_min:.4f} m  "
              f"— terminating episode to protect table!")
        return False
    return True

def load_act(checkpoint_path: str, device: torch.device):
    ckpt = Path(checkpoint_path)
    config_candidates = [
        ckpt / "config.json",
        ckpt / "pretrained_model" / "config.json",
        ckpt.parent / "config.json",
    ]
    config_path = next((c for c in config_candidates if c.exists()), None)
    if config_path is None:
        raise FileNotFoundError(f"config.json not found under {ckpt}")

    policy = ACTPolicy.from_pretrained(str(config_path.parent))
    policy.to(device).eval()

    pre_path = ckpt / "pretrained_model" / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    post_path = ckpt / "pretrained_model" / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    norm_stats   = load_safetensors(str(pre_path),  device="cpu")
    unnorm_stats = load_safetensors(str(post_path), device="cpu")

    state_mean   = norm_stats["observation.state.mean"].numpy()
    state_std    = norm_stats["observation.state.std"].numpy()
    action_mean  = unnorm_stats["action.mean"].numpy()
    action_std   = unnorm_stats["action.std"].numpy()

    return policy, state_mean, state_std, action_mean, action_std


def normalize_obs(raw_obs, state_mean, state_std):
    norm = {}
    s = raw_obs["observation.state"].astype(np.float32)
    norm["observation.state"] = (s - state_mean) / (state_std + 1e-8)

    img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    img_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    for k in ("observation.images.cam_high", "observation.images.cam_right_wrist"):
        if k not in raw_obs:
            continue
        img = raw_obs[k]
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        norm[k] = ((img - img_mean) / img_std).transpose(2, 0, 1)  # (3,H,W)
    return norm


def obs_to_batch(obs_norm, device):
    return {k: torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
            for k, v in obs_norm.items()}


def get_action(policy, obs_norm_batch, action_mean, action_std, device):
    with torch.no_grad():
        out = policy.select_action(obs_norm_batch)
    if isinstance(out, np.ndarray):
        out = torch.tensor(out, dtype=torch.float32, device=device)
    action_norm = out.squeeze(0).cpu().numpy()
    return action_norm * (action_std + 1e-8) + action_mean  # unnormalize


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FrameSaver] Device: {device}")

    policy, state_mean, state_std, action_mean, action_std = load_act(
        cfg.checkpoint_path, device)
    policy.reset()

    # Create output dirs — unsorted/ holds everything before you manually sort
    out = Path(cfg.out_dir)
    unsorted_wrist = out / "unsorted" / "wrist";  unsorted_wrist.mkdir(parents=True, exist_ok=True)
    unsorted_top   = out / "unsorted" / "top";    unsorted_top.mkdir(parents=True, exist_ok=True)
    # Pre-create the sort targets so you can drag-and-drop immediately
    for split in ("success", "failure"):
        (out / "wrist" / split).mkdir(parents=True, exist_ok=True)
        (out / "top"   / split).mkdir(parents=True, exist_ok=True)

    robot = UR10RobotInterface(host="0.0.0.0", port=cfg.robot_ws_port)

    total_frames = 0
    interrupted  = False

    def _sigint(sig, frame):
        nonlocal interrupted
        print("\n[FrameSaver] Ctrl+C — finishing current episode then stopping.")
        interrupted = True
    signal.signal(signal.SIGINT, _sigint)

    for ep in range(cfg.num_episodes):
        tcp_z_violated = False
        if interrupted:
            break

        print(f"\n{'='*60}")
        print(f"[FrameSaver] Episode {ep+1}/{cfg.num_episodes}")
        input("  Place object, then press Enter to reset arm and start...")

        raw_obs = robot.reset()
        policy.reset()  # clear temporal ensembler
        ep_frames = 0

        if tcp_z_violated:
            print(f"  Episode {ep+1} SAFETY STOPPED (TCP Z violation) — "
                f"saved {ep_frames} frames before stop.")
            print("  Resetting arm to home before next episode...")
            tcp_z_violated = False
            raw_obs = robot.reset()
        for step in range(cfg.max_episode_steps):
            if not check_tcp_z(robot, cfg.tcp_z_min):
                tcp_z_violated = True
                # safety_stops += 1
                break

            if interrupted:
                break

            # ── Save frames (before step — captures current scene state) ──
            if step % cfg.save_every_n_steps == 0:
                ts = f"ep{ep:03d}_s{step:04d}"

                wrist_img = raw_obs.get("observation.images.cam_right_wrist")
                top_img   = raw_obs.get("observation.images.cam_high")

                if wrist_img is not None:
                    # Images arrive as (H,W,3) uint8 from the WebSocket client
                    # If they're already float, convert back
                    img = wrist_img
                    if img.dtype != np.uint8:
                        img = (img * 255).clip(0, 255).astype(np.uint8)
                    save_frame(img, unsorted_wrist / f"{ts}.jpg")

                if top_img is not None:
                    img = top_img
                    if img.dtype != np.uint8:
                        img = (img * 255).clip(0, 255).astype(np.uint8)
                    save_frame(img, unsorted_top / f"{ts}.jpg")

                ep_frames += 1
                total_frames += 1

            # ── ACT action ──
            obs_norm = normalize_obs(raw_obs, state_mean, state_std)
            obs_batch = obs_to_batch(obs_norm, device)
            action = get_action(policy, obs_batch, action_mean, action_std, device)

            raw_obs, _, _ = robot.step(action)

            # Print classifier scores live if classifiers provided (optional debug)
            if cfg.classifier_wrist and step % cfg.save_every_n_steps == 0:
                _print_scores(cfg, raw_obs, device)

        print(f"  Episode {ep+1} done — saved {ep_frames} frame pairs "
              f"(total so far: {total_frames})")

    print(f"\n[FrameSaver] Collection complete. {total_frames} frame pairs saved.")
    print(f"\nNow sort frames into success/failure:")
    print(f"  {out}/unsorted/wrist/  →  {out}/wrist/success/  or  {out}/wrist/failure/")
    print(f"  {out}/unsorted/top/    →  {out}/top/success/    or  {out}/top/failure/")
    print(f"\nThen retrain classifiers:")
    print(f"  python examples/ur10_gello/train_reward_classifier.py \\")
    print(f"      --success_dir {out}/wrist/success \\")
    print(f"      --failure_dir {out}/wrist/failure \\")
    print(f"      --out ./examples/ur10_gello/reward_classifier_wrist.pt")
    print(f"  python examples/ur10_gello/train_reward_classifier.py \\")
    print(f"      --success_dir {out}/top/success \\")
    print(f"      --failure_dir {out}/top/failure \\")
    print(f"      --out ./examples/ur10_gello/reward_classifier_top.pt")


def _print_scores(cfg, raw_obs, device):
    """Optional: print live classifier scores during collection."""
    import torch.nn as nn
    import torchvision.models as tvm

    # Lazy-load classifiers only when --classifier_wrist is provided
    if not hasattr(_print_scores, "_clf_wrist"):
        _print_scores._clf_wrist = _load_clf(cfg.classifier_wrist, device)
    if cfg.classifier_top and not hasattr(_print_scores, "_clf_top"):
        _print_scores._clf_top = _load_clf(cfg.classifier_top, device)

    wrist_img = raw_obs.get("observation.images.cam_right_wrist")
    top_img   = raw_obs.get("observation.images.cam_high")

    scores = []
    if wrist_img is not None:
        scores.append(f"wrist={_score(_print_scores._clf_wrist, wrist_img, device):.2f}")
    if top_img is not None and hasattr(_print_scores, "_clf_top"):
        scores.append(f"top={_score(_print_scores._clf_top, top_img, device):.2f}")
    if scores:
        print(f"    [clf] {' | '.join(scores)}")


def _load_clf(path, device):
    import torchvision.models as tvm
    import torch.nn as nn
    base = tvm.resnet18(weights=None)
    base.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))
    ckpt = torch.load(path, map_location=device)
    base.load_state_dict(ckpt["model_state"], strict=False)
    return base.to(device).eval()


def _score(clf, img_hwc, device):
    img = img_hwc.astype(np.float32) / 255.0 if img_hwc.dtype == np.uint8 else img_hwc
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    img = ((img - mean) / std).transpose(2, 0, 1)
    t = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        return float(torch.sigmoid(clf(t).squeeze()).item())


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--robot-ws-port",   type=int, default=8766)
    p.add_argument("--num_episodes",    type=int, default=15,
                   help="Number of ACT rollout episodes to collect")
    p.add_argument("--max_episode_steps", type=int, default=300)
    p.add_argument("--out_dir",         type=str, default="./classifier_data")
    p.add_argument("--save_every_n_steps", type=int, default=5,
                   help="Save one frame pair every N steps (5 = ~60 frames/ep at 300 steps)")
    p.add_argument("--classifier_wrist", type=str, default=None,
                   help="Optional: path to existing wrist classifier to show live scores")
    p.add_argument("--classifier_top",   type=str, default=None,
                   help="Optional: path to existing top classifier to show live scores")
    p.add_argument("--tcp_z_min", type=float, default=0.005,
                   help="Minimum safe TCP Z height in meters — episode will terminate if violated")
    return p.parse_args()


if __name__ == "__main__":
    collect(parse_args())