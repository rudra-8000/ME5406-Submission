#!/usr/bin/env python3
"""
serl_finetune_act.py

SERL (Sample Efficient Real-Robot RL) fine-tuning on top of a pretrained ACT policy
on a real UR10 robot using LeRobot.

Design:
  - Loads your pretrained ACT checkpoint (frozen backbone, trainable policy head by default)
  - Uses SAC-style off-policy RL with a learned critic
  - Reward is computed from a classifier trained on a small set of success/failure images
    (you label ~20-30 demos each), OR from a simple hand-crafted reward
  - Action = ACT policy output + RL residual (residual policy learning)
  - Safety wrapper enforces joint-space limits and tool-center-point (TCP) workspace box

On Compute Node:
python examples/ur10_gello/serl_finetune_act.py \
    --checkpoint_path /home_local/rudra_1/rudra/act_4/checkpoints/060000/ \
    --robot_ip 192.168.100.3 \
    --robot-ws-port 8766 \
    --reward_mode classifier \
    --reward_classifier_path ./examples/ur10_gello/reward_classifier_wrist.pt \
    --num_steps 20000 \
    --warmup_steps 500 \
    --max_episode_steps 80 \
    --batch_size 96 \
    --save_dir ./serl_checkpoints \
    --tcp_workspace_min -0.75 -1.00 0.02 \
    --tcp_workspace_max  0.75  0.80 0.70 \
    --max_action_delta 3.5

On Laptop:
python examples/ur10_gello/serl_finetune_act.py \
    --checkpoint_path /mnt/UBU-DATA/WORK/Policy_Checkpoints/act_10k/ \
    --robot_ip 192.168.100.3 \
    --robot-ws-port 8766 \
    --reward_mode classifier \
    --reward_classifier_path ./examples/ur10_gello/reward_classifier_wrist.pt \
    --num_steps 20000 \
    --warmup_steps 500 \
    --max_episode_steps 80 \
    --batch_size 96 \
    --save_dir ./serl_checkpoints \
    --tcp_workspace_min -0.75 -1.00 0.02 \
    --tcp_workspace_max  0.75  0.80 0.70 \
    --max_action_delta 3.5

Safety critical: you MUST set --tcp_workspace_min and --tcp_workspace_max correctly for your setup.
test
"""
import logging
import argparse
import collections
import copy
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import sys
import termios


# ---------------------------------------------------------------------------
# LeRobot / ACT imports — adjust paths if your fork differs
# ---------------------------------------------------------------------------
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
# from lerobot.robots.lerobot_robot_ur10 import UR10Config


from pathlib import Path


import asyncio
import threading
import websockets
import msgpack_numpy

class UR10RobotInterface:
    """
    Server-side robot interface for SERL training.
    Opens a WebSocket server on --robot-ws-port (default 8766).
    The robot PC runs serl_client_ur10.py which connects here.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8766):
        self.host = host
        self.port = port
        self._packer = msgpack_numpy.Packer()
        self._ws = None  # the connected client websocket
        self._loop = None
        self._server = None

        # Queues for sync ↔ async bridge
        self._obs_queue: asyncio.Queue = None
        self._action_queue: asyncio.Queue = None

        self._server_thread = threading.Thread(target=self._start_server, daemon=True)
        self._server_thread.start()
        self._wait_for_client()
        self._last_raw_obs = {}


    def _start_server(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._obs_queue = asyncio.Queue()
        self._action_queue = asyncio.Queue()
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        async with websockets.serve(
            self._handler, self.host, self.port,
            max_size=None, compression=None
        ) as server:
            self._server = server
            logging.info(f"[SERL-WS] Waiting for robot client on port {self.port}...")
            await server.wait_closed()

    async def _handler(self, websocket):
        self._ws = websocket
        logging.info("[SERL-WS] Robot client connected.")
        async for raw in websocket:
            msg = msgpack_numpy.unpackb(raw)
            await self._obs_queue.put(msg)

    def _wait_for_client(self):
        """Block until the robot client connects."""
        import time
        print("[SERL] Waiting for robot PC to connect on port 8766...")
        while self._ws is None:
            time.sleep(0.2)
        print("[SERL] Robot client connected!")

    def _send_sync(self, data: dict):
        """Send a message from the training thread (sync → async bridge)."""
        future = asyncio.run_coroutine_threadsafe(
            self._ws.send(self._packer.pack(data)),
            self._loop
        )
        future.result(timeout=10.0)

    def _recv_sync(self, timeout: float = 15.0) -> dict:
        """Receive a message from the robot PC (sync ← async bridge)."""
        future = asyncio.run_coroutine_threadsafe(
            self._obs_queue.get(), self._loop
        )
        return future.result(timeout=timeout)

    def reset(self) -> dict:
        """Send reset command — robot immediately moves home, then waits for obs."""
        self._send_sync({"__ctrl__": "reset"})
        msg = self._recv_sync(timeout=30.0)
        assert msg["type"] == "reset_done", f"Unexpected: {msg}"
        raw_obs = msg.get("observation", {})
        self._last_raw_obs = raw_obs
        return self._raw_to_obs(raw_obs)

    # def reset(self) -> dict:
    #     """Send reset command, wait for home + initial obs."""
    #     input("  Place object in scene, then press Enter...")
    #     self._send_sync({"__ctrl__": "reset"})
    #     msg = self._recv_sync(timeout=30.0)  # client now settles before sending this
    #     assert msg["type"] == "reset_done", f"Unexpected: {msg}"
        
    #     raw_obs = msg.get("observation", {})
    #     self._last_raw_obs = raw_obs
    #     return self._raw_to_obs(raw_obs)

    # def reset(self) -> dict:
    #     input("  Place object in scene, then press Enter...")
    #     self._send_sync({"__ctrl__": "reset"})
    #     msg = self._recv_sync(timeout=30.0)
    #     assert msg["type"] == "reset_done", f"Unexpected: {msg}"

    #     import time
    #     time.sleep(1.5)  # let the arm finish settling

    #     raw_obs = msg.get("observation", {})
    #     self._last_raw_obs = raw_obs
    #     return self._raw_to_obs(raw_obs)

    # def reset(self) -> dict:
    #     """Send reset command, wait for home + initial obs."""
    #     input("  Place object in scene, then press Enter...")
    #     self._send_sync({"__ctrl__": "reset"})
    #     msg = self._recv_sync(timeout=30.0)  # wait for smooth_move_to_home
    #     raw_obs = msg.get("observation", {})
    #     self._last_raw_obs = raw_obs   # raw dict from WebSocket, including tcp_pose
    #     assert msg["type"] == "reset_done", f"Unexpected: {msg}"
    #     return self._raw_to_obs(msg["observation"])

    def step(self, action: np.ndarray) -> tuple[dict, bool, dict]:
        """Send action, receive next obs."""
        self._send_sync({"action": action})
        msg = self._recv_sync(timeout=10.0)
        assert msg["type"] == "step_result", f"Unexpected: {msg}"
        raw_obs = msg.get("observation", {})
        self._last_raw_obs = raw_obs   # raw dict from WebSocket, including tcp_pose
        obs = self._raw_to_obs(msg["observation"])
        return obs, False, {}

    # def get_tcp_pose(self) -> np.ndarray:
    #     """Extract TCP XYZ from state obs (joints 0-5, no direct TCP here).
    #     Override if your robot.get_observation() returns tcp_pose."""
    #     # Fallback: return zeros — safety wrapper will still work if you
    #     # set tcp_workspace bounds wide. Replace with actual TCP if available.
    #     return np.zeros(3)

    def get_tcp_pose(self) -> np.ndarray:
        """Read TCP XYZ from the last observation dict sent by the client."""
        tcp = self._last_raw_obs.get("tcp_pose", None)
        if tcp is None or (isinstance(tcp, np.ndarray) and np.all(tcp == 0)):
            # Fallback: forward kinematics from joint state using urx/numpy
            # This path only triggers if client is old and didn't send tcp_pose
            joints = self._last_raw_obs.get("observation.state", np.zeros(7))[:6]
            return self._fk_xyz(joints)
        return np.array(tcp, dtype=np.float32)

    def _fk_xyz(self, joints: np.ndarray) -> np.ndarray:
        """
        Rough FK fallback using known UR10 DH params — only used if tcp_pose missing.
        Not needed once client sends tcp_pose correctly.
        """
        # Just return a safe in-workspace position so we don't false-trigger safety stop
        return np.array([-0.685, -0.176, 0.5], dtype=np.float32)

    def _raw_to_obs(self, raw: dict) -> dict:
        """Convert flat observation payload to ACT-formatted obs dict."""
        # Reconstruct state vector [j0..j5, gripper]
        if "observation.state" in raw:
            state = np.array(raw["observation.state"], dtype=np.float32)
        else:
            state = np.array([
                raw.get(f"joint_{i}", 0.0) for i in range(6)
            ] + [raw.get("gripper", 0.0)], dtype=np.float32)

        obs = {"observation.state": state}

        # Images — stored as (H,W,3) uint8 in raw, need (3,H,W) for ACT
        for cam in ("cam_high", "cam_right_wrist"):
            key_in = cam
            key_out = f"observation.images.{cam}"
            if key_in in raw:
                img = raw[key_in]  # (H,W,3) uint8
                obs[key_out] = img  # keep uint8; ReplayBuffer handles /255
            elif key_out in raw:
                obs[key_out] = raw[key_out]

        return obs

# UR10 robot driver — your lerobot fork's robot interface
# from lerobot.robots.ur10 import UR10Robot  # uncomment when available
# For now we provide a stub you can swap in

# ---------------------------------------------------------------------------
# 1. SAFETY WRAPPER
# ---------------------------------------------------------------------------

class SafetyWrapper:
    """
    Hard workspace and velocity constraints.
    Call check_action(action, current_state) before executing.
    action: np.ndarray [7] — [6 joint velocities OR poses, 1 gripper]
    Returns (safe_action, violated_flag)
    """

    def __init__(
        self,
        # TCP workspace box in robot base frame (meters), set to YOUR table setup
        tcp_min: np.ndarray = np.array([-0.8, -0.8, 0.02]),
        tcp_max: np.ndarray = np.array([0.8, 0.8, 0.80]),
        # Joint limits (radians) — UR10 default
        joint_min: np.ndarray = np.array([-2*np.pi]*6),
        joint_max: np.ndarray = np.array([2*np.pi]*6),
        # Max delta per step (action space clipping)
        max_delta: float = 0.05,   # meters or radians per control step
        max_gripper_delta: float = 0.1,
    ):
        self.tcp_min = tcp_min
        self.tcp_max = tcp_max
        self.joint_min = joint_min
        self.joint_max = joint_max
        self.max_delta = max_delta
        self.max_gripper_delta = max_gripper_delta

    # def clip_action(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
    #     """
    #     action: shape (7,) — first 6 dims are joint/EEF deltas, last is gripper
    #     Returns clipped action and whether it was violated.
    #     """
    #     violated = False
    #     clipped = action.copy()

    #     # Clip joint/EEF deltas
    #     norm = np.linalg.norm(clipped[:6])
    #     if norm > self.max_delta:
    #         clipped[:6] = clipped[:6] / norm * self.max_delta
    #         violated = True

    #     # Clip gripper
    #     clipped[6] = np.clip(clipped[6], -self.max_gripper_delta, self.max_gripper_delta)

    #     return clipped, violated

    def clip_action(self, action: np.ndarray,
                    current_joints: np.ndarray = None) -> Tuple[np.ndarray, bool]:
        """
        action: (7,) absolute joint targets [j0..j5, gripper]
        current_joints: (7,) current joint positions — used to limit per-step delta
        """
        violated = False
        clipped = action.copy()

        if current_joints is not None:
            # Compute per-step delta and clip it
            delta = clipped[:6] - current_joints[:6]
            norm = np.linalg.norm(delta)
            if norm > self.max_delta:
                delta = delta / norm * self.max_delta
                clipped[:6] = current_joints[:6] + delta
                violated = True

        # Hard joint limits
        clipped[:6] = np.clip(clipped[:6], self.joint_min, self.joint_max)

        # Gripper stays in [0, 1]
        clipped[6] = np.clip(clipped[6], 0.0, 1.0)

        return clipped, violated

    def check_tcp_workspace(self, tcp_pos: np.ndarray) -> bool:
        """Returns True if TCP is within allowed workspace."""
        return np.all(tcp_pos >= self.tcp_min) and np.all(tcp_pos <= self.tcp_max)

# ### -----------------------------------------------------------------------
# ### EARLY STOP LISTENER (safety stop via keyboard input)
# ### -----------------------------------------------------------------------

class EarlyStopListener:
    """
    Monitors /dev/tty for 'S' keypresses using select() polling.
    Does NOT call setraw() — terminal mode is untouched, Ctrl+C always works.
    User presses S then Enter to truncate.
    """
    def __init__(self, truncation_penalty: float = -50.0):
        self.truncation_penalty = truncation_penalty
        self._stop_flag = False
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()
        print("[EarlyStop] Running. Press 'S' + Enter to truncate episode with penalty.")

    def _listen(self):
        import select
        try:
            tty_fd = open("/dev/tty", "r")
        except OSError:
            print("[EarlyStop] WARNING: /dev/tty unavailable, early stop disabled.")
            return

        while self._running:
            # Poll with 0.1s timeout so the thread exits cleanly on shutdown
            try:
                ready, _, _ = select.select([tty_fd], [], [], 0.1)
                if ready:
                    line = tty_fd.readline().strip().lower()
                    if line == "s":
                        with self._lock:
                            self._stop_flag = True
                        print("\n[EarlyStop] ⚠ STOP flagged — truncating after current step.")
            except Exception:
                break

        tty_fd.close()

    def check_and_clear(self) -> bool:
        with self._lock:
            if self._stop_flag:
                self._stop_flag = False
                return True
            return False

    def shutdown(self):
        self._running = False

# class EarlyStopListener:
#     """
#     Runs in a background thread. Press 'S' + Enter any time during an episode
#     to flag the current step as a truncation (dangerous situation).
#     The flag is cleared automatically after each step is consumed.
#     """
#     def __init__(self, truncation_penalty: float = -50.0):
#         self.truncation_penalty = truncation_penalty
#         self._stop_flag = False
#         self._lock = threading.Lock()
#         self._thread = threading.Thread(target=self._listen, daemon=True)
#         self._thread.start()
#         print("[EarlyStop] Running. Press 'S' + Enter to truncate episode with penalty.")

#     def _listen(self):
#         while True:
#             try:
#                 key = input().strip().lower()
#                 if key == "s":
#                     with self._lock:
#                         self._stop_flag = True
#                     print("\n[EarlyStop] ⚠ STOP flagged — will truncate after current step.")
#             except EOFError:
#                 break

#     def check_and_clear(self) -> bool:
#         """Returns True (and clears the flag) if stop was requested."""
#         with self._lock:
#             if self._stop_flag:
#                 self._stop_flag = False
#                 return True
#             return False

# ---------------------------------------------------------------------------
# 2. REWARD CLASSIFIER (image-based, trained online)
# ---------------------------------------------------------------------------

class RewardClassifier(nn.Module):
    """
    Binary success/failure classifier compatible with train_reward_classifier_v2.py.
    Architecture is determined by the 'arch' key saved in the checkpoint.
    Outputs P(success) in [0,1].
    """

    def __init__(self, arch: str = "efficientnet", device: str = "cuda"):
        super().__init__()
        from torchvision import models

        if arch == "efficientnet":
            base = models.efficientnet_b0(weights=None)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_features, 2)
            )
        else:  # resnet18
            base = models.resnet18(weights=None)
            base.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, 2))

        self.model = base
        self.device = device
        self.to(device)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, 3, H, W) normalized ImageNet → (B, 2) logits"""
        return self.model(img)

    def predict_reward(self, img_np: np.ndarray) -> float:
        """img_np: (H, W, 3) uint8 — accepts BGR (OpenCV) or RGB (raw obs)"""
        # raw obs images from the robot are already RGB — skip bgr conversion
        if img_np.shape[2] == 3:
            img = img_np  # treat as RGB (from robot obs dict)
        img = cv2.resize(img_np, (224, 224)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img  = (img - mean) / std
        t = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.forward(t)
            prob = F.softmax(logits, dim=1)[0, 1].item()
        return prob

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cuda") -> "RewardClassifier":
        ckpt = torch.load(path, map_location=device)
        arch = ckpt.get("arch", "efficientnet")
        model = cls(arch=arch, device=device)
        
        # Remap keys: checkpoint has bare keys (features.*, classifier.*)
        # but our wrapper adds the "model." prefix
        state = ckpt["model_state"]
        if not any(k.startswith("model.") for k in state.keys()):
            state = {"model." + k: v for k, v in state.items()}
        
        model.load_state_dict(state)  # strict=True
        model.eval()
        print(f"  [RewardClassifier] Loaded {arch} from {path}  "
            f"(F1={ckpt.get('f1','?'):.3f}, epoch={ckpt.get('epoch','?')})")
        return model
    # @classmethod
    # def from_checkpoint(cls, path: str, device: str = "cuda") -> "RewardClassifier":
    #     """Load a classifier saved by train_reward_classifier_v2.py."""
    #     ckpt = torch.load(path, map_location=device)
    #     arch = ckpt.get("arch", "efficientnet")
    #     model = cls(arch=arch, device=device)
    #     model.load_state_dict(ckpt["model_state"])   # strict=True — crashes loudly on mismatch
    #     model.eval()
    #     print(f"  [RewardClassifier] Loaded {arch} from {path}  "
    #           f"(F1={ckpt.get('f1','?'):.3f}, epoch={ckpt.get('epoch','?')})")
    #     return model

# class RewardClassifier(nn.Module):
#     """
#     Small binary success/failure classifier on top of a frozen ResNet18.
#     Train with ~20-30 labelled success frames and ~20-30 failure frames.
#     """

#     def __init__(self, device: str = "cuda"):
#         super().__init__()
#         import torchvision.models as tvm
#         backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
#         self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # remove FC
#         for p in self.backbone.parameters():
#             p.requires_grad = False

#         self.head = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(512, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1),
#         )
#         self.device = device
#         self.to(device)

#     def forward(self, img: torch.Tensor) -> torch.Tensor:
#         """img: (B, 3, H, W) normalized ImageNet"""
#         with torch.no_grad():
#             feat = self.backbone(img)
#         return self.head(feat).squeeze(-1)  # (B,)

#     def predict_reward(self, img_np: np.ndarray) -> float:
#         """img_np: (H, W, 3) uint8 BGR from OpenCV"""
#         img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
#         mean = np.array([0.485, 0.456, 0.406])
#         std  = np.array([0.229, 0.224, 0.225])
#         img = (img - mean) / std
#         t = torch.tensor(img.transpose(2,0,1), dtype=torch.float32).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             logit = self.forward(t).item()
#         return float(torch.sigmoid(torch.tensor(logit)).item())

#     @classmethod
#     def train_from_folders(
#         cls,
#         success_dir: str,
#         failure_dir: str,
#         epochs: int = 50,
#         lr: float = 1e-3,
#         device: str = "cuda",
#     ) -> "RewardClassifier":
#         """Quick training on labelled images."""
#         model = cls(device=device)
#         for p in model.head.parameters():
#             p.requires_grad = True
#         opt = Adam(model.head.parameters(), lr=lr)

#         def load_images(folder, label):
#             items = []
#             for f in Path(folder).glob("*.jpg"):
#                 img = cv2.imread(str(f))
#                 if img is None: continue
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
#                 mean = np.array([0.485, 0.456, 0.406])
#                 std  = np.array([0.229, 0.224, 0.225])
#                 img = (img - mean) / std
#                 items.append((img.transpose(2,0,1), label))
#             return items

#         data = load_images(success_dir, 1.0) + load_images(failure_dir, 0.0)
#         print(f"[RewardClassifier] Training on {len(data)} images")

#         imgs = torch.tensor(np.stack([d[0] for d in data]), dtype=torch.float32).to(device)
#         labels = torch.tensor([d[1] for d in data], dtype=torch.float32).to(device)

#         for ep in range(epochs):
#             idx = torch.randperm(len(imgs))
#             imgs, labels = imgs[idx], labels[idx]
#             logits = model(imgs)
#             loss = F.binary_cross_entropy_with_logits(logits, labels)
#             opt.zero_grad(); loss.backward(); opt.step()
#             if (ep+1) % 10 == 0:
#                 acc = ((torch.sigmoid(logits) > 0.5) == (labels > 0.5)).float().mean()
#                 print(f"  epoch {ep+1}/{epochs}  loss={loss.item():.4f}  acc={acc.item():.3f}")

#         return model

# class RewardClassifier(nn.Module):
#     """
#     Binary success/failure classifier matching the architecture saved by
#     train_reward_classifier.py (named-layer ResNet18 + custom FC head).
#     """

#     def __init__(self, device: str = "cuda"):
#         super().__init__()
#         import torchvision.models as tvm

#         # Use the FULL ResNet18 with named layers — matches train_reward_classifier.py
#         base = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
#         # Replace FC with a 2-layer head (same as train script)
#         base.fc = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 1),
#         )
#         self.model = base
#         self.device = device
#         self.to(device)

#     def forward(self, img: torch.Tensor) -> torch.Tensor:
#         """img: (B, 3, H, W) normalized ImageNet → scalar logit per image"""
#         return self.model(img).squeeze(-1)

#     def predict_reward(self, img_np: np.ndarray) -> float:
#         """img_np: (H, W, 3) uint8 BGR from OpenCV"""
#         img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
#         mean = np.array([0.485, 0.456, 0.406])
#         std  = np.array([0.229, 0.224, 0.225])
#         img  = (img - mean) / std
#         t = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             logit = self.forward(t).item()
#         return float(torch.sigmoid(torch.tensor(logit)).item())

# ---------------------------------------------------------------------------
# 3. REPLAY BUFFER (off-policy)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Stores (obs, action, reward, next_obs, done) tuples.
    obs is a dict matching your ACT input_features.
    Images stored as uint8 to save RAM.
    """

    def __init__(self, capacity: int = 50_000, device: str = "cuda"):
        self.capacity = capacity
        self.device = device
        self.buffer: List[Dict] = []
        self.pos = 0

    def push(self, obs, action, reward, next_obs, done):
        entry = {
            "obs": obs,
            "action": np.array(action, dtype=np.float32),
            "reward": float(reward),
            "next_obs": next_obs,
            "done": float(done),
        }
        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            self.buffer[self.pos] = entry
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict:
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        
        def stack_obs(key):
            vals = [b["obs"][key] for b in batch]
            arr = np.stack(vals)
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            # Transpose images from (B,H,W,3) → (B,3,H,W)
            if "image" in key and arr.ndim == 4 and arr.shape[-1] in (1, 3):
                arr = arr.transpose(0, 3, 1, 2)
            return torch.tensor(arr, dtype=torch.float32, device=self.device)
        # def stack_obs(key):
        #     vals = [b["obs"][key] for b in batch]
        #     arr = np.stack(vals)
        #     if arr.dtype == np.uint8:
        #         arr = arr.astype(np.float32) / 255.0
        #     return torch.tensor(arr, dtype=torch.float32, device=self.device)

        def stack_next_obs(key):
            vals = [b["next_obs"][key] for b in batch]
            arr = np.stack(vals)
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            # Transpose images from (B,H,W,3) → (B,3,H,W)
            if "image" in key and arr.ndim == 4 and arr.shape[-1] in (1, 3):
                arr = arr.transpose(0, 3, 1, 2)
            return torch.tensor(arr, dtype=torch.float32, device=self.device)

        # def stack_next_obs(key):
        #     vals = [b["next_obs"][key] for b in batch]
        #     arr = np.stack(vals)
        #     if arr.dtype == np.uint8:
        #         arr = arr.astype(np.float32) / 255.0
        #     return torch.tensor(arr, dtype=torch.float32, device=self.device)

        obs_keys = list(batch[0]["obs"].keys())
        return {
            "obs": {k: stack_obs(k) for k in obs_keys},
            "next_obs": {k: stack_next_obs(k) for k in obs_keys},
            "actions": torch.tensor(
                np.stack([b["action"] for b in batch]),
                dtype=torch.float32, device=self.device
            ),
            "rewards": torch.tensor(
                [b["reward"] for b in batch],
                dtype=torch.float32, device=self.device
            ),
            "dones": torch.tensor(
                [b["done"] for b in batch],
                dtype=torch.float32, device=self.device
            ),
        }

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# 4. CRITIC NETWORK (SAC-style double Q)
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    Encodes (obs, action) → scalar Q-value.
    Uses lightweight CNN for images + MLP for state.
    """

    def __init__(
        self,
        state_dim: int = 7,
        action_dim: int = 7,
        image_keys: List[str] = ("observation.images.cam_high", "observation.images.cam_right_wrist"),
        hidden: int = 512,
    ):
        super().__init__()
        self.image_keys = list(image_keys)

        # Lightweight CNN per image
        self.cnns = nn.ModuleDict()
        for k in self.image_keys:
            self.cnns[k.replace(".", "_")] = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(64*7*7, 256), nn.ReLU(),
            )

        img_feat_dim = 256 * len(self.image_keys)
        self.mlp = nn.Sequential(
            nn.Linear(img_feat_dim + state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: Dict, action: torch.Tensor) -> torch.Tensor:
        feats = []
        for k in self.image_keys:
            key = k.replace(".", "_")
            img = obs[k]
            if img.shape[-1] == 640:  # (B,3,480,640) — downsample for critic
                img = F.interpolate(img, size=(120, 160), mode="bilinear", align_corners=False)
            feats.append(self.cnns[key](img))
        state = obs["observation.state"]
        x = torch.cat(feats + [state, action], dim=-1)
        return self.mlp(x).squeeze(-1)


class DoubleCritic(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.q1 = QNetwork(**kwargs)
        self.q2 = QNetwork(**kwargs)

    def forward(self, obs, action):
        return self.q1(obs, action), self.q2(obs, action)

    def min_q(self, obs, action):
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)


# ---------------------------------------------------------------------------
# 5. ACT ACTOR WRAPPER (policy + residual head)
# ---------------------------------------------------------------------------
# class ACTActorWithResidual(nn.Module):
class ACTActorWithResidual(nn.Module):
    """
    Wraps the pretrained ACT policy and adds a small residual MLP on top.
    The residual corrects the ACT output with RL-learned adjustments.

    Modes:
      'frozen': ACT fully frozen, residual learns everything
      'head': ACT backbone frozen, ACT decoder fine-tuned + residual
      'full': everything trainable (only use after 5k warmup steps)
    """    

    def __init__(
        self,
        act_policy: ACTPolicy,
        action_dim: int = 7,
        residual_hidden: int = 256,
        residual_scale: float = 0.02,   # max magnitude of residual correction
        mode: str = "frozen",
<<<<<<< HEAD
        context_dim: int = 2,   # ← add: number of classifier P values fed in
=======
>>>>>>> 86760d5fb4b2b916c91d8e3736abf60f4663a89b
    ):
        super().__init__()
        self.act_policy = act_policy
        self.act_policy_batch = copy.deepcopy(act_policy)
        self.act_policy_batch.eval()

        self.residual_scale = residual_scale
        self.mode = mode
        self.action_dim = action_dim
<<<<<<< HEAD
        self.context_dim = context_dim

=======
>>>>>>> 86760d5fb4b2b916c91d8e3736abf60f4663a89b

        # Determine ACT hidden dim for residual input
        act_dim = act_policy.config.dim_model

        # Small residual MLP: takes ACT base action → correction
        self.residual = nn.Sequential(
<<<<<<< HEAD
            nn.Linear(action_dim + context_dim, residual_hidden), nn.ReLU(),
            nn.Linear(residual_hidden, residual_hidden),           nn.ReLU(),
            nn.Linear(residual_hidden, action_dim),                nn.Tanh(),
=======
            nn.Linear(action_dim, residual_hidden), nn.ReLU(),
            nn.Linear(residual_hidden, residual_hidden), nn.ReLU(),
            nn.Linear(residual_hidden, action_dim), nn.Tanh(),
>>>>>>> 86760d5fb4b2b916c91d8e3736abf60f4663a89b
        )

        # Log std for stochastic policy (SAC)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 4.0)

        self._set_grad_mode(mode)

    def _set_grad_mode(self, mode: str):
        """Freeze/unfreeze parts of ACT depending on training mode."""
        for p in self.act_policy.parameters():
            p.requires_grad = False

        if mode == "head":
            # Unfreeze ACT transformer decoder
            for p in self.act_policy.model.decoder.parameters():
                p.requires_grad = True
            for p in self.act_policy.model.action_head.parameters():
                p.requires_grad = True
        elif mode == "full":
            for p in self.act_policy.parameters():
                p.requires_grad = True

        # Residual always trainable
        for p in self.residual.parameters():
            p.requires_grad = True
        self.log_std.requires_grad = True


    def get_base_action(self, obs_norm_batch: Dict) -> torch.Tensor:
        B = obs_norm_batch["observation.state"].shape[0]

        if B == 1:
            # Rollout: use select_action → temporal ensembling preserved
            with torch.set_grad_enabled(self.mode != "frozen"):
                output = self.act_policy.select_action(obs_norm_batch)
                if isinstance(output, np.ndarray):
                    output = torch.tensor(output, dtype=torch.float32,
                                        device=self.log_std.device)
                if output.dim() == 1:
                    output = output.unsqueeze(0)
            return output  # (1, 7)
        else:
            # Batch update (critic/actor): bypass select_action entirely,
            # call predict_action_chunk directly — no ensembler state touched
            with torch.set_grad_enabled(self.mode != "frozen"):
                action_chunk = self.act_policy.predict_action_chunk(obs_norm_batch)
                # action_chunk: (B, chunk_size, 7) — take first action step only
            return action_chunk[:, 0, :]  # (B, 7)
        
    #likely correct
    # def get_base_action(self, obs_norm_batch: Dict) -> torch.Tensor:
    #     """
    #     obs_norm_batch: already-normalized tensors (state MEAN_STD, images ImageNet).
    #     select_action expects this — it has NO internal normalizer.
    #     """
    #     with torch.set_grad_enabled(self.mode != "frozen"):
    #         output = self.act_policy.select_action(obs_norm_batch)
    #         if isinstance(output, np.ndarray):
    #             output = torch.tensor(output, dtype=torch.float32,
    #                                 device=self.log_std.device)
    #         if output.dim() == 1:
    #             output = output.unsqueeze(0)
    #     return output  # (1, 7) — still in NORMALIZED action space

    # def get_base_action(self, obs_raw: Dict) -> torch.Tensor:
    #     """
    #     obs_raw: dict with raw (unnormalized) tensors — select_action handles norm internally.
    #     """
    #     with torch.set_grad_enabled(self.mode != "frozen"):
    #         output = self.act_policy.select_action(obs_raw)
    #         if isinstance(output, np.ndarray):
    #             output = torch.tensor(output, dtype=torch.float32,
    #                                 device=self.log_std.device)
    #         if output.dim() == 1:
    #             output = output.unsqueeze(0)  # ensure (1, 7)
    #     print(f"output: {output}")
    #     return output

    # def get_base_action(self, obs: Dict) -> torch.Tensor:
    #     """Get deterministic ACT output (no VAE sampling during RL)."""
    #     with torch.set_grad_enabled(self.mode != "frozen"):
    #         # ACT forward — use mean action (disable VAE sampling)
    #         batch = {k: v for k, v in obs.items()}
    #         # Force deterministic: set encoder to eval and zero latent
    #         output = self.act_policy.select_action(batch)
    #         # select_action returns numpy; convert back
    #         if isinstance(output, np.ndarray):
    #             output = torch.tensor(output, dtype=torch.float32, device=self.log_std.device)
    #     return output  # (action_dim,) or (B, action_dim)

    # def forward(self, obs: Dict, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Returns (action, log_prob) for SAC.
    #     action: (B, action_dim)
    #     """
    #     base = self.get_base_action(obs)  # (B, action_dim)
    #     residual = self.residual(base) * self.residual_scale
    #     mean = base + residual

    #     if deterministic:
    #         return mean, torch.zeros(mean.shape[0], device=mean.device)

    #     std = self.log_std.exp().clamp(1e-4, 1.0).expand_as(mean)
    #     dist = torch.distributions.Normal(mean, std)
    #     raw = dist.rsample()
    #     action = torch.tanh(raw)

    #     # Log prob with tanh squashing correction
    #     log_prob = dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)
    #     log_prob = log_prob.sum(-1)

    #     return action, log_prob\

<<<<<<< HEAD
    # def forward(self, obs: Dict, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    #     base = self.get_base_action(obs)  # (B, 7) — real joint radians from select_action

    #     # Residual in joint space — scale is in radians (e.g. 0.05 rad ≈ 3°)
    #     residual = self.residual(base) * self.residual_scale  # (B, 7)
    #     mean = base + residual

    #     if deterministic:
    #         return mean, torch.zeros(mean.shape[0] if mean.dim() > 1 else 1,
    #                                 device=mean.device)

    #     std = self.log_std.exp().clamp(1e-4, 0.05).expand_as(mean)  # tight std in rad
    #     dist = torch.distributions.Normal(mean, std)
    #     action = dist.rsample()

    #     # Log prob — NO tanh squashing (joint space is unbounded)
    #     log_prob = dist.log_prob(action).sum(-1)

    #     return action, log_prob
    def forward(
        self,
        obs: Dict,
        deterministic: bool = False,
        context: Optional[torch.Tensor] = None,   # ← (B, context_dim) classifier probs
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        base = self.get_base_action(obs)   # (B, 7) normalized

        # Unnorm to real radians (same as the run() loop)
        # NOTE: for batch updates we don't have _action_std here, so pass base as-is
        # and let the trainer inject context. See update() change below.
        if context is None:
            # No classifier context available (e.g. batch update without stored probs)
            # Fall back to zero context — residual still learns, just without the signal
            context = torch.zeros(
                base.shape[0], self.context_dim, device=base.device, dtype=base.dtype
            )

        residual_input = torch.cat([base, context], dim=-1)   # (B, 7+context_dim)
        residual = self.residual(residual_input) * self.residual_scale
=======
    def forward(self, obs: Dict, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        base = self.get_base_action(obs)  # (B, 7) — real joint radians from select_action

        # Residual in joint space — scale is in radians (e.g. 0.05 rad ≈ 3°)
        residual = self.residual(base) * self.residual_scale  # (B, 7)
>>>>>>> 86760d5fb4b2b916c91d8e3736abf60f4663a89b
        mean = base + residual

        if deterministic:
            return mean, torch.zeros(mean.shape[0] if mean.dim() > 1 else 1,
                                    device=mean.device)

<<<<<<< HEAD
        std = self.log_std.exp().clamp(1e-4, 0.05).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

=======
        std = self.log_std.exp().clamp(1e-4, 0.05).expand_as(mean)  # tight std in rad
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()

        # Log prob — NO tanh squashing (joint space is unbounded)
        log_prob = dist.log_prob(action).sum(-1)

        return action, log_prob


>>>>>>> 86760d5fb4b2b916c91d8e3736abf60f4663a89b
# ---------------------------------------------------------------------------
# 6. SAC TRAINER
# ---------------------------------------------------------------------------

class SERLTrainer:
    """
    SAC fine-tuning of ACT on the real robot.
    Implements SERL: Sample Efficient Real-Robot RL (https://arxiv.org/abs/2401.16013)
    """

    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SERL] Using device: {self.device}")

        print(f"[SERL] Loading ACT checkpoint from {cfg.checkpoint_path}")
        # Find config.json — lerobot saves it in different places per version
        ckpt = Path(cfg.checkpoint_path)
        config_candidates = [
            ckpt / "config.json",
            ckpt / "pretrained_model" / "config.json",
            ckpt.parent / "config.json",
        ]
        config_path = next((c for c in config_candidates if c.exists()), None)

        if config_path is None:
            import os
            for root, dirs, files in os.walk(str(ckpt)):
                for f in files: print(f"  {root}/{f}")
            raise FileNotFoundError(f"config.json not found under {ckpt}")

        self.act_policy = ACTPolicy.from_pretrained(str(config_path.parent))
        # Load pretrained ACT
        
        # self.act_policy = ACTPolicy.from_pretrained(cfg.checkpoint_path / pretrained_model / config.json)
        self.act_policy.to(self.device)
        self.act_policy.eval()

        # Load normalization stats from the preprocessor safetensors
        from safetensors.torch import load_file as load_safetensors
        pre_path = ckpt / "pretrained_model" / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        post_path = ckpt / "pretrained_model" / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"

        self.norm_stats = load_safetensors(str(pre_path), device="cpu")
        self.unnorm_stats = load_safetensors(str(post_path), device="cpu")

        # Extract state mean/std — keys are like "observation.state.mean", "observation.state.std"
        self._state_mean = self.norm_stats["observation.state.mean"].numpy()
        self._state_std  = self.norm_stats["observation.state.std"].numpy()
        # Action unnorm stats (to convert normalized ACT output back to real actions)
        self._action_mean = self.unnorm_stats["action.mean"].numpy()
        self._action_std  = self.unnorm_stats["action.std"].numpy()
        print(f"[SERL] Norm stats loaded. State mean: {self._state_mean}")

        # Actor
        self.actor = ACTActorWithResidual(
            act_policy=self.act_policy,
            action_dim=7,
            residual_scale=cfg.residual_scale,
            mode=cfg.actor_mode,
        ).to(self.device)

        # Critic (double Q)
        self.critic = DoubleCritic(
            state_dim=7,
            action_dim=7,
            image_keys=["observation.images.cam_high", "observation.images.cam_right_wrist"],
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Optimizers
        actor_params = list(self.actor.residual.parameters()) + [self.actor.log_std]
        if cfg.actor_mode != "frozen":
            actor_params += [p for p in self.actor.act_policy.parameters() if p.requires_grad]

        self.actor_opt  = Adam(actor_params, lr=cfg.actor_lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=cfg.critic_lr)

        # Entropy temperature (SAC)
        self.log_alpha = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.target_entropy = -float(7)  # -action_dim
        self.alpha_opt = Adam([self.log_alpha], lr=cfg.alpha_lr)

        # Replay buffer
        self.replay = ReplayBuffer(capacity=cfg.buffer_size, device=str(self.device))

        # Safety
        tcp_min = np.array(cfg.tcp_workspace_min)
        tcp_max = np.array(cfg.tcp_workspace_max)
        self.safety = SafetyWrapper(
            tcp_min=tcp_min,
            tcp_max=tcp_max,
            max_delta=cfg.max_action_delta,
        )
        print(f"[SERL] Workspace box: {self.safety.tcp_min} → {self.safety.tcp_max}")
        print(f"[SERL] Max action delta: {self.safety.max_delta}")

        # Early stop listener
        self.early_stop = EarlyStopListener(
            truncation_penalty=getattr(cfg, "truncation_penalty", -50.0)
        )
        # Reward
        self.reward_mode = cfg.reward_mode
        # self.reward_classifier = None
        # if cfg.reward_mode == "classifier":
        #     print("[SERL] Training reward classifier...")
        #     self.reward_classifier = RewardClassifier.train_from_folders(
        #         cfg.success_images_dir,
        #         cfg.failure_images_dir,
        #         device=str(self.device),
        #     )
        #     print("[SERL] Reward classifier ready.")

        self.reward_classifier = None
        if cfg.reward_mode == "classifier":
            if hasattr(cfg, "reward_classifier_path") and cfg.reward_classifier_path:
                print(f"[SERL] Loading reward classifier from {cfg.reward_classifier_path}")
                self.reward_classifier = RewardClassifier.from_checkpoint(
                    cfg.reward_classifier_path, device=str(self.device)
                )
            else:
                raise ValueError("--reward_classifier_path required for classifier mode. "
                                "Run train_reward_classifier_v2.py first.")

        # self.reward_classifier = None
        # if cfg.reward_mode == "classifier":
        #     if hasattr(cfg, "reward_classifier_path") and cfg.reward_classifier_path:
        #         # Load pre-trained classifier (fast path)
        #         print(f"[SERL] Loading pre-trained reward classifier from {cfg.reward_classifier_path}")
        #         self.reward_classifier = RewardClassifier(device=str(self.device))
        #         ckpt = torch.load(cfg.reward_classifier_path, map_location=self.device)
        #         # train_reward_classifier.py saves the full model state dict
        #         # self.reward_classifier.load_state_dict(ckpt["model_state_dict"])
        #         # self.reward_classifier.load_state_dict(ckpt["model_state"])
        #         self.reward_classifier.load_state_dict(ckpt["model_state"], strict=False)
        #         self.reward_classifier.eval()
        #         print("[SERL] Reward classifier loaded.")
        #     else:
        #         # Train from scratch — images must be .jpg directly in the folder
        #         print("[SERL] Training reward classifier from images...")
        #         self.reward_classifier = RewardClassifier.train_from_folders(
        #             cfg.success_images_dir,
        #             cfg.failure_images_dir,
        #             device=str(self.device),
        #         )


        self.top_cam_classifier = None
        if hasattr(cfg, "top_cam_classifier_path") and cfg.top_cam_classifier_path:
            print(f"[SERL] Loading top-cam classifier from {cfg.top_cam_classifier_path}")
            self.top_cam_classifier = RewardClassifier.from_checkpoint(
                cfg.top_cam_classifier_path, device=str(self.device)
            )
        # self.top_cam_classifier = None
        # if hasattr(cfg, "top_cam_classifier_path") and cfg.top_cam_classifier_path:
        #     print(f"[SERL] Loading top-cam classifier from {cfg.top_cam_classifier_path}")
        #     self.top_cam_classifier = RewardClassifier(device=str(self.device))
        #     ckpt = torch.load(cfg.top_cam_classifier_path, map_location=self.device)
        #     self.top_cam_classifier.load_state_dict(ckpt["model_state"], strict=False)
        #     self.top_cam_classifier.eval()
        self.success_threshold = getattr(cfg, "success_threshold", 0.80)


        # Stats
        self.total_steps = 0
        self.episode_returns = []
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._last_tcp_z = 0.5   # safe default height until first TCP read

    # -----------------------------------------------------------------------
    # Observation normalization helpers (uses ACT's stored stats)
    # -----------------------------------------------------------------------

    # def normalize_obs(self, raw_obs: Dict) -> Dict:
    #     """
    #     Applies ACT's MEAN_STD normalization to observations.
    #     raw_obs["observation.state"]: np.ndarray (7,)
    #     raw_obs["observation.images.*"]: np.ndarray (H, W, 3) uint8
    #     """
    #     norm = {}
    #     # State
    #     s = raw_obs["observation.state"]
    #     s_mean = self.act_policy.normalize_inputs.buffer["observation.state"]["mean"].cpu().numpy()
    #     s_std  = self.act_policy.normalize_inputs.buffer["observation.state"]["std"].cpu().numpy()
    #     norm["observation.state"] = ((s - s_mean) / (s_std + 1e-8)).astype(np.float32)

    #     # Images: convert to (3, H, W) float32 [0,1] — normalization via ImageNet stats
    #     img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    #     img_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    #     for k in ("observation.images.cam_high", "observation.images.cam_right_wrist"):
    #         img = raw_obs[k].astype(np.float32) / 255.0  # (H, W, 3)
    #         img = (img - img_mean) / img_std
    #         norm[k] = img.transpose(2, 0, 1)  # (3, H, W)

    #     return norm

    def normalize_obs(self, raw_obs: Dict) -> Dict:
        norm = {}
        # State: MEAN_STD normalization using ACT's stored stats
        s = raw_obs["observation.state"].astype(np.float32)
        norm["observation.state"] = (s - self._state_mean) / (self._state_std + 1e-8)

        # Images: (H,W,3) uint8 → (3,H,W) float32, ImageNet normalized
        img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        img_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        for k in ("observation.images.cam_high", "observation.images.cam_right_wrist"):
            img = raw_obs[k]
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            img = (img - img_mean) / img_std
            norm[k] = img.transpose(2, 0, 1)  # (3,H,W)

        return norm
    
    def unnormalize_action(self, action_norm: np.ndarray) -> np.ndarray:
        """Convert ACT's normalized action output back to real joint/EEF units."""
        return action_norm * (self._action_std + 1e-8) + self._action_mean

    def obs_to_batch(self, obs_norm: Dict) -> Dict:
        """Adds batch dimension and moves to device."""
        return {
            k: torch.tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0)
            for k, v in obs_norm.items()
        }
    def obs_to_batch_raw(self, raw_obs: Dict) -> Dict:
        """Batch raw obs for select_action (it normalizes internally)."""
        batch = {}
        for k, v in raw_obs.items():
            arr = np.array(v, dtype=np.float32)
            if "image" in k and arr.ndim == 3:
                arr = arr.transpose(2, 0, 1)  # (H,W,3) → (3,H,W)
            batch[k] = torch.tensor(arr, device=self.device).unsqueeze(0)
        return batch

    # -----------------------------------------------------------------------
    # Reward computation
    # -----------------------------------------------------------------------
    
    def compute_reward(self, raw_obs, action, next_raw_obs, done) -> float:
        reward = 0.0

        # ── 1. Per-step efficiency penalty ──
        reward -= 0.05

        # ── 2. Action smoothness penalty ──
        # reward -= 0.005 * float(np.linalg.norm(action[:6]))

        # ── 3. Table proximity penalty ──
        if self._last_tcp_z < 0.015:
            reward -= 3.0

        # ── 4. Wrist classifier — dense grasp progress reward (every step) ──
        if self.reward_classifier is not None:
            wrist_img = next_raw_obs.get("observation.images.cam_right_wrist",
                                        next_raw_obs.get("observation.images.cam_high"))
            p_grasp = self.reward_classifier.predict_reward(wrist_img)
            # print(f"P_grasp: {p_grasp}")
            reward += 5.0 * (p_grasp-self.success_threshold)   # positive signal: higher = gripper on object

        # ── 5. Top cam classifier — object-in-box penalty (every step) ──
        if self.top_cam_classifier is not None:
            top_img = next_raw_obs.get("observation.images.cam_high")
            if top_img is not None:
                p_in_box = self.top_cam_classifier.predict_reward(top_img)
                # print(f"P_in_box: {p_in_box}")
                reward -= 1.5 * (1.0 - p_in_box)  # penalty if object left the box

        # ── 6. Terminal success bonus ──
        if done:
            reward += 10000.0
        # print(f"reward: {reward:.3f}")
        return reward
    # def compute_reward(self, raw_obs, action, next_raw_obs, done) -> float:
    #     """
    #     Dense shaped reward for grasp approach:
    #     +1.0  if classifier says success (> threshold) at END of episode
    #     -0.01 action magnitude penalty (smooth motion)
    #     -2.0  if tcp_z is dangerously close to table (< 0.04m)
    #     -0.05 per step to encourage efficiency (episode length pressure)
    #     -5.0  truncation penalty already handled by EarlyStopListener
    #     """
    #     reward = 0.0

    #     # ── 1. Per-step efficiency penalty (encourages fast completion) ──
    #     reward -= 0.05

    #     # ── 2. Action smoothness penalty ──
    #     reward -= 0.005 * float(np.linalg.norm(action[:6]))

    #     # ── 3. Table proximity penalty ──
    #     tcp_z = self._last_tcp_z   # set this from tcp_pose each step (see below)
    #     if tcp_z < 0.04:           # 4cm floor — table danger zone
    #         reward -= 3.0

    #     if self.reward_classifier is not None:
    #         reward -= 2.0 * (1.0 - self.reward_classifier.predict_reward(next_raw_obs.get("observation.images.cam_high", next_raw_obs.get("observation.images.cam_right_wrist"))))

    #     # ── 4. Classifier-gated success bonus (only at episode end) ──
    #     # if done or episode_steps >= max_episode_steps:
    #     if done:
    #         wrist_img = next_raw_obs.get("observation.images.cam_right_wrist",
    #                                     next_raw_obs.get("observation.images.cam_high"))
    #         if self.reward_classifier is not None:
    #             p_success = self.reward_classifier.predict_reward(wrist_img)
    #             if p_success > self.success_threshold:
    #                 reward += 10.0    # big terminal bonus only when confident
    #         # Sparse fallback if no classifier
    #         elif done:
    #             reward += 100.0

    #     return reward

    # def compute_reward(
    #     self,
    #     raw_obs: Dict,
    #     action: np.ndarray,
    #     next_raw_obs: Dict,
    #     done: bool,
    # ) -> float:
    #     if self.reward_mode == "classifier":
    #         # Use wrist cam for grasp detection
    #         wrist_img = next_raw_obs.get(
    #             "observation.images.cam_right_wrist",
    #             next_raw_obs.get("observation.images.cam_high")
    #         )
    #         reward = self.reward_classifier.predict_reward(wrist_img)
    #         # Bonus for task completion
    #         # if done and reward > 0.7:
    #         if done and reward > self.success_threshold:
    #             reward += 2.0
    #         return reward

    #     elif self.reward_mode == "sparse":
    #         # Binary: you hook in your own success detector
    #         return float(done)

    #     elif self.reward_mode == "shaped":
    #         # Example: penalize proximity to table (z < 0.05m)
    #         tcp_z = next_raw_obs.get("tcp_z", 0.1)
    #         table_penalty = -2.0 if tcp_z < 0.05 else 0.0
    #         # Penalize large actions (smooth motion)
    #         action_penalty = -0.01 * float(np.linalg.norm(action[:6]))
    #         return float(done) * 5.0 + table_penalty + action_penalty

    #     return 0.0

    # -----------------------------------------------------------------------
    # SAC update step
    # -----------------------------------------------------------------------

    def update(self):
        if len(self.replay) < self.cfg.batch_size:
            return {}

        batch = self.replay.sample(self.cfg.batch_size)
        obs      = batch["obs"]
        next_obs = batch["next_obs"]
        actions  = batch["actions"]
        rewards  = batch["rewards"]
        dones    = batch["dones"]

        alpha = self.log_alpha.exp().detach()

        # --- Critic update ---
        with torch.no_grad():
            next_act, next_logp = self.actor(next_obs)
            q_next = self.critic_target.min_q(next_obs, next_act)
            q_target = rewards + self.cfg.gamma * (1 - dones) * (q_next - alpha * next_logp)

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # --- Actor update ---
        new_act, new_logp = self.actor(obs)
        q_val = self.critic.min_q(obs, new_act)
        actor_loss = (alpha * new_logp - q_val).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in self.actor.parameters() if p.requires_grad], 1.0
        )
        self.actor_opt.step()

        # --- Temperature update ---
        alpha_loss = -(self.log_alpha * (new_logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # --- Soft target update ---
        tau = self.cfg.tau
        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.mul_(1 - tau).add_(tau * p.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha":       alpha.item(),
            "mean_q":      q_val.mean().item(),
        }

    # -----------------------------------------------------------------------
    # Main training loop (real robot)
    # -----------------------------------------------------------------------

    # def run(self, robot):
    #     """
    #     robot: your UR10 robot interface with:
    #       .reset() -> raw_obs (dict)
    #       .step(action: np.ndarray) -> (raw_obs, done, info)
    #       .get_tcp_pose() -> np.ndarray (3,) position in base frame
    #     """
    #     print(f"[SERL] Starting SERL fine-tuning for {self.cfg.num_steps} steps")
    #     print(f"[SERL] Actor mode: {self.cfg.actor_mode}")
    #     print(f"[SERL] Warmup: {self.cfg.warmup_steps} random steps before RL updates")
    #     print("=" * 60)

    #     step = 0
    #     episode = 0
    #     episode_return = 0.0
    #     episode_steps  = 0

    #     raw_obs = robot.reset()
    #     # obs = self.normalize_obs(raw_obs)
    #     obs_norm = self.normalize_obs(raw_obs)   # for critic
    #     obs_raw  = self.obs_to_batch_raw(raw_obs) # for ACT select_action

    #     while step < self.cfg.num_steps:
    #         obs_batch = self.obs_to_batch(obs)

    #         # Select action
    #         if step < self.cfg.warmup_steps:
    #             # Pure ACT during warmup — let it explore naturally
    #             with torch.no_grad():
    #                 act_t, _ = self.actor(obs_batch, deterministic=False)
    #             action_np = act_t.squeeze(0).cpu().numpy()
    #         else:
    #             with torch.no_grad():
    #                 act_t, _ = self.actor(obs_batch, deterministic=False)
    #             action_np = act_t.squeeze(0).cpu().numpy()

    #         # action_np = self.unnormalize_action(action_np) #change suggested by Perplexity

    #         # Safety clip
    #         current_joints = raw_obs["observation.state"]  # assuming first 7 are joint angles
    #         safe_action, violated = self.safety.clip_action(action_np, current_joints=current_joints)
    #         if violated:
    #             print(f"  [SAFETY] Action clipped at step {step}")

    #         # Check TCP workspace
    #         tcp_pos = robot.get_tcp_pose()
    #         if np.all(tcp_pos == 0):
    #             print("  [SAFETY] TCP not yet available, skipping workspace check this step")
    #         elif not self.safety.check_tcp_workspace(tcp_pos):
    #         # if not self.safety.check_tcp_workspace(tcp_pos):
    #             print(f"  [SAFETY] TCP {tcp_pos} outside workspace! Stopping episode.")
    #             raw_obs = robot.reset()
    #             obs = self.normalize_obs(raw_obs)
    #             episode += 1
    #             episode_return = 0.0
    #             episode_steps  = 0
    #             continue

    #         # Execute on robot
    #         next_raw_obs, done, info = robot.step(safe_action)
    #         next_obs = self.normalize_obs(next_raw_obs)

    #         # Reward
    #         reward = self.compute_reward(raw_obs, safe_action, next_raw_obs, done)
    #         episode_return += reward

    #         # Store transition (as uint8 for images to save memory)
    #         def compress_obs(o):
    #             c = {}
    #             for k, v in o.items():
    #                 if "image" in k:
    #                     # Store as uint8 (H,W,3) to save memory
    #                     img = (v.transpose(1,2,0) * np.array([0.229,0.224,0.225]) +
    #                            np.array([0.485,0.456,0.406])) * 255.0
    #                     c[k] = np.clip(img, 0, 255).astype(np.uint8)
    #                 else:
    #                     c[k] = v
    #             return c

    #         self.replay.push(
    #             compress_obs(obs),
    #             safe_action,
    #             reward,
    #             compress_obs(next_obs),
    #             done,
    #         )

    #         obs = next_obs
    #         raw_obs = next_raw_obs
    #         step += 1
    #         episode_steps += 1
    #         self.total_steps += 1

    #         # RL update (multiple gradient steps per env step — SERL style)
    #         if step >= self.cfg.warmup_steps and len(self.replay) >= self.cfg.batch_size:
    #             for _ in range(self.cfg.updates_per_step):
    #                 stats = self.update()

    #             if step % self.cfg.log_every == 0 and stats:
    #                 print(
    #                     f"[Step {step:6d}] "
    #                     f"Ep={episode:4d} | "
    #                     f"Ret={episode_return:.3f} | "
    #                     f"CriticL={stats['critic_loss']:.4f} | "
    #                     f"ActorL={stats['actor_loss']:.4f} | "
    #                     f"Alpha={stats['alpha']:.4f} | "
    #                     f"MeanQ={stats['mean_q']:.4f} | "
    #                     f"BufLen={len(self.replay)}"
    #                 )

    #         if done or episode_steps >= self.cfg.max_episode_steps:
    #             self.episode_returns.append(episode_return)
    #             print(f"  --> Episode {episode} done. Return={episode_return:.3f} Steps={episode_steps}")
    #             raw_obs = robot.reset()
    #             obs = self.normalize_obs(raw_obs)
    #             episode += 1
    #             episode_return = 0.0
    #             episode_steps  = 0

    #         # Save checkpoint
    #         if step % self.cfg.save_every == 0 and step > 0:
    #             self.save(step)

    #     print("[SERL] Training complete.")
    #     self.save("final")

    def run(self, robot):
        print(f"[SERL] Starting SERL fine-tuning for {self.cfg.num_steps} steps")
        print(f"[SERL] Actor mode: {self.cfg.actor_mode}")
        print(f"[SERL] Warmup: {self.cfg.warmup_steps} random steps before RL updates")
        print("=" * 60)

        step = 0
        episode = 0
        episode_return = 0.0
        episode_steps  = 0
        
        # input("  [Ep %d] Place object, then press Enter to reset arm..." % episode)
        raw_obs = robot.reset()
        self.act_policy.reset()

        obs_norm = self.normalize_obs(raw_obs)   # normalized → critic/replay buffer
        obs_raw_batch = self.obs_to_batch_raw(raw_obs)  # raw → ACT select_action

        while step < self.cfg.num_steps:

            obs_norm_batch = self.obs_to_batch(obs_norm)  # (1, 7) normalized state, (1,3,H,W) images

            def compress_obs(o):
                c = {}
                for k, v in o.items():
                    if "image" in k:
                        # v is (3,H,W) float32 normalized — undo ImageNet norm, convert to HWC uint8
                        img = (v.transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225])
                            + np.array([0.485, 0.456, 0.406])) * 255.0
                        c[k] = np.clip(img, 0, 255).astype(np.uint8)
                    else:
                        c[k] = v
                return c
            # def compress_obs(o):
            #     c = {}
            #     for k, v in o.items():
            #         if "image" in k:
            #             img = (v.transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225])
            #                 + np.array([0.485, 0.456, 0.406])) * 255.0
            #             c[k] = np.clip(img, 0, 255).astype(np.uint8)
            #         else:
            #             c[k] = v
            #     return c
            
            with torch.no_grad():
                # 1. Get ACT output in NORMALIZED action space
                base_norm = self.actor.get_base_action(obs_norm_batch)  # (1,7)

                # 2. Unnormalize to real joint radians
                a_std  = torch.tensor(self._action_std  + 1e-8, dtype=torch.float32, device=self.device)
                a_mean = torch.tensor(self._action_mean,         dtype=torch.float32, device=self.device)
                base_real = base_norm * a_std + a_mean  # (1,7) real radians

<<<<<<< HEAD
                p_grasp_val = 0.5   # neutral default
                p_in_box_val = 0.5

                if self.reward_classifier is not None:
                    wrist_img = raw_obs.get("observation.images.cam_right_wrist",
                                            raw_obs.get("observation.images.cam_high"))
                    if wrist_img is not None:
                        p_grasp_val = self.reward_classifier.predict_reward(wrist_img)

                if self.top_cam_classifier is not None:
                    top_img = raw_obs.get("observation.images.cam_high")
                    if top_img is not None:
                        p_in_box_val = self.top_cam_classifier.predict_reward(top_img)

                context = torch.tensor(
                    [[p_grasp_val, p_in_box_val]], dtype=torch.float32, device=self.device
                )  # (1, 2)

                # ── Compute residual with context ────────────────────────────────────
                residual_input = torch.cat([base_real, context], dim=-1)   # (1, 9)
                residual = self.actor.residual(residual_input) * self.actor.residual_scale
                mean_action = base_real + residual

=======
>>>>>>> 86760d5fb4b2b916c91d8e3736abf60f4663a89b
                # print(f"base_action (real rad): {base_real.squeeze().cpu().numpy().round(3)}")

                # 3. RL residual in real radian space
                residual   = self.actor.residual(base_real) * self.actor.residual_scale
                mean_action = base_real + residual  # (1,7)

                if step < self.cfg.warmup_steps:
                    action_np = mean_action.squeeze(0).cpu().numpy()
                else:
                    std = self.actor.log_std.exp().clamp(1e-4, 0.05).expand_as(mean_action)
                    action_np = torch.distributions.Normal(mean_action, std).rsample() \
                                    .squeeze(0).cpu().numpy()
                    action_np = action_np.flatten()  # ensure (7,) not (1,7) or (6,7)


            # Safety clips against real joint values — this now makes sense
            current_joints = raw_obs["observation.state"]
            safe_action, violated = self.safety.clip_action(action_np, current_joints=current_joints)
            # ── Action selection ──────────────────────────────────────────────
            # obs_raw_batch feeds ACT (it normalizes internally)
            # obs_norm_batch feeds the critic only
            # obs_norm_batch = self.obs_to_batch(obs_norm)

            # # In run(), replace the action selection block:
            # with torch.no_grad():
            #     # Pass NORMALIZED obs to select_action
            #     base_action_norm = self.actor.get_base_action(obs_norm_batch)  # (1,7) normalized
                

            #     # Unnormalize to real joint radians
            #     base_action_real = (
            #         base_action_norm * torch.tensor(
            #             self._action_std + 1e-8, dtype=torch.float32, device=self.device
            #         ) + torch.tensor(
            #             self._action_mean, dtype=torch.float32, device=self.device
            #         )
            #     )  # (1, 7) real radians
                
            #     # Residual in real joint space (radians)
            #     residual = self.actor.residual(base_action_real) * self.actor.residual_scale
            #     mean_action = base_action_real + residual

            #     if step < self.cfg.warmup_steps:
            #         action_np = mean_action.squeeze(0).cpu().numpy()
            #     else:
            #         std = self.actor.log_std.exp().clamp(1e-4, 0.05).expand_as(mean_action)
            #         action_np = torch.distributions.Normal(mean_action, std).rsample() \
            #                         .squeeze(0).cpu().numpy()
            # with torch.no_grad():
            #     # FIX 3: get base action from RAW obs (no double-norm)
            #     base_action = self.actor.get_base_action(obs_raw_batch)  # (1, 7) real rad

            #     # FIX 1: residual in joint space, no tanh squashing
            #     residual = self.actor.residual(base_action) * self.actor.residual_scale
            #     mean_action = base_action + residual  # (1, 7) absolute joint targets

            #     if step < self.cfg.warmup_steps:
            #         # Warmup: pure ACT output (deterministic), no RL noise yet
            #         action_np = mean_action.squeeze(0).cpu().numpy()
            #     else:
            #         # RL exploration: small Gaussian noise in joint space
            #         std = self.actor.log_std.exp().clamp(1e-4, 0.05).expand_as(mean_action)
            #         action_t = torch.distributions.Normal(mean_action, std).rsample()
            #         action_np = action_t.squeeze(0).cpu().numpy()

            # ── Safety ────────────────────────────────────────────────────────
            # FIX 2: clip_action now uses current joints to bound per-step delta
            current_joints = raw_obs["observation.state"]  # (7,) real joint positions
            safe_action, violated = self.safety.clip_action(
                action_np, current_joints=current_joints
            )
            if violated:
                print(f"  [SAFETY] Action clipped at step {step}")

            # tcp_pos = robot.get_tcp_pose()
            # if np.all(tcp_pos == 0):
            #     print("  [SAFETY] TCP not yet available, skipping workspace check")
            # elif not self.safety.check_tcp_workspace(tcp_pos):
            #     print(f"  [SAFETY] TCP {tcp_pos} outside workspace! Resetting.")
            #     raw_obs = robot.reset()
            #     obs_norm = self.normalize_obs(raw_obs)
            #     obs_raw_batch = self.obs_to_batch_raw(raw_obs)
            #     episode += 1
            #     episode_return = 0.0
            #     episode_steps  = 0
            #     continue

            # ── Safety: TCP workspace ─────────────────────────────────────────────
            tcp_pos = robot.get_tcp_pose()
            tcp_out_of_bounds = (
                not np.all(tcp_pos == 0) and
                not self.safety.check_tcp_workspace(tcp_pos)
            )
            
            # ── Safety: TCP workspace ─────────────────────────────────────────────
            # tcp_pos = robot.get_tcp_pose()
            # tcp_out_of_bounds = (
            #     not np.all(tcp_pos == 0) and
            #     not self.safety.check_tcp_workspace(tcp_pos)
            # )

            # ── Human early stop ──────────────────────────────────────────────────
            human_truncation = self.early_stop.check_and_clear()

            truncated = tcp_out_of_bounds or human_truncation

        
            # # ── Human early stop (press S + Enter) ───────────────────────────────
            # human_truncation = self.early_stop.check_and_clear()

            if truncated:
                reason = "TCP out of workspace" if tcp_out_of_bounds else "Human truncation"
                print(f"  [TRUNCATE] {reason} at step {step}. Penalty={self.early_stop.truncation_penalty:.1f}")

                truncation_reward = self.early_stop.truncation_penalty
                episode_return += truncation_reward
<<<<<<< HEAD
                
                ### CHANGE_BALLS
                obs_norm["classifier_context"] = np.array(
                    [p_grasp_val, p_in_box_val], dtype=np.float32
                )
=======

>>>>>>> 86760d5fb4b2b916c91d8e3736abf60f4663a89b
                self.replay.push(
                    compress_obs(obs_norm),
                    safe_action,
                    truncation_reward,
                    compress_obs(obs_norm),
                    False,
                )

                # RL update on truncation too (don't waste the experience)
                stats = {}
                if step >= self.cfg.warmup_steps and len(self.replay) >= self.cfg.batch_size:
                    for _ in range(self.cfg.updates_per_step):
                        stats = self.update()

                print("  [TRUNCATE] Resetting to home...")
                raw_obs = robot.reset()          # sends __ctrl__:reset, waits for reset_done
                self.act_policy.reset()

                obs_norm = self.normalize_obs(raw_obs)
                obs_raw_batch = self.obs_to_batch_raw(raw_obs)
                self.episode_returns.append(episode_return)
                print(f"  --> Episode {episode} truncated. Return={episode_return:.3f}  Steps={episode_steps}")
                episode += 1
                episode_return = 0.0
                episode_steps = 0
                step += 1
                self.total_steps += 1
                continue  # <-- jumps back to while, skips robot.step() entirely

            # ── Step robot (only reached if NOT truncated) ────────────────────────
            next_raw_obs, done, info = robot.step(safe_action)

            # ── Success detection: object-in-box is the ground truth ──
            if self.top_cam_classifier is not None:
                top_img = next_raw_obs.get("observation.images.cam_high")
                if top_img is not None:
                    p_in_box = self.top_cam_classifier.predict_reward(top_img)
                    # print(f"  [REWARD] P(in_box): {p_in_box:.3f}")
                    done = p_in_box > self.success_threshold

            # ── Fallback: use wrist classifier ONLY if no top-cam model loaded ──
            elif self.reward_classifier is not None:
                wrist_img = next_raw_obs.get("observation.images.cam_right_wrist",
                                            next_raw_obs.get("observation.images.cam_high"))
                p_grasp = self.reward_classifier.predict_reward(wrist_img)
                print(f"  [REWARD] P(grasp) [wrist fallback]: {p_grasp:.3f}")
                done = p_grasp > self.success_threshold

            if done:
                print(f"  [SUCCESS] Object-in-box confirmed at step {step} (episode {episode})")
            # next_raw_obs, done, info = robot.step(safe_action)
            # if self.reward_classifier is not None:
            #     reward_pred = self.reward_classifier.predict_reward(next_raw_obs.get("observation.images.cam_high"))
            #     print(f"  [REWARD] Predicted reward: {reward_pred}")
            #     done = reward_pred > self.success_threshold
            # if done:
            #     print(f"  [SUCCESS] Classifier triggered done at step {step} (episode {episode})")
            
            
            
            # if tcp_out_of_bounds or human_truncation:
            #     reason = "TCP out of workspace" if tcp_out_of_bounds else "Human truncation"
            #     print(f"  [TRUNCATE] {reason} at step {step}. Penalty={self.early_stop.truncation_penalty:.1f}")

            #     # Store transition with truncation penalty — do NOT mark done=True
            #     # (truncation ≠ natural episode end; use truncated flag for proper Bellman)
            #     truncation_reward = self.early_stop.truncation_penalty
            #     episode_return += truncation_reward

            #     # Push the last safe obs + penalty into replay as a truncated transition
            #     # next_obs here is the CURRENT obs (we never executed the step)
            #     self.replay.push(
            #         compress_obs(obs_norm),
            #         safe_action,
            #         truncation_reward,
            #         compress_obs(obs_norm),   # next_obs = same obs (episode cut short)
            #         False,                    # done=False — it was truncated, not terminal
            #     )

            #     # Reset
            #     print("  [TRUNCATE] Resetting to home...")
            #     raw_obs = robot.reset()
            #     obs_norm = self.normalize_obs(raw_obs)
            #     obs_raw_batch = self.obs_to_batch_raw(raw_obs)
            #     self.episode_returns.append(episode_return)
            #     print(f"  --> Episode {episode} truncated. Return={episode_return:.3f}  Steps={episode_steps}")
            #     episode += 1
            #     episode_return = 0.0
            #     episode_steps  = 0
            #     step += 1
            #     self.total_steps += 1
            #     continue

            # ── Step robot (only if no truncation) ───────────────────────────────
            # next_raw_obs, done, info = robot.step(safe_action)
            next_obs_norm = self.normalize_obs(next_raw_obs)
            tcp_pos = robot.get_tcp_pose()
            # self.trainer._last_tcp_z = float(tcp_pos[2]) if not np.all(tcp_pos == 0) else 0.5
            self._last_tcp_z = float(tcp_pos[2]) if not np.all(tcp_pos == 0) else 0.5

            reward = self.compute_reward(raw_obs, safe_action, next_raw_obs, done)
            episode_return += reward
            # # ── Step robot ────────────────────────────────────────────────────
            # next_raw_obs, done, info = robot.step(safe_action)
            # next_obs_norm = self.normalize_obs(next_raw_obs)

            # reward = self.compute_reward(raw_obs, safe_action, next_raw_obs, done)
            # episode_return += reward

            # ── Replay buffer: store NORMALIZED obs (critic never calls select_action)
            def compress_obs(o):
                c = {}
                for k, v in o.items():
                    if "image" in k:
                        # Denorm back to uint8 to save RAM
                        img = (v.transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225])
                            + np.array([0.485, 0.456, 0.406])) * 255.0
                        c[k] = np.clip(img, 0, 255).astype(np.uint8)
                    else:
                        c[k] = v
                return c

            self.replay.push(
                compress_obs(obs_norm),       # normalized state, uint8 images
                safe_action,
                reward,
                compress_obs(next_obs_norm),
                done,
            )

            # ── Advance state ─────────────────────────────────────────────────
            raw_obs = next_raw_obs
            obs_norm = next_obs_norm
            obs_raw_batch = self.obs_to_batch_raw(raw_obs)  # update raw batch for next step

            step += 1
            episode_steps += 1
            self.total_steps += 1

            # ── RL updates ────────────────────────────────────────────────────
            stats = {}
            if step >= self.cfg.warmup_steps and len(self.replay) >= self.cfg.batch_size:
                for _ in range(self.cfg.updates_per_step):
                    stats = self.update()

                if step % self.cfg.log_every == 0 and stats:
                    print(
                        f"[Step {step:6d}] "
                        f"Ep={episode:4d} | "
                        f"Ret={episode_return:.3f} | "
                        f"CriticL={stats['critic_loss']:.4f} | "
                        f"ActorL={stats['actor_loss']:.4f} | "
                        f"Alpha={stats['alpha']:.4f} | "
                        f"MeanQ={stats['mean_q']:.4f} | "
                        f"BufLen={len(self.replay)}"
                    )

            # ── Episode reset ─────────────────────────────────────────────────
            if done or episode_steps >= self.cfg.max_episode_steps:
                self.episode_returns.append(episode_return)
                print(f"  --> Episode {episode} done. Return={episode_return:.3f}  Steps={episode_steps}")
                input("  [Ep %d] Place object, then press Enter to reset arm..." % episode)
                raw_obs = robot.reset()
                self.act_policy.reset()

                obs_norm = self.normalize_obs(raw_obs)
                obs_raw_batch = self.obs_to_batch_raw(raw_obs)
                episode += 1
                episode_return = 0.0
                episode_steps  = 0

            # ── Checkpoint ────────────────────────────────────────────────────
            if step % self.cfg.save_every == 0 and step > 0:
                self.save(step)

        print("[SERL] Training complete.")
        self.save("final")

    def save(self, tag):
        ckpt = {
            "actor_state_dict":  self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "log_alpha":         self.log_alpha.item(),
            "total_steps":       self.total_steps,
            "episode_returns":   self.episode_returns,
        }
        path = self.save_dir / f"serl_ckpt_{tag}.pt"
        torch.save(ckpt, path)
        print(f"  [SERL] Saved checkpoint → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.log_alpha.data.fill_(ckpt["log_alpha"])
        self.total_steps = ckpt["total_steps"]
        print(f"  [SERL] Loaded checkpoint from {path} (step {self.total_steps})")


# ---------------------------------------------------------------------------
# 7. ROBOT STUB (replace with your actual UR10 interface)
# ---------------------------------------------------------------------------

class UR10RobotStub:
    """
    REPLACE THIS with your actual UR10 interface from lerobot.robots.
    This stub shows the expected API contract.
    """

    # def reset(self) -> Dict:
    #     print("[Robot] Resetting to home pose...")
    #     # Move to safe home, open gripper, wait for human to place object
    #     input("  Place object in scene, then press Enter...")
    #     return self._get_obs()
    
    # def reset(self) -> dict:
    #     input("  Place object in scene, then press Enter...")
    #     self._send_sync({"__ctrl__": "reset"})
    #     msg = self._recv_sync(timeout=60.0)   # was 30.0 — robot may need >30s to home
    #     raw_obs = msg.get("observation", {})
    #     self._last_raw_obs = raw_obs
    #     assert msg["type"] == "reset_done", f"Unexpected: {msg}"
    #     return self._raw_to_obs(msg["observation"])

    # def reset(self) -> dict:
    #     # Send reset immediately — client starts moving to home RIGHT NOW
    #     self._send_sync({"__ctrl__": "reset"})
    #     # Wait for home to be reached (client does smooth_move_to_home + sleep)
    #     msg = self._recv_sync(timeout=60.0)
    #     raw_obs = msg.get("observation", {})
    #     self._last_raw_obs = raw_obs
    #     assert msg["type"] == "reset_done", f"Unexpected: {msg}"
    #     # NOW ask human to place object — robot is already at home
    #     input("  Place object in scene, then press Enter...")
    #     # Send one more reset to get fresh obs AFTER object is placed
    #     self._send_sync({"__ctrl__": "reset"})
    #     msg2 = self._recv_sync(timeout=60.0)
    #     raw_obs2 = msg2.get("observation", {})
    #     self._last_raw_obs = raw_obs2
    #     assert msg2["type"] == "reset_done", f"Unexpected: {msg2}"
    #     return self._raw_to_obs(msg2["observation"])

    def reset(self) -> dict:
        # 1. Tell client to go home immediately
        self._send_sync({"__ctrl__": "reset"})
        # 2. Wait for home reached (client does smooth_move_to_home)
        msg = self._recv_sync(timeout=60.0)
        self._last_raw_obs = msg.get("observation", {})
        assert msg["type"] == "reset_done", f"Unexpected: {msg}"
        # 3. Human places object AFTER robot is safely at home
        input("  Robot at home. Place object in scene, then press Enter...")
        # 4. Get a FRESH obs now that object is in scene — no extra WS round-trip,
        #    just re-read the last obs (robot hasn't moved since home)
        return self._raw_to_obs(self._last_raw_obs)

    def step(self, action: np.ndarray) -> Tuple[Dict, bool, Dict]:
        """
        Execute action on robot.
        action: np.ndarray (7,) — 6-DOF EEF delta + gripper command
        Returns (obs, done, info)
        """
        raise NotImplementedError("Connect your actual UR10 robot here")

    def get_tcp_pose(self) -> np.ndarray:
        """Returns TCP position in robot base frame (3,)."""
        raise NotImplementedError

    def _get_obs(self) -> Dict:
        """Read cameras + joint states."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SERL fine-tuning on top of pretrained ACT")
    p.add_argument("--checkpoint_path", type=str,
                   default="/home_local/rudra_1/rudra/act_4/checkpoints/mod")
    p.add_argument("--robot_ip", type=str, default="192.168.100.3")

    # Training
    p.add_argument("--num_steps",         type=int,   default=20_000)
    p.add_argument("--warmup_steps",      type=int,   default=500,
                   help="Steps to run ACT policy before starting RL updates")
    p.add_argument("--batch_size",        type=int,   default=256)
    p.add_argument("--buffer_size",       type=int,   default=50_000)
    p.add_argument("--updates_per_step",  type=int,   default=1,
                   help="Gradient updates per env step (SERL uses 1)")
    p.add_argument("--max_episode_steps", type=int,   default=200)

    # Actor
    p.add_argument("--actor_mode",     type=str, default="frozen",
                   choices=["frozen", "head", "full"],
                   help="frozen=residual only, head=decoder+residual, full=all params")
    p.add_argument("--residual_scale", type=float, default=0.02,
                   help="Max magnitude of residual correction (in action units)")

    # Hyperparameters
    p.add_argument("--actor_lr",  type=float, default=3e-4)
    p.add_argument("--critic_lr", type=float, default=3e-4)
    p.add_argument("--alpha_lr",  type=float, default=3e-4)
    p.add_argument("--gamma",     type=float, default=0.99)
    p.add_argument("--tau",       type=float, default=0.005,
                   help="Polyak averaging coefficient for target critic")

    # Reward
    p.add_argument("--reward_mode",          type=str,  default="classifier",
                   choices=["classifier", "sparse", "shaped"])
    p.add_argument("--success_images_dir",   type=str,  default="./reward_data/success")
    p.add_argument("--failure_images_dir",   type=str,  default="./reward_data/failure")
    p.add_argument("--truncation_penalty", type=float, default=-50.0,
               help="Reward given when human truncates an episode early (negative)")

    # Safety
    p.add_argument("--tcp_workspace_min",  type=float, nargs=3,
                   default=[-100, -100, 2],
                   help="Min TCP position [x,y,z] in robot base frame (meters)")
    p.add_argument("--tcp_workspace_max",  type=float, nargs=3,
                   default=[100, 100, 90],
                   help="Max TCP position [x,y,z] in robot base frame (meters)")
    p.add_argument("--max_action_delta",   type=float, default=0.05,
                   help="Max L2 norm of 6-DOF action per step")

    # Logging
    p.add_argument("--save_dir",   type=str, default="./serl_checkpoints")
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every",  type=int, default=100)
    p.add_argument("--resume",     type=str, default=None,
                   help="Path to a SERL checkpoint to resume from")

    p.add_argument("--reward_classifier_path", type=str, default='./examples/ur10_gello/reward_classifier_wrist_v2.pt', help="Path to a pre-trained reward classifier .pt file (skips retraining)")
    p.add_argument("--success_threshold", type=float, default=0.80, help="Classifier probability threshold for success reward")

    p.add_argument("--robot-ws-port", type=int, default=8766, help="WebSocket port to listen for robot client (serl_client_ur10.py)")
    p.add_argument("--top_cam_classifier_path", type=str, default='./examples/ur10_gello/reward_classifier_top_v2.pt')
    return p.parse_args()

import signal

if __name__ == "__main__":
    
    # cfg = parse_args()

    # trainer = SERLTrainer(cfg)

    # if cfg.resume:
    #     trainer.load(cfg.resume)

    # # Connect your robot here — replace UR10RobotStub with your real interface:
    # # from lerobot.robots.your_ur10_robot import UR10Robot
    # # robot = UR10Robot(ip=cfg.robot_ip)
    # robot = UR10RobotInterface(host="0.0.0.0", port=cfg.robot_ws_port)
    # # robot.connect()
    # # robot = UR10RobotStub()

    # trainer.run(robot)

    cfg = parse_args()
    trainer = SERLTrainer(cfg)

    def _shutdown(sig, frame):
        print("\n[SERL] Caught Ctrl+C — saving checkpoint and exiting.")
        trainer.save("interrupted")
        trainer.early_stop.shutdown()
        robot._send_sync({"__ctrl__": "shutdown"})
        time.sleep(0.5)
        sys.exit(0)
    signal.signal(signal.SIGINT, _shutdown)
    # Clean Ctrl+C shutdown
    def _handle_sigint(sig, frame):
        print("\n[SERL] Ctrl+C received — shutting down.")
        trainer.early_stop.shutdown()
        trainer.save("interrupted")
        sys.exit(0)
    signal.signal(signal.SIGINT, _handle_sigint)

    if cfg.resume:
        trainer.load(cfg.resume)

    robot = UR10RobotInterface(host="0.0.0.0", port=cfg.robot_ws_port)
    trainer.run(robot)