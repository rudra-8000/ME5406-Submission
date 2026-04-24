#!/usr/bin/env python3
"""
serl_finetune_act.py

SERL fine-tuning on top of a pretrained ACT policy on a real UR10 robot.

Architecture:
  - ACT policy output (base action, normalized) is the anchor
  - Residual MLP input: [base_action (7) | p_wrist (1) | p_inbox (1)] = 9-dim
  - Residual output added to unnormalized base action in joint space
  - SAC critic: double-Q with CNN image encoder + state MLP
  - Reward: wrist classifier (dense shaping) + top-cam classifier (success/done)

Usage (compute node):
python examples/ur10_gello/serl_finetune_act.py \
    --checkpoint_path /home_local/rudra_1/rudra/act_4/checkpoints/080000/ \
    --robot_ip 192.168.100.3 \
    --robot-ws-port 8766 \
    --reward_mode classifier \
    --num_steps 20000 \
    --warmup_steps 5000 \
    --max_episode_steps 600 \
    --batch_size 64 \
    --save_dir ./serl_checkpoints \
    --tcp_workspace_min -0.75 -1.00 0.0005 \
    --tcp_workspace_max  0.75  0.80 0.70 \
    --max_action_delta 2.5
"""

import argparse
import asyncio
import collections
import copy
import logging
import os
import signal
import sys
import termios
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import msgpack_numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import websockets
from torch.optim import Adam

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig


# ---------------------------------------------------------------------------
# WebSocket robot interface
# ---------------------------------------------------------------------------

class UR10RobotInterface:
    """
    Training-side WebSocket server. Pairs with serl_client_ur10.py on the robot PC.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8766):
        self.host = host
        self.port = port
        self._packer = msgpack_numpy.Packer()
        self._ws = None
        self._loop = None
        self._server = None
        self._obs_queue: asyncio.Queue = None
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
        print("[SERL] Waiting for robot PC to connect on port 8766...")
        while self._ws is None:
            time.sleep(0.2)
        print("[SERL] Robot client connected!")

    def _send_sync(self, data: dict):
        future = asyncio.run_coroutine_threadsafe(
            self._ws.send(self._packer.pack(data)), self._loop
        )
        future.result(timeout=10.0)

    def _recv_sync(self, timeout: float = 15.0) -> dict:
        future = asyncio.run_coroutine_threadsafe(
            self._obs_queue.get(), self._loop
        )
        return future.result(timeout=timeout)

    def reset(self) -> dict:
        self._send_sync({"__ctrl__": "reset"})
        msg = self._recv_sync(timeout=30.0)
        assert msg["type"] == "reset_done", f"Unexpected: {msg}"
        raw_obs = msg.get("observation", {})
        self._last_raw_obs = raw_obs
        return self._raw_to_obs(raw_obs)

    def step(self, action: np.ndarray) -> tuple:
        self._send_sync({"action": action})
        msg = self._recv_sync(timeout=10.0)
        assert msg["type"] == "step_result", f"Unexpected: {msg}"
        raw_obs = msg.get("observation", {})
        self._last_raw_obs = raw_obs
        return self._raw_to_obs(raw_obs), False, {}

    def get_tcp_pose(self) -> np.ndarray:
        tcp = self._last_raw_obs.get("tcp_pose", None)
        if tcp is None or (isinstance(tcp, np.ndarray) and np.all(tcp == 0)):
            return self._fk_xyz(
                self._last_raw_obs.get("observation.state", np.zeros(7))[:6]
            )
        return np.array(tcp, dtype=np.float32)

    def _fk_xyz(self, joints: np.ndarray) -> np.ndarray:
        # Safe fallback — returns a known-safe position
        return np.array([-0.685, -0.176, 0.5], dtype=np.float32)

    def _raw_to_obs(self, raw: dict) -> dict:
        if "observation.state" in raw:
            state = np.array(raw["observation.state"], dtype=np.float32)
        else:
            state = np.array(
                [raw.get(f"joint_{i}", 0.0) for i in range(6)] + [raw.get("gripper", 0.0)],
                dtype=np.float32
            )
        obs = {"observation.state": state}
        for cam in ("cam_high", "cam_right_wrist"):
            key_out = f"observation.images.{cam}"
            if cam in raw:
                obs[key_out] = raw[cam]
            elif key_out in raw:
                obs[key_out] = raw[key_out]
        return obs


# ---------------------------------------------------------------------------
# Safety wrapper
# ---------------------------------------------------------------------------

class SafetyWrapper:
    def __init__(
        self,
        tcp_min: np.ndarray = np.array([-0.8, -0.8, 0.02]),
        tcp_max: np.ndarray = np.array([0.8, 0.8, 0.80]),
        joint_min: np.ndarray = np.array([-2 * np.pi] * 6),
        joint_max: np.ndarray = np.array([2 * np.pi] * 6),
        max_delta: float = 0.05,
    ):
        self.tcp_min = tcp_min
        self.tcp_max = tcp_max
        self.joint_min = joint_min
        self.joint_max = joint_max
        self.max_delta = max_delta

    def clip_action(
        self, action: np.ndarray, current_joints: np.ndarray = None
    ) -> Tuple[np.ndarray, bool]:
        violated = False
        clipped = action.copy()
        if current_joints is not None:
            delta = clipped[:6] - current_joints[:6]
            norm = np.linalg.norm(delta)
            if norm > self.max_delta:
                clipped[:6] = current_joints[:6] + delta / norm * self.max_delta
                violated = True
        clipped[:6] = np.clip(clipped[:6], self.joint_min, self.joint_max)
        clipped[6] = np.clip(clipped[6], 0.0, 1.0)
        return clipped, violated

    def check_tcp_workspace(self, tcp_pos: np.ndarray) -> bool:
        return np.all(tcp_pos >= self.tcp_min) and np.all(tcp_pos <= self.tcp_max)


# ---------------------------------------------------------------------------
# Early stop listener (keyboard)
# ---------------------------------------------------------------------------

class EarlyStopListener:
    """Press 'S' + Enter during an episode to truncate it with a penalty."""

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


# ---------------------------------------------------------------------------
# Reward classifier
# ---------------------------------------------------------------------------

class RewardClassifier(nn.Module):
    """
    Binary success/failure classifier (EfficientNet-B0 or ResNet18).
    Compatible with train_reward_classifier_v2.py checkpoints.
    """

    def __init__(self, arch: str = "efficientnet", device: str = "cuda"):
        super().__init__()
        from torchvision import models
        if arch == "efficientnet":
            base = models.efficientnet_b0(weights=None)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, 2))
        else:
            base = models.resnet18(weights=None)
            base.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, 2))
        self.model = base
        self.device = device
        self.to(device)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.model(img)

    def predict_reward(self, img_np: np.ndarray) -> float:
        img = cv2.resize(img_np, (224, 224)).astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        t = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = F.softmax(self.forward(t), dim=1)[0, 1].item()
        return prob

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cuda") -> "RewardClassifier":
        ckpt = torch.load(path, map_location=device)
        arch = ckpt.get("arch", "efficientnet")
        model = cls(arch=arch, device=device)
        state = ckpt["model_state"]
        # Remap bare keys (features.*) to wrapper keys (model.features.*)
        if not any(k.startswith("model.") for k in state.keys()):
            state = {"model." + k: v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()
        print(f"  [RewardClassifier] Loaded {arch} from {path} "
              f"(F1={ckpt.get('f1','?')}, epoch={ckpt.get('epoch','?')})")
        return model


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Stores (obs, action, reward, next_obs, done) transitions.
    obs dict keys:
      observation.state          — float32 (7,)  normalized
      observation.images.cam_*   — uint8   (H,W,3)
      p_wrist                    — float32 scalar
      p_inbox                    — float32 scalar
    Images stored as uint8 to save RAM; converted to float on sample.
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

        def to_tensor(arr, key):
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            # (B,H,W,3) → (B,3,H,W) for image keys
            if "image" in key and arr.ndim == 4 and arr.shape[-1] in (1, 3):
                arr = arr.transpose(0, 3, 1, 2)
            return torch.tensor(arr, dtype=torch.float32, device=self.device)

        obs_keys = list(batch[0]["obs"].keys())
        return {
            "obs":     {k: to_tensor(np.stack([b["obs"][k]      for b in batch]), k) for k in obs_keys},
            "next_obs":{k: to_tensor(np.stack([b["next_obs"][k] for b in batch]), k) for k in obs_keys},
            "actions": torch.tensor(np.stack([b["action"]  for b in batch]), dtype=torch.float32, device=self.device),
            "rewards": torch.tensor([b["reward"] for b in batch],            dtype=torch.float32, device=self.device),
            "dones":   torch.tensor([b["done"]   for b in batch],            dtype=torch.float32, device=self.device),
        }

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Critic (SAC double-Q)
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    Q(obs, action) → scalar.
    obs includes: state (7), images (cam_high, cam_right_wrist), p_wrist (1), p_inbox (1).
    The classifier probabilities are part of the obs dict (scalars, stored as shape (1,)).
    """

    def __init__(
        self,
        state_dim: int = 7,
        action_dim: int = 7,
        image_keys: List[str] = (
            "observation.images.cam_high",
            "observation.images.cam_right_wrist",
        ),
        hidden: int = 512,
    ):
        super().__init__()
        self.image_keys = list(image_keys)

        self.cnns = nn.ModuleDict()
        for k in self.image_keys:
            self.cnns[k.replace(".", "_")] = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 256), nn.ReLU(),
            )

        # +2 for p_wrist and p_inbox fed directly into MLP
        img_feat_dim = 256 * len(self.image_keys)
        self.mlp = nn.Sequential(
            nn.Linear(img_feat_dim + state_dim + action_dim + 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: Dict, action: torch.Tensor) -> torch.Tensor:
        feats = []
        for k in self.image_keys:
            img = obs[k]
            if img.shape[-1] == 640:
                img = F.interpolate(img, size=(120, 160), mode="bilinear", align_corners=False)
            feats.append(self.cnns[k.replace(".", "_")](img))

        state   = obs["observation.state"]         # (B, 7)
        p_wrist = obs["p_wrist"].unsqueeze(-1) if obs["p_wrist"].dim() == 1 else obs["p_wrist"]  # (B, 1)
        p_inbox = obs["p_inbox"].unsqueeze(-1) if obs["p_inbox"].dim() == 1 else obs["p_inbox"]  # (B, 1)

        x = torch.cat(feats + [state, p_wrist, p_inbox, action], dim=-1)
        return self.mlp(x).squeeze(-1)


class DoubleCritic(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.q1 = QNetwork(**kwargs)
        self.q2 = QNetwork(**kwargs)

    def forward(self, obs, action):
        return self.q1(obs, action), self.q2(obs, action)

    def min_q(self, obs, action):
        return torch.min(*self.forward(obs, action))


# ---------------------------------------------------------------------------
# Actor: ACT + residual MLP conditioned on classifier probabilities
# ---------------------------------------------------------------------------

class ACTActorWithResidual(nn.Module):
    """
    Base action from ACT (frozen/partially frozen).
    Residual MLP input: [base_action (7) | p_wrist (1) | p_inbox (1)] → 9-dim.
    This gives the residual direct perceptual context:
      - p_wrist: how well the gripper is currently wrapping the object
      - p_inbox: whether the object is already in the target box
    Both signals are zero-cost to compute (classifiers run every step anyway).

    Modes:
      frozen — only residual + log_std train
      head   — ACT decoder + residual train
      full   — everything trains
    """

    def __init__(
        self,
        act_policy: ACTPolicy,
        action_dim: int = 7,
        residual_hidden: int = 256,
        residual_scale: float = 0.02,
        mode: str = "frozen",
    ):
        super().__init__()
        self.act_policy = act_policy
        self.residual_scale = residual_scale
        self.mode = mode
        self.action_dim = action_dim

        # Input dim: base_action (7) + p_wrist (1) + p_inbox (1)
        residual_input_dim = action_dim + 2

        self.residual = nn.Sequential(
            nn.Linear(residual_input_dim, residual_hidden), nn.ReLU(),
            nn.Linear(residual_hidden, residual_hidden),    nn.ReLU(),
            nn.Linear(residual_hidden, action_dim),         nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 4.0)
        self._set_grad_mode(mode)

    def _set_grad_mode(self, mode: str):
        for p in self.act_policy.parameters():
            p.requires_grad = False
        if mode == "head":
            for p in self.act_policy.model.decoder.parameters():
                p.requires_grad = True
            for p in self.act_policy.model.action_head.parameters():
                p.requires_grad = True
        elif mode == "full":
            for p in self.act_policy.parameters():
                p.requires_grad = True
        for p in self.residual.parameters():
            p.requires_grad = True
        self.log_std.requires_grad = True

    def get_base_action(self, obs_norm_batch: Dict) -> torch.Tensor:
        B = obs_norm_batch["observation.state"].shape[0]
        # Strip p_wrist/p_inbox — ACT doesn't know about them
        act_obs = {k: v for k, v in obs_norm_batch.items()
                   if k not in ("p_wrist", "p_inbox")}
        if B == 1:
            with torch.set_grad_enabled(self.mode != "frozen"):
                output = self.act_policy.select_action(act_obs)
                if isinstance(output, np.ndarray):
                    output = torch.tensor(output, dtype=torch.float32,
                                          device=self.log_std.device)
                if output.dim() == 1:
                    output = output.unsqueeze(0)
            return output  # (1, 7)
        else:
            with torch.set_grad_enabled(self.mode != "frozen"):
                action_chunk = self.act_policy.predict_action_chunk(act_obs)
            return action_chunk[:, 0, :]  # (B, 7)

    def forward(
        self, obs: Dict, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        obs must contain p_wrist and p_inbox (shape (B,) or (B,1)).
        Returns (action, log_prob).
        """
        base = self.get_base_action(obs)  # (B, 7) normalized ACT output

        # Extract classifier probs — ensure shape (B, 1)
        p_wrist = obs["p_wrist"]
        p_inbox = obs["p_inbox"]
        if p_wrist.dim() == 1:
            p_wrist = p_wrist.unsqueeze(-1)
        if p_inbox.dim() == 1:
            p_inbox = p_inbox.unsqueeze(-1)

        # Residual conditioned on base action + classifier context
        residual_input = torch.cat([base, p_wrist, p_inbox], dim=-1)  # (B, 9)
        residual = self.residual(residual_input) * self.residual_scale  # (B, 7)
        mean = base + residual

        if deterministic:
            return mean, torch.zeros(mean.shape[0], device=mean.device)

        std = self.log_std.exp().clamp(1e-4, 0.05).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob


# ---------------------------------------------------------------------------
# SAC trainer
# ---------------------------------------------------------------------------

class SERLTrainer:

    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SERL] Using device: {self.device}")

        # ── Load ACT checkpoint ──────────────────────────────────────────────
        print(f"[SERL] Loading ACT checkpoint from {cfg.checkpoint_path}")
        ckpt = Path(cfg.checkpoint_path)
        config_candidates = [
            ckpt / "config.json",
            ckpt / "pretrained_model" / "config.json",
            ckpt.parent / "config.json",
        ]
        config_path = next((c for c in config_candidates if c.exists()), None)
        if config_path is None:
            for root, dirs, files in os.walk(str(ckpt)):
                for f in files:
                    print(f"  {root}/{f}")
            raise FileNotFoundError(f"config.json not found under {ckpt}")

        self.act_policy = ACTPolicy.from_pretrained(str(config_path.parent))
        self.act_policy.to(self.device)
        self.act_policy.eval()

        # ── Normalization stats ──────────────────────────────────────────────
        from safetensors.torch import load_file as load_safetensors
        pre_path  = ckpt / "pretrained_model" / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        post_path = ckpt / "pretrained_model" / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        self.norm_stats   = load_safetensors(str(pre_path),  device="cpu")
        self.unnorm_stats = load_safetensors(str(post_path), device="cpu")
        self._state_mean  = self.norm_stats["observation.state.mean"].numpy()
        self._state_std   = self.norm_stats["observation.state.std"].numpy()
        self._action_mean = self.unnorm_stats["action.mean"].numpy()
        self._action_std  = self.unnorm_stats["action.std"].numpy()
        print(f"[SERL] Norm stats loaded. State mean: {self._state_mean}")

        # ── Reward classifiers ───────────────────────────────────────────────
        self.reward_classifier = None
        if cfg.reward_mode == "classifier":
            if not cfg.reward_classifier_path:
                raise ValueError("--reward_classifier_path required for classifier mode.")
            print(f"[SERL] Loading wrist reward classifier from {cfg.reward_classifier_path}")
            self.reward_classifier = RewardClassifier.from_checkpoint(
                cfg.reward_classifier_path, device=str(self.device)
            )

        self.top_cam_classifier = None
        if cfg.top_cam_classifier_path:
            print(f"[SERL] Loading top-cam classifier from {cfg.top_cam_classifier_path}")
            self.top_cam_classifier = RewardClassifier.from_checkpoint(
                cfg.top_cam_classifier_path, device=str(self.device)
            )

        self.success_threshold = getattr(cfg, "success_threshold", 0.80)

        # ── Actor ────────────────────────────────────────────────────────────
        self.actor = ACTActorWithResidual(
            act_policy=self.act_policy,
            action_dim=7,
            residual_scale=cfg.residual_scale,
            mode=cfg.actor_mode,
        ).to(self.device)

        # ── Critic ───────────────────────────────────────────────────────────
        self.critic = DoubleCritic(
            state_dim=7,
            action_dim=7,
            image_keys=[
                "observation.images.cam_high",
                "observation.images.cam_right_wrist",
            ],
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ── Optimizers ───────────────────────────────────────────────────────
        actor_params = list(self.actor.residual.parameters()) + [self.actor.log_std]
        if cfg.actor_mode != "frozen":
            actor_params += [p for p in self.actor.act_policy.parameters() if p.requires_grad]
        self.actor_opt  = Adam(actor_params,              lr=cfg.actor_lr)
        self.critic_opt = Adam(self.critic.parameters(),  lr=cfg.critic_lr)

        self.log_alpha    = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.target_entropy = -float(7)
        self.alpha_opt    = Adam([self.log_alpha], lr=cfg.alpha_lr)

        # ── Replay buffer ────────────────────────────────────────────────────
        self.replay = ReplayBuffer(capacity=cfg.buffer_size, device=str(self.device))

        # ── Safety ───────────────────────────────────────────────────────────
        self.safety = SafetyWrapper(
            tcp_min=np.array(cfg.tcp_workspace_min),
            tcp_max=np.array(cfg.tcp_workspace_max),
            max_delta=cfg.max_action_delta,
        )
        print(f"[SERL] Workspace box: {self.safety.tcp_min} → {self.safety.tcp_max}")
        print(f"[SERL] Max action delta: {self.safety.max_delta}")

        # ── Early stop ───────────────────────────────────────────────────────
        self.early_stop = EarlyStopListener(
            truncation_penalty=getattr(cfg, "truncation_penalty", -50.0)
        )

        # ── Misc ─────────────────────────────────────────────────────────────
        self.total_steps    = 0
        self.episode_returns = []
        self.save_dir       = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._last_tcp_z    = 0.5

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def normalize_obs(self, raw_obs: Dict) -> Dict:
        """Normalize state (MEAN_STD) and images (ImageNet). Does NOT add p_wrist/p_inbox."""
        norm = {}
        s = raw_obs["observation.state"].astype(np.float32)
        norm["observation.state"] = (s - self._state_mean) / (self._state_std + 1e-8)
        img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        img_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        for k in ("observation.images.cam_high", "observation.images.cam_right_wrist"):
            img = raw_obs[k]
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            norm[k] = ((img - img_mean) / img_std).transpose(2, 0, 1)  # (3,H,W)
        return norm

    def get_classifier_probs(self, raw_obs: Dict) -> Tuple[float, float]:
        """
        Compute p_wrist and p_inbox from current raw observation.
        Falls back to 0.0 if a classifier is not loaded.
        """
        p_wrist = 0.0
        if self.reward_classifier is not None:
            wrist_img = raw_obs.get(
                "observation.images.cam_right_wrist",
                raw_obs.get("observation.images.cam_high")
            )
            if wrist_img is not None:
                p_wrist = self.reward_classifier.predict_reward(wrist_img)

        p_inbox = 0.0
        if self.top_cam_classifier is not None:
            top_img = raw_obs.get("observation.images.cam_high")
            if top_img is not None:
                p_inbox = self.top_cam_classifier.predict_reward(top_img)

        return p_wrist, p_inbox

    def build_obs_with_probs(self, obs_norm: Dict, p_wrist: float, p_inbox: float) -> Dict:
        """Attach classifier probs to a normalized obs dict (for actor + critic)."""
        obs = dict(obs_norm)
        obs["p_wrist"] = np.float32(p_wrist)
        obs["p_inbox"] = np.float32(p_inbox)
        return obs

    def obs_to_batch(self, obs: Dict) -> Dict:
        """Add batch dim and move to device. Handles scalars (p_wrist, p_inbox) too."""
        batch = {}
        for k, v in obs.items():
            if isinstance(v, np.floating) or (isinstance(v, np.ndarray) and v.ndim == 0):
                # scalar → (1, 1) tensor
                batch[k] = torch.tensor([[float(v)]], dtype=torch.float32, device=self.device)
            else:
                batch[k] = torch.tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0)
        return batch

    def compress_obs(self, obs_norm: Dict, p_wrist: float, p_inbox: float) -> Dict:
        """
        Convert normalized obs → uint8 images (RAM saving) + scalar probs.
        This is what gets pushed into the replay buffer.
        """
        c = {}
        for k, v in obs_norm.items():
            if "image" in k:
                img = (v.transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225])
                       + np.array([0.485, 0.456, 0.406])) * 255.0
                c[k] = np.clip(img, 0, 255).astype(np.uint8)
            else:
                c[k] = v
        c["p_wrist"] = np.float32(p_wrist)
        c["p_inbox"] = np.float32(p_inbox)
        return c

    def unnormalize_action(self, action_norm: np.ndarray) -> np.ndarray:
        return action_norm * (self._action_std + 1e-8) + self._action_mean

    # -------------------------------------------------------------------------
    # Reward
    # -------------------------------------------------------------------------

    def compute_reward(
        self, raw_obs, action, next_raw_obs, done,
        p_wrist: float, p_inbox: float
    ) -> float:
        reward = 0.0

        # Per-step efficiency penalty
        reward -= 0.05

        # Table proximity penalty
        if self._last_tcp_z < 0.015:
            reward -= 3.0

        # Wrist classifier: dense shaping — only active when top-cam not loaded
        if self.reward_classifier is not None and self.top_cam_classifier is None:
            reward += 5.0 * (p_wrist - self.success_threshold)

        # Top-cam classifier: bipolar dense shaping toward object-in-box
        if self.top_cam_classifier is not None:
            reward += 8.0 * (p_inbox - 0.5)

        # Terminal success bonus
        if done:
            reward += 10000.0

        print(f"reward: {reward:.3f}  p_wrist={p_wrist:.3f}  p_inbox={p_inbox:.3f}")
        return reward

    # -------------------------------------------------------------------------
    # SAC update
    # -------------------------------------------------------------------------

    def update(self):
        if len(self.replay) < self.cfg.batch_size:
            return {}

        batch   = self.replay.sample(self.cfg.batch_size)
        obs      = batch["obs"]
        next_obs = batch["next_obs"]
        actions  = batch["actions"]
        rewards  = batch["rewards"]
        dones    = batch["dones"]
        alpha    = self.log_alpha.exp().detach()

        # Critic update
        with torch.no_grad():
            next_act, next_logp = self.actor(next_obs)
            q_next   = self.critic_target.min_q(next_obs, next_act)
            q_target = rewards + self.cfg.gamma * (1 - dones) * (q_next - alpha * next_logp)

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # Actor update
        new_act, new_logp = self.actor(obs)
        q_val = self.critic.min_q(obs, new_act)
        actor_loss = (alpha * new_logp - q_val).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in self.actor.parameters() if p.requires_grad], 1.0
        )
        self.actor_opt.step()

        # Temperature update
        alpha_loss = -(self.log_alpha * (new_logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Soft target update
        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha":       alpha.item(),
            "mean_q":      q_val.mean().item(),
        }

    # -------------------------------------------------------------------------
    # Main training loop
    # -------------------------------------------------------------------------

    def run(self, robot):
        print(f"[SERL] Starting SERL fine-tuning for {self.cfg.num_steps} steps")
        print(f"[SERL] Actor mode: {self.cfg.actor_mode}  |  Residual input: base_action + p_wrist + p_inbox")
        print(f"[SERL] Warmup: {self.cfg.warmup_steps} steps before RL updates")
        print("=" * 60)

        step = 0
        episode = 0
        episode_return = 0.0
        episode_steps  = 0

        raw_obs = robot.reset()
        self.act_policy.reset()
        obs_norm = self.normalize_obs(raw_obs)

        while step < self.cfg.num_steps:

            # ── Classifier probs (run every step; used by actor, critic, reward) ──
            p_wrist, p_inbox = self.get_classifier_probs(raw_obs)
            obs_full = self.build_obs_with_probs(obs_norm, p_wrist, p_inbox)
            obs_batch = self.obs_to_batch(obs_full)

            # ── Action selection ──────────────────────────────────────────────
            with torch.no_grad():
                # 1. ACT base action in normalized space
                base_norm = self.actor.get_base_action(obs_batch)  # (1, 7)

                # 2. Unnormalize to real joint radians
                a_std  = torch.tensor(self._action_std  + 1e-8, dtype=torch.float32, device=self.device)
                a_mean = torch.tensor(self._action_mean,         dtype=torch.float32, device=self.device)
                base_real = base_norm * a_std + a_mean  # (1, 7)

                # 3. Build residual input: [base_real | p_wrist | p_inbox]
                pw = torch.tensor([[p_wrist]], dtype=torch.float32, device=self.device)
                pi = torch.tensor([[p_inbox]], dtype=torch.float32, device=self.device)
                residual_in = torch.cat([base_real, pw, pi], dim=-1)  # (1, 9)

                # 4. Residual correction
                residual   = self.actor.residual(residual_in) * self.actor.residual_scale
                mean_action = base_real + residual  # (1, 7)

                if step < self.cfg.warmup_steps:
                    action_np = mean_action.squeeze(0).cpu().numpy()
                else:
                    std = self.actor.log_std.exp().clamp(1e-4, 0.05).expand_as(mean_action)
                    action_np = torch.distributions.Normal(mean_action, std).rsample() \
                                    .squeeze(0).cpu().numpy()

            # ── Safety ───────────────────────────────────────────────────────
            current_joints = raw_obs["observation.state"]
            safe_action, violated = self.safety.clip_action(action_np, current_joints=current_joints)
            if violated:
                print(f"  [SAFETY] Action clipped at step {step}")

            # ── TCP workspace + human early stop ─────────────────────────────
            tcp_pos = robot.get_tcp_pose()
            tcp_out = (not np.all(tcp_pos == 0) and
                       not self.safety.check_tcp_workspace(tcp_pos))
            human_trunc = self.early_stop.check_and_clear()
            truncated   = tcp_out or human_trunc

            if truncated:
                reason = "TCP out of workspace" if tcp_out else "Human truncation"
                print(f"  [TRUNCATE] {reason} at step {step}. "
                      f"Penalty={self.early_stop.truncation_penalty:.1f}")

                trunc_reward = self.early_stop.truncation_penalty
                episode_return += trunc_reward
                self.replay.push(
                    self.compress_obs(obs_norm, p_wrist, p_inbox),
                    safe_action,
                    trunc_reward,
                    self.compress_obs(obs_norm, p_wrist, p_inbox),
                    False,
                )

                if step >= self.cfg.warmup_steps and len(self.replay) >= self.cfg.batch_size:
                    for _ in range(self.cfg.updates_per_step):
                        self.update()

                print("  [TRUNCATE] Resetting to home...")
                raw_obs = robot.reset()
                self.act_policy.reset()
                obs_norm = self.normalize_obs(raw_obs)

                self.episode_returns.append(episode_return)
                print(f"  --> Episode {episode} truncated. Return={episode_return:.3f}  Steps={episode_steps}")
                episode += 1
                episode_return = 0.0
                episode_steps  = 0
                step += 1
                self.total_steps += 1
                continue

            # ── Step robot ────────────────────────────────────────────────────
            next_raw_obs, done, info = robot.step(safe_action)

            # ── Success detection ─────────────────────────────────────────────
            next_p_wrist, next_p_inbox = self.get_classifier_probs(next_raw_obs)

            if self.top_cam_classifier is not None:
                done = next_p_inbox > self.success_threshold
            elif self.reward_classifier is not None:
                done = next_p_wrist > self.success_threshold

            if done:
                print(f"  [SUCCESS] Done triggered at step {step} "
                      f"(p_inbox={next_p_inbox:.3f}, p_wrist={next_p_wrist:.3f})")

            # ── TCP z for table-proximity penalty ─────────────────────────────
            tcp_pos = robot.get_tcp_pose()
            self._last_tcp_z = float(tcp_pos[2]) if not np.all(tcp_pos == 0) else 0.5

            # ── Reward ────────────────────────────────────────────────────────
            reward = self.compute_reward(
                raw_obs, safe_action, next_raw_obs, done,
                p_wrist=next_p_wrist, p_inbox=next_p_inbox
            )
            episode_return += reward

            # ── Replay buffer ─────────────────────────────────────────────────
            next_obs_norm = self.normalize_obs(next_raw_obs)
            self.replay.push(
                self.compress_obs(obs_norm,      p_wrist,      p_inbox),
                safe_action,
                reward,
                self.compress_obs(next_obs_norm, next_p_wrist, next_p_inbox),
                done,
            )

            # ── Advance state ─────────────────────────────────────────────────
            raw_obs  = next_raw_obs
            obs_norm = next_obs_norm
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
                        f"[Step {step:6d}] Ep={episode:4d} | Ret={episode_return:.3f} | "
                        f"CriticL={stats['critic_loss']:.4f} | ActorL={stats['actor_loss']:.4f} | "
                        f"Alpha={stats['alpha']:.4f} | MeanQ={stats['mean_q']:.4f} | "
                        f"BufLen={len(self.replay)}"
                    )

            # ── Episode reset ─────────────────────────────────────────────────
            if done or episode_steps >= self.cfg.max_episode_steps:
                self.episode_returns.append(episode_return)
                print(f"  --> Episode {episode} done. Return={episode_return:.3f}  Steps={episode_steps}")
                input(f"  [Ep {episode}] Place object, then press Enter to reset arm...")
                raw_obs = robot.reset()
                self.act_policy.reset()
                obs_norm = self.normalize_obs(raw_obs)
                episode += 1
                episode_return = 0.0
                episode_steps  = 0

            # ── Checkpoint ────────────────────────────────────────────────────
            if step % self.cfg.save_every == 0 and step > 0:
                self.save(step)

        print("[SERL] Training complete.")
        self.save("final")

    # -------------------------------------------------------------------------
    # Save / load
    # -------------------------------------------------------------------------

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
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SERL fine-tuning on top of pretrained ACT")
    p.add_argument("--checkpoint_path",          type=str, default="/home_local/rudra_1/rudra/act_4/checkpoints/mod")
    p.add_argument("--robot_ip",                 type=str, default="192.168.100.3")
    p.add_argument("--robot-ws-port",            type=int, default=8766)

    # Training
    p.add_argument("--num_steps",         type=int,   default=20_000)
    p.add_argument("--warmup_steps",      type=int,   default=500)
    p.add_argument("--batch_size",        type=int,   default=256)
    p.add_argument("--buffer_size",       type=int,   default=50_000)
    p.add_argument("--updates_per_step",  type=int,   default=1)
    p.add_argument("--max_episode_steps", type=int,   default=200)

    # Actor
    p.add_argument("--actor_mode",     type=str,   default="frozen",
                   choices=["frozen", "head", "full"])
    p.add_argument("--residual_scale", type=float, default=0.02)

    # Hyperparameters
    p.add_argument("--actor_lr",  type=float, default=3e-4)
    p.add_argument("--critic_lr", type=float, default=3e-4)
    p.add_argument("--alpha_lr",  type=float, default=3e-4)
    p.add_argument("--gamma",     type=float, default=0.99)
    p.add_argument("--tau",       type=float, default=0.005)

    # Reward
    p.add_argument("--reward_mode",             type=str,   default="classifier",
                   choices=["classifier", "sparse", "shaped"])
    p.add_argument("--reward_classifier_path",  type=str,   default="./examples/ur10_gello/reward_classifier_wrist_v2.pt")
    p.add_argument("--top_cam_classifier_path", type=str,   default="./examples/ur10_gello/reward_classifier_top_v2.pt")
    p.add_argument("--success_threshold",       type=float, default=0.60)
    p.add_argument("--truncation_penalty",      type=float, default=-50.0)

    # Safety
    p.add_argument("--tcp_workspace_min", type=float, nargs=3, default=[-100, -100, 2])
    p.add_argument("--tcp_workspace_max", type=float, nargs=3, default=[100, 100, 90])
    p.add_argument("--max_action_delta",  type=float, default=0.05)

    # Logging
    p.add_argument("--save_dir",   type=str, default="./serl_checkpoints")
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every",  type=int, default=100)
    p.add_argument("--resume",     type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = parse_args()
    trainer = SERLTrainer(cfg)

    robot = UR10RobotInterface(host="0.0.0.0", port=cfg.robot_ws_port)

    def _shutdown(sig, frame):
        print("\n[SERL] Shutting down — saving checkpoint.")
        trainer.save("interrupted")
        trainer.early_stop.shutdown()
        try:
            robot._send_sync({"__ctrl__": "shutdown"})
            time.sleep(0.5)
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)

    if cfg.resume:
        trainer.load(cfg.resume)

    trainer.run(robot)