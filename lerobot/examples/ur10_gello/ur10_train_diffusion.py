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

"""Programmatic entry point equivalent to `lerobot-train` for UR10+GELLO workflows.

Edit the CONFIGURATION section below, or override with CLI flags.

FOR PI05
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes=2 examples/ur10_gello/ur10_train.py --dataset-repo-id /home_local/rudra_1/rudra/lerobot/datasets/grasp_place --dataset-root /home_local/rudra_1/rudra/lerobot/datasets/grasp_place/ --policy-type pi05 --output-dir /home_local/rudra_1/rudra/checkpoints_pi05 --job-name pi05_test --policy-device cuda --wandb-enable false --policy-repo-id rudra-8000/pi05_test

CUDA_VISIBLE_DEVICES=0 python examples/ur10_gello/ur10_train_diffusion.py --batch-size 32 --steps=16000 --save-freq 1000 --dataset-repo-id /home_local/rudra_1/rudra/lerobot/datasets/grasp_place --dataset-root /home_local/rudra_1/rudra/lerobot/datasets/grasp_place/ --policy-type pi05 --output-dir /home_local/rudra_1/rudra/checkpoints_pi05 --job-name pi05_test --wandb-enable false --policy-repo-id rudra-8000/pi05_test --num-workers 0 --resume --resume-config-path /home_local/rudra_1/rudra/checkpoints_pi05/checkpoints/last/pretrained_model/

Example (shell, same spirit as lerobot-train):
  lerobot-train \\
    --dataset.repo_id=... \\
    --policy.type=act \\
    --output_dir=... \\
    --job_name=... \\
    --policy.device=cuda \\
    --wandb.enable=false \\
    --policy.repo_id=...

This script:
  python examples/ur10_gello/ur10_train.py
  python examples/ur10_gello/ur10_train.py --policy-type diffusion --batch-size 16 --steps 50000

Serve a trained policy over WebSocket (remote inference; local robot sends observations, receives actions).
  Requires: pip install websockets msgpack-numpy
  Training step checkpoints look like: <output>/checkpoints/000200/pretrained_model/{config.json,model.safetensors,...}
  You may pass either the step folder (000200) or pretrained_model directly as --policy-path.
  python examples/ur10_gello/ur10_train.py --serve-policy --policy-path /path/to/checkpoints/000200 \\
    --serve-dataset-from-train-config
  Optional: --dataset-repo-id / --dataset-root override dataset metadata used for stats (must match training).
  Clients connect to ws://<server-ip>:8765/, send msgpack observation batches, receive msgpack action dicts
  (same idea as openpi scripts/serve_policy.py).

Multi-GPU (Hugging Face Accelerate; effective batch = batch_size * num_processes).
  Run from the LeRobot repo root. Do not put ``python`` after ``accelerate launch`` — the first
  positional argument must be the script path, or Accelerate will try to open a file named ``python``.

  CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes=2 \\
    examples/ur10_gello/ur10_train.py \\
    --dataset-repo-id /path/to/dataset \\
    --policy-type act \\
    --output-dir /path/to/checkpoints \\
    --job-name act_test \\
    --policy-device cuda \\
    --wandb-enable false \\
    --policy-repo-id your_hf/repo
"""

from __future__ import annotations

import argparse
import asyncio
import json
import http
import logging
import socket
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_SINGLE_FILE
from accelerate.utils import DistributedDataParallelKwargs

from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.scripts.lerobot_train import train
from lerobot.utils.constants import ACTION, POLICY_PREPROCESSOR_DEFAULT_NAME, PRETRAINED_MODEL_DIR
from lerobot.utils.import_utils import register_third_party_plugins

# Register bundled policy configs on PreTrainedConfig (same registry as `lerobot-train`).
import lerobot.policies.factory  # noqa: F401
import lerobot.policies.pi0_fast.configuration_pi0_fast  # noqa: F401  # not imported by factory.py


import sys


# -----------------------------------------------------------------------------
# CONFIGURATION (defaults mirror common `lerobot-train` flags)
# -----------------------------------------------------------------------------

# --dataset.*
DATASET_REPO_ID = "gnq/ur10_gello"
DATASET_ROOT: str | None = "/home_local/rudra_1/guningquan/dataset/testlerobot"
DATASET_EPISODES: list[int] | None = None
DATASET_REVISION: str | None = None
DATASET_USE_IMAGENET_STATS = True
DATASET_STREAMING = False

# --policy.type (registered names include: act, diffusion, vqbet, tdmpc, sac, sarm, smolvla,
# groot, pi0, pi05, pi0_fast, xvla, wall_x, reward_classifier, ...)
POLICY_TYPE = "act"
POLICY_DEVICE = "cuda"
POLICY_REPO_ID = "guningquan/act_policy"
POLICY_PUSH_TO_HUB = False
POLICY_USE_AMP = False
# If set, loads policy config/weights from this path (like `--policy.path=...`).
# When set, `POLICY_TYPE` is ignored (type comes from the checkpoint).
# POLICY_PRETRAINED_PATH: str | None = None
POLICY_PRETRAINED_PATH: str | None = None
RESUME_CONFIG_PATH: str | None = None   # <-- ADD THIS

# Top-level train args
OUTPUT_DIR = Path("/home_local/rudra_1/guningquan/checkpoints/testlerobot")
JOB_NAME = "act_test"
RESUME = True
SEED = 1000
CUDNN_DETERMINISTIC = False
NUM_WORKERS = 4
BATCH_SIZE = 64
STEPS = 100_000
EVAL_FREQ = 20_000
LOG_FREQ = 200
TOLERANCE_S = 1e-4
SAVE_CHECKPOINT = True
SAVE_FREQ = 20_000
USE_POLICY_TRAINING_PRESET = True

# --wandb.*
WANDB_ENABLE = False
WANDB_PROJECT = "lerobot"
WANDB_ENTITY: str | None = None
WANDB_NOTES: str | None = None
WANDB_MODE: str | None = None
WANDB_DISABLE_ARTIFACT = False
WANDB_ADD_TAGS = True

# --eval.*
EVAL_N_EPISODES = 50
EVAL_BATCH_SIZE = 50
EVAL_USE_ASYNC_ENVS = False

# RA-BC (--use_rabc, --rabc_*)
USE_RABC = False
RABC_PROGRESS_PATH: str | None = None
RABC_KAPPA = 0.01
RABC_EPSILON = 1e-6
RABC_HEAD_MODE: str | None = "sparse"

# Optional: observation rename map (usually empty)
RENAME_MAP: dict[str, str] = {}

# WebSocket policy server (--serve-policy); mirrors openpi scripts/serve_policy.py style.
SERVE_HOST = "0.0.0.0"
SERVE_PORT = 8765
# If None while serving, falls back to POLICY_PRETRAINED_PATH.
SERVE_POLICY_CHECKPOINT: str | None = None
def create_train_accelerator(cfg: TrainPipelineConfig) -> Accelerator:
    """Match `lerobot_train.train()` so DDP and device selection behave the same under `accelerate launch`."""
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    force_cpu = cfg.policy.device == "cpu"
    return Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
        cpu=force_cpu,
    )


def _build_policy() -> PreTrainedConfig:
    if POLICY_PRETRAINED_PATH:
        policy = PreTrainedConfig.from_pretrained(POLICY_PRETRAINED_PATH)
    else:
        try:
            config_cls = PreTrainedConfig.get_choice_class(POLICY_TYPE)
        except KeyError as e:
            raise ValueError(
                f"Unknown policy.type={POLICY_TYPE!r}. "
                "Use a name registered on PreTrainedConfig (e.g. act, diffusion, pi0), "
                "or install a plugin that registers your policy config."
            ) from e
        policy = config_cls()

    if POLICY_TYPE == "pi05":
        policy.dtype = "bfloat16"  # Pi0.5 benefits from mixed precision training
        policy.train_expert_only = True   # freeze VLM, train action expert only
        policy.gradient_checkpointing = True  # save memory for longer sequences / larger batch sizes
        # policy.freeze_vision_encoder = True  # alternative: freeze only vision
        # policy.use_peft = True   
    
    policy.device = POLICY_DEVICE
    policy.repo_id = POLICY_REPO_ID
    policy.push_to_hub = POLICY_PUSH_TO_HUB
    policy.use_amp = POLICY_USE_AMP
    if RESUME and POLICY_PRETRAINED_PATH is None and RESUME_CONFIG_PATH is not None:
        policy.pretrained_path = Path(RESUME_CONFIG_PATH)
    elif POLICY_PRETRAINED_PATH:
        policy.pretrained_path = Path(POLICY_PRETRAINED_PATH)
    return policy

    

def _serve_checkpoint_path() -> Path:
    raw = SERVE_POLICY_CHECKPOINT or POLICY_PRETRAINED_PATH
    if not raw:
        raise ValueError(
            "Policy serving requires a checkpoint path: set POLICY_PRETRAINED_PATH / SERVE_POLICY_CHECKPOINT "
            "or pass --policy-path / --serve-policy-checkpoint."
        )
    return Path(raw).expanduser().resolve()


def _resolve_policy_bundle_dir(user_checkpoint: Path) -> Path:
    """Resolve to the directory that contains policy config.json (Hub export or …/pretrained_model)."""
    if (user_checkpoint / CONFIG_NAME).is_file():
        return user_checkpoint
    nested = user_checkpoint / PRETRAINED_MODEL_DIR
    if (nested / CONFIG_NAME).is_file():
        return nested
    raise FileNotFoundError(
        f"No {CONFIG_NAME} found at {user_checkpoint} or {nested}. "
        "Use a training step directory (e.g. …/checkpoints/000200) or the inner pretrained_model folder."
    )


def _ensure_policy_weights(policy_dir: Path) -> None:
    weights = policy_dir / SAFETENSORS_SINGLE_FILE
    if not weights.is_file():
        raise FileNotFoundError(
            f"Missing {SAFETENSORS_SINGLE_FILE} under {policy_dir}. "
            "LeRobot training checkpoints store weights in pretrained_model/model.safetensors. "
            "If that file is absent, this save is incomplete; re-run training with saving enabled or use a "
            "checkpoint directory that contains the model file."
        )


def _preprocessors_pretrained_path(policy_dir: Path) -> str | None:
    proc = policy_dir / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
    return str(policy_dir) if proc.is_file() else None


def _read_train_config_dataset(policy_dir: Path) -> dict:
    train_cfg_path = policy_dir / "train_config.json"
    if not train_cfg_path.is_file():
        return {}
    with train_cfg_path.open(encoding="utf-8") as f:
        tc = json.load(f)
    return tc.get("dataset") or {}


def _resolve_serve_dataset(
    policy_dir: Path, args: argparse.Namespace
) -> tuple[str, str | None, str | None]:
    """Pick dataset repo_id/root/revision: CLI overrides train_config when both are used."""
    repo_id = DATASET_REPO_ID
    root = DATASET_ROOT
    revision = DATASET_REVISION
    if args.serve_dataset_from_train_config:
        tc_ds = _read_train_config_dataset(policy_dir)
        if not tc_ds:
            logging.warning(
                "--serve-dataset-from-train-config was set but %s is missing or has no dataset block.",
                policy_dir / "train_config.json",
            )
        else:
            if args.dataset_repo_id is None and tc_ds.get("repo_id") is not None:
                repo_id = tc_ds["repo_id"]
            if args.dataset_root is None and "root" in tc_ds:
                root = tc_ds["root"]
            if args.dataset_revision is None and tc_ds.get("revision") is not None:
                revision = tc_ds["revision"]
            logging.info(
                "Dataset for serving (after train_config merge): repo_id=%r root=%r revision=%r",
                repo_id,
                root,
                revision,
            )
    return repo_id, root, revision


def _policy_server_metadata(
    policy_cfg: PreTrainedConfig,
    ds_features: dict,
    user_checkpoint: Path,
    policy_dir: Path,
    dataset_repo_id: str,
) -> dict:
    action_names = list(ds_features[ACTION]["names"])
    return {
        "protocol": "lerobot_policy_v1",
        "policy_type": policy_cfg.type,
        "user_checkpoint": str(user_checkpoint),
        "policy_bundle_dir": str(policy_dir),
        "dataset_repo_id": dataset_repo_id,
        "action_names": action_names,
        "message_format": (
            "Send msgpack: either a flat observation dict (numpy arrays, same keys as training/robot), "
            "or {\"observation\": {...}, \"task\": optional str, \"robot_type\": optional str}. "
            "Send {\"__ctrl__\": \"reset\"} to call policy.reset()."
        ),
    }


def run_policy_server(args: argparse.Namespace) -> None:
    """Listen on WebSocket; receive observations (msgpack+numpy), return actions."""
    try:
        import msgpack_numpy
        from websockets.asyncio.server import ServerConnection, serve
        from websockets.exceptions import ConnectionClosed
        from websockets.http11 import Request
    except ImportError as e:
        raise ImportError(
            "Policy server needs optional dependencies. Install with: pip install websockets msgpack-numpy"
        ) from e

    user_checkpoint = _serve_checkpoint_path()
    policy_dir = _resolve_policy_bundle_dir(user_checkpoint)
    _ensure_policy_weights(policy_dir)

    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_dir))
    policy_cfg.device = POLICY_DEVICE
    policy_cfg.pretrained_path = policy_dir

    ds_repo_id, ds_root, ds_revision = _resolve_serve_dataset(policy_dir, args)
    dataset_meta = LeRobotDatasetMetadata(
        repo_id=ds_repo_id,
        root=ds_root,
        revision=ds_revision,
    )
    rename = dict(RENAME_MAP) if RENAME_MAP else None
    policy = make_policy(policy_cfg, ds_meta=dataset_meta, rename_map=rename)
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=_preprocessors_pretrained_path(policy_dir),
        dataset_stats=dataset_meta.stats,
    )
    policy.eval()
    device = torch.device(policy_cfg.device)
    metadata = _policy_server_metadata(
        policy_cfg, dataset_meta.features, user_checkpoint, policy_dir, ds_repo_id
    )
    packer = msgpack_numpy.Packer()

    def _health_check(connection: ServerConnection, request: Request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    async def _handler(websocket: ServerConnection):
        remote = websocket.remote_address
        logging.info("Connection from %s opened", remote)
        await websocket.send(packer.pack(metadata))
        prev_total_time: float | None = None
        while True:
            try:
                start_time = time.monotonic()
                raw = msgpack_numpy.unpackb(await websocket.recv())

                if isinstance(raw, dict) and raw.get("__ctrl__") == "reset":
                    policy.reset()
                    await websocket.send(packer.pack({"ok": True, "reset": True}))
                    continue

                if not isinstance(raw, dict):
                    raise TypeError(f"Expected observation dict, got {type(raw).__name__}")

                if "observation" in raw and isinstance(raw["observation"], dict):
                    obs_dict = raw["observation"]
                    task = raw.get("task")
                    robot_type = raw.get("robot_type")
                else:
                    obs_dict = raw
                    task = None
                    robot_type = None

                infer_start = time.monotonic()
                frame = build_inference_frame(
                    observation=obs_dict,
                    device=device,
                    ds_features=dataset_meta.features,
                    task=task,
                    robot_type=robot_type,
                )
                batch = preprocess(frame)
                with torch.inference_mode():
                    action_tensor = policy.select_action(batch)
                action_tensor = postprocess(action_tensor)
                robot_action = make_robot_action(action_tensor, dataset_meta.features)
                infer_time = time.monotonic() - infer_start

                action_names = metadata["action_names"]
                action_vec = np.asarray([robot_action[name] for name in action_names], dtype=np.float32)

                out = {
                    "action": robot_action,
                    "action_vector": action_vec,
                    "server_timing": {"infer_ms": infer_time * 1000.0},
                }
                if prev_total_time is not None:
                    out["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0

                await websocket.send(packer.pack(out))
                prev_total_time = time.monotonic() - start_time

            except asyncio.CancelledError:
                raise
            except ConnectionClosed:
                logging.info("Connection from %s closed", remote)
                break
            except Exception:
                await websocket.send(packer.pack({"error": traceback.format_exc()}))
                await websocket.close(code=1011, reason="Internal server error")
                raise

    async def _run():
        async with serve(
            _handler,
            SERVE_HOST,
            SERVE_PORT,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        local_ip = "(could not resolve; check `hostname -I` or `ip addr`)"

    logging.info(
        "LeRobot policy WebSocket server binding %s:%s (hostname=%s ip=%s)",
        SERVE_HOST,
        SERVE_PORT,
        hostname,
        local_ip,
    )
    asyncio.run(_run())


def build_train_config() -> TrainPipelineConfig:
    policy_cfg = _build_policy()

    return TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=DATASET_REPO_ID,
            root=DATASET_ROOT,
            episodes=DATASET_EPISODES,
            revision=DATASET_REVISION,
            use_imagenet_stats=DATASET_USE_IMAGENET_STATS,
            streaming=DATASET_STREAMING,
            video_backend="pyav",
        ),
        
        policy=policy_cfg,
        optimizer=None,
        scheduler=None,
        output_dir=OUTPUT_DIR,
        job_name=JOB_NAME,
        resume=RESUME,
        seed=SEED,
        cudnn_deterministic=CUDNN_DETERMINISTIC,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        steps=STEPS,
        eval_freq=EVAL_FREQ,
        log_freq=LOG_FREQ,
        tolerance_s=TOLERANCE_S,
        save_checkpoint=SAVE_CHECKPOINT,
        save_freq=SAVE_FREQ,
        use_policy_training_preset=USE_POLICY_TRAINING_PRESET,
        wandb=WandBConfig(
            enable=WANDB_ENABLE,
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            notes=WANDB_NOTES,
            mode=WANDB_MODE,
            disable_artifact=WANDB_DISABLE_ARTIFACT,
            add_tags=WANDB_ADD_TAGS,
        ),
        eval=EvalConfig(
            n_episodes=EVAL_N_EPISODES,
            batch_size=EVAL_BATCH_SIZE,
            use_async_envs=EVAL_USE_ASYNC_ENVS,
        ),
        use_rabc=USE_RABC,
        rabc_progress_path=RABC_PROGRESS_PATH,
        rabc_kappa=RABC_KAPPA,
        rabc_epsilon=RABC_EPSILON,
        rabc_head_mode=RABC_HEAD_MODE,
        rename_map=dict(RENAME_MAP),
    )


def _apply_cli_to_module_constants(args: argparse.Namespace) -> None:
    """Patch module-level CONFIGURATION from CLI (optional overrides)."""
    global DATASET_REPO_ID, DATASET_ROOT, DATASET_REVISION, OUTPUT_DIR, JOB_NAME, POLICY_DEVICE, POLICY_REPO_ID, POLICY_TYPE
    global WANDB_ENABLE, BATCH_SIZE, STEPS, RESUME, POLICY_PRETRAINED_PATH, NUM_WORKERS
    global EVAL_FREQ, LOG_FREQ, SAVE_FREQ, SEED, CUDNN_DETERMINISTIC, POLICY_PUSH_TO_HUB, POLICY_USE_AMP
    global SERVE_HOST, SERVE_PORT, SERVE_POLICY_CHECKPOINT, RESUME_CONFIG_PATH

    if args.dataset_repo_id is not None:
        DATASET_REPO_ID = args.dataset_repo_id
    if args.dataset_root is not None:
        DATASET_ROOT = args.dataset_root
    if args.dataset_revision is not None:
        DATASET_REVISION = args.dataset_revision
    if args.output_dir is not None:
        OUTPUT_DIR = Path(args.output_dir)
    if args.job_name is not None:
        JOB_NAME = args.job_name
    if args.policy_device is not None:
        POLICY_DEVICE = args.policy_device
    if args.policy_type is not None:
        POLICY_TYPE = args.policy_type
    if args.policy_repo_id is not None:
        POLICY_REPO_ID = args.policy_repo_id
    if args.policy_path is not None:
        POLICY_PRETRAINED_PATH = args.policy_path
    if args.wandb_enable is not None:
        WANDB_ENABLE = args.wandb_enable
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    if args.steps is not None:
        STEPS = args.steps
    if args.resume:
        RESUME = True
    if args.num_workers is not None:
        NUM_WORKERS = args.num_workers
    if args.eval_freq is not None:
        EVAL_FREQ = args.eval_freq
    if args.log_freq is not None:
        LOG_FREQ = args.log_freq
    if args.save_freq is not None:
        SAVE_FREQ = args.save_freq
    if args.seed is not None:
        SEED = args.seed
    if args.cudnn_deterministic:
        CUDNN_DETERMINISTIC = True
    if args.policy_push_to_hub is not None:
        POLICY_PUSH_TO_HUB = args.policy_push_to_hub
    if args.policy_use_amp is not None:
        POLICY_USE_AMP = args.policy_use_amp
    if args.serve_host is not None:
        SERVE_HOST = args.serve_host
    if args.serve_port is not None:
        SERVE_PORT = args.serve_port
    if args.serve_policy_checkpoint is not None:
        SERVE_POLICY_CHECKPOINT = args.serve_policy_checkpoint
    if args.resume_config_path is not None:
        RESUME_CONFIG_PATH = args.resume_config_path
        train_cfg_json = Path(args.resume_config_path) / "train_config.json"
        sys.argv.append(f"--config_path={train_cfg_json}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a policy (default ACT) using the same pipeline as lerobot_train.py.",
    )
    p.add_argument("--dataset-repo-id", type=str, default=None, help="Maps to dataset.repo_id")
    p.add_argument("--dataset-root", type=str, default=None, help="Maps to dataset.root")
    p.add_argument(
        "--dataset-revision",
        type=str,
        default=None,
        help="Maps to dataset.revision (metadata version / branch)",
    )
    p.add_argument("--output-dir", type=str, default=None, help="Maps to output_dir")
    p.add_argument("--job-name", type=str, default=None, help="Maps to job_name")
    p.add_argument("--policy-device", type=str, default=None, help="Maps to policy.device")
    p.add_argument(
        "--policy-type",
        type=str,
        default=None,
        help="Maps to policy.type (e.g. act, diffusion, pi0, smolvla)",
    )
    p.add_argument("--policy-repo-id", type=str, default=None, help="Maps to policy.repo_id")
    p.add_argument(
        "--policy-path",
        type=str,
        default=None,
        help="Load policy from local/HF path (like --policy.path=...)",
    )
    p.add_argument(
        "--policy-push-to-hub",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=None,
        help="Maps to policy.push_to_hub (true/false)",
    )
    p.add_argument(
        "--policy-use-amp",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=None,
        help="Maps to policy.use_amp (true/false)",
    )
    p.add_argument(
        "--wandb-enable",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=None,
        help="Maps to wandb.enable",
    )
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--eval-freq", type=int, default=None)
    p.add_argument("--log-freq", type=int, default=None)
    p.add_argument("--save-freq", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", action="store_true", help="Resume from output_dir checkpoint")
    p.add_argument(
        "--cudnn-deterministic",
        action="store_true",
        help="Enable deterministic cuDNN (slower, more reproducible)",
    )
    p.add_argument(
        "--serve-policy",
        action="store_true",
        help="Run WebSocket policy server instead of training (needs websockets, msgpack-numpy)",
    )
    p.add_argument(
        "--serve-host",
        type=str,
        default=None,
        help="WebSocket bind address (default from SERVE_HOST, usually 0.0.0.0)",
    )
    p.add_argument(
        "--serve-port",
        type=int,
        default=None,
        help="WebSocket port (default SERVE_PORT=8765)",
    )
    p.add_argument(
        "--serve-policy-checkpoint",
        type=str,
        default=None,
        help="Checkpoint directory for serving (overrides POLICY_PRETRAINED_PATH for --serve-policy only)",
    )
    p.add_argument(
        "--serve-dataset-from-train-config",
        action="store_true",
        help="When serving, set dataset repo_id/root/revision from <pretrained_model>/train_config.json",
    )

    p.add_argument(
        "--resume-config-path",
        type=str,
        default=None,
        help="Path to train_config.json for resuming (injected as config_path= for draccus)",
    )
    return p.parse_args()


def main() -> None:
    register_third_party_plugins()
    args = parse_args()
    _apply_cli_to_module_constants(args)
    if args.serve_policy:
        logging.basicConfig(level=logging.INFO, force=True)
        run_policy_server(args)
        return
    cfg = build_train_config()
    cfg.validate()

    if cfg.optimizer is None:
        cfg.optimizer = cfg.policy.get_optimizer_preset()
    if cfg.scheduler is None:
        cfg.scheduler = cfg.policy.get_scheduler_preset()

    accelerator = create_train_accelerator(cfg)
    train(cfg, accelerator=accelerator)


if __name__ == "__main__":
    main()
