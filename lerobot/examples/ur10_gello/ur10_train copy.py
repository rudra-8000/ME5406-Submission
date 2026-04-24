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
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.lerobot_train import train
from lerobot.utils.import_utils import register_third_party_plugins

# Register bundled policy configs on PreTrainedConfig (same registry as `lerobot-train`).
import lerobot.policies.factory  # noqa: F401
import lerobot.policies.pi0_fast.configuration_pi0_fast  # noqa: F401  # not imported by factory.py

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
POLICY_PRETRAINED_PATH: str | None = None

# Top-level train args
OUTPUT_DIR = Path("/home_local/rudra_1/guningquan/checkpoints/testlerobot")
JOB_NAME = "act_test"
RESUME = False
SEED = 1000
CUDNN_DETERMINISTIC = False
NUM_WORKERS = 4
BATCH_SIZE = 8
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
    policy.device = POLICY_DEVICE
    policy.repo_id = POLICY_REPO_ID
    policy.push_to_hub = POLICY_PUSH_TO_HUB
    policy.use_amp = POLICY_USE_AMP
    if POLICY_PRETRAINED_PATH:
        policy.pretrained_path = Path(POLICY_PRETRAINED_PATH)
    return policy


def build_train_config() -> TrainPipelineConfig:
    return TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=DATASET_REPO_ID,
            root=DATASET_ROOT,
            episodes=DATASET_EPISODES,
            revision=DATASET_REVISION,
            use_imagenet_stats=DATASET_USE_IMAGENET_STATS,
            streaming=DATASET_STREAMING,
        ),
        policy=_build_policy(),
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
    global DATASET_REPO_ID, DATASET_ROOT, OUTPUT_DIR, JOB_NAME, POLICY_DEVICE, POLICY_REPO_ID, POLICY_TYPE
    global WANDB_ENABLE, BATCH_SIZE, STEPS, RESUME, POLICY_PRETRAINED_PATH, NUM_WORKERS
    global EVAL_FREQ, LOG_FREQ, SAVE_FREQ, SEED, CUDNN_DETERMINISTIC, POLICY_PUSH_TO_HUB, POLICY_USE_AMP

    if args.dataset_repo_id is not None:
        DATASET_REPO_ID = args.dataset_repo_id
    if args.dataset_root is not None:
        DATASET_ROOT = args.dataset_root
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a policy (default ACT) using the same pipeline as lerobot_train.py.",
    )
    p.add_argument("--dataset-repo-id", type=str, default=None, help="Maps to dataset.repo_id")
    p.add_argument("--dataset-root", type=str, default=None, help="Maps to dataset.root")
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
    return p.parse_args()


def main() -> None:
    register_third_party_plugins()
    _apply_cli_to_module_constants(parse_args())
    cfg = build_train_config()
    cfg.validate()
    accelerator = create_train_accelerator(cfg)
    train(cfg, accelerator=accelerator)


if __name__ == "__main__":
    main()
