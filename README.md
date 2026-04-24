# ME5406 — SERL-ACT: Residual Reinforcement Learning for Robot Grasping on a UR10

**ME5406 Final Project — National University of Singapore**

This repository contains the full implementation of a **SERL-style residual reinforcement learning system** built on top of a **LeRobot ACT (Action Chunking with Transformers)** policy, targeting a real **Universal Robots UR10** arm for a pick-and-place grasping task. The system learns a small corrective residual on top of a pre-trained imitation learning policy using **Soft Actor-Critic (SAC)**, guided by vision-based reward classifiers trained from human-labelled frames.

Rollout videos are in [`videos_serl/`](#rollout-videos).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Key Components](#key-components)
   - [ACT Base Policy](#act-base-policy)
   - [Residual Policy (SAC)](#residual-policy-sac)
   - [Reward Classifiers](#reward-classifiers)
   - [Shaped Reward Function](#shaped-reward-function)
   - [Recording & Dataset Pipeline](#recording--dataset-pipeline)
4. [Repository Structure](#repository-structure)
5. [Rollout Videos](#rollout-videos)
6. [Setup & Usage](#setup--usage)
7. [Design Decisions & Lessons Learned](#design-decisions--lessons-learned)

---

## Project Overview

The core problem: a UR10 arm is trained via imitation learning (ACT) to grasp a object and place it in a box. ACT alone is brittle — small distribution shifts in object position or orientation cause failures. The goal of this project is to apply online reinforcement learning *on top of* ACT, letting the robot self-improve from its own rollout experience on the real robot, without any simulation.

The approach follows the **SERL** (Sample Efficient Robot Learning) paradigm:

- An **ACT policy** provides a strong base action at every timestep.
- A **residual MLP** (trained by SAC) adds a small corrective delta to that action.
- Two **CNN-based reward classifiers** (wrist camera and top camera) provide dense reward signals by predicting grasp success and in-box placement probability from live images.
- A **human operator** provides sparse terminal signals by pressing a key to flag success or dangerous states.

This avoids the need for simulation, reward engineering by hand, or a reset robot — the robot simply tries, gets feedback from its cameras, and gradually improves.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Robot Loop (30 Hz)                     │
│                                                                 │
│   Observation (joint angles, camera frames)                     │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────┐    base action a_ACT                          │
│   │  ACT Policy  │ ──────────────────────────────┐             │
│   └─────────────┘                                │             │
│                                                  ▼             │
│   ┌──────────────────────────────┐   ┌──────────────────────┐  │
│   │  Residual MLP (SAC Actor)    │   │   Action Addition    │  │
│   │  input: [a_ACT, c_g, c_t]   │──▶│  a = a_ACT + Δa_res  │  │
│   └──────────────────────────────┘   └──────────────────────┘  │
│          │                                       │             │
│          │  Δa_res (clipped to α·range)          │             │
│          │                                       ▼             │
│          │                            UR10 executes action     │
│          │                                       │             │
│          ▼                                       ▼             │
│   SAC Critic / Replay Buffer          Reward Signal            │
│   (off-policy updates from            ┌─────────────────┐     │
│    replay buffer)                     │ Wrist classifier│ r_g │
│                                       │ Top classifier  │ r_t │
│                                       │ Table penalty   │     │
│                                       │ Step penalty    │     │
│                                       │ Terminal bonus  │     │
│                                       └─────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

The residual action is **scaled and clipped** to at most `α × joint_range` (typically α = 0.1–0.15) so the SAC actor cannot override the ACT policy — it can only nudge it. This is what makes the system safe for a real robot: the policy's primary authority always belongs to the pre-trained ACT.

---

## Key Components

### ACT Base Policy

The ACT policy (`lerobot/src/lerobot/`) was pre-trained offline using ~80 demonstrations collected via a **GELLO teleoperation device** (a shadow puppet arm mirroring the UR10 joint-by-joint). Demonstrations were recorded at 30 Hz with two cameras (wrist + top) and 7-DoF joint states (6 UR10 joints + gripper). The LeRobot framework stores demonstrations as **Parquet + MP4 datasets** on Hugging Face.

At inference time, ACT ingests an observation window of recent images and joint states, and outputs a *chunk* of future actions via a CVAE encoder-decoder. Only the first action of each chunk is executed before re-querying the policy (standard ACT deployment).

### Residual Policy (SAC)

The residual policy is a small MLP:

```
Input: [a_ACT (7), c_grasp (1), c_task (1)]  →  ℝ⁹
Hidden: Linear(9 → 256) → ReLU → Linear(256 → 256) → ReLU → Linear(256 → 7) → tanh
Output: Δa_res, scaled to α × joint_range
```

The **critic** is a twin-Q SAC network (two separate Q-MLPs) that learns the value of `(state, Δa_res)` pairs. A **replay buffer** of capacity 50,000 transitions stores real-robot experience, and the SAC update runs asynchronously (every N environment steps) to avoid blocking the control loop.

During a **warm-up phase** (first 500 steps), the residual is held at zero — the robot runs purely on ACT — so the replay buffer collects initial experience before any SAC updates begin.

### Reward Classifiers

Two binary classifiers provide dense per-step reward signals, eliminating the need for physical sensors or manual reward coding:

| Classifier | Camera | Architecture | What it predicts |
|---|---|---|---|
| Wrist classifier | Wrist-mounted | EfficientNet-B0, Dropout(0.4) → Linear(1280, 2), softmax | P(gripper has grasped the object) |
| Top classifier | Overhead | EfficientNet-B0, same head | P(object is inside the target box) |

Training data was extracted from recorded rollout episodes using `extract_reward_frames.py`, which splits frames into `success/` and `failure/` folders based on human-labelled episode outcomes. Both classifiers are **fine-tuned from ImageNet-pretrained weights** using cross-entropy loss, then frozen during RL training — only the residual MLP is updated online.

### Shaped Reward Function

The reward at each timestep is a sum of dense and sparse terms:

| Term | Formula | Type | Rationale |
|---|---|---|---|
| Step penalty | −0.05 | Dense | Encourages faster task completion |
| Table proximity penalty | −3.0 · 𝟙[tcp_z < 0.015 m] | Sparse | Prevents TCP from scraping the table |
| Grasp reward | 5.0 · (p_grasp − 0.80) | Dense | Rewards confident grasping; centred at threshold 0.80 |
| In-box reward | −1.5 · (1 − p_inbox) | Dense | Continuously penalises the object being out of the box |
| Terminal success bonus | +10,000 | Terminal sparse | Human operator presses success key when object lands in box |

The large terminal bonus (+10,000) is intentional — it ensures that any trajectory leading to a true success dramatically dominates all partial-reward trajectories in the critic's Q-function, preventing the robot from "farming" partial rewards without completing the task.

### Recording & Dataset Pipeline

- `record.py` / `record_tactile.py` — live demonstration recording from GELLO → LeRobot dataset format
- `extract_reward_frames.py` — extracts labelled frames from rollout videos for classifier training
- `quick_eval_classifier.py` — offline evaluation of trained classifiers on held-out frames
- `lerobot_dataset_viz_tactile.py` — dataset visualisation tool
- `client_ur10_control.py` — low-level UR10 RTDE interface (sends joint position commands, reads state)
- `serl_finetune_act.py` — the main online RL training loop integrating all components

---

## Repository Structure

```
ME5406-Submission/
├── lerobot/                        # Modified LeRobot framework
│   ├── src/lerobot/                # Core library: ACT model, dataset tools, training
│   ├── examples/ur10_gello/        # All UR10-specific scripts
│   │   ├── serl_finetune_act.py    # ★ Main SERL online RL training script
│   │   ├── record.py               # Demonstration recording
│   │   ├── record_tactile.py       # Recording with tactile sensor integration
│   │   ├── client_ur10_control.py  # UR10 RTDE robot controller
│   │   ├── extract_reward_frames.py# Frame extraction for classifier training
│   │   ├── quick_eval_classifier.py# Classifier offline evaluation
│   │   ├── lerobot_dataset_viz_tactile.py # Dataset visualisation
│   │   ├── reward_classifier_top_v2.pt    # Trained top-camera classifier
│   │   ├── reward_classifier_wrist_best.pt # Trained wrist classifier
│   │   └── reward_classifier_top_best.pt  # Best top-camera classifier checkpoint
│   ├── classifier_data/            # Labelled frames for classifier training
│   └── datasets/                   # LeRobot-format demonstration datasets
└── videos_serl/                    # ★ Rollout videos (see below)
```

---

## Rollout Videos

All videos are real robot rollouts on the physical UR10. They are not simulated.

### ✅ Successes

| Video | Description |
|---|---|
| [Success_vertical.mp4](https://github.com/rudra-8000/ME5406-Submission/blob/main/videos_serl/Success_vertical.mp4) | Full successful grasp and box placement — object oriented vertically. The residual policy visibly nudges the wrist to align before closing the gripper. |
| [Success_Horizontal.mp4](https://github.com/rudra-8000/ME5406-Submission/blob/main/videos_serl/Success_Horizontal.mp4) | Successful grasp with the object in a horizontal orientation — a harder case the ACT policy alone often failed on. |

### ⚠️ Instructive Partial Successes

| Video | Description |
|---|---|
| [Failed_To_Grasp_Successfully.mp4](https://github.com/rudra-8000/ME5406-Submission/blob/main/videos_serl/Failed_To_Grasp_Successfully.mp4) | Robot reaches and closes the gripper but the object slips — the wrist classifier correctly assigns low reward, driving the SAC critic to penalise this grasp geometry. |
| [Supported_But_Table_Close.mp4](https://github.com/rudra-8000/ME5406-Submission/blob/main/videos_serl/Supported_But_Table_Close.mp4) | Grasp succeeds but the TCP descends too close to the table surface, triggering the −3.0 proximity penalty. Illustrates the table penalty shaping arm behaviour upward. |
| [Unassisted_Successful_Localization_Of_Object_But_Too_Close_Table.mp4](https://github.com/rudra-8000/ME5406-Submission/blob/main/videos_serl/Unassisted_Successful_Localization_Of_Object_But_Too_Close_Table.mp4) | ACT base policy alone successfully locates the object but brings the wrist dangerously low — the residual (not yet trained sufficiently) fails to correct altitude in time. |

### ❌ Failure Modes

| Video | Description |
|---|---|
| [Failure_Missed_Object_Completely.mp4](https://github.com/rudra-8000/ME5406-Submission/blob/main/videos_serl/Failure_Missed_Object_Completely.mp4) | Early-training failure: ACT reaches for the object but the residual perturbs the approach enough to miss entirely. Demonstrates the exploration-exploitation tradeoff before the critic converges. |
| [Failure_Misaligned and Failed Grasp.mp4](https://github.com/rudra-8000/ME5406-Submission/blob/main/videos_serl/Failure_Misaligned%20and%20Failed%20Grasp.mp4) | Gripper is slightly misaligned — a common ACT failure case on off-centre object positions that RL training is intended to correct. |
| [Failure_Object_Hit.mp4](https://github.com/rudra-8000/ME5406-Submission/blob/main/videos_serl/Failure_Object_Hit.mp4) | Robot knocks the object instead of grasping it. The table penalty and wrist classifier both fire, producing a large negative reward that discourages this trajectory. |

### 🔄 Training Dynamics

| Video | Description |
|---|---|
| [Transition from ACT+Critic to ACT+SAC.mp4](https://github.com/rudra-8000/ME5406-Submission/blob/main/videos_serl/Transition%20from%20ACT%2BCritic%20to%20ACT%2BSAC.mp4) | Side-by-side comparison of a rollout with only the ACT+Critic (value estimation, no residual correction) versus the full ACT+SAC system. The SAC system's wrist alignment is visibly smoother. |
| [Human_Terminated_Object_Hit_Prevented.mp4](https://github.com/rudra-8000/ME5406-Submission/blob/main/videos_serl/Human_Terminated_Object_Hit_Prevented.mp4) | Human operator terminates the episode early after detecting an impending object collision — this is the human-in-the-loop safety mechanism. The episode is flagged as a failure in the replay buffer. |

---

## Setup & Usage

### Prerequisites

- Ubuntu 20.04 / 22.04
- Python ≥ 3.10
- UR10 with RTDE enabled (network reachable)
- Two USB cameras (wrist-mounted + overhead)
- GELLO teleoperation device (for demonstration collection only)

### Installation

```bash
# Clone this repository
git clone https://github.com/rudra-8000/ME5406-Submission.git
cd ME5406-Submission/lerobot

# Install LeRobot and dependencies
pip install -e ".[ur10]"
# or with full requirements:
pip install -r requirements-ubuntu.txt
```

### Step 1 — Collect Demonstrations

```bash
python examples/ur10_gello/record.py \
    --robot-ip 192.168.1.100 \
    --num-episodes 80 \
    --dataset-name my_grasp_demos
```

Demonstrations are saved as LeRobot-format Parquet + video datasets under `~/.cache/huggingface/lerobot/`.

### Step 2 — Train ACT Policy

```bash
python -m lerobot.scripts.train \
    policy=act \
    dataset_repo_id=my_grasp_demos \
    training.num_steps=100000
```

### Step 3 — Train Reward Classifiers

Extract frames from recorded episodes and label them, then train:

```bash
# Extract frames from episodes
python examples/ur10_gello/extract_reward_frames.py \
    --dataset-path ~/.cache/huggingface/lerobot/my_grasp_demos

# Train classifiers (wrist and top)
# [See classifier training script for training command]
```

Pre-trained classifiers (`reward_classifier_wrist_best.pt`, `reward_classifier_top_best.pt`, `reward_classifier_top_v2.pt`) are already included in the repo.

### Step 4 — Run SERL Online Fine-Tuning

```bash
python examples/ur10_gello/serl_finetune_act.py \
    --robot-ip 192.168.1.100 \
    --policy-path outputs/train/act_grasp/checkpoints/last \
    --wrist-classifier examples/ur10_gello/reward_classifier_wrist_best.pt \
    --top-classifier examples/ur10_gello/reward_classifier_top_best.pt \
    --max-episode-steps 80 \
    --warmup-steps 500 \
    --residual-scale 0.1 \
    --batch-size 256 \
    --gamma 0.99
```

The operator uses keyboard keys during rollouts:
- `s` — mark episode as **success** (triggers +10,000 terminal bonus, resets robot)
- `f` — mark episode as **failure / dangerous** (terminates episode, negative reward)
- `q` — quit training

---

## Design Decisions & Lessons Learned

**Why residual RL instead of full RL from scratch?**
A UR10 with no prior policy would spend the vast majority of rollouts in unsafe configurations. ACT provides a prior that keeps the arm near the task-relevant workspace, making exploration safe and sample-efficient. SAC only needs to learn the last few centimetres of correction.

**Why vision-based reward instead of a force/torque sensor?**
The UR10 lacks a wrist F/T sensor in this configuration. Vision classifiers trained from ~100 labelled frames per class are cheap to build and surprisingly robust once the lighting is controlled. The key insight is that grasping produces a consistent visual signature (object occluded by gripper fingers) that a fine-tuned EfficientNet-B0 can reliably detect.

**Why a large terminal bonus (+10,000)?**
With dense rewards only, SAC tends to converge to "good approach trajectories that never complete" — the agent maximises partial reward without paying the cost of the final grasp. The large terminal bonus ensures the Q-function correctly assigns massive value to states that lead to completion, overriding any partial-reward local optima.

**Human-in-the-loop termination**
Real robot RL without a simulator requires a safety abort mechanism. The human operator watches every rollout and terminates dangerous episodes (e.g., TCP heading toward the table or knocking the object). Aborted episodes are stored in the replay buffer as failures, which actually *helps* training by providing negative examples of dangerous joint configurations.

**Failure modes observed**
The most common failure was the ACT policy producing an approach trajectory that was spatially correct but rotationally misaligned — the object was in reach but the gripper fingers straddled it rather than closing around it. The residual successfully corrected wrist yaw in these cases after ~300 online steps. The remaining hard failures were cases where the object had rolled outside the ACT training distribution (>8 cm from the typical starting position).
