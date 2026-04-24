# ME5406 — SERL-ACT: Residual Reinforcement Learning for Robot Grasping on a UR10

**ME5406 Final Project — National University of Singapore**

This repository contains the full implementation of a **SERL-style residual reinforcement learning system** built on top of a **LeRobot ACT (Action Chunking with Transformers)** policy, targeting a real **Universal Robots UR10** arm for a pick-and-place grasping task. The system learns a small corrective residual on top of a pre-trained imitation learning policy using **Soft Actor-Critic (SAC)**, guided by vision-based reward classifiers trained from human-labelled frames.

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
8. [Bonus: Fine-tuned π₀ (Pi0) Rollouts](#bonus-fine-tuned-pi0-rollouts)
---

## Project Overview

The core problem: a UR10 arm is trained via imitation learning (ACT) to grasp a object and place it in a box. ACT alone is brittle — small distribution shifts in object position or orientation cause failures. The goal of this project is to apply online reinforcement learning *on top of* ACT, letting the robot self-improve from its own rollout experience on the real robot, without any simulation.

The approach follows the **SERL** (Sample Efficient Robot Learning) paradigm:

- An **ACT policy** provides a strong base action at every timestep.
- A **residual MLP** (trained by SAC) adds a small corrective delta to that action.
- Two **CNN-based reward classifiers** (wrist camera and top camera) provide dense reward signals by predicting grasp success and in-box placement probability from live images.
- A **human operator** provides sparse terminal signals by pressing a key to flag success or dangerous states — including imminent object collisions that the system cannot detect on its own.

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

The ACT policy (`lerobot/src/lerobot/`) was trained offline using 40 demonstrations collected via a **GELLO teleoperation device** (a shadow puppet arm mirroring the UR10 joint-by-joint). Demonstrations were recorded at 30 Hz with two cameras (wrist + top) and 7-DoF joint states (6 UR10 joints + gripper). The LeRobot framework stores demonstrations as **Parquet + MP4 datasets** on Hugging Face https://huggingface.co/datasets/rudy8k/grasp_place.

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
│   └── examples/ur10_gello/        # All UR10-specific scripts
│       ├── serl_finetune_act.py    # ★ Main SERL online RL training script
│       ├── record.py               # Demonstration recording
│       ├── record_tactile.py       # Recording with tactile sensor integration
│       ├── client_ur10_control.py  # UR10 RTDE robot controller
│       ├── extract_reward_frames.py# Frame extraction for classifier training
│       ├── quick_eval_classifier.py# Classifier offline evaluation
│       ├── lerobot_dataset_viz_tactile.py # Dataset visualisation
│       ├── reward_classifier_top_v2.pt    # Trained top-camera classifier
│       ├── reward_classifier_wrist_best.pt # Trained wrist classifier
│       └── reward_classifier_top_best.pt  # Best top-camera checkpoint
├── GIFS_SERL/                      # ★ Embedded rollout GIFs (see below)
└── videos_serl/                    # Full-resolution rollout videos
```

---

## Rollout Videos

All rollouts are on the **physical UR10**, no simulation. GIFs are embedded inline below; click any filename link to watch the full-resolution video.

---

### ✅ Successes

**Success — Vertical Object Orientation**

![Success vertical](https://raw.githubusercontent.com/rudra-8000/ME5406-Submission/main/GIFS_SERL/Success_vertical.gif)

Assissted but successful grasp and box placement with the object oriented vertically. The residual policy visibly nudges the wrist to align before closing the gripper. [📹 Original video](https://github.com/rudra-8000/ME5406-Submission/blob/main/videos_serl/Success_vertical.mp4)

---

**Success — Horizontal object Orientation**

![Success horizontal](https://raw.githubusercontent.com/rudra-8000/ME5406-Submission/main/GIFS_SERL/Success_Horizontal.gif)

Assisted but successful grasp of the object in a horizontal orientation and drop in the box — a harder case the ACT policy alone often failed on due to wrist misalignment. [📹 Original video](https://github.com/rudra-8000/ME5406-Submission/blob/main/videos_serl/Success_Horizontal.mp4)

---

### ⚠️ Instructive Partial Successes

**Grasp Attempt — Slip Failure**

![Failed to grasp successfully](https://raw.githubusercontent.com/rudra-8000/ME5406-Submission/main/GIFS_SERL/Failed_To_Grasp_Successfully.gif)

Robot reaches and closes the gripper but the object slips. The wrist classifier correctly assigns low reward, driving the SAC critic to penalise this grasp geometry in future rollouts. [📹 Original video](https://github.com/rudra-8000/ME5406-Submission/blob/main/videos_serl/Failed_To_Grasp_Successfully.mp4)

---

**Grasp Success — TCP Too Close to Table**

![Supported but table close](https://raw.githubusercontent.com/rudra-8000/ME5406-Submission/main/GIFS_SERL/Supported_But_Table_Close.gif)

Grasp succeeds but the TCP descends too close to the table surface, triggering the −3.0 proximity penalty. Illustrates how the table penalty shapes the arm to maintain safe clearance. [📹 Original video](https://github.com/rudra-8000/ME5406-Submission/blob/main/videos_serl/Supported_But_Table_Close.mp4)

---

**ACT Unassisted — Object Located, Too Close to Table**

![Unassisted localization too close table](https://raw.githubusercontent.com/rudra-8000/ME5406-Submission/main/GIFS_SERL/Unassisted_Successful_Localization_Of_Object_But_Too_Close_Table.gif)

The ACT base policy alone successfully locates the object but brings the wrist dangerously low. Without the residual trained sufficiently, altitude correction does not happen in time. [📹 Original video](https://github.com/rudra-8000/ME5406-Submission/blob/main/videos_serl/Unassisted_Successful_Localization_Of_Object_But_Too_Close_Table.mp4)

---

### ❌ Failure Modes

**Failure — Missed Object Completely**

![Failure missed object completely](https://raw.githubusercontent.com/rudra-8000/ME5406-Submission/main/GIFS_SERL/Failure_Missed_Object_Completely.gif)

Early-training failure: ACT reaches for the object but the residual perturbs the approach enough to miss entirely. Demonstrates the exploration-exploitation tradeoff before the critic has converged. [📹 Original video](https://github.com/rudra-8000/ME5406-Submission/blob/main/videos_serl/Failure_Missed_Object_Completely.mp4)

---

**Failure — Misaligned Gripper**

![Failure misaligned and failed grasp](https://raw.githubusercontent.com/rudra-8000/ME5406-Submission/main/GIFS_SERL/Failure_Misaligned-and-Failed-Grasp.gif)

Gripper is slightly misaligned — a common ACT failure case on off-centre object positions that RL training is intended to correct over successive rollouts. [📹 Original video](https://github.com/rudra-8000/ME5406-Submission/blob/main/videos_serl/Failure_Misaligned%20and%20Failed%20Grasp.mp4)

---

**Failure — Object Hit**

![Failure object hit](https://raw.githubusercontent.com/rudra-8000/ME5406-Submission/main/GIFS_SERL/Failure_Object_Hit.gif)

Robot knocks the object instead of grasping it. The table penalty and wrist classifier both fire, producing a large negative reward that discourages this trajectory. [📹 Original video](https://github.com/rudra-8000/ME5406-Submission/blob/main/videos_serl/Failure_Object_Hit.mp4)

---

### 🛑 Human-in-the-Loop Termination

**HIL — Object Hit Prevented by Operator**

![Human terminated object hit prevented](https://raw.githubusercontent.com/rudra-8000/ME5406-Submission/main/GIFS_SERL/Human_Terminated_Object_Hit_Prevented.gif)

The human operator manually terminates the episode after detecting that the robot is about to knock the object — a collision the automated reward system cannot anticipate. The operator presses the abort key, the episode is logged as a failure in the replay buffer, and the robot resets. This human-in-the-loop mechanism is a critical safety layer: the system has no predictive model of impending collisions, so the operator acts as the real-time safety monitor. [📹 Original video](https://github.com/rudra-8000/ME5406-Submission/blob/main/videos_serl/Human_Terminated_Object_Hit_Prevented.mp4)

---

### 🔄 Training Dynamics

**Transition: ACT+Critic → ACT+SAC**

![Transition from ACT Critic to ACT SAC](https://raw.githubusercontent.com/rudra-8000/ME5406-Submission/main/GIFS_SERL/Transition-from-ACT-Critic-to-ACT-SAC.gif)

Transtion of a rollout from only the ACT+Critic (value estimation, no residual correction applied) to the full ACT+SAC system with the trained residual. The SERL Pipeline's failure is clearly visible here as the robot switches from smooth to random jerky actions. [📹 Original video](https://github.com/rudra-8000/ME5406-Submission/blob/main/videos_serl/Transition%20from%20ACT%2BCritic%20to%20ACT%2BSAC.mp4)

---

## Setup & Usage

### Prerequisites

- Ubuntu 20.04 / 22.04
- Python ≥ 3.12
- UR10 with RTDE enabled (network reachable)
- Two USB cameras (wrist-mounted + overhead)
- GELLO teleoperation device (for demonstration collection only)

### Installation

First Install LeRobot: https://huggingface.co/docs/lerobot/installation 

Merge the files in the folder "lerobot/" of this repository with the cloned repo of https://github.com/huggingface/lerobot.

### Step 1 — Collect Demonstrations

```bash
python examples/ur10_gello/record.py \
    --robot-ip 192.168.1.100 \
    --num-episodes 80 \
    --dataset-name my_grasp_demos
```



### Step 2 — Train ACT Policy

```bash
  python examples/ur10_gello/ur10_train.py \
    --dataset-repo-id /path/to/dataset \
    --policy-type act \
    --output-dir /path/to/checkpoints \
    --job-name act_test \
    --policy-device cuda \
    --wandb-enable false
    --batch-size 64
```

### Step 3 — Train Reward Classifiers

```bash
python examples/ur10_gello/extract_reward_frames.py \
    --dataset-path ~/.cache/huggingface/lerobot/my_grasp_demos
# Label frames into success/ and failure/ subdirectories, then train classifiers.
```
```bash
python examples/ur10_gello/save_classifier_frames.py \
    --checkpoint_path /path/to/checkpoint/ \
    --robot-ws-port 8766 \
    --num_episodes 15 \
    --max_episode_steps 300 \
    --out_dir ./classifier_data \
    --save_every_n_steps 5
# Run ACT Policy and collect frames and manually classify collected frames into success/ and failure/ subdirectories, then train classifiers.
```
Pre-trained classifiers (`reward_classifier_wrist_v2.pt`, `reward_classifier_top_v2.pt`) are already included in the repo and can be used directly.

### Step 4 — Run SERL Online Fine-Tuning

```bash
python examples/ur10_gello/serl_finetune_act.py \
    --robot-ip 192.168.1.100 \
    --policy-path path/to/policy/checkpoint \
    --wrist-classifier examples/ur10_gello/reward_classifier_wrist_v2.pt \
    --top-classifier examples/ur10_gello/reward_classifier_top_v2.pt \
    --max-episode-steps 600 \
    --warmup-steps 5000 \
    --residual-scale 0.02 \
    --batch-size 64 \
    --gamma 0.99
```

**Operator keyboard controls during rollouts:**

| Key | Action |
|---|---|
| `s+Enter` | Mark episode as **truncated** → triggers -50 truncation penalty, robot resets |
| `Ctrl+C` | Quit training |

> ⚠️ The operator must actively monitor every rollout. There is no automated collision prediction — the `s+Enter` key is the only mechanism to prevent the robot from hitting the object or entering an unsafe configuration.

---

## Design Decisions & Lessons Learned

**Why residual RL instead of full RL from scratch?**
A UR10 with no prior policy would spend the vast majority of rollouts in unsafe configurations. ACT provides a prior that keeps the arm near the task-relevant workspace, making exploration safe and sample-efficient. SAC only needs to learn the last few centimetres of correction.

**Why vision-based reward instead of a force/torque sensor?**
The UR10 lacks a wrist F/T sensor in this configuration. Vision classifiers trained from ~100 labelled frames per class are cheap to build and surprisingly robust once lighting is controlled. The key insight is that grasping produces a consistent visual signature (object occluded by gripper fingers) that a fine-tuned EfficientNet-B0 can reliably detect.

**Why a large terminal bonus (+10,000)?**
With dense rewards only, SAC tends to converge to "good approach trajectories that never complete" — the agent maximises partial reward without paying the cost of the final grasp. The large terminal bonus ensures the Q-function correctly assigns massive value to states that lead to completion, overriding any partial-reward local optima.

**Human-in-the-loop termination**
The system has no predictive model of impending collisions. When the robot is about to hit the object or enter a dangerous configuration, only the human operator can intervene by pressing the abort key. These aborted episodes are stored in the replay buffer as failures, which helps training by providing negative Q-value targets for dangerous joint configurations — turning every near-miss into a useful learning signal.

**Failure modes observed**
The most common failure was the ACT policy producing an approach trajectory that was spatially correct but rotationally misaligned — the object was in reach but the gripper fingers straddled it rather than closing around it. The residual successfully corrected wrist yaw in these cases after ~300 online steps. The remaining hard failures were cases where the object had rolled outside the ACT training distribution (>8 cm from the typical starting position).


---

## Bonus: Fine-tuned Pi0 Rollouts

As a comparison baseline, the **π₀ (Pi0)** vision-language-action model from [Physical Intelligence](https://github.com/Physical-Intelligence/openpi) was fine-tuned on the same UR10 demonstration dataset used to train ACT.

**Fine-tuning details:**
- **Base model:** π₀ (openpi)
- **Steps:** 37,000
- **Batch size:** 96
- **Task language prompt:** `"pick up object and place in box"`
- **Adapted from:** [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)

π₀ is a flow-matching VLA policy that conditions on both camera observations and a natural language task description. Fine-tuning it on the UR10 demonstrations allows direct comparison against the ACT + SERL residual RL approach in terms of raw policy quality from offline training alone, without any online RL correction.

---

### Rollout

**Rollout 0001**

![Pi0 rollout 0001](https://raw.githubusercontent.com/rudra-8000/ME5406-Submission/main/Pi0_Rollouts/rollout_0001.gif)

[📹 Original video](https://github.com/rudra-8000/ME5406-Submission/blob/main/Pi0_Rollouts/rollout_0001.mp4)