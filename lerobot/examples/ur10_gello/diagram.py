
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(22, 15))
ax.set_xlim(0, 22)
ax.set_ylim(0, 15)
ax.axis('off')
fig.patch.set_facecolor('#0F0F14')
ax.set_facecolor('#0F0F14')

# ─── PALETTE ────────────────────────────────────────────────────────────────
C = {
    'bg':       '#0F0F14',
    'surf1':    '#1A1A24',
    'surf2':    '#1F1F2E',
    'surf3':    '#252538',
    'act_hdr':  '#1E3A5F',
    'act_body': '#0D2035',
    'sac_hdr':  '#3A1E1E',
    'sac_body': '#220D0D',
    'obs_hdr':  '#1E3A28',
    'obs_body': '#0D2015',
    'rew_hdr':  '#3A3A1E',
    'rew_body': '#22220D',
    'grid_hdr': '#2A1E3A',
    'grid_body':'#150D22',
    'border_act':'#4A9ED4',
    'border_sac':'#D44A4A',
    'border_obs':'#4AD47A',
    'border_rew':'#D4C94A',
    'border_grid':'#9A4AD4',
    'text_w':   '#F0F0F8',
    'text_m':   '#B0B0C8',
    'text_f':   '#7070A0',
    'arrow':    '#8888BB',
    'arrow_hl': '#D4AA4A',
    'divider':  '#2A2A40',
}

def box(ax, x, y, w, h, facecolor, edgecolor, lw=1.2, radius=0.18, alpha=1.0):
    b = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(b)
    return b

def hdr(ax, x, y, w, h, facecolor, edgecolor, lw=1.2, radius=0.18):
    b = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=4)
    ax.add_patch(b)

def txt(ax, x, y, s, size=8.5, color='#F0F0F8', ha='center', va='center',
        weight='normal', style='normal', zorder=5):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color,
            fontweight=weight, fontstyle=style, zorder=zorder,
            fontfamily='DejaVu Sans')

def arrow(ax, x1, y1, x2, y2, color='#8888BB', lw=1.5, style='->', zorder=6):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                        connectionstyle='arc3,rad=0.0'),
        zorder=zorder)

def arrow_label(ax, x, y, s, color='#D4AA4A', size=7.0):
    ax.text(x, y, s, ha='center', va='center', fontsize=size,
            color=color, fontweight='bold', zorder=7,
            bbox=dict(boxstyle='round,pad=0.2', fc='#1A1A24', ec='none', alpha=0.85))

# ═══════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════
txt(ax, 11, 14.55, 'HIL-SERL Pick-and-Place System Architecture', size=15, weight='bold', color='#F0F0F8')
txt(ax, 11, 14.1, 'UR10 + ACT + Residual SAC Policy  •  ME5406, NUS 2026', size=9, color='#9090B8')

# ─── thin top divider ───────────────────────────────────────────────────────
ax.plot([0.3, 21.7], [13.8, 13.8], color=C['divider'], lw=0.8)

# ═══════════════════════════════════════════════════════════════════════════
# COLUMN GUIDES (invisible – just for layout reference)
# Columns:  A=0.3  B=5.1  C=9.9  D=14.7  E=19.5
# ═══════════════════════════════════════════════════════════════════════════

# ─── SECTION LABELS ────────────────────────────────────────────────────────
for label, xc, yc, clr in [
    ('OBSERVATION\nSTACK', 1.65, 13.3, C['border_obs']),
    ('BASE POLICY\n(Frozen ACT)', 6.5, 13.3, C['border_act']),
    ('RESIDUAL\nCORRECTOR', 11.5, 13.3, C['border_sac']),
    ('REWARD\nSIGNALS', 16.0, 13.3, C['border_rew']),
    ('ROBOT\nEXECUTION', 20.2, 13.3, '#AAAACC'),
]:
    txt(ax, xc, yc, label, size=7.5, color=clr, weight='bold')

ax.plot([0.3, 21.7], [12.75, 12.75], color=C['divider'], lw=0.5, ls='--')


# ═══════════════════════════════════════════════════════════════════════════
# COLUMN A:  OBSERVATION STACK
# ═══════════════════════════════════════════════════════════════════════════
OBS_X, OBS_Y, OBS_W = 0.3, 1.3, 3.0

# Outer frame
box(ax, OBS_X, OBS_Y, OBS_W, 11.2, C['obs_body'], C['border_obs'], lw=1.8, radius=0.25)
hdr(ax, OBS_X, OBS_Y + 9.8, OBS_W, 1.4, C['obs_hdr'], C['border_obs'], lw=1.8)
txt(ax, OBS_X + OBS_W/2, OBS_Y + 10.5, 'Observation  $o_t$', size=9.5, weight='bold', color=C['border_obs'])
txt(ax, OBS_X + OBS_W/2, OBS_Y + 10.08, 'collected @ 30 Hz', size=7.5, color=C['text_m'])

# sub-boxes inside OBS
def obs_sub(y_top, label, detail, color):
    bx = OBS_X + 0.15
    by = OBS_Y + y_top - 0.95
    bw = OBS_W - 0.3
    bh = 0.88
    box(ax, bx, by, bw, bh, C['surf2'], color, lw=1.0, radius=0.12)
    txt(ax, bx + bw/2, by + bh*0.62, label, size=8.0, weight='bold', color='#F0F0F8')
    txt(ax, bx + bw/2, by + bh*0.22, detail, size=7.0, color=C['text_m'])
    return bx + bw/2, by + bh/2

cxA = OBS_X + OBS_W/2
obs_sub(9.4,  r'Joint Angles  $\mathbf{q} \in \mathbb{R}^6$', 'RTDE @ 30 Hz, radians', '#6ABEFF')
obs_sub(8.2,  r'Gripper State  $g \in [0,1]$', 'Normalised position', '#6ABEFF')
obs_sub(7.0,  r'TCP Pose  $\mathbf{p} \in \mathbb{R}^3$', 'Cartesian (x,y,z)', '#6ABEFF')
obs_sub(5.8,  r'$I_{\mathrm{top}}$: D435 cam_high', '640×480 RGB @ 30 FPS', '#66FFAA')
obs_sub(4.6,  r'$I_{\mathrm{wrist}}$: D435 cam_right_wrist', '640×480 RGB @ 30 FPS', '#66FFAA')
obs_sub(3.4,  r'State vector  $\mathbf{s}_t \in \mathbb{R}^7$', r'[$q_0..q_5$, $g$]  →  normalised (μ,σ)', '#FFAA55')
obs_sub(2.2,  r'Prev. action  $\mathbf{a}_{t-1}$', 'Fed to residual MLP', '#AA88FF')


# ═══════════════════════════════════════════════════════════════════════════
# COLUMN B:  FROZEN ACT POLICY
# ═══════════════════════════════════════════════════════════════════════════
ACT_X, ACT_Y, ACT_W = 3.65, 1.3, 5.15

box(ax, ACT_X, ACT_Y, ACT_W, 11.2, C['act_body'], C['border_act'], lw=1.8, radius=0.25)
hdr(ax, ACT_X, ACT_Y + 9.8, ACT_W, 1.4, C['act_hdr'], C['border_act'], lw=1.8)
txt(ax, ACT_X + ACT_W/2, ACT_Y + 10.5, 'Frozen ACT Policy', size=9.5, weight='bold', color=C['border_act'])
txt(ax, ACT_X + ACT_W/2, ACT_Y + 10.08,
    'ACT: Action Chunking with Transformers  (Zhao et al., 2023)', size=7.2, color=C['text_m'])

# Frozen badge
box(ax, ACT_X + 3.2, ACT_Y + 9.85, 1.7, 0.52, '#1A3A1A', '#55CC55', lw=0.9, radius=0.12)
txt(ax, ACT_X + 4.05, ACT_Y + 10.12, '🔒  FROZEN  (80k steps)', size=7.0, color='#55CC55')

# ── Visual encoder ──────────────────────────────────────────────────────────
bx = ACT_X + 0.2;  by = ACT_Y + 8.05;  bw = 4.75;  bh = 1.55
box(ax, bx, by, bw, bh, C['surf3'], C['border_act'], lw=0.9, radius=0.14)
txt(ax, bx + bw/2, by + bh - 0.28, 'Visual Encoder  (ResNet-18, ImageNet)', size=8.0, weight='bold', color='#8BC8F8')
# two sub-encoders
for xi, lbl in [(0.25, r'$I_{\mathrm{top}}$'), (2.55, r'$I_{\mathrm{wrist}}$')]:
    sx = bx + xi; sy = by + 0.15; sw = 2.0; sh = 0.82
    box(ax, sx, sy, sw, sh, '#0D1E2E', C['border_act'], lw=0.7, radius=0.1)
    txt(ax, sx + sw/2, sy + sh*0.65, lbl, size=8.5, weight='bold', color='#8BC8F8')
    txt(ax, sx + sw/2, sy + sh*0.25, '→ feature  512-d', size=7.0, color=C['text_m'])

# ── cVAE Encoder (train-time) ───────────────────────────────────────────────
bx = ACT_X + 0.2;  by = ACT_Y + 6.4;  bw = 4.75;  bh = 1.4
box(ax, bx, by, bw, bh, '#0E1A2A', '#4499CC', lw=0.9, radius=0.14)
txt(ax, bx + bw/2, by + bh - 0.30, 'cVAE Encoder  (train only)', size=8.0, weight='bold', color='#4499CC')
txt(ax, bx + bw/2, by + bh - 0.65, r'$q_\phi(\mathbf{z}\mid a_{t:t+T_c},\,o_t)$', size=8.5, color='#99CCFF')
txt(ax, bx + bw/2, by + 0.26,
    r'$\mathbf{z}\sim\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\sigma})$   latent dim = 32', size=7.5, color=C['text_m'])

# ── Transformer Decoder ─────────────────────────────────────────────────────
bx = ACT_X + 0.2;  by = ACT_Y + 4.55;  bw = 4.75;  bh = 1.6
box(ax, bx, by, bw, bh, '#0D1E35', '#5599DD', lw=0.9, radius=0.14)
txt(ax, bx + bw/2, by + bh - 0.30, 'Transformer Decoder', size=8.0, weight='bold', color='#5599DD')
txt(ax, bx + bw/2, by + bh - 0.66, 'dim=512, 8 heads, 1 dec layer', size=7.5, color=C['text_m'])
txt(ax, bx + bw/2, by + bh - 0.97,
    r'Input: $\mathbf{z}$ + obs tokens + positional embeddings', size=7.0, color=C['text_m'])
txt(ax, bx + bw/2, by + 0.28,
    r'Output: chunk $\hat{\mathbf{a}}_{t:t+T_c}\in\mathbb{R}^{100\times7}$', size=8.0, color='#99CCFF', weight='bold')

# ── Temporal Ensembling ──────────────────────────────────────────────────────
bx = ACT_X + 0.2;  by = ACT_Y + 2.75;  bw = 4.75;  bh = 1.55
box(ax, bx, by, bw, bh, C['surf3'], '#44BBCC', lw=0.9, radius=0.14)
txt(ax, bx + bw/2, by + bh - 0.3, 'Temporal Ensembling', size=8.0, weight='bold', color='#44BBCC')
txt(ax, bx + bw/2, by + bh - 0.65,
    r'$w_i = e^{-m \cdot i},\ m=0.06$', size=8.5, color='#AAFFEE')
txt(ax, bx + bw/2, by + bh - 0.97,
    r'$\hat{\mathbf{a}}_t = \frac{\sum w_i \hat{\mathbf{a}}_{t|t-i}}{\sum w_i}$', size=8.5, color='#AAFFEE')
txt(ax, bx + bw/2, by + 0.28, r'Smooth base action $\mathbf{a}_\mathrm{ACT}\in\mathbb{R}^7$',
    size=7.5, color=C['text_m'])

# ── ACT output label ─────────────────────────────────────────────────────────
bx = ACT_X + 0.2;  by = ACT_Y + 1.5;  bw = 4.75;  bh = 1.0
box(ax, bx, by, bw, bh, '#0A1520', '#44BBCC', lw=0.9, radius=0.12)
txt(ax, bx + bw/2, by + bh/2 + 0.14,
    r'$\mathbf{a}_\mathrm{ACT} \in \mathbb{R}^7$  (abs. joint targets + gripper)',
    size=8.5, color='#AAFFEE', weight='bold')
txt(ax, bx + bw/2, by + 0.22, 'Clipped gripper to [0, 0.95]', size=7.0, color=C['text_m'])


# ═══════════════════════════════════════════════════════════════════════════
# COLUMN C:  RESIDUAL SAC CORRECTOR
# ═══════════════════════════════════════════════════════════════════════════
SAC_X, SAC_Y, SAC_W = 9.15, 1.3, 5.1

box(ax, SAC_X, SAC_Y, SAC_W, 11.2, C['sac_body'], C['border_sac'], lw=1.8, radius=0.25)
hdr(ax, SAC_X, SAC_Y + 9.8, SAC_W, 1.4, C['sac_hdr'], C['border_sac'], lw=1.8)
txt(ax, SAC_X + SAC_W/2, SAC_Y + 10.5, 'Residual SAC Corrector', size=9.5, weight='bold', color=C['border_sac'])
txt(ax, SAC_X + SAC_W/2, SAC_Y + 10.08,
    'Soft Actor-Critic  (Haarnoja et al., 2018)  —  trainable', size=7.2, color=C['text_m'])

# trainable badge
box(ax, SAC_X + 3.0, SAC_Y + 9.85, 1.85, 0.52, '#3A1010', '#FF7070', lw=0.9, radius=0.12)
txt(ax, SAC_X + 3.93, SAC_Y + 10.12, '⚡ TRAINABLE (online RL)', size=7.0, color='#FF7070')

# ── Input feature ────────────────────────────────────────────────────────────
bx = SAC_X + 0.2;  by = SAC_Y + 8.05;  bw = 4.7;  bh = 1.55
box(ax, bx, by, bw, bh, C['surf3'], '#DD6666', lw=0.9, radius=0.14)
txt(ax, bx + bw/2, by + bh - 0.3, 'Input Feature  φ(o_t)', size=8.0, weight='bold', color='#FF9999')
# v1 / v2 side by side
for xi, lbl, sub, clr in [
    (0.1, 'v1 (state-only)', r'$[q_0..q_5, g]\in\mathbb{R}^7$', '#CC8888'),
    (2.45, 'v2 (state+clf)', r'$[s_t, c_\mathrm{grasp}, c_\mathrm{task}]\in\mathbb{R}^9$', '#DD9999'),
]:
    sx = bx + xi; sy = by + 0.13; sw = 2.2; sh = 0.88
    box(ax, sx, sy, sw, sh, '#1A0D0D', '#996666', lw=0.7, radius=0.1)
    txt(ax, sx + sw/2, sy + sh*0.67, lbl, size=7.5, color=clr, weight='bold')
    txt(ax, sx + sw/2, sy + sh*0.25, sub, size=7.0, color=C['text_m'])

# ── Residual MLP ─────────────────────────────────────────────────────────────
bx = SAC_X + 0.2;  by = SAC_Y + 5.9;  bw = 4.7;  bh = 1.9
box(ax, bx, by, bw, bh, '#1A0A0A', '#EE6666', lw=0.9, radius=0.14)
txt(ax, bx + bw/2, by + bh - 0.3, 'Residual MLP  (Actor π_θ)', size=8.0, weight='bold', color='#FF9999')
# layer boxes
ly = by + bh - 0.72
for lbl, lw2 in [
    ('Linear(7/9)', 0.72), ('→ ReLU', 0.35),
    ('Linear(256)', 0.72), ('→ ReLU', 0.35),
    ('Linear(256)', 0.72), ('→ ReLU', 0.35),
    ('Linear(128)', 0.72), ('→ ReLU', 0.35),
    ('Linear(7)', 0.65), ('→ Tanh', 0.35),
]:
    box(ax, bx + 0.15, ly - 0.26, lw2 + 0.05, 0.27, '#280D0D', '#996666', lw=0.7, radius=0.06)
    txt(ax, bx + 0.15 + (lw2 + 0.05)/2, ly - 0.13, lbl, size=6.8, color='#FFAAAA')
    ly -= 0.295
txt(ax, bx + bw/2, by + 0.22,
    r'Output $\mathbf{a}_\mathrm{res}\in[-1,1]^7$  →  clamp then scale $\alpha$',
    size=7.2, color='#FFCCCC', weight='bold')

# ── SAC Critics ──────────────────────────────────────────────────────────────
bx = SAC_X + 0.2;  by = SAC_Y + 3.95;  bw = 4.7;  bh = 1.7
box(ax, bx, by, bw, bh, '#1A0808', '#CC5555', lw=0.9, radius=0.14)
txt(ax, bx + bw/2, by + bh - 0.3, 'Double-Q Critics  $Q_{ψ_1},\\ Q_{ψ_2}$', size=8.0, weight='bold', color='#FF8888')
txt(ax, bx + bw/2, by + bh - 0.65,
    r'$y_t = R_t + \gamma\min_j Q_j(s_{t+1},a^\prime) - \alpha_\mathrm{ent}\log\pi(a^\prime|s_{t+1})$',
    size=7.2, color='#FFBBBB')
txt(ax, bx + bw/2, by + bh - 1.0,
    r'$\mathcal{L}_\mathrm{critic} = \mathbb{E}[(Q_i(s,a)-y)^2]$',
    size=7.5, color='#FFBBBB')
txt(ax, bx + bw/2, by + 0.52,
    r'$\mathcal{L}_\mathrm{actor}= \mathbb{E}[\alpha_\mathrm{ent}\log\pi - \min_j Q_j]$',
    size=7.5, color='#FFBBBB')
txt(ax, bx + bw/2, by + 0.22,
    r'$\gamma=0.99\ \ H_\mathrm{target}=-7\ \ \alpha_\mathrm{ent}$ auto-tuned',
    size=7.0, color=C['text_m'])

# ── Warm-up / Replay ─────────────────────────────────────────────────────────
bx = SAC_X + 0.2;  by = SAC_Y + 2.3;  bw = 2.15;  bh = 1.4
box(ax, bx, by, bw, bh, C['surf3'], '#AA4444', lw=0.9, radius=0.12)
txt(ax, bx + bw/2, by + bh - 0.28, 'Warm-up Phase', size=7.5, weight='bold', color='#FF9999')
txt(ax, bx + bw/2, by + bh - 0.60, 'N=5000 steps', size=7.5, color='#FFBBBB')
txt(ax, bx + bw/2, by + 0.60, 'ACT-only (α=0)', size=7.0, color=C['text_m'])
txt(ax, bx + bw/2, by + 0.30, 'critics train', size=7.0, color=C['text_m'])

bx2 = SAC_X + 2.75;  by2 = SAC_Y + 2.3;  bw2 = 2.15;  bh2 = 1.4
box(ax, bx2, by2, bw2, bh2, C['surf3'], '#AA4444', lw=0.9, radius=0.12)
txt(ax, bx2 + bw2/2, by2 + bh2 - 0.28, 'Replay Buffer', size=7.5, weight='bold', color='#FF9999')
txt(ax, bx2 + bw2/2, by2 + bh2 - 0.60, 'off-policy SAC', size=7.2, color=C['text_m'])
txt(ax, bx2 + bw2/2, by2 + 0.60, 'demo seeds  +', size=7.0, color=C['text_m'])
txt(ax, bx2 + bw2/2, by2 + 0.30, 'online transitions', size=7.0, color=C['text_m'])

# ── Summation block ───────────────────────────────────────────────────────────
bx = SAC_X + 0.2;  by = SAC_Y + 1.5;  bw = 4.7;  bh = 0.62
box(ax, bx, by, bw, bh, '#200808', '#FF7070', lw=1.2, radius=0.12)
txt(ax, bx + bw/2, by + bh/2 + 0.1,
    r'$\mathbf{a}_\mathrm{total} = \mathbf{a}_\mathrm{ACT} + \alpha\,\mathrm{clip}(\mathbf{a}_\mathrm{res},[-1,1])$',
    size=8.5, color='#FFAAAA', weight='bold')
txt(ax, bx + bw/2, by + 0.15, r'$\alpha\in\{0.05,\,0.02\}$  —  max correction ≈ 2.9° per joint',
    size=7.0, color=C['text_m'])


# ═══════════════════════════════════════════════════════════════════════════
# COLUMN D:  REWARD SIGNALS
# ═══════════════════════════════════════════════════════════════════════════
REW_X, REW_Y, REW_W = 14.6, 1.3, 4.65

box(ax, REW_X, REW_Y, REW_W, 11.2, C['rew_body'], C['border_rew'], lw=1.8, radius=0.25)
hdr(ax, REW_X, REW_Y + 9.8, REW_W, 1.4, C['rew_hdr'], C['border_rew'], lw=1.8)
txt(ax, REW_X + REW_W/2, REW_Y + 10.5, 'Reward Signals', size=9.5, weight='bold', color=C['border_rew'])
txt(ax, REW_X + REW_W/2, REW_Y + 10.08,
    r'$R_\mathrm{total} = R_\mathrm{task} + R_\mathrm{grasp} + R_\mathrm{smooth}$', size=7.8, color=C['text_m'])

# ── EfficientNet classifiers ─────────────────────────────────────────────────
for yi, cam, clf_lbl, oc in [
    (8.05, r'$I_\mathrm{top}$  (cam_high)', 'Object-in-box\ndetector', '#EE9900'),
    (5.85, r'$I_\mathrm{wrist}$  (cam_right_wrist)', 'Grasp-success\ndetector', '#DDBB00'),
]:
    bx = REW_X + 0.2;  by = REW_Y + yi;  bw = 4.25;  bh = 1.85
    box(ax, bx, by, bw, bh, '#1A1500', oc, lw=0.9, radius=0.14)
    txt(ax, bx + bw/2, by + bh - 0.3, cam, size=8.0, weight='bold', color=oc)
    txt(ax, bx + bw/2, by + bh - 0.62, 'EfficientNet-B0  (ImageNet pretrained)', size=7.5, color=C['text_m'])
    txt(ax, bx + bw/2, by + bh - 0.91, 'Linear(1280)→ReLU→Dropout(0.3)→Linear(1)→σ', size=6.8, color=C['text_m'])
    # v1 note
    box(ax, bx + 0.1, by + 0.65, bw - 0.2, 0.38, '#111100', '#886600', lw=0.6, radius=0.08)
    txt(ax, bx + bw/2, by + 0.84, 'v1 ResNet-18 → uninformative (~0.5 both classes)', size=6.6, color='#AA9900', style='italic')
    txt(ax, bx + bw/2, by + 0.34, clf_lbl, size=7.5, color=oc, weight='bold')
    txt(ax, bx + bw/2, by + 0.15, 'success ∈ [0.7,0.9]  |  failure ∈ [0.01,0.1]', size=6.8, color=C['text_m'])

# ── Reward component boxes ───────────────────────────────────────────────────
for yi, lbl, formula, clr in [
    (4.55, r'$R_\mathrm{task}$ (sparse)',
     r'$+10\cdot\mathbf{1}[f_\mathrm{top}>0.7]$,  once/ep', '#FFCC44'),
    (3.35, r'$R_\mathrm{grasp}$ (dense)',
     r'$f_\mathrm{wrist}(I_t)\in[0,1]$  every step', '#FFAA33'),
    (2.15, r'$R_\mathrm{smooth}$ (penalty)',
     r'$-0.01\|\mathbf{a}_\mathrm{res}\|_2^2$  smoothness', '#FF8844'),
]:
    bx = REW_X + 0.2;  by = REW_Y + yi;  bw = 4.25;  bh = 0.95
    box(ax, bx, by, bw, bh, '#1A1200', clr, lw=0.9, radius=0.12)
    txt(ax, bx + bw/2, by + bh - 0.28, lbl, size=8.0, weight='bold', color=clr)
    txt(ax, bx + bw/2, by + 0.26, formula, size=7.5, color=C['text_m'])

# ── Total reward summary ──────────────────────────────────────────────────────
bx = REW_X + 0.2;  by = REW_Y + 1.5;  bw = 4.25;  bh = 0.48
box(ax, bx, by, bw, bh, '#201800', C['border_rew'], lw=1.1, radius=0.1)
txt(ax, bx + bw/2, by + bh/2,
    r'$\gamma=0.99\ \ T=150$ steps  ($\approx$5 s/ep)  @ 30 Hz',
    size=7.5, color='#FFEEAA')


# ═══════════════════════════════════════════════════════════════════════════
# COLUMN E:  ROBOT EXECUTION
# ═══════════════════════════════════════════════════════════════════════════
ROB_X, ROB_Y, ROB_W = 19.6, 1.3, 2.1

box(ax, ROB_X, ROB_Y, ROB_W, 11.2, '#0E0E1A', '#8888CC', lw=1.5, radius=0.25)
hdr(ax, ROB_X, ROB_Y + 9.8, ROB_W, 1.4, '#1C1C30', '#8888CC', lw=1.5)
txt(ax, ROB_X + ROB_W/2, ROB_Y + 10.5, 'UR10 Robot', size=8.5, weight='bold', color='#AAAAFF')
txt(ax, ROB_X + ROB_W/2, ROB_Y + 10.1, '6-DOF, RTDE', size=7.0, color=C['text_m'])

for yi, lbl, sub in [
    (8.9, 'Joint Servo', 'RTDE @ 30 Hz'),
    (7.7, 'PincOpen Gripper', 'Dynamixel XM430'),
    (6.5, 'U2D2 Adapter', '1 Mbaud, P2.0'),
    (5.3, 'Mode: CurrPos', 'Op. Mode = 5'),
    (4.1, r'τ_lim = 0.4×I_max', '≈ 477 mA'),
    (2.9, 'Reset: home', 'smooth interp'),
    (1.8, 'Rollout video', 'MP4 side-by-side'),
]:
    bx = ROB_X + 0.15;  by = ROB_Y + yi;  bw = ROB_W - 0.3;  bh = 0.82
    box(ax, bx, by, bw, bh, C['surf2'], '#555588', lw=0.7, radius=0.1)
    txt(ax, bx + bw/2, by + bh*0.64, lbl, size=7.0, weight='bold', color='#AAAAFF')
    txt(ax, bx + bw/2, by + bh*0.22, sub, size=6.5, color=C['text_m'])


# ═══════════════════════════════════════════════════════════════════════════
# ── ARROWS ──────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

# obs → ACT encoder
arrow(ax, OBS_X + OBS_W, ACT_Y + 9.5, ACT_X, ACT_Y + 9.5, color=C['border_obs'], lw=1.5)
arrow_label(ax, (OBS_X + OBS_W + ACT_X)/2, ACT_Y + 9.78, r'images  $I_\mathrm{top},I_\mathrm{wrist}$')

# obs state → temporal ensembling + decoder
arrow(ax, OBS_X + OBS_W, ACT_Y + 5.75, ACT_X, ACT_Y + 5.75, color='#6ABEFF', lw=1.3)
arrow_label(ax, (OBS_X + OBS_W + ACT_X)/2, ACT_Y + 6.0, r'$\mathbf{s}_t$')

# cVAE -> decoder
arrow(ax, ACT_X + ACT_W/2, ACT_Y + 6.4, ACT_X + ACT_W/2, ACT_Y + 6.15, color='#4499CC', lw=1.3)

# decoder -> temporal ens
arrow(ax, ACT_X + ACT_W/2, ACT_Y + 4.55, ACT_X + ACT_W/2, ACT_Y + 4.3, color='#5599DD', lw=1.3)

# temporal ens -> ACT output
arrow(ax, ACT_X + ACT_W/2, ACT_Y + 2.75, ACT_X + ACT_W/2, ACT_Y + 2.5, color='#44BBCC', lw=1.3)

# ACT output -> summation (residual col)
arr_mx = SAC_X + 0.2 + 4.7/2
arrow(ax, ACT_X + ACT_W/2, ACT_Y + 1.5,
      ACT_X + ACT_W, ACT_Y + 1.75, color='#44BBCC', lw=1.8)
ax.annotate('', xy=(SAC_X + 2.55, ACT_Y + 1.79),
            xytext=(ACT_X + ACT_W, ACT_Y + 1.79),
            arrowprops=dict(arrowstyle='->', color='#44BBCC', lw=1.8), zorder=6)
arrow_label(ax, (ACT_X + ACT_W + SAC_X + 2.55)/2, ACT_Y + 2.05, r'$\mathbf{a}_\mathrm{ACT}$')

# obs → residual MLP
arrow(ax, OBS_X + OBS_W, SAC_Y + 9.4, SAC_X, SAC_Y + 9.4, color=C['border_obs'], lw=1.3)
arrow_label(ax, (OBS_X + OBS_W + SAC_X)/2, SAC_Y + 9.65, r'$\phi(o_t)$')

# feature v down to input then MLP
arrow(ax, SAC_X + SAC_W/2, SAC_Y + 8.05, SAC_X + SAC_W/2, SAC_Y + 7.8, color='#EE9999', lw=1.2)

# MLP -> critics
arrow(ax, SAC_X + SAC_W/2, SAC_Y + 5.9, SAC_X + SAC_W/2, SAC_Y + 5.65, color='#DD6666', lw=1.2)

# critics -> replay
arrow(ax, SAC_X + 3.4, SAC_Y + 3.95, SAC_X + 3.4, SAC_Y + 3.7, color='#CC5555', lw=1.2)

# Summation → UR10
arrow(ax, SAC_X + SAC_W, SAC_Y + 1.81, ROB_X, SAC_Y + 1.81, color=C['arrow_hl'], lw=2.0)
arrow_label(ax, (SAC_X + SAC_W + ROB_X)/2, SAC_Y + 2.1, r'$\mathbf{a}_\mathrm{total}\in\mathbb{R}^7$')

# UR10 images → reward classifiers (back arrow)
ax.annotate('', xy=(REW_X + REW_W, REW_Y + 9.5),
            xytext=(ROB_X, ROB_Y + 8.2),
            arrowprops=dict(arrowstyle='<-', color='#FFCC44', lw=1.6,
                            connectionstyle='arc3,rad=-0.25'), zorder=6)
arrow_label(ax, 20.55, REW_Y + 9.5, r'$I_t$', color='#FFCC44', size=7.0)

# obs images → reward classifiers
arrow(ax, OBS_X + OBS_W, REW_Y + 7.0, REW_X, REW_Y + 7.0, color='#FFAA44', lw=1.4)
arrow_label(ax, (OBS_X + OBS_W + REW_X)/2 + 2.0, REW_Y + 7.25, r'$I_\mathrm{top},I_\mathrm{wrist}$', color='#FFAA44')

# reward scalars → replay buffer
ax.annotate('', xy=(SAC_X + 3.85, SAC_Y + 2.95),
            xytext=(REW_X, SAC_Y + 2.95),
            arrowprops=dict(arrowstyle='<-', color='#FFCC44', lw=1.6,
                            connectionstyle='arc3,rad=0.0'), zorder=6)
arrow_label(ax, (SAC_X + 3.85 + REW_X)/2, SAC_Y + 3.22, r'$r_t$', color='#FFCC44')


# ─── GELLO TELEOPERATION note ────────────────────────────────────────────────
bx = OBS_X + 0.1;  by = OBS_Y - 0.95;  bw = OBS_W - 0.2;  bh = 0.75
box(ax, bx, by, bw, bh, '#0A0A18', '#6688FF', lw=0.9, radius=0.12)
txt(ax, bx + bw/2, by + bh*0.65, 'GELLO Teleoperation', size=7.5, weight='bold', color='#8899FF')
txt(ax, bx + bw/2, by + bh*0.25, 'joint-space @ 100 Hz → dataset (LeRobot HF fmt)', size=6.8, color=C['text_m'])

# demo arrow
arrow(ax, bx + bw/2, by + bh, bx + bw/2, OBS_Y + 1.2, color='#6688FF', lw=1.2)


# ─── WebSocket server-client bridge label ────────────────────────────────────
bx2 = SAC_X + 0.2;  by2 = SAC_Y - 0.95;  bw2 = 4.7;  bh2 = 0.75
box(ax, bx2, by2, bw2, bh2, '#1A0A0A', '#CC4444', lw=0.9, radius=0.12)
txt(ax, bx2 + bw2/2, by2 + bh2*0.65, 'GPU Server  (serl_finetune_act.py)', size=7.5, weight='bold', color='#FF8888')
txt(ax, bx2 + bw2/2, by2 + bh2*0.25, 'WebSocket ↔ Robot PC  (serl_client_ur10.py)', size=6.8, color=C['text_m'])

bx3 = OBS_X + 0.1;  by3 = by2;  bw3 = OBS_W - 0.2;  bh3 = 0.75
box(ax, bx3, by3, bw3, bh3, '#080818', '#6688CC', lw=0.9, radius=0.12)
txt(ax, bx3 + bw3/2, by3 + bh3*0.65, 'Robot PC  (serl_client_ur10.py)', size=7.2, weight='bold', color='#88AAFF')
txt(ax, bx3 + bw3/2, by3 + bh3*0.25, 'RTDE + D435 + Dynamixel', size=6.8, color=C['text_m'])


# ─── Legend ─────────────────────────────────────────────────────────────────
legend_items = [
    (C['border_obs'], 'Observation stream'),
    (C['border_act'], 'ACT policy (frozen)'),
    (C['border_sac'], 'SAC residual (trainable)'),
    (C['border_rew'], 'Reward / classifier'),
    (C['arrow_hl'],   'Action flow'),
    ('#6688FF',       'Data collection (GELLO)'),
]
lx = 14.65; ly = 0.7
for i, (clr, lbl) in enumerate(legend_items):
    x = lx + (i % 3) * 2.55
    y = ly - (i // 3) * 0.35
    ax.plot([x, x + 0.35], [y, y], color=clr, lw=2.2, solid_capstyle='round')
    txt(ax, x + 0.5, y, lbl, size=7.0, color=C['text_m'], ha='left', va='center')

txt(ax, 11.0, 0.55, 'Repos:  rudra-8000/lerobot_ur10  ·  lerobot_ur10_gello  ·  PincOpen_Dynamixel_XM430-W350-T',
    size=6.8, color=C['text_f'], ha='center')

plt.tight_layout(pad=0.3)
plt.savefig('./hil_serl_architecture.png', dpi=180, bbox_inches='tight',
            facecolor=C['bg'], edgecolor='none')
plt.close()
print("Saved.")