# # quick_eval_classifier.py
# import torch, cv2, sys

# import sys, os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
# from serl_finetune_act import RewardClassifier
# clf = RewardClassifier(device="cuda")
# ckpt = torch.load(sys.argv[1], map_location="cuda")
# clf.load_state_dict(ckpt["model_state"], strict=False)
# clf.eval()

# for path in sys.argv[2:]:
#     img = cv2.imread(path)
#     p = clf.predict_reward(img)
#     print(f"{p:.3f}  {path}")


"""
quick_eval_classifier.py  — compatible with train_reward_classifier_v2.py

Usage:
  python quick_eval_classifier.py reward_classifier.pt img1.jpg img2.jpg ...
"""

import sys
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image

# ── Load checkpoint ────────────────────────────────────────────────────────────
ckpt_path = sys.argv[1]
image_paths = sys.argv[2:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(ckpt_path, map_location=device)

arch = ckpt.get("arch", "efficientnet")   # default to efficientnet (v2 default)
print(f"Checkpoint arch: {arch}  |  best epoch: {ckpt.get('epoch','?')}  "
      f"|  F1={ckpt.get('f1', '?'):.3f}  val_acc={ckpt.get('val_acc','?'):.3f}")

# ── Rebuild model (must match train_reward_classifier_v2.py exactly) ───────────
if arch == "efficientnet":
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 2)
    )
else:
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, 2))

model.load_state_dict(ckpt["model_state"])
model = model.to(device)
model.eval()

# ── Val transform (identical to get_transforms(False) in v2 trainer) ──────────
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Evaluate ───────────────────────────────────────────────────────────────────
print(f"\n{'P(success)':>12}  {'Decision':>10}  {'File'}")
print("-" * 60)
for path in image_paths:
    img = Image.open(path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = F.softmax(logits, dim=1)[0, 1].item()
    decision = "SUCCESS ✓" if prob > 0.5 else "failure ✗"
    flag = "✓" if (prob > 0.6 or prob < 0.4) else "~"   # ~ = uncertain
    print(f"{flag}  {prob:>10.3f}  {decision:>10}  {path}")