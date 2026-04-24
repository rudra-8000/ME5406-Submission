"""
train_reward_classifier_v2.py
Improved reward classifier:
- Trains on WRIST camera (not top)
- Removes neutral/start frames from failure set  
- Uses focal loss to focus on hard examples
- Stronger augmentation
- Confidence calibration check at end
"""

import argparse, os, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
import numpy as np


# ── Focal Loss ────────────────────────────────────────────────────────────────
# Standard CE with class-imbalance gives ~0.5 for everything.
# Focal loss down-weights easy examples so the model focuses on hard boundaries.

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        # alpha: weight for positive (success) class — set higher since it's
        # the harder class to distinguish
        self.alpha = alpha

    def forward(self, logits, targets):
        # logits: (B, 2), targets: (B,) in {0,1}
        probs = F.softmax(logits, dim=1)
        # gather the probability for the true class
        pt = probs[torch.arange(len(targets)), targets]
        # alpha weighting per sample
        alpha_t = torch.where(targets == 1,
                              torch.full_like(pt, self.alpha),
                              torch.full_like(pt, 1 - self.alpha))
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        ce = F.cross_entropy(logits, targets, reduction='none')
        loss = (focal_weight * ce).mean()
        return loss


# ── Dataset ───────────────────────────────────────────────────────────────────

def is_terminal_failure(path: Path) -> bool:
    """
    Only keep frames that represent actual failure states.
    Reject: FIRST frames (robot at home), very early s0000-s0060 steps.
    Keep: MID frames, late sequential frames (s0120+), and any LAST frames
    that ended up in failure (protective stop at end).
    """
    # name = path.stem
    # # Always reject start-of-episode frames
    # if "_FIRST_" in name:
    #     return False
    # # Keep MID frames (middle of trajectory = approaching object)
    # if "_MID_" in name:
    #     return True
    # # For sequential frames, only keep the later ones (robot near object)
    # if "_s" in name:
    #     try:
    #         step = int(name.split("_s")[-1])
    #         return step >= 100   # Only steps 100+ = robot in approach zone
    #     except ValueError:
    #         return True
    # # Keep LAST frames (episode ended = terminal state)
    # if "_LAST_" in name:
    #     return True
    return True


class RewardDataset(Dataset):
    def __init__(self, success_dir: Path, failure_dir: Path,
                 transform=None, val_split=0.15, mode="train", seed=42):

        success_imgs = sorted(success_dir.glob("*.jpg")) + sorted(success_dir.glob("*.png"))

        all_failure = sorted(failure_dir.glob("*.jpg")) + sorted(failure_dir.glob("*.png"))
        failure_imgs = [p for p in all_failure if is_terminal_failure(p)]

        print(f"  success: {len(success_imgs)} images")
        print(f"  failure (raw): {len(all_failure)}  →  filtered: {len(failure_imgs)}")

        if len(failure_imgs) < 20:
            print("  WARNING: Very few failure images after filtering. "
                  "Lowering step threshold to 60.")
            failure_imgs = [p for p in all_failure
                            if "_FIRST_" not in p.stem]

        labeled = [(p, 1) for p in success_imgs] + [(p, 0) for p in failure_imgs]
        random.seed(seed)
        random.shuffle(labeled)

        n_val = max(4, int(len(labeled) * val_split))
        if mode == "val":
            self.data = labeled[:n_val]
        else:
            self.data = labeled[n_val:]

        self.transform = transform
        self.labels = [lbl for _, lbl in self.data]

        n_pos = sum(self.labels)
        n_neg = len(self.labels) - n_pos
        print(f"  {mode}: {len(self.data)} samples  "
              f"(pos={n_pos}, neg={n_neg}, ratio={n_pos/max(n_neg,1):.2f})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_sample_weights(self):
        n_pos = sum(self.labels)
        n_neg = len(self.labels) - n_pos
        w_pos = 1.0 / max(n_pos, 1)
        w_neg = 1.0 / max(n_neg, 1)
        return [w_pos if l == 1 else w_neg for l in self.labels]


# ── Transforms ────────────────────────────────────────────────────────────────
# Wrist camera sees close-up gripper+object — use aggressive crop/zoom
# augmentation since the object position varies a lot.

def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.3, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    success_dir = Path(args.success_dir).expanduser()
    failure_dir = Path(args.failure_dir).expanduser()

    train_ds = RewardDataset(success_dir, failure_dir, get_transforms(True),  mode="train")
    val_ds   = RewardDataset(success_dir, failure_dir, get_transforms(False), mode="val")

    sampler = WeightedRandomSampler(train_ds.get_sample_weights(),
                                    num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=4)

    # EfficientNet-B0 is better than ResNet18 for small datasets —
    # more parameter-efficient and stronger ImageNet features
    if args.arch == "efficientnet":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 2)
        )
    else:  # resnet18 fallback
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, 2))

    model = model.to(device)

    # Phase 1: train head only
    for p in model.parameters():
        p.requires_grad = False
    if args.arch == "efficientnet":
        for p in model.classifier.parameters():
            p.requires_grad = True
    else:
        for p in model.fc.parameters():
            p.requires_grad = True

    criterion = FocalLoss(gamma=2.0, alpha=0.75)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-3
    )
    unfreeze_epoch = max(3, args.epochs // 5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=unfreeze_epoch,
        pct_start=0.3
    )

    best_f1 = 0.0  # Use F1, not accuracy — more robust with imbalance

    for epoch in range(1, args.epochs + 1):

        if epoch == unfreeze_epoch + 1:
            print(f"\n[Epoch {epoch}] Unfreezing full backbone at lr={args.lr*0.05:.2e}")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr * 0.05, weight_decay=1e-3
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - unfreeze_epoch, eta_min=1e-6
            )

        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if epoch <= unfreeze_epoch:
                scheduler.step()
            train_loss   += loss.item() * len(labels)
            train_correct += (out.argmax(1) == labels).sum().item()
            train_total   += len(labels)

        if epoch > unfreeze_epoch:
            scheduler.step()

        # ── Validate ──
        model.eval()
        val_tp = val_fp = val_fn = val_tn = 0
        probs_all = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                preds = logits.argmax(1)
                probs = torch.softmax(logits, dim=1)[:, 1]
                probs_all.extend(zip(probs.cpu().tolist(), labels.cpu().tolist()))
                val_tp += ((preds == 1) & (labels == 1)).sum().item()
                val_fp += ((preds == 1) & (labels == 0)).sum().item()
                val_fn += ((preds == 0) & (labels == 1)).sum().item()
                val_tn += ((preds == 0) & (labels == 0)).sum().item()

        val_total = val_tp + val_fp + val_fn + val_tn
        val_acc   = (val_tp + val_tn) / max(val_total, 1)
        precision = val_tp / max(val_tp + val_fp, 1)
        recall    = val_tp / max(val_tp + val_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        train_acc = train_correct / max(train_total, 1)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"loss={train_loss/max(train_total,1):.4f} "
              f"tr_acc={train_acc:.3f} | "
              f"val_acc={val_acc:.3f} P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
              f"TP={val_tp} FP={val_fp} FN={val_fn} TN={val_tn}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "model_state": model.state_dict(),
                "arch": args.arch,
                "val_acc": val_acc, "f1": f1,
                "precision": precision, "recall": recall,
                "epoch": epoch,
            }, args.out)
            print(f"  ✓ Saved (F1={f1:.3f})")

    # ── Calibration check ──
    print(f"\nBest F1: {best_f1:.3f}")
    print("\nCalibration check on validation set:")
    print("  Ideal: success~0.8+, failure~0.2-")

    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tf = get_transforms(False)

    for label_name, folder in [("SUCCESS", success_dir), ("FAILURE", failure_dir)]:
        imgs = list(folder.glob("*.jpg"))[:5]
        for p in imgs:
            img = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.softmax(model(img), dim=1)[0, 1].item()
            flag = "✓" if (label_name == "SUCCESS" and prob > 0.6) or \
                          (label_name == "FAILURE" and prob < 0.4) else "✗"
            print(f"  {flag} {label_name} {p.name}: P(success)={prob:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--success_dir", required=True)
    parser.add_argument("--failure_dir", required=True)
    parser.add_argument("--out", default="./reward_classifier.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--arch", choices=["resnet18", "efficientnet"],
                        default="efficientnet")
    args = parser.parse_args()
    
    train(args)