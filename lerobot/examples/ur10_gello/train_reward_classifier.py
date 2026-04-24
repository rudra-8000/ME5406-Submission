"""
train_reward_classifier.py
Train a ResNet18 binary reward classifier from labelled frames.
Handles class imbalance via weighted sampler.

Usage:
    python train_reward_classifier.py \\
        --success_dir ~/rudra/lerobot/reward_data/success/wrist \\
        --failure_dir ~/rudra/lerobot/reward_data/failure/wrist \\
        --out ./reward_classifier_wrist.pt \\
        --epochs 30
"""

import argparse, os, random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ── Dataset ──────────────────────────────────────────────────────────────────

class RewardDataset(Dataset):
    def __init__(self, success_dir: Path, failure_dir: Path, transform=None, val_split=0.15, mode="train", seed=42):
        success_imgs = sorted(success_dir.glob("*.jpg")) + sorted(success_dir.glob("*.png"))
        failure_imgs = sorted(failure_dir.glob("*.jpg")) + sorted(failure_dir.glob("*.png"))

        print(f"  success: {len(success_imgs)} images")
        print(f"  failure: {len(failure_imgs)} images")

        labeled = [(p, 1) for p in success_imgs] + [(p, 0) for p in failure_imgs]
        random.seed(seed)
        random.shuffle(labeled)

        n_val = max(1, int(len(labeled) * val_split))
        if mode == "val":
            self.data = labeled[:n_val]
        else:
            self.data = labeled[n_val:]

        self.transform = transform
        self.labels = [lbl for _, lbl in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_sample_weights(self):
        """Per-sample weights for WeightedRandomSampler to balance classes."""
        n_pos = sum(self.labels)
        n_neg = len(self.labels) - n_pos
        w_pos = 1.0 / max(n_pos, 1)
        w_neg = 1.0 / max(n_neg, 1)
        return [w_pos if l == 1 else w_neg for l in self.labels]


# ── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ── Training ─────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    success_dir = Path(args.success_dir).expanduser()
    failure_dir = Path(args.failure_dir).expanduser()

    train_ds = RewardDataset(success_dir, failure_dir, get_transforms(True),  mode="train")
    val_ds   = RewardDataset(success_dir, failure_dir, get_transforms(False), mode="val")

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # Weighted sampler to balance class imbalance
    sample_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,   num_workers=4)

    # ResNet18 with pretrained weights, replace head
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 2)
    )
    model = model.to(device)

    # Freeze backbone for first few epochs, then unfreeze
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    unfreeze_epoch = max(1, args.epochs // 4)

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone partway through training
        if epoch == unfreeze_epoch:
            print(f"\n[Epoch {epoch}] Unfreezing full backbone")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - unfreeze_epoch)

        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            train_correct += (out.argmax(1) == labels).sum().item()
            train_total += len(labels)

        scheduler.step()

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        val_tp = val_fp = val_fn = val_tn = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
                val_tp += ((preds == 1) & (labels == 1)).sum().item()
                val_fp += ((preds == 1) & (labels == 0)).sum().item()
                val_fn += ((preds == 0) & (labels == 1)).sum().item()
                val_tn += ((preds == 0) & (labels == 0)).sum().item()

        train_acc = train_correct / max(train_total, 1)
        val_acc   = val_correct   / max(val_total, 1)
        precision = val_tp / max(val_tp + val_fp, 1)
        recall    = val_tp / max(val_tp + val_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss/max(train_total,1):.4f} "
              f"train_acc={train_acc:.3f} | "
              f"val_acc={val_acc:.3f} prec={precision:.3f} rec={recall:.3f} F1={f1:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "epoch": epoch,
            }, args.out)
            print(f"  ✓ Saved best model (val_acc={val_acc:.3f})")

    print(f"\\nBest val_acc: {best_val_acc:.3f} → {args.out}")

    # Final check: verify the saved model loads and does inference
    print("\\nRunning quick inference test on 3 success images...")
    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tf = get_transforms(False)
    success_imgs = list(Path(args.success_dir).expanduser().glob("*.jpg"))[:3]
    with torch.no_grad():
        for p in success_imgs:
            img = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            logits = model(img)
            prob_success = torch.softmax(logits, dim=1)[0, 1].item()
            print(f"  {p.name}: P(success)={prob_success:.3f}")

    print("\\nDone. Use this classifier in serl_finetune_act.py:")
    print(f"  --reward_mode classifier \\\\")
    print(f"  --reward_classifier_path {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--success_dir", required=True)
    parser.add_argument("--failure_dir", required=True)
    parser.add_argument("--out", default="./reward_classifier_wrist.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    train(args)