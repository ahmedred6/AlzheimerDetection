import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score

from monai.transforms import Compose, RandAffine, RandGaussianNoise, RandAdjustContrast, RandGaussianSmooth




# =========================================================
# CONFIG
# =========================================================
@dataclass
class CFG:
    data_root: str = "AlzhiemerDisease/Hippocampus_Cubes"  # has AD/CN/MCI/*.npy
    groups: Tuple[str, ...] = ("AD", "NC", "MCI")
    n_splits: int = 5
    seed: int = 42

    # training
    epochs: int = 60
    batch_size: int = 16
    num_workers: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-4
    dropout_p: float = 0.2

    # scheduler
    use_cosine: bool = True

    # early stopping
    patience: int = 10   # epochs with no improvement in patient-level macro-F1

    # pretrained (MedicalNet hook)
    pretrained_path: str = "AlzhiemerDisease/models/resnet_18_23dataset.pth"  # set to a .pth if you have MedicalNet weights
    strict_pretrained: bool = False  # True if your keys match exactly

    # output
    out_dir: str = "runs_resnet3d_roi"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = CFG()


# =========================================================
# REPRODUCIBILITY
# =========================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic-ish (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# DATA INDEXING
# =========================================================
def build_index(data_root: str, groups: Tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for label, group in enumerate(groups):
        gdir = os.path.join(data_root, group)
        for fn in os.listdir(gdir):
            if not fn.endswith(".npy"):
                continue
            stem = fn[:-4]  # remove .npy
            patient_id = "_".join(stem.split("_")[:3])  # XXX_S_YYYY
            date = stem.split("_")[3] if len(stem.split("_")) > 3 else "UnknownDate"
            rows.append(
                dict(
                    path=os.path.join(gdir, fn),
                    label=label,
                    group=group,
                    patient_id=patient_id,
                    date=date,
                )
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No .npy files found under {data_root}/{groups}")
    return df


# =========================================================
# AUGMENTATION (SAFE FOR MRI ROI CUBES)
# =========================================================
def get_train_transforms():
    # NOTE: avoid left-right swap because you explicitly store Left channel=0, Right=1.
    # If you ever flip L/R, you'd also need to swap channels. We'll avoid that.
    return Compose([
        RandAffine(
            prob=0.5,
            rotate_range=(0.10, 0.10, 0.10),       # ~ +/- 5-6 degrees
            translate_range=(4, 4, 4),
            scale_range=(0.05, 0.05, 0.05),
            padding_mode="border",
        ),
        RandGaussianNoise(prob=0.25, mean=0.0, std=0.01),
        RandGaussianSmooth(prob=0.15, sigma_x=(0.3, 0.8), sigma_y=(0.3, 0.8), sigma_z=(0.3, 0.8)),
        RandAdjustContrast(prob=0.25, gamma=(0.9, 1.1)),
    ])


# =========================================================
# DATASET
# =========================================================
class HippocampusNPYDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.tf = get_train_transforms() if train else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        x = np.load(row["path"]).astype(np.float32)  # (2,64,64,64)
        y = int(row["label"])
        pid = row["patient_id"]

        x = torch.from_numpy(x)  # torch float32

        if self.train:
            x = self.tf(x)

        return x, torch.tensor(y, dtype=torch.long), pid


# =========================================================
# MODEL
# =========================================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes)
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


from models.resnet import resnet18
import torch.nn as nn
import torch

class MedicalNetClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # Load backbone exactly as trained
        self.backbone = resnet18(
            sample_input_D=64,
            sample_input_H=64,
            sample_input_W=64,
            shortcut_type='A',   # IMPORTANT
            num_seg_classes=1    # dummy, we won't use conv_seg
        )

        # Remove segmentation head
        self.backbone.conv_seg = nn.Identity()

        # Add classification head
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Run backbone up to layer4
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def load_medicalnet_weights(model, path):
    ckpt = torch.load(path, map_location="cpu")

    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    new_state = {}
    for k, v in ckpt.items():
        if k.startswith("module."):
            k = k[7:]
        new_state[k] = v

    missing, unexpected = model.backbone.load_state_dict(new_state, strict=False)

    print("MedicalNet weights loaded.")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)


# =========================================================
# METRICS: PATIENT-LEVEL AGGREGATION
# =========================================================
@torch.no_grad()
def evaluate_patient_level(model, loader, device) -> Dict:
    model.eval()

    # patient_id -> list of logits, list of labels
    logits_by_pid: Dict[str, List[np.ndarray]] = {}
    label_by_pid: Dict[str, int] = {}

    for x, y, pid in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).detach().cpu().numpy()
        y = y.numpy()

        for i in range(len(pid)):
            p = pid[i]
            logits_by_pid.setdefault(p, []).append(logits[i])
            # label is same for all visits in your folder structure;
            # still guard consistency
            if p in label_by_pid and label_by_pid[p] != int(y[i]):
                raise RuntimeError(f"Label mismatch for patient {p}")
            label_by_pid[p] = int(y[i])

    pids = sorted(logits_by_pid.keys())
    y_true = []
    y_pred = []

    for p in pids:
        avg_logits = np.mean(np.stack(logits_by_pid[p], axis=0), axis=0)
        pred = int(np.argmax(avg_logits))
        y_pred.append(pred)
        y_true.append(label_by_pid[p])

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "macro_f1": macro_f1,
        "balanced_acc": bal_acc,
        "confusion_matrix": cm,
        "n_patients": len(pids),
    }


# =========================================================
# TRAINING
# =========================================================
def compute_class_weights(train_df: pd.DataFrame, num_classes: int, device: str):
    counts = train_df["label"].value_counts().reindex(range(num_classes), fill_value=0).values.astype(np.float32)
    inv = 1.0 / np.maximum(counts, 1.0)
    w = inv / inv.sum()
    return torch.tensor(w, dtype=torch.float32, device=device)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for x, y, _pid in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / max(len(loader), 1)


def run_fold(fold: int, train_df: pd.DataFrame, val_df: pd.DataFrame):
    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.out_dir, f"fold{fold}_best.pt")

    train_ds = HippocampusNPYDataset(train_df, train=True)
    val_ds = HippocampusNPYDataset(val_df, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False
    )

    model = MedicalNetClassifier(num_classes=3)
    load_medicalnet_weights(model, cfg.pretrained_path)

    # Now expand first conv to 2 channels
    old_weight = model.backbone.conv1.weight.data

    new_conv = nn.Conv3d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

    new_conv.weight.data[:, 0] = old_weight[:, 0]
    new_conv.weight.data[:, 1] = old_weight[:, 0]

    model.backbone.conv1 = new_conv

    # replace final FC
    model.fc = nn.Linear(512, 3)

    model = model.to(cfg.device)
    # ========================================
    # PHASE 1 — Freeze backbone
    # ========================================
    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False
    
    class_w = compute_class_weights(train_df, num_classes=len(cfg.groups), device=cfg.device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    else:
        scheduler = None

    best_f1 = -1.0
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        # After 5 epochs, unfreeze everything and reduce LR
        if epoch == 6:
            print("Unfreezing backbone and reducing LR...")
            for param in model.parameters():
                param.requires_grad = True

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-5,   # lower LR for fine-tuning
                weight_decay=cfg.weight_decay
            )

            if cfg.use_cosine:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cfg.epochs - 5
                )
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg.device)
        metrics = evaluate_patient_level(model, val_loader, cfg.device)

        if scheduler is not None:
            scheduler.step()
            lr_now = scheduler.get_last_lr()[0]
        else:
            lr_now = optimizer.param_groups[0]["lr"]

        val_f1 = metrics["macro_f1"]

        print(
            f"[Fold {fold}] Epoch {epoch:03d} | "
            f"loss={tr_loss:.4f} | val_macroF1={val_f1:.4f} | "
            f"val_balAcc={metrics['balanced_acc']:.4f} | "
            f"patients={metrics['n_patients']} | lr={lr_now:.6e}"
        )

        # Early stopping + checkpointing
        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_macro_f1": best_f1,
                    "cfg": cfg.__dict__,
                },
                ckpt_path
            )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"[Fold {fold}] Early stopping triggered. Best macro-F1={best_f1:.4f}")
                break

    # Load best and return final fold metrics
    best = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(best["model_state"])
    final_metrics = evaluate_patient_level(model, val_loader, cfg.device)
    final_metrics["best_epoch"] = best["epoch"]
    final_metrics["best_macro_f1"] = best["best_macro_f1"]
    final_metrics["ckpt"] = ckpt_path
    return final_metrics


def main():
    seed_everything(cfg.seed)
    df = build_index(cfg.data_root, cfg.groups)

    print("Total scans:", len(df))
    print("Unique patients:", df["patient_id"].nunique())
    print(df.groupby("group").size())

    X = df["path"].values
    y = df["label"].values
    g = df["patient_id"].values

    sgkf = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)

    all_fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, g), start=1):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        # Sanity: no patient overlap
        inter = set(train_df["patient_id"]).intersection(set(val_df["patient_id"]))
        if inter:
            raise RuntimeError(f"Leakage! Patients overlap in fold {fold}: {list(inter)[:5]}")

        m = run_fold(fold, train_df, val_df)
        all_fold_metrics.append(m)

        print(f"\n[Fold {fold}] BEST macro-F1={m['best_macro_f1']:.4f} @ epoch {m['best_epoch']}")
        print(f"[Fold {fold}] Confusion matrix (patient-level):\n{m['confusion_matrix']}\n")

    # Summary
    f1s = [m["best_macro_f1"] for m in all_fold_metrics]
    bals = [m["balanced_acc"] for m in all_fold_metrics]
    print("=================================================")
    print(f"5-fold macro-F1: mean={np.mean(f1s):.4f}  std={np.std(f1s):.4f}")
    print(f"5-fold bal-acc:  mean={np.mean(bals):.4f} std={np.std(bals):.4f}")
    print("Checkpoints saved under:", cfg.out_dir)


if __name__ == "__main__":
    main()