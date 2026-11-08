#!/usr/bin/env python3
"""Supplementary modality ablations, separate from the five fusion architectures."""

import argparse
import os
from pathlib import Path


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["all", "wli", "afi", "clinical", "wli_afi", "multimodal"], default="all")
    p.add_argument("--root", default=".")
    p.add_argument("--bert_path", default="google-bert/bert-base-chinese")
    p.add_argument("--train_csv")
    p.add_argument("--val_csv")
    p.add_argument("--image_root")
    p.add_argument("--gpu", default="0")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    return p


# Parse --help and select the GPU before importing torch.
_ARGS = build_parser().parse_args() if __name__ == "__main__" else build_parser().parse_args([])
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = _ARGS.gpu

import csv, json, time, random, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
from torchvision.transforms import functional as TF, InterpolationMode
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from opmd_data import load_opmd_csv, assert_no_patient_overlap, resolve_image_path

# ============================================================
# OPMD-CLIP202609: reviewer-oriented ablation training
# Modes:
#   wli          = WLI-only
#   afi          = AFI-only
#   clinical     = Clinical-only
#   wli_afi      = WLI + AFI
#   multimodal   = WLI + AFI + Clinical
#
# IMPORTANT:
# - clinical prompt uses PRE-BIOPSY variables only
# - OED grade / binary risk are NEVER included in text input
# - model selection is PATIENT-LEVEL macro-F1
# - patient-balanced + inverse-sqrt class sampling
# - paired WLI/AFI geometric augmentation
# ============================================================

ROOT = Path(_ARGS.root)
BERT_PATH = _ARGS.bert_path

TRAIN_CSV = Path(_ARGS.train_csv) if _ARGS.train_csv else ROOT / "data/processed/csv/split/train.csv"
VAL_CSV = Path(_ARGS.val_csv) if _ARGS.val_csv else ROOT / "data/processed/csv/split/val.csv"

COL_ID = "id"
COL_WLI = "whitelight_image"
COL_AFI = "fluorescent_image"
COL_SITE = "病损部位"
COL_DX = "疾病诊断"
COL_Y5 = "上皮异常增生程度（0：无；1：轻度；2：中度；3：重度；4：癌变）"
COL_Y2 = "上皮异常增生程度（0：低级别；1：高级别）"

MODES = ["wli", "afi", "clinical", "wli_afi", "multimodal"]

SEED = 20260917
IMG_SIZE = 224
BATCH_SIZE = _ARGS.batch
EPOCHS = _ARGS.epochs
PATIENCE = 10
NUM_WORKERS = _ARGS.workers
EMBED_DIM = 384

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Optimized multimodal losses
BINARY_LAMBDA = 0.35
ORDINAL_LAMBDA = 0.20
HIER_LAMBDA = 0.10
CLIP_LAMBDA = 0.015
CLIP_START_EPOCH = 5


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def clean_text(x):
    if pd.isna(x):
        return "未知"
    x = str(x).strip()
    return x if x else "未知"


def check_paths():
    print("=" * 80)
    print("ROOT       :", ROOT)
    print("TRAIN_CSV  :", TRAIN_CSV)
    print("VAL_CSV    :", VAL_CSV)
    print("BERT_PATH  :", BERT_PATH)
    print("=" * 80)
    for p, name in [(ROOT, "ROOT"), (TRAIN_CSV, "TRAIN_CSV"),
                    (VAL_CSV, "VAL_CSV")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")


def validate_dataframe(df, name):
    required = [COL_ID, COL_WLI, COL_AFI, COL_SITE, COL_DX, COL_Y5, COL_Y2]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{name}: missing columns: {missing}")

    # Prevent patient leakage across train/val later.
    bad5 = df.groupby(COL_ID)[COL_Y5].nunique()
    bad2 = df.groupby(COL_ID)[COL_Y2].nunique()
    if (bad5 > 1).any():
        print(f"WARNING: {name} has patients with inconsistent 5-class labels:",
              bad5[bad5 > 1].index.tolist()[:20])
    if (bad2 > 1).any():
        print(f"WARNING: {name} has patients with inconsistent binary labels:",
              bad2[bad2 > 1].index.tolist()[:20])


def print_dataset_summary(train_df, val_df):
    tr_ids = set(train_df[COL_ID].astype(str))
    va_ids = set(val_df[COL_ID].astype(str))
    overlap = tr_ids & va_ids
    if overlap:
        raise RuntimeError(
            f"PATIENT LEAKAGE: {len(overlap)} IDs occur in both train and val. "
            f"Examples: {list(overlap)[:10]}"
        )

    for name, df in [("TRAIN", train_df), ("VAL", val_df)]:
        print("\n" + "=" * 80)
        print(name)
        print("rows / image pairs :", len(df))
        print("patients           :", df[COL_ID].astype(str).nunique())
        pairs_per_patient = df.groupby(COL_ID).size()
        print("pairs/patient mean :", round(float(pairs_per_patient.mean()), 3))
        print("pairs/patient median:", round(float(pairs_per_patient.median()), 3))
        print("5-class rows:")
        print(df[COL_Y5].value_counts().sort_index().to_dict())
        print("2-class rows:")
        print(df[COL_Y2].value_counts().sort_index().to_dict())

        # Patient-level distribution (first label after consistency check).
        pdf = df.groupby(COL_ID).first().reset_index()
        print("5-class patients:")
        print(pdf[COL_Y5].value_counts().sort_index().to_dict())
        print("2-class patients:")
        print(pdf[COL_Y2].value_counts().sort_index().to_dict())


class PairedTrainTransform:
    def __init__(self, size=IMG_SIZE):
        self.size = size

    def __call__(self, wli, afi):
        wli = TF.resize(wli, [self.size, self.size],
                        interpolation=InterpolationMode.BILINEAR, antialias=True)
        afi = TF.resize(afi, [self.size, self.size],
                        interpolation=InterpolationMode.BILINEAR, antialias=True)

        # SAME geometry for both modalities.
        if random.random() < 0.5:
            wli, afi = TF.hflip(wli), TF.hflip(afi)

        angle = random.uniform(-7.0, 7.0)
        wli = TF.rotate(wli, angle, interpolation=InterpolationMode.BILINEAR)
        afi = TF.rotate(afi, angle, interpolation=InterpolationMode.BILINEAR)

        # Weak WLI photometric augmentation only.
        if random.random() < 0.8:
            brightness = random.uniform(0.94, 1.06)
            contrast = random.uniform(0.94, 1.06)
            saturation = random.uniform(0.97, 1.03)
            wli = TF.adjust_brightness(wli, brightness)
            wli = TF.adjust_contrast(wli, contrast)
            wli = TF.adjust_saturation(wli, saturation)

        wli = TF.normalize(TF.to_tensor(wli), IMAGENET_MEAN, IMAGENET_STD)
        afi = TF.normalize(TF.to_tensor(afi), IMAGENET_MEAN, IMAGENET_STD)
        return wli, afi


class PairedEvalTransform:
    def __init__(self, size=IMG_SIZE):
        self.size = size

    def __call__(self, wli, afi):
        wli = TF.resize(wli, [self.size, self.size],
                        interpolation=InterpolationMode.BILINEAR, antialias=True)
        afi = TF.resize(afi, [self.size, self.size],
                        interpolation=InterpolationMode.BILINEAR, antialias=True)
        wli = TF.normalize(TF.to_tensor(wli), IMAGENET_MEAN, IMAGENET_STD)
        afi = TF.normalize(TF.to_tensor(afi), IMAGENET_MEAN, IMAGENET_STD)
        return wli, afi


class OPMDDataset(Dataset):
    def __init__(self, csv_path, bert_path, train=False):
        self.df = load_opmd_csv(csv_path).reset_index(drop=True)
        self.csv_path = Path(csv_path)
        validate_dataframe(self.df, str(csv_path))
        self.tokenizer = BertTokenizer.from_pretrained(str(bert_path))
        self.transform = PairedTrainTransform() if train else PairedEvalTransform()

    def __len__(self):
        return len(self.df)

    def build_prompt(self, row):
        # PRE-BIOPSY clinical information only.
        # Do NOT add OED grade, pathology, binary risk, or post-biopsy information.
        return (
            f"病损部位：{clean_text(row[COL_SITE])}。"
            f"临床诊断：{clean_text(row[COL_DX])}。"
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wli_raw = Image.open(resolve_image_path(row[COL_WLI], self.csv_path, _ARGS.image_root)).convert("RGB")
        afi_raw = Image.open(resolve_image_path(row[COL_AFI], self.csv_path, _ARGS.image_root)).convert("RGB")

        # Online pixel-distance WLI/AFI rotation matching was intentionally removed.
        # If orientation correction is necessary, do it once as audited preprocessing.
        wli, afi = self.transform(wli_raw, afi_raw)

        enc = self.tokenizer(
            self.build_prompt(row),
            padding="max_length",
            truncation=True,
            max_length=48,
            return_tensors="pt"
        )

        return (
            wli, afi,
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            torch.tensor(int(row[COL_Y5]), dtype=torch.long),
            torch.tensor(int(row[COL_Y2]), dtype=torch.long),
            str(row[COL_ID])
        )


def make_patient_balanced_sampler(df):
    patient_counts = df[COL_ID].astype(str).value_counts().to_dict()
    class_counts = df[COL_Y5].astype(int).value_counts().to_dict()
    weights = []

    for _, row in df.iterrows():
        pid = str(row[COL_ID])
        y = int(row[COL_Y5])
        # Equalize patient contribution + moderately emphasize rare classes.
        w_patient = 1.0 / patient_counts[pid]
        w_class = 1.0 / np.sqrt(max(class_counts[y], 1))
        weights.append(w_patient * w_class)

    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.mean()

    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=len(df),
        replacement=True
    )


class VisualEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        d = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.proj = nn.Sequential(
            nn.Linear(d, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.10)
        )

    def forward(self, x):
        return self.proj(self.backbone(x))


class TextEncoder(nn.Module):
    def __init__(self, bert_path, embed_dim=EMBED_DIM):
        super().__init__()
        self.bert = BertModel.from_pretrained(str(bert_path))
        for p in self.bert.parameters():
            p.requires_grad = False
        self.proj = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.10)
        )

    def forward(self, ids, mask):
        # BERT frozen; projection layer remains trainable.
        with torch.no_grad():
            out = self.bert(input_ids=ids, attention_mask=mask)
        return self.proj(out.last_hidden_state[:, 0])


class FlexibleOPMDModel(nn.Module):
    def __init__(self, bert_path, mode="multimodal", embed_dim=EMBED_DIM):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode

        self.wli = VisualEncoder(embed_dim) if mode in ["wli", "wli_afi", "multimodal"] else None
        self.afi = VisualEncoder(embed_dim) if mode in ["afi", "wli_afi", "multimodal"] else None
        self.txt = TextEncoder(bert_path, embed_dim) if mode in ["clinical", "multimodal"] else None

        self.n_modalities = {
            "wli": 1, "afi": 1, "clinical": 1,
            "wli_afi": 2, "multimodal": 3
        }[mode]

        in_dim = embed_dim * self.n_modalities

        # Fair ablation: same classifier/fusion family; input width changes only.
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        self.cls5 = nn.Linear(embed_dim, 5)
        self.cls2 = nn.Linear(embed_dim, 1)
        self.ordinal = nn.Linear(embed_dim, 4)

    def forward(self, wli, afi, ids, mask):
        feats = {}
        if self.wli is not None:
            feats["wli"] = self.wli(wli)
        if self.afi is not None:
            feats["afi"] = self.afi(afi)
        if self.txt is not None:
            feats["clinical"] = self.txt(ids, mask)

        order = {
            "wli": ["wli"],
            "afi": ["afi"],
            "clinical": ["clinical"],
            "wli_afi": ["wli", "afi"],
            "multimodal": ["wli", "afi", "clinical"]
        }[self.mode]

        h = self.fusion(torch.cat([feats[k] for k in order], dim=1))
        log5 = self.cls5(h)
        log2 = self.cls2(h).squeeze(1)
        ordlog = self.ordinal(h)
        return feats, h, log5, log2, ordlog


def ordinal_targets(y5):
    thresholds = torch.arange(4, device=y5.device).unsqueeze(0)
    return (y5.unsqueeze(1) > thresholds).float()


def symmetric_clip(a, b, temperature=0.10):
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    logits = a @ b.T / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) +
        F.cross_entropy(logits.T, labels)
    )


def compute_loss(mode, epoch, feats, log5, log2, ordlog, y5, y2):
    loss5 = F.cross_entropy(log5, y5, label_smoothing=0.03)
    loss2 = F.binary_cross_entropy_with_logits(log2, y2.float())
    loss_ord = F.binary_cross_entropy_with_logits(ordlog, ordinal_targets(y5))

    p5 = F.softmax(log5, dim=1)
    p_high5 = p5[:, 2:].sum(1)
    p_high2 = torch.sigmoid(log2)
    loss_hier = F.mse_loss(p_high2, p_high5)

    loss_clip = torch.zeros((), device=log5.device)
    if mode == "multimodal" and epoch >= CLIP_START_EPOCH:
        # Weak CLIP only makes sense when clinical text and images coexist.
        loss_clip = 0.5 * (
            symmetric_clip(feats["wli"], feats["clinical"]) +
            symmetric_clip(feats["afi"], feats["clinical"])
        )

    total = (
        loss5 +
        BINARY_LAMBDA * loss2 +
        ORDINAL_LAMBDA * loss_ord +
        HIER_LAMBDA * loss_hier +
        CLIP_LAMBDA * loss_clip
    )
    return total


def macro_f1(y_true, y_pred, n_classes=5):
    vals = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        vals.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(vals))


def balanced_accuracy(y_true, y_pred, n_classes=5):
    recalls = []
    for c in range(n_classes):
        m = y_true == c
        if m.sum():
            recalls.append(float((y_pred[m] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def class_recalls(y_true, y_pred):
    out = {}
    for c in range(5):
        m = y_true == c
        out[f"recall_{c}"] = float((y_pred[m] == c).mean()) if m.sum() else np.nan
    return out


def aggregate_patient(ids, y5, prob5, y2, prob2):
    bucket = defaultdict(lambda: {"y5": [], "p5": [], "y2": [], "p2": []})

    for pid, a, b, c, d in zip(ids, y5, prob5, y2, prob2):
        z = bucket[str(pid)]
        z["y5"].append(int(a))
        z["p5"].append(np.asarray(b))
        z["y2"].append(int(c))
        z["p2"].append(float(d))

    py5, pp5, py2, pp2 = [], [], [], []
    for pid, z in bucket.items():
        if len(set(z["y5"])) != 1 or len(set(z["y2"])) != 1:
            raise ValueError(f"Inconsistent patient labels: {pid}")
        py5.append(z["y5"][0])
        py2.append(z["y2"][0])

        # Main analysis = mean probability.
        # Max-risk should be evaluated separately as reviewer-requested sensitivity analysis.
        pp5.append(np.mean(np.stack(z["p5"]), axis=0))
        pp2.append(np.mean(z["p2"]))

    return np.asarray(py5), np.stack(pp5), np.asarray(py2), np.asarray(pp2)


def metric_dict(y5, p5, y2, p2):
    pred5 = p5.argmax(1)
    pred2 = (p2 >= 0.5).astype(int)
    out = {
        "acc5": float((pred5 == y5).mean()),
        "bal_acc5": balanced_accuracy(y5, pred5),
        "macro_f1_5": macro_f1(y5, pred5),
        "acc2": float((pred2 == y2).mean())
    }
    out.update(class_recalls(y5, pred5))
    return out


@torch.no_grad()
def evaluate(model, loader, device, mode, epoch):
    model.eval()
    ids_all, y5_all, p5_all, y2_all, p2_all = [], [], [], [], []
    losses = []

    for wli, afi, ids, mask, y5, y2, pids in loader:
        wli, afi = wli.to(device), afi.to(device)
        ids, mask = ids.to(device), mask.to(device)
        y5, y2 = y5.to(device), y2.to(device)

        feats, h, log5, log2, ordlog = model(wli, afi, ids, mask)
        loss = compute_loss(mode, epoch, feats, log5, log2, ordlog, y5, y2)
        losses.append(loss.item())

        ids_all.extend(list(pids))
        y5_all.extend(y5.cpu().numpy().tolist())
        p5_all.extend(F.softmax(log5, 1).cpu().numpy().tolist())
        y2_all.extend(y2.cpu().numpy().tolist())
        p2_all.extend(torch.sigmoid(log2).cpu().numpy().tolist())

    y5 = np.asarray(y5_all)
    p5 = np.asarray(p5_all)
    y2 = np.asarray(y2_all)
    p2 = np.asarray(p2_all)

    pair_metrics = metric_dict(y5, p5, y2, p2)

    py5, pp5, py2, pp2 = aggregate_patient(ids_all, y5, p5, y2, p2)
    patient_metrics = metric_dict(py5, pp5, py2, pp2)
    patient_metrics["n_patients"] = len(py5)

    return float(np.mean(losses)), pair_metrics, patient_metrics


def parameter_groups(model):
    backbone = []
    new = []

    if model.wli is not None:
        backbone += list(model.wli.backbone.parameters())
        new += list(model.wli.proj.parameters())
    if model.afi is not None:
        backbone += list(model.afi.backbone.parameters())
        new += list(model.afi.proj.parameters())
    if model.txt is not None:
        # BERT is frozen; only projection goes here.
        new += list(model.txt.proj.parameters())

    new += (
        list(model.fusion.parameters()) +
        list(model.cls5.parameters()) +
        list(model.cls2.parameters()) +
        list(model.ordinal.parameters())
    )

    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": 2e-5, "name": "backbone"})
    if new:
        groups.append({"params": new, "lr": 1e-4, "name": "new"})
    return groups


def save_ckpt(path, model, optimizer, scheduler, epoch, mode, pair, patient):
    torch.save({
        "epoch": epoch,
        "mode": mode,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "pair_metrics": pair,
        "patient_metrics": patient,
        "config": {
            "root": str(ROOT),
            "bert_path": str(BERT_PATH),
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "seed": SEED,
            "binary_lambda": BINARY_LAMBDA,
            "ordinal_lambda": ORDINAL_LAMBDA,
            "hier_lambda": HIER_LAMBDA,
            "clip_lambda": CLIP_LAMBDA,
            "clip_start_epoch": CLIP_START_EPOCH,
            "aggregation": "mean_probability",
            "selection": "patient_macro_f1"
        }
    }, path)


def train_one_mode(mode, train_ds, val_ds, device):
    print("\n" + "#" * 90)
    print(f"START MODE: {mode}")
    print("#" * 90)

    # Reset seed so each ablation begins reproducibly.
    seed_everything(SEED)

    exp_dir = ROOT / "experiments" / "reviewer_ablation_202609" / mode
    model_dir = ROOT / "models" / "reviewer_ablation_202609" / mode
    output_dir = ROOT / "outputs" / "reviewer_ablation_202609" / mode
    figure_dir = ROOT / "figures" / "reviewer_ablation_202609" / mode
    for p in [exp_dir, model_dir, output_dir, figure_dir]:
        p.mkdir(parents=True, exist_ok=True)

    sampler = make_patient_balanced_sampler(train_ds.df)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=NUM_WORKERS > 0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=NUM_WORKERS > 0
    )

    model = FlexibleOPMDModel(BERT_PATH, mode=mode).to(device)
    optimizer = torch.optim.AdamW(
        parameter_groups(model),
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=5e-7
    )

    log_csv = exp_dir / "train_log.csv"
    fields = [
        "epoch", "train_loss", "val_loss",
        "pair_acc5", "pair_bal_acc5", "pair_macro_f1_5", "pair_acc2",
        "patient_acc5", "patient_bal_acc5", "patient_macro_f1_5", "patient_acc2",
        "patient_recall_none", "patient_recall_mild", "patient_recall_moderate",
        "patient_recall_severe", "patient_recall_cancer"
    ]
    with open(log_csv, "w", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_f1 = -1.0
    best_acc = -1.0
    bad_epochs = 0
    best_record = None

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []

        for wli, afi, ids, mask, y5, y2, _ in tqdm(
            train_loader, desc=f"{mode} | epoch {epoch:02d}"
        ):
            wli, afi = wli.to(device), afi.to(device)
            ids, mask = ids.to(device), mask.to(device)
            y5, y2 = y5.to(device), y2.to(device)

            optimizer.zero_grad(set_to_none=True)
            feats, h, log5, log2, ordlog = model(wli, afi, ids, mask)
            loss = compute_loss(mode, epoch, feats, log5, log2, ordlog, y5, y2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(loss.item())

        val_loss, pair, patient = evaluate(
            model, val_loader, device, mode, epoch
        )
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "pair_acc5": pair["acc5"],
            "pair_bal_acc5": pair["bal_acc5"],
            "pair_macro_f1_5": pair["macro_f1_5"],
            "pair_acc2": pair["acc2"],
            "patient_acc5": patient["acc5"],
            "patient_bal_acc5": patient["bal_acc5"],
            "patient_macro_f1_5": patient["macro_f1_5"],
            "patient_acc2": patient["acc2"],
            "patient_recall_none": patient["recall_0"],
            "patient_recall_mild": patient["recall_1"],
            "patient_recall_moderate": patient["recall_2"],
            "patient_recall_severe": patient["recall_3"],
            "patient_recall_cancer": patient["recall_4"]
        }
        with open(log_csv, "a", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(record)

        print(
            f"[{mode}] epoch={epoch:02d} "
            f"patient ACC5={patient['acc5']:.4f} "
            f"BalAcc={patient['bal_acc5']:.4f} "
            f"MacroF1={patient['macro_f1_5']:.4f} "
            f"ACC2={patient['acc2']:.4f} "
            f"SevereRecall={patient['recall_3']:.4f}"
        )

        score = patient["macro_f1_5"]
        improved = (
            score > best_f1 + 1e-6 or
            (abs(score - best_f1) <= 1e-6 and patient["acc5"] > best_acc)
        )

        if improved:
            best_f1 = score
            best_acc = patient["acc5"]
            bad_epochs = 0
            best_record = record.copy()
            save_ckpt(
                model_dir / "best_patient_macro_f1.pt",
                model, optimizer, scheduler, epoch, mode, pair, patient
            )
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"[{mode}] early stop at epoch {epoch}")
                break

    save_ckpt(
        model_dir / "last.pt",
        model, optimizer, scheduler, epoch, mode, pair, patient
    )

    with open(output_dir / "best_validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(best_record, f, indent=2, ensure_ascii=False)

    return best_record


def main():
    args = _ARGS
    if min(EPOCHS, BATCH_SIZE) <= 0 or NUM_WORKERS < 0:
        raise ValueError("epochs/batch must be positive and workers nonnegative")
    assert_no_patient_overlap(TRAIN_CSV, VAL_CSV)
    for parent in ["experiments", "models", "outputs", "figures"]:
        target = ROOT / parent / "reviewer_ablation_202609"
        if target.exists() and any(target.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Existing ablation output: {target}. Use a new --root or --overwrite.")

    check_paths()
    seed_everything()

    train_df = load_opmd_csv(TRAIN_CSV)
    val_df = load_opmd_csv(VAL_CSV)
    validate_dataframe(train_df, "TRAIN")
    validate_dataframe(val_df, "VAL")
    print_dataset_summary(train_df, val_df)

    train_ds = OPMDDataset(TRAIN_CSV, BERT_PATH, train=True)
    val_ds = OPMDDataset(VAL_CSV, BERT_PATH, train=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDEVICE:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    modes = MODES if args.mode == "all" else [args.mode]
    results = {}

    for mode in modes:
        results[mode] = train_one_mode(mode, train_ds, val_ds, device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_dir = ROOT / "outputs" / "reviewer_ablation_202609"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode, r in results.items():
        row = {"mode": mode}
        row.update(r)
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(
        summary_dir / "ablation_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    print("\n" + "=" * 90)
    print("ABLATION FINISHED")
    print(summary[
        ["mode", "patient_acc5", "patient_bal_acc5",
         "patient_macro_f1_5", "patient_acc2",
         "patient_recall_severe"]
    ].to_string(index=False))
    print("\nSummary saved to:")
    print(summary_dir / "ablation_summary.csv")


if __name__ == "__main__":
    main()
