#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import random
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
from torchvision.transforms import functional as TF
from transformers import BertTokenizer, BertModel, BertConfig

from opmd_cli import ROOT_DEFAULT, BERT_DEFAULT, build_argparser, validate_training_args
from opmd_data import (
    COL5, COL2, load_opmd_csv, detect_id_col, assert_no_patient_overlap,
    choose_one_fixed_pair, resolve_image_path, clinical_prompt,
)
from opmd_metrics import (
    add_pair_predictions, patient_aggregate, calculate_metrics, save_confusions, write_json,
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class PairedAugment:
    def __init__(self, train=True, size=224):
        self.train = train
        self.size = size

    def __call__(self, wli, afi):
        wli = TF.resize(wli, [self.size, self.size])
        afi = TF.resize(afi, [self.size, self.size])
        if self.train:
            if random.random() < 0.5:
                wli = TF.hflip(wli)
                afi = TF.hflip(afi)
            angle = random.uniform(-5.0, 5.0)
            wli = TF.rotate(wli, angle)
            afi = TF.rotate(afi, angle)
            wli = TF.adjust_brightness(wli, random.uniform(0.92, 1.08))
            wli = TF.adjust_contrast(wli, random.uniform(0.92, 1.08))
            wli = TF.adjust_saturation(wli, random.uniform(0.96, 1.04))
        wli = TF.to_tensor(wli)
        afi = TF.to_tensor(afi)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return TF.normalize(wli, mean, std), TF.normalize(afi, mean, std)


class OPMDDataset(Dataset):
    def __init__(self, csv_path, bert_path, train=False, pair_policy="all", size=224, max_text_len=64, image_root=None):
        self.csv_path = Path(csv_path)
        self.image_root = image_root
        self.df = load_opmd_csv(csv_path)
        self.id_col = detect_id_col(self.df)
        self.df[self.id_col] = self.df[self.id_col].astype(str)
        if pair_policy == "one_fixed":
            self.df = choose_one_fixed_pair(self.df, self.id_col)
        elif pair_policy != "all":
            raise ValueError("pair_policy must be all or one_fixed")
        self.aug = PairedAugment(train=train, size=size)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.df)

    def build_prompt(self, row):
        return clinical_prompt(row)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        wli_path = str(resolve_image_path(r["whitelight_image"], self.csv_path, self.image_root))
        afi_path = str(resolve_image_path(r["fluorescent_image"], self.csv_path, self.image_root))
        with Image.open(wli_path) as im:
            wli = im.convert("RGB")
        with Image.open(afi_path) as im:
            afi = im.convert("RGB")
        wli, afi = self.aug(wli, afi)
        enc = self.tokenizer(
            self.build_prompt(r),
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        return {
            "wli": wli,
            "afi": afi,
            "ids": enc["input_ids"].squeeze(0),
            "mask": enc["attention_mask"].squeeze(0),
            "y5": torch.tensor(int(r[COL5]), dtype=torch.long),
            "y2": torch.tensor(int(r[COL2]), dtype=torch.float32),
            "pid": str(r[self.id_col]),
            "row_index": int(i),
            "wli_path": wli_path,
            "afi_path": afi_path,
        }


def make_patient_balanced_sampler(dataset, seed):
    counts = dataset.df[dataset.id_col].value_counts()
    weights = dataset.df[dataset.id_col].map(lambda x: 1.0 / counts[x]).to_numpy()
    g = torch.Generator()
    g.manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=g,
    )


def patient_level_class_weights(csv_path, pair_policy):
    df = load_opmd_csv(csv_path)
    id_col = detect_id_col(df)
    df[id_col] = df[id_col].astype(str)
    if pair_policy == "one_fixed":
        df = choose_one_fixed_pair(df, id_col)
    patient_df = df.groupby(id_col, sort=False).first().reset_index()
    counts = np.bincount(patient_df[COL5].astype(int), minlength=5).astype(float)
    w = np.sqrt(counts.sum() / (5.0 * np.maximum(counts, 1.0)))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32), counts.astype(int).tolist()


class GlobalVisualEncoder(nn.Module):
    def __init__(self, dim=512, pretrained=True):
        super().__init__()
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        in_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.proj = nn.Sequential(
            nn.Linear(in_dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(0.10)
        )

    def forward(self, x):
        return self.proj(self.backbone(x))


class GlobalTextEncoder(nn.Module):
    def __init__(self, bert_path, dim=512, pretrained=True):
        super().__init__()
        self.bert = (BertModel.from_pretrained(bert_path) if pretrained
                     else BertModel(BertConfig.from_pretrained(bert_path)))
        for p in self.bert.parameters():
            p.requires_grad = False
        self.proj = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, dim),
            nn.LayerNorm(dim), nn.GELU(), nn.Dropout(0.10)
        )

    def forward(self, ids, mask):
        h = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.proj(h)


class SpatialVisualEncoder(nn.Module):
    def __init__(self, dim=512, max_tokens=49, pretrained=True):
        super().__init__()
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.features = nn.Sequential(*list(net.children())[:-2])
        self.proj = nn.Linear(2048, dim)
        self.norm = nn.LayerNorm(dim)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        f = self.features(x)
        f = f.flatten(2).transpose(1, 2)
        z = self.norm(self.proj(f))
        if z.size(1) > self.pos.size(1):
            raise RuntimeError("Visual token count exceeds positional table. Keep --size 224.")
        return z + self.pos[:, :z.size(1)]


class TokenTextEncoder(nn.Module):
    def __init__(self, bert_path, dim=512, pretrained=True):
        super().__init__()
        self.bert = (BertModel.from_pretrained(bert_path) if pretrained
                     else BertModel(BertConfig.from_pretrained(bert_path)))
        for p in self.bert.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(self.bert.config.hidden_size, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, ids, mask):
        h = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.norm(self.proj(h))


def masked_mean(x, mask):
    m = mask.float().unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp_min(1.0)


class ResidualCrossAttention(nn.Module):
    def __init__(self, dim=512, heads=8, dropout=0.10):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q, kv, key_padding_mask=None):
        h, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        q = self.norm1(q + h)
        return self.norm2(q + self.ffn(q))


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.w_from_a = ResidualCrossAttention(dim, heads)
        self.a_from_w = ResidualCrossAttention(dim, heads)
        self.v_from_t = ResidualCrossAttention(dim, heads)
        self.t_from_v = ResidualCrossAttention(dim, heads)
        self.fuse = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(0.25)
        )

    def forward(self, w, a, t, text_mask):
        w2 = self.w_from_a(w, a)
        a2 = self.a_from_w(a, w)
        visual = torch.cat([w2, a2], dim=1)
        text_pad = ~text_mask.bool()
        visual2 = self.v_from_t(visual, t, key_padding_mask=text_pad)
        text2 = self.t_from_v(t, visual)
        v_pool = visual2.mean(1)
        t_pool = masked_mean(text2, text_mask)
        fused = self.fuse(torch.cat([v_pool, t_pool], dim=1))
        fw = w2.mean(1)
        fa = a2.mean(1)
        ft = masked_mean(t, text_mask)
        return fused, fw, fa, ft


class BaselineConcatModel(nn.Module):
    def __init__(self, bert_path, dim=512, pretrained=True):
        super().__init__()
        self.wli = GlobalVisualEncoder(dim, pretrained=pretrained)
        self.afi = GlobalVisualEncoder(dim, pretrained=pretrained)
        self.txt = GlobalTextEncoder(bert_path, dim, pretrained=pretrained)
        self.cls5 = nn.Linear(dim * 3, 5)
        self.cls2 = nn.Linear(dim * 3, 1)

    def forward(self, b):
        fw = self.wli(b["wli"])
        fa = self.afi(b["afi"])
        ft = self.txt(b["ids"], b["mask"])
        h = torch.cat([fw, fa, ft], dim=1)
        return {"log5": self.cls5(h), "log2": self.cls2(h).squeeze(1), "fw": fw, "fa": fa, "ft": ft}


class MLPFusionModel(nn.Module):
    def __init__(self, bert_path, dim=512, pretrained=True):
        super().__init__()
        self.wli = GlobalVisualEncoder(dim, pretrained=pretrained)
        self.afi = GlobalVisualEncoder(dim, pretrained=pretrained)
        self.txt = GlobalTextEncoder(bert_path, dim, pretrained=pretrained)
        self.fusion = nn.Sequential(
            nn.Linear(dim * 3, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.20)
        )
        self.cls5 = nn.Linear(256, 5)
        self.cls2 = nn.Linear(256, 1)

    def forward(self, b):
        fw = self.wli(b["wli"])
        fa = self.afi(b["afi"])
        ft = self.txt(b["ids"], b["mask"])
        h = self.fusion(torch.cat([fw, fa, ft], dim=1))
        return {"log5": self.cls5(h), "log2": self.cls2(h).squeeze(1), "fw": fw, "fa": fa, "ft": ft}


class CrossAttentionModel(nn.Module):
    def __init__(self, bert_path, dim=512, heads=8, with_ordinal=False, pretrained=True):
        super().__init__()
        self.with_ordinal = with_ordinal
        self.wli = SpatialVisualEncoder(dim, pretrained=pretrained)
        self.afi = SpatialVisualEncoder(dim, pretrained=pretrained)
        self.txt = TokenTextEncoder(bert_path, dim, pretrained=pretrained)
        self.cross = CrossAttentionFusion(dim, heads)
        self.cls5 = nn.Linear(dim, 5)
        self.cls2 = nn.Linear(dim, 1)
        if with_ordinal:
            self.ordinal = nn.Linear(dim, 4)

    def forward(self, b):
        w = self.wli(b["wli"])
        a = self.afi(b["afi"])
        t = self.txt(b["ids"], b["mask"])
        h, fw, fa, ft = self.cross(w, a, t, b["mask"])
        out = {"log5": self.cls5(h), "log2": self.cls2(h).squeeze(1), "fw": fw, "fa": fa, "ft": ft}
        if self.with_ordinal:
            out["log_ord"] = self.ordinal(h)
        return out


def build_model(model_kind, args, pretrained=True):
    if model_kind == "baseline_concat":
        return BaselineConcatModel(args.bert_path, args.dim, pretrained=pretrained)
    if model_kind == "mlp_fusion":
        return MLPFusionModel(args.bert_path, args.dim, pretrained=pretrained)
    if model_kind == "cross_attention":
        return CrossAttentionModel(args.bert_path, args.dim, args.heads, with_ordinal=False, pretrained=pretrained)
    if model_kind in ["cross_attention_ordinal", "full"]:
        return CrossAttentionModel(args.bert_path, args.dim, args.heads, with_ordinal=True, pretrained=pretrained)
    raise ValueError(model_kind)


def ordinal_targets(y5):
    thresholds = torch.arange(4, device=y5.device).unsqueeze(0)
    return (y5.unsqueeze(1) > thresholds).float()


def ordinal_and_monotonic_loss(log_ord, y5):
    target = ordinal_targets(y5)
    l_ord = F.binary_cross_entropy_with_logits(log_ord, target)
    p = torch.sigmoid(log_ord)
    l_mono = F.relu(p[:, 1:] - p[:, :-1]).mean()
    return l_ord, l_mono


def patient_aware_contrastive(a, b, pids, temperature=0.07):
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    logits = a @ b.T / temperature
    pids = [str(x) for x in pids]
    pos = torch.tensor([[pi == pj for pj in pids] for pi in pids], device=logits.device, dtype=torch.bool)
    neg_inf = torch.finfo(logits.dtype).min

    def one_direction(x, mask):
        num = torch.logsumexp(x.masked_fill(~mask, neg_inf), dim=1)
        den = torch.logsumexp(x, dim=1)
        return -(num - den).mean()

    return 0.5 * (one_direction(logits, pos) + one_direction(logits.T, pos.T))


def compute_loss(model_kind, out, y5, y2, w5, epoch, pids, args):
    l5 = F.cross_entropy(out["log5"], y5, weight=w5, label_smoothing=args.label_smoothing)
    l2 = F.binary_cross_entropy_with_logits(out["log2"], y2)
    l_ord = torch.zeros((), device=y5.device)
    l_mono = torch.zeros((), device=y5.device)
    l_hier = torch.zeros((), device=y5.device)
    l_con = torch.zeros((), device=y5.device)
    total = l5 + args.binary_lambda * l2

    if model_kind in ["cross_attention_ordinal", "full"]:
        l_ord, l_mono = ordinal_and_monotonic_loss(out["log_ord"], y5)
        p5 = F.softmax(out["log5"], dim=1)
        p_high5 = p5[:, 2:].sum(1)
        p_high2 = torch.sigmoid(out["log2"])
        l_hier = F.mse_loss(p_high2, p_high5)
        total = total + args.ordinal_lambda * l_ord + args.monotonic_lambda * l_mono + args.hier_lambda * l_hier

    # Full model: contrastive alignment is active from the beginning,
    # but its weight is linearly warmed up to avoid destabilizing early classification.
    contrastive_lambda_effective = 0.0
    if model_kind == "full" and epoch >= args.contrastive_start:
        warmup_epochs = max(int(args.contrastive_warmup_epochs), 1)
        warmup_step = epoch - int(args.contrastive_start) + 1
        warmup_ratio = min(1.0, warmup_step / float(warmup_epochs))
        contrastive_lambda_effective = float(args.contrastive_lambda) * warmup_ratio

        l_wa = patient_aware_contrastive(out["fw"], out["fa"], pids, args.temperature)
        l_wt = patient_aware_contrastive(out["fw"], out["ft"], pids, args.temperature)
        l_at = patient_aware_contrastive(out["fa"], out["ft"], pids, args.temperature)
        l_con = (l_wa + l_wt + l_at) / 3.0
        total = total + contrastive_lambda_effective * l_con

    return total, {
        "loss": float(total.detach()), "l5": float(l5.detach()), "l2": float(l2.detach()),
        "ordinal": float(l_ord.detach()), "monotonic": float(l_mono.detach()),
        "hier": float(l_hier.detach()), "contrastive": float(l_con.detach()),
        "contrastive_lambda_effective": float(contrastive_lambda_effective),
    }


def move_tensor_batch(b, device):
    for k in ["wli", "afi", "ids", "mask", "y5", "y2"]:
        b[k] = b[k].to(device, non_blocking=True)
    return b


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    rows = []
    for b in loader:
        pids = list(b["pid"])
        row_idx = b["row_index"].numpy().tolist()
        wli_paths = list(b["wli_path"])
        afi_paths = list(b["afi_path"])
        b = move_tensor_batch(b, device)
        out = model(b)
        p5 = F.softmax(out["log5"], 1).cpu().numpy()
        p2 = torch.sigmoid(out["log2"]).cpu().numpy()
        ord_p = torch.sigmoid(out["log_ord"]).cpu().numpy() if "log_ord" in out else None
        for i, pid in enumerate(pids):
            row = {
                "pid": str(pid), "row_index": int(row_idx[i]),
                "wli_path": wli_paths[i], "afi_path": afi_paths[i],
                "y5": int(b["y5"][i].item()), "y2": int(b["y2"][i].item()),
                "p5": json.dumps(p5[i].tolist()), "p2": float(p2[i])
            }
            if ord_p is not None:
                row["p_ord"] = json.dumps(ord_p[i].tolist())
            rows.append(row)
    return pd.DataFrame(rows)


def run_experiment(model_kind, default_exp, args=None):
    # IMPORTANT:
    # CUDA_VISIBLE_DEVICES must already have been set by the entry script
    # BEFORE this module imported torch. Do not change it here.
    args = args or build_argparser(default_exp).parse_args()
    validate_training_args(args)
    seed_all(args.seed)

    split = Path(args.root) / "data" / "processed" / "csv" / "split"
    args.train_csv = args.train_csv or str(split / "train.csv")
    args.val_csv = args.val_csv or str(split / "val.csv")
    if args.test_csv is None and (split / "test.csv").exists():
        args.test_csv = str(split / "test.csv")
    assert_no_patient_overlap(args.train_csv, args.val_csv, args.test_csv)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(
            f"[CUDA] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} | "
            f"visible_gpus={torch.cuda.device_count()} | "
            f"logical_device=0 | name={torch.cuda.get_device_name(0)}"
        )
    else:
        device = torch.device("cpu")
        print(
            f"[CUDA] CUDA is not available. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )
    outdir = Path(args.root) / "experiments" / args.exp
    if outdir.exists() and any(outdir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Experiment directory is not empty: {outdir}. Use a new --exp or --overwrite.")
    outdir.mkdir(parents=True, exist_ok=True)

    train_ds = OPMDDataset(args.train_csv, args.bert_path, True, args.pair_policy, args.size, args.max_text_len, args.image_root)
    val_ds = OPMDDataset(args.val_csv, args.bert_path, False, args.pair_policy, args.size, args.max_text_len, args.image_root)

    print(f"[{default_exp}] train={len(train_ds)} pairs/{train_ds.df[train_ds.id_col].nunique()} patients")
    print(f"[{default_exp}] val={len(val_ds)} pairs/{val_ds.df[val_ds.id_col].nunique()} patients")
    if model_kind == "full":
        print(
            f"[{default_exp}] contrastive: start_epoch={args.contrastive_start}, "
            f"warmup_epochs={args.contrastive_warmup_epochs}, "
            f"target_lambda={args.contrastive_lambda}, temperature={args.temperature}"
        )

    if args.no_patient_balanced_sampling:
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    else:
        sampler = make_patient_balanced_sampler(train_ds, args.seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, shuffle=False, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    w5, counts = patient_level_class_weights(args.train_csv, args.pair_policy)
    w5 = w5.to(device)
    model = build_model(model_kind, args).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    config = vars(args).copy()
    config.update({
        "model_kind": model_kind,
        "checkpoint_schema_version": 2,
        "patient_level_class_counts5": counts,
        "class_weights5": w5.detach().cpu().tolist(),
        "primary_patient_aggregation": "mean_prob",
        "secondary_patient_aggregations": ["max_prob", "max_risk"],
        "checkpoint_metric": "val patient mean-probability 5-class macro-F1",
        "clinical_variables": ["病损部位", "疾病诊断"],
        "test_used_for_model_selection": False,
    })
    write_json(outdir / "config.json", config)

    best = -1.0
    best_epoch = -1
    bad = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        meters = defaultdict(float)
        nb = 0
        for batch in train_loader:
            pids = list(batch["pid"])
            batch = move_tensor_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(batch)
                loss, parts = compute_loss(model_kind, out, batch["y5"], batch["y2"], w5, epoch, pids, args)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            nb += 1
            for k, v in parts.items():
                meters[k] += v
        scheduler.step()

        pair_raw = predict(model, val_loader, device)
        pair_eval = add_pair_predictions(pair_raw)
        m_pair = calculate_metrics(pair_eval, "pairs")
        pt_mean = patient_aggregate(pair_raw, "mean_prob")
        pt_max = patient_aggregate(pair_raw, "max_prob")
        pt_risk = patient_aggregate(pair_raw, "max_risk")
        m_mean = calculate_metrics(pt_mean, "patients")
        m_max = calculate_metrics(pt_max, "patients")
        m_risk = calculate_metrics(pt_risk, "patients")

        score = m_mean["macro_f1_5"]
        record = {
            "epoch": epoch, "seconds": time.time() - t0, "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v / max(nb, 1) for k, v in meters.items()},
            **{f"pair_{k}": v for k, v in m_pair.items()},
            **{f"patient_mean_{k}": v for k, v in m_mean.items()},
            **{f"patient_maxprob_{k}": v for k, v in m_max.items()},
            **{f"patient_maxrisk_{k}": v for k, v in m_risk.items()},
        }
        history.append(record)
        pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)

        print(
            f"[{default_exp}] ep={epoch:03d} "
            f"PAIR MacroF1={m_pair['macro_f1_5']:.4f} Acc5={m_pair['acc5']:.4f} AUC2={m_pair['auc2']:.4f} | "
            f"PATIENT-MEAN MacroF1={m_mean['macro_f1_5']:.4f} Acc5={m_mean['acc5']:.4f} AUC2={m_mean['auc2']:.4f} | "
            f"SevereRecall={m_pair['recall_grade_3']:.4f}/{m_mean['recall_grade_3']:.4f}"
        )

        if score > best + args.min_delta:
            best = score
            best_epoch = epoch
            bad = 0
            torch.save({
                "model": model.state_dict(), "model_kind": model_kind, "epoch": epoch,
                "image_pair_metrics": m_pair, "patient_mean_metrics": m_mean,
                "patient_maxprob_metrics": m_max, "patient_maxrisk_metrics": m_risk,
                "config": config,
            }, outdir / "best_patient_mean_macro_f1.pt")
            pair_eval.to_csv(outdir / "val_image_pair_predictions.csv", index=False)
            pt_mean.to_csv(outdir / "val_patient_mean_probability.csv", index=False)
            pt_max.to_csv(outdir / "val_patient_max_probability.csv", index=False)
            pt_risk.to_csv(outdir / "val_patient_max_risk.csv", index=False)
            save_confusions(pair_eval, outdir, "val_image_pair")
            save_confusions(pt_mean, outdir, "val_patient_mean")
            save_confusions(pt_max, outdir, "val_patient_maxprob")
            save_confusions(pt_risk, outdir, "val_patient_maxrisk")
            write_json(outdir / "best_val_metrics.json", {
                "image_pair": m_pair,
                "patient_mean_probability": m_mean,
                "patient_max_probability": m_max,
                "patient_max_risk": m_risk,
            })
        else:
            bad += 1

        if epoch + 1 >= args.min_epochs and bad >= args.patience:
            print(f"Early stop at epoch {epoch}; best epoch={best_epoch}, score={best:.4f}")
            break

    write_json(outdir / "training_summary.json", {
        "model_kind": model_kind,
        "best_epoch": best_epoch,
        "best_val_patient_mean_macro_f1": best,
        "output_directory": str(outdir),
    })
