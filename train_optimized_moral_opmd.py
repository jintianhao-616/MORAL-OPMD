# ============================================================
# Optimized (GITHUB / PAPER VERSION)
# Reproducible version (NO hard-coded paths)
# ============================================================

import os
import argparse
import json
import csv
import datetime
import time
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================
# 0. Argument parser (⭐ KEY FOR GITHUB)
# ============================================================
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--bert_path", type=str, required=True)

    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)

    return parser.parse_args()


# ============================================================
# 1. Dataset
# ============================================================
class OPMDCLIPDataset(Dataset):
    def __init__(self, csv_path, transform_wli, transform_afi, bert_path):
        self.df = pd.read_csv(csv_path)
        self.transform_wli = transform_wli
        self.transform_afi = transform_afi
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

    def auto_align_afi(self, afi_img, wli_img):
        resize = transforms.Resize((224,224))
        wli_t = transforms.ToTensor()(resize(wli_img))

        best_img, min_dist = afi_img, 1e9
        for angle in [0, 90, 180, 270]:
            rot = afi_img.rotate(angle)
            afi_t = transforms.ToTensor()(resize(rot))
            dist = torch.norm(afi_t - wli_t).item()
            if dist < min_dist:
                min_dist = dist
                best_img = rot
        return best_img

    def build_prompt(self, row):
        return (
            f"Lesion location:{row['Lesion location']}。"
            f"Clinical diagnosis:{row['Clinical diagnosis']}。"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        wli = Image.open(row["whitelight_image"]).convert("RGB")
        afi = Image.open(row["fluorescent_image"]).convert("RGB")
        afi = self.auto_align_afi(afi, wli)

        wli = self.transform_wli(wli)
        afi = self.transform_afi(afi)

        enc = self.tokenizer(
            self.build_prompt(row),
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        y5 = int(row["label_5cls"])
        y2 = int(row["label_2cls"])

        return (
            wli,
            afi,
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            y5,
            y2
        )


# ============================================================
# 2. Model
# ============================================================
class VisualEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        hidden = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.proj = nn.Linear(hidden, embed_dim)

    def forward(self, x):
        return self.proj(self.backbone(x))


class TextEncoder(nn.Module):
    def __init__(self, bert_path, embed_dim=512):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        for p in self.bert.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(self.bert.config.hidden_size, embed_dim)

    def forward(self, ids, mask):
        out = self.bert(input_ids=ids, attention_mask=mask)
        return self.proj(out.last_hidden_state[:, 0])


class OptimizedV2(nn.Module):
    def __init__(self, bert_path):
        super().__init__()
        self.wli = VisualEncoder()
        self.afi = VisualEncoder()
        self.txt = TextEncoder(bert_path)

        self.fusion = nn.Sequential(
            nn.Linear(512 * 3, 512),
            nn.ReLU(inplace=True)
        )

        self.cls5 = nn.Linear(512, 5)
        self.cls2 = nn.Linear(512, 1)

    def forward(self, wli, afi, ids, mask):
        fw = self.wli(wli)
        fa = self.afi(afi)
        ft = self.txt(ids, mask)

        fused = self.fusion(torch.cat([fw, fa, ft], dim=1))
        return fw, fa, ft, self.cls5(fused), self.cls2(fused).squeeze(1)


# ============================================================
# 3. Loss
# ============================================================
def clip_loss(a, b, t=0.07):
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    sim = a @ b.T / t
    labels = torch.arange(sim.size(0), device=sim.device)
    return (F.cross_entropy(sim, labels) +
            F.cross_entropy(sim.T, labels)) / 2


# ============================================================
# 4. Train / Eval
# ============================================================
def evaluate(model, loader):
    model.eval()
    correct5, correct2, total = 0, 0, 0

    with torch.no_grad():
        for wli, afi, ids, mask, y5, y2 in loader:
            wli, afi = wli.cuda(), afi.cuda()
            ids, mask = ids.cuda(), mask.cuda()
            y5, y2 = y5.cuda(), y2.cuda()

            _, _, _, log5, log2 = model(wli, afi, ids, mask)

            correct5 += (log5.argmax(1) == y5).sum().item()
            correct2 += ((log2 > 0).long() == y2).sum().item()
            total += y5.size(0)

    return correct5 / total, correct2 / total


# ============================================================
# 5. Main
# ============================================================
def main():
    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_eval = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    train_loader = DataLoader(
        OPMDCLIPDataset(args.train_csv, transform_train, transform_train, args.bert_path),
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        OPMDCLIPDataset(args.val_csv, transform_eval, transform_eval, args.bert_path),
        batch_size=args.batch_size,
        shuffle=False
    )

    model = OptimizedV2(args.bert_path).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()

        for wli, afi, ids, mask, y5, y2 in train_loader:
            wli, afi = wli.cuda(), afi.cuda()
            ids, mask = ids.cuda(), mask.cuda()
            y5, y2 = y5.cuda(), y2.cuda()

            _, _, _, log5, log2 = model(wli, afi, ids, mask)

            loss = (
                F.cross_entropy(log5, y5) +
                F.binary_cross_entropy_with_logits(log2, y2.float())
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc5, acc2 = evaluate(model, val_loader)

        print(f"Epoch {epoch}: ACC5={acc5:.4f}, ACC2={acc2:.4f}")

    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))


if __name__ == "__main__":
    main()