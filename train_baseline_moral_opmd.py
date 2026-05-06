# ============================================================
# Baseline  (GITHUB / PAPER VERSION)
# ============================================================

import os
import argparse
import json
import csv
import time
import datetime
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
# 0. Args
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

    def build_text(self, row):
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
            self.build_text(row),
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
# 2. Model (baseline)
# ============================================================
class VisualEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        base.fc = nn.Linear(base.fc.in_features, embed_dim)
        self.encoder = base

    def forward(self, x):
        return self.encoder(x)


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


class FusionModel(nn.Module):
    def __init__(self, bert_path, embed_dim=512):
        super().__init__()
        self.wli = VisualEncoder(embed_dim)
        self.afi = VisualEncoder(embed_dim)
        self.txt = TextEncoder(bert_path, embed_dim)

        self.cls5 = nn.Linear(embed_dim * 3, 5)
        self.cls2 = nn.Linear(embed_dim * 3, 1)

    def forward(self, wli, afi, ids, mask):
        fw = self.wli(wli)
        fa = self.afi(afi)
        ft = self.txt(ids, mask)

        h = torch.cat([fw, fa, ft], dim=1)
        return fw, fa, ft, self.cls5(h), self.cls2(h).squeeze(1)


# ============================================================
# 3. Loss
# ============================================================
def contrastive_loss(a, b, t=0.07):
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    sim = a @ b.T / t
    labels = torch.arange(sim.size(0), device=sim.device)

    return (F.cross_entropy(sim, labels) +
            F.cross_entropy(sim.T, labels)) / 2


def compute_loss(fw, fa, ft, log5, log2, y5, y2):
    loss_clip = 0.1 * (
        contrastive_loss(fw, ft) +
        contrastive_loss(fa, ft)
    )

    loss5 = F.cross_entropy(log5, y5)
    loss2 = F.binary_cross_entropy_with_logits(log2, y2.float())
    loss_h = F.binary_cross_entropy_with_logits(log2, (y5 >= 2).float())

    total = loss_clip + loss5 + 0.3 * loss2 + 0.3 * loss_h

    return total


# ============================================================
# 4. Eval
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
        transforms.ColorJitter(0.1,0.1,0.05),
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

    model = FusionModel(args.bert_path).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()

        for wli, afi, ids, mask, y5, y2 in train_loader:
            wli, afi = wli.cuda(), afi.cuda()
            ids, mask = ids.cuda(), mask.cuda()
            y5, y2 = y5.cuda(), y2.cuda()

            fw, fa, ft, log5, log2 = model(wli, afi, ids, mask)

            loss = compute_loss(fw, fa, ft, log5, log2, y5, y2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc5, acc2 = evaluate(model, val_loader)
        print(f"Epoch {epoch}: ACC5={acc5:.4f}, ACC2={acc2:.4f}")

    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))


if __name__ == "__main__":
    main()