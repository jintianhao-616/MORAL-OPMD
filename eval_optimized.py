# ============================================================
# OPMD Evaluation Script (GITHUB + PAPER VERSION)
# Patient-level evaluation
# ============================================================

import os
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix
)


# ============================================================
# 0. Args
# ============================================================
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--bert_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./eval_outputs")

    return parser.parse_args()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1. Dataset
# ============================================================
class OPMDCLIPDataset(Dataset):
    def __init__(self, csv_path, bert_path, transform):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

    def auto_align_afi(self, afi_img, wli_img):
        resize = transforms.Resize((224, 224))
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

        wli = self.transform(wli)
        afi = self.transform(afi)

        enc = self.tokenizer(
            self.build_prompt(row),
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        return (
            wli, afi,
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            int(row["label_5cls"]),
            int(row["label_2cls"]),
            row["id"]
        )


# ============================================================
# 2. Model
# ============================================================
class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )
        hidden = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.proj = nn.Linear(hidden, 512)

    def forward(self, x):
        return self.proj(self.backbone(x))


class TextEncoder(nn.Module):
    def __init__(self, bert_path):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        for p in self.bert.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(self.bert.config.hidden_size, 512)

    def forward(self, ids, mask):
        out = self.bert(input_ids=ids, attention_mask=mask)
        return self.proj(out.last_hidden_state[:, 0])


class Model(nn.Module):
    def __init__(self, bert_path):
        super().__init__()
        self.wli = VisualEncoder()
        self.afi = VisualEncoder()
        self.txt = TextEncoder(bert_path)

        self.fusion = nn.Sequential(
            nn.Linear(512 * 3, 512),
            nn.ReLU()
        )

        self.cls5 = nn.Linear(512, 5)
        self.cls2 = nn.Linear(512, 1)

    def forward(self, wli, afi, ids, mask):
        fw = self.wli(wli)
        fa = self.afi(afi)
        ft = self.txt(ids, mask)

        fused = self.fusion(torch.cat([fw, fa, ft], dim=1))
        return self.cls5(fused), self.cls2(fused).squeeze(1)


# ============================================================
# 3. Inference
# ============================================================
@torch.no_grad()
def inference(model, loader):
    model.eval()

    y5_t, y5_p, y2_t, y2_p, ids = [], [], [], [], []

    for wli, afi, ids_t, mask, y5, y2, pid in tqdm(loader):
        wli, afi = wli.to(DEVICE), afi.to(DEVICE)
        ids_t, mask = ids_t.to(DEVICE), mask.to(DEVICE)

        log5, log2 = model(wli, afi, ids_t, mask)

        y5_t.extend(y5.numpy())
        y5_p.extend(log5.argmax(1).cpu().numpy())
        y2_t.extend(y2.numpy())
        y2_p.extend((log2 > 0).long().cpu().numpy())
        ids.extend(pid)

    return pd.DataFrame({
        "id": ids,
        "gt_5cls": y5_t,
        "pred_5cls": y5_p,
        "gt_2cls": y2_t,
        "pred_2cls": y2_p
    })


# ============================================================
# 4. Patient-level aggregation
# ============================================================
def aggregate(df):
    patient_df = df.groupby("id").apply(lambda x: pd.Series({
        "gt_5cls": x["gt_5cls"].iloc[0],
        "gt_2cls": x["gt_2cls"].iloc[0],
        "pred_5cls": x["pred_5cls"].value_counts().idxmax(),
        "pred_2cls": x["pred_2cls"].max()
    })).reset_index()

    gt2 = patient_df["gt_2cls"]
    pr2 = patient_df["pred_2cls"]

    tn, fp, fn, tp = confusion_matrix(gt2, pr2).ravel()

    metrics = {
        "accuracy": accuracy_score(gt2, pr2),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "macro_f1": f1_score(gt2, pr2, average="macro")
    }

    return metrics, patient_df


# ============================================================
# 5. Main
# ============================================================
def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    loader = DataLoader(
        OPMDCLIPDataset(args.test_csv, args.bert_path, transform),
        batch_size=16,
        shuffle=False
    )

    model = Model(args.bert_path).to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))

    df = inference(model, loader)
    metrics, patient_df = aggregate(df)

    df.to_csv(f"{args.output_dir}/pair_level.csv", index=False)
    patient_df.to_csv(f"{args.output_dir}/patient_level.csv", index=False)

    with open(f"{args.output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(metrics)


if __name__ == "__main__":
    main()