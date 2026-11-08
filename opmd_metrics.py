"""Prediction aggregation and metrics shared by training and evaluation."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    average_precision_score, precision_recall_fscore_support, confusion_matrix,
)

def add_pair_predictions(df):
    out = df.copy()
    out["pred5"] = out["p5"].map(lambda s: int(np.argmax(np.asarray(json.loads(s), dtype=float))))
    out["pred2"] = (out["p2"].astype(float) >= 0.5).astype(int)
    return out


def patient_aggregate(df, method="mean_prob"):
    if method not in ["mean_prob", "max_prob", "max_risk", "majority_max_risk"]:
        raise ValueError(method)
    rows = []
    for pid, g in df.groupby("pid", sort=False):
        y5u = g["y5"].astype(int).unique()
        y2u = g["y2"].astype(int).unique()
        if len(y5u) != 1 or len(y2u) != 1:
            raise RuntimeError(f"Patient {pid} has inconsistent labels")
        p5 = np.vstack(g["p5"].map(json.loads))
        p2 = g["p2"].to_numpy(float)
        if method == "mean_prob":
            q5, q2 = p5.mean(0), float(p2.mean())
        elif method == "max_prob":
            q5 = p5.max(0)
            q5 = q5 / np.clip(q5.sum(), 1e-12, None)
            q2 = float(p2.max())
        elif method == "max_risk":
            k = int(p5.argmax(1).max())
            q5 = np.zeros(5, dtype=float)
            q5[k] = 1.0
            q2 = float(p2.max())
        elif method == "majority_max_risk":
            # Argmax breaks vote ties toward the lower grade, explicitly and reproducibly.
            votes = np.bincount(p5.argmax(1), minlength=5)
            k = int(votes.argmax())
            # Keep continuous probabilities for AUC; votes determine the hard label.
            q5 = p5.mean(0)
            q2 = float(p2.max())
        else:
            raise ValueError(method)
        rows.append({
            "pid": str(pid), "n_pairs": int(len(g)), "y5": int(y5u[0]), "y2": int(y2u[0]),
            "p5": json.dumps(q5.tolist()), "p2": q2,
            "pred5": k if method == "majority_max_risk" else int(np.argmax(q5)),
            "pred2": int(q2 >= 0.5)
        })
    return pd.DataFrame(rows)


def binary_stats(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    return sens, spec, ppv, npv


def calculate_metrics(df, level):
    y5 = df["y5"].to_numpy(int)
    pr5 = df["pred5"].to_numpy(int)
    y2 = df["y2"].to_numpy(int)
    pr2 = df["pred2"].to_numpy(int)
    p2 = df["p2"].to_numpy(float)
    p5 = np.vstack(df["p5"].map(json.loads))
    d = {
        f"n_{level}": int(len(df)),
        "acc5": float(accuracy_score(y5, pr5)),
        "bal_acc5": float(balanced_accuracy_score(y5, pr5)),
        "macro_f1_5": float(f1_score(y5, pr5, labels=[0, 1, 2, 3, 4], average="macro", zero_division=0)),
        "weighted_f1_5": float(f1_score(y5, pr5, average="weighted", zero_division=0)),
        "acc2": float(accuracy_score(y2, pr2)),
        "bal_acc2": float(balanced_accuracy_score(y2, pr2)),
        "macro_f1_2": float(f1_score(y2, pr2, labels=[0, 1], average="macro", zero_division=0)),
    }
    if len(np.unique(y2)) == 2:
        d["auc2"] = float(roc_auc_score(y2, p2))
        d["pr_auc2"] = float(average_precision_score(y2, p2))
    else:
        d["auc2"] = float("nan")
        d["pr_auc2"] = float("nan")
    try:
        if len(np.unique(y5)) != 5:
            raise ValueError("Macro five-class AUC requires all five reference classes")
        d["macro_auc5_ovr"] = float(roc_auc_score(y5, p5, multi_class="ovr", average="macro", labels=[0,1,2,3,4]))
    except ValueError:
        d["macro_auc5_ovr"] = float("nan")
    precision, recall, f1v, support = precision_recall_fscore_support(y5, pr5, labels=[0,1,2,3,4], zero_division=0)
    for k in range(5):
        d[f"precision_grade_{k}"] = float(precision[k])
        d[f"recall_grade_{k}"] = float(recall[k])
        d[f"f1_grade_{k}"] = float(f1v[k])
        d[f"support_grade_{k}"] = int(support[k])
    sens, spec, ppv, npv = binary_stats(y2, pr2)
    d.update({"sensitivity2": float(sens), "specificity2": float(spec), "ppv2": float(ppv), "npv2": float(npv)})
    return d


def write_json(path, payload):
    """Write standard JSON; undefined metrics are null instead of nonstandard NaN."""
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    Path(path).write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding="utf-8")


def save_confusions(df, outdir, prefix):
    cm5 = confusion_matrix(df["y5"], df["pred5"], labels=[0,1,2,3,4])
    cm2 = confusion_matrix(df["y2"], df["pred2"], labels=[0,1])
    pd.DataFrame(cm5).to_csv(outdir / f"{prefix}_confusion_5class.csv", index=False)
    pd.DataFrame(cm2).to_csv(outdir / f"{prefix}_confusion_2class.csv", index=False)
