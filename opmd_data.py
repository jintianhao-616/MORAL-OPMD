"""CSV schema, patient checks and image paths; no model dependencies."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

COL5 = "上皮异常增生程度（0：无；1：轻度；2：中度；3：重度；4：癌变）"
COL2 = "上皮异常增生程度（0：低级别；1：高级别）"
ALIASES = {
    "Lesion location": "病损部位",
    "Clinical diagnosis": "疾病诊断",
    "label_5cls": COL5,
    "label_2cls": COL2,
}


def detect_id_col(df):
    for c in ["id", "patient_id", "患者ID", "患者编号", "病例号"]:
        if c in df.columns:
            return c
    raise ValueError("No patient ID column found")


def validate_required_columns(df):
    required = ["whitelight_image", "fluorescent_image", "病损部位", "疾病诊断", COL5, COL2]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_patient_labels(df, id_col, name):
    bad = []
    for pid, g in df.groupby(id_col, sort=False):
        if g[COL5].nunique() != 1 or g[COL2].nunique() != 1:
            bad.append(str(pid))
    if bad:
        raise RuntimeError(f"{name}: inconsistent patient labels. Examples: {bad[:10]}")


def load_opmd_csv(path):
    # Read IDs as text so leading zeroes and values such as 'NA' survive.
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"{path}: CSV has no rows")
    for alias, canonical in ALIASES.items():
        if alias in df.columns:
            if canonical in df.columns:
                if not df[alias].equals(df[canonical]):
                    raise ValueError(f"{path}: conflicting columns {alias!r} and {canonical!r}")
                df = df.drop(columns=[alias])
            else:
                df = df.rename(columns={alias: canonical})
    validate_required_columns(df)
    id_col = detect_id_col(df)
    if id_col != "id":
        df = df.rename(columns={id_col: "id"})
        id_col = "id"
    df[id_col] = df[id_col].str.strip()
    if df[id_col].eq("").any():
        raise ValueError(f"{path}: blank patient IDs")
    for column, upper in [(COL5, 4), (COL2, 1)]:
        values = pd.to_numeric(df[column], errors="coerce")
        if not (np.isfinite(values) & values.between(0, upper) & (values % 1 == 0)).all():
            raise ValueError(f"{path}: {column} must contain integers from 0 to {upper}")
        df[column] = values.astype(int)
    if not df[COL2].eq((df[COL5] >= 2).astype(int)).all():
        raise ValueError(f"{path}: binary labels must equal (five-class label >= 2)")
    for column in ["whitelight_image", "fluorescent_image"]:
        df[column] = df[column].str.strip()
        if df[column].eq("").any():
            raise ValueError(f"{path}: blank {column} paths")
    validate_patient_labels(df, id_col, str(path))
    return df


def assert_no_patient_overlap(train_csv, val_csv, test_csv=None):
    paths = [train_csv, val_csv] + ([test_csv] if test_csv else [])
    names = ["train", "val"] + (["test"] if test_csv else [])
    sets = []
    for path in paths:
        df = load_opmd_csv(path)
        sets.append(set(df[detect_id_col(df)]))
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            if overlap:
                raise RuntimeError(f"PATIENT LEAKAGE: {names[i]} vs {names[j]} share "
                                   f"{len(overlap)} IDs; examples={sorted(overlap)[:10]}")


def choose_one_fixed_pair(df, id_col):
    keep = []
    for _, g in df.groupby(id_col, sort=False):
        keys = g["whitelight_image"].astype(str) + "||" + g["fluorescent_image"].astype(str)
        # Same hash rule as the supplied training code, independent of row order.
        keep.append(min(zip(g.index, keys), key=lambda x: hashlib.sha1(x[1].encode("utf-8")).hexdigest())[0])
    return df.loc[keep].reset_index(drop=True)


def resolve_image_path(value, csv_path, image_root=None):
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p
    if image_root is not None:
        return Path(image_root).expanduser().resolve() / p
    project_dir = Path(__file__).resolve().parent
    if Path(csv_path).resolve() == project_dir / "data" / "sample.csv":
        return project_dir / p
    # Preserve the existing sample CSV's repository-relative paths.
    if p.is_file():
        return p.resolve()
    return Path(csv_path).resolve().parent / p


def clinical_prompt(row):
    def safe(value):
        return "未知" if pd.isna(value) or str(value).strip() == "" else str(value).strip()
    return f"病损部位：{safe(row.get('病损部位'))}。临床诊断：{safe(row.get('疾病诊断'))}。"
