#!/usr/bin/env python3
"""Evaluate the five downloaded checkpoint folders on one fixed cohort."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from opmd_cli import MODEL_SPECS, BERT_DEFAULT

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TEST_CSV = PROJECT_DIR / "data" / "sample.csv"

SUMMARY_FIELDS = [
    "model_kind", "checkpoint", "best_epoch", "aggregation", "pair_policy", "n_patients",
    "patient_acc5", "patient_bal_acc5", "patient_macro_f1_5", "patient_macro_auc5_ovr",
    "patient_acc2", "patient_bal_acc2", "patient_auc2", "patient_pr_auc2",
    "patient_sensitivity2", "patient_specificity2", "patient_severe_recall",
    "pair_acc5", "pair_macro_f1_5", "pair_macro_auc5_ovr", "pair_auc2", "pair_pr_auc2",
]


def build_commands(args):
    evaluator = Path(__file__).resolve().parent / "eval_moral_opmd.py"
    image_root = args.image_root
    if image_root is None and Path(args.csv).resolve() == DEFAULT_TEST_CSV:
        image_root = str(PROJECT_DIR)
    commands = []
    for kind, (_, folder) in MODEL_SPECS.items():
        checkpoint = Path(args.weights_dir) / folder / "best_patient_mean_macro_f1.pt"
        command = [sys.executable, str(evaluator), "--checkpoint", str(checkpoint),
                   "--model_kind", kind, "--csv", args.csv,
                   "--output_dir", str(Path(args.output_dir) / folder),
                   "--gpu", args.gpu, "--batch", str(args.batch), "--workers", str(args.workers)]
        for name in ["bert_path", "image_root", "aggregation", "pair_policy"]:
            value = image_root if name == "image_root" else getattr(args, name)
            if value is not None:
                command.extend([f"--{name}", value])
        if args.overwrite:
            command.append("--overwrite")
        commands.append((kind, folder, checkpoint, command))
    return commands


def write_summary(output_dir):
    root = Path(output_dir)
    rows = []
    for kind, (_, folder) in MODEL_SPECS.items():
        metrics = json.loads((root / folder / "metrics.json").read_text(encoding="utf-8"))
        if metrics["model_kind"] != kind:
            raise ValueError(f"Unexpected model in {folder}: {metrics['model_kind']}")
        method = metrics["primary_patient_aggregation"]
        patient, pair = metrics[f"patient_{method}"], metrics["image_pair"]
        row = {"model_kind": kind, "checkpoint": metrics["checkpoint"], "best_epoch": metrics["epoch"],
               "aggregation": method, "pair_policy": metrics["pair_policy"],
               "n_patients": patient["n_patients"], "patient_severe_recall": patient["recall_grade_3"]}
        for field in SUMMARY_FIELDS:
            if field in row:
                continue
            if field.startswith("patient_"):
                row[field] = patient[field[len("patient_"):]]
            elif field.startswith("pair_"):
                row[field] = pair[field[len("pair_"):]]
        rows.append(row)
    destination = root / "five_model_summary.csv"
    with destination.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights_dir", default=str(PROJECT_DIR / "models"), help="Parent of the five numbered downloaded folders")
    p.add_argument("--csv", "--test_csv", dest="csv", default=str(DEFAULT_TEST_CSV),
                   help="Default: the unchanged data/sample.csv supplied in this repository")
    p.add_argument("--output_dir", default=str(PROJECT_DIR / "eval_outputs" / "five_models"))
    p.add_argument("--bert_path", default=BERT_DEFAULT)
    p.add_argument("--image_root", default=None)
    p.add_argument("--gpu", default="0")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--aggregation", choices=["mean_prob", "max_prob", "max_risk", "majority_max_risk"],
                   default="mean_prob", help="Use one prespecified aggregation policy for all five models")
    p.add_argument("--pair_policy", choices=["all", "one_fixed"], default="all",
                   help="Use one pair policy for all five models; choose before reviewing test results")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="Print expected paths and commands without loading weights")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.batch <= 0 or args.workers < 0:
        raise ValueError("--batch must be positive and --workers nonnegative")
    commands = build_commands(args)
    if not args.dry_run:
        missing = [str(checkpoint) for _, _, checkpoint, _ in commands if not checkpoint.is_file()]
        if missing:
            raise FileNotFoundError("Missing downloaded checkpoints:\n" + "\n".join(missing))
        if not Path(args.csv).is_file():
            raise FileNotFoundError(args.csv)
        existing = [str(Path(args.output_dir) / folder) for _, folder, _, _ in commands
                    if (Path(args.output_dir) / folder).exists() and any((Path(args.output_dir) / folder).iterdir())]
        if existing and not args.overwrite:
            raise FileExistsError("Existing evaluation outputs; use a new --output_dir or --overwrite: " + ", ".join(existing))
        summary = Path(args.output_dir) / "five_model_summary.csv"
        if summary.exists() and not args.overwrite:
            raise FileExistsError(f"Summary already exists: {summary}; use --overwrite or a new --output_dir")
    for _, _, _, command in commands:
        print(json.dumps(command, ensure_ascii=False), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    if not args.dry_run:
        print(f"Saved comparison to {write_summary(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
