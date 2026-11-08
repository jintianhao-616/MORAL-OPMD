"""Model registry and command-line parsing without importing torch."""

import argparse
import os

ROOT_DEFAULT = "."
BERT_DEFAULT = "google-bert/bert-base-chinese"
MODEL_SPECS = {
    "baseline_concat": ("01_baseline_concat.py", "01_baseline_concat"),
    "mlp_fusion": ("02_mlp_fusion.py", "02_mlp_fusion"),
    "cross_attention": ("03_cross_attention.py", "03_cross_attention"),
    "cross_attention_ordinal": ("04_cross_attention_ordinal.py", "04_cross_attention_ordinal"),
    "full": ("05_full_cross_attention_ordinal_contrastive.py", "05_full_cross_attention_ordinal_contrastive_warmup"),
}


def build_argparser(default_exp):
    p = argparse.ArgumentParser(description="Train one MORAL-OPMD architecture.")
    p.add_argument("--root", default=ROOT_DEFAULT)
    p.add_argument("--bert_path", default=BERT_DEFAULT)
    p.add_argument("--image_root", default=None, help="Base directory for relative image paths")
    p.add_argument("--overwrite", action="store_true", help="Allow reuse of an existing experiment directory")
    p.add_argument("--train_csv", default=None)
    p.add_argument("--val_csv", default=None)
    p.add_argument("--test_csv", default=None)
    p.add_argument("--exp", default=default_exp)
    p.add_argument("--gpu", default="0")
    p.add_argument("--pair_policy", choices=["all", "one_fixed"], default="all")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--min_epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min_delta", type=float, default=1e-5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--max_text_len", type=int, default=64)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--label_smoothing", type=float, default=0.03)
    p.add_argument("--binary_lambda", type=float, default=0.50)
    p.add_argument("--ordinal_lambda", type=float, default=0.20)
    p.add_argument("--monotonic_lambda", type=float, default=0.05)
    p.add_argument("--hier_lambda", type=float, default=0.20)
    p.add_argument("--contrastive_start", type=int, default=0)
    p.add_argument("--contrastive_warmup_epochs", type=int, default=5)
    p.add_argument("--contrastive_lambda", type=float, default=0.02)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--no_patient_balanced_sampling", action="store_true")
    p.add_argument("--seed", type=int, default=20260922)
    return p


def validate_training_args(args):
    for name in ["epochs", "min_epochs", "patience", "batch", "size", "max_text_len", "dim", "heads", "contrastive_warmup_epochs"]:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.workers < 0 or args.contrastive_start < 0:
        raise ValueError("--workers and --contrastive_start must be nonnegative")
    if args.dim % args.heads:
        raise ValueError("--dim must be divisible by --heads")
    if args.size > 224:
        raise ValueError("--size must not exceed 224 for the 49-token visual position table")
    if args.temperature <= 0 or args.lr <= 0:
        raise ValueError("--temperature and --lr must be positive")


def run_model_entry(model_kind, default_exp=None):
    if model_kind not in MODEL_SPECS:
        raise ValueError(f"Unknown model kind: {model_kind}")
    args = build_argparser(default_exp or MODEL_SPECS[model_kind][1]).parse_args()
    validate_training_args(args)
    # This happens before the first import of torch. Physical IDs are mapped to cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from opmd_common import run_experiment
    run_experiment(model_kind, default_exp or MODEL_SPECS[model_kind][1], args=args)
