#!/usr/bin/env python3
"""Evaluate any of the five architectures using its saved configuration."""

import argparse
import os
from pathlib import Path

from opmd_cli import MODEL_SPECS, BERT_DEFAULT

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TEST_CSV = PROJECT_DIR / "data" / "sample.csv"


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", "--model_path", dest="checkpoint", required=True)
    p.add_argument("--csv", "--test_csv", dest="csv", default=str(DEFAULT_TEST_CSV),
                   help="Default: the unchanged data/sample.csv supplied in this repository")
    p.add_argument("--output_dir", default="eval_outputs")
    p.add_argument("--model_kind", choices=list(MODEL_SPECS), default=None)
    p.add_argument("--bert_path", default=BERT_DEFAULT, help="BERT config/tokenizer model ID or local folder")
    p.add_argument("--image_root", default=None)
    p.add_argument("--gpu", default="0", help="Physical CUDA ID, or -1 for CPU")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--pair_policy", choices=["all", "one_fixed"], default=None)
    p.add_argument("--aggregation", choices=["mean_prob", "max_prob", "max_risk", "majority_max_risk"], default=None)
    p.add_argument("--dim", type=int, default=None, help="Required to match a bare state_dict")
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--max_text_len", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.batch <= 0 or args.workers < 0:
        raise ValueError("--batch must be positive and --workers nonnegative")
    if args.image_root is None and Path(args.csv).resolve() == DEFAULT_TEST_CSV:
        args.image_root = str(PROJECT_DIR)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Import model libraries after selecting the GPU and handling --help.
    import torch
    from torch.utils.data import DataLoader
    from opmd_common import OPMDDataset, build_model, predict
    from opmd_checkpoint import resolve_checkpoint
    from opmd_metrics import add_pair_predictions, patient_aggregate, calculate_metrics, save_confusions, write_json

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    kind, state, cfg, original_config = resolve_checkpoint(
        checkpoint, args.model_kind, bert_path=args.bert_path, dim=args.dim,
        heads=args.heads, size=args.size, max_text_len=args.max_text_len,
        pair_policy=args.pair_policy,
    )
    aggregation = args.aggregation or original_config.get("primary_patient_aggregation", "mean_prob")
    if aggregation not in ["mean_prob", "max_prob", "max_risk", "majority_max_risk"]:
        raise ValueError(f"Unknown aggregation in checkpoint: {aggregation}")
    outdir = Path(args.output_dir)
    if outdir.exists() and any(outdir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {outdir}; use --overwrite or a new --output_dir")
    dataset = OPMDDataset(args.csv, cfg.bert_path, train=False, pair_policy=cfg.pair_policy,
                          size=cfg.size, max_text_len=cfg.max_text_len, image_root=args.image_root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # All trained parameters, including BERT, are restored from the checkpoint.
    # This avoids downloading ImageNet or BERT weights during evaluation.
    model = build_model(kind, cfg, pretrained=False)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError("Checkpoint parameters do not match the selected model. "
                           "Use the checkpoint in its matching numbered model folder.") from error
    model.to(device)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"))
    raw = predict(model, loader, device)
    pairs = add_pair_predictions(raw)
    outdir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(outdir / "image_pair_predictions.csv", index=False)
    save_confusions(pairs, outdir, "image_pair")
    metrics = {"model_kind": kind, "checkpoint": str(Path(args.checkpoint).resolve()),
               "epoch": checkpoint.get("epoch") if "model" in checkpoint else None,
               "primary_patient_aggregation": aggregation, "pair_policy": cfg.pair_policy,
               "binary_threshold": 0.5, "image_pair": calculate_metrics(pairs, "pairs")}
    for method in ["mean_prob", "max_prob", "max_risk", "majority_max_risk"]:
        patients = patient_aggregate(raw, method)
        patients.to_csv(outdir / f"patient_{method}_predictions.csv", index=False)
        save_confusions(patients, outdir, f"patient_{method}")
        metrics[f"patient_{method}"] = calculate_metrics(patients, "patients")
    write_json(outdir / "metrics.json", metrics)
    effective = {**vars(args), "model_kind": kind, "resolved_model_config": vars(cfg),
                 "primary_patient_aggregation": aggregation}
    write_json(outdir / "evaluation_config.json", effective)
    print(f"{kind} | {device} | primary aggregation: {aggregation}")
    print(metrics[f"patient_{aggregation}"])
    print(f"Saved evaluation to {outdir.resolve()}")


if __name__ == "__main__":
    main()
