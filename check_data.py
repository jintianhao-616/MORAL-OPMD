#!/usr/bin/env python3
"""Check CSV labels, patient splits and optional image paths without loading models."""

import argparse
from pathlib import Path

from opmd_data import COL5, COL2, load_opmd_csv, detect_id_col, assert_no_patient_overlap, resolve_image_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv")
    p.add_argument("--train_csv")
    p.add_argument("--val_csv")
    p.add_argument("--test_csv")
    p.add_argument("--image_root")
    p.add_argument("--check_images", action="store_true")
    args = p.parse_args(argv)
    if not any([args.csv, args.train_csv, args.val_csv, args.test_csv]):
        args.csv = str(Path(__file__).resolve().parent / "data" / "sample.csv")
    if args.csv:
        if args.train_csv or args.val_csv or args.test_csv:
            p.error("Use --csv alone, or --train_csv/--val_csv/--test_csv")
        paths = [args.csv]
    else:
        if not args.train_csv or not args.val_csv:
            p.error("Supply --csv or both --train_csv and --val_csv")
        assert_no_patient_overlap(args.train_csv, args.val_csv, args.test_csv)
        paths = [args.train_csv, args.val_csv] + ([args.test_csv] if args.test_csv else [])
    for path in paths:
        df = load_opmd_csv(path)
        patients = df.drop_duplicates(detect_id_col(df))
        if args.check_images:
            missing = {str(resolve_image_path(value, path, args.image_root))
                       for col in ["whitelight_image", "fluorescent_image"] for value in df[col]
                       if not resolve_image_path(value, path, args.image_root).is_file()}
            if missing:
                raise FileNotFoundError(f"{path}: {len(missing)} missing images: {sorted(missing)[:10]}")
        print(f"{Path(path)}: {len(df)} image pairs; {len(patients)} patients; "
              f"grades={patients[COL5].value_counts().sort_index().to_dict()}; "
              f"risk={patients[COL2].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
