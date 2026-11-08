#!/usr/bin/env python3
"""Train the five architectures in separate processes with the same arguments."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from opmd_cli import MODEL_SPECS


def build_commands(kinds, forwarded):
    # Each model needs a separate experiment directory.
    if any(a == "--exp" or a.startswith("--exp=") for a in forwarded):
        raise ValueError("--exp is only supported by individual model scripts")
    here = Path(__file__).resolve().parent
    return [[sys.executable, str(here / MODEL_SPECS[kind][0]), *forwarded] for kind in kinds]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, epilog="Other arguments are passed to every training script. See 01_baseline_concat.py --help.")
    p.add_argument("--models", nargs="+", choices=list(MODEL_SPECS), default=list(MODEL_SPECS))
    p.add_argument("--dry_run", action="store_true", help="Print commands without importing torch or training")
    args, forwarded = p.parse_known_args(argv)
    commands = build_commands(args.models, forwarded)
    for command in commands:
        # JSON is unambiguous on Windows and Linux; subprocess uses an argument list.
        print(json.dumps(command, ensure_ascii=False), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
