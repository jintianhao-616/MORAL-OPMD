#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

def _early_gpu_from_argv(default="0"):
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--gpu" and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith("--gpu="):
            return arg.split("=", 1)[1]
    return default

os.environ["CUDA_VISIBLE_DEVICES"] = _early_gpu_from_argv("0")

import torch

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch version        =", torch.__version__)
print("torch CUDA version   =", torch.version.cuda)
print("cuda available       =", torch.cuda.is_available())
print("visible device count =", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"logical cuda:{i} -> {torch.cuda.get_device_name(i)}")
    x = torch.tensor([1.0], device="cuda:0")
    print("CUDA tensor test     =", x, "OK")
else:
    raise SystemExit("CUDA is not available in this Python environment.")
