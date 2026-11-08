#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the full architecture."""

from opmd_cli import run_model_entry

if __name__ == "__main__":
    run_model_entry("full", "05_full_cross_attention_ordinal_contrastive_warmup")
