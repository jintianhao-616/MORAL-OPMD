import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from opmd_cli import MODEL_SPECS, build_argparser
from opmd_checkpoint import resolve_checkpoint
from opmd_data import (COL5, COL2, load_opmd_csv, assert_no_patient_overlap,
                       choose_one_fixed_pair, clinical_prompt, resolve_image_path)
from opmd_metrics import add_pair_predictions, patient_aggregate, calculate_metrics, write_json
from train_all_models import build_commands
from eval_all_models import (build_parser as build_eval_all_parser,
                             build_commands as build_eval_commands, write_summary, SUMMARY_FIELDS)
from eval_moral_opmd import build_parser as build_single_eval_parser

ROOT = Path(__file__).resolve().parents[1]


def example_rows():
    return pd.DataFrame({
        "id": ["001", "001", "002"], "whitelight_image": ["a.jpg", "b.jpg", "c.jpg"],
        "fluorescent_image": ["aa.jpg", "bb.jpg", "cc.jpg"],
        "Lesion location": ["舌", "舌", "颊"], "Clinical diagnosis": ["白斑"] * 3,
        "label_5cls": [2, 2, 0], "label_2cls": [1, 1, 0],
    })


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "data.csv"

    def save(self, rows):
        rows.to_csv(self.path, index=False)
        return load_opmd_csv(self.path)

    def test_aliases_ids_and_no_pathology_in_prompt(self):
        df = self.save(example_rows())
        self.assertEqual(df["id"].tolist(), ["001", "001", "002"])
        self.assertEqual(df[COL5].tolist(), [2, 2, 0])
        row = df.iloc[0].copy()
        original = clinical_prompt(row)
        row[COL5], row[COL2], row["病理诊断"] = 4, 0, "THIS_MUST_NOT_BE_INPUT"
        self.assertEqual(clinical_prompt(row), original)
        self.assertEqual(original, "病损部位：舌。临床诊断：白斑。")

    def test_label_errors_are_rejected(self):
        for column, value in [("label_5cls", 2.5), ("label_5cls", 5),
                              ("label_2cls", 0), ("id", "")]:
            with self.subTest(column=column, value=value):
                rows = example_rows()
                rows[column] = rows[column].astype(object)
                rows.loc[0, column] = value
                with self.assertRaises(ValueError):
                    self.save(rows)
        rows = example_rows()
        rows.loc[1, ["label_5cls", "label_2cls"]] = [3, 1]
        with self.assertRaises(RuntimeError):
            self.save(rows)

    def test_alternative_id_header_is_canonicalized(self):
        df = self.save(example_rows().rename(columns={"id": "患者编号"}))
        self.assertEqual(df["id"].tolist(), ["001", "001", "002"])

    def test_patient_overlap_includes_optional_test_set(self):
        train, val, test = [Path(self.temp.name) / f"{x}.csv" for x in ["train", "val", "test"]]
        rows = example_rows()
        rows.iloc[:2].to_csv(train, index=False)
        rows.iloc[2:].to_csv(val, index=False)
        rows.iloc[:1].to_csv(test, index=False)
        assert_no_patient_overlap(train, val)
        with self.assertRaisesRegex(RuntimeError, "PATIENT LEAKAGE"):
            assert_no_patient_overlap(train, val, test)

    def test_fixed_pair_does_not_depend_on_csv_order(self):
        df = self.save(example_rows())
        left = choose_one_fixed_pair(df, "id").set_index("id")["whitelight_image"].to_dict()
        right = choose_one_fixed_pair(df.iloc[::-1].reset_index(drop=True), "id").set_index("id")["whitelight_image"].to_dict()
        self.assertEqual(left, right)
        self.assertEqual(len(left), 2)

    def test_image_root_and_csv_relative_paths(self):
        self.assertEqual(resolve_image_path("a.jpg", self.path), self.path.parent / "a.jpg")
        self.assertEqual(resolve_image_path("a.jpg", self.path, self.path.parent), self.path.parent / "a.jpg")
        self.assertEqual(resolve_image_path(str(self.path), self.path), self.path)


class MetricTests(unittest.TestCase):
    def test_aggregation_methods_and_vote_tie(self):
        raw = pd.DataFrame({"pid": ["001"] * 3, "y5": [2] * 3, "y2": [1] * 3,
                            "p5": [json.dumps(x) for x in [[.51, 0, .49, 0, 0], [.51, 0, .49, 0, 0], [0, 0, 1, 0, 0]]],
                            "p2": [.2, .3, .9]})
        mean = patient_aggregate(raw, "mean_prob").iloc[0]
        majority = patient_aggregate(raw, "majority_max_risk").iloc[0]
        self.assertEqual((mean.pred5, mean.pred2), (2, 0))
        self.assertEqual((majority.pred5, majority.pred2), (0, 1))
        self.assertEqual(patient_aggregate(raw, "max_risk").iloc[0].pred5, 2)
        tie = patient_aggregate(raw.iloc[[0, 2]], "majority_max_risk").iloc[0]
        self.assertEqual(tie.pred5, 0)
        self.assertAlmostEqual(sum(json.loads(majority.p5)), 1)
        self.assertAlmostEqual(sum(json.loads(patient_aggregate(raw, "max_prob").iloc[0].p5)), 1)

    def test_perfect_five_class_metrics(self):
        rows = pd.DataFrame({"pid": list("abcde"), "y5": list(range(5)), "y2": [0, 0, 1, 1, 1],
                             "p5": [json.dumps(x.tolist()) for x in np.eye(5)], "p2": [.1, .2, .8, .9, .95]})
        metrics = calculate_metrics(add_pair_predictions(rows), "pairs")
        for name in ["acc5", "macro_f1_5", "macro_auc5_ovr", "auc2", "pr_auc2", "sensitivity2", "specificity2"]:
            self.assertEqual(metrics[name], 1.0, name)

    def test_single_class_metrics_and_strict_json(self):
        rows = pd.DataFrame({"pid": ["a"], "y5": [0], "y2": [0],
                             "p5": [json.dumps([1, 0, 0, 0, 0])], "p2": [.1]})
        metrics = calculate_metrics(add_pair_predictions(rows), "patients")
        self.assertTrue(math.isnan(metrics["auc2"]))
        self.assertTrue(math.isnan(metrics["sensitivity2"]))
        self.assertEqual(metrics["macro_f1_5"], .2)  # fixed five-class average
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "metrics.json"
            write_json(p, metrics)
            self.assertNotIn("NaN", p.read_text())
            self.assertIsNone(json.loads(p.read_text())["auc2"])


class WorkflowTests(unittest.TestCase):
    def test_all_five_entrypoints_and_eval_help(self):
        scripts = [x[0] for x in MODEL_SPECS.values()] + ["eval_moral_opmd.py", "eval_all_models.py", "check_data.py", "train_all_models.py"]
        self.assertEqual(len(MODEL_SPECS), 5)
        for script in scripts:
            with self.subTest(script=script):
                result = subprocess.run([sys.executable, str(ROOT / script), "--help"], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_runner_preserves_arguments_with_spaces(self):
        forwarded = ["--root", "/a path/project", "--gpu=1"]
        commands = build_commands(list(MODEL_SPECS), forwarded)
        self.assertEqual(len(commands), 5)
        self.assertEqual(len({x[1] for x in commands}), 5)
        self.assertTrue(all(x[-3:] == forwarded for x in commands))
        with self.assertRaises(ValueError):
            build_commands(["full"], ["--exp=shared"])

    def test_checkpoint_configuration_for_each_model(self):
        for kind in MODEL_SPECS:
            ckpt = {"model_kind": kind, "model": {"module.x": 1},
                    "config": {"model_kind": kind, "dim": 384, "heads": 8}}
            resolved, state, cfg, _ = resolve_checkpoint(ckpt)
            self.assertEqual(resolved, kind)
            self.assertEqual(state, {"x": 1})
            self.assertEqual(cfg.dim, 384)
            with self.assertRaises(ValueError):
                resolve_checkpoint(ckpt, dim=512)
        with self.assertRaises(ValueError):
            resolve_checkpoint(ckpt, "baseline_concat")
        with self.assertRaises(ValueError):
            resolve_checkpoint({"mode": "multimodal", "model": {"x": 1}})
        with self.assertRaises(ValueError):
            resolve_checkpoint({"x": 1})
        self.assertEqual(resolve_checkpoint({"x": 1}, "full")[0], "full")

    def test_warmup_defaults_are_explicit(self):
        args = build_argparser("test").parse_args([])
        self.assertEqual((args.contrastive_start, args.contrastive_warmup_epochs, args.contrastive_lambda), (0, 5, .02))

    def test_download_manifest_matches_five_checkpoint_folders(self):
        manifest = json.loads((ROOT / "models/checkpoints.json").read_text())
        self.assertEqual(len(manifest["models"]), 5)
        for entry in manifest["models"]:
            self.assertEqual(entry["folder"], MODEL_SPECS[entry["model_kind"]][1])
            self.assertEqual(entry["relative_checkpoint_path"], entry["folder"] + "/best_patient_mean_macro_f1.pt")
            self.assertGreater(entry["size_bytes"], 0)

    def test_default_evaluation_uses_the_original_sample(self):
        batch_args = build_eval_all_parser().parse_args([])
        single_args = build_single_eval_parser().parse_args(["--checkpoint", "model.pt"])
        self.assertEqual(Path(batch_args.csv), ROOT / "data" / "sample.csv")
        self.assertEqual(Path(single_args.csv), ROOT / "data" / "sample.csv")
        self.assertEqual(batch_args.workers, 0)
        self.assertEqual(single_args.workers, 0)
        sample = load_opmd_csv(batch_args.csv)
        self.assertEqual(len(sample), 9)
        self.assertEqual(sample["id"].nunique(), 4)

    def test_default_data_paths_work_outside_the_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(ROOT / "eval_all_models.py"), "--dry_run"],
                                    cwd=tmp, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(commands), 5)
        for command in commands:
            self.assertEqual(Path(command[command.index("--csv") + 1]), ROOT / "data" / "sample.csv")
            self.assertEqual(Path(command[command.index("--image_root") + 1]), ROOT)
        sample = load_opmd_csv(ROOT / "data" / "sample.csv")
        self.assertEqual(resolve_image_path(sample.iloc[0]["whitelight_image"], ROOT / "data" / "sample.csv"),
                         ROOT / sample.iloc[0]["whitelight_image"])

    def test_batch_evaluation_preserves_folders_and_common_protocol(self):
        args = build_eval_all_parser().parse_args(["--csv", "test cohort.csv", "--weights_dir", "weights with spaces", "--gpu", "1"])
        commands = build_eval_commands(args)
        self.assertEqual(len(commands), 5)
        paths = set()
        for kind, folder, checkpoint, command in commands:
            self.assertEqual(checkpoint, Path("weights with spaces") / folder / "best_patient_mean_macro_f1.pt")
            self.assertEqual(command[command.index("--model_kind") + 1], kind)
            self.assertEqual(command[command.index("--csv") + 1], "test cohort.csv")
            self.assertEqual(command[command.index("--aggregation") + 1], "mean_prob")
            self.assertEqual(command[command.index("--pair_policy") + 1], "all")
            paths.add(checkpoint)
        self.assertEqual(len(paths), 5)

    def test_batch_summary_uses_the_selected_patient_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kind, (_, folder) in MODEL_SPECS.items():
                patient = {"n_patients": 50, "recall_grade_3": .75}
                pair = {}
                for field in SUMMARY_FIELDS:
                    if field.startswith("patient_") and field != "patient_severe_recall":
                        patient[field[len("patient_"):]] = .8
                    elif field.startswith("pair_") and field != "pair_policy":
                        pair[field[len("pair_"):]] = .7
                payload = {"model_kind": kind, "checkpoint": "test-only.pt", "epoch": 8,
                           "primary_patient_aggregation": "majority_max_risk", "pair_policy": "all",
                           "patient_majority_max_risk": patient, "image_pair": pair}
                directory = Path(tmp) / folder
                directory.mkdir()
                (directory / "metrics.json").write_text(json.dumps(payload))
            summary = pd.read_csv(write_summary(tmp))
            self.assertEqual(summary.columns.tolist(), SUMMARY_FIELDS)
            self.assertEqual(len(summary), 5)
            self.assertTrue((summary["aggregation"] == "majority_max_risk").all())
            self.assertTrue((summary["patient_severe_recall"] == .75).all())


@unittest.skipUnless(all(importlib.util.find_spec(x) for x in ["torch", "torchvision", "transformers"]),
                     "PyTorch, torchvision and transformers are needed for forward/backward smoke tests")
class ModelSmokeTests(unittest.TestCase):
    def test_five_architectures_forward_backward_and_restore(self):
        import torch
        from torch import nn
        from unittest.mock import patch
        from transformers import BertConfig
        from opmd_common import build_model, compute_loss

        class TinyResNet(nn.Module):
            # Lightweight stand-in for ResNet50; fusion/loss/BERT paths are real.
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 2048, 1)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.flatten = nn.Flatten(1)
                self.fc = nn.Linear(2048, 1000)

            def forward(self, x):
                return self.fc(self.flatten(self.pool(self.conv(x))))

        with tempfile.TemporaryDirectory() as tmp, patch("opmd_common.models.resnet50", side_effect=lambda **_: TinyResNet()):
            BertConfig(vocab_size=20, hidden_size=32, num_hidden_layers=1,
                       num_attention_heads=4, intermediate_size=64).save_pretrained(tmp)
            args = build_argparser("test").parse_args(["--bert_path", tmp, "--dim", "32", "--heads", "4"])
            batch = {"wli": torch.rand(2, 3, 16, 16), "afi": torch.rand(2, 3, 16, 16),
                     "ids": torch.ones(2, 6, dtype=torch.long), "mask": torch.ones(2, 6, dtype=torch.long)}
            y5, y2 = torch.tensor([0, 3]), torch.tensor([0., 1.])
            for kind in MODEL_SPECS:
                with self.subTest(kind=kind):
                    model = build_model(kind, args, pretrained=False)
                    model.train()
                    out = model(batch)
                    self.assertEqual(tuple(out["log5"].shape), (2, 5))
                    self.assertEqual(tuple(out["log2"].shape), (2,))
                    self.assertEqual("log_ord" in out, kind in ["cross_attention_ordinal", "full"])
                    loss, parts = compute_loss(kind, out, y5, y2, torch.ones(5), 0, ["a", "b"], args)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertIsNotNone(model.cls5.weight.grad)
                    self.assertTrue(all(not p.requires_grad for p in model.txt.bert.parameters()))
                    if kind == "full":
                        self.assertAlmostEqual(parts["contrastive_lambda_effective"], .004)
                    restored = build_model(kind, args, pretrained=False)
                    restored.load_state_dict(model.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
