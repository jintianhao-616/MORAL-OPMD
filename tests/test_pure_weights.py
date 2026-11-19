"""Public pure-weight loading without private training configs or directories."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from opmd_cli import MODEL_SPECS, MODEL_WEIGHT_FILENAMES, INFERENCE_DEFAULTS
from opmd_checkpoint import model_kind_from_filename, resolve_checkpoint
from eval_all_models import build_parser, build_commands, main as batch_main, write_summary, SUMMARY_FIELDS
from eval_moral_opmd import main as evaluate_one
from test_workflows import example_rows

ROOT = Path(__file__).resolve().parents[1]


class PureIdentityTests(unittest.TestCase):
    def test_all_published_filenames_load_without_private_metadata(self):
        for number, kind in enumerate(MODEL_SPECS, 1):
            with self.subTest(kind=kind):
                path = Path('/unrelated/download') / f'M{number}_weights.pt'
                self.assertEqual(model_kind_from_filename(path), kind)
                state = {'cls5.weight': object()}
                actual, restored, cfg, original = resolve_checkpoint(state, checkpoint_path=path)
                self.assertEqual(actual, kind)
                self.assertIs(restored, state)
                self.assertEqual(original, {})
                self.assertEqual(vars(cfg), INFERENCE_DEFAULTS)

    def test_renamed_weights_require_model_identity(self):
        with self.assertRaisesRegex(ValueError, '--model_kind'):
            resolve_checkpoint({'x': 1}, checkpoint_path='renamed.pt')
        kind, _, _, _ = resolve_checkpoint({'x': 1}, 'full', checkpoint_path='renamed.pt')
        self.assertEqual(kind, 'full')

    def test_filename_and_explicit_or_recorded_kind_must_agree(self):
        for number, kind in enumerate(MODEL_SPECS, 1):
            wrong = 'full' if kind != 'full' else 'baseline_concat'
            with self.assertRaisesRegex(ValueError, 'Filename'):
                resolve_checkpoint({'x': 1}, wrong, checkpoint_path=f'M{number}_weights.pt')
            with self.assertRaisesRegex(ValueError, 'Filename'):
                resolve_checkpoint({'model': {'x': 1}, 'model_kind': wrong},
                                   checkpoint_path=f'M{number}_weights.pt')

    def test_flat_layout_and_custom_inference_arguments(self):
        args = build_parser().parse_args(['--weights_layout', 'flat', '--weights_dir', 'a path',
                                         '--dim', '384', '--heads', '6', '--size', '192',
                                         '--max_text_len', '48'])
        for kind, _, path, command in build_commands(args):
            self.assertEqual(path, Path('a path') / MODEL_WEIGHT_FILENAMES[kind])
            for flag, value in [('--dim', '384'), ('--heads', '6'), ('--size', '192'), ('--max_text_len', '48')]:
                self.assertEqual(command[command.index(flag) + 1], value)

    def test_old_filenames_are_not_silently_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for _, folder in MODEL_SPECS.values():
                (directory / folder).mkdir()
                (directory / folder / 'best_patient_mean_macro_f1.pt').write_bytes(b'old')
            with self.assertRaisesRegex(FileNotFoundError, 'M1_weights.pt'):
                batch_main(['--weights_dir', str(directory)])

    def test_manifest_documents_all_public_defaults_and_distinct_links(self):
        manifest = json.loads((ROOT / 'models/checkpoints.json').read_text())
        self.assertEqual(manifest['weights_format'], 'pure_state_dict')
        self.assertEqual(manifest['inference_defaults'], INFERENCE_DEFAULTS)
        self.assertEqual(len({m['file_id'] for m in manifest['models']}), 5)
        for m in manifest['models']:
            self.assertEqual(m['filename'], MODEL_WEIGHT_FILENAMES[m['model_kind']])
            self.assertTrue(m['public_reader_access'])
            for text in [(ROOT / 'README.md').read_text(), (ROOT / 'models/README.md').read_text(),
                         (ROOT / 'models' / m['folder'] / 'README.md').read_text()]:
                self.assertIn(m['file_url'], text)


class PureEvaluationIntegrationTests(unittest.TestCase):
    def test_five_pure_files_evaluate_with_no_experiments_or_config_json(self):
        """Mock only tensor/network IO; run actual resolver, evaluator and metrics."""
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv = root / 'cohort.csv'
            example_rows().to_csv(csv, index=False)
            output = root / 'results'
            raw = pd.DataFrame({'pid': ['001', '001', '002'], 'row_index': [0, 1, 2],
                                'y5': [2, 2, 0], 'y2': [1, 1, 0],
                                'p5': [json.dumps([0, 0, 1, 0, 0])] * 2 + [json.dumps([1, 0, 0, 0, 0])],
                                'p2': [.9, .8, .1]})
            class Model:
                def load_state_dict(self, state, strict=False):
                    if not strict or 'cls5.weight' not in state:
                        raise AssertionError('Strict loading is required')
                def to(self, device): return self
            class Dataset:
                def __init__(self, *args, **kwargs):
                    if kwargs['train'] is not False: raise AssertionError('Evaluation must disable augmentation')
            common = types.ModuleType('opmd_common')
            common.OPMDDataset = Dataset
            def build(kind, cfg, pretrained=True):
                self.assertFalse(pretrained)
                self.assertEqual(vars(cfg), INFERENCE_DEFAULTS)
                calls.append(kind)
                return Model()
            common.build_model = build
            common.predict = lambda *args: raw.copy()
            torch = types.ModuleType('torch')
            torch.is_tensor = lambda value: isinstance(value, np.ndarray)
            torch.cuda = types.SimpleNamespace(is_available=lambda: False)
            torch.device = lambda *args: types.SimpleNamespace(type='cpu')
            def load(path, map_location=None, weights_only=False):
                self.assertTrue(weights_only)
                self.assertEqual(map_location, 'cpu')
                self.assertIn(Path(path).name, MODEL_WEIGHT_FILENAMES.values())
                return {'cls5.weight': np.zeros((5, 512))}
            torch.load = load
            loader = types.ModuleType('torch.utils.data')
            loader.DataLoader = lambda *args, **kwargs: object()
            patches = {'torch': torch, 'torch.utils': types.ModuleType('torch.utils'),
                       'torch.utils.data': loader, 'opmd_common': common}
            with patch.dict(sys.modules, patches):
                for kind, (_, folder) in MODEL_SPECS.items():
                    weight = root / MODEL_WEIGHT_FILENAMES[kind]
                    weight.write_bytes(b'unchanged exported weights')
                    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
                    evaluate_one(['--checkpoint', str(weight), '--csv', str(csv),
                                  '--output_dir', str(output / folder)])
                    self.assertEqual(digest, hashlib.sha256(weight.read_bytes()).hexdigest())
                    metrics = json.loads((output / folder / 'metrics.json').read_text())
                    self.assertEqual(metrics['model_kind'], kind)
                    self.assertEqual(metrics['weights_format'], 'pure_state_dict')
                    self.assertIsNone(metrics['epoch'])
                    self.assertEqual(metrics['patient_mean_prob']['acc5'], 1)
            self.assertEqual(calls, list(MODEL_SPECS))
            table = pd.read_csv(write_summary(output))
            self.assertEqual(list(table.columns), SUMMARY_FIELDS)
            self.assertTrue(table.best_epoch.isna().all())
            self.assertTrue(table.patient_acc5.eq(1).all())
            self.assertFalse((root / 'experiments').exists())
            self.assertFalse(list(root.rglob('config.json')))

    def test_evaluation_uses_restricted_load_and_strict_restore(self):
        tree = ast.parse((ROOT / 'eval_moral_opmd.py').read_text())
        loads = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and isinstance(n.func.value, ast.Name) and n.func.value.id == 'torch' and n.func.attr == 'load']
        self.assertEqual(len(loads), 1)
        self.assertTrue(any(k.arg == 'weights_only' and isinstance(k.value, ast.Constant) and k.value.value is True
                            for k in loads[0].keywords))


if __name__ == '__main__':
    unittest.main()
