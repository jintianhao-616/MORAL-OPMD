# MORAL-OPMD

MultimOdal Risk-Aware Learning for Oral Potentially Malignant Disorders (OPMDs).
Five multimodal architectures combine white-light images (WLI), autofluorescence
images (AFI), and pre-biopsy clinical information. Each model predicts five-class
histopathological grading and binary risk.

## Quick start with the five pure weight files

1. Install dependencies: `pip install -r requirements.txt`.
2. Download **M1_weights.pt through M5_weights.pt** from the individual links below.
3. Put each file in its matching numbered directory under `models/`, preserving
   the downloaded filename.
4. Run:

```bash
python eval_all_models.py
```

This evaluates all five models on the existing **data/sample.csv** and
**data/sample_images/**. These data are read without rewriting headers, labels,
paths or images. The CSV and default paths also work when the evaluator is
launched from another working directory.

The public `.pt` files are **pure parameter/buffer dictionaries**. Evaluation
uses the model definitions and public inference defaults in this repository;
**no training `experiments/` directory, original checkpoint or training
config.json is needed**. A BERT configuration and tokenizer are still required,
as described below. They are different from a private training configuration.

## Model downloads

[All five Drive folders](https://drive.google.com/drive/folders/1ulnSnaLbFlO7_KqerZzFEHOLW2msXOAd). Each direct link below
points to a separate file; do not substitute one model's weights for another.

| Model | Weight file | Architecture | Direct file link |
| --- | --- | --- | --- |
| M1 | `M1_weights.pt` | Global feature concatenation | [Download](https://drive.google.com/file/d/11buhRsn_bTiOQXvxPQN3yHeYTE7x08uu/view?usp=drivesdk) |
| M2 | `M2_weights.pt` | Multilayer-perceptron fusion | [Download](https://drive.google.com/file/d/1FI0_jVBPl6Z2nQWvjY_53f17IhXKLeAX/view?usp=drivesdk) |
| M3 | `M3_weights.pt` | Cross-attention fusion | [Download](https://drive.google.com/file/d/1DAROyskK845muTC3pZNhCFZ2BP4ZiiU8/view?usp=drivesdk) |
| M4 | `M4_weights.pt` | Cross-attention with ordinal/hierarchical objectives | [Download](https://drive.google.com/file/d/1QRD-homu06JWshibfqXUqCxSWFIHUoIG/view?usp=drivesdk) |
| M5 | `M5_weights.pt` | Full MORAL-OPMD with contrastive alignment | [Download](https://drive.google.com/file/d/1hKcpSsTKB4v6S6KZK1PeYVx7JcqHOiHJ/view?usp=drivesdk) |

The default local layout is:

```text
models/
  01_baseline_concat/M1_weights.pt
  02_mlp_fusion/M2_weights.pt
  03_cross_attention/M3_weights.pt
  04_cross_attention_ordinal/M4_weights.pt
  05_full_cross_attention_ordinal_contrastive_warmup/M5_weights.pt
```

If all five files are in one directory, use the explicit flat layout instead:

```bash
python eval_all_models.py --weights_dir ./weights --weights_layout flat
```

The batch evaluator does not fall back to older, identically named training
checkpoints. It expects the five new filenames. Download links, numbered paths,
observed file sizes and public reader-access checks are recorded in
[models/checkpoints.json](models/checkpoints.json) and [models/README.md](models/README.md).
The `.pt` files are hosted on Drive, not in the Git repository.

Drive may describe a `.pt` file as a ZIP because PyTorch uses a ZIP-based storage
container. Keep it as `.pt` and pass it directly to the evaluator; do not unpack
or rename it to `.zip`.

## BERT and inference settings

The configuration/tokenizer default is **google-bert/bert-base-chinese**. The
first run needs network access or a populated local cache. For offline use,
provide a directory with the matching BERT **config.json**, **vocab.txt** and
any associated tokenizer configuration files:

```bash
python eval_all_models.py --bert_path ./bert-base-chinese
```

The checked-in `bert-base-chinese/` directory contains instructions only.
Evaluation constructs ResNet50 without ImageNet pretrained weights and BERT
from its architecture configuration, then restores all trained parameters from
the pure state_dict. Separate pretrained ResNet or BERT weight files are not
needed for evaluation. The tokenizer must match the one used by these weights.

The public inference defaults are fixed in `opmd_cli.py`:

| Setting | Value |
| --- | --- |
| Visual/fusion dimension | 512 |
| Cross-attention heads | 8 |
| Image size | 224 × 224, RGB |
| Image normalization mean | 0.485, 0.456, 0.406 |
| Image normalization standard deviation | 0.229, 0.224, 0.225 |
| Maximum token length | 64 |
| Image-pair policy | all |
| Primary patient aggregation | mean_prob |
| Binary decision threshold | 0.5 |

The bare files have no configuration object; the inference settings are
declared here. Do not select them from trial performance. CLI options `--dim`, `--heads`, `--size` and `--max_text_len` exist
for other compatible exports and must match their training protocol.

## Evaluate one model or another cohort

```bash
python eval_moral_opmd.py --checkpoint models/05_full_cross_attention_ordinal_contrastive_warmup/M5_weights.pt
```

The five exact filenames identify their corresponding architectures. A renamed
bare state_dict requires an explicit architecture, for example:

```bash
python eval_moral_opmd.py --checkpoint /path/to/renamed.pt --model_kind full
```

M4 and M5 have the same inference architecture but represent different
training objectives. Tensor shapes cannot establish which model a renamed file came
from. Keep the verified download-to-model mapping. A conflicting known filename
and `--model_kind` is rejected. All parameter keys and tensor shapes are loaded
strictly; a partial load is not used.

To evaluate a separate labeled cohort, pass `--csv /path/to/cohort.csv`, optionally
with `--image_root /path/to/images`. Both evaluators also accept `--gpu`,
`--batch`, `--workers`, `--bert_path`, `--output_dir`, `--aggregation` and
`--pair_policy`. `--gpu 1` selects physical GPU 1 before torch is imported;
`--gpu -1` selects CPU. Existing result directories require a new `--output_dir`
or explicit `--overwrite`.

Check the five expected paths and commands without loading any weights:

```bash
python eval_all_models.py --dry_run
```

## Evaluation outputs

The default directory is **eval_outputs/five_models/**:

| Output | Contents |
| --- | --- |
| Numbered model directories | Image-pair/patient predictions, confusion-matrix CSVs and metrics |
| five_model_summary.csv | One summary row for each model |
| evaluation_config.json in each model directory | Effective input, weights, preprocessing and aggregation |

The same data, all-pairs policy and mean-probability patient aggregation are
used for every model by default. Other exported patient aggregations are
`max_prob`, `max_risk` and `majority_max_risk` (vote ties favor the lower grade).
Binary predictions use 0.5; the evaluator does not select weights or tune the
threshold using the evaluation results.

Pure weights do not retain a training epoch: the `epoch`/`best_epoch` output is
null/blank for these files. It is not guessed from parameters. Evaluation output
files record the supplied evaluation data/settings; they are separate from the
unchanged pure weight files.

## Data and tasks

The unchanged sample CSV contains 9 image pairs from 4 patients:

| Column | Meaning |
| --- | --- |
| id | Patient identifier, read as text |
| whitelight_image | WLI path |
| fluorescent_image | AFI path |
| Lesion location | Pre-biopsy lesion site |
| Clinical diagnosis | Pre-biopsy clinical diagnosis |
| label_5cls | Histopathological target |
| label_2cls | Binary risk target |

Chinese training-header aliases are mapped in memory. Labels must be consistent
within each patient, and binary labels must equal `(label_5cls >= 2)`.
The clinical prompt uses only lesion site and clinical diagnosis, never the
histopathological targets.

| Five-class target | Category | Binary risk |
| --- | --- | --- |
| 0 | No dysplasia | Low, 0 |
| 1 | Mild OED | Low, 0 |
| 2 | Moderate OED | High, 1 |
| 3 | Severe OED | High, 1 |
| 4 | Carcinoma / OSCC | High, 1 |

This sample demonstrates functionality; it is not a clinical validation cohort
and its membership relative to any downloaded model's training data is not
established by this workflow. A new cohort must be independently checked for
training overlap before its results are called external validation. Removing
checkpoint metadata does not change the data a model was trained on.

The sample's reference grades are 0 and 2 only. Metrics needing all five classes,
including macro five-class ROC-AUC, are undefined and saved as JSON null. Metrics
include accuracy, balanced accuracy, fixed-five-class Macro-F1, weighted F1,
per-grade precision/recall/F1, macro OvR AUC, binary ROC-AUC, average precision,
sensitivity, specificity, PPV and NPV. `pr_auc2` means average precision.

## Training and code layout

| File | Purpose |
| --- | --- |
| 01_*.py through 05_*.py | The five architecture training entries |
| train_all_models.py | Train the five architectures sequentially |
| opmd_common.py | Unchanged encoders, fusion, losses and training loop |
| opmd_cli.py | Architecture registry, public filenames, inference defaults and training CLI |
| opmd_data.py | Schema, patient checks, prompts and image paths |
| opmd_checkpoint.py | Resolve a pure state_dict or a legacy configured checkpoint |
| opmd_metrics.py | Aggregation and metrics |
| eval_moral_opmd.py | Evaluate one model |
| eval_all_models.py | Evaluate all five models |
| check_data.py, 00_check_cuda.py | Check inputs / CUDA |
| tests/test_workflows.py | Workflow and optional model smoke tests |

Retraining is not required to run the demonstration. For a new research cohort
with pre-existing patient-disjoint training and validation CSVs:

```bash
python train_all_models.py --root /path/to/project --train_csv /path/train.csv --val_csv /path/val.csv --bert_path /path/to/bert-base-chinese --gpu 0
```

The training entries still save their full research checkpoint as
`best_patient_mean_macro_f1.pt` in their experiment directory. Single-model
evaluation remains compatible with such checkpoints when supplied explicitly;
the public batch command now defaults to the pure weight release.

The architecture definitions, augmentation, losses and validation checkpoint
criterion are unchanged. Selection uses validation patient mean-probability
five-class Macro-F1. M5 retains its contrastive warmup: zero-based start epoch 0,
five warmup epochs, target weight 0.02 (effective weights 0.004–0.020 in epochs
0–4). Test data are not used by the training loop for checkpoint selection.

## Checks and environment

Python 3.10–3.12; requirements pin torch 2.3.1 and torchvision 0.18.1. Install the
CUDA build appropriate for your machine.

```bash
python check_data.py
python check_data.py --check_images
python -m unittest discover -s tests -v
```

Download filenames, sizes and public reader permissions were checked. The code
update tests defaults, filename-to-model identity, CLI generation, CSV parsing,
aggregation, safe JSON and mocked evaluation with bare weight dictionaries.
The optional forward/backward smoke test needs torch, torchvision and
transformers. Real downloaded weight loading and inference on the photographs
were not run in the update environment; no numerical equivalence claim is made
from the mock tests alone.

## License and contact

MIT License; see LICENSE. Publication details will be added when available.
Contact: [drshilinjun@126.com](mailto:drshilinjun@126.com).
