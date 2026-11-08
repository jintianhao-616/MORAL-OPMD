# MORAL-OPMD

MultimOdal Risk-Aware Learning for Oral Potentially Malignant Disorders (OPMDs).
Five multimodal architectures combine white-light images (WLI), autofluorescence
images (AFI), and pre-biopsy clinical information. Each model predicts five-class
histopathological grading and binary risk.

## Test using the existing data

Keep the repository's existing **data/** directory exactly as it is:

- **data/sample.csv**: original CSV, including its English headers and clinical text.
- **data/sample_images/**: original WLI and AFI photographs.

The evaluator maps headers internally. It does not rewrite the CSV, rename the
images, change labels, or make new train/test splits.

1. Install dependencies.
2. Download the five checkpoint folders listed below and place them directly
   under **models/**, keeping their names.
3. From the repository, run:

~~~bash
pip install -r requirements.txt
python eval_all_models.py
~~~

The command defaults to **data/sample.csv** and resolves its image paths relative
to the repository. This also works when the script is launched from another
working directory. All five models use the same original data.

BERT configuration and tokenizer files default to **google-bert/bert-base-chinese**.
For a complete local BERT folder, use:

~~~bash
python eval_all_models.py --bert_path ./bert-base-chinese
~~~

The bert-base-chinese/ directory in this repository contains instructions only
until the corresponding files have been downloaded. Evaluation restores the
trained BERT and ResNet parameters from each checkpoint.

## Five models and checkpoints

Download from the author's
[Google Drive model folder](https://drive.google.com/drive/folders/1ulnSnaLbFlO7_KqerZzFEHOLW2msXOAd).

| Model | Training entry | Architecture | Checkpoint folder |
| --- | --- | --- | --- |
| 1 | 01_baseline_concat.py | Global feature concatenation | 01_baseline_concat |
| 2 | 02_mlp_fusion.py | Multilayer-perceptron fusion | 02_mlp_fusion |
| 3 | 03_cross_attention.py | Cross-attention fusion | 03_cross_attention |
| 4 | 04_cross_attention_ordinal.py | Cross-attention with ordinal, monotonicity and hierarchical constraints | 04_cross_attention_ordinal |
| 5 | 05_full_cross_attention_ordinal_contrastive.py | Full MORAL-OPMD with patient-aware contrastive alignment | 05_full_cross_attention_ordinal_contrastive_warmup |

Every folder contains **best_patient_mean_macro_f1.pt**. Keep each file inside
its own numbered folder. For example, Model 5 is loaded from:

~~~text
models/05_full_cross_attention_ordinal_contrastive_warmup/best_patient_mean_macro_f1.pt
~~~

The five checkpoints are separate files despite their identical filenames.
Their complete paths distinguish the models. Each checkpoint supports both
output tasks through the corresponding model's classification heads.

Direct folder links and observed file sizes are recorded in
[models/README.md](models/README.md) and
[models/checkpoints.json](models/checkpoints.json). Checkpoint files are hosted
separately on Drive and are not included in the code archive.

## Evaluation outputs

The default results directory is **eval_outputs/five_models/**:

| Output | Contents |
| --- | --- |
| Numbered model directories | Image-pair predictions, patient predictions, confusion matrices and metrics |
| five_model_summary.csv | One result row per model |
| evaluation_config.json, within each model directory | Effective data path, checkpoint and evaluation settings |

The batch evaluator explicitly matches every folder to its architecture. It
uses the same patient aggregation policy (**mean_prob**) and pair policy (**all**)
for all five models by default. Binary predictions use a fixed threshold of 0.5.
It does not use the sample results to select a checkpoint or tune a threshold.

Inspect the five commands without loading weights:

~~~bash
python eval_all_models.py --dry_run
~~~

Evaluate only Model 5 on the same existing data:

~~~bash
python eval_moral_opmd.py --checkpoint models/05_full_cross_attention_ordinal_contrastive_warmup/best_patient_mean_macro_f1.pt
~~~

Optional arguments include --gpu, --batch, --workers, --bert_path, --output_dir,
--aggregation and --pair_policy. Worker count defaults to 0 for the small sample
and Windows compatibility. --gpu 1 selects physical GPU 1 before torch loads;
--gpu -1 selects CPU. Existing result directories require a new --output_dir
or explicit --overwrite.

To evaluate a separate cohort, pass --csv /path/to/cohort.csv. This is an optional
override; the normal demonstration command uses the unchanged data/sample.csv.

## Data and tasks

The supplied sample CSV has 9 image-pair rows from 4 patients. It retains these
original columns:

| Column | Meaning |
| --- | --- |
| id | Patient identifier |
| whitelight_image | WLI image path |
| fluorescent_image | AFI image path |
| Lesion location | Pre-biopsy lesion site |
| Clinical diagnosis | Pre-biopsy clinical diagnosis |
| label_5cls | Histopathological target |
| label_2cls | Binary risk target |

Chinese training-header aliases are also supported internally. IDs are read as
text. Labels must be consistent within each patient, and the binary label must
equal (label_5cls >= 2). Histopathological targets never enter the clinical prompt;
the prompt uses lesion site and clinical diagnosis only.

| Five-class target | Category | Binary risk |
| --- | --- | --- |
| 0 | No dysplasia | Low, 0 |
| 1 | Mild OED | Low, 0 |
| 2 | Moderate OED | High, 1 |
| 3 | Severe OED | High, 1 |
| 4 | Carcinoma / OSCC | High, 1 |

The sample is a functional demonstration rather than a clinical validation
cohort. It contains reference grades 0 and 2 only; metrics requiring all five
reference classes, including macro five-class ROC-AUC, are undefined and saved
as JSON null.

Metrics include accuracy, balanced accuracy, five-class Macro-F1 and weighted F1,
per-grade precision/recall/F1, five-class macro OvR AUC, binary ROC-AUC, average
precision, sensitivity, specificity, PPV and NPV. Macro-F1 uses the fixed task
label set. pr_auc2 denotes average precision. The primary patient-level policy
is mean_prob. Additional saved aggregation policies are max_prob, max_risk and
majority_max_risk; the latter resolves vote ties toward the lower grade.
The primary policy is recorded explicitly.

## Files

| File | Purpose |
| --- | --- |
| 00_check_cuda.py | Check CUDA |
| 01_*.py through 05_*.py | The five architecture training entries |
| train_all_models.py | Train all five architectures sequentially |
| opmd_common.py | Encoders, fusion modules, losses and training loop |
| opmd_cli.py | Registry and training arguments |
| opmd_data.py | Schema mapping, patient checks and image paths |
| opmd_checkpoint.py | Checkpoint configuration and architecture validation |
| opmd_metrics.py | Aggregation and metrics |
| eval_moral_opmd.py | Evaluate one of the five models |
| eval_all_models.py | Evaluate all five models on the existing data |
| check_data.py | Check CSV and image paths |
| tests/test_workflows.py | Workflow tests |

## Training the five architectures

The downloaded weights can be tested directly; retraining is not needed for
the data/sample.csv demonstration. To train on the research cohort with its
existing patient-disjoint splits:

~~~bash
python train_all_models.py --root /path/to/project --train_csv /path/train.csv --val_csv /path/val.csv --bert_path /path/to/bert-base-chinese --gpu 0
~~~

Run any numbered entry with --help for the full argument list. Training defaults
retain the supplied encoders, augmentation, losses and validation checkpoint
criterion. The selected checkpoint maximizes validation patient mean-probability
five-class Macro-F1. Test data are not used for training or checkpoint selection.

Model 5 uses the supplied warmup configuration: zero-based contrastive start
epoch 0, five warmup epochs and target weight 0.02. The effective weight rises
from 0.004 to 0.020 over epochs 0–4. The schedule is saved in config and history.
Models 4 and 5 share an inference architecture; Model 5 adds a training objective.

## Checks and environment

Python 3.10–3.12; the dependency file pins torch 2.3.1 and torchvision 0.18.1.
Retain the appropriate CUDA build for your machine.

~~~bash
python check_data.py
python check_data.py --check_images
python -m unittest discover -s tests -v
~~~

The code update was checked without torch, torchvision or transformers installed.
CSV parsing, default data paths, command generation, metadata and metrics are
tested. The optional forward/backward smoke test needs the model libraries and
uses a small backbone stand-in. Real checkpoint loading and inference on the
photographs were not run in the update environment.

## License and contact

MIT License; see LICENSE. Publication details will be added when available.
Contact: [drshilinjun@126.com](mailto:drshilinjun@126.com).
