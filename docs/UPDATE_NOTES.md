# Five-model release

The active project contains exactly five numbered architecture entries and one
shared five-model evaluation workflow. The migration removes the obsolete
standalone evaluation/training entries and superseded model-folder placeholders.
No archived copies of their source are distributed in the update package.

## Existing data

`data/sample.csv` and `data/sample_images/` are retained unchanged. They are not
included in the update layer and are never copied, renamed or deleted by the
installer. Both evaluators default to the original sample CSV. Relative paths
inside that file resolve against the repository even when the command runs
from another working directory. English column names are mapped in memory;
no data conversion or rewrite is performed.

The existing sample has 9 image pairs from 4 patients. Five-class reference
labels include 0 and 2; metrics requiring all five reference classes are null.

## Weights and evaluation

The original Drive download URL hosts the five numbered folders. Every folder
contains `best_patient_mean_macro_f1.pt`. The registry matches those folder names
exactly. The default `python eval_all_models.py` command runs all five on the
same unchanged data and writes one comparison row per model.

The batch evaluation defaults to `all` pairs and `mean_prob` aggregation for
every model, with a fixed binary threshold of 0.5. Checkpoints are selected
before this evaluation. BERT config/tokenizer defaults to the explicit
`google-bert/bert-base-chinese` model ID; a local directory can be supplied.
The small demonstration defaults to zero loader workers.

## Training

The five architectures, augmentations, forward paths, losses and checkpoint
criterion remain based on the supplied numbered/common training scripts.
Model 5 retains the supplied five-epoch contrastive warmup configuration.
The data demonstration does not require retraining.

## Validation limits

Data/default-path/CLI/aggregation/metadata/summary checks and syntax compilation
run without GPU libraries. The optional model smoke test is skipped without
torch, torchvision and transformers. The update environment did not run real
checkpoint loading or inference on the original photographs. The installer
and archive were checked to preserve the original dataset and remove only the
specified obsolete code files.
