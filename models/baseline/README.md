# Legacy baseline checkpoints

This directory is retained for checkpoints from the earlier two-model
implementation. Its corresponding old evaluator is `eval_baseline.py`.

The shared [Google Drive folder](https://drive.google.com/drive/folders/1ulnSnaLbFlO7_KqerZzFEHOLW2msXOAd)
now contains five new architecture folders, each with
`best_patient_mean_macro_f1.pt`. Those checkpoints require `eval_moral_opmd.py`
or `eval_all_models.py`, not this legacy evaluator. See the parent `models/README.md`.

Keep any old weights separately if you still need to reproduce the old model.
Do not relabel old checkpoints as the new models, or use new checkpoints with
old architecture definitions.
