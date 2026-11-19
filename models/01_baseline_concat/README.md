# Model 1: Global feature concatenation

[Download **M1_weights.pt**](https://drive.google.com/file/d/11buhRsn_bTiOQXvxPQN3yHeYTE7x08uu/view?usp=drivesdk) · [Drive folder](https://drive.google.com/drive/folders/1VjxiKc2NZifzlEzzWiOIjKm_Sgw__bvl)

Place the file here without changing its name:

```text
models/01_baseline_concat/M1_weights.pt
```

Evaluate this model on the unchanged demonstration data:

```bash
python eval_moral_opmd.py --checkpoint models/01_baseline_concat/M1_weights.pt
```

The filename identifies `baseline_concat`. For a renamed file, explicitly pass
`--model_kind baseline_concat`. The model is loaded strictly from the pure
parameter dictionary; no original training config or experiment folder is read.
Use `--bert_path /path/to/bert-base-chinese` for a matching offline BERT
configuration and tokenizer. The public defaults are dim=512, heads=8,
size=224 and max_text_len=64. This weight file does not record an epoch.

To evaluate all five files together, use `python eval_all_models.py`.
Observed Drive file size: **607,843,148 bytes**. Parameter contents and
real inference were not checked during the repository update.
