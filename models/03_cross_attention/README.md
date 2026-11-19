# Model 3: Cross-attention fusion

[Download **M3_weights.pt**](https://drive.google.com/file/d/1DAROyskK845muTC3pZNhCFZ2BP4ZiiU8/view?usp=drivesdk) · [Drive folder](https://drive.google.com/drive/folders/107reNXyOPpUwNa_eKDluHeHn2RiQ3a8P)

Place the file here without changing its name:

```text
models/03_cross_attention/M3_weights.pt
```

Evaluate this model on the unchanged demonstration data:

```bash
python eval_moral_opmd.py --checkpoint models/03_cross_attention/M3_weights.pt
```

The filename identifies `cross_attention`. For a renamed file, explicitly pass
`--model_kind cross_attention`. The model is loaded strictly from the pure
parameter dictionary; no original training config or experiment folder is read.
Use `--bert_path /path/to/bert-base-chinese` for a matching offline BERT
configuration and tokenizer. The public defaults are dim=512, heads=8,
size=224 and max_text_len=64. This weight file does not record an epoch.

To evaluate all five files together, use `python eval_all_models.py`.
Observed Drive file size: **660,573,856 bytes**. Parameter contents and
real inference were not checked during the repository update.
