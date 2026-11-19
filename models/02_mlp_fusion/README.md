# Model 2: Multilayer-perceptron fusion

[Download **M2_weights.pt**](https://drive.google.com/file/d/1FI0_jVBPl6Z2nQWvjY_53f17IhXKLeAX/view?usp=drivesdk) · [Drive folder](https://drive.google.com/drive/folders/1PnjegR1RZJGF6SmHfP_0LYEzIiyZN7I5)

Place the file here without changing its name:

```text
models/02_mlp_fusion/M2_weights.pt
```

Evaluate this model on the unchanged demonstration data:

```bash
python eval_moral_opmd.py --checkpoint models/02_mlp_fusion/M2_weights.pt
```

The filename identifies `mlp_fusion`. For a renamed file, explicitly pass
`--model_kind mlp_fusion`. The model is loaded strictly from the pure
parameter dictionary; no original training config or experiment folder is read.
Use `--bert_path /path/to/bert-base-chinese` for a matching offline BERT
configuration and tokenizer. The public defaults are dim=512, heads=8,
size=224 and max_text_len=64. This weight file does not record an epoch.

To evaluate all five files together, use `python eval_all_models.py`.
Observed Drive file size: **611,494,012 bytes**. Parameter contents and
real inference were not checked during the repository update.
