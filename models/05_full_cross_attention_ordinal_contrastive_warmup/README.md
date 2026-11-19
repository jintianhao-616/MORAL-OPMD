# Model 5: Full MORAL-OPMD with contrastive alignment

[Download **M5_weights.pt**](https://drive.google.com/file/d/1hKcpSsTKB4v6S6KZK1PeYVx7JcqHOiHJ/view?usp=drivesdk) · [Drive folder](https://drive.google.com/drive/folders/1scbg7H2DirkktIKjY8v22PBSlUBu2jCd)

Place the file here without changing its name:

```text
models/05_full_cross_attention_ordinal_contrastive_warmup/M5_weights.pt
```

Evaluate this model on the unchanged demonstration data:

```bash
python eval_moral_opmd.py --checkpoint models/05_full_cross_attention_ordinal_contrastive_warmup/M5_weights.pt
```

The filename identifies `full`. For a renamed file, explicitly pass
`--model_kind full`. The model is loaded strictly from the pure
parameter dictionary; no original training config or experiment folder is read.
Use `--bert_path /path/to/bert-base-chinese` for a matching offline BERT
configuration and tokenizer. The public defaults are dim=512, heads=8,
size=224 and max_text_len=64. This weight file does not record an epoch.

To evaluate all five files together, use `python eval_all_models.py`.
Observed Drive file size: **660,582,620 bytes**. Parameter contents and
real inference were not checked during the repository update.
