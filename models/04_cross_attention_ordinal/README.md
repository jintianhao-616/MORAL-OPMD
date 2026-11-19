# Model 4: Cross-attention with ordinal/hierarchical objectives

[Download **M4_weights.pt**](https://drive.google.com/file/d/1QRD-homu06JWshibfqXUqCxSWFIHUoIG/view?usp=drivesdk) · [Drive folder](https://drive.google.com/drive/folders/1huhpVFPqK2AyDHN-bZJVqMg0i_3IFRUw)

Place the file here without changing its name:

```text
models/04_cross_attention_ordinal/M4_weights.pt
```

Evaluate this model on the unchanged demonstration data:

```bash
python eval_moral_opmd.py --checkpoint models/04_cross_attention_ordinal/M4_weights.pt
```

The filename identifies `cross_attention_ordinal`. For a renamed file, explicitly pass
`--model_kind cross_attention_ordinal`. The model is loaded strictly from the pure
parameter dictionary; no original training config or experiment folder is read.
Use `--bert_path /path/to/bert-base-chinese` for a matching offline BERT
configuration and tokenizer. The public defaults are dim=512, heads=8,
size=224 and max_text_len=64. This weight file does not record an epoch.

To evaluate all five files together, use `python eval_all_models.py`.
Observed Drive file size: **660,582,620 bytes**. Parameter contents and
real inference were not checked during the repository update.
