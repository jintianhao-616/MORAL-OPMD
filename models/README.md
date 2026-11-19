# Five pure model weight files

[Open the Drive collection](https://drive.google.com/drive/folders/1ulnSnaLbFlO7_KqerZzFEHOLW2msXOAd). Download the files
below and keep each in the matching directory under `models/`.

| Model | Path relative to models/ | Direct download page | Drive folder | Bytes |
| --- | --- | --- | --- | ---: |
| M1 | `01_baseline_concat/M1_weights.pt` | [File](https://drive.google.com/file/d/11buhRsn_bTiOQXvxPQN3yHeYTE7x08uu/view?usp=drivesdk) | [Folder](https://drive.google.com/drive/folders/1VjxiKc2NZifzlEzzWiOIjKm_Sgw__bvl) | 607,843,148 |
| M2 | `02_mlp_fusion/M2_weights.pt` | [File](https://drive.google.com/file/d/1FI0_jVBPl6Z2nQWvjY_53f17IhXKLeAX/view?usp=drivesdk) | [Folder](https://drive.google.com/drive/folders/1PnjegR1RZJGF6SmHfP_0LYEzIiyZN7I5) | 611,494,012 |
| M3 | `03_cross_attention/M3_weights.pt` | [File](https://drive.google.com/file/d/1DAROyskK845muTC3pZNhCFZ2BP4ZiiU8/view?usp=drivesdk) | [Folder](https://drive.google.com/drive/folders/107reNXyOPpUwNa_eKDluHeHn2RiQ3a8P) | 660,573,856 |
| M4 | `04_cross_attention_ordinal/M4_weights.pt` | [File](https://drive.google.com/file/d/1QRD-homu06JWshibfqXUqCxSWFIHUoIG/view?usp=drivesdk) | [Folder](https://drive.google.com/drive/folders/1huhpVFPqK2AyDHN-bZJVqMg0i_3IFRUw) | 660,582,620 |
| M5 | `05_full_cross_attention_ordinal_contrastive_warmup/M5_weights.pt` | [File](https://drive.google.com/file/d/1hKcpSsTKB4v6S6KZK1PeYVx7JcqHOiHJ/view?usp=drivesdk) | [Folder](https://drive.google.com/drive/folders/1scbg7H2DirkktIKjY8v22PBSlUBu2jCd) | 660,582,620 |

These are bare `state_dict` files, named `M1_weights.pt` through `M5_weights.pt`.
The public evaluator does not require the author's training config or an
`experiments/` directory. It uses the repository's explicit inference defaults
and needs a matching Chinese BERT configuration/tokenizer.

```bash
python eval_all_models.py
```

All five files can instead be placed together in a directory and evaluated with
`python eval_all_models.py --weights_dir ./weights --weights_layout flat`.
The evaluator uses the existing `data/sample.csv` and images without rewriting
them. Different filenames for M4/M5 are important: they share tensor shapes.

Observed filenames, sizes and public reader permissions are recorded in
`checkpoints.json`. The large files were not downloaded or deserialized in this
update, so these checks are not a tensor/content or performance verification.
Drive's ZIP MIME type does not mean you should extract the `.pt` file.
