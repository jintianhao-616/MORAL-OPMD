# Model 2: 02_mlp_fusion

[Download this checkpoint](https://drive.google.com/drive/folders/1PnjegR1RZJGF6SmHfP_0LYEzIiyZN7I5).
Keep the filename `best_patient_mean_macro_f1.pt` inside this directory.

Local checkpoint path:
`models/02_mlp_fusion/best_patient_mean_macro_f1.pt`.

Run `python eval_all_models.py` from the repository to test the five numbered
folders on the original `data/sample.csv` and `data/sample_images/`.
To test this model alone, use:

```bash
python eval_moral_opmd.py --checkpoint models/02_mlp_fusion/best_patient_mean_macro_f1.pt
```

The dataset files are read without modification. A matching BERT configuration
and tokenizer are required; pass `--bert_path` for a local model folder.
