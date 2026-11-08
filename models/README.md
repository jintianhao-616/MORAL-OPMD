# Five model checkpoints

[Download the five folders](https://drive.google.com/drive/folders/1ulnSnaLbFlO7_KqerZzFEHOLW2msXOAd)
and place the numbered folders directly under `models/`.

| Model | Folder | Download |
| --- | --- | --- |
| 1 | `01_baseline_concat` | [Google Drive](https://drive.google.com/drive/folders/1VjxiKc2NZifzlEzzWiOIjKm_Sgw__bvl) |
| 2 | `02_mlp_fusion` | [Google Drive](https://drive.google.com/drive/folders/1PnjegR1RZJGF6SmHfP_0LYEzIiyZN7I5) |
| 3 | `03_cross_attention` | [Google Drive](https://drive.google.com/drive/folders/107reNXyOPpUwNa_eKDluHeHn2RiQ3a8P) |
| 4 | `04_cross_attention_ordinal` | [Google Drive](https://drive.google.com/drive/folders/1huhpVFPqK2AyDHN-bZJVqMg0i_3IFRUw) |
| 5 | `05_full_cross_attention_ordinal_contrastive_warmup` | [Google Drive](https://drive.google.com/drive/folders/1scbg7H2DirkktIKjY8v22PBSlUBu2jCd) |

Every folder contains `best_patient_mean_macro_f1.pt`. Preserve the five folders
so the identically named files retain distinct complete paths.

Run `python eval_all_models.py` to evaluate the five models on the unchanged
`data/sample.csv` and images under `data/sample_images/`. Results are written to
`eval_outputs/five_models/`, including `five_model_summary.csv`.

The weights are hosted separately on Drive. `checkpoints.json` records the
observed links and file sizes. The code update verified folder listings but
did not load the trained checkpoint contents.
