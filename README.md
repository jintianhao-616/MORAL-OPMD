# MORAL-OPMD: Multimodal Risk-Aware Learning for Oral Potentially Malignant Disorders

## Overview

**MORAL-OPMD** is a multimodal deep learning framework for assessing **oral potentially malignant disorders (OPMDs)** by integrating:

* White-light imaging (WLI)
* Autofluorescence imaging (AFI)
* Structured clinical text

The model supports:

* **Five-class oral epithelial dysplasia (OED) grading (0–4)**
* **Binary risk stratification (low vs high risk)**

This repository provides **training and evaluation code aligned with the manuscript**, with a focus on **patient-level clinical evaluation**.

---

## Repository Structure

```bash
MORAL-OPMD/
│
├── bert-base-chinese/   # (optional local BERT model)
│
├── data/
│   ├── sample.csv
│   └── sample_images/
│
├── models/
│   ├── baseline/
│   └── optimized/
│
├── eval_outputs/   # optional
│
├── train_baseline_moral_opmd.py
├── train_optimized_moral_opmd.py
├── eval_baseline.py
├── eval_optimized.py
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Task Definition

### 1. Five-class OED grading

| Label | Description       |
| ----- | ----------------- |
| 0     | No dysplasia      |
| 1     | Mild              |
| 2     | Moderate          |
| 3     | Severe            |
| 4     | Carcinoma in situ |

---

### 2. Binary risk stratification

* Low risk: OED < 2
* High risk: OED ≥ 2

---

## Dataset Format

Due to privacy and ethical restrictions, the original clinical dataset is not publicly available.
A small **sample dataset** is provided for demonstration.

```csv
id,whitelight_image,fluorescent_image,Lesion location,Clinical diagnosis,label_5cls,label_2cls
001,data/sample_images/001_wli_1.jpg,data/sample_images/001_afi_1.jpg,left ventral tongue,oral leukoplakia,2,1
```

---

## Data Structure (Important)

Each patient may have:

* multiple WLI images (n)
* multiple AFI images (m)

Pairs are constructed using a **many-to-many (n × m) strategy**.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## BERT Model Setup

We use the pretrained Chinese BERT model.

### Option 1 (Recommended)

Automatically download from HuggingFace:

```bash
--bert_path bert-base-chinese
```

---

### Option 2 (Offline / Restricted network)

If automatic download is not available, you can manually download the model:

https://drive.google.com/drive/folders/1DubroiWinQUGrqyyMURQnqo3uoW3COsS

After downloading, use:

```bash
--bert_path ./bert-base-chinese
```

---

## Quick Demo (Recommended)

Run the following command:

```bash
python eval_baseline.py \
  --test_csv data/sample.csv \
  --model_path models/baseline/best_5cls.pt \
  --bert_path bert-base-chinese \
  --output_dir eval_outputs/demo
```

This will output example evaluation metrics.

**Note:** The sample dataset is for demonstration only and does not reflect full model performance.

If you encounter issues with model loading, please ensure that the evaluation script matches the model type (baseline vs optimized).

---

## Additional Evaluation Modes

### Baseline (2-class)

```bash
python eval_baseline.py \
  --test_csv data/sample.csv \
  --model_path models/baseline/best_2cls.pt \
  --bert_path bert-base-chinese \
  --output_dir eval_outputs/baseline_2cls
```

---

### Optimized (5-class)

```bash
python eval_optimized.py \
  --test_csv data/sample.csv \
  --model_path models/optimized/best_5cls.pt \
  --bert_path bert-base-chinese \
  --output_dir eval_outputs/optimized_5cls
```

---

### Optimized (2-class)

```bash
python eval_optimized.py \
  --test_csv data/sample.csv \
  --model_path models/optimized/best_2cls.pt \
  --bert_path bert-base-chinese \
  --output_dir eval_outputs/optimized_2cls
```

---

## Evaluation Protocol

### Patient-level evaluation

All results are computed at the **patient level**.

---

### Aggregation strategy

* 5-class: majority voting
* 2-class: max-risk

If any image pair is predicted high-risk → patient is high-risk.

---

## Metrics

* Accuracy
* Sensitivity
* Specificity
* Macro-F1

---

## Model Weights

Download from:

https://drive.google.com/drive/folders/1ulnSnaLbFlO7_KqerZzFEHOLW2msXOAd

Place under:

```bash
models/
 ├── baseline/
 └── optimized/
```

The folder structure matches the repository layout.

---

## Important Notes

### No data leakage

Only the following fields are used as input:

* Lesion location
* Clinical diagnosis

---

### Demo limitation

The sample dataset is for demonstration only and does not reflect real performance.

---

### Model compatibility

Ensure:

```text
eval_baseline.py  ↔ baseline model  
eval_optimized.py ↔ optimized model
```

---

## License

MIT License

---

## Citation

(To be updated after publication)

---

## Contact

For inquiries:

[drshilinjun@126.com](mailto:drshilinjun@126.com)
