# MORAL-OPMD: Multimodal Risk-Aware Learning for Oral Potentially Malignant Disorders

## Overview

**MORAL-OPMD** is a multimodal deep learning framework for assessing **oral potentially malignant disorders (OPMDs)** by integrating:

* White-light imaging (WLI)
* Autofluorescence imaging (AFI)
* Structured clinical text

The model performs:

* **Five-class oral epithelial dysplasia (OED) grading (0–4)**
* **Binary risk stratification (low vs high risk)**

This repository provides **training and evaluation code aligned with the manuscript**, supporting reproducible experiments under a **patient-level clinical setting**.

---

## Repository Structure

```bash
gitihub/
│
├── data/
│   ├── sample.csv
│   └── sample_images/
│
├── models/
│   ├── baseline/
│   │    ├── best_2cls.pt
│   │    └── best_5cls.pt
│   └── optimized/
│        ├── best_2cls.pt
│        └── best_5cls.pt
│
├── eval_outputs/   # (optional, can be empty)
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
| 4     | Carcinoma |

---

### 2. Binary risk stratification

* Low risk: OED < 2
* High risk: OED ≥ 2

---

## Dataset Format

Due to privacy and ethical restrictions, the original clinical dataset is not publicly available.
A sample dataset is provided to illustrate the required format:

```csv
id,whitelight_image,fluorescent_image,Lesion location,Clinical diagnosis,label_5cls,label_2cls
001,data/sample_images/001_wli_1.jpg,data/sample_images/001_afi_1.jpg,left ventral tongue,oral leukoplakia,2,1
```

### Field Description

* `id`: patient identifier (used for patient-level aggregation)
* `whitelight_image`: WLI image path
* `fluorescent_image`: AFI image path
* `Lesion location`: anatomical site
* `Clinical diagnosis`: clinical diagnosis (full English term)
* `label_5cls`: OED grade
* `label_2cls`: binary risk label

---

## Data Structure (Important)

Each patient may have:

* multiple WLI images (n)
* multiple AFI images (m)

Pairing is constructed using a **many-to-many (n × m) strategy**, forming multiple image pairs per patient.

---

## Installation

```bash
pip install -r requirements.txt
```

---

# Training (demo with sample data)
python train_baseline_moral_opmd.py \
  --train_csv data/sample.csv \
  --val_csv data/sample.csv \
  --bert_path bert-base-chinese \
  --output_dir ./outputs_baseline

# Evaluation
python eval_baseline.py \
  --test_csv data/sample.csv \
  --model_path models/baseline/best_5cls.pt \
  --bert_path bert-base-chinese \
  --output_dir eval_outputs/baseline

---

## Evaluation Protocol (Clinical Setting)

### Patient-level evaluation

All results are computed at the **patient level**, not image level.

---

### Aggregation strategy

* **5-class task**: majority voting across image pairs
* **2-class task**: max-risk aggregation

  * If any image pair is predicted as high-risk → patient is classified as high-risk

---

## Metrics

Reported metrics include:

* Accuracy
* Sensitivity (recall for high-risk cases)
* Specificity
* Macro-F1 score

---

## Model Weights

* `models/baseline/best_5cls.pt`: main model for 5-class classification
* `models/baseline/best_2cls.pt`: binary classification head
* `models/optimized/`: optimized model counterparts

---

## Important Notes

### 1. No data leakage

Only the following fields are used as model input:

* Lesion location
* Clinical diagnosis

Pathological labels are **not included in the input text**.

---

### 2. Sensitivity interpretation

Under the conservative aggregation strategy:

* Zero false negatives may occur
* This reflects **clinical prioritization of sensitivity**, not perfect classification

---

### 3. Data availability

The original clinical images are **not publicly available** due to privacy and ethical restrictions.

---

## License

This project is released under the MIT License.

---

## Citation

If you use this work, please cite:

```
(Your paper citation here)
```

---

## Contact

For questions, please contact:
(Your email)
