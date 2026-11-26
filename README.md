# Cost-Sensitive Churn Prediction with Segment-Level Thresholds

MSc Data Analytics thesis, National College of Ireland
Ibrahim A I Hameed (x25189042)

**Title:** A Value-Segmented Churn Prediction Framework with Data-Driven KPI Development for Telecommunications Customer Retention
**Data:** IBM Telco Customer Churn (7,043 customers, 21 columns)

## Project structure

```
churn-thesis/
├── config.py              all settings and business assumptions (paths, seed, costs)
├── requirements.txt
├── run_all.py             regenerates every result end to end (to be written)
├── data/
│   ├── raw/               original CSV, never edited
│   └── processed/         train.csv / test.csv (not in Git, recreated by notebook 01)
├── notebooks/
│   ├── 01_data_preparation.ipynb   Stage 1, Part 1: cleaning, feature decisions, split
│   ├── 02_eda.ipynb                Stage 1, Part 2: exploratory data analysis
│   ├── 03_segmentation.ipynb       Stage 2: value segmentation (K-Means, k = 3)
│   └── 04_modelling.ipynb          Stage 3: 4 models x 4 training conditions x calibration
├── src/
│   ├── features.py        reusable, model-agnostic preparation functions
│   └── modelling.py       Stage 3+: profit function, weights, models, calibration, metrics, tests
├── outputs/
│   ├── figures/
│   ├── tables/            00–03 data preparation, 04–07 EDA
│   └── archive_week2/     outputs of the first exploratory version (Week 2 meeting)
└── app/                   Streamlit dashboard (later stage)
```

## Pipeline stages (as in the idea diagram)

| Stage | Content | Where | Status |
|---|---|---|---|
| 1. Data | Part 1: data preparation · Part 2: EDA (Q2 who, Q3 when, Q4 how costly) | `notebooks/01_data_preparation.ipynb`, `notebooks/02_eda.ipynb` | Done |
| 2. Segmentation | K-Means on tenure and MonthlyCharges (training set only), k = 3 | `notebooks/03_segmentation.ipynb` | Done |
| 3. Prediction | LR, RF, LightGBM, SVM; no/class/value weights, SMOTE reference; cross-fitted Platt/isotonic calibration | `notebooks/04_modelling.ipynb`, `src/modelling.py` | Done (awaiting supervisor review) |
| 4. Evaluation | Per-segment cost-based evaluation, segment-level thresholds, sensitivity analysis | — | Planned |
| 5. Act | Segment KPIs and thresholds in a Streamlit dashboard | `app/` | Planned |

## How to run

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name churn-thesis --display-name "Python (churn-thesis)"
jupyter lab
```

Open `notebooks/01_data_preparation.ipynb` and run all cells with the "Python (churn-thesis)" kernel.
