# A Value-Segmented Churn Prediction Framework with Data-Driven KPI Development for Telecommunications Customer Retention

MSc in Data Analytics – Research Practicum, National College of Ireland
Ibrahim A I Hameed (x25189042)

## Overview

Churn models are usually judged by accuracy and used with a fixed 0.5 cut-off, which ignores customer value and campaign costs. This project builds a leakage-free framework on the public IBM Telco Customer Churn dataset that links:

1. **Value segmentation** – K-Means on tenure and monthly charge
2. **Calibrated churn prediction** – logistic regression, random forest, LightGBM, SVM and the tabular foundation model TabICL v2
3. **Profit-based decision thresholds** – one global threshold vs one threshold per segment
4. **Costed retention actions** – counterfactual explanations (DiCE)
5. **Decision KPIs** – segment KPIs validated against real outcomes

Everything that learns from data is fitted on the training set only; the test set (1,407 customers) is used once for the final evaluation. All assumptions (seed, costs, acceptance rate, thresholds) are in `config.py`.

## Main results

| | Result |
|---|---|
| Segments | Low-value (30%, churn 12.5%), new high-value (38%, churn 47%, 70% of lost revenue), established high-value (32%, churn 15.5%) |
| Models | Test AUC 0.83–0.84 for all five models; unweighted models already calibrated |
| H1 – segment thresholds | +0.65 per customer over one global threshold (95% CI −0.27 to +1.69): not supported |
| H2 – calibration | Needed only for weighted / resampled models: partly supported |
| H3 – value weighting | No profit gain: not supported |
| H4 – calibrated KPIs | KPI error 84% → 7% after calibration: supported |
| Main message | Any profit-based rule beats the 0.5 cut-off by +11 to +17 per customer; the decision rule matters more than the model |

## Project structure

```
churn-thesis/
├── config.py                 all settings and business assumptions
├── requirements.txt          Python packages
├── src/
│   ├── features.py           data cleaning, encoding, segment assignment
│   └── modelling.py          profit function, weights, models, calibration, metrics, statistical tests
├── notebooks/                one notebook per stage (run in this order)
├── data/
│   ├── raw/                  original IBM Telco CSV (never edited)
│   └── processed/            train/test files and predictions (created by the notebooks, not in Git)
├── outputs/
│   ├── tables/               result tables 00–49
│   ├── figures/              figures from the notebooks
│   └── models/               fitted segmentation, thresholds and cached results
├── report_figures.py         redraws the figures used in the report
├── dashboard/                optional Streamlit dashboard (not part of the report)
└── pretrained/               TabICL v2 checkpoint (download separately, not in Git)
```

## Notebooks

| # | Notebook | Stage | Main outputs |
|---|---|---|---|
| 01 | `01_data_preparation.ipynb` | Cleaning, feature decisions, stratified 80/20 split | `train.csv`, `test.csv`, Tables 00–03 |
| 02 | `02_eda.ipynb` | Exploratory analysis: who churns, when, and at what cost | Tables 04–07, Figures 01–06 |
| 03 | `03_segmentation.ipynb` | K-Means value segments (k chosen by a pre-set rule) | segmented train/test, Tables 08–10 |
| 04 | `04_modelling.ipynb` | 4 models × 4 training conditions × 3 calibration options | OOF and test predictions, Tables 11–19 |
| 04b | `04b_tabicl.ipynb` | TabICL v2 added as a fifth model | Tables 20–24 |
| 05 | `05_thresholds.ipynb` | Profit thresholds, H1 test, budgets, sensitivity | thresholds, Tables 25–35 |
| 06 | `06_counterfactuals.ipynb` | Costed counterfactual retention actions (DiCE) | Tables 36–43 |
| 07 | `07_kpis.ipynb` | Segment KPIs and the H4 test | Tables 44–49 |

Each notebook reads the files written by the previous ones, so run them in order (01 → 07).

## How to run

Tested with Python 3.12 on macOS (Intel). PyTorch 2.2 is the last release for Intel Macs and needs NumPy 1.x, which is why `requirements.txt` pins `numpy<2`.

```bash
# 1. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install the packages
pip install -r requirements.txt

# 3. Download the TabICL v2 checkpoint into pretrained/ (needed by notebook 04b)
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='jingang/TabICL', filename='tabicl-classifier-v2-20260212.ckpt', local_dir='pretrained')"

# 4. Open Jupyter and run the notebooks in order
jupyter lab
```

Run times on a laptop CPU: most notebooks take seconds to a few minutes. Model tuning (04, about 15–40 minutes) and TabICL predictions (04b, about 16 minutes) are cached in `outputs/models/`; delete the cache files to recompute them.

### Report figures

```bash
python report_figures.py <output-folder>
```

### Optional dashboard

```bash
python dashboard/export_data.py      # once, after the notebooks
streamlit run dashboard/app.py
```

## Data

IBM Telco Customer Churn sample dataset (7,043 customers, 21 columns), publicly available on Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The data describes a fictional company, so no personal data is involved.
