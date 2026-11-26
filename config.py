"""All assumptions in one place.

Every number that affects a result lives here, so the thesis can state
its assumptions in one table and the sensitivity analysis can vary them.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "raw" / "WA_FnUseC-Telco-Customer-Churn.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "outputs" / "figures"
TABLES_DIR = ROOT / "outputs" / "tables"
MODELS_DIR = ROOT / "outputs" / "models"

# ---------------------------------------------------------- reproducibility
RANDOM_STATE = 42          # fixed seed used everywhere
TEST_SIZE = 0.20           # stratified 80/20 split

# ------------------------------------------------------------ data columns
ID_COL = "customerID"
TARGET = "Churn"           # mapped to 1 = churned, 0 = retained
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# ------------------------------------------------------------ segmentation
SEGMENT_FEATURES = ["tenure", "MonthlyCharges"]  # kept to two on purpose
N_SEGMENTS = 3                                   # expected k; the notebook checks it with the rule below
K_RANGE = range(2, 7)                            # k = 2 to 6 evaluated
MIN_CHURNERS_PER_SEGMENT = 100                   # k rule: best silhouette among k where every TRAIN
                                                 # segment has >= 100 churners (needed for Stage 4 thresholds)
ROBUSTNESS_K = 4                                 # statistically best k, re-run in Stage 4 as a robustness check

# -------------------------------------------------------- campaign-profit assumptions
# Approved 21 Oct 2026 (Stage 3 design). Sources: Verbraken et al. (2013),
# Hoppner et al. (2020), Gomez-Vargas et al. (2025), Petrides & Verbeke (2022),
# Vanderschueren et al. (2022). Profit of contacting customer i:
#   churner:      ACCEPTANCE_RATE * (CLV_i - d) - CONTACT_COST
#   non-churner:  -(d + CONTACT_COST)
VALUE_HORIZON_MONTHS = 12      # CLV_i = 12 x MonthlyCharges_i (revenue, not margin)
INCENTIVE_RATIO = 1 / 20       # d = mean training CLV / 20 (delta = 0.05) -> about 39
CONTACT_COST = 1.0             # f, cost of contacting one customer
ACCEPTANCE_RATE = 0.30         # gamma, share of contacted churners who accept and stay
EMP_BETA = (6, 14)             # gamma ~ Beta(6, 14) (mean 0.30) for the expected maximum profit

SENSITIVITY = {
    "INCENTIVE_DIVISOR": [20, 15, 10, 5, 3],       # d = mean CLV / divisor
    "ACCEPTANCE_RATE": [0.10, 0.20, 0.30, 0.40, 0.50],
    "VALUE_HORIZON_MONTHS": [6, 12, 24],
}

# ------------------------------------------------------------ modelling (Stage 3)
CV_FOLDS = 5                   # stratified folds on the training set, same for every model
TUNING_METRIC = "roc_auc"      # not changed by monotone calibration (Bischl et al., 2023)
N_RANDOM_CONFIGS = {"Random forest": 30, "LightGBM": 40}
PROB_CLIP = (0.001, 0.999)     # clip calibrated probabilities (isotonic can return 0 or 1)
ECE_BINS = 10                  # equal-frequency bins for the expected calibration error
N_BOOTSTRAP = 2000             # paired bootstrap resamples on the test set

# ------------------------------------------------------------ Stage 3b: tabular foundation model
# TabICL v2 checkpoint (Hugging Face: jingang/TabICL), downloaded once into pretrained/
TABICL_CHECKPOINT = ROOT / "pretrained" / "tabicl-classifier-v2-20260212.ckpt"

# ------------------------------------------------------------ thresholds
THRESHOLD_GRID = [round(t / 100, 2) for t in range(5, 96)]  # 0.05 ... 0.95
DEFAULT_THRESHOLD = 0.50       # the one-size-fits-all baseline
