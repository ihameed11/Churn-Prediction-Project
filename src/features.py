"""Model-agnostic feature preparation for the churn thesis.

Every stage (EDA, segmentation, modelling, dashboard) imports these functions,
so the same rules are applied everywhere. Nothing here learns from data except
make_encoder(), which must be fitted on the training set only.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------
INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]
SERVICE_FLAGS = ["PhoneService", "MultipleLines"] + INTERNET_ADDONS   # Yes/No columns

# Role of every column that is NOT a plain model input (Decision 6)
ROLES = {
    C.ID_COL:       ("identifier", "Unique ID with no predictive meaning"),
    C.TARGET:       ("target",     "Outcome to predict (1 = churned)"),
    "TotalCharges": ("excluded",   "≈ tenure × MonthlyCharges (median ratio 1.00); kept for EDA and revenue"),
    "n_services":   ("optional",   "Engineered: count of services; r = 0.85 with MonthlyCharges"),
}
OPTIONAL = [col for col, (role, _) in ROLES.items() if role == "optional"]
NOT_INPUTS = list(ROLES)


# ---------------------------------------------------------------------------
# Fixed rules (do not learn from data - safe before the split)
# ---------------------------------------------------------------------------
def clean_raw(df_raw):
    """Decisions 1 and 3: drop tenure-0 rows, Churn to 0/1, SeniorCitizen to No/Yes."""
    df = df_raw.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    df[C.TARGET] = (df[C.TARGET] == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df


def merge_redundant_levels(df):
    """Decision 4: 'No internet service' / 'No phone service' -> 'No'."""
    df = df.copy()
    df[INTERNET_ADDONS] = df[INTERNET_ADDONS].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    return df


def add_n_services(df):
    """Decision 5: number of services held (1-9)."""
    df = df.copy()
    df["n_services"] = ((df[SERVICE_FLAGS] == "Yes").sum(axis=1)
                        + (df["InternetService"] != "No").astype(int))
    return df


def prepare(df_raw):
    """Full fixed-rule pipeline: raw CSV -> cleaned, merged, with n_services."""
    return add_n_services(merge_redundant_levels(clean_raw(df_raw)))


# ---------------------------------------------------------------------------
# Feature dictionary (Decision 6)
# ---------------------------------------------------------------------------
def _column_type(s):
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 2:
        return "numeric"
    return "binary" if s.nunique() == 2 else "multi-class"


def _column_levels(s):
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 2:
        return f"{s.min():g} to {s.max():g}"
    return ", ".join(map(str, sorted(s.unique())))


def feature_dictionary(df):
    """Type, levels/range and role of every column except the ID."""
    return pd.DataFrame([{
        "feature":        col,
        "type":           _column_type(df[col]),
        "levels / range": _column_levels(df[col]),
        "role":           ROLES.get(col, ("input", ""))[0],
        "note":           ROLES.get(col, ("input", ""))[1],
    } for col in df.columns if col != C.ID_COL])


# ---------------------------------------------------------------------------
# Inputs and encoding (Decision 9)
# ---------------------------------------------------------------------------
def get_X_y(df, include_optional=False):
    """Inputs (X) and target (y). Optional features are added only on request."""
    keep = OPTIONAL if include_optional else []
    drop = [c for c in NOT_INPUTS if c not in keep and c in df.columns]
    return df.drop(columns=drop), df[C.TARGET]


def make_encoder(X, scale_numeric=True, drop="if_binary"):
    """Scale numeric and one-hot encode categorical columns. FIT ON TRAIN ONLY.

    scale_numeric : True for distance/coefficient models, False for trees.
    drop          : "if_binary" (default) or "first" for unregularised linear models.
    """
    numeric_cols = X.select_dtypes("number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler() if scale_numeric else "passthrough", numeric_cols),
        ("cat", OneHotEncoder(drop=drop, handle_unknown="ignore", sparse_output=False),
         categorical_cols),
    ])


# ---------------------------------------------------------------------------
# Segments (Stage 2)
# ---------------------------------------------------------------------------
def assign_segment(df, k=None):
    """Assign customers to value segments with the scaler and K-Means fitted on
    the TRAINING set in 03_segmentation.ipynb. New or test customers are placed
    in the nearest segment; nothing is refitted."""
    import joblib
    k = k or C.N_SEGMENTS
    model = joblib.load(C.MODELS_DIR / f"segmentation_k{k}.joblib")
    ids = model["kmeans"].predict(model["scaler"].transform(df[model["features"]]))
    return pd.Series([model["names"][i] for i in ids], index=df.index, name="segment")
