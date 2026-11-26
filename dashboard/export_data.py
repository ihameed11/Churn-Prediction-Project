"""Stage 6 - exports the data the dashboard reads.

Run from the project folder:   python dashboard/export_data.py
The dashboard never loads a model: it only reads the CSV/JSON files written here,
which come from the saved results of Stages 2-5. A reconciliation check makes sure
the dashboard numbers equal the report tables (Table 44).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
import config as C
from src import modelling as M

OUT = ROOT / "dashboard" / "data"
OUT.mkdir(parents=True, exist_ok=True)
SEGMENTS = ["Low-value", "New high-value", "Established high-value"]

test = pd.read_csv(C.PROCESSED_DIR / "test_segmented.csv")
tst = pd.read_csv(C.PROCESSED_DIR / "stage3_test_predictions.csv")
cfs = pd.read_csv(C.PROCESSED_DIR / "stage5_counterfactuals.csv")
s4 = joblib.load(C.MODELS_DIR / "stage4_thresholds.joblib")
s5b = joblib.load(C.MODELS_DIR / "stage5b_kpis.joblib")
assert (tst[C.ID_COL] == test[C.ID_COL]).all()

D, F, GAMMA = float(s4["incentive_d"]), C.CONTACT_COST, C.ACCEPTANCE_RATE
TH = s4["thresholds"]["LightGBM"]["P2"]

# ---------------------------------------------------------------- customers
p = tst["LightGBM|none|raw"].to_numpy()
clv = np.asarray(M.customer_value(test))
benefit, cost = M.profit_terms(clv, D, F, GAMMA)
E = p * benefit - (1 - p) * cost
t = test["segment"].map(TH).to_numpy()

cust = pd.DataFrame({
    "customerID": test[C.ID_COL], "segment": test["segment"], "tenure": test["tenure"],
    "contract": test["Contract"], "internet": test["InternetService"], "payment": test["PaymentMethod"],
    "monthly_charge": test["MonthlyCharges"], "annual_value": clv.round(2),
    "churn_probability": p.round(4), "threshold": t, "expected_profit": E.round(2),
    "contact": p > t, "churned": test[C.TARGET].astype(int),
})

# recommended retention action (Stage 5a): best counterfactual per explained customer
NICE = {"Contract": "contract", "MonthlyCharges": "discount", "PaymentMethod": "automatic payment",
        "OnlineSecurity": "free OnlineSecurity", "OnlineBackup": "free OnlineBackup",
        "DeviceProtection": "free DeviceProtection", "TechSupport": "free TechSupport"}

def describe(r, cur):
    parts = []
    for ch in str(r["changes"]).split(" + "):
        if ch == "Contract":
            parts.append(f"Upgrade contract: {cur['contract']} → {r['new_Contract']}")
        elif ch == "MonthlyCharges":
            parts.append(f"Discount {r['discount_pct']:.0f}% ({cur['monthly_charge']:.2f} → {r['new_MonthlyCharges']:.2f} per month)")
        elif ch == "PaymentMethod":
            parts.append(f"Switch to {r['new_PaymentMethod']}")
        elif ch in NICE:
            parts.append(f"Offer {NICE[ch]} (free)")
    return "; ".join(parts)

used = cfs[cfs["usable"]]
best = used.sort_values("net_value", ascending=False).groupby("customerID").head(1).set_index("customerID")
cust = cust.set_index("customerID")
cust["explained"] = cust.index.isin(cfs["customerID"])
cust["action"] = ""
for cid, r in best.iterrows():
    cust.loc[cid, "action"] = describe(r, cust.loc[cid])
cust.loc[best.index, "action_type"] = best["changes"].str.replace("MonthlyCharges", "Discount")
cust.loc[best.index, "prob_after_action"] = best["p_after"].round(4)
cust.loc[best.index, "action_cost_12m"] = best["cost_12m"].round(2)
cust.loc[best.index, "action_net_value"] = best["net_value"].round(2)
cust = cust.reset_index()
cust.to_csv(OUT / "customers.csv", index=False)

# ------------------------------------------------------------ segment KPIs
k = s5b["kpis"]["Primary (LightGBM, no weights)"].copy()
k.index.name = "segment"
val = pd.read_csv(C.TABLES_DIR / "44_kpi_rar_validation.csv").set_index("segment")
k["RaR relative error %"] = val["primary: relative error %"]
k["RaR 95% CI"] = val["primary: 95% CI"]
k["threshold"] = pd.Series(TH)
k.to_csv(OUT / "segment_kpis.csv")
val.to_csv(OUT / "kpi_validation.csv")

# ---------------------------------------------------------------- playbooks
pd.read_csv(C.TABLES_DIR / "41_segment_playbook.csv").to_csv(OUT / "playbook.csv", index=False)
pd.read_csv(C.TABLES_DIR / "43_segment_playbook_plan_b.csv").to_csv(OUT / "playbook_plan_b.csv", index=False)

# -------------------------------------------------------------- assumptions
json.dump({"incentive_d": round(D, 2), "contact_cost_f": F, "acceptance_rate": GAMMA,
           "value_horizon_months": C.VALUE_HORIZON_MONTHS, "thresholds": TH,
           "model": "LightGBM (no weights, raw probabilities = calibrated)",
           "test_customers": int(len(cust)), "h4_delta_sae": float(s5b["h4"]["delta"]),
           "h4_ci": [float(x) for x in s5b["h4"]["ci"]]},
          open(OUT / "assumptions.json", "w"), indent=2)

# ------------------------------------------------- reconciliation with Table 44
seg = cust.groupby("segment")
chk = pd.DataFrame({
    "RaR estimated": seg.apply(lambda g: (g.churn_probability * g.annual_value).sum(), include_groups=False),
    "RaR realised": seg.apply(lambda g: (g.churned * g.annual_value).sum(), include_groups=False),
    "contacted": seg["contact"].sum(),
}).reindex(SEGMENTS)
assert np.allclose(chk["RaR estimated"], val.loc[SEGMENTS, "primary: estimated RaR"], rtol=1e-3)
assert np.allclose(chk["RaR realised"], val.loc[SEGMENTS, "realised RaR"], rtol=1e-3)
assert (chk["contacted"].to_numpy() == val.loc[SEGMENTS, "contacted (primary)"].to_numpy()).all()
print("Reconciliation with report Table 44: OK")
print("Exported:", sorted(f.name for f in OUT.iterdir()))
