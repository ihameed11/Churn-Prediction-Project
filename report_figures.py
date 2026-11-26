"""Re-draws the report figures at print size (text width 6.27 in) so fonts stay readable."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.append(str(Path(__file__).parent))
import config as C
from src import modelling as M

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
W = 6.27
SEG = ["Low-value", "New high-value", "Established high-value"]
SEGC = {"Low-value": "#2a78d6", "New high-value": "#1baf7a", "Established high-value": "#4a3aa7"}
MODC = {"Logistic regression": "#2a78d6", "Random forest": "#1baf7a", "LightGBM": "#eda100", "SVM": "#4a3aa7"}
GREY = "#8a8984"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#ebeae6", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.edgecolor": "#9a9993", "savefig.dpi": 300, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02})
def save(fig, name): fig.savefig(OUT / f"{name}.png"); plt.close(fig); print("saved", name)

P = C.PROCESSED_DIR
train = pd.read_csv(P / "train_segmented.csv"); test = pd.read_csv(P / "test_segmented.csv")
oof = pd.read_csv(P / "stage3_oof_predictions.csv"); tst = pd.read_csv(P / "stage3_test_predictions.csv")
T = C.TABLES_DIR

# ---------- Figure: segments (map + lost revenue) ----------
fig, ax = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"width_ratios": [1.05, 1]})
for s in SEG:
    d = train[train.segment == s]
    ax[0].scatter(d.tenure, d.MonthlyCharges, s=3, alpha=0.35, color=SEGC[s], lw=0, label=s, rasterized=True)
    ax[0].scatter(d.tenure.mean(), d.MonthlyCharges.mean(), marker="X", s=60, color="black", edgecolor="white", lw=0.8, zorder=3)
ax[0].set_xlabel("Tenure (months)"); ax[0].set_ylabel("Monthly charge")
ax[0].legend(markerscale=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.5, handletextpad=0.2, columnspacing=0.8)
rev = train.MonthlyCharges * train.Churn
share_c = train.segment.value_counts(normalize=True).reindex(SEG) * 100
share_r = rev.groupby(train.segment).sum().reindex(SEG) / rev.sum() * 100
churn = train.groupby("segment").Churn.mean().reindex(SEG) * 100
y = np.arange(3); h = 0.38
ax[1].barh(y - h/2, share_c, h, color="#2a78d6", label="% of customers")
ax[1].barh(y + h/2, share_r, h, color="#e8663a", label="% of revenue lost to churn")
for i in range(3):
    ax[1].text(share_c.iloc[i] + 1, y[i] - h/2, f"{share_c.iloc[i]:.0f}%", va="center", fontsize=7)
    ax[1].text(share_r.iloc[i] + 1, y[i] + h/2, f"{share_r.iloc[i]:.0f}%", va="center", fontsize=7)
ax[1].set_yticks(y, [f"{s}\n(churn {c:.0f}%)" for s, c in zip(SEG, churn)]); ax[1].invert_yaxis()
ax[1].set_xlim(0, 85); ax[1].set_title("Lost revenue by segment")
ax[1].legend(loc="upper center", bbox_to_anchor=(0.4, -0.22), ncol=2); ax[1].grid(axis="y", visible=False)
fig.tight_layout(w_pad=1.5); save(fig, "r_segments")

# ---------- Figure: reliability diagrams ----------
def rel(y, p, n=10):
    b = pd.qcut(pd.Series(p).rank(method="first"), n, labels=False)
    g = pd.DataFrame({"y": y, "p": p, "b": b}).groupby("b").mean()
    return g.p.values, g.y.values
y_tr = train.Churn.to_numpy()
models = ["Logistic regression", "Random forest", "LightGBM", "SVM"]
fig, axes = plt.subplots(2, 4, figsize=(W, 3.5), sharex=True, sharey=True)
styles = {"raw": ":", "Platt": "--", "isotonic": "-"}
for r, cond in enumerate(["none", "value"]):
    for c, m in enumerate(models):
        a = axes[r, c]; a.plot([0, 1], [0, 1], color=GREY, lw=0.7)
        for cal, ls in styles.items():
            col = f"{m}|{cond}|{cal}"
            if col in oof and oof[col].notna().all():
                px, py = rel(y_tr, oof[col].to_numpy())
                a.plot(px, py, ls, marker="o", ms=2, lw=1, color=MODC[m], label=cal)
        a.set_title(f"{m if m != 'Logistic regression' else 'Log. regression'}\n{'no weights' if cond == 'none' else 'value weights'}", fontsize=7.5)
        a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        if c == 0: a.set_ylabel("Observed churn rate")
        if r == 1: a.set_xlabel("Mean predicted")
        if r == 0 and c == 0: a.legend(loc="upper left", fontsize=6.5, handlelength=1.8)
fig.tight_layout(h_pad=0.6, w_pad=0.4); save(fig, "r_reliability")

# ---------- Figure: segment calibration with / without value weights ----------
sel = pd.read_csv(T / "15_cv_selected_calibration.csv").set_index(["model", "condition"])["calibration"]
fig, axes = plt.subplots(1, 3, figsize=(W, 2.2))
x = np.arange(4); h = 0.38
for a, s in zip(axes, SEG):
    msk = (test.segment == s).to_numpy()
    for k, (cond, colr, lab) in enumerate([("none", "#9bbfee", "no weights"), ("value", "#2a78d6", "value weights")]):
        vals = [tst.loc[msk, f"{m}|{cond}|{sel[(m, cond)]}"].mean() for m in models]
        a.bar(x + (k - 0.5) * h, vals, h, color=colr, label=lab)
    a.axhline(test.Churn[msk].mean(), color="black", ls="--", lw=1, label="observed churn rate")
    a.set_xticks(x, ["LR", "RF", "LGBM", "SVM"]); a.set_title(s)
    a.grid(axis="x", visible=False)
axes[0].set_ylabel("Mean predicted churn probability")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, "r_segment_calibration")

# ---------- Figure: threshold profit curves and stability ----------
clv_tr = M.customer_value(train).to_numpy() if hasattr(M.customer_value(train), "to_numpy") else M.customer_value(train)
D = clv_tr.mean() * C.INCENTIVE_RATIO
b_, c_ = M.profit_terms(clv_tr, D)
g = np.where(y_tr == 1, b_, -c_)
p = oof["LightGBM|none|raw"].to_numpy()
GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)
curve = lambda pp, gg: ((pp[None, :] > GRID[:, None]) * gg[None, :]).sum(axis=1)
def choose(pp, gg, nb=200):
    rng = np.random.default_rng(C.RANDOM_STATE); best = np.empty(nb)
    for i in range(nb):
        idx = rng.integers(0, len(pp), len(pp)); best[i] = GRID[np.argmax(curve(pp[idx], gg[idx]))]
    return np.round(np.median(best), 2), best
seg = train.segment.to_numpy()
th = pd.read_csv(T / "25_thresholds_by_model.csv").set_index("model")
fig, ax = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"width_ratios": [1.6, 1]})
boots, theo = {}, {}
for s in SEG:
    m_ = seg == s
    ax[0].plot(GRID, curve(p[m_], g[m_]) / m_.sum(), color=SEGC[s], lw=1.3, label=s)
    t_s = th.loc["LightGBM", f"P2 {s}"]; ax[0].axvline(t_s, color=SEGC[s], ls="--", lw=0.8)
    tt = th.loc["Theory (profit formula)", f"P2 {s}"]; theo[s] = tt
    ax[0].plot(tt, np.interp(tt, GRID, curve(p[m_], g[m_]) / m_.sum()), "o", ms=4, color=SEGC[s])
    boots[s] = choose(p[m_], g[m_])[1]
boots["Global"] = choose(p, g)[1]; theo["Global"] = th.loc["Theory (profit formula)", "P1 global"]
ax[0].set_xlim(0, 0.8); ax[0].set_xlabel("Threshold"); ax[0].set_ylabel("Profit per customer (training, OOF)")
ax[0].set_title("Profit curves (dashed = chosen, dot = theory)"); ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3)
labs = ["Low-value", "New high-value", "Established high-value", "Global"]
ax[1].boxplot([boots[k] for k in labs], orientation="horizontal", widths=0.5, flierprops={"markersize": 2},
              medianprops={"color": "#e8663a"})
for i, k in enumerate(labs): ax[1].plot(theo[k], i + 1, "D", color="#d6402a", ms=4)
ax[1].set_yticks(range(1, 5), ["Low", "New high", "Est. high", "Global"]); ax[1].invert_yaxis()
ax[1].set_xlabel("Bootstrap optimum"); ax[1].set_title("Threshold stability (red = theory)")
fig.tight_layout(w_pad=1.2); save(fig, "r_thresholds")

# ---------- Figure: budgets + sensitivity ----------
bud = pd.read_csv(T / "32_budget_scenarios.csv"); sen = pd.read_csv(T / "33_sensitivity.csv")
fig, ax = plt.subplots(1, 2, figsize=(W, 2.4), gridspec_kw={"width_ratios": [1, 1.25]})
for rk, col, lab in [("rank by churn probability", GREY, "rank by churn probability"), ("rank by expected profit", "#2a78d6", "rank by expected profit")]:
    d = bud[bud.ranking == rk]; xs = d.budget.str.rstrip("%").astype(int)
    ax[0].plot(xs, d["Profit per customer"], "o-", color=col, ms=4, lw=1.3, label=lab)
nolim = bud.loc[bud.budget == "no limit", "Profit per customer"].iloc[0]
ax[0].axhline(nolim, color="#d6402a", ls="--", lw=1, label="no budget limit")
ax[0].set_xticks([10, 20, 30], ["10%", "20%", "30%"]); ax[0].set_xlabel("Customers that can be contacted")
ax[0].set_ylabel("Profit per customer (test)"); ax[0].set_title("Contact budgets"); ax[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.93))
names = {"base (d = CLV/20, gamma = 0.3)": "base case", "offer cost d = CLV/10": "offer cost CLV/10",
         "offer cost d = CLV/5": "offer cost CLV/5", "acceptance gamma = 0.1": "acceptance 0.10",
         "acceptance gamma = 0.5": "acceptance 0.50", "proportional offer d = 5% of CLV": "proportional offer"}
yy = np.arange(len(sen))
ax[1].errorbar(sen["P2 - P1"], yy, xerr=[sen["P2 - P1"] - sen["CI 2.5%"], sen["CI 97.5%"] - sen["P2 - P1"]],
               fmt="o", color="#2a78d6", ms=4, capsize=2, lw=1)
ax[1].axvline(0, color="black", lw=0.8); ax[1].set_yticks(yy, [names.get(n, n) for n in sen.scenario]); ax[1].invert_yaxis()
ax[1].set_xlabel("P2 $-$ P1 profit per customer (95% CI)"); ax[1].set_title("Sensitivity of the segment-threshold gain")
ax[1].grid(axis="y", visible=False)
fig.tight_layout(w_pad=1.5); save(fig, "r_budget_sensitivity")

# ---------- Figure: counterfactual actions ----------
cf = pd.read_csv(P / "stage5_counterfactuals.csv")
best = cf[cf.usable].sort_values("net_value", ascending=False).groupby("test_index").head(1)
acts = {"Contract": "Contract upgrade", "MonthlyCharges": "Discount", "PaymentMethod": "Automatic payment",
        "OnlineSecurity": "Free OnlineSecurity", "OnlineBackup": "Free OnlineBackup",
        "DeviceProtection": "Free DeviceProtection", "TechSupport": "Free TechSupport"}
fig, ax = plt.subplots(1, 3, figsize=(W, 2.4), gridspec_kw={"width_ratios": [1.25, 1, 0.9]})
yb = np.arange(len(acts)); h = 0.27
for k, s in enumerate(SEG):
    b = best[best.segment == s]
    vals = [100 * b.changes.fillna("").str.contains(a).mean() for a in acts]
    ax[0].barh(yb + (k - 1) * h, vals, h, color=SEGC[s], label=s)
ax[0].set_yticks(yb, list(acts.values())); ax[0].invert_yaxis(); ax[0].set_xlabel("% of recommended actions")
ax[0].set_title("Actions included"); ax[0].legend(loc="lower right", fontsize=6.5); ax[0].grid(axis="y", visible=False)
for s in SEG:
    b = best[best.segment == s]
    ax[1].scatter(b.cost_12m, b.delta_p, s=8, alpha=0.7, color=SEGC[s], lw=0)
ax[1].set_xlabel("12-month cost"); ax[1].set_ylabel("Drop in churn probability"); ax[1].set_title("Effect vs cost")
bp = ax[2].boxplot([best.loc[best.segment == s, "net_value"] for s in SEG], widths=0.55, patch_artist=True,
                   flierprops={"markersize": 2}, medianprops={"color": "black"})
for patch, s in zip(bp["boxes"], SEG): patch.set_facecolor(SEGC[s]); patch.set_alpha(0.6)
ax[2].axhline(0, color=GREY, ls="--", lw=0.8); ax[2].set_xticks([1, 2, 3], ["Low", "New\nhigh", "Est.\nhigh"])
ax[2].set_ylabel("Net value"); ax[2].set_title("Net value"); ax[2].grid(axis="x", visible=False)
fig.tight_layout(w_pad=1.0); save(fig, "r_counterfactuals")

# ---------- Figure: KPI validation (relative RaR error with 95% CI) ----------
v = pd.read_csv(T / "44_kpi_rar_validation.csv").set_index("segment")
src = [("primary", "Primary (LightGBM, no weights)", "#eda100"), ("VW raw", "Value weights, raw", "#9a9993"),
       ("VW calibrated", "Value weights, calibrated", "#2a78d6")]
fig, ax = plt.subplots(figsize=(W, 2.3))
rows = SEG + ["Total"]
for j, (k, lab, col) in enumerate(src):
    y = np.arange(len(rows)) + (j - 1) * 0.24
    e = v.loc[rows, f"{k}: relative error %"].to_numpy()
    lo_hi = v.loc[rows, f"{k}: 95% CI"].str.strip("[]").str.split(",", expand=True).astype(float).to_numpy()
    ax.errorbar(e, y, xerr=[e - lo_hi[:, 0], lo_hi[:, 1] - e], fmt="o", color=col, ms=4, capsize=2, lw=1, label=lab)
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(np.arange(len(rows)), rows); ax.invert_yaxis()
ax.set_xlabel("Relative error of estimated revenue at risk (%, 95% CI)")
ax.legend(loc="lower right", fontsize=7); ax.grid(axis="y", visible=False)
fig.tight_layout(); save(fig, "r_kpi_validation")
