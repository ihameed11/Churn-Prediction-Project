"""Stage 3 helpers: campaign profit, instance weights, models, calibration and metrics.

Used by notebooks/04_modelling.ipynb and, later, by the Stage 4 threshold notebook,
so that every stage computes profit and metrics in exactly the same way.

Profit of contacting customer i (Verbraken et al., 2013, with individual CLV):
    would-be churner:  gamma * (CLV_i - d) - f
    non-churner:       -(d + f)
    not contacted:     0
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             matthews_corrcoef, roc_auc_score)
from sklearn.svm import SVC

import config as C

MODELS = ["Logistic regression", "Random forest", "LightGBM", "SVM"]
CONDITIONS = ["none", "class", "value", "SMOTE"]
CALIBRATIONS = ["raw", "Platt", "isotonic"]


# ---------------------------------------------------------------------------
# Campaign profit
# ---------------------------------------------------------------------------
def customer_value(df, horizon=None):
    """CLV_i = horizon (months) x MonthlyCharges_i."""
    return (horizon or C.VALUE_HORIZON_MONTHS) * df["MonthlyCharges"].to_numpy()


def profit_terms(clv, d, f=None, gamma=None):
    """Return (benefit if a contacted customer is a churner, cost if not)."""
    f = C.CONTACT_COST if f is None else f
    gamma = C.ACCEPTANCE_RATE if gamma is None else gamma
    clv = np.asarray(clv, dtype=float)
    return gamma * (clv - d) - f, np.broadcast_to(np.asarray(d + f, dtype=float), clv.shape).copy()


def campaign_cost(y, contact, clv, d, f=None, gamma=None):
    """Money spent on the campaign: contact cost for everyone contacted, plus the
    incentive for non-churners (who take it) and for churners who accept (gamma)."""
    f = C.CONTACT_COST if f is None else f
    gamma = C.ACCEPTANCE_RATE if gamma is None else gamma
    y = np.asarray(y)
    d = np.broadcast_to(np.asarray(d, dtype=float), y.shape)
    return float(np.sum(contact * (f + d * (y * gamma + (1 - y)))))


def theoretical_threshold(clv, d, f=None, gamma=None):
    """Contact when p > (d + f) / (gamma * (CLV - d) + d)."""
    benefit, cost = profit_terms(clv, d, f, gamma)
    return cost / (benefit + cost)


def contact_decision(p, clv, d, f=None, gamma=None):
    """Contact when the expected profit p*benefit - (1-p)*cost is positive."""
    benefit, cost = profit_terms(clv, d, f, gamma)
    return p * benefit - (1 - p) * cost > 0


def realised_profit(y, contact, clv, d, f=None, gamma=None):
    """Campaign profit (expected over offer acceptance) of the contacted customers."""
    benefit, cost = profit_terms(clv, d, f, gamma)
    y = np.asarray(y)
    return float(np.sum(contact * (y * benefit - (1 - y) * cost)))


def emp(y, p, clv, d, f=None, beta=None, n_gamma=200):
    """Expected maximum profit per customer (Verbraken et al., 2013) with individual CLV.

    Customers are ranked by p; for each acceptance rate gamma (drawn from a
    Beta distribution) the best targeting depth is chosen, and the maximum
    profits are averaged. Depends only on the ranking, not on calibration."""
    f = C.CONTACT_COST if f is None else f
    a, b = beta or C.EMP_BETA
    gammas = stats.beta.ppf((np.arange(n_gamma) + 0.5) / n_gamma, a, b)
    order = np.argsort(-np.asarray(p), kind="stable")
    y_s, clv_s = np.asarray(y)[order], np.asarray(clv)[order]
    A = np.concatenate([[0], np.cumsum(y_s * (clv_s - d))])
    B = np.concatenate([[0], np.cumsum(y_s * f + (1 - y_s) * (d + f))])
    best = (gammas[:, None] * A[None, :] - B[None, :]).max(axis=1)
    return float(best.mean() / len(y_s))


# ---------------------------------------------------------------------------
# Instance weights
# ---------------------------------------------------------------------------
def class_weights(y):
    """Class-balanced weights n / (2 * n_class)."""
    y = np.asarray(y)
    n1 = y.sum()
    n0 = len(y) - n1
    return np.where(y == 1, len(y) / (2 * n1), len(y) / (2 * n0))


def value_weights(y, clv, d, f=None, gamma=None):
    """Profit lost if the model gets customer i wrong, rescaled to mean 1.

    Churner: gamma*(CLV_i - d) - f (missed retention benefit).
    Non-churner: d + f (wasted incentive and contact)."""
    benefit, cost = profit_terms(clv, d, f, gamma)
    w = np.where(np.asarray(y) == 1, benefit, cost)
    return w / w.mean()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def make_model(name, params):
    """One classifier per model family, with the given hyperparameters."""
    if name == "Logistic regression":
        return LogisticRegression(max_iter=5000, **params)
    if name == "Random forest":
        return RandomForestClassifier(n_estimators=500, n_jobs=-1,
                                      random_state=C.RANDOM_STATE, **params)
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=C.RANDOM_STATE, n_jobs=-1, verbose=-1,
                              deterministic=True, force_row_wise=True, **params)
    if name == "TabICL v2":
        # Stage 3b: pre-trained tabular foundation model (Qu et al., 2025; 2026), used
        # as released (no tuning). The checkpoint is loaded from a local file so the
        # run is reproducible and works offline.
        from tabicl import TabICLClassifier
        return TabICLClassifier(model_path=C.TABICL_CHECKPOINT, allow_auto_download=False,
                                random_state=C.RANDOM_STATE, **params)
    if name == "SVM":
        return SVC(kernel="rbf", **params)   # probability=False: we calibrate ourselves
    raise ValueError(name)


def model_outputs(model, X):
    """Return (score, raw probability).

    score: input for Platt/isotonic calibration (log-odds or SVM decision value).
    raw probability: the model's own probability; None for the SVM."""
    if isinstance(model, SVC):
        return model.decision_function(X), None
    p = model.predict_proba(X)[:, 1]
    if isinstance(model, LogisticRegression):
        return model.decision_function(X), p
    return logit(np.clip(p, 1e-6, 1 - 1e-6)), p


def training_weights(condition, y, clv, d):
    if condition == "class":
        return class_weights(y)
    if condition == "value":
        return value_weights(y, clv, d)
    return None


def fit_one(name, params, condition, X_tr, y_tr, clv_tr, d, encoder_factory):
    """Fit encoder, (SMOTE), weights and model on one training set."""
    from imblearn.over_sampling import SMOTE
    enc = encoder_factory(X_tr)
    Z = enc.fit_transform(X_tr)
    y_fit = np.asarray(y_tr)
    w = training_weights(condition, y_fit, clv_tr, d)
    if condition == "SMOTE":
        Z, y_fit = SMOTE(random_state=C.RANDOM_STATE).fit_resample(Z, y_fit)
    model = make_model(name, params)
    if w is None:
        model.fit(Z, y_fit)          # TabICL (Stage 3b) has no sample_weight argument
    else:
        model.fit(Z, y_fit, sample_weight=w)
    return enc, model


def cross_validate(name, params, condition, X, y, clv, d, fold_id, encoder_factory):
    """Out-of-fold scores and raw probabilities; everything fitted inside each fold."""
    y = np.asarray(y)
    oof_score = np.zeros(len(y))
    oof_prob = np.full(len(y), np.nan)
    fold_auc = []
    for k in np.unique(fold_id):
        tr, va = fold_id != k, fold_id == k
        enc, model = fit_one(name, params, condition, X[tr], y[tr], clv[tr], d, encoder_factory)
        s, p = model_outputs(model, enc.transform(X[va]))
        oof_score[va] = s
        if p is not None:
            oof_prob[va] = p
        fold_auc.append(roc_auc_score(y[va], s))
    return oof_score, oof_prob, float(np.mean(fold_auc))


# ---------------------------------------------------------------------------
# Calibration (fitted WITHOUT instance weights)
# ---------------------------------------------------------------------------
def fit_calibrator(method, score, y):
    if method == "Platt":
        return LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(score).reshape(-1, 1), y)
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=C.PROB_CLIP[0],
                                  y_max=C.PROB_CLIP[1]).fit(score, y)
    raise ValueError(method)


def apply_calibrator(cal, score):
    if isinstance(cal, LogisticRegression):
        p = cal.predict_proba(np.asarray(score).reshape(-1, 1))[:, 1]
    else:
        p = cal.predict(score)
    return np.clip(p, *C.PROB_CLIP)


def cross_fit_calibration(method, score, y, fold_id):
    """Calibrated OOF probabilities: fold k is calibrated by a map fitted on the other folds."""
    y = np.asarray(y)
    out = np.zeros(len(y))
    for k in np.unique(fold_id):
        va = fold_id == k
        cal = fit_calibrator(method, score[~va], y[~va])
        out[va] = apply_calibrator(cal, score[va])
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def ece(y, p, n_bins=None):
    """Expected calibration error with equal-frequency bins."""
    n_bins = n_bins or C.ECE_BINS
    bins = pd.qcut(pd.Series(p).rank(method="first"), n_bins, labels=False)
    df = pd.DataFrame({"y": y, "p": p, "b": bins})
    g = df.groupby("b").agg(n=("y", "size"), obs=("y", "mean"), pred=("p", "mean"))
    return float((g["n"] / len(df) * (g["obs"] - g["pred"]).abs()).sum())


def calibration_slope_intercept(y, p):
    """Slope of logit(p) in a logistic recalibration model (ideal 1) and the
    calibration intercept with logit(p) as offset (ideal 0)."""
    y = np.asarray(y)
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y).coef_[0, 0]

    def nll(a):
        q = np.clip(expit(z + a), 1e-12, 1 - 1e-12)
        return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))

    intercept = minimize_scalar(nll, bounds=(-5, 5), method="bounded").x
    return float(slope), float(intercept)


def evaluate(y, p, clv, d, score=None):
    """All Stage 3 metrics for one set of predictions.

    p     : probabilities used for calibration metrics and the profit decision
            (None for an uncalibrated SVM).
    score : ranking score for AUC/PR-AUC/EMP when p is None."""
    y = np.asarray(y)
    rank = p if p is not None else score
    out = {"AUC": roc_auc_score(y, rank),
           "PR-AUC": average_precision_score(y, rank),
           "EMP per customer": emp(y, rank, clv, d)}
    if p is not None:
        slope, intercept = calibration_slope_intercept(y, p)
        contact = contact_decision(p, clv, d)
        out.update({
            "Brier": brier_score_loss(y, p),
            "Log loss": log_loss(y, p),
            "ECE": ece(y, p),
            "Cal. slope": slope,
            "Cal. intercept": intercept,
            "Mean predicted": float(np.mean(p)),
            "Observed rate": float(np.mean(y)),
            "Profit per customer": realised_profit(y, contact, clv, d) / len(y),
            "Targeted share": float(np.mean(contact)),
            "MCC": matthews_corrcoef(y, contact),
        })
    return out


# ---------------------------------------------------------------------------
# Statistical comparison on the test set
# ---------------------------------------------------------------------------
def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong_test(y, s1, s2):
    """DeLong et al. (1988) test for two correlated AUCs. Returns (AUC1 - AUC2, p)."""
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, v01, v10 = [], [], []
    for s in (np.asarray(s1), np.asarray(s2)):
        tx, ty, tz = _midrank(s[pos]), _midrank(s[neg]), _midrank(s)
        aucs.append((tz[pos].sum() - m * (m + 1) / 2) / (m * n))
        v01.append((tz[pos] - tx) / n)
        v10.append(1 - (tz[neg] - ty) / m)
    sx, sy = np.cov(np.vstack(v01)), np.cov(np.vstack(v10))
    var = (sx[0, 0] + sx[1, 1] - 2 * sx[0, 1]) / m + (sy[0, 0] + sy[1, 1] - 2 * sy[0, 1]) / n
    diff = aucs[0] - aucs[1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    return float(diff), float(2 * stats.norm.sf(abs(z)))


def paired_bootstrap(stat, n, n_boot=None, seed=None):
    """Paired bootstrap of a difference stat(idx) between two prediction sets.

    stat : function of an index array returning the difference (A - B).
    Returns (observed difference, 2.5% and 97.5% percentiles, two-sided p)."""
    n_boot = n_boot or C.N_BOOTSTRAP
    rng = np.random.default_rng(C.RANDOM_STATE if seed is None else seed)
    observed = stat(np.arange(n))
    diffs = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return observed, np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), min(1.0, max(p, 1 / n_boot))


def holm(p_values, alpha=0.05):
    """Holm step-down: returns a list of booleans (significant) in the input order."""
    p = np.asarray(p_values)
    order = np.argsort(p)
    m = len(p)
    sig = np.zeros(m, dtype=bool)
    still = True
    for i, idx in enumerate(order):
        still = still and p[idx] < alpha / (m - i)
        sig[idx] = still
    return sig.tolist()
