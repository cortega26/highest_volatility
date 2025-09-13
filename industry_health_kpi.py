
"""
industry_health_kpi.py

Compute a bias-minimized composite KPI of company financial health.

Usage example:
    python - <<'PY'
    import pandas as pd
    from industry_health_kpi import compute_industry_health_kpi

    df = pd.read_csv("your_financials.csv")
    feature_cols = [
        "net_margin","roe","current_ratio","quick_ratio",
        "debt_to_equity","interest_coverage","fcf_margin","asset_turnover"
    ]
    higher_is_better = {
        "net_margin": True,
        "roe": True,
        "current_ratio": True,
        "quick_ratio": True,
        "debt_to_equity": False,   # lower is better
        "interest_coverage": True,
        "fcf_margin": True,
        "asset_turnover": True
    }
    res = compute_industry_health_kpi(
        df, feature_cols,
        company_col="company", date_col="date", sector_col="sector",
        label_col=None,  # or "distress_flag" if available (1 = distress)
        higher_is_better=higher_is_better,
        do_smoothing=True, smoothing_window=4,
        winsorize=True, lower_q=0.01, upper_q=0.99,
        use_bootstrap=True, n_boot=200
    )
    res.df_scores.to_csv("health_scores.csv", index=False)
    print(res.meta)
    PY
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict

try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def robust_sector_scale(df: pd.DataFrame, sector_col: str, feature_cols: List[str]):
    params = defaultdict(dict)
    df_scaled = df.copy()
    for sec, g in df.groupby(sector_col):
        for f in feature_cols:
            med = g[f].median(skipna=True)
            iqr = g[f].quantile(0.75) - g[f].quantile(0.25)
            if pd.isna(iqr) or iqr == 0:
                iqr = 1.0
            params[sec][f] = (med, iqr)
            df_scaled.loc[g.index, f] = (g[f] - med) / iqr
    return df_scaled, params


def winsorize_within_sector(df: pd.DataFrame, sector_col: str, feature_cols: List[str], lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
    df_w = df.copy()
    for sec, g in df.groupby(sector_col):
        for f in feature_cols:
            lo = g[f].quantile(lower_q)
            hi = g[f].quantile(upper_q)
            df_w.loc[g.index, f] = g[f].clip(lower=lo, upper=hi)
    return df_w


def rolling_company_mean(df: pd.DataFrame, company_col: str, date_col: str, feature_cols: List[str], window: int = 4) -> pd.DataFrame:
    df_s = df.copy()
    df_s = df_s.sort_values([company_col, date_col])
    for f in feature_cols:
        df_s[f] = df_s.groupby(company_col, group_keys=False)[f].rolling(
            window, min_periods=1).mean().reset_index(level=0, drop=True)
    return df_s


def align_feature_direction(df: pd.DataFrame, feature_cols: List[str], higher_is_better: Dict[str, bool]):
    mults = {}
    df_a = df.copy()
    for f in feature_cols:
        mult = 1 if higher_is_better.get(f, True) else -1
        mults[f] = mult
        df_a[f] = df_a[f] * mult
    return df_a, mults


@dataclass
class SupervisedResult:
    score: pd.Series
    contribs: pd.DataFrame
    model_info: Dict


def supervised_sector_logit(df_scaled, sector_col, feature_cols, label_col, company_col, C=1.0, max_iter=200, n_splits=5):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError(
            "scikit-learn not available; supervised mode cannot run.")

    preds = pd.Series(index=df_scaled.index, dtype=float)
    all_contribs = pd.DataFrame(
        index=df_scaled.index, columns=feature_cols, dtype=float)
    sector_models = {}

    for sec, g in df_scaled.groupby(sector_col):
        X = g[feature_cols].values
        y = g[label_col].values.astype(int)
        if len(np.unique(y)) < 2 or len(g) < (n_splits + 2):
            lr = LogisticRegression(
                C=C, penalty='l2', solver='lbfgs', max_iter=max_iter, class_weight='balanced')
            lr.fit(X, y)
            proba = lr.predict_proba(X)[:, 1]
            contrib = (X * lr.coef_[0]).astype(float)
            all_contribs.loc[g.index, feature_cols] = contrib
            preds.loc[g.index] = 1.0 - proba
            sector_models[sec] = {"coef": lr.coef_[
                0].tolist(), "intercept": float(lr.intercept_[0])}
        else:
            skf = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=42)
            lr = LogisticRegression(
                C=C, penalty='l2', solver='lbfgs', max_iter=max_iter, class_weight='balanced')
            proba_cv = cross_val_predict(
                lr, X, y, cv=skf, method='predict_proba')[:, 1]
            lr.fit(X, y)
            contrib = (X * lr.coef_[0]).astype(float)
            all_contribs.loc[g.index, feature_cols] = contrib
            preds.loc[g.index] = 1.0 - proba_cv
            sector_models[sec] = {"coef": lr.coef_[
                0].tolist(), "intercept": float(lr.intercept_[0])}

    score = (preds.clip(0, 1) * 100).astype(float)
    return SupervisedResult(score=score, contribs=all_contribs, model_info={"sector_models": sector_models})


@dataclass
class UnsupervisedResult:
    score: pd.Series
    contribs: pd.DataFrame
    model_info: Dict


def unsupervised_sector_pca(df_scaled, sector_col, feature_cols, variance_target=0.7):
    if not SKLEARN_AVAILABLE:
        raw = df_scaled.groupby(sector_col)[
            feature_cols].transform('mean').sum(axis=1)
        score = raw.groupby(df_scaled[sector_col]).transform(
            lambda s: 100*(s.rank(pct=True)))
        contribs = df_scaled[feature_cols].copy()
        return UnsupervisedResult(score=score, contribs=contribs, model_info={"note": "SKLearn unavailable; equal-weight fallback."})

    from sklearn.decomposition import PCA
    pc_scores = pd.Series(index=df_scaled.index, dtype=float)
    contribs = pd.DataFrame(index=df_scaled.index,
                            columns=feature_cols, dtype=float)
    sector_info = {}

    for sec, g in df_scaled.groupby(sector_col):
        X = g[feature_cols].values
        pca = PCA(svd_solver='full', random_state=42)
        pca.fit(X)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        k = int(np.searchsorted(cumvar, variance_target)) + 1
        k = min(k, len(feature_cols))
        pcs = pca.transform(X)[:, :k]
        weights = pca.explained_variance_ratio_[:k]
        composite = (pcs * weights).sum(axis=1)

        loadings = pca.components_[:k, :]
        approx_contrib = (pcs[:, :, None] * loadings[None, :, :]).sum(axis=1)
        contribs.loc[g.index, feature_cols] = approx_contrib

        s = pd.Series(composite, index=g.index)
        lo = s.quantile(0.02)
        hi = s.quantile(0.98)
        s_clip = s.clip(lo, hi)
        pc_scores.loc[g.index] = 100 * (s_clip.rank(pct=True))

        sector_info[sec] = {"k_components": int(
            k), "explained_variance_ratio": pca.explained_variance_ratio_[:k].tolist()}

    return UnsupervisedResult(score=pc_scores.astype(float), contribs=contribs, model_info=sector_info)


def bootstrap_scores(scores: pd.Series, group: pd.Series, n_boot=200, random_state=42):
    rng = np.random.default_rng(random_state)
    se = pd.Series(0.0, index=scores.index)
    for sec, idx in group.groupby(group).groups.items():
        idx = list(idx)
        if len(idx) < 5:
            se.loc[idx] = float(scores.loc[idx].std(ddof=1)
                                ) if len(idx) > 1 else 0.0
            continue
        boot = []
        for _ in range(n_boot):
            sample_idx = rng.choice(idx, size=len(idx), replace=True)
            boot.append(scores.loc[sample_idx].values)
        boot = np.array(boot)
        se_vals = boot.std(axis=0, ddof=1)
        se.loc[idx] = se_vals
    return se


@dataclass
class KPIResult:
    df_scores: pd.DataFrame
    meta: Dict


def compute_industry_health_kpi(
    df: pd.DataFrame,
    feature_cols: List[str],
    company_col: str = "company",
    date_col: str = "date",
    sector_col: str = "sector",
    label_col: Optional[str] = None,
    higher_is_better: Optional[Dict[str, bool]] = None,
    do_smoothing: bool = True,
    smoothing_window: int = 4,
    winsorize: bool = True,
    lower_q: float = 0.01, upper_q: float = 0.99,
    use_bootstrap: bool = True,
    n_boot: int = 200
) -> KPIResult:

    df_proc = df.copy()

    for f in feature_cols:
        df_proc[f] = df_proc.groupby(sector_col)[f].transform(
            lambda s: s.fillna(s.median()))

    if do_smoothing:
        df_proc = rolling_company_mean(
            df_proc, company_col, date_col, feature_cols, window=smoothing_window)

    if winsorize:
        df_proc = winsorize_within_sector(
            df_proc, sector_col, feature_cols, lower_q, upper_q)

    df_scaled, scale_params = robust_sector_scale(
        df_proc, sector_col, feature_cols)

    if higher_is_better is None:
        higher_is_better = {f: True for f in feature_cols}
    df_aligned, multipliers = align_feature_direction(
        df_scaled, feature_cols, higher_is_better)

    if label_col is not None and label_col in df.columns and df[label_col].notna().any():
        sup = supervised_sector_logit(
            df_aligned, sector_col, feature_cols, label_col, company_col)
        score = sup.score
        contribs = sup.contribs
        model_info = {"mode": "supervised", **sup.model_info}
    else:
        unsup = unsupervised_sector_pca(df_aligned, sector_col, feature_cols)
        score = unsup.score
        contribs = unsup.contribs
        model_info = {"mode": "unsupervised", **unsup.model_info}

    se = bootstrap_scores(score, df_aligned[sector_col], n_boot=n_boot) if use_bootstrap else pd.Series(
        0.0, index=score.index)

    out = df.copy()
    out["health_score"] = score
    out["health_se"] = se
    lo = out["health_score"].quantile(0.01)
    hi = out["health_score"].quantile(0.99)
    out["health_score"] = ((out["health_score"].clip(
        lo, hi) - lo) / max(hi - lo, 1e-9) * 100).astype(float)

    contrib_df = contribs.copy()
    contrib_df.columns = [f"contrib_{c}" for c in contrib_df.columns]
    result = pd.concat([out, contrib_df], axis=1)

    meta = {
        "feature_multipliers_(+1=higher_is_better)": multipliers,
        "robust_scale_params_(median, IQR)": {sec: {f: (float(m), float(i)) for f, (m, i) in secmap.items()} for sec, secmap in scale_params.items()},
        "model_info": model_info,
        "notes": [
            "Scores are sector-relative with robust scaling to minimize cross-industry bias.",
            "Supervised mode uses regularized logistic regression with class balancing; unsupervised uses PCA with variance-weighted PCs.",
            "Rolling smoothing and winsorization reduce noise and outlier impact.",
            "Uncertainty (health_se) is an empirical bootstrap SE within sector."
        ]
    }

    return KPIResult(df_scores=result, meta=meta)
