
# core/signals.py (robust)
import math, itertools
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
from scipy.stats import zscore
from statsmodels.tsa.stattools import coint

def _as_series(df: pd.DataFrame, key: str) -> pd.Series:
    """
    Robustly fetch a 1D Series for OHLCV columns from possibly tricky frames
    (duplicate columns or MultiIndex). Falls back across levels.
    """
    col = None
    if key in df.columns:
        col = df[key]
    elif isinstance(df.columns, pd.MultiIndex):
        # try level 0, then level 1
        try:
            col = df.xs(key, axis=1, level=0)
        except Exception:
            try:
                col = df.xs(key, axis=1, level=1)
            except Exception:
                col = None
    if isinstance(col, pd.DataFrame):
        # pick the first matching column deterministically
        col = col.iloc[:, 0]
    if col is None:
        raise KeyError(f"Column '{key}' not found in DataFrame with columns: {list(df.columns)}")
    return pd.to_numeric(col, errors="coerce")

def _px(df: pd.DataFrame) -> pd.Series:
    try:
        return _as_series(df, "Adj Close").astype(float)
    except Exception:
        return _as_series(df, "Close").astype(float)

# ---------- Strategy 1: Overnight Gap Mean-Reversion ----------
def s1_gap_meanrev(daily: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for t, df in daily.items():
        if df is None or df.empty or len(df) < 40:
            continue
        try:
            open_s = _as_series(df, "Open").astype(float)
            close_s = _as_series(df, "Close").astype(float)
        except Exception:
            continue
        if len(close_s) < 2:
            continue
        ret1d = close_s.pct_change()
        vol20 = ret1d.rolling(20).std().iloc[-1]
        prev_close = close_s.iloc[-2]
        open_last = open_s.iloc[-1]
        if not np.isfinite(prev_close) or prev_close == 0 or not np.isfinite(open_last) or not np.isfinite(vol20) or vol20 == 0:
            continue
        gap_last = (open_last - prev_close) / prev_close
        rows.append({"ticker": t, "gap": float(gap_last), "vol20": float(vol20)})
    if not rows:
        return pd.DataFrame(columns=["score"])
    out = pd.DataFrame(rows).set_index("ticker")
    out["gap_z"] = (out["gap"] - out["gap"].mean()) / (out["gap"].std(ddof=0) + 1e-12)
    out["score"] = -out["gap_z"] / (out["vol20"] + 1e-9)
    return out.sort_values("score", ascending=False)

# ---------- Strategy 2: Mini-Trend (EMA on returns) ----------
def s2_minitrend(daily: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for t, df in daily.items():
        if df is None or df.empty or len(df) < 80:
            continue
        px = _px(df)
        rets = px.pct_change().dropna()
        if len(rets) < 60:
            continue
        ema_s = rets.ewm(span=10, adjust=False).mean().iloc[-1]
        ema_l = rets.ewm(span=40, adjust=False).mean().iloc[-1]
        sigma = rets.rolling(20).std().iloc[-1]
        if not np.isfinite(sigma) or sigma == 0:
            continue
        rows.append({"ticker": t, "slope": float(ema_s - ema_l), "sigma": float(sigma)})
    if not rows:
        return pd.DataFrame(columns=["score"])
    out = pd.DataFrame(rows).set_index("ticker")
    out["score"] = out["slope"] / (out["sigma"] + 1e-9)
    return out.sort_values("score", ascending=False)

# ---------- Strategy 3: Pairs / Cointegration ----------
def s3_pairs_cointegration(daily: Dict[str, pd.DataFrame], max_pairs_from: int = 60) -> pd.DataFrame:
    tickers = list(daily.keys())[:max_pairs_from]
    closes = {}
    for t in tickers:
        df = daily.get(t)
        if df is None or df.empty or len(df) < 120:
            continue
        try:
            closes[t] = _px(df).rename(t)
        except Exception:
            continue
    if len(closes) < 2:
        return pd.DataFrame(columns=["tickA","tickB","pvalue","z","half_life_est","score"]).set_index("pair")
    X = pd.concat(closes.values(), axis=1).dropna(how="any")
    pairs = list(itertools.combinations(X.columns, 2))
    rows = []
    for a, b in pairs:
        x = X[a].astype(float)
        y = X[b].astype(float)
        if x.std() == 0 or y.std() == 0:
            continue
        try:
            _score, pvalue, _ = coint(np.log(x), np.log(y))
        except Exception:
            continue
        if not np.isfinite(pvalue):
            continue
        beta = np.polyfit(np.log(y), np.log(x), 1)[0]
        spread = np.log(x) - beta * np.log(y)
        z = (spread - spread.mean()) / (spread.std(ddof=0) + 1e-12)
        z_latest = float(z.iloc[-1])
        # half-life via AR(1)
        half_life = np.nan
        spr = spread.dropna()
        if len(spr) > 30:
            spr_diff = spr.diff().dropna()
            spr_lag = spr.shift(1).loc[spr_diff.index]
            try:
                bcoef = np.polyfit(spr_lag, spr_diff, 1)[0]
                kappa = -bcoef if bcoef < 0 else 1e-6
                half_life = float(np.log(2) / max(kappa, 1e-6))
            except Exception:
                pass
        rows.append({
            "pair": f"{a}/{b}", "tickA": a, "tickB": b, "pvalue": float(pvalue),
            "z": z_latest, "half_life_est": half_life, "score": abs(z_latest) * (1 if pvalue < 0.2 else 0.5)
        })
    if not rows:
        return pd.DataFrame(columns=["tickA","tickB","pvalue","z","half_life_est","score"]).set_index("pair")
    return pd.DataFrame(rows).set_index("pair").sort_values("score", ascending=False)

# ---------- Strategy 4: Residual Reversion (factor-neutral) ----------
def s4_residual_reversion(daily: Dict[str, pd.DataFrame], market: Optional[pd.Series]) -> pd.DataFrame:
    if market is None or market.isna().all():
        return pd.DataFrame(columns=["score"])
    rows = []
    mret = market.pct_change().dropna()
    for t, df in daily.items():
        if df is None or df.empty or len(df) < 80:
            continue
        r = _px(df).pct_change().dropna()
        idx = mret.index.intersection(r.index)
        r = r.loc[idx]
        mr = mret.loc[idx]
        if len(r) < 60:
            continue
        cov = r.rolling(60).cov(mr).iloc[-1]
        var = mr.rolling(60).var().iloc[-1]
        if var is None or var == 0 or pd.isna(cov) or pd.isna(var):
            continue
        beta = cov / var
        resid = r.iloc[-1] - beta * mr.iloc[-1]
        sigma = r.rolling(20).std().iloc[-1]
        if not np.isfinite(sigma) or sigma == 0:
            continue
        rows.append({"ticker": t, "residual": float(resid), "sigma": float(sigma)})
    if not rows:
        return pd.DataFrame(columns=["score"])
    out = pd.DataFrame(rows).set_index("ticker")
    out["resid_z"] = (out["residual"] - out["residual"].mean()) / (out["residual"].std(ddof=0) + 1e-12)
    out["score"] = -out["resid_z"] / (out["sigma"] + 1e-9)
    return out.sort_values("score", ascending=False)

# ---------- Strategy 5: Microstructure Impact-Reversal (intraday proxy) ----------
def s5_micro_reversal(intraday: Dict[str, pd.DataFrame], lookback_min: int = 60) -> pd.DataFrame:
    rows = []
    for t, df in intraday.items():
        if df is None or df.empty:
            continue
        try:
            d = df.iloc[-lookback_min-10:]
            close_s = _as_series(d, "Close").astype(float)
            high_s  = _as_series(d, "High").astype(float)
            low_s   = _as_series(d, "Low").astype(float)
            vol_s   = _as_series(d, "Volume").astype(float)
        except Exception:
            continue
        if len(close_s) < 6:
            continue
        last5 = close_s.iloc[-1] / close_s.iloc[-6] - 1.0
        vol_avg = vol_s.iloc[-lookback_min:].mean() if len(vol_s) >= lookback_min else vol_s.mean()
        vol_surge = vol_s.iloc[-1] / (vol_avg + 1e-12)
        intraday_range = (high_s.iloc[-1] - low_s.iloc[-1]) / max(1e-9, close_s.iloc[-1])
        score = (-last5) * vol_surge / (intraday_range + 1e-9)
        rows.append({"ticker": t, "last5m_ret": float(last5), "vol_surge": float(vol_surge), "range_pct": float(intraday_range), "score": float(score)})
    if not rows:
        return pd.DataFrame(columns=["score"])
    out = pd.DataFrame(rows).set_index("ticker")
    out["score"] = (out["score"] - out["score"].mean()) / (out["score"].std(ddof=0) + 1e-12)
    return out.sort_values("score", ascending=False)

# ---------- Strategy 6: Ensemble of Many Tiny Edges ----------
def s6_ensemble(signal_frames: dict, weights: Optional[dict]=None) -> pd.DataFrame:
    usable = {}
    for name, df in signal_frames.items():
        if df is None or df.empty or "score" not in df.columns:
            continue
        s = df["score"].copy()
        s = (s - s.mean()) / (s.std(ddof=0) + 1e-12)
        usable[name] = s
    if not usable:
        return pd.DataFrame(columns=["score"])
    M = pd.concat(usable.values(), axis=1).fillna(0.0)
    if weights is None:
        weights = {k: 1.0 for k in M.columns}
    w = pd.Series(weights).reindex(M.columns).fillna(0.0)
    meta = (M * w).sum(axis=1)
    return pd.DataFrame({"score": meta}).sort_values("score", ascending=False)

# ---------- Strategy 7: Monday Effect ----------
def s7_monday_effect(daily: Dict[str, pd.DataFrame], intraday: Optional[Dict[str, pd.DataFrame]]=None, tz="America/New_York"):
    today = pd.Timestamp.now(tz=tz).tz_localize(None).date()
    is_monday = pd.Timestamp(today).weekday() == 0
    rows = []
    for t, df in daily.items():
        if df is None or df.empty or len(df) < 10:
            continue
        try:
            d = df.copy()
            d.index = pd.to_datetime(d.index)
            close_s = _as_series(d, "Close").astype(float)
            high_s  = _as_series(d, "High").astype(float)
            low_s   = _as_series(d, "Low").astype(float)
        except Exception:
            continue
        # ATR14
        tr1 = (high_s - low_s).abs()
        tr2 = (high_s - close_s.shift(1)).abs()
        tr3 = (low_s  - close_s.shift(1)).abs()
        atr14 = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean().iloc[-1]
        if not np.isfinite(atr14) or atr14 == 0:
            continue
        last_idx = close_s.index[-4:]
        # Find Thursday and Friday among last 4 sessions
        try:
            idx_fri = [i for i in last_idx if i.weekday() == 4][-1]
            idx_thu = [i for i in last_idx if i.weekday() == 3][-1]
        except Exception:
            continue
        last_close = close_s.iloc[-1]
        if intraday is not None and t in intraday and not intraday[t].empty:
            try:
                last_close = _as_series(intraday[t], "Close").astype(float).iloc[-1]
            except Exception:
                pass
        fri_close = float(close_s.loc[idx_fri])
        thu_close = float(close_s.loc[idx_thu])
        cond2 = fri_close < thu_close
        cond3 = last_close < fri_close
        if cond2 and cond3 and is_monday:
            two_day_drop = (fri_close/thu_close - 1.0)
            today_vs_fri = (last_close/fri_close - 1.0)
            mag = abs(two_day_drop) + abs(today_vs_fri)
            score = -mag / ((atr14/thu_close) + 1e-9)
            rows.append({"ticker": t, "two_day_drop": float(two_day_drop), "today_vs_fri": float(today_vs_fri), "atr14": float(atr14), "score": float(score)})
    df = pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame(columns=["score"])
    return df.sort_values("score", ascending=False), is_monday
