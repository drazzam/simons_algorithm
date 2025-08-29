
# core/utils.py
import numpy as np
import pandas as pd
TRADING_DAYS = 252
def annualize_sharpe(daily_ret: pd.Series) -> float:
    r = daily_ret.dropna()
    if r.std(ddof=0) == 0 or len(r) < 5: return 0.0
    return np.sqrt(TRADING_DAYS) * r.mean() / r.std(ddof=0)
def max_drawdown(equity_curve: pd.Series) -> float:
    eq = equity_curve.dropna().astype(float)
    if eq.empty: return 0.0
    cummax = eq.cummax(); dd = (eq - cummax) / cummax
    return float(dd.min())
def cagr(equity_curve: pd.Series, freq_days: int = TRADING_DAYS) -> float:
    eq = equity_curve.dropna().astype(float)
    if eq.empty: return 0.0
    total_return = eq.iloc[-1] / eq.iloc[0] - 1.0 if eq.iloc[0] != 0 else 0.0
    years = max(1e-9, len(eq) / freq_days)
    return (1.0 + total_return) ** (1/years) - 1.0
def safe_div(a,b,eps=1e-12): return a/(b+eps)
