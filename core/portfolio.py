
# core/portfolio.py
from typing import Tuple
import pandas as pd
def weights_from_scores(df: pd.DataFrame, long_only: bool=False, top_n: int=10, max_weight: float=0.03) -> pd.Series:
    if df is None or df.empty or "score" not in df.columns: return pd.Series(dtype=float)
    s = df["score"].dropna()
    if long_only:
        s = s[s>0].sort_values(ascending=False).head(top_n)
        w = s.clip(lower=0) / s.clip(lower=0).sum()
    else:
        longs = s.sort_values(ascending=False).head(top_n)
        shorts= s.sort_values(ascending=True ).head(top_n)
        w = pd.concat([ longs/longs.abs().sum(), -shorts/shorts.abs().sum() ])
    w = w.clip(lower=-max_weight, upper=max_weight)
    if long_only:
        if w.sum()!=0: w = w/w.sum()
    else:
        if w.abs().sum()!=0: w = w/w.abs().sum()
    return w
def rebalance_to_targets(prev_positions: pd.Series, target_weights: pd.Series, prices: pd.Series, capital: float, turnover_cap: float=0.3) -> pd.Series:
    prev_positions = prev_positions.reindex(prices.index).fillna(0.0)
    notional = capital * target_weights.reindex(prices.index).fillna(0.0)
    target_shares = notional / prices.clip(lower=1e-9)
    return target_shares
