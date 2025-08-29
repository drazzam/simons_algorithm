
# core/papertrade.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np, pandas as pd
from .utils import annualize_sharpe, max_drawdown, cagr, TRADING_DAYS
@dataclass
class CostModel:
    base_spread_bps: float = 3.0
    impact_coef: float = 15.0
    min_bps: float = 1.0
    max_bps: float = 50.0
@dataclass
class BrokerState:
    cash: float
    positions: Dict[str, float] = field(default_factory=dict)
    equity_curve: List[float] = field(default_factory=list)
    dates: List[pd.Timestamp] = field(default_factory=list)
    trade_log: List[dict] = field(default_factory=list)
class PaperBroker:
    def __init__(self, initial_capital: float = 1_000_000.0, cost_model: Optional[CostModel]=None):
        self.state = BrokerState(cash=initial_capital)
        self.initial_capital = initial_capital
        self.cost_model = cost_model or CostModel()
    def mark_to_market(self, prices: pd.Series):
        return self.state.cash + float((pd.Series(self.state.positions) * prices.reindex(self.state.positions.keys()).fillna(0.0)).sum())
    def update(self, dt: pd.Timestamp, price_panel: pd.DataFrame, target_shares: Optional[pd.Series]=None, adv_panel: Optional[pd.Series]=None, sigma_panel: Optional[pd.Series]=None):
        close_prices = price_panel["Close"]
        eq = self.mark_to_market(close_prices)
        self.state.equity_curve.append(eq); self.state.dates.append(dt)
    def results(self):
        eq = pd.Series(self.state.equity_curve, index=pd.to_datetime(self.state.dates), name="equity").sort_index()
        ret = eq.pct_change().fillna(0.0)
        summary = pd.DataFrame([{
            "final_equity": float(eq.iloc[-1]) if not eq.empty else self.initial_capital,
            "total_return_%": float((eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0) if len(eq)>1 else 0.0,
            "CAGR_%": float((cagr(eq) * 100.0) if len(eq)>1 else 0.0),
            "vol_%": float(ret.std(ddof=0) * np.sqrt(TRADING_DAYS) * 100.0),
            "sharpe": float(annualize_sharpe(ret)),
            "max_drawdown_%": float(max_drawdown(eq) * 100.0),
        }])
        return eq, ret, summary, pd.DataFrame(self.state.trade_log)
