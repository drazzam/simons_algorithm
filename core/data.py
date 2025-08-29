
# core/data.py
from typing import Dict, List
import pandas as pd, yfinance as yf, time
DEFAULT_UNIVERSE = ["AAPL","MSFT","AMZN","NVDA","GOOGL","META"]
def fetch_sp500_tickers() -> List[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        if "BRK-B" not in tickers: tickers.append("BRK-B")
        return sorted(set(tickers))
    except Exception:
        return DEFAULT_UNIVERSE
def download_ohlcv(tickers: List[str], period: str, interval: str) -> Dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        try:
            df = yf.download(t, period=period, interval=interval, auto_adjust=False, progress=False)
            if df is not None and not df.empty:
                out[t] = df.dropna(how="any")
        except Exception:
            pass
        time.sleep(0.01)
    return out
def download_market_proxy(period: str = "1y") -> pd.Series:
    s = yf.download("SPY", period=period, interval="1d", auto_adjust=False, progress=False)["Adj Close"]
    s.name = "SPY"; return s
