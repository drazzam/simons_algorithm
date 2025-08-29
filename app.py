
# app.py (minimal to demonstrate robust signals)
import streamlit as st, pandas as pd
from core.data import fetch_sp500_tickers, download_ohlcv, download_market_proxy
from core.signals import s1_gap_meanrev, s2_minitrend, s3_pairs_cointegration, s4_residual_reversion, s5_micro_reversal, s6_ensemble, s7_monday_effect

st.set_page_config(page_title="Simons Signals — Robust", layout="wide")
tickers = fetch_sp500_tickers()[:50]
daily_data = download_ohlcv(tickers, period="1y", interval="1d")
intraday_data = download_ohlcv(tickers[:60], period="5d", interval="1m")
spy = download_market_proxy(period="1y")
st.write("Daily tickers:", len(daily_data), "Intraday tickers:", len(intraday_data))
s1 = s1_gap_meanrev(daily_data); st.write("S1 top 5", s1.head(5))
s2 = s2_minitrend(daily_data);  st.write("S2 top 5", s2.head(5))
s4 = s4_residual_reversion(daily_data, spy); st.write("S4 top 5", s4.head(5))
s5 = s5_micro_reversal(intraday_data); st.write("S5 top 5", s5.head(5))
s7,_ = s7_monday_effect(daily_data, intraday=intraday_data); st.write("S7 top 5", s7.head(5))
