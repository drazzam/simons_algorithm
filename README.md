
# Simons‑Style Signals — Pro (Streamlit)

**Seven strategies + paper trading**, built around free Yahoo Finance data (`yfinance`).  
Educational research tool only — **not** investment advice.

## Features
- Live screeners for 7 strategies (gap MR, mini‑trend, pairs, residual reversion, microstructure proxy, ensemble, Monday effect).
- Paper‑trading engine with cost/impact model, turnover caps, gross leverage control.
- Top‑10 picks per strategy under current conditions.
- S&P 500 universe from Wikipedia (or custom list).

## Run locally
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Create a **public GitHub repo**, add these files, and push.
2. Go to https://share.streamlit.io/ → *New app* → connect your repo → select `app.py` as the entrypoint.
3. Streamlit Cloud installs `requirements.txt` and runs the app automatically.

## Notes
- Intraday data availability is limited on Yahoo (about 7 days for 1‑minute bars). For broader intraday research, use a paid feed.
- The paper‑trading layer executes **EOD signals at next‑day open** with a simple cost/impact model.
- All models are simplified; the aim is clarity, robustness, and reproducibility.
