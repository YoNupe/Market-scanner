import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import logging

# ── Silence extra logs ──
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ── Page setup ──
st.set_page_config(page_title="Smith'n The Market", layout="wide")
st.title("Smith'n The Market – Fast Scanner")

# ── Inputs ──
ticker_input = st.text_input(
    "Tickers (comma-separated)",
    "SPY, QQQ, TSLA, BTC-USD, ETH-USD"
)
interval = st.selectbox("Interval", ["1m","3m","5m","15m"], index=1)
symbols = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# ── Fetch & indicator computation ──
@st.cache_data(ttl=60)
def fetch(sym):
    # choose intraday period
    if interval == "1m":
        period = "1d"
    else:
        period = "5d"

    df = pd.DataFrame()

    # crypto via Binance
    if sym.endswith("-USD"):
        pair = sym.replace("-USD", "USDT")
        url = f"https://api.binance.com/api/v3/klines?symbol={pair}&interval={interval}&limit=500"
        try:
            data = requests.get(url, timeout=5).json()
            df = pd.DataFrame(data, columns=range(12))[[0,1,2,3,4,5]]
            df.columns = ["open_time","Open","High","Low","Close","Volume"]
            df[["Open","High","Low","Close","Volume"]] = df[["Open","High","Low","Close","Volume"]].astype(float)
            df.index = pd.to_datetime(df["open_time"], unit="ms")
            df.drop(columns="open_time", inplace=True)
        except:
            df = pd.DataFrame()
    else:
        # stocks via yfinance
        try:
            df = yf.download(sym, period=period, interval=interval,
                             progress=False, threads=False)
        except:
            df = pd.DataFrame()

    # fallback to daily if empty
    if df.empty:
        try:
            df = yf.download(sym, period="90d", interval="1d",
                             progress=False, threads=False)
        except:
            return None

    if df is None or df.empty:
        return None

    # compute RSI(14)
    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs       = avg_gain.div(avg_loss).replace([pd.NA,float("inf")],0)
    df["RSI"] = 100 - 100/(1+rs)

    # compute EMAs
    df["EMA9"]  = df["Close"].ewm(span=9,  adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()

    return df.dropna()

# ── Build signals ──
records = []
for s in symbols:
    df = fetch(s)
    if df is None:
        continue

    last  = df.iloc[-1]
    price = float(last["Close"])
    rsi   = float(last["RSI"])
    e9    = float(last["EMA9"])
    e21   = float(last["EMA21"])

    sig = "HOLD"
    if rsi < 30 and e9 > e21:
        sig = "BUY"
    elif rsi > 70 and e9 < e21:
        sig = "SELL"

    records.append({
        "Ticker": s,
        "Price":   round(price,2),
        "RSI":     round(rsi,2),
        "EMA9":    round(e9,2),
        "EMA21":   round(e21,2),
        "Signal":  sig
    })

# ── Display table & chart ──
if records:
    df_signals = pd.DataFrame(records)
    st.dataframe(df_signals)

    choice = st.selectbox("Chart ticker", df_signals["Ticker"].tolist())
    chart_df = fetch(choice)

    if chart_df is None or chart_df.empty:
        st.warning(f"No chart data for {choice}. Try a different interval or wait for market hours.")
    else:
        st.subheader(f"{choice} – Price & EMA Chart")
        # debug: show actual cols
        # st.write("Columns:", chart_df.columns.tolist())

        to_plot = ["Close"] + [c for c in ["EMA9","EMA21"] if c in chart_df.columns]
        if not to_plot:
            st.error("No Close/EMA columns found.")
        else:
            st.line_chart(chart_df[to_plot])
else:
    st.error("⚠️ No data fetched. Check tickers, market hours, or interval.")