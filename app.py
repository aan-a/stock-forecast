import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from prophet import Prophet
from io import BytesIO

st.set_page_config(
    page_title="AAPL Forecast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.header("⏳ Forecast Settings")

start_date = st.sidebar.date_input(
    "Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input(
    "End Date", pd.to_datetime("2024-12-31"))

horizon = st.sidebar.slider(
    "Forecast Horizon (days)", min_value=30, max_value=180, value=60)

st.sidebar.markdown("**Select regressors**")
use_vol = st.sidebar.checkbox("Volume", value=True)
use_ma  = st.sidebar.checkbox("MA 20-day", value=True)
use_ret = st.sidebar.checkbox("Daily Return", value=True)

@st.cache_data
def load_data(start, end):
    df = yf.download("AAPL", start=start, end=end)
    # flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # select and rename
    df = df[["Close", "Volume"]].rename(columns={"Close": "y"})
    df.reset_index(inplace=True)
    df.rename(columns={"Date": "ds"}, inplace=True)

    # features
    df["MA_20"]  = df["y"].rolling(window=20).mean()
    df["Return"] = df["y"].pct_change()

    # enforce types
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    for col in ["y", "Volume", "MA_20", "Return"]:
        raw = df[col]
        if isinstance(raw, pd.DataFrame):
            raw = raw.squeeze()
        df[col] = pd.to_numeric(raw, errors="coerce")

    df.dropna(subset=["ds", "y", "Volume", "MA_20", "Return"], inplace=True)
    return df

df = load_data(start_date, end_date)
st.sidebar.success(f"Loaded {len(df)} rows")

model = Prophet()
regs = []
if use_vol:
    model.add_regressor("Volume"); regs.append("Volume")
if use_ma:
    model.add_regressor("MA_20");  regs.append("MA_20")
if use_ret:
    model.add_regressor("Return"); regs.append("Return")

model.fit(df[["ds", "y"] + regs])

future = model.make_future_dataframe(periods=horizon)
last_vals = df[regs].iloc[-1]
for r in regs:
    future[r] = last_vals[r]

forecast = model.predict(future)

st.title("📈 AAPL Stock Forecast")

latest = forecast.iloc[-1]
c1, c2, c3 = st.columns(3)
c1.metric("Forecast",      f"${latest.yhat:.2f}")
c2.metric("Lower Bound",   f"${latest.yhat_lower:.2f}")
c3.metric("Upper Bound",   f"${latest.yhat_upper:.2f}")

st.subheader("📉 Forecast Plot")
fig1, ax1 = plt.subplots(figsize=(10, 4))
model.plot(forecast, ax=ax1)
ax1.set_title("Prophet Forecast with Confidence Interval")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price (USD)")
st.pyplot(fig1)

st.subheader("📊 Forecast Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

csv_data = forecast.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    "Download Forecast CSV",
    data=csv_data,
    file_name="aapl_forecast.csv",
    mime="text/csv"
)

# PNG download for forecast plot
buf = BytesIO()
fig1.savefig(buf, format="png", bbox_inches="tight")
buf.seek(0)
st.sidebar.download_button(
    "Download Forecast Plot PNG",
    data=buf,
    file_name="forecast_plot.png",
    mime="image/png"
)

st.markdown("---")
st.write("Built by Aarna")
