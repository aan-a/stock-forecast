import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from prophet import Prophet

st.set_page_config(page_title="AAPL Forecast", layout="wide")
st.title("📈 Apple Stock Forecast with Prophet + Market Features")

start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date   = st.date_input("End Date",   pd.to_datetime("2024-12-31"))

@st.cache_data
def load_data(start, end):
    # 1. Download & flatten MultiIndex
    df = yf.download("AAPL", start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. Keep only Close & Volume, then rename
    df = df[['Close', 'Volume']].copy()
    df.rename(columns={'Close': 'y'}, inplace=True)

    # 3. Reset index → get 'ds' column
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'ds'}, inplace=True)

    # 4. Feature engineering
    df['MA_20']  = df['y'].rolling(window=20).mean()
    df['Return'] = df['y'].pct_change()

    # 5. Clean & convert to numeric
    df['ds']      = pd.to_datetime(df['ds'], errors='coerce')
    for col in ['y', 'Volume', 'MA_20', 'Return']:
        raw = df[col]
        if isinstance(raw, pd.DataFrame):
            raw = raw.squeeze()
        df[col] = pd.to_numeric(raw, errors='coerce')

    # 6. Drop any rows missing essentials
    df.dropna(subset=['ds','y','Volume','MA_20','Return'], inplace=True)
    return df

df = load_data(start_date, end_date)
st.success(f"✅ Loaded {len(df)} rows of clean AAPL data.")

# Build & fit Prophet
model = Prophet()
for reg in ['Volume','MA_20','Return']:
    model.add_regressor(reg)
model.fit(df[['ds','y','Volume','MA_20','Return']])

# Forecast
future = model.make_future_dataframe(periods=60)
future[['Volume','MA_20','Return']] = df[['Volume','MA_20','Return']].iloc[-1].values
forecast = model.predict(future)

# Plot results
st.subheader("📉 Forecast")
st.pyplot(model.plot(forecast))

st.subheader("📊 Components")
st.pyplot(model.plot_components(forecast))

st.subheader("🧮 Latest Forecast")
st.dataframe(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(1))

st.markdown("---")
st.write("Built by Aarna")
