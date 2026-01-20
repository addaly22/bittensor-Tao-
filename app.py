import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime
import os

# ================= CONFIG =================
WINDOW = 10
USD_TO_IDR = 16909.10

FEATURES = [
    'priceOpen','priceHigh','priceLow','volume',
    'hour_sin','hour_cos','day_sin','day_cos','month_sin','month_cos'
]

# ================= LOAD RESOURCE =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = load_model("model_bittensor.keras")
scaler_X = joblib.load("scaler_X_bittensor.pkl")
scaler_y = joblib.load("scaler_y_bittensor.pkl")

df = pd.read_excel("bittensor.xlsx")
df['timeOpen'] = pd.to_datetime(df['timeOpen'])
df = df.sort_values('timeOpen')

# ================= UI =================
st.set_page_config(page_title="Prediksi Harga Bittensor", page_icon="📈")
st.title("📈 Prediksi Harga Crypto Bittensor")

tanggal = st.date_input("Pilih Tanggal")
jam = st.slider("Pilih Jam", 0, 23, 12)

if st.button("Prediksi"):
    cutoff = pd.to_datetime(f"{tanggal} {jam}:00:00")
    df_cut = df[df['timeOpen'] < cutoff].tail(WINDOW)

    if len(df_cut) < WINDOW:
        st.error("❌ Data historis tidak cukup")
    else:
        hour = jam
        day = cutoff.weekday()
        month = cutoff.month

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin  = np.sin(2 * np.pi * day / 7)
        day_cos  = np.cos(2 * np.pi * day / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        X = df_cut[['priceOpen','priceHigh','priceLow','volume']].copy()
        X['hour_sin'] = hour_sin
        X['hour_cos'] = hour_cos
        X['day_sin']  = day_sin
        X['day_cos']  = day_cos
        X['month_sin'] = month_sin
        X['month_cos'] = month_cos

        X = scaler_X.transform(X.values)
        X = X.reshape(1, WINDOW, len(FEATURES))

        pred_scaled = model.predict(X)
        price_usd = scaler_y.inverse_transform(pred_scaled)[0][0]
        price_idr = price_usd * USD_TO_IDR

        st.success("✅ Hasil Prediksi")
        st.metric("USD", f"${price_usd:,.2f}")
        st.metric("IDR", f"Rp {price_idr:,.0f}")
