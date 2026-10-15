from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.title("🌦️ Weather Forecast ML")

CITIES = {
    "Chennai": (13.08, 80.27),
    "Bengaluru": (12.97, 77.59),
    "Delhi": (28.61, 77.21),
    "Mumbai": (19.08, 72.88),
}
COLS = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"]


def add_features(df):
    """Add yesterday and 2-days-ago values as extra inputs."""
    df = df.copy()
    for c in COLS:
        df[c + "_yesterday"] = df[c].shift(1)
        df[c + "_2days_ago"] = df[c].shift(2)
    return df


@st.cache_resource
def train(lat, lon):
    """Download history, train the model, measure the error."""
    end = (date.today() - timedelta(days=7)).isoformat()
    params = {"latitude": lat, "longitude": lon,
              "start_date": "2020-01-01", "end_date": end,
              "daily": ",".join(COLS), "timezone": "auto"}
    data = requests.get("https://archive-api.open-meteo.com/v1/archive",
                        params=params, timeout=30).json()["daily"]

    df = add_features(pd.DataFrame(data))
    df["tomorrow_temp"] = df["temperature_2m_max"].shift(-1)
    df = df.dropna()

    X = df.drop(columns=["time", "tomorrow_temp"])
    y = df["tomorrow_temp"]
    split = int(len(df) * 0.8)  # train on old days, test on newer days

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X[:split], y[:split])

    error = mean_absolute_error(y[split:], model.predict(X[split:]))
    baseline = mean_absolute_error(y[split:], X[split:]["temperature_2m_max"])
    return model, error, baseline, list(X.columns), len(df)


city = st.selectbox("Choose city", list(CITIES))
lat, lon = CITIES[city]

with st.spinner("Training model on 6 years of data..."):
    model, error, baseline, columns, rows = train(lat, lon)

st.subheader("Model accuracy")
c1, c2 = st.columns(2)
c1.metric("Model error", f"{error:.2f} °C")
c2.metric("Simple guess error (tomorrow = today)", f"{baseline:.2f} °C")
st.caption(f"Trained on {rows} days of data. Smaller error is better.")

if st.button("Predict tomorrow"):
    params = {"latitude": lat, "longitude": lon, "daily": ",".join(COLS),
              "past_days": 2, "forecast_days": 1, "timezone": "auto"}
    recent = pd.DataFrame(requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params=params, timeout=30).json()["daily"])

    row = add_features(recent).iloc[[-1]][columns]
    st.metric(f"Predicted max temperature in {city}",
              f"{model.predict(row)[0]:.1f} °C")
    st.subheader("Last 3 days used as input")
    st.dataframe(recent)