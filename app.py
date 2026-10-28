import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Home Electricity Predictor", page_icon="⚡", layout="wide")
st.title("⚡ Home Electricity Consumption Predictor")
st.caption("A simple dashboard for exploring household electricity consumption and estimating future usage.")

DATA_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
FEATURES = ["Power (kW)", "Reactive power", "Voltage (V)", "Current (A)", "Kitchen submeter (Wh)", "Laundry submeter (Wh)", "Other submeter (Wh)"]

@st.cache_data(show_spinner="Downloading and preparing the real UCI dataset (first run only)...")
def load_data():
    response = requests.get(DATA_URL, timeout=180)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        name = next(n for n in archive.namelist() if n.endswith("household_power_consumption.txt"))
        raw = pd.read_csv(archive.open(name), sep=";", na_values="?", low_memory=False)
    raw["DateTime"] = pd.to_datetime(raw["Date"] + " " + raw["Time"], dayfirst=True, errors="coerce")
    raw = raw.dropna(subset=["DateTime"]).set_index("DateTime").sort_index()
    cols = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    for c in cols:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw[cols].dropna()
    h = pd.DataFrame(index=raw["Global_active_power"].resample("h").mean().index)
    h[FEATURES[0]] = raw["Global_active_power"].resample("h").mean()
    h[FEATURES[1]] = raw["Global_reactive_power"].resample("h").mean()
    h[FEATURES[2]] = raw["Voltage"].resample("h").mean()
    h[FEATURES[3]] = raw["Global_intensity"].resample("h").mean()
    h[FEATURES[4]] = raw["Sub_metering_1"].resample("h").sum()
    h[FEATURES[5]] = raw["Sub_metering_2"].resample("h").sum()
    h[FEATURES[6]] = raw["Sub_metering_3"].resample("h").sum()
    return h.dropna()

@st.cache_resource(show_spinner="Training and comparing prediction models (first run only)...")
def train_model():
    data = load_data()
    x = data[FEATURES].iloc[:-1]
    y = data[FEATURES[0]].shift(-1).dropna()
    split = int(len(x) * 0.8)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    candidates = {
        "Random Forest": RandomForestRegressor(n_estimators=80, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(max_iter=5000),
        "Decision Tree": DecisionTreeRegressor(min_samples_leaf=5, random_state=42),
    }
    results = []
    fitted = {}
    for name, candidate in candidates.items():
        candidate.fit(x_train, y_train)
        prediction = np.maximum(0, candidate.predict(x_test))
        results.append({
            "Model": name,
            "MAE (kW)": mean_absolute_error(y_test, prediction),
            "RMSE (kW)": np.sqrt(mean_squared_error(y_test, prediction)),
            "R² score": r2_score(y_test, prediction),
        })
        fitted[name] = candidate
    comparison = pd.DataFrame(results).sort_values("MAE (kW)").reset_index(drop=True)
    # Keep Random Forest as the model used by the forecasting page.
    model = fitted["Random Forest"]
    forest_row = comparison.loc[comparison["Model"] == "Random Forest"].iloc[0]
    metrics = {"MAE (kW)": float(forest_row["MAE (kW)" ]), "RMSE (kW)": float(forest_row["RMSE (kW)" ]), "R²": float(forest_row["R² score"])}
    return model, metrics, comparison

try:
    hourly = load_data()
    model, metrics, model_comparison = train_model()
except Exception as e:
    st.error("The dataset could not be downloaded or processed. Check your internet connection and try refreshing the app.")
    st.caption(f"Technical detail: {e}")
    st.stop()

latest_time = hourly.index.max()
latest_row = hourly.loc[[latest_time], FEATURES]

# Display-only date mapping: preserve row order and all measured values, while mapping
# the historical series across Jan 2010–Sep 2026. Calculations still use original rows.
display_start = pd.Timestamp("2010-01-01 00:00:00")
display_end = pd.Timestamp("2026-09-30 23:00:00")
original_ns = hourly.index.view("int64")
if len(hourly) > 1:
    fractions = np.linspace(0.0, 1.0, len(hourly))
    display_ns = display_start.value + (display_end.value - display_start.value) * fractions
    display_index = pd.to_datetime(display_ns.astype("int64"))
else:
    display_index = pd.DatetimeIndex([display_end])
display_dates = pd.Series(display_index, index=hourly.index)
display_latest_time = display_index[-1]

with st.sidebar:
    st.header("Menu")
    page = st.radio("Go to", ["Dashboard", "Predict consumption", "About the data"])
    tariff = st.number_input("Electricity rate (₹ per unit / kWh)", min_value=0.0, value=8.0, step=0.5)
    st.caption("Default rate ₹8 is only an example. Enter your own rate.")

if page == "Dashboard":
    st.subheader("At a glance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Hourly records", f"{len(hourly):,}")
    c2.metric("Latest recorded power", f"{hourly.loc[latest_time, FEATURES[0]]:.2f} kW")
    c3.metric("Latest display date", display_latest_time.strftime("%d %b %Y, %I:%M %p"))

    period = st.radio("Show history for", ["Last 7 days", "Last 30 days", "Last 90 days"], horizontal=True)
    hours_to_show = {"Last 7 days": 24*7, "Last 30 days": 24*30, "Last 90 days": 24*90}[period]
    view = hourly.tail(hours_to_show).copy()
    chart_data = view[[FEATURES[0]]].copy()
    chart_data.index = display_dates.loc[view.index].dt.strftime("%d %b %Y, %I %p")
    st.subheader("Household power over time")
    st.line_chart(chart_data, y_label="Average power (kW)")
    st.caption("Each point is a real recorded hourly value. Dates are mapped to the demonstration timeline (Jan 2010–Sep 2026); they are not the original measurement dates.")

    st.subheader("Which time of day uses more power?")
    hourly_avg = hourly.groupby(hourly.index.hour)[FEATURES[0]].mean()
    labels = [pd.Timestamp("2000-01-01") + pd.Timedelta(hours=int(hr)) for hr in hourly_avg.index]
    by_time = pd.DataFrame({"Time of day": [x.strftime("%I %p").lstrip("0") for x in labels], "Average power (kW)": hourly_avg.values}).set_index("Time of day")
    st.bar_chart(by_time, y_label="Average power (kW)")

    st.subheader("Model comparison")
    st.caption("Models are compared on the same chronological test data. Lower MAE and RMSE are better; a higher R² score is better.")
    st.dataframe(model_comparison.style.format({"MAE (kW)": "{:.3f}", "RMSE (kW)": "{:.3f}", "R² score": "{:.3f}"}), use_container_width=True, hide_index=True)
    st.bar_chart(model_comparison.set_index("Model")[["MAE (kW)", "RMSE (kW)"]], y_label="Prediction error (kW)")

elif page == "Predict consumption":
    st.subheader("1. Choose the readings")
    input_mode = st.radio("How would you like to provide readings?", ["Use latest dataset reading", "Choose a date and time", "Enter readings manually"], horizontal=True)
    if input_mode == "Use latest dataset reading":
        chosen_time = latest_time
        row = latest_row.copy()
        st.success(f"Using latest available real dataset reading (shown as {display_latest_time.strftime('%A, %d %B %Y at %I:%M %p')} on the demo timeline).")
    elif input_mode == "Choose a date and time":
        min_date, max_date = display_index[0].date(), display_index[-1].date()
        chosen_display_date = st.date_input("Choose display date", value=max_date, min_value=min_date, max_value=max_date)
        display_day_mask = pd.Series(display_index.date == chosen_display_date)
        day_positions = np.flatnonzero(display_day_mask.to_numpy())
        if len(day_positions) == 0:
            st.warning("No data points map to that display date. Please choose another date.")
            st.stop()
        time_options = [hourly.index[pos] for pos in day_positions]
        chosen_time = st.selectbox("Choose time", time_options, index=len(time_options)-1, format_func=lambda t: display_dates.loc[t].strftime("%I:%M %p"))
        row = hourly.loc[[chosen_time], FEATURES].copy()
        chosen_display_time = display_dates.loc[chosen_time]
        st.success(f"Selected real reading shown as: {chosen_display_time.strftime('%A, %d %B %Y at %I:%M %p')} (display date only).")
    else:
        st.caption("Enter values from a meter or dataset. Leave the example values unchanged only if you want to test the form.")
        base = latest_row.iloc[0]
        defaults = {}
        for feature in FEATURES:
            defaults[feature] = st.number_input(feature, min_value=0.0, value=float(base[feature]), key=f"manual_{feature}")
        row = pd.DataFrame([defaults], columns=FEATURES)
        chosen_time = latest_time

    with st.expander("See the readings being used"):
        st.dataframe(row.rename_axis("Selected timestamp").reset_index(), use_container_width=True, hide_index=True)

    st.subheader("2. Estimate energy and bill")
    forecast_days = st.slider("Estimate how many days ahead?", min_value=1, max_value=30, value=7)
    st.caption("The app forecasts up to 30 days. Longer forecasts are more uncertain because future household behaviour and weather are unknown.")
    if st.button("Predict consumption", type="primary", use_container_width=True):
        # Simple recursive baseline: future non-power measurements are unknown, so they are held at the selected reading.
        next_row = row[FEATURES].iloc[0].astype(float).to_dict()
        forecast = []
        for step in range(1, forecast_days * 24 + 1):
            predicted_kw = max(0.0, float(model.predict(pd.DataFrame([next_row], columns=FEATURES))[0]))
            stamp = chosen_time + pd.Timedelta(hours=step)
            display_stamp = display_dates.loc[chosen_time] + pd.Timedelta(hours=step)
            forecast.append({"Date and time": display_stamp, "Time": display_stamp.strftime("%d %b %Y, %I %p"), "Predicted average power (kW)": predicted_kw, "Estimated energy (kWh)": predicted_kw})
            next_row[FEATURES[0]] = predicted_kw
        forecast_df = pd.DataFrame(forecast)
        units = float(forecast_df["Estimated energy (kWh)"].sum())
        cost = units * tariff
        daily = forecast_df.copy()
        daily["Date"] = pd.to_datetime(daily["Date and time"]).dt.strftime("%d %b %Y")
        daily_summary = daily.groupby("Date", sort=False).agg(
            **{"Estimated energy (kWh)": ("Estimated energy (kWh)", "sum"),
               "Average predicted power (kW)": ("Predicted average power (kW)", "mean")}
        ).reset_index()
        daily_summary["Estimated cost (₹)"] = daily_summary["Estimated energy (kWh)"] * tariff
        st.subheader("Estimated bill for selected period")
        a, b, c = st.columns(3)
        a.metric("Estimated energy", f"{units:.2f} units (kWh)")
        b.metric("Estimated cost", f"₹{cost:.2f}")
        c.metric("Forecast length", f"{forecast_days} day(s)")
        st.subheader("Daily estimated energy")
        st.bar_chart(daily_summary.set_index("Date")[["Estimated energy (kWh)"]], y_label="Energy (kWh/day)")
        st.subheader("Daily bill estimate")
        st.dataframe(daily_summary, use_container_width=True, hide_index=True)
        st.caption("Calculations use the model's predicted hourly kW values, converted to hourly kWh and multiplied by your tariff. This is an educational estimate, not a guaranteed bill. Forecast dates are display-only dates; readings remain from the historical dataset. Actual bills may differ due to usage changes, tariff slabs, fixed charges and taxes.")

else:
    st.subheader("About the dataset")
    st.write("This app uses the UCI Individual Household Electric Power Consumption dataset. Its real measurements were recorded between December 2006 and November 2010.")
    st.write(f"**Original measurement period:** {hourly.index.min().strftime('%d %B %Y')} to {latest_time.strftime('%d %B %Y')}.")
    st.write(f"**Demonstration display timeline:** {display_index[0].strftime('%d %B %Y')} to {display_index[-1].strftime('%d %B %Y')}.")
    st.subheader("Model quality on held-out historical data")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{metrics['MAE (kW)']:.3f} kW")
    m2.metric("RMSE", f"{metrics['RMSE (kW)']:.3f} kW")
    m3.metric("R² score", f"{metrics['R²']:.3f}")
    st.caption("These scores describe performance on a chronological test portion of historical data; they do not guarantee future accuracy.")
    st.download_button("Download hourly dataset (CSV)", hourly.to_csv(index_label="DateTime").encode("utf-8"), "household_hourly_data.csv", "text/csv")
