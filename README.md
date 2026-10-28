# Home Electricity Consumption Prediction & Dashboard ⚡

A machine learning and data analysis project for forecasting household electricity consumption and exploring energy usage patterns using the Individual Household Electric Power Consumption dataset from the UCI Machine Learning Repository.

## 📌 Project Overview

- **Machine Learning Models:** Random Forest, Decision Tree, Linear Regression, Ridge, and Lasso Regressors benchmarked on chronological test splits.
- **Interactive Web App:** A Streamlit dashboard (`app.py`) allowing users to visualize energy usage trends, compare predictive models, and estimate power consumption and tariff costs.
- **Analysis Notebook:** `Electricity_Consumption_Prediction.ipynb` providing end-to-end data preprocessing, exploratory data analysis (EDA), model training, statistical evaluation, and multi-step forecasting.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/24pal/siproject.git
cd siproject
```

### 2. Set up environment & install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```

---

## 📊 Features

- **Dashboard:** At-a-glance metrics, 7/30/90-day power trends, diurnal usage breakdown by hour of day, and model performance comparisons (MAE, RMSE, R²).
- **Consumption Predictor:** Predict next-hour power consumption using real recorded readings, date selection, or manual inputs with custom tariff calculations.
- **Dataset Exploration:** Overview of features including active power, reactive power, voltage, global intensity, and sub-metering breakdown.
