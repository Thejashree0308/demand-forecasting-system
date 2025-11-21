Demand Forecasting System using Machine Learning & Time Series

A forecasting application built with Python, Streamlit, and Machine Learning to predict weekly product demand for retail businesses.
It supports ARIMA, Prophet, and XGBoost regression, compares performance, and visualizes trends with interactive graphs.

🚀 Project Overview

Demand forecasting helps businesses prevent stockouts, optimize inventory, and plan supply chain operations.
This project predicts next-week product sales using:

Historical sales

Seasonal trends

Promotions

Price changes

Holiday effects

The app provides visual insights + predictions through a clean Streamlit UI.

🛠 Tech Stack
Machine Learning

ARIMA / SARIMA

Facebook Prophet

XGBoost Regressor

Feature Engineering (lag features, moving averages, rolling windows)

Frontend

Streamlit (interactive UI)

Plotly / Matplotlib (visual graphs)

Backend

Python (Pandas, Scikit-learn, Statsmodels)

📊 Features

✔ Upload retail sales dataset (CSV)
✔ Automatic data cleaning
✔ Time-series decomposition (trend, seasonality)
✔ Train 3 models: ARIMA, Prophet, XGBoost
✔ Compare RMSE, MAPE, MAE
✔ Forecast next week / next month
✔ Beautiful Streamlit UI with charts
✔ Download forecast results

📁 Project Structure
demand_forecasting_system/
│── app.py                     # Streamlit frontend
│── model_training.ipynb       # Experiments & model comparison
│── utils.py                   # Feature engineering utilities
│── requirements.txt           # Dependencies
│── README.md                  # Project documentation
│── data/
│      └── sample_sales.csv    # Example dataset

▶️ How to Run Locally

Make sure you have Python 3.10+ installed.

1. Clone Repository
git clone https://github.com/Thejashree0308/demand-forecasting-system.git
cd demand-forecasting-system

2. Install Dependencies
pip install -r requirements.txt

3. Run Streamlit App
streamlit run app.py

📉 Model Evaluation

Models compared using:

RMSE (Root Mean Squared Error)

MAPE (Mean Absolute Percentage Error)

MAE (Mean Absolute Error)

Streamlit displays:

Error table

Predicted vs Actual graph

Forecast line chart
