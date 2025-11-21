import io
import base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import timedelta

from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

from prophet import Prophet
from xgboost import XGBRegressor
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def preprocess_data(df, date_col, target_col, product_col=None, selected_product=None):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, target_col])

    if product_col is not None and selected_product is not None:
        df = df[df[product_col] == selected_product]

    df = df.sort_values(date_col)
    df = df.groupby(date_col)[target_col].sum().reset_index()
    df = df.rename(columns={date_col: "ds", target_col: "y"})

    inferred_freq = pd.infer_freq(df["ds"])
    if inferred_freq is None:
        inferred_freq = "D"

    df = df.set_index("ds").asfreq(inferred_freq)

    y = df["y"].copy()
    stockout_mask = (y == 0) & (y.shift(1) > 0) & (y.shift(-1) > 0)
    y[stockout_mask] = np.nan

    y = y.interpolate(method="linear").ffill().bfill()

    q1, q3 = y.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr > 0:
        low_iqr = q1 - 1.5 * iqr
        high_iqr = q3 + 1.5 * iqr
        mean = y.mean()
        std = y.std(ddof=0)
        if std > 0:
            low_z = mean - 3 * std
            high_z = mean + 3 * std
            low_cap = max(low_iqr, low_z)
            high_cap = min(high_iqr, high_z)
        else:
            low_cap = low_iqr
            high_cap = high_iqr
        y = y.clip(lower=low_cap, upper=high_cap)

    df["y"] = y
    df = df.reset_index()
    return df


def plot_time_series(df):
    fig = px.line(df, x="ds", y="y", title="Sales Over Time", labels={"ds": "Date", "y": "Sales"})
    fig.update_layout(template="plotly_white")
    return fig


def plot_distribution(df):
    fig = px.histogram(df, x="y", nbins=30, title="Sales Distribution", labels={"y": "Sales"})
    fig.update_layout(template="plotly_white")
    return fig


def plot_day_month_patterns(df):
    temp = df.copy()
    temp["day_of_week"] = temp["ds"].dt.day_name()
    temp["month"] = temp["ds"].dt.month_name().str.slice(stop=3)

    fig1 = px.box(temp, x="day_of_week", y="y", title="Sales by Day of Week")
    fig2 = px.box(temp, x="month", y="y", title="Sales by Month")
    fig1.update_layout(template="plotly_white")
    fig2.update_layout(template="plotly_white")
    return fig1, fig2


def plot_month_year_heatmap(df):
    temp = df.copy()
    temp["year"] = temp["ds"].dt.year
    temp["month"] = temp["ds"].dt.month
    pivot = temp.pivot_table(index="year", columns="month", values="y", aggfunc="sum")

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[f"{m:02d}" for m in pivot.columns],
            y=pivot.index.astype(str),
            colorscale="Purples",
            colorbar_title="Sales",
        )
    )
    fig.update_layout(
        title="Month vs Year Sales Heatmap",
        xaxis_title="Month",
        yaxis_title="Year",
        template="plotly_white",
    )
    return fig


def seasonal_decomposition_plot(df):
    df_sd = df.set_index("ds")["y"].asfreq("D")
    df_sd = df_sd.interpolate(method="linear")
    result = seasonal_decompose(df_sd, model="additive", period=7)

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    result.observed.plot(ax=axes[0], title="Observed")
    result.trend.plot(ax=axes[1], title="Trend")
    result.seasonal.plot(ax=axes[2], title="Seasonal")
    result.resid.plot(ax=axes[3], title="Residual")
    plt.tight_layout()
    return fig


def train_test_split_time_series(df, test_size):
    df = df.copy()
    if isinstance(test_size, int):
        split_point = len(df) - test_size
    else:
        split_point = int(len(df) * (1 - test_size))
    train = df.iloc[:split_point]
    test = df.iloc[split_point:]
    return train, test


def train_arima(train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
    model = SARIMAX(train["y"], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    return results


def forecast_arima(model, train, steps):
    pred = model.get_forecast(steps=steps)
    pred_mean = pred.predicted_mean
    conf_int = pred.conf_int(alpha=0.05)

    last_date = train["ds"].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=steps, freq="D")

    df_forecast = pd.DataFrame({
        "ds": future_dates,
        "yhat": pred_mean.values,
        "yhat_lower": conf_int.iloc[:, 0].values,
        "yhat_upper": conf_int.iloc[:, 1].values,
    })
    return df_forecast


def train_prophet(train):
    m = Prophet(interval_width=0.95, daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
    m.fit(train[["ds", "y"]])
    return m


def forecast_prophet(model, train, periods):
    future = model.make_future_dataframe(periods=periods, freq="D")
    forecast = model.predict(future)
    forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    forecast_future = forecast[forecast["ds"] > train["ds"].max()]
    return forecast_future, forecast


def create_lag_features(df, feature_df=None, n_lags=7):
    df_lag = df.copy()
    for lag in range(1, n_lags + 1):
        df_lag[f"lag_{lag}"] = df_lag["y"].shift(lag)
    df_lag = df_lag.dropna().reset_index(drop=True)
    if feature_df is not None:
        df_lag = df_lag.merge(feature_df, on="ds", how="left")
    return df_lag


def train_xgboost(train_lag, test_lag):
    feature_cols = [c for c in train_lag.columns if c not in ["ds", "y"]]
    X_train, y_train = train_lag[feature_cols], train_lag["y"]
    X_test, y_test = test_lag[feature_cols], test_lag["y"]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred, feature_cols


def rolling_forecast_xgboost(model, df, feature_cols, horizon):
    history = df.copy().set_index("ds")
    forecasts = []
    last_date = history.index.max()

    current_series = history["y"].copy()
    for i in range(horizon):
        lags = []
        for lag in range(1, len(feature_cols) + 1):
            lags.append(current_series.iloc[-lag])
        X = np.array(lags).reshape(1, -1)
        yhat = model.predict(X)[0]
        next_date = last_date + timedelta(days=i + 1)
        forecasts.append((next_date, yhat))
        current_series.loc[next_date] = yhat

    df_forecast = pd.DataFrame(forecasts, columns=["ds", "yhat"])
    df_forecast["yhat_lower"] = np.nan
    df_forecast["yhat_upper"] = np.nan
    return df_forecast


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape_val = mape(y_true, y_pred)
    return mae, rmse, mape_val


def get_table_download_link(df, filename="forecast.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download Forecast as CSV</a>'
    return href
def main():
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("Upload a retail sales CSV to begin.")

        uploaded_file = st.file_uploader("Upload Sales CSV", type=["csv"])

        st.markdown("---")
        st.markdown("### 🔮 Model & Forecast Settings")
        model_choice = st.selectbox(
            "Choose Forecasting Model",
            ["Auto (Best)", "ARIMA", "Prophet", "XGBoost"],
            index=1,
        )
        horizon = st.slider("Forecast Horizon (days)", 7, 60, 30, step=1)

        st.markdown("---")
        st.markdown("### 📊 Train/Test Split")
        test_size_days = st.slider("Test size (last N days)", 7, 90, 30, step=1)

        st.markdown("---")
        st.markdown("### 🧠 Actions")
        train_button = st.button("🚀 Train Model & Forecast", use_container_width=True)

    st.markdown(
        """
        <h1 style="text-align:center; background: -webkit-linear-gradient(45deg,#7F00FF,#E100FF); -webkit-background-clip: text; color: transparent;">
        Retail Demand Forecasting System
        </h1>
        <p style="text-align:center; font-size:16px;">
        Interactive time series forecasting for retail sales using ARIMA, Prophet, and XGBoost.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    if uploaded_file is None:
        st.info("Upload a CSV file from the sidebar to get started.")
        return

    # Read data
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return

    st.subheader("📄 Dataset Preview")
    st.write(raw_df.head())

    if raw_df.empty:
        st.error("Uploaded dataset is empty.")
        return

    with st.expander("🧮 Column Configuration", expanded=True):
        date_col = st.selectbox("Select Date Column", options=raw_df.columns)
        target_col = st.selectbox("Select Target (Sales) Column", options=raw_df.columns)

        product_col = st.selectbox(
            "Optional Product / Item Column (for filtering)",
            options=["None"] + list(raw_df.columns),
            index=0,
        )

        selected_product = None
        if product_col != "None":
            unique_values = raw_df[product_col].dropna().unique()
            if len(unique_values) > 0:
                selected_product = st.selectbox("Select Product / Item", options=unique_values)

        exclude_cols = {date_col, target_col}
        if product_col != "None":
            exclude_cols.add(product_col)
        candidate_extra = [c for c in raw_df.columns if c not in exclude_cols]
        extra_feature_cols = st.multiselect(
            "Optional external feature columns (holidays/promos/weather/etc. for XGBoost)",
            options=candidate_extra,
        )

    try:
        df = preprocess_data(
            raw_df,
            date_col=date_col,
            target_col=target_col,
            product_col=None if product_col == "None" else product_col,
            selected_product=selected_product,
        )
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return

    feature_df_for_xgb = df[["ds"]].copy()
    feature_df_for_xgb["day_of_week"] = feature_df_for_xgb["ds"].dt.weekday
    feature_df_for_xgb["month"] = feature_df_for_xgb["ds"].dt.month
    feature_df_for_xgb["is_weekend"] = feature_df_for_xgb["day_of_week"].isin([5, 6]).astype(int)

    if extra_feature_cols:
        ext = raw_df.copy()
        ext[date_col] = pd.to_datetime(ext[date_col], errors="coerce")
        ext = ext.dropna(subset=[date_col])
        if product_col != "None" and selected_product is not None:
            ext = ext[ext[product_col] == selected_product]
        agg_dict = {col: "mean" for col in extra_feature_cols}
        ext_agg = ext.groupby(date_col)[extra_feature_cols].mean().reset_index()
        ext_agg = ext_agg.rename(columns={date_col: "ds"})
        feature_df_for_xgb = feature_df_for_xgb.merge(ext_agg, on="ds", how="left")
        feature_df_for_xgb = feature_df_for_xgb.ffill().bfill()

    st.success("Data successfully preprocessed for time series analysis.")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🤖 Model & Metrics", "📈 Forecast", "📦 Optimization"])

    with tab1:
        st.markdown("### Exploratory Data Analysis")

        with st.expander("📈 Sales Over Time", expanded=True):
            fig_ts = plot_time_series(df)
            st.plotly_chart(fig_ts, use_container_width=True)

        with st.expander("🧬 Seasonality & Decomposition", expanded=False):
            try:
                fig_decomp = seasonal_decomposition_plot(df)
                st.pyplot(fig_decomp, clear_figure=True)
            except Exception as e:
                st.warning(f"Seasonal decomposition could not be computed: {e}")

        with st.expander("📦 Distributions & Patterns", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                fig_dist = plot_distribution(df)
                st.plotly_chart(fig_dist, use_container_width=True)
                # Additional graph: rolling mean to smooth the series
                rolling_window = min(30, max(3, len(df) // 20))
                df_roll = df.copy()
                df_roll["rolling_mean"] = df_roll["y"].rolling(window=rolling_window).mean()
                fig_roll = px.line(
                    df_roll,
                    x="ds",
                    y="rolling_mean",
                    title=f"Rolling Mean (window={rolling_window} days)",
                    labels={"ds": "Date", "rolling_mean": "Sales (Smoothed)"},
                )
                fig_roll.update_layout(template="plotly_white")
                st.plotly_chart(fig_roll, use_container_width=True)
            with col2:
                fig_dow, fig_mon = plot_day_month_patterns(df)
                st.plotly_chart(fig_dow, use_container_width=True)
                st.plotly_chart(fig_mon, use_container_width=True)

        with st.expander("🔥 Month vs Year Heatmap", expanded=False):
            fig_heat = plot_month_year_heatmap(df)
            st.plotly_chart(fig_heat, use_container_width=True)

    train, test = train_test_split_time_series(df, test_size=test_size_days)

    forecast_df = None
    backtest_pred = None
    metrics = None
    all_model_metrics = None
    selected_model_name = None

    if train_button:
        with st.spinner("Training selected model and generating forecast..."):
            def run_arima(train_df, test_df, full_df):
                model = train_arima(train_df)
                steps_test_local = len(test_df)
                df_test_fc = forecast_arima(model, train_df, steps=steps_test_local)
                bt_pred = df_test_fc.set_index("ds").loc[test_df["ds"]]["yhat"].values
                mae_l, rmse_l, mape_l = compute_metrics(test_df["y"].values, bt_pred)
                fc_full = forecast_arima(model, full_df, steps=horizon)
                return bt_pred, fc_full, {"MAE": mae_l, "RMSE": rmse_l, "MAPE": mape_l}

            def run_prophet(train_df, test_df, full_df):
                model = train_prophet(train_df)
                total_periods_local = len(test_df)
                forecast_future_local, full_forecast_local = forecast_prophet(model, train_df, periods=total_periods_local)
                merged_local = test_df.merge(forecast_future_local[["ds", "yhat"]], on="ds", how="left")
                bt_pred = merged_local["yhat"].values
                mae_l, rmse_l, mape_l = compute_metrics(merged_local["y"].values, bt_pred)
                forecast_df_future_local, full_forecast_final_local = forecast_prophet(model, full_df, periods=horizon)
                if forecast_df_future_local is None or forecast_df_future_local.empty:
                    fc_full = full_forecast_final_local.tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
                else:
                    fc_full = forecast_df_future_local
                return bt_pred, fc_full, {"MAE": mae_l, "RMSE": rmse_l, "MAPE": mape_l}

            def run_xgb(train_df, test_df, full_df):
                lag_data_local = create_lag_features(full_df, feature_df=feature_df_for_xgb, n_lags=7)
                lag_train_local = lag_data_local[lag_data_local["ds"] <= train_df["ds"].max()]
                lag_test_local = lag_data_local[lag_data_local["ds"] > train_df["ds"].max()]
                if lag_test_local.empty:
                    return None, None, None
                model, y_pred_test_local, feature_cols_local = train_xgboost(lag_train_local, lag_test_local)
                mae_l, rmse_l, mape_l = compute_metrics(lag_test_local["y"].values, y_pred_test_local)
                fc_full = rolling_forecast_xgboost(model, full_df, feature_cols_local, horizon=horizon)
                return y_pred_test_local, fc_full, {"MAE": mae_l, "RMSE": rmse_l, "MAPE": mape_l}

            results = {}

            if model_choice in ("ARIMA", "Auto (Best)"):
                bt_arima, fc_arima, met_arima = run_arima(train, test, df)
                results["ARIMA"] = (bt_arima, fc_arima, met_arima)

            if model_choice in ("Prophet", "Auto (Best)"):
                bt_prophet, fc_prophet, met_prophet = run_prophet(train, test, df)
                results["Prophet"] = (bt_prophet, fc_prophet, met_prophet)

            if model_choice in ("XGBoost", "Auto (Best)"):
                bt_xgb, fc_xgb, met_xgb = run_xgb(train, test, df)
                if bt_xgb is not None:
                    results["XGBoost"] = (bt_xgb, fc_xgb, met_xgb)

            if model_choice == "Auto (Best)":
                if not results:
                    st.error("No models could be trained. Check data length and settings.")
                else:
                    best_name = None
                    best_mape = None
                    for name, (_, _, met) in results.items():
                        mape_val_local = met.get("MAPE")
                        if mape_val_local is not None and not np.isnan(mape_val_local):
                            if best_mape is None or mape_val_local < best_mape:
                                best_mape = mape_val_local
                                best_name = name
                    if best_name is None:
                        best_name = list(results.keys())[0]
                    backtest_pred, forecast_df, metrics = results[best_name]
                    selected_model_name = best_name
                    all_model_metrics = {name: met for name, (_, _, met) in results.items()}
            else:
                if model_choice in results:
                    backtest_pred, forecast_df, metrics = results[model_choice]
                    selected_model_name = model_choice
                    all_model_metrics = {model_choice: metrics}

        if metrics is not None:
            st.success("Model training & forecasting complete.")

    with tab2:
        st.markdown("### Model Performance & Backtest")

        if metrics is None:
            st.info("Train a model from the sidebar to see metrics.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{metrics['MAE']:.2f}")
            col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
            col3.metric("MAPE (%)", f"{metrics['MAPE']:.2f}")

            if all_model_metrics is not None and len(all_model_metrics) > 1:
                st.markdown("#### Model Comparison (Auto Selection)")
                comp_rows = []
                for name, met in all_model_metrics.items():
                    comp_rows.append({
                        "Model": name,
                        "MAE": met["MAE"],
                        "RMSE": met["RMSE"],
                        "MAPE": met["MAPE"],
                    })
                comp_df = pd.DataFrame(comp_rows).sort_values("MAPE")
                st.dataframe(comp_df.reset_index(drop=True))
                if selected_model_name is not None:
                    st.markdown(f"Selected model: **{selected_model_name}**")

            st.markdown("#### Actual vs Predicted (Backtest)")
            bt_df = test.copy()
            bt_df["yhat"] = backtest_pred
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=bt_df["ds"], y=bt_df["y"], mode="lines+markers", name="Actual"))
            fig_bt.add_trace(go.Scatter(x=bt_df["ds"], y=bt_df["yhat"], mode="lines+markers", name="Predicted"))
            fig_bt.update_layout(template="plotly_white", title="Backtest: Actual vs Predicted")
            st.plotly_chart(fig_bt, use_container_width=True)

    with tab3:
        st.markdown("### Future Forecast")

        if forecast_df is None:
            st.info("Train a model from the sidebar to generate a forecast.")
        else:
            st.markdown("#### Forecast Data")
            st.dataframe(forecast_df.head())

            st.markdown("#### Forecast vs History")
            hist_df = df.copy()
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=hist_df["ds"], y=hist_df["y"], mode="lines", name="Historical"))
            fig_f.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat"], mode="lines+markers", name="Forecast", line=dict(color="purple")))

            if "yhat_lower" in forecast_df.columns and forecast_df["yhat_lower"].notna().any():
                fig_f.add_trace(
                    go.Scatter(
                        x=pd.concat([forecast_df["ds"], forecast_df["ds"][::-1]]),
                        y=pd.concat([forecast_df["yhat_upper"], forecast_df["yhat_lower"][::-1]]),
                        fill="toself",
                        fillcolor="rgba(155, 89, 182,0.2)",
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip",
                        showlegend=True,
                        name="Confidence Interval",
                    )
                )

            fig_f.update_layout(template="plotly_white", title="Historical & Forecasted Sales")
            st.plotly_chart(fig_f, use_container_width=True)

            st.markdown("#### Download Forecast")
            st.markdown(get_table_download_link(forecast_df, filename="forecast.csv"), unsafe_allow_html=True)

    with tab4:
        st.markdown("### Inventory Optimization")

        if forecast_df is None:
            st.info("Train a model and generate a forecast to see optimization suggestions.")
        else:
            avg_daily_demand = df["y"].mean()
            std_daily_demand = df["y"].std(ddof=0)

            col_o1, col_o2 = st.columns(2)
            with col_o1:
                lead_time_days = st.number_input("Lead time (days)", min_value=1.0, value=7.0, step=1.0)
                order_cost = st.number_input("Order cost per order (K)", min_value=1.0, value=100.0, step=10.0)
                holding_cost = st.number_input("Annual holding cost per unit (H)", min_value=0.1, value=5.0, step=0.5)
            with col_o2:
                service_level_z = st.number_input("Service level z-score", min_value=0.0, value=1.65, step=0.05)
                days_per_year = st.number_input("Days per year", min_value=1, value=365, step=1)

            annual_demand = avg_daily_demand * days_per_year
            if annual_demand <= 0 or holding_cost <= 0:
                st.warning("Not enough demand data or invalid holding cost to compute EOQ.")
            else:
                eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost)
                safety_stock = service_level_z * std_daily_demand * np.sqrt(lead_time_days)
                reorder_point = avg_daily_demand * lead_time_days + safety_stock

                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("EOQ (units)", f"{eoq:.2f}")
                col_r2.metric("Safety Stock", f"{safety_stock:.2f}")
                col_r3.metric("Reorder Point", f"{reorder_point:.2f}")

                st.markdown("These values are based on the historical demand distribution and your cost parameters.")


if __name__ == "__main__":
    main()
