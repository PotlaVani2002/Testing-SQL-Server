import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------
# Load Data from CSV
# ---------------------------
@st.cache_data
def load_growth_percentage():
    file_path = Path(__file__).parent / "data.csv"
    df = pd.read_csv(file_path)

    # Ensure Date column
    df["Date"] = pd.to_datetime(df["yr"].astype(str) + "-" + df["mn"].astype(str) + "-01")
    return df


# ---------------------------
# Load Pre-trained ARIMA Model
# ---------------------------
def load_best_arima_model(meta_file="best_arima_meta.pkl", series=None):
    file_path = Path(__file__).parent / meta_file   # ✅ pathlib usage
    meta = joblib.load(file_path)  
    order = meta["order"]

    # retrain ARIMA on full series with best params
    final_fit = ARIMA(series, order=order).fit()
    return final_fit, meta


# ---------------------------
# Streamlit UI
# ---------------------------
df = load_growth_percentage()

servers = ["All Servers"] + sorted(df['servername'].unique())
selected_server = st.sidebar.selectbox("Select Server", servers)

if selected_server != "All Servers":
    db_list = df[df["servername"] == selected_server]['databasename'].unique()
else:
    db_list = df['databasename'].unique()

databases = ["All Databases"] + sorted(db_list)
selected_db = st.sidebar.selectbox("Select Database", databases)

forecast_months = st.sidebar.number_input("Months to Forecast", min_value=1, value=6, step=1)

selected_model = st.sidebar.selectbox("Select Model", ["ARIMA", "SARIMA"])
chart_type = st.sidebar.selectbox("Select Chart Type", ["Line Chart", "Bar Chart"])

if selected_server != "All Servers":
    df = df[df["servername"] == selected_server]

if selected_db != "All Databases":
    df = df[df["databasename"] == selected_db]

st.subheader("Database Growth Forecast")

metrics_list = []
plot_data = pd.DataFrame()


# ---------------------------
# Forecasting Logic
# ---------------------------
def forecast_group(group, server_name, db_name):
    ts = group.set_index("Date")["per"].asfreq("MS").fillna(method="ffill")
    if len(ts) < 8:
        st.warning(f"Not enough data for {server_name} | {db_name} (need >= 8 months).")
        return pd.DataFrame()

    try:
        if selected_model == "ARIMA":
            # load best ARIMA model from pickle
            final_fit, meta = load_best_arima_model("best_arima_meta.pkl", series=ts)

            # forecast
            future = final_fit.forecast(steps=forecast_months)

            # metrics (we don’t have test split here, so just show saved meta)
            metrics_list.append({
                "RMSE": round(np.sqrt(meta["mse"]), 3),
                "MAE": round(meta["mae"], 3),
                "MAPE (%)": round(np.mean(np.abs((ts - final_fit.fittedvalues) /
                                                 np.where(np.abs(ts) < 1e-8, 1e-8, np.abs(ts)))) * 100, 3),
                "Model": "ARIMA"
            })
        else:
            # Train SARIMA fresh (since only ARIMA is pre-trained)
            train_size = max(int(len(ts) * 0.8), 6)
            train, test = ts.iloc[:train_size], ts.iloc[train_size:]
            model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12),
                            enforce_stationarity=False, enforce_invertibility=False)
            final_fit = model.fit()
            future = final_fit.forecast(steps=forecast_months)

            y_true, y_pred = np.asarray(test, dtype=float), np.asarray(final_fit.forecast(steps=len(test)), dtype=float)
            mse = mean_squared_error(y_true, y_pred)
            metrics_list.append({
                "RMSE": round(np.sqrt(mse), 3),
                "MAE": round(mean_absolute_error(y_true, y_pred), 3),
                "MAPE (%)": round(np.mean(np.abs((y_true - y_pred) /
                                                 np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true)))) * 100, 3),
                "Model": "SARIMA"
            })

        hist_df = group.copy()
        hist_df["Type"] = "Historical"

        # forecast dataframe
        future_idx = pd.date_range(ts.index[-1] + pd.offsets.MonthBegin(1),
                                   periods=forecast_months, freq="MS")
        forecast_df = pd.DataFrame({
            "Date": future_idx,
            "per": future.values,
            "servername": server_name,
            "databasename": db_name,
            "Type": "Forecast"
        })
        forecast_df["yr"] = forecast_df["Date"].dt.year
        forecast_df["mn"] = forecast_df["Date"].dt.month

        # calculate used from growth %
        last_used = group["used"].iloc[-1]
        used_forecast, current_used = [], last_used
        for perc in forecast_df["per"]:
            growth_value = (perc / 100.0) * current_used
            current_used += growth_value
            used_forecast.append(current_used)
        forecast_df["used"] = used_forecast

        combined = pd.concat([hist_df, forecast_df])
        return combined

    except Exception as e:
        st.warning(f"Could not build {selected_model} model for {server_name} | {db_name}: {e}")
        return pd.DataFrame()


if selected_db == "All Databases":
    group = df.groupby("Date", as_index=False)[["per", "used"]].sum()
    group["servername"] = selected_server if selected_server != "All Servers" else "All Servers"
    group["databasename"] = "All Databases"
    plot_data = forecast_group(group, selected_server if selected_server != "All Servers" else "All Servers", "All Databases")
else:
    for (server, db), group in df.groupby(["servername", "databasename"]):
        group = group.sort_values("Date")
        plot_data = pd.concat([plot_data, forecast_group(group, server, db)])

# ---------------------------
# Rest of visualization & metrics remain unchanged
# ---------------------------
if not plot_data.empty:
    plot_data["used_display"] = plot_data["used"].apply(lambda x: f"{x:,.2f} MB" if x < 1024 else f"{x/1024:,.2f} GB")
    plot_data["Label"] = plot_data["servername"] + " | " + plot_data["databasename"]

    if selected_server == "All Servers" and selected_db == "All Databases":
        server_capacity = 4
    else:
        server_capacity = 2

    if chart_type == "Line Chart":
        fig = px.line(
            plot_data,
            x="Date", y="per", color="Type", line_group="Label",
            markers=True,
            hover_data={"servername": True, "databasename": True, "per": ':.2f', "used_display": True, "Type": True},
            title="Server-Database Growth Forecast",
            color_discrete_map={"Historical": "#1f77b4", "Forecast": "#ff7f0e"}
        )
    else:
        fig = px.bar(
            plot_data,
            x="Date", y="per", color="Type", barmode="group",
            hover_data={"servername": True, "databasename": True, "per": ':.2f', "used_display": True, "Type": True},
            title="Server-Database Growth Forecast",
            color_discrete_map={"Historical": "#1f77b4", "Forecast": "#ff7f0e"}
        )

    fig.add_hline(y=0, line_color="black")
    fig.add_vline(x=plot_data["Date"].min(), line_color="black")
    fig.add_trace(
        px.line(
            x=[plot_data["Date"].min(), plot_data["Date"].max()],
            y=[server_capacity, server_capacity],
        ).data[0]
    )
    fig.data[-1].update(name="DB Limit", mode="lines", line=dict(color="red", dash="dot", width=2), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Metrics Table
# ---------------------------
st.subheader("Forecast Model Metrics")
if metrics_list:
    metrics_df = pd.DataFrame(metrics_list)[["Model", "RMSE", "MAE", "MAPE (%)"]]
    st.dataframe(metrics_df, use_container_width=True)
else:
    st.info("No metrics to display (insufficient data).")

# ---------------------------
# Raw + Predicted Data
# ---------------------------
st.subheader("Raw + Predicted Data")
raw_df = df.copy()
raw_df["Type"] = "Historical"
pred_df = plot_data[plot_data["Type"] == "Forecast"].copy()
combined_df = pd.concat([raw_df, pred_df], ignore_index=True)
combined_df = combined_df.sort_values(["servername", "databasename", "Date"])

if "Year-Month" in combined_df.columns:
    combined_df = combined_df.drop(columns=["Year-Month"])

st.dataframe(combined_df, use_container_width=True)
