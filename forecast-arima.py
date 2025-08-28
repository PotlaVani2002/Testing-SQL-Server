import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

# =========================
# Data & Model Loading
# =========================
@st.cache_data
def load_growth_percentage():
    file_path = Path(__file__).parent / "new-data5.csv"
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01")
    return df

def load_best_arima_model(meta_file="best_arima_meta3.pkl", series=None):
    meta = joblib.load(Path(__file__).parent / meta_file)
    order = (12, 1, 6)  # (7,1,6)
    final_fit = ARIMA(series, order=order).fit()
    return final_fit, meta

# =========================
# Helpers
# =========================
def agg_weighted(x):
    return pd.Series({
        "Growth%": np.average(x["Growth%"], weights=x["Size_Used"]),
        "Size_Used": x["Size_Used"].sum()
    })

def server_capacity_value(srv):
    if srv == "Server1":
        return 16
    if srv == "Server2":
        return 11
    return 13.5

# =========================
# UI
# =========================
df = load_growth_percentage()
default_server = "Server1" if "Server1" in df['ServerName'].unique() else sorted(df['ServerName'].unique())[0]
servers = ["All Servers"] + sorted(df['ServerName'].unique())

selected_server = st.sidebar.selectbox("Select Server", servers, index=servers.index(default_server))

db_list = (df[df["ServerName"] == selected_server]['DatabaseName'].unique()
           if selected_server != "All Servers" else df['DatabaseName'].unique())
databases = ["All Databases"] + sorted(db_list)
default_db = "All Databases" if selected_server == "Server1" else ("All Databases"if len(db_list) > 0 else sorted(db_list)[0]  )

selected_db = st.sidebar.selectbox("Select Database", databases, index=databases.index(default_db))
forecast_months = st.sidebar.number_input("Months to Forecast", min_value=1, value=6, step=1)
selected_model = st.sidebar.selectbox("Select Model", ["ARIMA", "SARIMA"])
chart_type = st.sidebar.selectbox("Select Chart Type", ["Line Chart", "Bar Chart"])

if selected_server != "All Servers":
    df = df[df["ServerName"] == selected_server]
if selected_db != "All Databases":
    df = df[df["DatabaseName"] == selected_db]

st.markdown(
    "<h2 style='text-align: center; color: #1E90FF; font-size: 40px;'>Database Growth Forecast</h2>",
    unsafe_allow_html=True
)

metrics_list, plot_data = [], pd.DataFrame()

# =========================
# Forecasting
# =========================
def forecast_group(group, server_name, db_name):
    ts = group.set_index("Date")["Growth%"].asfreq("MS").fillna(method="ffill")
    if len(ts) < 8:
        st.warning(f"Not enough data for {server_name} | {db_name} (need >= 8 months).")
        return pd.DataFrame()
    try:
        if selected_model == "ARIMA":
            final_fit, meta = load_best_arima_model("best_arima_meta2.pkl", series=ts)
            future = final_fit.forecast(steps=forecast_months)
            mape = np.mean(np.abs((ts - final_fit.fittedvalues) / np.where(np.abs(ts) < 1e-8, 1e-8, np.abs(ts)))) * 100
            metrics_list.append({"RMSE": round(np.sqrt(meta["mse"]), 3),
                                 "MAE": round(meta["mae"], 3),
                                 "MAPE (%)": round(mape, 3),
                                 "Model": "ARIMA"})
        else:
            n = max(int(len(ts) * 0.7), 6)
            train, test = ts.iloc[:n], ts.iloc[n:]
            model = SARIMAX(train, order=(1, 0, 3), seasonal_order=(1, 1, 1, 12),
                            enforce_stationarity=False, enforce_invertibility=False)
            final_fit = model.fit()
            future = final_fit.forecast(steps=forecast_months)
            y_true = np.asarray(test, dtype=float)
            y_pred = np.asarray(final_fit.forecast(steps=len(test)), dtype=float)
            mse = mean_squared_error(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true)))) * 100
            metrics_list.append({"RMSE": round(np.sqrt(mse), 3),
                                 "MAE": round(mean_absolute_error(y_true, y_pred), 3),
                                 "MAPE (%)": round(mape, 3),
                                 "Model": "SARIMA"})

        hist_df = group.copy()
        hist_df["Type"] = "Historical"

        future_idx = pd.date_range(ts.index[-1] + pd.offsets.MonthBegin(1), periods=forecast_months, freq="MS")
        forecast_df = pd.DataFrame({
            "Date": future_idx,
            "Growth%": future.values,
            "ServerName": server_name,
            "DatabaseName": db_name,
            "Type": "Forecast"
        })
        forecast_df["Year"] = forecast_df["Date"].dt.year
        forecast_df["Month"] = forecast_df["Date"].dt.month

        # Size_Used from growth% (keep original logic: percentage * 70000 each month)
        forecast_df["Size_Used"] = (forecast_df["Growth%"] / 100.0) * 70000

        return pd.concat([hist_df, forecast_df], ignore_index=True)
    except Exception as e:
        st.warning(f"Could not build {selected_model} model for {server_name} | {db_name}: {e}")
        return pd.DataFrame()

if selected_server == "All Servers" and selected_db == "All Databases":
    grouped = df.groupby("Date", as_index=False).apply(agg_weighted).reset_index(drop=True)
    grouped["ServerName"], grouped["DatabaseName"] = "All Servers", "All Databases"
    plot_data = forecast_group(grouped, "All Servers", "All Databases")
elif selected_db == "All Databases":
    grouped = df.groupby("Date", as_index=False).apply(agg_weighted).reset_index(drop=True)
    grouped["ServerName"], grouped["DatabaseName"] = selected_server, "All Databases"
    plot_data = forecast_group(grouped, selected_server, "All Databases")
else:
    for (server, db), grp in df.sort_values("Date").groupby(["ServerName", "DatabaseName"]):
        if (selected_server in ["All Servers", server]) and (selected_db in ["All Databases", db]):
            plot_data = pd.concat([plot_data, forecast_group(grp, server, db)], ignore_index=True)

# =========================
# Visualization 1: Growth%
# =========================
st.subheader("DB Growth Percent")
if not plot_data.empty:
    plot_data["used_display"] = plot_data["Size_Used"]
    plot_data["Label"] = plot_data["ServerName"] + " | " + plot_data["DatabaseName"]
    hist_data = plot_data[plot_data["Type"] == "Historical"]
    forecast_data = plot_data[plot_data["Type"] == "Forecast"]
    hover = {"ServerName": True, "DatabaseName": True, "Growth%": ':.2f', "used_display": True, "Type": True}

    if chart_type == "Bar Chart":
        fig = px.bar(hist_data, x="Date", y="Growth%", color="Type",
                     color_discrete_map={"Historical": "#1f77b4"}, hover_data=hover,
                     title="Server-Database Growth Forecast")
        forecast_fig = px.bar(forecast_data, x="Date", y="Growth%", color="Type",
                              color_discrete_map={"Forecast": "#ff7f0e"}, hover_data=hover)
    else:
        fig = px.line(hist_data, x="Date", y="Growth%", line_group="Label", color="Type",
                      color_discrete_map={"Forecast": "#1f77b4"}, color_discrete_sequence=["#1f77b4"],
                      markers=True, hover_data=hover, title="Server-Database Growth Forecast")
        forecast_fig = px.line(forecast_data, x="Date", y="Growth%", line_group="Label", color="Type",
                               color_discrete_map={"Forecast": "#ff7f0e"}, color_discrete_sequence=["#ff7f0e"],
                               markers=True, hover_data=hover)
    for tr in forecast_fig.data:
        fig.add_trace(tr)

    if chart_type == "Line Chart":
        for lbl in forecast_data["Label"].unique():
            last_hist = hist_data[hist_data["Label"] == lbl].iloc[-1]
            first_fore = forecast_data[forecast_data["Label"] == lbl].iloc[0]
            fig.add_scatter(x=[last_hist["Date"], first_fore["Date"]],
                            y=[last_hist["Growth%"], first_fore["Growth%"]],
                            mode="lines", line=dict(color="#ff7f0e", width=2), showlegend=False)

    cap = server_capacity_value(selected_server)
    fig.add_trace(px.line(x=[plot_data["Date"].min(), plot_data["Date"].max()],
                          y=[cap, cap]).data[0])
    fig.data[-1].update(name="DB Limit", mode="lines",
                        line=dict(color="red", dash="dot", width=2), showlegend=True)
    fig.add_hline(y=0, line_color="white")
    fig.add_vline(x=plot_data["Date"].min(), line_color="white")
    fig.update_layout(
        title=dict(text="Server-Database Growth Forecast", x=0.4, font=dict(color="black", size=18)),
        xaxis=dict(showline=True, linecolor="white", gridcolor="rgba(200, 200, 200, 0.3)",
                   zerolinecolor="rgba(0, 0, 0, 0.2)", title=dict(text="Date", font=dict(color="black", size=18)),
                   tickfont=dict(color="black", size=14)),
        yaxis=dict(showline=True, linecolor="white", gridcolor="rgba(200, 200, 200, 0.3)",
                   zerolinecolor="rgba(0, 0, 0, 0.2)", title=dict(text="Growth %", font=dict(color="black", size=18)),
                   tickfont=dict(color="black", size=14)),
        plot_bgcolor="rgba(1,3,10, 0.9)", paper_bgcolor="rgb(240, 242, 246)",
        font=dict(color="black", size=12),
        legend=dict(bgcolor="rgba(255, 255, 255, 0.6)", bordercolor="rgba(0, 0, 0, 0.1)", borderwidth=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# Visualization 2: Size Used
# =========================
st.subheader("DB Size Usage")
if not plot_data.empty:
    plot_data["Cumulative_Size_Used"] = plot_data.groupby(["ServerName", "DatabaseName"])["Size_Used"].cumsum()
    plot_data["Cumulative_Size_Used_GB"] = plot_data["Cumulative_Size_Used"] / 1024
    hist_data = plot_data[plot_data["Type"] == "Historical"]
    forecast_data = plot_data[plot_data["Type"] == "Forecast"]

    if chart_type == "Bar Chart":
        fig_cum = px.bar(plot_data, x="Date", y="Cumulative_Size_Used_GB", color="Type", barmode="group",
                         color_discrete_map={"Historical": "#1f77b4", "Forecast": "#ff7f0e"},
                         hover_data={"ServerName": True, "DatabaseName": True, "Cumulative_Size_Used_GB": ':.2f', "Date": True},
                         title=" Size Used (Bar Chart)")
    else:
        fig_cum = px.line(hist_data, x="Date", y="Cumulative_Size_Used_GB", color="Type",
                          color_discrete_map={"Historical": "#1f77b4"}, markers=True,
                          hover_data={"ServerName": True, "DatabaseName": True, "Cumulative_Size_Used_GB": ':.2f', "Date": True},
                          title="DB Size Used (Line Chart)")
        forecast_fig = px.line(forecast_data, x="Date", y="Cumulative_Size_Used_GB", color="Type",
                               color_discrete_map={"Forecast": "#ff7f0e"}, markers=True)
        for tr in forecast_fig.data:
            fig_cum.add_trace(tr)
        for db in forecast_data["DatabaseName"].unique():
            last_hist = hist_data[hist_data["DatabaseName"] == db].iloc[-1]
            first_fore = forecast_data[forecast_data["DatabaseName"] == db].iloc[0]
            fig_cum.add_scatter(x=[last_hist["Date"], first_fore["Date"]],
                                y=[last_hist["Cumulative_Size_Used_GB"], first_fore["Cumulative_Size_Used_GB"]],
                                mode="lines", line=dict(color="#ff7f0e", width=2), showlegend=False)

    fig_cum.add_hline(y=0, line_color="white")
    fig_cum.add_vline(x=plot_data["Date"].min(), line_color="white")
    fig_cum.update_layout(
        title=dict(text="Database Size Used In GB", x=0.4, font=dict(color="black", size=18)),
        xaxis=dict(showline=True, linecolor="white", gridcolor="rgba(200, 200, 200, 0.3)",
                   zerolinecolor="rgba(0, 0, 0, 0.2)", title=dict(text="Date", font=dict(color="black", size=18)),
                   tickfont=dict(color="black", size=14)),
        yaxis=dict(showline=True, linecolor="white", gridcolor="rgba(200, 200, 200, 0.3)",
                   zerolinecolor="rgba(0, 0, 0, 0.2)", title=dict(text="Size Used(GB)", font=dict(color="black", size=18)),
                   tickfont=dict(color="black", size=14)),
        plot_bgcolor="rgba(1,3,10, 0.9)", paper_bgcolor="rgb(240, 242, 246)",
        font=dict(color="black", size=12),
        legend=dict(bgcolor="rgba(255, 255, 255, 0.6)", bordercolor="rgba(0, 0, 0, 0.1)", borderwidth=1)
    )
    st.plotly_chart(fig_cum, use_container_width=True)

# =========================
# Metrics Table
# =========================
st.subheader("Forecast Model Metrics")
if metrics_list:
    st.dataframe(pd.DataFrame(metrics_list)[["Model", "RMSE", "MAE", "MAPE (%)"]], use_container_width=True)
    st.markdown("""
**Metric Definitions:**
- **RMSE (Root Mean Square Error):** Square root of average squared differences between actual and predicted values.
- **MAE (Mean Absolute Error):** Average of absolute differences between actual and predicted values.
- **MAPE (Mean Absolute Percentage Error):** Average percentage error between actual and predicted values.
""")
else:
    st.info("No metrics to display (insufficient data).")

# =========================
# Raw + Predicted Data
# =========================
st.subheader("Raw + Predicted Data")
raw_df = df.copy()
raw_df["Type"] = "Historical"
pred_df = plot_data[plot_data["Type"] == "Forecast"].copy() if not plot_data.empty else pd.DataFrame(columns=df.columns.tolist() + ["Type"])
combined_df = pd.concat([raw_df, pred_df], ignore_index=True).sort_values(["ServerName", "DatabaseName", "Date"])
combined_df.drop(columns=["used_display"], errors="ignore", inplace=True)

combined_df["Cumulative_Size_Used"] = combined_df.groupby(["ServerName", "DatabaseName"])["Size_Used"].cumsum()
combined_df.drop(columns=["Cumulative_Size_Used_GB"], errors="ignore", inplace=True)
if "Year-Month" in combined_df.columns:
    combined_df.drop(columns=["Year-Month"], inplace=True)

st.dataframe(combined_df, use_container_width=True)
