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

st.set_page_config(layout="wide")
# ---------------------------
# Load Data from CSV
# ---------------------------
@st.cache_data
def load_growth_percentage():
    file_path = Path(__file__).parent / "new-data.csv"
    df = pd.read_csv(file_path)

    # Ensure Date column
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01")
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

servers = ["All Servers"] + sorted(df['ServerName'].unique())
selected_server = st.sidebar.selectbox("Select Server", servers)

if selected_server != "All Servers":
    db_list = df[df["ServerName"] == selected_server]['DatabaseName'].unique()
else:
    db_list = df['DatabaseName'].unique()

databases = ["All Databases"] + sorted(db_list)
selected_db = st.sidebar.selectbox("Select Database", databases)

forecast_months = st.sidebar.number_input("Months to Forecast", min_value=1, value=6, step=1)

selected_model = st.sidebar.selectbox("Select Model", ["ARIMA", "SARIMA"])
chart_type = st.sidebar.selectbox("Select Chart Type", ["Line Chart", "Bar Chart"])

if selected_server != "All Servers":
    df = df[df["ServerName"] == selected_server]

if selected_db != "All Databases":
    df = df[df["DatabaseName"] == selected_db]

st.markdown(
    """
    <h2 style='text-align: center; color: #1E90FF; font-size: 40px;'>
        Database Growth Forecast
    </h2>
    """,
    unsafe_allow_html=True
)


metrics_list = []
plot_data = pd.DataFrame()


# ---------------------------
# Forecasting Logic
# ---------------------------
def forecast_group(group, server_name, db_name):
    ts = group.set_index("Date")["Growth%"].asfreq("MS").fillna(method="ffill")
    if len(ts) < 8:
        st.warning(f"Not enough data for {server_name} | {db_name} (need >= 8 months).")
        return pd.DataFrame()

    try:
        if selected_model == "ARIMA":
            # load best ARIMA model from pickle
            final_fit, meta = load_best_arima_model("best_arima_meta2.pkl", series=ts)

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
            "Growth%": future.values,
            "ServerName": server_name,
            "DatabaseName": db_name,
            "Type": "Forecast"
        })
        forecast_df["Year"] = forecast_df["Date"].dt.year
        forecast_df["Month"] = forecast_df["Date"].dt.month

        # calculate used from growth %
        last_used = group["Size_Used"].iloc[-1]
        used_forecast, current_used = [], last_used
        for perc in forecast_df["Growth%"]:
            growth_value = (perc / 100.0) * current_used
            current_used += growth_value
            used_forecast.append(current_used)
        forecast_df["Size_Used"] = used_forecast

        combined = pd.concat([hist_df, forecast_df])
        return combined

    except Exception as e:
        st.warning(f"Could not build {selected_model} model for {server_name} | {db_name}: {e}")
        return pd.DataFrame()


if selected_db == "All Databases":
    group = df.groupby("Date", as_index=False)[["Growth%", "Size_Used"]].sum()
    group["ServerName"] = selected_server if selected_server != "All Servers" else "All Servers"
    group["DatabaseName"] = "All Databases"
    plot_data = forecast_group(group, selected_server if selected_server != "All Servers" else "All Servers", "All Databases")
else:
    for (server, db), group in df.groupby(["ServerName", "DatabaseName"]):
        group = group.sort_values("Date")
        plot_data = pd.concat([plot_data, forecast_group(group, server, db)])


# ---------------------------
# Extra Plot: Cumulative Size Used (Dynamic - Line OR Bar)
# ---------------------------
st.subheader("DB Size Usage")

if not plot_data.empty:
    # Calculate cumulative size
    plot_data["Cumulative_Size_Used"] = plot_data.groupby(
        ["ServerName", "DatabaseName"]
    )["Size_Used"].cumsum()

    # Convert to GB
    plot_data["Cumulative_Size_Used_GB"] = plot_data["Cumulative_Size_Used"] / 1024

    # Separate Historical and Forecast data
    hist_data = plot_data[plot_data["Type"] == "Historical"]
    forecast_data = plot_data[plot_data["Type"] == "Forecast"]

    if chart_type == "Bar Chart":
        # BAR CHART
        fig_cum = px.bar(
            plot_data,
            x="Date",
            y="Cumulative_Size_Used_GB",
            color="Type",
            barmode="group",
            color_discrete_map={"Historical": "#1f77b4", "Forecast": "#ff7f0e"},
            hover_data={
                "ServerName": True,
                "DatabaseName": True,
                "Cumulative_Size_Used_GB": ':.2f',
                "Date": True
            },
            title=" Size Used (Bar Chart)"
        )

    else:
        # LINE CHART
        fig_cum = px.line(
            hist_data,
            x="Date",
            y="Cumulative_Size_Used_GB",
            color="Type",
            color_discrete_map={"Historical": "#1f77b4"},
            markers=True,
            hover_data={
                "ServerName": True,
                "DatabaseName": True,
                "Cumulative_Size_Used_GB": ':.2f',
                "Date": True
            },
            title="DB Size Used (Line Chart)"
        )

        # Add Forecast traces in orange
        forecast_fig = px.line(
            forecast_data,
            x="Date",
            y="Cumulative_Size_Used_GB",
            color="Type",
            color_discrete_map={"Forecast": "#ff7f0e"},
            markers=True
        )
        for trace in forecast_fig.data:
            fig_cum.add_trace(trace)

        # Connect last historical & first forecast point
        for label in forecast_data["DatabaseName"].unique():
            last_hist = hist_data[hist_data["DatabaseName"] == label].iloc[-1]
            first_fore = forecast_data[forecast_data["DatabaseName"] == label].iloc[0]
            fig_cum.add_scatter(
                x=[last_hist["Date"], first_fore["Date"]],
                y=[last_hist["Cumulative_Size_Used_GB"], first_fore["Cumulative_Size_Used_GB"]],
                mode="lines",
                line=dict(color="#ff7f0e", width=2),
                showlegend=False
            )
    fig_cum.add_hline(y=0, line_color="white") 
    fig_cum.add_vline(x=plot_data["Date"].min(), line_color="white")
    # Common Layout Styling
    fig_cum.update_layout(
    title=dict(
        text="Database Size Used In GB",
        x=0.4,
        font=dict(color="black", size=18)
    ),
    xaxis=dict(
        showline=True,               # Enable axis line
        linecolor="white",           # Set line color to white
        gridcolor="rgba(200, 200, 200, 0.3)",
        zerolinecolor="rgba(0, 0, 0, 0.2)",
        title=dict(text="Date", font=dict(color="black", size=18)),
        tickfont=dict(color="black", size=14)
    ),
    yaxis=dict(
        showline=True,               # Enable axis line
        linecolor="white",           # Set line color to white
        gridcolor="rgba(200, 200, 200, 0.3)",
        zerolinecolor="rgba(0, 0, 0, 0.2)",
        title=dict(text="Size Used(GB)", font=dict(color="black", size=18)),
        tickfont=dict(color="black", size=14)
    ),
    plot_bgcolor="rgba(1,3,10, 0.9)",
    paper_bgcolor="rgb(240, 242, 246)",
    font=dict(color="black", size=12),
    legend=dict(
        bgcolor="rgba(255, 255, 255, 0.6)",
        bordercolor="rgba(0, 0, 0, 0.1)",
        borderwidth=1
    )
)


    st.plotly_chart(fig_cum, use_container_width=True)


# ---------------------------
# visualization 
# ---------------------------
st.subheader("DB Growth Percent")
if not plot_data.empty:
    plot_data["used_display"] = plot_data["Size_Used"].apply(
        lambda x: f"{x:,.2f} MB" if x < 1024 else f"{x/1024:,.2f} GB"
    )
    plot_data["Label"] = plot_data["ServerName"] + " | " + plot_data["DatabaseName"]

    # Separate historical and forecast data
    hist_data = plot_data[plot_data["Type"] == "Historical"]
    forecast_data = plot_data[plot_data["Type"] == "Forecast"]

    if chart_type == "Bar Chart":
        # BAR CHART for historical
        fig = px.bar(
            hist_data,
            x="Date", y="Growth%",
            color="Type",
            color_discrete_map={"Historical": "#1f77b4"},
            hover_data={
                "ServerName": True,
                "DatabaseName": True,
                "Growth%": ':.2f',
                "used_display": True,
                "Type": True
            },
            title="Server-Database Growth Forecast"
        )
        # Add forecast bars
        forecast_fig = px.bar(
            forecast_data,
            x="Date", y="Growth%",
            color="Type",
            color_discrete_map={"Forecast": "#ff7f0e"},
            hover_data={
                "ServerName": True,
                "DatabaseName": True,
                "Growth%": ':.2f',
                "used_display": True,
                "Type": True
            }
        )
        for trace in forecast_fig.data:
            fig.add_trace(trace)

    else:
        # LINE CHART for historical
        fig = px.line(
            hist_data,
            x="Date", y="Growth%",
            line_group="Label",
            color="Type",
            color_discrete_map={"Forecast": "#1f77b4"},
            color_discrete_sequence=["#1f77b4"],
            markers=True,
            hover_data={
                "ServerName": True,
                "DatabaseName": True,
                "Growth%": ':.2f',
                "used_display": True,
                "Type": True
            },
            title="Server-Database Growth Forecast"
        )

        # Add forecast lines
        forecast_fig = px.line(
            forecast_data,
            x="Date", y="Growth%",
            line_group="Label",
            color="Type",
            color_discrete_map={"Forecast": "#ff7f0e"},
            color_discrete_sequence=["#ff7f0e"],
            markers=True,
            hover_data={
                "ServerName": True,
                "DatabaseName": True,
                "Growth%": ':.2f',
                "used_display": True,
                "Type": True
            }
        )
        for trace in forecast_fig.data:
            fig.add_trace(trace)

        # Connect last historical and first forecast point for each label
        for label in forecast_data["Label"].unique():
            last_hist = hist_data[hist_data["Label"] == label].iloc[-1]
            first_fore = forecast_data[forecast_data["Label"] == label].iloc[0]
            fig.add_scatter(
                x=[last_hist["Date"], first_fore["Date"]],
                y=[last_hist["Growth%"], first_fore["Growth%"]],
                mode="lines",
                line=dict(color="#ff7f0e", width=2),
                showlegend=False
            )

    # Add capacity line
    server_capacity = 3
    fig.add_trace(
        px.line(
            x=[plot_data["Date"].min(), plot_data["Date"].max()],
            y=[server_capacity, server_capacity]
        ).data[0]
    )
    fig.data[-1].update(
        name="DB Limit",
        mode="lines",
        line=dict(color="red", dash="dot", width=2),
        showlegend=True
    )
    fig.add_hline(y=0, line_color="white") 
    fig.add_vline(x=plot_data["Date"].min(), line_color="white")

    # Layout styling
    fig.update_layout(
        title=dict(
            text="Server-Database Growth Forecast",
            x=0.4,
            font=dict(color="black", size=18)
        ),
        xaxis=dict(
            showline=True,
            linecolor="white",
            gridcolor="rgba(200, 200, 200, 0.3)",
            zerolinecolor="rgba(0, 0, 0, 0.2)",
            title=dict(text="Date", font=dict(color="black", size=18)),
            tickfont=dict(color="black", size=14)
        ),
        yaxis=dict(
            showline=True,
            linecolor="white",
            gridcolor="rgba(200, 200, 200, 0.3)",
            zerolinecolor="rgba(0, 0, 0, 0.2)",
            title=dict(text="Growth %", font=dict(color="black", size=18)),
            tickfont=dict(color="black", size=14)
        ),
        plot_bgcolor="rgba(1,3,10, 0.9)",
        paper_bgcolor="rgb(240, 242, 246)",
        font=dict(color="black", size=12),
        legend=dict(
            bgcolor="rgba(255, 255, 255, 0.6)",
            bordercolor="rgba(0, 0, 0, 0.1)",
            borderwidth=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# Metrics Table
# ---------------------------
st.subheader("Forecast Model Metrics")
if metrics_list:
    metrics_df = pd.DataFrame(metrics_list)[["Model", "RMSE", "MAE", "MAPE (%)"]]
    st.dataframe(metrics_df, use_container_width=True)
    st.markdown("""
    **Metric Definitions:**
    - **RMSE (Root Mean Square Error):** Square root of average squared differences between actual and predicted values.
    - **MAE (Mean Absolute Error):** Average of absolute differences between actual and predicted values.
    - **MAPE (Mean Absolute Percentage Error):** Average percentage error between actual and predicted values.
    """)
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
combined_df = combined_df.sort_values(["ServerName", "DatabaseName", "Date"])
combined_df = combined_df.drop(columns=["used_display"])

# ---- Add Previous Size_Used ----
combined_df["Prev_Size_Used"] = combined_df.groupby(
    ["ServerName", "DatabaseName"]
)["Size_Used"].shift(1)

# ---- Size Change (Current + Previous) ----
combined_df["Size_Change"] = combined_df["Size_Used"] + combined_df["Prev_Size_Used"]
combined_df = combined_df.drop(columns=["Size_Change"])
combined_df = combined_df.drop(columns=["Prev_Size_Used"])
# ---- Add Cumulative Size_Used ----
combined_df["Cumulative_Size_Used"] = combined_df.groupby(
    ["ServerName", "DatabaseName"]
)["Size_Used"].cumsum()
combined_df = combined_df.drop(columns=["Cumulative_Size_Used_GB"])
if "Year-Month" in combined_df.columns:
    combined_df = combined_df.drop(columns=["Year-Month"])

st.dataframe(combined_df, use_container_width=True)
