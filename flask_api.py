import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

st.set_page_config(page_title="SARIMA Forecast App", layout="wide")
st.title("📈 SARIMA Forecasting App")

# ----------------------------
# Load data
# ----------------------------
@st.cache_data
def load_data():
    file_path = Path(__file__).parent / "Month_Value_1.csv"   # <-- keep CSV in repo
    data = pd.read_csv(file_path)
    data['Period'] = pd.to_datetime(data['Period'], format='%d.%m.%Y')
    data.set_index('Period', inplace=True)
    return data

data = load_data()
average_cost = data['Average_cost'].dropna()

st.subheader("📊 Historical Data Preview")
st.write(data.head())

# Plot historical data
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(average_cost, marker='o', label='Average Cost')
ax.set_title("Average Cost Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Average Cost")
ax.legend()
ax.grid()
st.pyplot(fig)

# ----------------------------
# Load trained model
# ----------------------------
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / 'model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model




model = load_model()

# ----------------------------
# Forecasting
# ----------------------------
st.subheader("🔮 Forecasting")

forecast_horizon = st.slider("Select forecast horizon (steps)", min_value=5, max_value=50, value=10)

if st.button("Generate Forecast"):
    with st.spinner("Generating forecast..."):
        forecast = model.get_forecast(steps=forecast_horizon)
        pred_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()

    st.success("✅ Forecast generated successfully!")

    # Plot forecast
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(average_cost, label="Historical Data")
    ax2.plot(pred_mean.index, pred_mean, label="Forecast", color="green")
    ax2.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], 
                     color="lightgreen", alpha=0.5)
    ax2.set_title("SARIMA Forecast for Average Cost")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Average Cost")
    ax2.legend()
    ax2.grid()
    st.pyplot(fig2)

    # Show forecast values
    st.subheader("📄 Forecasted Values")
    st.write(pred_mean)

    # Download forecast as CSV
    forecast_df = pd.DataFrame({
        "Date": pred_mean.index,
        "Forecast": pred_mean.values,
        "Lower CI": conf_int.iloc[:, 0].values,
        "Upper CI": conf_int.iloc[:, 1].values
    })
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Forecast (CSV)",
        data=csv,
        file_name="sarima_forecast.csv",
        mime="text/csv"
    )
