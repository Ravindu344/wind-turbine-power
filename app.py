import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

# Set page configuration
st.set_page_config(page_title="⚡ Wind Turbine Power Predictor", layout="centered")

st.title("⚡ Wind Turbine Power Predictor")
st.markdown("Enter wind speed and direction to predict **LV Active Power** and **Theoretical Power Curve**.")

# Load the models (two separate XGBoost models for two outputs)
with open("xgb_lv_active_power.json", "rb") as f:
    model_lv = xgb.XGBRegressor()
    model_lv.load_model("xgb_lv_active_power.json")

with open("xgb_theoretical_power.json", "rb") as f:
    model_theoretical = xgb.XGBRegressor()
    model_theoretical.load_model("xgb_theoretical_power.json")

# Optional: load scaler if used
# with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# Input sliders
wind_speed = st.number_input("🌬️ Wind Speed (m/s)", min_value=0.0, max_value=25.0, value=0.0, step=0.1)
wind_direction = st.number_input("🧭 Wind Direction (°)", min_value=0.0, max_value=360.0, value=0.0, step=1.0)

if st.button("🚀 Predict Power Output"):
    input_df = pd.DataFrame([{
        'Wind Speed (m/s)': wind_speed,
        'Wind Direction (°)': wind_direction
    }])

    # If scaled, apply transformation
    # input_df_scaled = scaler.transform(input_df)

    # Predict
    active_power = model_lv.predict(input_df)[0]
    theoretical_power = model_theoretical.predict(input_df)[0]

    # Clip to avoid negative predictions
    active_power = max(0, round(active_power, 2))
    theoretical_power = max(0, round(theoretical_power, 2))

    # Display results
    st.success(f"🔋 LV Active Power: {active_power:.2f} kW")
    st.info(f"💡 Theoretical Power Curve: {theoretical_power:.2f} kWh")

st.markdown("---")
st.caption("Built with ❤️ using XGBoost and Streamlit")
