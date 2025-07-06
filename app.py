import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np

# Set page config
st.set_page_config(page_title="Wind Turbine Power Predictor", layout="centered")

# Load the XGBoost model
model = xgb.XGBRegressor()
model.load_model("xgb_model.json")

# App title
st.title("⚡ Wind Turbine Power Predictor")
st.markdown("Enter wind speed and direction to predict **LV Active Power** and **Theoretical Power Curve**.")

# User inputs
wind_speed = st.number_input("🌬️ Wind Speed (m/s)", min_value=0.0, max_value=25.0, value=0.0, step=0.1)
wind_direction = st.number_input("🧭 Wind Direction (°)", min_value=0.0, max_value=360.0, value=0.0, step=1.0)

# Predict button
if st.button("🚀 Predict Power Output"):
    input_df = pd.DataFrame([{
        'Wind Speed (m/s)': wind_speed,
        'Wind Direction (°)': wind_direction
    }])

    # Predict
    prediction = model.predict(input_df)
    
    # Handle single or multi-output prediction
    if isinstance(prediction[0], (np.ndarray, list)):
        active_power = prediction[0][0]
        theoretical_power = prediction[0][1]
    else:
        active_power = prediction[0]
        theoretical_power = None

    # Display predictions
    st.success(f"🔋 LV Active Power: {active_power:.2f} kW")
    if theoretical_power is not None:
        st.info(f"💡 Theoretical Power Curve: {theoretical_power:.2f} kWh")
