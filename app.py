
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("co2_emissions_model.pkl")

st.title("🚗 CO₂ Emissions Predictor")
st.write("Predict vehicle CO₂ emissions using vehicle and fuel characteristics.")

# Inputs
engine_size = st.number_input(
    "Engine Size (L)",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.1
)

mpg = st.number_input(
    "Fuel Consumption Combined (mpg)",
    min_value=1.0,
    max_value=100.0,
    value=30.0,
    step=1.0
)

make = st.text_input("Make", "ACURA")

vehicle_class = st.text_input("Vehicle Class", "COMPACT")

fuel_type = st.text_input("Fuel Type", "D")

# Prediction
if st.button("Predict CO₂ Emissions"):

    input_data = pd.DataFrame({
        "Engine Size(L)": [engine_size],
        "Fuel Consumption Comb (mpg)": [mpg],
        "Make": [make],
        "Vehicle Class": [vehicle_class],
        "Fuel Type": [fuel_type]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted CO₂ Emissions: {prediction:.2f} g/km")
