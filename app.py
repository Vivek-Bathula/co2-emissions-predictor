
import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("co2_emissions_model.pkl")

# Get valid categories from the trained encoder
encoder = model.named_steps["preprocessor"].named_transformers_["cat"]

make_options = encoder.categories_[0]
vehicle_class_options = encoder.categories_[1]
fuel_type_options = encoder.categories_[2]

# Page title
st.title("🚗 CO₂ Emissions Predictor")

st.write(
    "Enter the vehicle details below to predict its CO₂ emissions."
)

# Input: Engine Size
engine_size = st.number_input(
    "Engine Size (L)",
    min_value=0.1,
    max_value=10.0,
    value=2.0,
    step=0.1
)

# Input: Fuel Consumption
mpg = st.number_input(
    "Fuel Consumption Combined (mpg)",
    min_value=1.0,
    max_value=100.0,
    value=30.0,
    step=1.0
)

# Input: Make
make = st.selectbox(
    "Make",
    make_options
)

# Input: Vehicle Class
vehicle_class = st.selectbox(
    "Vehicle Class",
    vehicle_class_options
)

# Input: Fuel Type
fuel_type = st.selectbox(
    "Fuel Type",
    fuel_type_options
)

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

    st.success(
        f"Predicted CO₂ Emissions: {prediction:.2f} g/km"
    )

# Footer
st.markdown("---")
st.caption("Built by Vivek Bathula | First Machine Learning Project")
