import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Page layout
st.set_page_config(
    page_title="California Power Plant Capacity Prediction",
    layout="wide"
)

# Load model
model = joblib.load("power_plant_model.pkl")

# Load model features
features = joblib.load("model_features.pkl")

# App title
st.title("California Power Plant Capacity Prediction")

st.write(
    "This application predicts power plant generation capacity based on plant characteristics."
)

# Layout
col1, col2 = st.columns(2)

# INPUT PANEL
with col1:

    st.subheader("Enter Plant Characteristics")

    x = st.number_input("Longitude", value=-120.0)
    y = st.number_input("Latitude", value=36.0)

    Plant_Age = st.slider('Plant Age', 0, 100, 10)
    
    Start_Year = st.number_input('Start Year', value=2010)

    County = st.selectbox('County', ['Los Angeles', 'San Diego', 'Orange', 'Riverside', 'Alameda'])
    
    CEC = st.selectbox('CEC Jurisdiction', ['Yes', 'No'])
    
    PriEnergySource = st.selectbox('Energy Source', ['Natural Gas', 'Solar', 'Wind', 'Hydro', 'Coal', 'Nuclear'])


# MAP PANEL
with col2:

    st.subheader("Plant Location")

    map_data = pd.DataFrame({
        "lat": [y],
        "lon": [x]
    })

    st.map(map_data, zoom=5)


# Build input dataframe
input_dict = {
    "x": x,
    "y": y,
    "Plant_Age": Plant_Age,
    "StartYear": Start_Year,
    "PriEnergySource": PriEnergySource,
    "County": County,
    "CEC_Jurisdictional": CEC
}

input_df = pd.DataFrame([input_dict])

# One-hot encode input
input_df = pd.get_dummies(input_df)

# Align with training features
for col in features:
    if col not in input_df:
        input_df[col] = 0

# Remove extra columns not seen in training
input_df = input_df[features]

current_year = datetime.datetime.now().year

if Start_Year > current_year:
    st.warning("Start year is in the future")

calculated_age = current_year - Start_Year

if abs(calculated_age - Plant_Age) > 2:
    st.info("Plant Age and Start Year seem inconsistent")

if not (-180 <= x <= 180):
    st.error("Longitude must be between -180 and 180")

if not (-90 <= y <= 90):
    st.error("Latitude must be between -90 and 90")

# Prediction button
if st.button("Predict Capacity"):
    
    try:
        log_prediction = model.predict(input_df)[0]
        prediction = np.expm1(log_prediction)
    except Exception as e:
        st.error("Prediction failed. Please check inputs.")
        st.stop()

    st.divider()

    st.subheader("Prediction Results")

    st.success(f"Predicted Capacity: {prediction:.2f} MW")

    if prediction < 50:
        st.warning("Small Power Plant")

    elif prediction < 300:
        st.info("Medium Scale Power Plant")

    else:
        st.success("Large Utility Scale Power Plant")
        
    st.write(f"Estimated Range: {prediction*0.85:.1f} – {prediction*1.15:.1f} MW")

    st.caption('Prediction is based on historical power plant data using a trained machine learning model.')

    # Create side-by-side charts
    chart1, chart2 = st.columns(2)

    # Capacity Chart
    with chart1:

        fig, ax = plt.subplots()

        ax.bar(["Predicted Capacity"], [prediction], color="orange")

        ax.set_ylabel("Megawatts (MW)")
        ax.set_title("Estimated Power Plant Capacity")

        st.pyplot(fig)


    # Feature Importance Chart
    with chart2:

        rf_model = model
        
        # Handle pipeline case
        if hasattr(model, "named_steps"):
            rf_model = model.named_steps.get("rf", model)
        
        if hasattr(rf_model, "feature_importances_"):
            
            importance_df = pd.DataFrame({
                "Feature": features,
                "Importance": rf_model.feature_importances_
                })
            
            importance_df = importance_df.sort_values(
                by="Importance",
                ascending=False
            ).head(10)
            
            fig2, ax2 = plt.subplots()
            
            ax2.barh(
                importance_df["Feature"],
                importance_df["Importance"]
            )
            
            ax2.set_title("Top Features Influencing Capacity")
            
            st.pyplot(fig2)
        
        else:
            st.info("Feature importance not available for this model.")

       
