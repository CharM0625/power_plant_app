import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

    Plant_Age = st.slider('Plant Age', 0, 100, 0)
    
    Start_Year = st.number_input('Start Year', value=2010)
    
    PriEnergySource = st.selectbox('Energy Source', ['Nuclear', 'Sun', 'Sub-Bitumous Coal', 'Bitumous', 'Natural Gas', 'Wind', 'Water'])


# MAP PANEL
with col2:

    st.subheader("Plant Location")

    map_data = pd.DataFrame({
        "lat": [y],
        "lon": [x]
    })

    st.map(map_data)


# Build input dataframe
input_dict = {
    "x": x,
    "y": y,
    "Plant_Age": Plant_Age,
    "StartYear": Start_Year,
    "PriEnergySource": PriEnergySource
}

input_df = pd.DataFrame([input_dict])

# Match training features
for col in features:
    if col not in input_df:
        input_df[col] = 0

input_df = input_df[features]

# Prediction button
if st.button("Predict Capacity"):

    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Results")

    st.success(f"Predicted Capacity: {prediction:.2f} MW")

    if prediction < 50:
        st.warning("Small Power Plant")

    elif prediction < 300:
        st.info("Medium Scale Power Plant")

    else:
        st.success("Large Utility Scale Power Plant")


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

        if hasattr(model, "feature_importances_"):

            importance_df = pd.DataFrame({
                "Feature": features,
                "Importance": model.feature_importances_
            })

            importance_df = importance_df.sort_values(
                by="Importance",
                ascending=False
            ).head(10)

            fig2, ax2 = plt.subplots()

            colors = plt.cm.viridis(np.linspace(0,1,len(importance_df)))

            ax2.barh(
                importance_df["Feature"],
                importance_df["Importance"],
                color=colors
            )

            ax2.set_title("Top Features Influencing Capacity")

            st.pyplot(fig2)

        else:
            st.info("Feature importance not available for this model.")
