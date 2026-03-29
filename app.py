import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

#Load the model
model = joblib.load("power_plant_model.pkl")

#Load features
features = joblib.load("model_features.pkl")

#Create the app name
st.title("California Power Plant Capacity Prediction")

#Explain the app
st.write("This application predicts power plant generation capacity based on plant characteristics.")

#Create input fields

col1, col2 = st.columns(2)

with col1:
    
    x = st.number_input("Longitude", value=-120.0)
    y = st.number_input("Latitude", value=36.0)
    
    plant_age = st.slider("Plant Age", 0, 100, 10)
    
    start_year = st.number_input("Start Year", value=2010)
    
    solar = st.selectbox("Solar Plant", [0,1])
    hydro = st.selectbox("Hydropower Plant", [0,1])


with col2:
    
    st.subheader("Plant Location")

    map_data = pd.DataFrame({
        "lat":[y],
        "lon":[x]
    })

    st.map(map_data)

#Build the input dataframe
input_dict = {
    "x": x,
    "y": y,
    "Plant_Age": plant_age,
    "StartYear": start_year,
    "PriEnergySource_SUN": solar,
    "PriEnergySource_WAT": hydro
}

input_df = pd.DataFrame([input_dict])

# Match model training features
for col in features:
    if col not in input_df:
        input_df[col] = 0

input_df = input_df[features]

#Create the predict button
if st.button("Predict Capacity"):
    
    prediction = model.predict(input_df)[0]

    st.subheader("Predicted Capacity")

    st.success(f"{prediction:.2f} MW")
    
    if prediction < 50:
        st.warning("Small Power Plant")

    elif prediction < 300:
        st.info("Medium Scale Power Plant")

    else:
        st.success("Large Utility Scale Power Plant")

#Visualization
    fig, ax = plt.subplots()

    ax.bar(["Predicted Capacity"], [prediction])

    ax.set_ylabel("Megawatts (MW)")
    ax.set_title("Estimated Power Plant Capacity")

    st.pyplot(fig)

    fig, ax = plt.subplots()

    ax.bar(["Predicted Capacity"], [prediction])

    ax.set_ylabel("Megawatts (MW)")
    ax.set_title("Estimated Power Plant Capacity")

    st.pyplot(fig)

    fig2, ax2 = plt.subplots()

    ax2.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    ax2.set_title("Top Features Influencing Capacity")

    st.pyplot(fig2)