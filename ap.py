import streamlit as st
import pandas as pd
import joblib

# Load model
@st.cache_resource
def load_model():
    return joblib.load("crop_recommendation_model.pkl")

model = load_model()

# Load feature column names (should match training)
feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Set page config
st.set_page_config(page_title="Crop Recommendation System", layout="centered")

st.title("Crop Recommendation System")
st.markdown("Provide soil and environmental parameters to receive a crop recommendation.")

# Input form
with st.form("input_form"):
    N = st.number_input("Nitrogen (N)", min_value=0, value=50)
    P = st.number_input("Phosphorus (P)", min_value=0, value=50)
    K = st.number_input("Potassium (K)", min_value=0, value=50)
    temperature = st.number_input("Temperature (°C)", format="%.2f", value=25.0)
    humidity = st.number_input("Humidity (%)", format="%.2f", value=80.0)
    ph = st.number_input("Soil pH", format="%.2f", value=6.5)
    rainfall = st.number_input("Rainfall (mm)", format="%.2f", value=200.0)
    
    submit = st.form_submit_button("Predict")

# Prediction logic
if submit:
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=feature_columns)
    prediction = model.predict(input_data)[0]
    st.success(f"Recommended Crop: **{prediction}**")

    # Store in session
    if "history" not in st.session_state:
        st.session_state["history"] = []

    st.session_state["history"].append({
        "N": N, "P": P, "K": K,
        "Temperature": temperature,
        "Humidity": humidity,
        "pH": ph,
        "Rainfall": rainfall,
        "Prediction": prediction
    })

# Show prediction history
if "history" in st.session_state and st.session_state["history"]:
    st.subheader("Prediction History")
    st.dataframe(pd.DataFrame(st.session_state["history"]), use_container_width=True)
