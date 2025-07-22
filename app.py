import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load('best_model.pkl')

st.set_page_config(page_title="Employee Income Prediction", page_icon="💼", layout="centered")
st.title("💼 Employee Income Prediction App")
st.markdown("Enter employee details to predict whether income >50K or ≤50K.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")
age = st.sidebar.slider("Age", 18, 100, 30)
education_num = st.sidebar.slider("Education Number", 1, 20, 10)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 100, 40)
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, value=0)

# Input as 2D array
input_data = np.array([[age, education_num, hours_per_week, capital_gain, capital_loss]])

st.write("### 🔎 Input Data")
st.json({
    "age": age,
    "education_num": education_num,
    "hours_per_week": hours_per_week,
    "capital_gain": capital_gain,
    "capital_loss": capital_loss
})

if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("✅ Prediction: Income >50K")
    else:
        st.info("ℹ️ Prediction: Income ≤50K")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with columns [age, education_num, hours_per_week, capital_gain, capital_loss]", type="csv")

if uploaded_file is not None:
    import pandas as pd
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())
    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
