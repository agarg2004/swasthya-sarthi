import streamlit as st
import pickle
import pandas as pd

# Load models
model = pickle.load(open('random.pkl', 'rb'))
scaler = pickle.load(open('scaler_random.pkl', 'rb'))
encoder = pickle.load(open('label_encoder.pkl', 'rb'))

st.title("🏋️ Personalized Fitness Recommendation")

# Inputs
sex = st.selectbox("Sex", ["Female", "Male"])
age = st.slider("Age", 10, 80, 25)
height = st.number_input("Height (m)", value=1.75)
weight = st.number_input("Weight (kg)", value=70.0)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
bmi = weight / (height ** 2)
level = st.selectbox("BMI Level", ["Normal", "Obese", "Overweight", "Underweight"])
goal = st.selectbox("Fitness Goal", ["Weight Gain", "Weight Loss"])
ftype = st.selectbox("Fitness Type", ["Cardio Fitness", "Muscular Fitness"])

if st.button("Get Recommendation"):
    input_data = {
        'Sex': 1 if sex == "Male" else 0,
        'Age': age,
        'Height': height,  # Changed from 'height' to 'Height'
        'Weight': weight,
        'Hypertension': 1 if hypertension == "Yes" else 0,
        'Diabetes': 1 if diabetes == "Yes" else 0,
        'BMI': bmi,
        'Level': {"Normal": 0, "Obese": 1, "Overweight": 2, "Underweight": 3}[level],
        'Fitness Goal': 0 if goal == "Weight Gain" else 1,
        'Fitness Type': 0 if ftype == "Cardio Fitness" else 1
    }

    df = pd.DataFrame([input_data])
    # Make sure the columns in the transform match exactly what was used during training
    df[['Age', 'Height', 'Weight', 'BMI']] = scaler.transform(df[['Age', 'Height', 'Weight', 'BMI']])
    prediction = model.predict(df)
    decoded = encoder.inverse_transform(prediction)

    st.success(f"🏆 Recommended Plan: {decoded[0]}")