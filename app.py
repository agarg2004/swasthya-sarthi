import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load data
data = pd.read_excel('gym recommendation.xlsx')

# Load models
model = pickle.load(open('random.pkl', 'rb'))
scaler = pickle.load(open('scaler_random.pkl', 'rb'))
encoder = pickle.load(open('label_encoder.pkl', 'rb'))

st.title("Personalized Fitness Recommendation")

# Inputs
sex = st.selectbox("Sex", ["Female", "Male"])
age = st.slider("Age", 10, 80, 25)
height = st.number_input("Height (m)", value=1.75)
weight = st.number_input("Weight (kg)", value=70.0)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
bmi = weight / (height ** 2)
goal = st.selectbox("Fitness Goal", ["Weight Gain", "Weight Loss"])
ftype = st.selectbox("Fitness Type", ["Cardio Fitness", "Muscular Fitness"])

if st.button("Get Recommendation"):
    if bmi < 18.5:
        level="Underweight"
        st.warning("Your BMI indicates you are Underweight. Please consult a healthcare provider for personalized advice.")
    elif bmi < 24.9:
        level="Normal"
        st.success("Your BMI is in the Normal range. Keep up the good work!")
    elif bmi < 29.9:
        level="Overweight"
        st.warning("Your BMI indicates you are Overweight or Obese. Consider consulting a healthcare provider for personalized advice.")
    else:
        level="Obese"
        st.error("Your BMI indicates you are Obese. Please consult a healthcare provider for personalized advice.")
    st.write(f"Calculated BMI: {bmi:.2f}")

    input_data = {
        'Sex': 1 if sex == "Male" else 0,
        'Age': age,
        'Height': height, 
        'Weight': weight,
        'Hypertension': 1 if hypertension == "Yes" else 0,
        'Diabetes': 1 if diabetes == "Yes" else 0,
        'BMI': bmi,
        'Level': {"Normal": 0, "Obese": 1, "Overweight": 2, "Underweight": 3}[level],
        'Fitness Goal': 0 if goal == "Weight Gain" else 1,
        'Fitness Type': 0 if ftype == "Cardio Fitness" else 1
    }

    df = pd.DataFrame([input_data])
    df[['Age', 'Height', 'Weight', 'BMI']] = scaler.transform(df[['Age', 'Height', 'Weight', 'BMI']])
     #Predict
    prediction = model.predict(df)

    # st.write("Scaled Input to Model:", df)
    decoded = encoder.inverse_transform(prediction)[0]
    decoded_value = encoder.inverse_transform(prediction)[0]

    # Show unique recommendations
    st.markdown("**📋 Unique Recommendation in Dataset:**")
    st.markdown(data['Recommendation'].unique().tolist()[0])

    st.markdown("**💪 Exercises:**")
    st.markdown(data['Exercises'].unique().tolist()[0])

    st.markdown("**🍽️ Diet:**")
    st.markdown(data['Diet'].unique().tolist()[0])

    st.markdown("**🏋️ Equipment:**")
    st.markdown(data['Equipment'].unique().tolist()[0])

   