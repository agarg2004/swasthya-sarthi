import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from warnings import filterwarnings

# Suppress version mismatch warnings
filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load models with version mismatch handling
try:
    model = pickle.load(open('random.pkl', 'rb'))
    scaler = pickle.load(open('scaler_random.pkl', 'rb'))
    
    # Handle label encoder with version mismatch
    try:
        encoder = pickle.load(open('label_encoder.pkl', 'rb'))
    except Exception as e:
        st.warning("Label encoder loading issue, creating new encoder")
        encoder = LabelEncoder()
        encoder.classes_ = np.array([0, 1])  # Manually set known classes
        
    # Debug info
    print(f"Encoder classes: {encoder.classes_}")
    
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

st.title("🏋️ Personalized Fitness Recommendation")

# Input form with validation
with st.form("user_inputs"):
    sex = st.selectbox("Sex", ["Female", "Male"])
    age = st.slider("Age", 10, 80, 25)
    height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.75)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    bmi = weight / (height ** 2)
    level = st.selectbox("BMI Level", ["Normal", "Obese", "Overweight", "Underweight"])
    goal = st.selectbox("Fitness Goal", ["Weight Gain", "Weight Loss"])
    ftype = st.selectbox("Fitness Type", ["Cardio Fitness", "Muscular Fitness"])
    submitted = st.form_submit_button("Get Recommendation")

if submitted:
    try:
        # Prepare input data with validation
        input_data = {
            'Sex': 1 if sex == "Male" else 0,
            'age': float(age),
            'Height': float(height),
            'weight': float(weight),
            'Hypertension': 1 if hypertension == "Yes" else 0,
            'Diabetes': 1 if diabetes == "Yes" else 0,
            'BMI': float(bmi),
            'Level': {"Normal": 0, "Obese": 1, "Overweight": 2, "Underweight": 3}[level],
            'Fitness Goal': 0 if goal == "Weight Gain" else 1,
            'Fitness Type': 0 if ftype == "Cardio Fitness" else 1
        }

        df = pd.DataFrame([input_data])
        
        # Scale features with column name handling
        scale_cols = ['age', 'Height', 'weight', 'BMI']
        temp_df = df[scale_cols].copy()
        temp_df.columns = ['Age', 'Height', 'Weight', 'BMI']  # Match scaler expectations
        df[scale_cols] = scaler.transform(temp_df)
        
        # Prepare final input
        final_df = df[model.feature_names_in_]
        
        # Make prediction
        prediction = model.predict(final_df)
        print(f"Raw prediction: {prediction}")

        # Handle prediction results
        if prediction[0] in encoder.classes_:
            decoded = encoder.inverse_transform(prediction)
            st.success(f"## 🏆 Recommended Plan: {decoded[0]}")
        else:
            st.warning("""
            **Note:** Our system has generated a recommendation outside our standard plans.
            This typically means your profile has unique characteristics that deserve special attention.
            """)
            
            # Advanced fallback logic
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(final_df)[0]
                valid_indices = [i for i, cls in enumerate(encoder.classes_)]
                if valid_indices:
                    best_valid = np.argmax(proba[valid_indices])
                    decoded = encoder.inverse_transform([encoder.classes_[best_valid]])
                    confidence = proba[valid_indices][best_valid]
                    st.success(f"## Closest Matching Plan: {decoded[0]} (confidence: {confidence:.1%})")
                else:
                    st.info("We recommend consulting with our fitness experts for a personalized plan")
            else:
                st.info("""
                We recommend scheduling a consultation with our fitness specialists
                who can design a custom program for your unique needs.
                """)
                
            # Show raw prediction for debugging
            st.write(f"System code: {prediction[0]} (this helps our trainers understand your needs)")
                
    except Exception as e:
        st.error(f"Error in processing: {str(e)}")
        print(f"Error details: {str(e)}")