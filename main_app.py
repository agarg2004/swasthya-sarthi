import streamlit as st
import pickle
import pandas as pd

# ---------- Load models ----------
model = pickle.load(open('random.pkl', 'rb'))
scaler = pickle.load(open('scaler_random.pkl', 'rb'))
encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# ---------- Session State ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- Sidebar Navigation ----------
st.sidebar.title("🏋️ Navigation")
page = st.sidebar.radio("Go to", ["Fitness Recommendation", "Chatbot"])

# ---------- Page 1: Fitness Recommendation ----------
if page == "Fitness Recommendation":
    st.title("🏋️ Personalized Fitness Recommendation")

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
        prediction = model.predict(df)
        decoded = encoder.inverse_transform(prediction)

        st.success(f"🏆 Recommended Plan: {decoded[0]}")

# ---------- Page 2: Chatbot ----------
elif page == "Chatbot":
    st.title("💬 AI Fitness Chatbot")
    st.write("Ask me about workouts, diets, BMI, or anything fitness related.")

    user_input = st.text_input("You:", key="chat_input")

    if st.button("Send", key="send_button"):
        if user_input:
            # Simulated chatbot response (replace this with real LLM or logic if needed)
            response = f"I'm a fitness assistant! You said: {user_input}"

            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", response))

    # Display chat history
    for sender, msg in st.session_state.chat_history:
        st.markdown(f"**{sender}:** {msg}")
