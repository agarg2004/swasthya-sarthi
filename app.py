import os
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ---- Load TogetherAI Chatbot tools ----
try:
    from langchain_together import ChatTogether
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False

# ---- Sidebar Navigation ----
st.set_page_config(page_title="Swasthya Sarthi", layout="centered")
PAGES = {
    "🏋️ Fitness Recommendation": "fitness",
    "💬 AI Fitness Chatbot": "chatbot",
}
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4697/4697975.png", width=100)
st.sidebar.title("Swasthya Sarthi")
sel_page = st.sidebar.radio("Go to", list(PAGES.keys()), label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.info(
    "Your smart fitness, workout and nutrition companion. "
    "**Switch tabs above** to chat with the AI or get personalized plans!"
)

# ---- Page 1: Fitness Recommendation ----
if PAGES[sel_page] == "fitness":
    st.title("🏋️ Personalized Fitness Recommendation")

    # Load data and models
    @st.cache_resource
    def load_fitness_assets():
        data = pd.read_excel('gym recommendation.xlsx')
        model = pickle.load(open('models/random.pkl', 'rb'))
        scaler = pickle.load(open('models/scaler_random.pkl', 'rb'))
        encoder = pickle.load(open('models/label_encoder.pkl', 'rb'))
        return data, model, scaler, encoder

    try:
        data, model, scaler, encoder = load_fitness_assets()
    except Exception as e:
        st.error(f"Model/Data error: {e}")
        st.stop()

    sex = st.selectbox("Sex", ["Female", "Male"])
    age = st.slider("Age", 10, 80, 25)
    height = st.number_input("Height (m)", value=1.75, min_value=1.0, max_value=2.5, step=0.01)
    weight = st.number_input("Weight (kg)", value=70.0, min_value=20.0, max_value=200.0, step=0.5)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    goal = st.selectbox("Fitness Goal", ["Weight Gain", "Weight Loss"])
    ftype = st.selectbox("Fitness Type", ["Cardio Fitness", "Muscular Fitness"])

    bmi = weight / (height ** 2)
    st.markdown(f"**BMI:** `{bmi:.2f}`")

    if st.button("Get Recommendation"):
        if bmi < 18.5:
            level = "Underweight"
            st.warning("Your BMI indicates you are Underweight. Please consult a healthcare provider for personalized advice.")
        elif bmi < 24.9:
            level = "Normal"
            st.success("Your BMI is in the Normal range. Keep up the good work!")
        elif bmi < 29.9:
            level = "Overweight"
            st.warning("Your BMI indicates you are Overweight or Obese. Consider consulting a healthcare provider for personalized advice.")
        else:
            level = "Obese"
            st.error("Your BMI indicates you are Obese. Please consult a healthcare provider for personalized advice.")

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

        # Scale numeric inputs
        X = pd.DataFrame([input_data])
        X[['Age', 'Height', 'Weight', 'BMI']] = scaler.transform(X[['Age', 'Height', 'Weight', 'BMI']])

        try:
            prediction = model.predict(X)
            decoded_value = encoder.inverse_transform(prediction)[0]

            st.markdown(f"### 🧠 **AI Recommendation:** `{decoded_value}`")
            st.markdown("**📋 Main Recommendation:**")
            st.info(data['Recommendation'].unique().tolist()[0])
            st.markdown("**💪 Exercises:**")
            st.success(data['Exercises'].unique().tolist()[0])
            st.markdown("**🍽️ Diet:**")
            st.success(data['Diet'].unique().tolist()[0])
            st.markdown("**🏋️ Equipment:**")
            st.success(data['Equipment'].unique().tolist()[0])
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---- Page 2: AI Fitness Chatbot ----
elif PAGES[sel_page] == "chatbot":
    st.title("💬 AI Fitness Chatbot")
    st.caption("Ask anything and everything about fitness, nutrition, exercises, or gym equipment.")

    if not TOGETHER_AVAILABLE:
        st.warning("Chatbot dependencies not installed. Please run `pip install langchain-together`.")
        st.stop()

    load_dotenv()
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")

    if not TOGETHER_API_KEY:
        st.error("TOGETHER_API_KEY not found in .env file.")
        st.stop()

    @st.cache_resource(show_spinner="⚡️ Starting chatbot...")
    def get_chain():
        llm = ChatTogether(
            together_api_key=TOGETHER_API_KEY,
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=0.5,
            max_tokens=512,
        )
        memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        return ConversationChain(llm=llm, memory=memory)

    chain = get_chain()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if not st.session_state.chat_history:
        with st.chat_message("assistant"):
            st.markdown(
                "<h3 style='margin-bottom: 0.5rem;'>Hi! I'm your Fitness AI Coach.</h3>"
                "<p>Ask me about workouts, diets, or fitness gear. Let's get you healthy!</p>", 
                unsafe_allow_html=True
            )

    # Show chat history
    for msg in st.session_state.chat_history:
        role, content = msg["role"], msg["message"]
        avatar = "🧑" if role == "user" else "🗣️"
        with st.chat_message(role, avatar=avatar):
            st.markdown(content)

    # Input and response
    if prompt := st.chat_input("Type your question..."):
        st.session_state.chat_history.append({"role": "user", "message": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🗣️"):
            with st.spinner("Thinking..."):
                try:
                    response = chain.run(prompt)
                    st.markdown(response)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "message": response
                    })
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "message": "Sorry, I ran into an error. Please try again."
                    })

    # Sidebar download chat
    def export_chat():
        import io
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for m in st.session_state.chat_history:
            who = "User" if m['role'] == "user" else "ChatBot"
            text = m["message"].replace('*', '').replace('#', '')
            pdf.multi_cell(0, 10, f"{who}: {text}\n")
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        return io.BytesIO(pdf_bytes)

    with st.sidebar:
        if st.session_state.chat_history:
            st.markdown("---\n**Download your session**")
            st.download_button(
                "📥 Download as PDF",
                data=export_chat(),
                file_name="fitness_chat_history.pdf",
                mime="application/pdf"
            )

# ---- END ----

