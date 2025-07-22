🏋️ Swasthya Sarthi: AI-Powered Fitness Recommendation System
![Streamlit UI](https://img.shields.io/badge/Built-with%20Streamlit](https://img.shields.io/badge/ML-RandomForest%20%2B](https://img.shields.io/badge/License-MIT-lightgreyarthi** is an AI-driven fitness assistant and chatbot that gives personalized workout, diet, and fitness equipment recommendations based on user input. Built with 🧠 machine learning, 🤖 Streamlit, and ⚡ LangChain/Together AI.

✅ Features
💪 Fitness Plan Recommendations (based on RandomForestClassifier)

🤖 Interactive Fitness Chatbot (via Together AI’s Mixtral model)

📊 KMeans-based Cluster Recommendations

📈 BMI calculation with health insights

🧠 Memory and Context-aware conversation

💾 Trainable ML pipeline with exported .pkl models

💬 PDF Export of chat sessions

✅ Voice input (optional)

📁 Clean modular structure

📁 Folder Structure
text
├── app.py                            # Main Streamlit app
├── train_model.py                    # Training script for RandomForest
├── train_kmeans.py                   # (optional) KMeans training
├── requirements.txt                  # Required Python libraries
├── gym recommendation.xlsx           # Dataset
├── models/                           # Saved models and encoders
│   ├── random.pkl
│   ├── scaler_random.pkl
│   ├── label_encoders.pkl
│   ├── target_encoder.pkl
│   ├── kmeans.pkl
│   ├── scaler_kmeans.pkl
│   └── kmeans_features.json
├── .env                              # Environment variables (for API keys)
└── README.md                         # This file!
🧠 Technologies Used
Category	Tools Used
Frontend	Streamlit
Machine Learning	scikit-learn (RandomForest, KMeans)
NLP / LLM Integration	OpenAI / Together AI (LangChain)
Data	Pandas, NumPy
File I/O	joblib, pickle, openpyxl
Export & Voice (extra)	FPDF, pyttsx3, SpeechRecognition
🚀 How to Run This App Locally
🔧 1. Clone the Repo
bash
git clone https://github.com/your-username/swasthya-sarthi.git
cd swasthya-sarthi
📦 2. Install Requirements
bash
pip install -r requirements.txt
🔑 3. Set Up Environment Variable
Create a .env file in the root folder:

text
TOGETHER_API_KEY=your_together_api_key_here
▶️ 4. Run Streamlit App
bash
streamlit run app.py
📊 Dataset Format
Your gym recommendation.xlsx should include:

Age	Height	Weight	BMI	Sex	...	Recommendation	Exercises	Diet	Equipment
🔐 Security Note
Do not expose your .env or API keys publicly.

.env should be added in .gitignore.

📚 Future Enhancements
✅ Multi-page UI with chatbot/document reader tabs

📂 File uploader for personalized document analysis (LangChain Q&A)

🧠 Hybrid recommender (KMeans filter → RandomForest ranker)

📶 Streamlit Cloud & HuggingFace Space deployment

📱 Mobile UI view optimization

💡 Contributing
Contributions are welcome! Please open issues or submit a pull request with suggestions.

📝 License
This project is licensed under the MIT License. See the LICENSE file for details.

🙌 Acknowledgements
Streamlit for making rapid app dev a breeze

Together AI for free access to large language models

scikit-learn for powerful machine learning algorithms

📬 Contact
Author: [Your Name]
Email: your.email@example.com
GitHub: @your-username
