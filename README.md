````markdown
# 💬 Fitness Chatbot using Together AI + Streamlit

A web-based chatbot built using [Streamlit](https://streamlit.io/) and [Together AI's](https://together.ai) large language models (LLMs), such as **Mixtral-8x7B**, via the Together API. This chatbot can help users with fitness questions, workout tips, and more!

---

## 🚀 Features

- 🧠 Powered by **Together AI LLMs** (Mixtral, LLaMA, etc.)
- ⚡️ Simple Streamlit interface
- 💾 Conversational memory for context-aware responses
- 🔐 `.env`-based secure API key handling

---

## 🖥️ Demo

<img src="https://i.imgur.com/CNUQEf2.png" alt="Demo Screenshot" width="600">

---

## 🛠️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/fitness-chatbot-together.git
cd fitness-chatbot-together
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit langchain langchain-community together python-dotenv
```

### 3️⃣ Set Your Together API Key

Create a `.env` file in the root directory:

```bash
TOGETHER_API_KEY=your_actual_api_key_here
```

You can get a free key from: [https://app.together.ai](https://app.together.ai)

### 4️⃣ Run the App

```bash
streamlit run app.py
```

---

## 🧠 How It Works

* Uses `langchain_community.chat_models.ChatTogether` to access Together's models
* Maintains a conversation history using LangChain’s `ConversationBufferMemory`
* UI powered by Streamlit
* Accepts one question at a time and provides AI-generated responses

---

## 🔐 Environment Variables

| Variable           | Description              |
| ------------------ | ------------------------ |
| `TOGETHER_API_KEY` | Your Together AI API key |

---

## 🤖 Supported Models

You can change the model in `app.py`:

```python
model="mistralai/Mixtral-8x7B-Instruct-v0.1"
```

Or try:

* `meta-llama/Llama-2-70b-chat-hf`
* `tiiuae/falcon-180B-chat`
* and others listed on [https://docs.together.ai/docs/inference-models](https://docs.together.ai/docs/inference-models)

---

## 🧾 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Support

Feel free to open an [Issue](https://github.com/<your-username>/fitness-chatbot-together/issues) or submit a [Pull Request](https://github.com/<your-username>/fitness-chatbot-together/pulls) if you'd like to contribute or need help.

---

## 💡 Future Ideas

* 🧍 Personalized fitness recommendations
* 🗣️ Voice input & TTS response
* 🧾 Chat history saving
* 📊 Recommendation analytics

---

Made with ❤️ by **Anirudh Garg**

```
