
````markdown
# 🥗 Diet and Exercise Recommendation Web App

This is a web-based application built with Streamlit that provides personalized diet and exercise recommendations based on user inputs like age, weight, height, gender, and fitness goals. The recommendations are generated using a trained **Random Forest Classifier** model.

---

## 🚀 Features

- Accepts user input for age, weight, height, gender, and goal.
- Uses a machine learning model (Random Forest Classifier) for prediction.
- Recommends tailored diet and exercise plans.
- Clean and responsive Streamlit interface.

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/agarg2004/swasthya-sarthi
````

2. **Create a virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure your model file `random_forest_model.pkl` is placed in the root directory.**

---

## 🧠 Model Training (Optional)

If you'd like to train your own model, you can use the `model.ipynb` script provided in the repository:

```bash
python model.ipynb
```

This script trains a `RandomForestClassifier` on your dataset and saves the model to `random_forest.pkl`.

---

## 🚦 How to Run the App

Once everything is set up, run the app with:

```bash
streamlit run app1.py
```

This will launch a local development server and open the app in your web browser.

---

## 🖼️ User Interface

The app allows users to input the following:

* Age
* Gender (Male/Female)
* Height (in m)
* Weight (in kg)
* Goal (e.g., Lose Weight, Gain Weight)

And it returns:

* Recommended diet plan
* Suggested exercise types


---

## 📜 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Author

**Anirudh Garg**
For questions or feedback, feel free to reach out!

---

## ⭐ Acknowledgements

* [Streamlit](https://streamlit.io/)
* [Scikit-learn](https://scikit-learn.org/)
* [Pandas](https://pandas.pydata.org/)

---
