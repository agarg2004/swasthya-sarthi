from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import random

# Initialize Flask app
app = Flask(__name__)

# Mock data and scaler (replace with your actual dataset and scaler)
data = pd.DataFrame({
    'Sex': [1, 0, 1],
    'Age': [25, 30, 35],
    'Height': [1.75, 1.60, 1.80],
    'Weight': [70, 65, 80],
    'Hypertension': [0, 1, 0],
    'Diabetes': [0, 0, 1],
    'BMI': [22.9, 25.4, 24.7],
    'Level': [0, 2, 1],
    'Fitness Goal': [1, 0, 1],
    'Fitness Type': [1, 0, 1],
    'Exercises': ['Yoga', 'Running', 'Weightlifting'],
    'Diet': ['High Protein', 'Balanced', 'Low Carb'],
    'Equipment': ['Mat', 'None', 'Weights']
})

class MockScaler:
    def transform(self, values):
        values = np.array(values)
        return (values - np.mean(values, axis=0)) / np.std(values, axis=0)

scaler = MockScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Extract form input
    user_input = {
        'Sex': int(request.form['Sex']),
        'Age': float(request.form['Age']),
        'Height': float(request.form['Height']),
        'Weight': float(request.form['Weight']),
        'Hypertension': int(request.form['Hypertension']),
        'Diabetes': int(request.form['Diabetes']),
        'BMI': float(request.form['BMI']),
        'Level': int(request.form['Level']),
        'Fitness Goal': int(request.form['FitnessGoal']),
        'Fitness Type': int(request.form['FitnessType'])
    }

    # Normalize input
    num_features = ['Age', 'Height', 'Weight', 'BMI']
    user_df = pd.DataFrame([user_input], columns=num_features)
    user_df[num_features] = scaler.transform(user_df[num_features])
    user_input.update(user_df.iloc[0].to_dict())
    user_df = pd.DataFrame([user_input])

    # Calculate similarity scores
    user_features = data[['Sex', 'Age', 'Height', 'Weight', 'Hypertension', 'Diabetes', 'BMI', 'Level', 'Fitness Goal', 'Fitness Type']]
    similarity_scores = cosine_similarity(user_features, user_df).flatten()
    similar_user_indices = similarity_scores.argsort()[-5:][::-1]
    similar_users = data.iloc[similar_user_indices]

    # Recommendation 1: Exact match
    recommendation_1 = similar_users[['Exercises', 'Diet', 'Equipment']].mode().iloc[0]

    # Simulated recommendations
    simulated_recommendations = []
    for _ in range(2):
        modified_input = user_input.copy()
        modified_input['Age'] += random.randint(-5, 5)
        modified_input['Weight'] += random.uniform(-5, 5)
        modified_input['BMI'] += random.uniform(-1, 1)

        modified_user_df = pd.DataFrame([modified_input], columns=num_features)
        modified_user_df[num_features] = scaler.transform(modified_user_df[num_features])
        modified_input.update(modified_user_df.iloc[0].to_dict())

        modified_similarity_scores = cosine_similarity(user_features, pd.DataFrame([modified_input])).flatten()
        modified_similar_user_indices = modified_similarity_scores.argsort()[-5:][::-1]
        modified_similar_users = data.iloc[modified_similar_user_indices]
        recommendation = modified_similar_users[['Exercises', 'Diet', 'Equipment']].mode().iloc[0]

        if not any(rec.equals(recommendation) for rec in simulated_recommendations):
            simulated_recommendations.append(recommendation)

    # Render recommendations
    return render_template('index.html', 
                           recommendation_1=recommendation_1.to_dict(), 
                           simulated_recommendations=[rec.to_dict() for rec in simulated_recommendations])

if __name__ == "__main__":
    app.run(debug=True)
