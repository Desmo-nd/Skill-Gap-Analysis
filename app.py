from flask import Flask, render_template, request
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your training data with tokenized skills
training = pd.read_csv('working_best.csv')
skills = training['tokenized_skills'].tolist()

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(skills)

# Apply K-means clustering
n_clusters = 6  # Define the number of clusters
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X)

# Calculate the distribution of samples across clusters
skill_clusters = kmeans.predict(X)
cluster_counts = np.bincount(skill_clusters)

# Find the cluster with the highest and lowest demand
highest_demand_cluster = np.argmin(cluster_counts)
lowest_demand_cluster = np.argmax(cluster_counts)

# Save the trained model and vectorizer
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
joblib.dump(kmeans, 'kmeans_model.joblib')

# Render the index.html template for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Handle the form submission and predict skill demand levels
@app.route('/predict', methods=['POST'])
def predict():
    # Load the trained TF-IDF vectorizer and KMeans model
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    kmeans = joblib.load('kmeans_model.joblib')

    # Get the user input from the form
    user_input = request.form['skills']
    user_skills = [skill.strip() for skill in user_input.split(',')]

    # Transform user skills using the TF-IDF vectorizer
    user_skills_transformed = vectorizer.transform(user_skills)

    # Predict the cluster for each user skill
    user_skill_clusters = kmeans.predict(user_skills_transformed)

    # Determine the demand level of each skill
    demand_levels = []
    for skill, cluster in zip(user_skills, user_skill_clusters):
        if cluster == highest_demand_cluster:
            demand_levels.append("High demand")
        elif cluster == lowest_demand_cluster:
            demand_levels.append("Low demand")
        else:
            demand_levels.append("Middle demand")

    # Display the demand level for each skill
    predictions = []
    for skill, demand_level in zip(user_skills, demand_levels):
        predictions.append({"Skill": skill, "Demand Level": demand_level})

    return render_template('result.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
