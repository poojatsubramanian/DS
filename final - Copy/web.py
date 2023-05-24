from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Read the dataset from CSV
data = pd.read_csv(r'C:\Users\ACER\Desktop\final\extracted_output (1).csv', encoding='utf-8')

# Split the data into features (messages) and labels
X = data['clean_text']
y = data['sentiment']

# Convert text data into numerical features using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open(r'C:\Users\ACER\Desktop\final\deploy.pickle','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['text']
    # Preprocess the user input and convert it into numerical features
    user_input_features = vectorizer.transform([user_input])

# Make predictions on the user input
    prediction = model.predict(user_input_features)

# Convert the prediction label to a human-readable sentiment
    sentiment = "Positive" if prediction[0] == 1 else "Neutral" if prediction[0] == 0 else "Negative"



# Print the sentiment prediction
    print("Sentiment:", sentiment)
   
    return render_template('result.html', prediction_text=f'Sentiment: {sentiment}')
if __name__ == '__main__':
    app.run(port=5100,debug=True)