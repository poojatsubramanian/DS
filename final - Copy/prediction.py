import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



data=pd.read_csv(r'C:\Users\ACER\Desktop\final\extracted_output (1).csv')

# Create a mapping dictionary
sentiment_mapping = {'positive': 1, 'neutral':0,'negative': -1}

# Replace values in the 'sentiment' column using the mapping dictionary
data['sentiment'] = data['sentiment'].replace(sentiment_mapping)

X=data['clean_text']
y=data['sentiment']

vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression

model =LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Take input from the user
user_input = input("Enter a text message: ")

# Preprocess the user input and convert it into numerical features
user_input_features = vectorizer.transform([user_input])

# Make predictions on the user input
prediction = model.predict(user_input_features)

# Convert the prediction label to a human-readable sentiment
if prediction[0] == 1:
    sentiment = "Positive"
elif prediction[0] == 0:
    sentiment = "Neutral"
else:
    sentiment = "Negative"

# Print the sentiment prediction
print("Sentiment:", sentiment)

import pickle

# Save the trained model as a pickle file
with open('deploy.pickle', 'wb') as file:
    pickle.dump(model, file)

