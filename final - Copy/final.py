# -*- coding: utf-8 -*-
"""final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cRyqt8crmOQmIXJP3hYQOT_RhpCz3hW7
"""

pip install scikit-plot

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_excel('sorted.xlsx')
data

"""# Exploratory Data Analysis"""

data.shape

data.info()

data.describe()

data.dtypes

#Checking null values

data.isna().sum()

#Checking unique values in columns

for column in data.columns:
   print(f"Unique values in {column}: {data[column].nunique()}")

#Displaying Categorical columns

cat_data=data.select_dtypes(include=['object','bool'])
cat_data

#Dropping unnecessary columns

data=data.drop(['user_name'],axis=1)
data=data.drop(['user_username'],axis=1)
data=data.drop(['user_location'],axis=1)

#Displaying numerical columns

num_data=data.select_dtypes(include=['float','int64'])
num_data

#Dropping unnecessary columns

data=data.drop(['source'],axis=1)

pd.set_option('display.max_columns', None)
data

"""#2.3 Generating New Time Variables"""

# Generating new time variables
data['tweet_created'] = pd.to_datetime(data['tweet_created'])
data['OnlyDate'] = data['tweet_created'].dt.date
data['OnlyDate'] = pd.to_datetime(data['OnlyDate'])
data['OnlyHour'] = data['tweet_created'].dt.hour
data['OnlyMin'] = data['tweet_created'].dt.minute

"""#2.4 First Exploratory Data Analysis Works"""

def plot_time_variable(col, ylim_lower = 100, ylim_upper = 3000):
    """
    Given a pandas dataframe and the name of a time column, this function will plot a line graph of time counts
    using the specified column.
    
    Parameters:
    -----------
    col : str
        The name of a column in the pandas dataframe.
    """
    
    if data[col].dtype == "int64":
        time_variable_counts = data[col].value_counts().sort_index()
        
    else:
        # calculate the count of dates using resample
        time_variable_counts = data[col].value_counts().resample('D').sum()

    # set the size of the figure
    plt.figure(figsize=(12, 8))

    # plot the counts using a line graph
    time_variable_counts.plot(kind='line', marker='o', markersize=8)

    # set the y-axis limits to a specific range
    plt.ylim(ylim_lower, ylim_upper)

    # add graph labels and titles
    plt.title(f"{col} Counts", fontsize=16)
    plt.xlabel(f"{col}", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(rotation=45, ha='right') # rotate x-axis labels for readability
    plt.grid(axis='y', linestyle='--')

    # display the graph
    plt.show()

plot_time_variable('OnlyDate')

plot_time_variable('OnlyHour', 100, 3000)

"""# Handling missing values"""

data.isna().sum()

data.dropna(inplace=True)

data.isna().sum()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer

import string

import nltk

nltk.download('punkt')
nltk.download('wordnet')

stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove @mentions
    text = re.sub(r'#\w+', '', text) # remove hashtags
    text = re.sub(r'RT[\s]+', '', text) # remove retweets
    text = re.sub(r'https?:\/\/\S+', '', text) # remove hyperlinks
    text = re.sub(r'[^\x00-\x7F]+', '', text) # remove non-ASCII characters
    text = re.sub(r'\d+', '', text)  # remove numeric values
    text = re.sub(r'[^\w\s]', '', text) # remove symbols
    text = re.sub(r'http|co', '', text)  # remove 'http', 'https', 'co'
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) # remove punctuations
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens] # stemming
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens] # lemmatization
    clean_text = ' '.join(lemmatized_tokens)
    return clean_text

data['clean_text'] = data['text'].apply(clean_text)
data['clean_text'] = data['clean_text'].str.lower() # lowercase

data

"""# Sentiment Analysis and Model


"""

# Function to perform sentiment analysis on a piece of text using the TextBlob library in Python.
# get_sentiment_polarity() function uses the TextBlob library to analyze the sentiment polarity of a given text.
from textblob import TextBlob
import numpy as np

def get_sentiment_polarity(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

data['sentiment_polarity'] = data['clean_text'].apply(get_sentiment_polarity)
data['sentiment'] = np.where(data['sentiment_polarity'] > 0, 'positive',
                             np.where(data['sentiment_polarity'] < 0, 'negative', 'neutral'))

data

# count the frequency of each sentiment label
sentiment_counts = data.sentiment.value_counts()

# create a column plot
fig, ax = plt.subplots(figsize=(8,6))
sentiment_counts.plot(kind='bar', ax=ax)

# set the plot title and axis labels
ax.set_title('Sentiment Label Frequencies')
ax.set_xlabel('Sentiment Label')
ax.set_ylabel('Frequency')

# add data labels to the top of each column
for i, freq in enumerate(sentiment_counts):
    ax.text(i, freq, str(freq), ha='center', va='bottom')

# display the plot
plt.show()

import plotly.offline as pyo
import plotly.graph_objects as go
# Get the value counts for user_verified
value_counts = data['sentiment'].value_counts()

## Create a pie chart using Plotly
labels = value_counts.index
values = value_counts.values

trace = go.Pie(labels=labels, values=values)
layout = go.Layout(title='Sentiment Classification', width=600, height=400)

fig = go.Figure(data=[trace], layout=layout)

# Display the chart
pyo.iplot(fig)

# Create a new DataFrame with counts of each sentiment per user_verified
count_data = data.groupby(['user_verified', 'sentiment']).size().unstack(fill_value=0)


# Set up the triple bar chart
fig, ax = plt.subplots()

bar_width = 0.25
x_pos = np.arange(len(count_data.index))

ax.bar(x_pos - bar_width, count_data['negative'], width=bar_width, label='Negative')
ax.bar(x_pos, count_data['positive'], width=bar_width, label='Positive')
ax.bar(x_pos + bar_width, count_data['neutral'], width=bar_width, label='Neutral')

# Add labels and legend
ax.set_xlabel('User Verified')
ax.set_ylabel('Number of Tweets')
ax.set_title("Sentiment Analysis of Tweets by User Verification")
ax.set_xticks(x_pos)
ax.set_xticklabels(count_data.index, rotation=45)
ax.legend()

# Show the plot
plt.show()

# Create a new DataFrame with counts of each sentiment per user_verified
count_data = data.groupby(['OnlyDate', 'sentiment']).size().unstack(fill_value=0)

# Set up the triple bar chart
fig, ax = plt.subplots()

bar_width = 0.25
x_pos = np.arange(len(count_data.index))

ax.bar(x_pos - bar_width, count_data['negative'], width=bar_width, label='Negative')
ax.bar(x_pos, count_data['positive'], width=bar_width, label='Positive')
ax.bar(x_pos + bar_width, count_data['neutral'], width=bar_width, label='Neutral')

# Add labels and legend
ax.set_xlabel('OnlyDate')
ax.set_ylabel('Number of Tweets')
ax.set_title("Sentiment Analysis of Tweets by dates")
ax.set_xticks(x_pos)
ax.set_xticklabels(count_data.index, rotation=45)
ax.legend()

# Show the plot
plt.show()

data.head()

specific_row = data['clean_text'][3]
print(specific_row)

import nltk
nltk.download('stopwords')

# Define stop words
stop_words = set(stopwords.words('english'))

# Define function to remove stop words
def remove_stopwords(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stop words
    filtered_tokens = [token for token in tokens if not token in stop_words]
    # Join the filtered tokens back into a string
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

# Apply the function to the dataframe column
data['clean_text'] = data['clean_text'].apply(remove_stopwords)

from nltk.stem import PorterStemmer

def perform_stemming(text):
    stemmer = PorterStemmer()
    new_list = []
    words = word_tokenize(text)
    for word in words:
        new_list.append(stemmer.stem(word))

    return " ".join(new_list)

data['clean_text'] = data['clean_text'].apply(perform_stemming)

data.head()

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from wordcloud import WordCloud
import matplotlib.pyplot as plt

"""# Wordcloud for positive tweets about chatgpt"""

text = " ".join(data[data['sentiment'] == 'positive']['clean_text'])
plt.figure(figsize = (15, 10))
wordcloud = WordCloud(max_words=500, height= 800, width = 1500,  background_color="black", colormap= 'viridis').generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

text = " ".join(data[data['sentiment'] == 'negative']['clean_text'])
plt.figure(figsize = (15, 10))
wordcloud = WordCloud(max_words=500, height= 800, width = 1500,  background_color="black", colormap= 'viridis').generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

text = " ".join(data[data['sentiment'] == 'neutral']['clean_text'])
plt.figure(figsize = (15, 10))
wordcloud = WordCloud(max_words=500, height= 800, width = 1500,  background_color="black", colormap= 'viridis').generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

from collections import Counter

all_nodep_words = []
for sentence in data[data['sentiment'] == 'positive']['clean_text'].to_list():
    for word in sentence.split():
        all_nodep_words.append(word)

df = pd.DataFrame(Counter(all_nodep_words).most_common(25), columns= ['Word', 'Frequency'])

sns.set_context('notebook', font_scale= 1.3)
plt.figure(figsize=(18,8))
sns.barplot(y = df['Word'], x= df['Frequency'], palette= 'summer')
plt.title("Most Commonly Used Words fo positive comments ")
plt.xlabel("Frequnecy")
plt.ylabel("Words")
plt.show()

all_dep_words = []
for sentence in data[data['sentiment'] == 'negative']['clean_text'].to_list():
    for word in sentence.split():
        all_dep_words.append(word)

df = pd.DataFrame(Counter(all_dep_words).most_common(25), columns= ['Word', 'Frequency'])

sns.set_context('notebook', font_scale= 1.3)
plt.figure(figsize=(18,8))
sns.barplot(y = df['Word'], x= df['Frequency'], palette= 'summer')
plt.title("Most Commonly Used Words fo negative comments")
plt.xlabel("Frequnecy")
plt.ylabel("Words")
plt.show()

all_dep_words = []
for sentence in data[data['sentiment'] == 'neutral']['clean_text'].to_list():
    for word in sentence.split():
        all_dep_words.append(word)

df = pd.DataFrame(Counter(all_dep_words).most_common(25), columns= ['Word', 'Frequency'])

sns.set_context('notebook', font_scale= 1.3)
plt.figure(figsize=(18,8))
sns.barplot(y = df['Word'], x= df['Frequency'], palette= 'summer')
plt.title("Most Commonly Used Words fo neutral comments")
plt.xlabel("Frequnecy")
plt.ylabel("Words")
plt.show()

# Specify the file path and name for the output CSV file
output_file_path = 'output.csv'

# Output the DataFrame as a CSV file
data.to_csv(output_file_path, index=False)





data.isna().sum()

"""# Vectorization"""

'''import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score'''



data.head()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Split the data into training and testing sets
X = data['clean_text']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Define the models to evaluate
models = [
    ('Gaussian Naive Bayes', GaussianNB()),
    ('Multinomial Naive Bayes', MultinomialNB()),
    ('Bernoulli Naive Bayes', BernoulliNB()),
    ('Complement Naive Bayes', ComplementNB()),
    ('Random Forest', RandomForestClassifier())
]

# Evaluate each model
for model_name, model in models:
    print(f'----- {model_name} -----')
    # Train the model
    model.fit(X_train_vec.toarray(), y_train)
  
    # Make predictions on the test set
    y_pred = model.predict(X_test_vec.toarray())
  
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
  
    # Print the evaluation metrics
    print('Confusion Matrix:')
    print(cm)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('\n')

data.head()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Split the data into features (messages) and labels
X = data['clean_text']
y = data['sentiment']

# Convert text data into numerical features using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
X

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Take input from the user
user_input = input("Enter a text message: ")

# Preprocess the user input and convert it into numerical features
user_input_features = vectorizer.transform([user_input])

# Make predictions on the user input
prediction = model.predict(user_input_features)


# Convert the numeric sentiment label to text
if sentiment == "positive":
    sentiment_text = "Positive"
elif sentiment == "neutral":
    sentiment_text = "Neutral"
else:
    sentiment_text = "Negative"

# Print the sentiment prediction
print("Sentiment:", sentiment_text)

print(prediction)











