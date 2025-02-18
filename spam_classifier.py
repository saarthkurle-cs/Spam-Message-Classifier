import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv")
df = df[['label', 'tweet']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({0: 'ham', 1: 'spam'})

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub("\d+", "", text)
    return text.strip()

df['message'] = df['message'].apply(preprocess_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Build model
model = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

# Train model
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# Predict function
def predict_message(message):
    processed_message = preprocess_text(message)
    return model.predict([processed_message])[0]

# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("Enter a message (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        print("Prediction:", predict_message(user_input))
