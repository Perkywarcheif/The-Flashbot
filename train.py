import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load intents from JSON file
with open("intents.json", "r") as file:
    intents = json.load(file)

# Prepare training data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train TfidfVectorizer and Logistic Regression model
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(patterns)
y = tags
clf = LogisticRegression(random_state=0, max_iter=10000)
clf.fit(x, y)

# Save the trained model and vectorizer
joblib.dump(clf, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved as model.pkl and vectorizer.pkl")