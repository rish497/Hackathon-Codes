# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle

# Load sample dataset (you can swap with real one later)
df = pd.read_csv("https://raw.githubusercontent.com/saranya97/Fake-News-Detection/main/news.csv")

# Prepare data
X = df['text'].fillna('')
y = df['label']

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_vectorized, y)

# Save model and vectorizer
import os
os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved.")
