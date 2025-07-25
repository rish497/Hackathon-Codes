import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

# ✅ USE WORKING LINK
df = pd.read_csv("https://raw.githubusercontent.com/GeorgeMcIntire/fake_real_news_dataset/main/fake_and_real_news_dataset.csv")

# Basic cleanup
df = df.dropna(subset=['text', 'label'])
X = df['text']
y = df['label'].map({'REAL': 0, 'FAKE': 1})

# Train
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('clf', PassiveAggressiveClassifier(max_iter=50))
])
pipeline.fit(X, y)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/fake_news_model.pkl")
print("✅ Model trained and saved at model/fake_news_model.pkl")
