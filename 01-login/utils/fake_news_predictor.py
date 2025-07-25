import joblib
import os
import re

# Load model + vectorizer
MODEL_PATH = os.path.abspath("fake_news_model.pkl")
VECTORIZER_PATH = os.path.abspath("vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def predict_fake_news(text):
    cleaned = clean_text(text)
    vec_text = vectorizer.transform([cleaned])
    prediction = model.predict(vec_text)[0]
    return "FAKE" if prediction == 1 else "REAL"
