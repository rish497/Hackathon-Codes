# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load datasets
true = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

true['label'] = 0
fake['label'] = 1

df = pd.concat([true, fake]).sample(frac=1).reset_index(drop=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# Evaluate
y_pred = clf.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save
joblib.dump(clf, "fake_news_model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")
print("✅ Model and vectorizer saved.")
