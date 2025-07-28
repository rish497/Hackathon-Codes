# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Load original datasets
true = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

true['label'] = 0  # Real
fake['label'] = 1  # Fake
true['source'] = 'original'
fake['source'] = 'original'

# Combine original datasets
df = pd.concat([true, fake], ignore_index=True)

# ✅ Add feedback data if it exists
if os.path.exists("feedback.csv"):
    print("🔁 Loading user feedback...")
    feedback = pd.read_csv("feedback.csv", names=["text", "label"])
    feedback = feedback[feedback["text"].str.strip().astype(bool)]
    feedback["label"] = feedback["label"].astype(int)
    feedback["source"] = "user"
    df = pd.concat([df, feedback], ignore_index=True)

# Shuffle data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split
X = df["text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# Predict
y_pred = clf.predict(X_test_vec)

# Evaluate: Overall
print("\n🎯 Overall Model Evaluation:")
overall_acc = accuracy_score(y_test, y_pred)
print("Accuracy:", overall_acc)
report = classification_report(y_test, y_pred)
print(report)

# ✅ Evaluate on just feedback samples
X_test_df = df.iloc[X_test.index]
feedback_test = X_test_df[X_test_df["source"] == "user"]

if not feedback_test.empty:
    feedback_vec = vectorizer.transform(feedback_test["text"])
    feedback_preds = clf.predict(feedback_vec)
    feedback_acc = accuracy_score(feedback_test["label"], feedback_preds)
    
    print("\n📊 Evaluation on User Feedback Samples:")
    print("Accuracy:", feedback_acc)
    print(classification_report(feedback_test["label"], feedback_preds))
else:
    feedback_acc = None

# Save model and vectorizer
joblib.dump(clf, "fake_news_model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")
print("✅ Model and vectorizer saved.")

# ✅ Log results to file
with open("training_log.txt", "a", encoding="utf-8") as log:
    log.write("=== New Training Run ===\n")
    log.write(f"Total Samples: {len(df)}\n")
    log.write(f"Accuracy (Overall): {overall_acc}\n")
    if feedback_acc is not None:
        log.write(f"Accuracy (Feedback): {feedback_acc}\n")
    log.write(report + "\n")
    log.write("-" * 40 + "\n")
