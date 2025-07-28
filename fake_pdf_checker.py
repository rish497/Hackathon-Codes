import os
import json
import io
import joblib
import fitz  # PyMuPDF
import pytesseract
import threading
from PIL import Image
import pandas as pd
from flask import Flask, request, render_template, redirect, session, url_for, jsonify, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv, find_dotenv
from authlib.integrations.flask_client import OAuth
from urllib.parse import quote_plus, urlencode
import google.generativeai as genai
import requests
from difflib import SequenceMatcher
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
from flask import send_file
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from waitress import serve


print("hello WOrld")
nltk.download('punkt')
nltk.download('stopwords')

# --- ENV / INIT ---
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

app = Flask(__name__)
app.secret_key = os.getenv("APP_SECRET_KEY")
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}

# --- Auth0 Setup ---
oauth = OAuth(app)
oauth.register(
    "auth0",
    client_id=os.getenv("AUTH0_CLIENT_ID"),
    client_secret=os.getenv("AUTH0_CLIENT_SECRET"),
    client_kwargs={"scope": "openid profile email"},
    server_metadata_url=f'https://{os.getenv("AUTH0_DOMAIN")}/.well-known/openid-configuration',
)

# --- Load Model + Vectorizer ---
clf = joblib.load("fake_news_model.joblib")
vectorizer_model = joblib.load("vectorizer.joblib")

# --- Configure Gemini ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
chat_session = gemini_model.start_chat(history=[])

# --- NEWS API ---
NEWSDATA_API_KEY = "pub_857237df36ae42f6b66ceb19f1ceff73"

# --- Utils ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()
    text = ""

    if ext == "pdf":
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        if not text.strip():
            for page in doc:
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes()))
                text += pytesseract.image_to_string(img)
    elif ext in ["png", "jpg", "jpeg"]:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)

    return text.strip()

def retrain_model_async():
    def retrain():
        global clf, vectorizer_model
        true = pd.read_csv("True.csv")
        fake = pd.read_csv("Fake.csv")
        true['label'] = 0
        fake['label'] = 1
        df = pd.concat([true, fake])
        if os.path.exists("feedback.csv"):
            feedback_df = pd.read_csv("feedback.csv", names=["text", "label"])
            df = pd.concat([df, feedback_df], ignore_index=True)
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        X_train, _, y_train, _ = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        X_train_vec = vectorizer.fit_transform(X_train)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_vec, y_train)
        joblib.dump(model, "fake_news_model.joblib")
        joblib.dump(vectorizer, "vectorizer.joblib")
        clf = model
        vectorizer_model = vectorizer
        print("✅ Model retrained and reloaded from feedback.")

    threading.Thread(target=retrain).start()

def fetch_news_articles(query):
    url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&q={query}&language=en&category=top"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return [article['title'] + " " + article.get('description', '') for article in data.get("results", [])]
        else:
            print(f"❌ News API Error - Status Code: {response.status_code}")
            return []
    except Exception as e:
        print("❌ News API Error:", e)
        return []

def second_check_news_similarity(text):
    from nltk import pos_tag, ne_chunk
    from nltk.tree import Tree

    def extract_keywords(text, max_keywords=10):
        words = word_tokenize(text)
        filtered = [w for w in words if w.isalnum() and w.lower() not in stopwords.words('english')]
        return filtered[:max_keywords]

    def extract_named_entities(text):
        named_entities = []
        for chunk in ne_chunk(pos_tag(word_tokenize(text))):
            if isinstance(chunk, Tree):
                ne = " ".join(c[0] for c in chunk)
                named_entities.append(ne)
        return named_entities

    levels = []

    # Level 1: Top 10 keywords (most specific)
    keywords = extract_keywords(text, max_keywords=10)
    levels.append(("Top 10 Keywords", " ".join(keywords)))

    # Level 2: Top 5 keywords (less specific)
    levels.append(("Top 5 Keywords", " ".join(keywords[:5])))

    # Level 3: Named Entities (even broader)
    entities = extract_named_entities(text)
    if entities:
        levels.append(("Named Entities", " ".join(entities)))

    # Level 4: First line/headline fallback
    first_line = text.strip().splitlines()[0] if text.strip().splitlines() else text[:100]
    levels.append(("First Line", first_line))

    # Try each level until we fetch articles
    for level_name, query in levels:
        print(f"\n🔎 Trying query level: {level_name}")
        print(f"🧠 Query: \"{query}\"")
        articles = fetch_news_articles(query)
        if articles:
            print(f"✅ Found {len(articles)} article(s) using {level_name}:\n")
            for i, article in enumerate(articles, 1):
                print(f"{i}. 📰 {article[:200]}...\n")
            # Similarity check
            for article in articles:
                sim = SequenceMatcher(None, text.lower(), article.lower()).ratio()
                if sim > 0.6:
                    print(f"✅ Similarity Match Found! Ratio: {sim:.2f}")
                    return True, articles
            print("❌ No article matched closely enough. Continuing fallback...\n")
        else:
            print(f"⚠️ No articles found with query level: {level_name}")

    print("❌ Final Verdict: No matching articles found through any fallback method.")
    return False, []



# --- Routes ---
@app.route("/")
def home():
    return render_template("home.html", session=session.get("user"))

@app.route("/login")
def login():
    return oauth.auth0.authorize_redirect(redirect_uri=url_for("callback", _external=True))

@app.route("/callback", methods=["GET", "POST"])
def callback():
    token = oauth.auth0.authorize_access_token()
    session["user"] = token
    return redirect("/")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(
        f"https://{os.getenv('AUTH0_DOMAIN')}/v2/logout?" + urlencode({
            "returnTo": url_for("home", _external=True),
            "client_id": os.getenv("AUTH0_CLIENT_ID"),
        }, quote_via=quote_plus)
    )

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    try:
        response = chat_session.send_message(user_input)
        return jsonify({"response": response.candidates[0].content.parts[0].text})
    except Exception as e:
        return jsonify({"error": f"Gemini API Error: {str(e)}"}), 500

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        flash("No file uploaded.")
        return redirect(url_for("home"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("home"))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        extracted_text = extract_text(filepath)
        if not extracted_text:
            return render_template("result.html", prediction="No readable text found.", extracted_text="")

        text_vector = vectorizer_model.transform([extracted_text])
        prediction_label = clf.predict(text_vector)[0]
        prediction_result = "Fake News" if prediction_label == 1 else "Real News"

        # Second check
        articles = []
        if prediction_result == "Fake News":
            found_match, articles = second_check_news_similarity(extracted_text)

            # DEBUG: Print all fetched articles to terminal
            print("\n📦 Articles returned from second check:")
            if articles:
                for i, article in enumerate(articles, 1):
                    print(f"{i}. 📰 {article[:200]}...\n")
            else:
                print("⚠️ No articles fetched from NewsData.io.")

            if found_match:
                prediction_result = "Real News ✅ (Verified by news data)"

        return render_template(
            "result.html",
            prediction=prediction_result,
            extracted_text=extracted_text,
            articles=articles
        )

    flash("Invalid file type.")
    return redirect(url_for("home"))

@app.route("/feedback", methods=["POST"])
def feedback():
    text = request.form.get("text")
    label = request.form.get("label")
    if not text or label not in ["0", "1"]:
        return "Invalid feedback", 400
    with open("feedback.csv", "a", encoding="utf-8") as f:
        safe_text = text.strip().replace('"', "'").replace("\n", " ")
        f.write(f'"{safe_text}",{label}\n')
    flash("✅ Thanks! Your feedback was saved and model will learn shortly.")
    retrain_model_async()
    return redirect(url_for("home"))

@app.route("/download-report", methods=["POST"])
def download_report():
    extracted_text = request.form.get("text", "")
    prediction = request.form.get("prediction", "")
    response = request.form.get("response", "")
    articles = request.form.get("articles", "").split("|||")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"news_report_{timestamp}.pdf"
    filepath = os.path.join("static", "reports", filename)
    os.makedirs("static/reports", exist_ok=True)

    c = canvas.Canvas(filepath, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Fake News Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Timestamp: {timestamp}")
    c.drawString(50, height - 100, f"Prediction: {prediction}")

    y = height - 130
    if response:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Gemini Explanation:")
        y -= 20
        c.setFont("Helvetica", 10)
        for line in response.splitlines():
            c.drawString(60, y, line)
            y -= 15

    if articles:
        y -= 30
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"News Articles Fetched ({len(articles)}):")
        y -= 20
        c.setFont("Helvetica", 9)
        for article in articles:
            for line in article.splitlines():
                if y < 50:
                    c.showPage()
                    y = height - 50
                c.drawString(60, y, line[:100])
                y -= 12

    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Extracted Text:")
    y -= 20
    c.setFont("Helvetica", 9)
    for line in extracted_text.splitlines():
        if y < 50:
            c.showPage()
            y = height - 50
        c.drawString(60, y, line[:100])
        y -= 12

    c.save()
    return send_file(filepath, as_attachment=True)
if __name__ == "__main__":
    from waitress import serve
    print("🚀 App running at: http://127.0.0.1:3000 or http://localhost:3000")
    print("✅ Using Waitress (production WSGI server)")
    serve(app, host="0.0.0.0", port=3000)
