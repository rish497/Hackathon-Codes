import os
import json
import io
import joblib
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from flask import Flask, request, render_template, redirect, session, url_for, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv, find_dotenv
from authlib.integrations.flask_client import OAuth
from urllib.parse import quote_plus, urlencode
import google.generativeai as genai

# --- ENV / INIT ---
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

app = Flask(__name__)
app.secret_key = os.getenv("APP_SECRET_KEY")
app.config["UPLOAD_FOLDER"] = "static/uploads"
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

# --- Load ML Model + Vectorizer ---
model = joblib.load("fake_news_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# --- Configure Gemini ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
chat_session = gemini_model.start_chat(history=[])

# --- Utilities ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()
    text = ""

    if ext == "pdf":
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()

        if not text.strip():  # fallback to OCR
            for page in doc:
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes()))
                text += pytesseract.image_to_string(img)
    elif ext in ["png", "jpg", "jpeg"]:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)

    return text.strip()

# --- Routes ---

@app.route("/")
def home():
    return render_template(
        "home.html",
        session=session.get("user"),
        pretty=json.dumps(session.get("user"), indent=4),
    )

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
        "https://" + os.getenv("AUTH0_DOMAIN") + "/v2/logout?" + urlencode(
            {
                "returnTo": url_for("home", _external=True),
                "client_id": os.getenv("AUTH0_CLIENT_ID"),
            },
            quote_via=quote_plus,
        )
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
        import traceback
        print("Gemini API Error:", traceback.format_exc())
        return jsonify({"error": f"Gemini API Error: {str(e)}"}), 500

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return render_template("home.html", error="No file uploaded.")

    file = request.files["file"]
    if file.filename == "":
        return render_template("home.html", error="No file selected.")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        extracted_text = extract_text(filepath)
        if not extracted_text:
            return render_template("result.html", prediction="No readable text found.")

        # Local model prediction
        text_vector = vectorizer.transform([extracted_text])
        prediction_label = model.predict(text_vector)[0]
        prediction_result = "Fake News" if prediction_label == 1 else "Real News"

        return render_template("result.html", prediction=prediction_result)

    return render_template("home.html", error="Invalid file type.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 3000)))
