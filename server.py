"""Python Flask WebApp Auth0 integration example
"""

import json
from os import environ as env
from urllib.parse import quote_plus, urlencode

from werkzeug.utils import secure_filename

from authlib.integrations.flask_client import OAuth
from dotenv import find_dotenv, load_dotenv
from flask import Flask, redirect, render_template, session, url_for

import google.generativeai as genai
from flask import request, jsonify
import os

from PIL import Image
import pytesseract
import io
import fitz  # PyMuPDF, for PDFs

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = env.get("APP_SECRET_KEY")

# Load Gemini API key from .env
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

genai.configure(api_key=api_key)

# Load Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")
chat_session = model.start_chat(history=[])

oauth = OAuth(app)

oauth.register(
    "auth0",
    client_id=env.get("AUTH0_CLIENT_ID"),
    client_secret=env.get("AUTH0_CLIENT_SECRET"),
    client_kwargs={
        "scope": "openid profile email",
    },
    server_metadata_url=f'https://{env.get("AUTH0_DOMAIN")}/.well-known/openid-configuration',
)


# Controllers API
@app.route("/")
def home():
    return render_template(
        "home.html",
        session=session.get("user"),
        pretty=json.dumps(session.get("user"), indent=4),
    )


@app.route("/callback", methods=["GET", "POST"])
def callback():
    token = oauth.auth0.authorize_access_token()
    session["user"] = token
    return redirect("/")


@app.route("/login")
def login():
    return oauth.auth0.authorize_redirect(
        redirect_uri=url_for("callback", _external=True)
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(
        "https://"
        + env.get("AUTH0_DOMAIN")
        + "/v2/logout?"
        + urlencode(
            {
                "returnTo": url_for("home", _external=True),
                "client_id": env.get("AUTH0_CLIENT_ID"),
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

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)

        extracted_text = ""
        if filename.lower().endswith(".pdf"):
            doc = fitz.open(path)
            for page in doc:
                extracted_text += page.get_text()
        else:
            image = Image.open(path)
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            extracted_text = pytesseract.image_to_string(image)

        try:
            response = chat_session.send_message(
                f"Tell me if the following text is real news or fake news, and why:\n\n{extracted_text}"
            )
            return render_template("detector.html", prediction="Check below 👇", response=response.text)
        except Exception as e:
            return f"Gemini API Error: {str(e)}", 500

    return "Invalid file type", 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=env.get("PORT", 3000))