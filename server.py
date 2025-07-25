import json
from os import environ as env
from urllib.parse import quote_plus, urlencode

from authlib.integrations.flask_client import OAuth
from dotenv import find_dotenv, load_dotenv
from flask import Flask, redirect, render_template, session, url_for, request, flash
import os
from werkzeug.utils import secure_filename

from utils.ocr_predictor import extract_text_from_file
import pickle

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

app = Flask(__name__)
app.secret_key = env.get("APP_SECRET_KEY")

# Auth0 setup
oauth = OAuth(app)
oauth.register(
    "auth0",
    client_id=env.get("AUTH0_CLIENT_ID"),
    client_secret=env.get("AUTH0_CLIENT_SECRET"),
    client_kwargs={"scope": "openid profile email"},
    server_metadata_url=f'https://{env.get("AUTH0_DOMAIN")}/.well-known/openid-configuration',
)

# Upload folder
UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load ML model and vectorizer
with open("model/fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Routes
@app.route("/")
def home():
    return render_template("home.html", session=session.get("user"), pretty=json.dumps(session.get("user"), indent=4))

@app.route("/callback", methods=["GET", "POST"])
def callback():
    token = oauth.auth0.authorize_access_token()
    session["user"] = token
    return redirect("/")

@app.route("/login")
def login():
    return oauth.auth0.authorize_redirect(redirect_uri=url_for("callback", _external=True))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(
        "https://" + env.get("AUTH0_DOMAIN") + "/v2/logout?" +
        urlencode({
            "returnTo": url_for("home", _external=True),
            "client_id": env.get("AUTH0_CLIENT_ID"),
        }, quote_via=quote_plus)
    )

@app.route("/detector", methods=["GET", "POST"])
def detector():
    prediction = None
    extracted_text = None

    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            extracted_text = extract_text_from_file(filepath)
            if extracted_text:
                vect_text = vectorizer.transform([extracted_text])
                prediction = model.predict(vect_text)[0]
            else:
                prediction = "No readable text found."

    return render_template("detector.html", prediction=prediction, extracted_text=extracted_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=env.get("PORT", 3000))
