# server.py
import json
from os import environ as env
from urllib.parse import quote_plus, urlencode

from authlib.integrations.flask_client import OAuth
from dotenv import find_dotenv, load_dotenv



from flask import Flask, render_template, request, redirect, url_for, session


import os
import joblib
from utils.ocr_predictor import extract_text_from_file

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("APP_SECRET_KEY")

# Auth0 setup
oauth = OAuth(app)
auth0 = oauth.register(
    'auth0',
    client_id=os.getenv("AUTH0_CLIENT_ID"),
    client_secret=os.getenv("AUTH0_CLIENT_SECRET"),
    api_base_url=os.getenv("AUTH0_DOMAIN"),
    access_token_url=os.getenv("AUTH0_DOMAIN") + "/oauth/token",
    authorize_url=os.getenv("AUTH0_DOMAIN") + "/authorize",
    client_kwargs={
        'scope': 'openid profile email',
    },
)

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


# Load fake news model
model = joblib.load("model/fake_news_model.pkl")

@app.route('/detector', methods=['GET', 'POST'])
def detector():
    prediction = None
    extracted_text = ""
    
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            file_path = os.path.join('static/uploads', uploaded_file.filename)
            uploaded_file.save(file_path)

            extracted_text = extract_text_from_file(file_path)
            if extracted_text.strip():
                prediction = model.predict([extracted_text])[0]

    return render_template('detector.html', prediction=prediction, text=extracted_text)

if __name__ == '__main__':
    app.run(debug=True, port=3000)
