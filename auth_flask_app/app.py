from flask import Flask, redirect, render_template, session, url_for
from flask_session import Session  # NEW
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ["APP_SECRET_KEY"]

# Session configuration
app.config['SESSION_TYPE'] = 'filesystem'  # NEW: prevents CSRF issues
Session(app)  # NEW: initialize server-side session

# OAuth Setup
oauth = OAuth(app)

auth0 = oauth.register(
    'auth0',
    client_id=os.environ["AUTH0_CLIENT_ID"],
    client_secret=os.environ["AUTH0_CLIENT_SECRET"],
    api_base_url=f"https://{os.environ['AUTH0_DOMAIN']}",
    access_token_url=f"https://{os.environ['AUTH0_DOMAIN']}/oauth/token",
    authorize_url=f"https://{os.environ['AUTH0_DOMAIN']}/authorize",
    client_kwargs={
        'scope': 'openid profile email',
    },
)

# ... [rest of your routes]

# Routes

@app.route('/')
def home():
    return render_template('home.html')  # Must have templates/home.html

@app.route('/login')
def login():
    return auth0.authorize_redirect(redirect_uri='http://localhost:5000/callback')

@app.route('/callback')
def callback():
    token = auth0.authorize_access_token()
    session['user'] = token['userinfo']
    return redirect('/dashboard')

@app.route('/dashboard')
def dashboard():
    userinfo = session.get('user')
    if not userinfo:
        return redirect('/')
    return f"Welcome {userinfo['name']}! Your email: {userinfo['email']}"

@app.route('/logout')
def logout():
    session.clear()
    return redirect(
        f"https://{os.environ['AUTH0_DOMAIN']}/v2/logout?returnTo=http://localhost:5000/&client_id={os.environ['AUTH0_CLIENT_ID']}"
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)
