from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import magic  # pip install python-magic

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Replace with the generated secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Initialize the database
with app.app_context():
    db.create_all()

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            # Store user information in session
            session['user_id'] = user.id
            session['username'] = user.username
            # Redirect to index page upon successful login
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password')
    
    return render_template('login.html')

@app.route('/index')
def index():
    if 'username' in session:
        user_name = session['username']
    else:
        user_name = 'Current User'  # Fallback in case session data is missing
    return render_template('index.html', user_name=user_name)

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']

    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
    new_user = User(username=username, email=email, password=hashed_password)

    try:
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    except:
        flash('Email address already exists')
        return redirect(url_for('login'))
'''
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    bot_response = f"You said: {user_message}"
    return {'response': bot_response}
'''
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    bot_response = f"{user_message}"

    # Check if a file was uploaded
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            filename = secure_filename(file.filename)
            
            # Get file metadata before saving
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)

            # Get file type using python-magic
            mime = magic.Magic(mime=True)
            file_type = mime.from_buffer(file.read(1024))
            file.seek(0)

            # Save the file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Get file creation and modification times
            file_stats = os.stat(file_path)
            creation_time = datetime.fromtimestamp(file_stats.st_ctime)
            modification_time = datetime.fromtimestamp(file_stats.st_mtime)

            metadata = f"""
            File '{filename}' has been saved.
            Size: {file_size} bytes
            Type: {file_type}
            Created: {creation_time}
            Modified: {modification_time}
            """

            bot_response += "\n" + metadata

    return {'response': bot_response}

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('user_id', None)
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
