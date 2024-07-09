from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
import google.generativeai as genai
import time
from requests.exceptions import HTTPError
import re
from fpdf import FPDF
from flask import send_file
import io
from flask import jsonify
import traceback
import unicodedata

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Set the API key for the generative model
os.environ['GOOGLE_API_KEY'] = 'YOUR_API_KEY'
# Configure the generative AI with the API key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Define the generative model
generative_model = genai.GenerativeModel('gemini-1.5-flash')

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Initialize the database
with app.app_context():
    db.create_all()

def preprocess_data(file_path, missing_value_threshold=0.1):
    df = pd.read_csv(file_path)
    num_rows = df.shape[0]

    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            z_scores = np.abs(stats.zscore(df[col]))
            outlier_indices = np.where(z_scores > 3)[0]
            df = df.drop(df.index[outlier_indices])

    missing_values = df.isna().sum()
    missing_proportion = missing_values / df.shape[0]

    for col in df.columns:
        if missing_proportion[col] > missing_value_threshold:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].mean())
            elif df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df = df.dropna(subset=[col])

    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].astype(int)
            except ValueError:
                try:
                    df[col] = df[col].astype(float)
                except ValueError:
                    pass

    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_cols)

    small_df = df.sample(frac=0.05, random_state=42)

    return df, small_df

def generate_text(prompt, retries=3, delay=5):
    for i in range(retries):
        try:
            response = generative_model.generate_content(prompt)
            return response._result.candidates[0].content.parts[0].text
        except HTTPError as e:
            if e.response.status_code == 429:
                print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise
    raise Exception("Max retries exceeded")

# Function to parse query and generate appropriate DataFrame operation
def parse_query_gpt(query, file_info):
    prompt = f"""
    Dataset info: {file_info}
    Query: {query}
    Provide the appropriate answer according to the given dataset info.
    """
    return generate_text(prompt)


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
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password')
    
    return render_template('login.html')

@app.route('/index')
def index():
    if 'username' in session:
        user_name = session['username']
    else:
        user_name = 'Current User'
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


@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    bot_response = ""

    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '' and file.filename.endswith('.csv'):
            session.clear()
            session.modified = True  
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the uploaded CSV file
            df, smalldf = preprocess_data(file_path)
            
            shape = df.shape
            col = df.columns.tolist()
            dtypes = df.dtypes.apply(lambda x: x.name).to_dict()  # Convert dtype objects to string names
            nulls = df.isnull().sum().to_dict()
            stats = df.describe().to_dict()
            uniques = df.nunique().to_dict()

            file_info = {
                'shape': shape,
                'columns': col,
                'dtypes': dtypes,
                'nulls': nulls,
                'stats': stats,
                'uniques': uniques
            }
            
            query = f"Please provide information about the file based on this data:"
            bot_response = parse_query_gpt(query, file_info)
            bot_response = re.sub(r'\*\*(.*?)\*\*', r'<h3>\1</h3>', bot_response )
            bot_response = bot_response.replace('*', '')
            txt = {'Information': bot_response}
            session['bot'] = txt
        else:
            bot_response = "Please upload a valid CSV file."
    elif 'bot' in session:
        if 'file' in user_message.lower() or 'data' in user_message.lower() or 'rows' in user_message.lower() or 'column' in user_message.lower():
            txt = session['bot']
            prompt = f"Query: '{user_message}'. Answer according to this provided info: {txt}"
        else:
            prompt = f"User's message: '{user_message}'. Please respond to this message."

        bot_response = generate_text(prompt)
        bot_response = re.sub(r'\*\*(.*?)\*\*', r'<h3>\1</h3>', bot_response )
        bot_response = bot_response.replace('*', '')
    else:
        prompt = f"User's message: '{user_message}'. Please respond to this message."
        bot_response = generate_text(prompt)
        bot_response = re.sub(r'\*\*(.*?)\*\*', r'<h3>\1</h3>', bot_response )
        bot_response = bot_response.replace('*', '')

    return ({'response': bot_response})

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Chat History', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def write_message(self, sender, text):
        filtered_text = self.filter_text(text)
        if filtered_text.strip():  # Only add non-empty lines
            self.set_font("Arial", 'B', 10)
            self.multi_cell(0, 10, f"{sender}:", 0, 'L')
            self.set_font("Arial", '', 10)
            self.multi_cell(0, 10, filtered_text)
            self.ln(5)

    def filter_text(self, text):
        # Remove characters not supported by latin-1 encoding
        return ''.join(c for c in text if unicodedata.category(c)[0] != 'C')

@app.route('/save_pdf', methods=['POST'])
def save_pdf():
    try:
        chat_messages = request.json['messages']
        
        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)
        pdf.set_auto_page_break(auto=True, margin=15)
        
        for message in chat_messages:
            pdf.write_message(message['sender'], message['text'])
        
        pdf_output = io.BytesIO()
        pdf_output.write(pdf.output(dest='S').encode('latin-1'))
        pdf_output.seek(0)
        
        return send_file(
            pdf_output,
            as_attachment=True,
            download_name="chat_history.pdf",
            mimetype="application/pdf"
        )
    except Exception as e:
        app.logger.error(f"Error generating PDF: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('user_id', None)
    session.pop('dataset_info', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
