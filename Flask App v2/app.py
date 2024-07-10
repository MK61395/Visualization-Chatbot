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
from flask_mail import Mail, Message
import re
import unicodedata
from datetime import datetime



app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db = SQLAlchemy(app)
migrate = Migrate(app, db)





app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'm.kashi613@gmail.com' # Use an app password
app.config['MAIL_PASSWORD'] = 'your password'
app.config['MAIL_DEFAULT_SENDER'] = ('PlotPal', 'm.kashi613@gmail.com')

mail = Mail(app)





# Set the API key for the generative model
# os.environ['GOOGLE_API_KEY'] = ''
os.environ['GOOGLE_API_KEY'] = 'Your_API_Key'
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

def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_non_printable(text):
    return ''.join(c for c in text if c.isprintable() or c.isspace())


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
            
            session['file_info'] = file_info
            query = f"Please provide information about the file based on this data:"
            bot_response = parse_query_gpt(query, file_info)
            bot_response = re.sub(r'\*\*(.*?)\*\*', r'<h3>\1</h3>', bot_response )
            bot_response = bot_response.replace('*', '')
        else:
            bot_response = "Please upload a valid CSV file."
    elif 'file_info' in session:
        if 'file' in user_message.lower() or 'data' in user_message.lower() or 'rows' in user_message.lower() or 'column' in user_message.lower():
            file_info = session['file_info']
            # Example of handling additional user messages that relate to the CSV data
            if 'shape' in user_message.lower():
                prompt = f"The shape of the uploaded CSV is {file_info['shape']}."
            elif 'columns' in user_message.lower():
                prompt = f"The columns in the uploaded CSV are {file_info['columns']}."
            elif 'nulls' in user_message.lower():
                prompt = f"The number of null values in each column is {file_info['nulls']}."
            elif 'stats' in user_message.lower():
                prompt = f"The statistical summary of the CSV is {file_info['stats']}."
            else:
                prompt = parse_query_gpt(user_message, file_info)
            bot_response = generate_text(prompt)
            bot_response = re.sub(r'\*\*(.*?)\*\*', r'<h3>\1</h3>', bot_response )
            bot_response = bot_response.replace('*', '')
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
        cleaned_text = remove_non_printable(remove_emojis(text))
        if cleaned_text.strip():  # Only add non-empty lines
            self.set_font("Arial", 'B', 10)
            self.multi_cell(0, 10, f"{sender}:", 0, 'L')
            self.set_font("Arial", '', 10)
            self.multi_cell(0, 10, cleaned_text)
            self.ln(5)

@app.route('/save_pdf', methods=['POST'])
def save_pdf():
    try:
        chat_messages = request.json['messages']
        
        pdf = PDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Add date and time at the top
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, f"Chat History - Generated on {current_time}", 0, 1, "C")
        pdf.ln(10)  # Add some space after the header
        
        # Reset font for messages
        pdf.set_font("Arial", "", 12)
        
        for message in chat_messages:
            pdf.write_message(message['sender'], message['text'])
        
        pdf_output = io.BytesIO()
        pdf_output.write(pdf.output(dest='S').encode('latin-1', errors='ignore'))
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
    


@app.route('/email_chat', methods=['POST'])
def email_chat():
    try:
        if 'user_id' not in session:
            return jsonify({"error": "User not logged in"}), 401

        user_id = session['user_id']
        user = User.query.filter_by(id=user_id).first()

        if not user:
            return jsonify({"error": "User not found"}), 404

        chat_messages = request.json['messages']

        email_body = "Chat History:\n\n"
        for message in chat_messages:
            email_body += f"{message['sender']}: {message['text']}\n\n"

        msg = Message('Chat History', recipients=[user.email])
        msg.body = email_body

        mail.send(msg)

        return jsonify({"message": "Email sent successfully"}), 200
    except Exception as e:
        app.logger.error(f"Error sending email: {str(e)}")
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
