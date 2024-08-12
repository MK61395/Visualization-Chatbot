from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_from_directory
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
from fpdf import FPDF, HTMLMixin
from flask import send_file
import io
from flask import jsonify
import traceback
import unicodedata
from flask_mail import Mail, Message
import re
import unicodedata
from datetime import datetime
import base64
from io import BytesIO 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import plotly.express as px
import plotly.io as pio
import base64
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from PIL import Image as PILImage
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

load_dotenv()



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
app.config['MAIL_PASSWORD'] = 'srrp pxgw ixxz rmnh'
app.config['MAIL_DEFAULT_SENDER'] = ('PlotPal', 'm.kashi613@gmail.com')

mail = Mail(app)

#key for second account backup: 'AIzaSyAO-ux6eN_If4qrEkqk5KjWYE7KbUdwL3s'
# Set the API key for the generative model
# os.environ['GOOGLE_API_KEY'] = 'AIzaSyBDclYIUCOkq9gQ8NQwFpJ55mFC-IC3Koo'
google_api_key = os.getenv('GOOGLE_API_KEY')

# Ensure the API key was loaded successfully
if not google_api_key:
    raise ValueError("Google API key not found in environment variables")

# Configure the generative AI with the API key
genai.configure(api_key=google_api_key)

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


def fig_to_base64(fig):
     """Convert a matplotlib figure to a base64 encoded string."""
     buf = BytesIO()
     fig.savefig(buf, format='png')
     buf.seek(0)
     img_base64 = base64.b64encode(buf.read()).decode('utf-8')
     buf.close()
     return img_base64

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

    # categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
    # df = pd.get_dummies(df, columns=categorical_cols)

    # small_df = df.sample(frac=0.05, random_state=42)

    return df

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



# Function to determine the context of the query
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def execute_query(data, query_code):
    try:
        # Ensure the graphs directory exists
        graphs_dir = 'graphs'
        ensure_dir(graphs_dir)

        # Extract code from within the triple backticks
        query_code = query_code.strip("```python").strip()
        local_vars = {'df': data, 'plt': plt, 'px': px, 'result': None}

        # Execute the code within a try-except block
        try:
            exec(query_code, {}, local_vars)
        except Exception as code_error:
            error_message = f"Error in generated code: {str(code_error)}"
            print(error_message)
            return {'type': 'error', 'content': error_message}

        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fig_{timestamp}.png"
        image_path = os.path.join(graphs_dir, filename)

        if 'fig' in local_vars:
            fig = local_vars['fig']
            if isinstance(fig, plt.Figure):
                # Matplotlib figure
                fig.savefig(image_path)
                plt.close(fig)  # Close the figure to free up memory
            elif 'plotly.graph_objs._figure.Figure' in str(type(fig)):
                # Plotly figure
                pio.write_image(fig, image_path)
            else:
                raise ValueError("Unsupported figure type")

            return {'type': 'image', 'content': f'/graphs/{filename}'}
        elif local_vars['result'] is not None:
            if isinstance(local_vars['result'], pd.DataFrame):
                # Convert DataFrame to HTML table
                html_table = local_vars['result'].to_html(classes='dataframe', index=False)
                return {'type': 'table', 'content': html_table}
            else:
                return {'type': 'text', 'content': str(local_vars['result'])}
        else:
            return {'type': 'text', 'content': "Query executed successfully, but no result or figure was produced."}

    except Exception as e:
        import traceback
        error_message = f"Error executing query: {str(e)}\n{traceback.format_exc()}"
        print(error_message)  # Log the full error
        return {'type': 'error', 'content': f"Error executing query: {str(e)}"}
    
n = 0
def get_query_context(query):
    prompt = f"""
    Query: {query}
    Based on the query, determine if it is related to pandas operations, matplotlib visualizations, general data operations, or if it is a general query not related to the dataset.
    Respond with one of the following keywords: 'pandas', 'matplotlib', 'general', 'non-data'.
    """
    return generate_text(prompt).strip().lower()


def is_tree_structure(dataframe):
    # Step 1: Create the graph from the dataframe
    graph = defaultdict(list)
    in_degree = defaultdict(int)

    for index, row in dataframe.iterrows():
        parent = row.iloc[0] 
        for child in row[1:]:
            if pd.notna(child):  # Check if the child value is not NaN
                graph[parent].append(child)
                in_degree[child] += 1
                if parent not in in_degree:
                    in_degree[parent] = 0
    
    # Step 2: Find all nodes with in-degree 0 (potential roots)
    roots = [node for node, degree in in_degree.items() if degree == 0]
    
    # There must be exactly one root node for a valid tree
    if len(roots) != 1:
        return False
    
    # Step 3: Check if the graph is connected and acyclic
    root = roots[0]
    visited = set()
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        if node in visited:
            return False
        visited.add(node)
        for neighbor in graph[node]:
            queue.append(neighbor)
    
    # All nodes must be visited once
    if len(visited) != len(in_degree):
        return False
    
    return True


def _graphs(df):
    relationships = []

    # Fill NaN values or drop rows with NaN values
    df = df.dropna()

    # Checking pairwise relationships
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2:
                # Numerical Correlation
                if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                    if df[col1].std() != 0 and df[col2].std() != 0:  # Check for zero standard deviation
                        corr = df[col1].corr(df[col2])
                        if abs(corr) > 0.5:  # Arbitrary threshold for correlation strength
                            relationships.append((col1, col2, {'weight': corr}))
                # Categorical Co-occurrence
                elif pd.api.types.is_object_dtype(df[col1]) or pd.api.types.is_object_dtype(df[col2]):
                    common_values = len(set(df[col1]).intersection(set(df[col2])))
                    if common_values > 0:
                        relationships.append((col1, col2, {'weight': common_values}))
    return relationships

    
# Function to parse query and generate appropriate visualization code
def parse_query_gpt(file_path, context, query, columns, dtypes, nulls, stats, uniques, shape):
    if context == 'matplotlib':
        df = preprocess_data(file_path)
    #    # relationships = identify_relationships(df)
    #     if is_tree_structure(df):
    #         print(f"Identified Relationship Tree")
    #     else:
    #         print("No hierarchical relationships detected between columns. NO TREE DATA STRUCTURE")

    #     relationships =_graphs(df)
    #     if relationships:
    #         print(f"Identified Relationship Graph")
    #     else:
    #         print("No hierarchical relationships detected between columns. NO GRAPH DATA STRUCTURE")

#  The figure size must be, Width: 9.5 inches, Height: 6 inches. 
        print(context)
        prompt = f"""
        Dataframe: {df}
        Dataset info: {columns, dtypes, nulls, stats, uniques, shape}
        Query: {query}
        Provide the appropriate visualization code using only Matplotlib or Plotly library to fulfill this query. Do not use any other library. Labels and Title should be proper readable and everything should be inside the boundary of graph. The background of graph should be light seagreen color, But the labels in the graph should be in white color and their background should be dark, so user can understand the text labels. (Only for Pie chart: If you are drawing pie chart then use different color combination, but make sure text should be visible and shows what every color represents). If graph requires only numerical columns like float or int then ignore all categorical columns for that graph and vice versa. Ensure to assign the plot object to a variable named 'fig'. Only give code output. Give clear code that will handle all possible cases, I will not bear any kind of error because this is really serious, Understand. Remember that all files are uploaded and saved. No need to import csv file, it's dataframe is already given to you. Remember that file name is {file_path}
        I will kick your ass if your code contains any error or did not work properly."""
    elif context == 'pandas':
        print(context)
        prompt = f"""
        Dataset columns: {columns, dtypes, nulls, stats, uniques, shape}
        Query: {query}
        Provide the appropriate DataFrame operation to fulfill this query in pandas. Give one line answer only. Ensure to assign the result to a variable named 'result'. Remember that all files are uploaded and saved. No need to import csv file.
        """
    elif context == 'general':
        print(context)
        prompt = f"""
        Dataset columns: {columns}
        Data types: {dtypes}
        Null values: {nulls}
        Descriptive stats: {stats}
        Unique values: {uniques}
        Shape: {shape}
        Query: {query}
        Provide the appropriate general information about the dataset.
        """
        
    elif context == 'non-data':
        print(context)
        prompt = query
        
    
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
            session['email'] = email
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
    

@app.route('/graphs/<path:filename>')
def serve_image(filename):
    return send_from_directory('graphs', filename)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # File successfully uploaded, enable options
        return jsonify({'success': True, 'message': 'File uploaded successfully'})
    else:
        return jsonify({'error': 'Invalid file type'})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}


@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    bot_response = ""
    max_retries = 3

    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '' and file.filename.endswith('.csv'):
            session.clear()
            session.modified = True  
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            small_df = preprocess_data(file_path)
            
            shape = small_df.shape
            columns = small_df.columns.tolist()
            dtypes = {col: str(dtype) for col, dtype in small_df.dtypes.to_dict().items()}
            nulls = small_df.isnull().sum().to_dict()
            stats = small_df.describe().to_dict()
            uniques = small_df.nunique().to_dict()
            
            file_info = {
                'shape': shape,
                'columns': columns,
                'dtypes': dtypes,
                'nulls': nulls,
                'stats': stats,
                'uniques': uniques
            }

            session['file_info'] = file_info
            session['file_path'] = file_path
            context = 'general'
            query = f"{user_message} Please provide information about the file based on this data:"
            bot_response = parse_query_gpt(file_path, context, query, columns, dtypes, nulls, stats, uniques, shape)
            bot_response = re.sub(r'\*\*(.*?)\*\*', r'<h3>\1</h3>', bot_response)
            bot_response = bot_response.replace('*', '')
        else:
            bot_response = "Please upload a valid CSV file."
    elif 'file_info' in session and 'file_path' in session:
        file_info = session['file_info']
        file_path = session['file_path']
        small_df = preprocess_data(file_path)
        prompt = f"Query: '{user_message}'."
        shape = file_info['shape']
        columns = file_info['columns']
        dtypes = file_info['dtypes']
        nulls = file_info['nulls']
        stats = file_info['stats']
        uniques = file_info['uniques']
        context = get_query_context(user_message)
        if context=='non-data':
            print('c')
            bot_response = re.sub(r'\*\*(.*?)\*\*', r'<h3>\1</h3>', bot_response )
            bot_response = bot_response.replace('*', '')
        
        retries = 0
        error_message = None

        while retries < max_retries:
            bot_response = parse_query_gpt(file_path, context, prompt, columns, dtypes, nulls, stats, uniques, shape)
            if context=='non-data' or 'general':
                print('c')
                bot_response = re.sub(r'\*\*(.*?)\*\*', r'<h3>\1</h3>', bot_response )
                bot_response = bot_response.replace('*', '')
            if isinstance(bot_response, str) and bot_response.startswith('```python'):
                query_code = bot_response[len('```python'):].strip()
                result = execute_query(small_df, query_code)
                print(result)
                if result['type'] == 'error':
                    error_message = result['content']
                    retries += 1
                    prompt += f" Error: {error_message}"
                    continue
                elif result['type'] == 'image':
                    bot_response = f"<img src='{result['content']}' alt='Generated Plot' />"
                elif result['type'] == 'table':
                    bot_response = result['content']  # This is now the HTML table
                else:
                    bot_response = re.sub(r'\*\*(.*?)\*\*', r'<h3>\1</h3>', result['content'])
                    bot_response = bot_response.replace('*', '')
                break
            else:
                break

        if retries == max_retries:
            bot_response = "Failed to generate a valid query after multiple attempts. Please rewrite your query."

    else:
        prompt = f"User's message: '{user_message}'. Please respond to this message."
        bot_response = generate_text(prompt)
        bot_response = re.sub(r'\*\*(.*?)\*\*', r'<h3>\1</h3>', bot_response)
        bot_response = bot_response.replace('*', '')

    return jsonify({'response': bot_response})


@app.route('/option_action', methods=['POST'])
def option_action():
    action = request.json['action']
    file_path = session.get('file_path')
    
    if not file_path:
        return jsonify({'user_message': action, 'response': "Please upload a CSV file first."})

    df = pd.read_csv(file_path)
    
    user_message = f"Perform action: {action}"
    
    if action == "Generate Ideas":
        prompt = f"""
        Analyze the following dataset:
        Columns: {df.columns.tolist()}
        Data types: {df.dtypes.to_dict()}
        
        Please provide:
        1. Potential relationships between columns
        2. Suggested types of graphs or visualizations based on the data
        3. Any interesting patterns or insights that could be explored
        
        Give a concise response focusing on the most important points.
        """
        response = generate_text(prompt)
        response = re.sub(r'\*\*(.*?)\*\*', r'<h3>\1</h3>', response)
        response = response.replace('*', '')
        
    elif action == "Visualize Trends":
        prompt = f"""
        Based on the following dataset:
        Columns: {df.columns.tolist()}
        Data types: {df.dtypes.to_dict()}
        
        Generate Python code to create either a pie chart, bar graph, or another appropriate chart that best represents a key trend or relationship in the data. Use either matplotlib or plotly for visualization. Assign the plot to a variable named 'fig'.
        """
        response = parse_query_gpt(file_path, 'matplotlib', prompt, df.columns.tolist(), df.dtypes.to_dict(), df.isnull().sum().to_dict(), df.describe().to_dict(), df.nunique().to_dict(), df.shape)
        result = execute_query(df, response)
        if result['type'] == 'image':
            response = f"<img src='{result['content']}' alt='Generated Plot' />"
        else:
            response = "Failed to generate visualization. Please try a different request."
        
    elif action == "Analyze Data":
        response = df.head(10).to_html(classes='dataframe', index=False, max_cols=5)
        
        
    elif action == "Report Insights":
        prompt = f"""
        Analyze the following dataset and provide key insights:
        Columns: {df.columns.tolist()}
        Data types: {df.dtypes.to_dict()}
        Shape: {df.shape}
        Summary statistics: {df.describe().to_dict()}
        
        Please provide:
        1. Overview of the dataset (number of rows, columns, data types)
        2. Any notable patterns or trends
        3. Potential data quality issues (missing values, outliers)
        4. Suggestions for further analysis
        
        Give a concise response focusing on the most important points.
        """
        response = generate_text(prompt)
        response = re.sub(r'\*\*(.*?)\*\*', r'<h3>\1</h3>', response)
        response = response.replace('*', '')
    
    else:
        response = "Invalid action requested."

    return jsonify({'user_message': user_message, 'response': response})



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

    def write_image(self, image_data):
        # Convert base64 image to a temporary file
        image_file = io.BytesIO(base64.b64decode(image_data))
        image = Image.open(image_file)
        temp_image_path = 'temp_image.png'
        image.save(temp_image_path)

        # Add image to PDF
        self.image(temp_image_path, x=None, y=None, w=0, h=60)
        os.remove(temp_image_path)
        self.ln(5)

@app.route('/save_pdf', methods=['POST'])
def save_pdf():
    try:
        chat_messages = request.json['messages']
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []

        styles = getSampleStyleSheet()
        title_style = styles['Heading1']
        heading_style = styles['Heading2']
        normal_style = styles['Normal']
        sender_style = ParagraphStyle(
            'SenderStyle',
            parent=styles['Heading2'],
            textColor=colors.blue,
            spaceAfter=6
        )

        # Add title
        elements.append(Paragraph(f"Chat History - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", title_style))
        elements.append(Spacer(1, 12))

        def is_heading(text):
            # Check if the text is less than 20 characters and ends with a colon
            return len(text) < 20 and text.strip().endswith(':')

        for message in chat_messages:
            sender = message['sender']
            elements.append(Paragraph(f"{sender}:", sender_style))

            if 'text' in message:
                text = message['text']
                # Split the text into paragraphs
                paragraphs = text.split('\n')
                for para in paragraphs:
                    if is_heading(para):
                        elements.append(Paragraph(para, heading_style))
                    else:
                        elements.append(Paragraph(para, normal_style))
                    elements.append(Spacer(1, 3))  # Small space between paragraphs

            if 'image' in message:
                image_data = base64.b64decode(message['image'])
                img = PILImage.open(BytesIO(image_data))
                img_width, img_height = img.size
                aspect = img_height / float(img_width)
                
                # Set a max width for the image in the PDF
                max_width = 500
                width = min(img_width, max_width)
                height = width * aspect

                img = Image(BytesIO(image_data), width=width, height=height)
                elements.append(img)

            if 'table' in message:
                # Parse the HTML table
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(message['table'], 'html.parser')
                table = soup.find('table')
                
                if table:
                    data = []
                    for row in table.find_all('tr'):
                        cols = row.find_all(['th', 'td'])
                        data.append([col.text.strip() for col in cols])
                    
                    if data:
                        t = Table(data)
                        t.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 12),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                            ('FONTSIZE', (0, 1), (-1, -1), 12),
                            ('TOPPADDING', (0, 1), (-1, -1), 6),
                            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))
                        elements.append(t)

            elements.append(Spacer(1, 12))  # Add space after each message

        doc.build(elements)
        buffer.seek(0)
        
        return send_file(
            buffer,
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
