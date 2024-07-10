# PlotPal - Visualization Chatbot

PlotPal is a Flask-based web application designed to help users visualize data trends, generate ideas, analyze data, and report insights through an interactive chatbot interface. The application integrates features such as user authentication, file upload and analysis, data visualisation, chatbot interaction using Google Generative AI (Gemini API), PDF generation, and email functionalities.

## Features

- **User Authentication**:
  - Sign up, login, and logout functionalities
  - Session management for seamless user experience

- **Data Visualization**:
  - Upload CSV files for analysis
  - Data preprocessing and cleaning
  - Display of key information and statistical summaries

- **Chatbot Interaction**:
  - Interact with a chatbot to generate ideas and visualize trends
  - Analyze data and report insights
  - Uses Google Generative AI (Gemini API) for generating responses

- **PDF Generation and Emailing**:
  - Save chat history as a PDF
  - Email chat history to the user

## Technology Stack

- **Backend**: Flask, SQLAlchemy
- **Database**: SQLite
- **Frontend**: HTML, CSS (Bootstrap), JavaScript
- **APIs and Libraries**: Google Generative AI (Gemini API), FPDF, Flask-Mail

## Installation

### Prerequisites

- Python 3.7+
- Virtual Environment (optional but recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/MK61395/Visualisation-Chatbot.git
   cd Visualisation-Chatbot
   
2. Setup a viryual environment (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install dependencies
   `pip install -r requirements.txt`

5. Set up database
   `flask db upgrade`

6. Set up environment variables such as your gemini api-key

7. Run the application
   `python app.py'

8. Access the application at `http://127.0.0.1:5000/.`


