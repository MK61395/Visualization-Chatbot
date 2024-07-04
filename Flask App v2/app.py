from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    user_name = "Current User"  
    return render_template('index.html', user_name=user_name)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    bot_response = f"You said: {user_message}"
    return {'response': bot_response}

if __name__ == '__main__':
    app.run(debug=True)
