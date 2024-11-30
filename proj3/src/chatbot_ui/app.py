# from flask import Flask, render_template, request, jsonify

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/get_response', methods=['POST'])
# def get_response():
#     user_message = request.json.get('message', '')
#     selected_categories = request.json.get('categories', [])
#     categories_text = ', '.join(selected_categories)
#     response = f"{user_message} [Selected Categories: {categories_text}]"
#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Replace <ip>:<port> with the actual IP and port of the external API
API_URL = "http://<ip>:<port>"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.json.get('message', '')
    selected_categories = request.json.get('categories', [])
    
    # Format the data for the external API
    payload = {
        "message": user_message,
        "topics": selected_categories
    }

    try:
        # Make the API call to the external service
        response = requests.post(API_URL, json=payload, timeout=5)
        response_data = response.json()
        bot_response = response_data.get('response', 'No response received from API.')
    except requests.exceptions.RequestException as e:
        # Handle errors in the API call
        bot_response = f"Error contacting the external API: {str(e)}"

    # return jsonify({'response': bot_response})
    return jsonify({'response': user_message})

if __name__ == '__main__':
    app.run(debug=True)
