from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.json.get('message', '')
    selected_categories = request.json.get('categories', [])
    categories_text = ', '.join(selected_categories)
    response = f"{user_message} [Selected Categories: {categories_text}]"
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
