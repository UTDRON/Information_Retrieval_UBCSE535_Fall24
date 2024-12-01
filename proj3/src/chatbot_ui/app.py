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
from faiss_wikibot import get_top_k
import time
import psycopg2
# import pg8000

app = Flask(__name__)

# Replace <ip>:<port> with the actual IP and port of the external API
# API_URL = "http://<ip>:<port>"

@app.route('/')
def index():
    return render_template('index.html')


def get_topics(query):
    topic_url = "http://34.122.76.122:9999/classify_query"
    payload = {
        "query": query,
        "multi_topic": True
    }

    try:
        response = requests.post(topic_url, json=payload)
        response_data = response.json()
    except requests.exceptions.RequestException as e:
        response_data = f"Error contacting the topic model: {str(e)}"

    if response.status_code == 200:
        print("Response:", response_data)
    else:
        response_data = f"Failed with status code {response.status_code}: {response.text}"
    
    return response_data["topics"]


def writeToDatabase(topics, execution_time):
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="pratiktt",
        host="34.169.90.241",
        port="5432"
    )
    cursor = conn.cursor()

    try:
        for topic in topics:
            new_data = [(topic, execution_time)]
            insert_query = "INSERT INTO chatlogs (msg_type, response_time) VALUES (%s, %s)"
            cursor.executemany(insert_query, new_data)
        conn.commit()
    except:
        print("XXXXXX________________________Issue in Writing to db________________________XXXXXX")
        
    cursor.close()
    conn.close()


@app.route('/get_response', methods=['POST'])
def get_response():
    start_time = time.time()
    user_message = request.json.get('message', '')
    selected_categories = request.json.get('categories', [])
    
    payload = {
        "message": user_message,
        "topics": selected_categories
    }

    if not selected_categories:
        topic = get_topics(user_message)
        print("CLASSIFIED/ANALYZED CATEGORIES:", topic, type(topic))
    else:
        topic = selected_categories
        print("SELECTED CATEGORIES: ",topic)

    try:
        wiki_response = get_top_k(user_message)
    except:
        wiki_response = "sorry, i don't know about "+ user_message

    end_time = time.time()
    execution_time = end_time - start_time
    writeToDatabase(topic, execution_time)
    return jsonify({'response': wiki_response})

if __name__ == '__main__':
    app.run(debug=True)
