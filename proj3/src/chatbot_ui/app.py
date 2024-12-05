from flask import Flask, render_template, request, jsonify
import requests
from faiss_wikibot import get_top_k
import time
import psycopg2

app = Flask(__name__)

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

def get_chitchat(query):
    
    chitchat_url = "http://34.23.122.204:5000/chat"
    payload = {
        "input": query
    }
    try:
        response = requests.post(chitchat_url, json = payload)
        response_data = response.json()
    except requests.exceptions.RequestException as e:
        response_data = f"Error contacting the classifier model: {str(e)}"

    if response.status_code == 200:
        print("Response:", response_data)
    else:
        response_data = f"Failed with status code {response.status_code}: {response.text}"
    
    return response_data

@app.route('/get_chart_data', methods=['GET'])
def get_chart_data():
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="pratiktt",
        host="34.169.90.241",
        port="5432"
    )
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT msg_type, response_time FROM chatlogs")
        rows = cursor.fetchall()
        conn.commit()

        category_counts = {}
        response_time_sums = {}
        response_time_counts = {}

        for row in rows:
            category = row[0].strip().lower()
            response_time = float(row[1])
            
            # Count occurrences
            if category not in category_counts:
                category_counts[category] = 0
                response_time_sums[category] = 0
                response_time_counts[category] = 0
            
            category_counts[category] += 1
            response_time_sums[category] += response_time
            response_time_counts[category] += 1

        average_response_times = {
            category: response_time_sums[category] / response_time_counts[category]
            for category in response_time_sums
        }

        data = {
            "categories": list(category_counts.keys()),
            "counts": list(category_counts.values()),
            "average_response_times": list(average_response_times.values())
        }

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        cursor.close()
        conn.close()



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
    
    # payload = {
    #     "message": user_message,
    #     "topics": selected_categories
    # }

    chitchat_response = get_chitchat(user_message)
    if "hello" in user_message or "hi" in user_message or "bye" in user_message or "good" in user_message or "plan" in user_message or "you" in user_message:
        try:
            bot_message = chitchat_response["response"]
            message_type = chitchat_response["type"]
            if message_type == "chitchat":
                end_time = time.time()
                execution_time = end_time - start_time
                writeToDatabase(['chitchat'], execution_time)
                return jsonify({'response': bot_message, 'topics': ['chitchat']})
        except:
            if "Failed" in chitchat_response or "Error" in chitchat_response:
                failure_response = "sorry, i don't know about "+ user_message
                return jsonify({'response': failure_response, 'topics': selected_categories})


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
    return jsonify({'response': wiki_response, 'topics': topic})

if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=6000)
