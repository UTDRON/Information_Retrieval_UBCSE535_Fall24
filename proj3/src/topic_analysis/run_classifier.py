import torch
from transformers import BertTokenizer, BertForSequenceClassification
from flask import Flask, request, jsonify

MODEL_PATH = "bert_topic_classifier"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

unique_topics = [
    "politics",
    "travel",
    "environment",
    "technology",
    "education",
    "entertainment",
    "sports",
    "food",
    "health",
    "economy"
]

def predict_single_topic(query):
    inputs = tokenizer.encode_plus(
        query,
        add_special_tokens=True,
        max_length=128,
        truncation=True,
        return_tensors="pt",
        padding="max_length",
        return_attention_mask=True
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()

    topic = unique_topics[predicted_label]

    return topic

def predict_multiple_topics(query, threshold=0.6):
    inputs = tokenizer.encode_plus(
        query,
        add_special_tokens=True,
        max_length=128,
        truncation=True,
        return_tensors="pt",
        padding="max_length",
        return_attention_mask=True
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

    topics_with_probabilities = [
        (unique_topics[id], probability)
        for id, probability in enumerate(probabilities)
        if probability >= threshold
    ]

    topics_with_probabilities = sorted(topics_with_probabilities, key=lambda x: x[1], reverse=True)
    sorted_topics = [topic for topic, _ in topics_with_probabilities]

    return sorted_topics

# Flask setup
app = Flask(__name__)

@app.route("/classify_query", methods=["POST"])
def classify_query():

    request_data = request.get_json()

    if not request_data or "query" not in request_data:
        return jsonify({"error": "Invalid request. JSON must have 'query' string and optional 'multi_topic' Boolean fields"}), 400

    query = request_data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    # multi_topic should be False by default
    multi_topic = request_data.get("multi_topic", False)

    if multi_topic:
        predicted_topics = predict_multiple_topics(query)
        response = {
            "topics": predicted_topics,
        }
    else:
        predicted_topic = predict_single_topic(query)
        response = {
            "topic": predicted_topic,
        }

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9999)