import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import base64
from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import time
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

key = get_random_bytes(16)
sid = SentimentIntensityAnalyzer()

def encrypt_text(text):
    cipher = AES.new(key, AES.MODE_CBC)
    data = text.encode('utf-8')
    padded_data = pad(data, AES.block_size)
    ciphertext = cipher.encrypt(padded_data)
    return base64.b64encode(cipher.iv + ciphertext).decode('utf-8')

def decrypt_text(encrypted_text):
    raw = base64.b64decode(encrypted_text)
    iv = raw[:16]
    ciphertext = raw[16:]
    decipher = AES.new(key, AES.MODE_CBC, iv=iv)
    decrypted_padded = decipher.decrypt(ciphertext)
    return unpad(decrypted_padded, AES.block_size).decode('utf-8')

def extract_location(text):
    pattern = r"(in|near|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
    match = re.search(pattern, text)
    return match.group(2) if match else "Unknown"

def analyze_tweets(limit=None):
    start_time = time.time()
    data = pd.read_csv("train.csv")
    posts = data[data["target"] == 1]["text"].tolist()
    if limit:
        posts = posts[:limit]

    urgent_keywords = {
        "trapped": 10,
        "flood": 8,
        "water": 7,
        "fire": 6,
        "food": 5,
        "help": 4,
        "urgent": 3,
        "rescue": 3,
        "need help": 4,
        "send help": 5
    }

    results = []
    for post in posts:
        encrypted_post = encrypt_text(post)
        decrypted_post = decrypt_text(encrypted_post)

        tokens = word_tokenize(decrypted_post.lower())
        urgency_score = 0
        detected_need = "None"

        for token in tokens:
            if token in urgent_keywords:
                urgency_score = urgent_keywords[token]
                detected_need = token
                break

        post_lower = decrypted_post.lower()
        for phrase in urgent_keywords:
            if len(phrase.split()) > 1 and phrase in post_lower:
                urgency_score = urgent_keywords[phrase]
                detected_need = phrase
                break

        sentiment = sid.polarity_scores(decrypted_post)
        sentiment_score = sentiment['compound']
        adjusted_urgency = urgency_score + (5 * max(0, -sentiment_score))
        location = extract_location(decrypted_post)

        results.append({
            "post": decrypted_post,
            "need": detected_need,
            "urgency": round(adjusted_urgency, 2),
            "sentiment": round(sentiment_score, 2),
            "location": location,
            "encrypted": encrypted_post
        })

    results.sort(key=lambda x: x["urgency"], reverse=True)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

    end_time = time.time()
    print(f"Processed {len(results)} tweets in {end_time - start_time:.2f} seconds")
    return results

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "Server is running", "message": "Welcome to Disaster Response Simulator"})

@app.route('/analyze', methods=['GET'])
def get_analysis():
    limit = request.args.get('limit', default=None, type=int)
    results = analyze_tweets(limit)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)