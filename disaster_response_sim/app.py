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
import ast

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Encryption key for AES (16 bytes)
key = get_random_bytes(16)

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Load the dataset with coordinates from GeoNames-enhanced CSV
df = pd.read_csv("train_with_coords.csv")  # Generated from add_coordinates.py using GeoNames

def encrypt_text(text):
    """Encrypts text using AES-CBC with a random IV."""
    cipher = AES.new(key, AES.MODE_CBC)
    data = text.encode('utf-8')
    padded_data = pad(data, AES.block_size)
    ciphertext = cipher.encrypt(padded_data)
    return base64.b64encode(cipher.iv + ciphertext).decode('utf-8')

def decrypt_text(encrypted_text):
    """Decrypts text encrypted with AES-CBC."""
    raw = base64.b64decode(encrypted_text)
    iv = raw[:16]
    ciphertext = raw[16:]
    decipher = AES.new(key, AES.MODE_CBC, iv=iv)
    decrypted_padded = decipher.decrypt(ciphertext)
    return unpad(decrypted_padded, AES.block_size).decode('utf-8')

def extract_location(text):
    """Extracts a simple location (city/country) from text using regex."""
    pattern = r"(in|near|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
    match = re.search(pattern, text)
    return match.group(2) if match else "Unknown"

def analyze_tweets(limit=None):
    """Analyzes tweets for need, urgency, sentiment, and location."""
    start_time = time.time()

    # Filter disaster-related tweets (target == 1) and sample if limit is provided
    data = df[df["target"] == 1]
    if limit:
        data = data.sample(n=min(limit, len(data)))

    # Define urgency keywords and their scores
    urgent_keywords = {
        "trapped": 10, "flood": 8, "water": 7, "fire": 6, "food": 5,
        "help": 4, "urgent": 3, "rescue": 3, "need help": 4, "send help": 5
    }

    results = []
    for _, row in data.iterrows():
        post = row["text"]
        encrypted_post = encrypt_text(post)
        decrypted_post = decrypt_text(encrypted_post)

        # Tokenize and analyze urgency
        tokens = word_tokenize(decrypted_post.lower())
        urgency_score = 0
        detected_need = "None"
        for token in tokens:
            if token in urgent_keywords:
                urgency_score = urgent_keywords[token]
                detected_need = token
                break

        # Check multi-word phrases
        post_lower = decrypted_post.lower()
        for phrase in urgent_keywords:
            if len(phrase.split()) > 1 and phrase in post_lower:
                urgency_score = urgent_keywords[phrase]
                detected_need = phrase
                break

        # Sentiment analysis and urgency adjustment
        sentiment = sid.polarity_scores(decrypted_post)
        sentiment_score = sentiment['compound']
        adjusted_urgency = urgency_score + (5 * max(0, -sentiment_score))

        # Use location from CSV if available, else extract from text
        location = row["location"] if pd.notna(row["location"]) else extract_location(decrypted_post)
        coordinates = ast.literal_eval(row["coordinates"])  # Convert string "[lat, lng]" to list

        # Append result with coordinates
        results.append({
            "post": decrypted_post,
            "need": detected_need,
            "urgency": round(adjusted_urgency, 2),
            "sentiment": round(sentiment_score, 2),
            "location": location,
            "coordinates": coordinates,
            "encrypted": encrypted_post
        })

    # Sort by urgency descending
    results.sort(key=lambda x: x["urgency"], reverse=True)

    # Save to JSON file
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

    end_time = time.time()
    print(f"Processed {len(results)} tweets in {end_time - start_time:.2f} seconds")
    return results

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint to confirm server is running."""
    return jsonify({"status": "Server is running", "message": "Welcome to Disaster Response Simulator"})

@app.route('/analyze', methods=['GET'])
def get_analysis():
    """API endpoint to analyze tweets and return results with coordinates."""
    limit = request.args.get('limit', default=None, type=int)
    results = analyze_tweets(limit)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)