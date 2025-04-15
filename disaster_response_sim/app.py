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
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import math

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Encryption key for AES (16 bytes)
key = get_random_bytes(16)

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Load the dataset with coordinates
df = pd.read_csv("train_with_coords.csv")

# Load pre-trained DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=7)
model.eval()

# Expanded need categories and keywords
need_categories = ['flood', 'fire', 'food', 'trapped', 'earthquake', 'help', 'none']
need_to_idx = {need: idx for idx, need in enumerate(need_categories)}

urgent_keywords = {
    "flood": 8, "flooding": 8, "water": 7, "drown": 7,
    "fire": 6, "burning": 6, "smoke": 5, "wildfire": 6,
    "earthquake": 9, "quake": 9, "tremor": 8,
    "storm": 7, "hurricane": 8, "tornado": 8, "wind": 6,
    "food": 5, "hungry": 5, "starving": 6, "waterless": 5,
    "trapped": 10, "stuck": 9, "buried": 10, "collapse": 9,
    "help": 4, "rescue": 5, "save": 5, "aid": 4,
    "urgent": 3, "now": 3, "immediately": 4, "asap": 3,
    "need help": 4, "send help": 5, "please help": 4, "emergency": 5
}

# Custom Dataset for BERT
class TweetDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], return_tensors="pt", truncation=True, padding='max_length', max_length=128)
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx])
        }

# Auto-label tweets based on keywords
def label_tweets(df):
    labels = []
    for _, row in df.iterrows():
        text = row["text"].lower()
        if row["target"] == 0:
            labels.append(need_to_idx["none"])
        else:
            detected_need = "none"
            for phrase in urgent_keywords:
                if phrase in text:
                    if phrase in need_categories:  # Only assign if it's a need category
                        detected_need = phrase
                        break
            labels.append(need_to_idx[detected_need])
    return labels

# Fine-tune BERT on the full dataset
def fine_tune_model():
    # Label the dataset
    texts = df["text"].tolist()
    labels = label_tweets(df)
    
    # Split into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = TweetDataset(train_texts, train_labels)
    val_dataset = TweetDataset(val_texts, val_labels)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    num_epochs = 3
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for batch in train_loader:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == batch['labels']).sum().item()
                total += batch['labels'].size(0)
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {accuracy:.4f}")
        model.train()
    
    # Save the fine-tuned model
    model.save_pretrained("fine_tuned_bert")
    tokenizer.save_pretrained("fine_tuned_bert")
    print("Model fine-tuned and saved to 'fine_tuned_bert'")

# Load or fine-tune model
try:
    model = DistilBertForSequenceClassification.from_pretrained("fine_tuned_bert")
    tokenizer = DistilBertTokenizer.from_pretrained("fine_tuned_bert")
    print("Loaded fine-tuned model")
except:
    print("No fine-tuned model found, training now...")
    fine_tune_model()
model.eval()

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

def classify_needs_batch(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_idxs = torch.argmax(probs, dim=-1)
    return [need_categories[idx] for idx in predicted_idxs]

def clean_location(location):
    if pd.isna(location) or location == 'NaN' or location == 'None':
        return None
    return str(location)

def clean_coordinates(coords):
    try:
        if isinstance(coords, str):
            coords = ast.literal_eval(coords)
        if not isinstance(coords, list) or len(coords) != 2:
            return None
        if any(not isinstance(x, (int, float)) or math.isnan(x) for x in coords):
            return None
        return [float(coords[0]), float(coords[1])]
    except:
        return None

def analyze_tweets(limit=None):
    start_time = time.time()
    data = df[df["target"] == 1].reset_index(drop=True)
    if limit:
        data = data.sample(n=min(limit, len(data))).reset_index(drop=True)

    posts = data["text"].tolist()
    encrypted_posts = [encrypt_text(post) for post in posts]
    decrypted_posts = [decrypt_text(ep) for ep in encrypted_posts]
    
    # Batch classify with BERT
    batch_size = 16
    needs = []
    for i in range(0, len(decrypted_posts), batch_size):
        batch = decrypted_posts[i:i + batch_size]
        needs.extend(classify_needs_batch(batch))

    results = []
    for idx, row in data.iterrows():
        decrypted_post = decrypted_posts[idx]
        detected_need = needs[idx]
        post_lower = decrypted_post.lower()

        # Fallback to keywords if "none"
        if detected_need == "none":
            tokens = word_tokenize(post_lower)
            urgency_score = 0
            for token in tokens:
                if token in urgent_keywords:
                    urgency_score = urgent_keywords[token]
                    detected_need = token
                    break
            for phrase in urgent_keywords:
                if len(phrase.split()) > 1 and phrase in post_lower:
                    urgency_score = urgent_keywords[phrase]
                    detected_need = phrase
                    break
        else:
            urgency_score = urgent_keywords.get(detected_need, 4)

        # Boost urgency with modifiers
        modifier_boost = 0
        for mod in urgent_keywords:
            if mod in post_lower and mod in ["now", "immediately", "asap", "urgent", "emergency"]:
                modifier_boost += urgent_keywords[mod]
        urgency_score += modifier_boost

        sentiment = sid.polarity_scores(decrypted_post)
        sentiment_score = sentiment['compound']
        adjusted_urgency = urgency_score + (5 * max(0, -sentiment_score))

        location = clean_location(row["location"])
        coordinates = clean_coordinates(row["coordinates"])

        results.append({
            "post": decrypted_post,
            "need": detected_need,
            "urgency": round(adjusted_urgency, 2),
            "sentiment": round(sentiment_score, 2),
            "location": location if location else None,
            "coordinates": coordinates if coordinates else None,
            "encrypted": encrypted_posts[idx]
        })

    results.sort(key=lambda x: x["urgency"], reverse=True)
    
    # Convert to JSON-serializable format
    final_results = []
    for item in results:
        serializable_item = item.copy()
        if serializable_item["coordinates"] is None:
            serializable_item["coordinates"] = None
        final_results.append(serializable_item)

    end_time = time.time()
    print(f"Processed {len(results)} tweets in {end_time - start_time:.2f} seconds")
    return final_results

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