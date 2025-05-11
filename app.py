from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Initialize Flask app
app = Flask(__name__)

# Load model and tokenizer (once at startup)
MODEL_PATH = "saved_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Label mapping
label_map = {0: "negative", 1: "neutral", 2: "positive"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()

    return jsonify({
        "text": text,
        "sentiment": label_map[prediction],
        "confidence": round(probs[0][prediction].item(), 3)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
