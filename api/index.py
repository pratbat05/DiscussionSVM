from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='../static')

# Load the trained model
try:
    model = joblib.load('../models/svm.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return send_from_directory('../static', 'index.html')

@app.route('/api')
def api_info():
    return jsonify({
        "message": "Sentiment Analysis API",
        "description": "Analyze sentiment of developer comments",
        "endpoints": {
            "/predict": "POST - Analyze sentiment of text",
            "/health": "GET - Check API health"
        }
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Please provide 'text' in request body"}), 400
        
        text = data['text']
        
        if not text or len(text.strip()) == 0:
            return jsonify({"error": "Text cannot be empty"}), 400
        
        # Make prediction
        prediction = model.predict([text])[0]
        
        # Map prediction to sentiment label
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = sentiment_map.get(prediction, "unknown")
        
        # Get prediction probabilities if available
        try:
            probabilities = model.predict_proba([text])[0]
            confidence = max(probabilities)
        except:
            confidence = None
        
        return jsonify({
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "prediction": int(prediction)
        })
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({"error": "Please provide 'texts' array in request body"}), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({"error": "Texts must be a non-empty array"}), 400
        
        # Make predictions
        predictions = model.predict(texts)
        
        # Map predictions to sentiment labels
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        results = []
        
        for i, text in enumerate(texts):
            sentiment = sentiment_map.get(predictions[i], "unknown")
            results.append({
                "text": text,
                "sentiment": sentiment,
                "prediction": int(predictions[i])
            })
        
        return jsonify({
            "results": results,
            "count": len(results)
        })
        
    except Exception as e:
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 500

# Export the Flask app for Vercel
app.debug = False 