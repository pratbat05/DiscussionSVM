import joblib
import pandas as pd
from preprocessor import TextPreprocessor

class SentimentPredictor:
    def __init__(self, model_name='logistic_regression'):
        """Initialize the predictor with the specified model"""
        self.model = joblib.load(f'models/{model_name}.joblib')
        self.vectorizer = joblib.load('models/vectorizer.joblib')
        self.preprocessor = TextPreprocessor()
        self.label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    def predict(self, text):
        """Predict sentiment for a single text"""
        # Clean the text
        cleaned_text = self.preprocessor.clean_text(text)
        
        # Vectorize the text
        X = self.vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        return {
            'text': text,
            'sentiment': self.label_map[prediction],
            'sentiment_code': prediction
        }
    
    def predict_batch(self, texts):
        """Predict sentiment for multiple texts"""
        return [self.predict(text) for text in texts]

def main():
    # Example usage
    predictor = SentimentPredictor()  # Default: logistic regression
    
    # Example comments
    example_comments = [
        "This is a great feature, thanks for implementing it!",
        "The code is broken and crashes constantly.",
        "I updated the documentation as requested.",
        "This bug is really annoying and needs to be fixed ASAP.",
        "Thanks for the quick response, works perfectly now!"
    ]
    
    # Make predictions
    print("\nPredicting sentiments for example comments:")
    predictions = predictor.predict_batch(example_comments)
    
    # Print results
    for pred in predictions:
        print(f"\nText: {pred['text']}")
        print(f"Sentiment: {pred['sentiment']}")
    
    # Try SVM model
    print("\nUsing SVM model:")
    svm_predictor = SentimentPredictor('svm')
    svm_predictions = svm_predictor.predict_batch(example_comments)
    
    for pred in svm_predictions:
        print(f"\nText: {pred['text']}")
        print(f"Sentiment: {pred['sentiment']}")

if __name__ == "__main__":
    main() 