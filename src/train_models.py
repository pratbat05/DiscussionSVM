import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from preprocessor import TextPreprocessor
import os

class SentimentModelTrainer:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
            'svm': LinearSVC(max_iter=1000, class_weight='balanced')
        }
        
    def prepare_data(self):
        """Load and prepare the data for training"""
        df = pd.read_csv('data/processed_comments.csv')
        
        # Vectorize the text
        X = self.preprocessor.vectorize_text(df['cleaned_text'])
        y = df['sentiment']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_and_evaluate(self):
        """Train and evaluate both models"""
        X_train, X_test, y_train, y_test = self.prepare_data()
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            report = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'report': report,
                'confusion_matrix': conf_matrix
            }
            
            print(f"\nResults for {name}:")
            print("\nClassification Report:")
            print(report)
            print("\nConfusion Matrix:")
            print(conf_matrix)
            
            # Save the model
            self.save_model(name, model)
        
        return results
    
    def save_model(self, name, model):
        """Save the trained model"""
        if not os.path.exists('models'):
            os.makedirs('models')
        
        model_path = f'models/{name}.joblib'
        joblib.dump(model, model_path)
        print(f"\nModel saved to {model_path}")
        
        # Also save the vectorizer
        vectorizer_path = 'models/vectorizer.joblib'
        if not os.path.exists(vectorizer_path):
            joblib.dump(self.preprocessor.vectorizer, vectorizer_path)
            print(f"Vectorizer saved to {vectorizer_path}")

def main():
    trainer = SentimentModelTrainer()
    results = trainer.train_and_evaluate()
    
    # Print final comparison
    print("\nModel Comparison Summary:")
    for name, result in results.items():
        print(f"\n{name.upper()} PERFORMANCE:")
        print(result['report'])

if __name__ == "__main__":
    main() 