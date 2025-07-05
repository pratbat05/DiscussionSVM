import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class SentimentAccuracyTester:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
    def create_test_dataset(self):
        """Create a comprehensive test dataset with known sentiments"""
        test_data = [
            # Positive comments
            ("This is amazing! Great work on the implementation.", 2),
            ("Thanks for the quick fix, it works perfectly now!", 2),
            ("Excellent documentation, very helpful.", 2),
            ("Love this feature, exactly what I needed.", 2),
            ("Outstanding performance, much faster than before.", 2),
            ("Fantastic job on the UI improvements.", 2),
            ("This solution is brilliant, thank you!", 2),
            ("Awesome work, the code is clean and efficient.", 2),
            ("Perfect! This resolves all my issues.", 2),
            ("Incredible improvement, much better now.", 2),
            
            # Negative comments
            ("This is broken and crashes constantly.", 0),
            ("Terrible performance, unusable.", 0),
            ("Buggy code, needs immediate fixing.", 0),
            ("This is a disaster, nothing works.", 0),
            ("Horrible documentation, impossible to understand.", 0),
            ("The worst implementation I've ever seen.", 0),
            ("This is completely useless.", 0),
            ("Broken functionality, waste of time.", 0),
            ("Terrible user experience.", 0),
            ("This is a nightmare to work with.", 0),
            
            # Neutral comments
            ("I updated the configuration file.", 1),
            ("The code has been refactored.", 1),
            ("Documentation has been updated.", 1),
            ("Added new test cases.", 1),
            ("Modified the build script.", 1),
            ("Updated dependencies to latest versions.", 1),
            ("Changed the default settings.", 1),
            ("Added comments to the code.", 1),
            ("Renamed the function for clarity.", 1),
            ("Moved the file to a different directory.", 1),
            
            # Mixed sentiment comments (more realistic)
            ("The feature works but the documentation is poor.", 1),
            ("Good performance but needs better error handling.", 1),
            ("Nice UI but too many bugs.", 1),
            ("Fast execution but confusing API.", 1),
            ("Clean code but missing features.", 1),
            ("Easy to use but crashes sometimes.", 1),
            ("Well documented but slow performance.", 1),
            ("Good design but hard to configure.", 1),
            ("Reliable but outdated interface.", 1),
            ("Functional but needs optimization.", 1)
        ]
        
        df = pd.DataFrame(test_data, columns=['text', 'sentiment'])
        return df
    
    def clean_text(self, text):
        """Clean text for processing"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def prepare_data(self, df):
        """Prepare data for training and testing"""
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        return df
    
    def test_models(self):
        """Test both models and compare their accuracy"""
        # Create test dataset
        df = self.create_test_dataset()
        df = self.prepare_data(df)
        
        # Vectorize the text
        X = self.vectorizer.fit_transform(df['cleaned_text'])
        y = df['sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Initialize models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM': LinearSVC(max_iter=1000, random_state=42)
        }
        
        results = {}
        
        print("Testing Model Accuracy\n" + "="*50)
        
        for name, model in models.items():
            print(f"\n{name}:")
            print("-" * 30)
            
            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X, y, cv=5)
            
            results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'true_labels': y_test
            }
            
            print(f"Test Accuracy: {accuracy:.3f}")
            print(f"Cross-validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))
            
            # Confusion matrix
            print("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
        
        # Compare models
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        for name, result in results.items():
            print(f"{name}:")
            print(f"  Test Accuracy: {result['accuracy']:.3f}")
            print(f"  CV Accuracy: {result['cv_mean']:.3f} (+/- {result['cv_std'] * 2:.3f})")
        
        # Find best model
        best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
        print(f"\nBest performing model: {best_model}")
        
        return results

def main():
    tester = SentimentAccuracyTester()
    results = tester.test_models()
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("The models have been tested on a balanced dataset with:")
    print("- 10 positive comments")
    print("- 10 negative comments") 
    print("- 10 neutral comments")
    print("- 10 mixed sentiment comments")
    print("\nThis gives us a realistic view of how well the models perform")
    print("on different types of developer comments.")

if __name__ == "__main__":
    main() 