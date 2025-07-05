import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import joblib
import os
import re

class EnhancedSentimentTrainer:
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.best_model = None
        self.best_accuracy = 0
        
    def load_data(self, filepath='data/enhanced_comments.csv'):
        """Load the enhanced dataset"""
        if not os.path.exists(filepath):
            print(f"Data file not found: {filepath}")
            print("Please run the enhanced data collector first.")
            return None
            
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} comments from {df['repo'].nunique()} repositories")
        print(f"Sentiment distribution:")
        print(df['sentiment'].value_counts().sort_index())
        return df
    
    def clean_text(self, text):
        """Enhanced text cleaning"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove GitHub-specific patterns
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'#\w+', '', text)  # Remove hashtags
        text = re.sub(r'`[^`]*`', '', text)  # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)  # Remove multi-line code blocks
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def prepare_data(self, df):
        """Prepare data for training"""
        print("Preparing data...")
        
        # Clean text
        df['cleaned_text'] = df['body'].apply(self.clean_text)
        
        # Remove empty or very short texts
        df = df[df['cleaned_text'].str.len() > 5]
        
        # Balance the dataset if needed
        sentiment_counts = df['sentiment'].value_counts()
        min_count = sentiment_counts.min()
        
        balanced_df = pd.DataFrame()
        for sentiment in [0, 1, 2]:
            sentiment_data = df[df['sentiment'] == sentiment]
            if len(sentiment_data) > min_count:
                sentiment_data = sentiment_data.sample(n=min_count, random_state=42)
            balanced_df = pd.concat([balanced_df, sentiment_data])
        
        print(f"Balanced dataset: {len(balanced_df)} samples")
        print("Final sentiment distribution:")
        print(balanced_df['sentiment'].value_counts().sort_index())
        
        return balanced_df
    
    def create_vectorizer(self):
        """Create an optimized TF-IDF vectorizer"""
        return TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),  # Include bigrams
            sublinear_tf=True  # Apply sublinear tf scaling
        )
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and compare performance"""
        print("Training models...")
        
        # Define models to test
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                C=1.0
            ),
            'SVM': LinearSVC(
                max_iter=1000, 
                random_state=42,
                C=1.0
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            ),
            'Naive Bayes': MultinomialNB(
                alpha=1.0
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('vectorizer', self.create_vectorizer()),
                ('classifier', model)
            ])
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
            
            results[name] = {
                'pipeline': pipeline,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'true_labels': y_test
            }
            
            print(f"Test Accuracy: {accuracy:.3f}")
            print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Store the best model
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = name
                self.models[name] = pipeline
        
        return results
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning on the best model"""
        print(f"\nPerforming hyperparameter tuning for {self.best_model}...")
        
        if self.best_model == 'Logistic Regression':
            param_grid = {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__penalty': ['l1', 'l2'],
                'vectorizer__max_features': [5000, 10000]
            }
        elif self.best_model == 'SVM':
            param_grid = {
                'classifier__C': [0.1, 1.0, 10.0],
                'vectorizer__max_features': [5000, 10000]
            }
        else:
            print("Hyperparameter tuning not implemented for this model type")
            return
        
        # Create base pipeline
        if self.best_model == 'Logistic Regression':
            base_pipeline = Pipeline([
                ('vectorizer', self.create_vectorizer()),
                ('classifier', LogisticRegression(max_iter=1000, random_state=42))
            ])
        elif self.best_model == 'SVM':
            base_pipeline = Pipeline([
                ('vectorizer', self.create_vectorizer()),
                ('classifier', LinearSVC(max_iter=1000, random_state=42))
            ])
        
        # Grid search
        grid_search = GridSearchCV(
            base_pipeline, 
            param_grid, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.3f}")
        
        # Update the best model
        self.models[self.best_model] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def evaluate_models(self, results):
        """Evaluate and compare all models"""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        for name, result in results.items():
            print(f"\n{name}:")
            print("-" * 40)
            print(f"Test Accuracy: {result['accuracy']:.3f}")
            print(f"CV Accuracy: {result['cv_mean']:.3f} (+/- {result['cv_std'] * 2:.3f})")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(
                result['true_labels'], 
                result['predictions'], 
                target_names=['Negative', 'Neutral', 'Positive']
            ))
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        print(f"\nBest performing model: {best_model_name}")
        print(f"Best accuracy: {results[best_model_name]['accuracy']:.3f}")
    
    def save_models(self):
        """Save the trained models"""
        os.makedirs('models', exist_ok=True)
        
        for name, model in self.models.items():
            filename = f"models/{name.lower().replace(' ', '_')}.pkl"
            joblib.dump(model, filename)
            print(f"Saved {name} to {filename}")
    
    def run_training(self):
        """Main training pipeline"""
        # Load data
        df = self.load_data()
        if df is None:
            return
        
        # Prepare data
        df = self.prepare_data(df)
        
        # Split data
        X = df['cleaned_text']
        y = df['sentiment']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train models
        results = self.train_models(X_train, X_test, y_train, y_test)
        
        # Evaluate models
        self.evaluate_models(results)
        
        # Hyperparameter tuning
        self.hyperparameter_tuning(X_train, y_train)
        
        # Save models
        self.save_models()
        
        print("\nTraining completed successfully!")
        return results

def main():
    trainer = EnhancedSentimentTrainer()
    results = trainer.run_training()

if __name__ == "__main__":
    main() 