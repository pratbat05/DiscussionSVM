import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from improved_preprocessor import ImprovedTextPreprocessor

class ImprovedModelTrainer:
    def __init__(self):
        self.preprocessor = ImprovedTextPreprocessor()
        self.scaler = StandardScaler()
        
    def prepare_data(self):
        """Load and prepare the improved data"""
        df = pd.read_csv('data/improved_processed_comments.csv')
        
        # Vectorize text
        X_text = self.preprocessor.vectorize_text(df['cleaned_text'])
        
        # Prepare numerical features
        numerical_features = ['char_count', 'word_count', 'exclamation_count', 
                            'question_count', 'uppercase_ratio', 'code_blocks', 
                            'has_issue_number', 'has_mention', 'emoji_count']
        
        X_num = df[numerical_features].fillna(0)
        X_num_scaled = self.scaler.fit_transform(X_num)
        
        # Combine text and numerical features
        from scipy.sparse import hstack
        X_combined = hstack([X_text, X_num_scaled])
        
        y = df['sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_models_with_tuning(self):
        """Train models with hyperparameter tuning"""
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        print("Training models with hyperparameter tuning...")
        
        # Define models with parameter grids
        models = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'class_weight': ['balanced', None],
                    'solver': ['liblinear', 'saga']
                }
            },
            'SVM': {
                'model': LinearSVC(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'class_weight': ['balanced', None],
                    'loss': ['hinge', 'squared_hinge']
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'class_weight': ['balanced', 'balanced_subsample']
                }
            }
        }
        
        best_models = {}
        results = {}
        
        for name, model_info in models.items():
            print(f"\nTuning {name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                model_info['model'],
                model_info['params'],
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            best_models[name] = best_model
            
            # Evaluate
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
            
            results[name] = {
                'model': best_model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'best_params': grid_search.best_params_,
                'predictions': y_pred,
                'true_labels': y_test
            }
            
            print(f"{name} - Best Accuracy: {accuracy:.3f}")
            print(f"Best Parameters: {grid_search.best_params_}")
        
        return best_models, results
    
    def create_ensemble(self, models, X_train, y_train, X_test, y_test):
        """Create an ensemble of the best models"""
        print("\nCreating ensemble model...")
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=[(name, model) for name, model in models.items()],
            voting='soft'  # Use probability voting
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        y_pred = ensemble.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5)
        
        results = {
            'model': ensemble,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'true_labels': y_test
        }
        
        print(f"Ensemble Accuracy: {accuracy:.3f}")
        
        return ensemble, results
    
    def save_models(self, models, ensemble):
        """Save all trained models"""
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Save individual models
        for name, model in models.items():
            model_path = f'models/improved_{name.lower().replace(" ", "_")}.joblib'
            joblib.dump(model, model_path)
            print(f"Saved {name} to {model_path}")
        
        # Save ensemble
        ensemble_path = 'models/improved_ensemble.joblib'
        joblib.dump(ensemble, ensemble_path)
        print(f"Saved ensemble to {ensemble_path}")
        
        # Save preprocessor and scaler
        joblib.dump(self.preprocessor, 'models/improved_preprocessor.joblib')
        joblib.dump(self.scaler, 'models/improved_scaler.joblib')
        print("Saved preprocessor and scaler")

def main():
    trainer = ImprovedModelTrainer()
    
    # Train individual models with tuning
    best_models, results = trainer.train_models_with_tuning()
    
    # Get training data for ensemble
    X_train, X_test, y_train, y_test = trainer.prepare_data()
    
    # Create and evaluate ensemble
    ensemble, ensemble_results = trainer.create_ensemble(
        best_models, X_train, y_train, X_test, y_test
    )
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Test Accuracy: {result['accuracy']:.3f}")
        print(f"  CV Accuracy: {result['cv_mean']:.3f} (+/- {result['cv_std'] * 2:.3f})")
    
    print(f"\nEnsemble:")
    print(f"  Test Accuracy: {ensemble_results['accuracy']:.3f}")
    print(f"  CV Accuracy: {ensemble_results['cv_mean']:.3f} (+/- {ensemble_results['cv_std'] * 2:.3f})")
    
    # Save models
    trainer.save_models(best_models, ensemble)
    
    # Print best model
    best_individual = max(results.keys(), key=lambda x: results[x]['accuracy'])
    print(f"\nBest Individual Model: {best_individual}")
    print(f"Best Individual Accuracy: {results[best_individual]['accuracy']:.3f}")
    print(f"Ensemble Accuracy: {ensemble_results['accuracy']:.3f}")

if __name__ == "__main__":
    main() 