import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os

# Load the test set and predictions from the last training run
# We'll use the enhanced_trainer's balanced dataset and SVM model

def load_balanced_data():
    df = pd.read_csv('data/enhanced_comments.csv')
    # Use the same balancing logic as in the trainer
    sentiment_counts = df['sentiment'].value_counts()
    min_count = sentiment_counts.min()
    balanced_df = pd.DataFrame()
    for sentiment in [0, 1, 2]:
        sentiment_data = df[df['sentiment'] == sentiment]
        if len(sentiment_data) > min_count:
            sentiment_data = sentiment_data.sample(n=min_count, random_state=42)
        balanced_df = pd.concat([balanced_df, sentiment_data])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df

def plot_class_distribution(df, save_path=None):
    plt.figure(figsize=(6,4))
    sns.countplot(x='sentiment', data=df, palette='viridis')
    plt.title('Class Distribution (Balanced)')
    plt.xlabel('Sentiment (0=Neg, 1=Neutral, 2=Pos)')
    plt.ylabel('Count')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Class distribution plot saved to {save_path}")
    plt.show()
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved to {save_path}")
    plt.show()
    plt.close()

def main():
    os.makedirs('plots', exist_ok=True)
    print('Loading balanced dataset...')
    df = load_balanced_data()
    print(f"Loaded {len(df)} samples.")
    plot_class_distribution(df, save_path='plots/class_distribution.png')

    # Split into train/test as in enhanced_trainer
    from sklearn.model_selection import train_test_split
    X = df['body']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Load the best model (SVM)
    print('Loading trained SVM model...')
    svm_model = joblib.load('models/svm.pkl')
    print('Predicting on test set...')
    y_pred = svm_model.predict(X_test)

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, save_path='plots/confusion_matrix.png')

    # Classification report
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

if __name__ == '__main__':
    main() 