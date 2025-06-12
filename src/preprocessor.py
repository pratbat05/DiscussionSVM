import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class TextPreprocessor:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
    def clean_text(self, text):
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]*`', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def prepare_data(self, df):
        """Prepare data for training"""
        # Clean the text
        df['cleaned_text'] = df['body'].apply(self.clean_text)
        
        # Remove empty comments
        df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
        
        return df

    def vectorize_text(self, texts, is_training=True):
        """Convert text to TF-IDF vectors"""
        if is_training:
            return self.vectorizer.fit_transform(texts)
        return self.vectorizer.transform(texts)

def assign_sentiment_labels(text):
    """
    Simple rule-based sentiment labeling
    Returns: 0 (negative), 1 (neutral), 2 (positive)
    """
    # Define simple keyword lists
    positive_words = {'thanks', 'great', 'good', 'awesome', 'excellent', 'nice', 'solved', 'works', 'helpful', 'love'}
    negative_words = {'bug', 'issue', 'error', 'problem', 'fail', 'crash', 'broken', 'wrong', 'bad', 'terrible'}
    
    text_words = set(text.lower().split())
    
    pos_count = len(text_words.intersection(positive_words))
    neg_count = len(text_words.intersection(negative_words))
    
    if pos_count > neg_count:
        return 2  # Positive
    elif neg_count > pos_count:
        return 0  # Negative
    else:
        return 1  # Neutral

def main():
    # Load the raw data
    df = pd.read_csv('data/raw_comments.csv')
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Prepare the data
    df = preprocessor.prepare_data(df)
    
    # Assign sentiment labels
    df['sentiment'] = df['cleaned_text'].apply(assign_sentiment_labels)
    
    # Save processed data
    df.to_csv('data/processed_comments.csv', index=False)
    print("Preprocessing completed and data saved.")

if __name__ == "__main__":
    main() 