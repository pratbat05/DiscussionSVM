import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import emoji

class ImprovedTextPreprocessor:
    def __init__(self, max_features=10000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
            sublinear_tf=True    # Apply sublinear tf scaling
        )
        self.scaler = StandardScaler()
        
    def clean_text(self, text):
        """Advanced text cleaning for GitHub comments"""
        if pd.isna(text) or text == '':
            return ''
            
        # Convert to string
        text = str(text)
        
        # Convert emojis to text
        text = emoji.demojize(text, delimiters=(' ', ' '))
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove code blocks but preserve inline code
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]*`', 'CODE', text)  # Replace inline code with CODE token
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Handle special GitHub patterns
        text = re.sub(r'#\d+', 'ISSUE_NUMBER', text)  # Replace issue numbers
        text = re.sub(r'@[\w-]+', 'MENTION', text)    # Replace mentions
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s!?.,;:()\[\]{}"\'-]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_features(self, text):
        """Extract additional features from text"""
        features = {}
        
        # Length features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        
        # Sentiment indicators
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Code-related features
        features['code_blocks'] = text.count('CODE')
        features['has_issue_number'] = 1 if 'ISSUE_NUMBER' in text else 0
        features['has_mention'] = 1 if 'MENTION' in text else 0
        
        # Emoji features
        features['emoji_count'] = len([c for c in text if c in emoji.EMOJI_DATA])
        
        return features
    
    def prepare_data(self, df):
        """Prepare data for training with enhanced features"""
        print("Cleaning text data...")
        df['cleaned_text'] = df['body'].apply(self.clean_text)
        
        # Remove empty comments
        df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
        
        # Extract additional features
        print("Extracting features...")
        feature_data = df['cleaned_text'].apply(self.extract_features)
        feature_df = pd.DataFrame(feature_data.tolist())
        
        # Combine text and features
        df = pd.concat([df, feature_df], axis=1)
        
        return df
    
    def vectorize_text(self, texts, is_training=True):
        """Convert text to TF-IDF vectors"""
        if is_training:
            return self.vectorizer.fit_transform(texts)
        return self.vectorizer.transform(texts)
    
    def scale_features(self, features, is_training=True):
        """Scale numerical features"""
        if is_training:
            return self.scaler.fit_transform(features)
        return self.scaler.transform(features)

def create_sentiment_labels(df):
    """
    Create sentiment labels based on content analysis
    This is a more sophisticated approach than the simple keyword matching
    """
    def analyze_sentiment(text):
        # Enhanced positive indicators
        positive_patterns = [
            r'\b(thanks|thank you|thx)\b',
            r'\b(great|awesome|excellent|amazing|fantastic|brilliant)\b',
            r'\b(works|working|fixed|resolved|solved)\b',
            r'\b(good|nice|perfect|love|wonderful)\b',
            r'\b(helpful|useful|beneficial)\b',
            r'\b(improved|better|faster|cleaner)\b',
            r'\+1', r':\+1:', r'👍', r'❤️', r'🎉'
        ]
        
        # Enhanced negative indicators
        negative_patterns = [
            r'\b(bug|error|issue|problem|fail|crash)\b',
            r'\b(broken|wrong|bad|terrible|horrible|awful)\b',
            r'\b(doesn\'t work|not working|broken)\b',
            r'\b(slow|slowly|performance issue)\b',
            r'\b(confusing|unclear|difficult)\b',
            r'\b(waste|useless|pointless)\b',
            r'\-1', r':\-1:', r'👎', r'😞', r'😡'
        ]
        
        # Neutral indicators
        neutral_patterns = [
            r'\b(update|updated|change|changed)\b',
            r'\b(add|added|remove|removed)\b',
            r'\b(test|testing|tested)\b',
            r'\b(doc|documentation|comment)\b',
            r'\b(refactor|refactored|cleanup)\b'
        ]
        
        text_lower = text.lower()
        
        # Count matches
        pos_count = sum(len(re.findall(pattern, text_lower)) for pattern in positive_patterns)
        neg_count = sum(len(re.findall(pattern, text_lower)) for pattern in negative_patterns)
        neu_count = sum(len(re.findall(pattern, text_lower)) for pattern in neutral_patterns)
        
        # Decision logic
        if pos_count > neg_count and pos_count > 0:
            return 2  # Positive
        elif neg_count > pos_count and neg_count > 0:
            return 0  # Negative
        elif neu_count > 0 or (pos_count == 0 and neg_count == 0):
            return 1  # Neutral
        else:
            return 1  # Default to neutral
    
    df['sentiment'] = df['cleaned_text'].apply(analyze_sentiment)
    return df

def main():
    # Load the raw data
    df = pd.read_csv('data/raw_comments.csv')
    
    # Initialize improved preprocessor
    preprocessor = ImprovedTextPreprocessor()
    
    # Prepare the data
    df = preprocessor.prepare_data(df)
    
    # Create sentiment labels
    df = create_sentiment_labels(df)
    
    # Save processed data
    df.to_csv('data/improved_processed_comments.csv', index=False)
    print("Improved preprocessing completed and data saved.")
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"Total comments: {len(df)}")
    print(f"Positive: {len(df[df['sentiment'] == 2])}")
    print(f"Neutral: {len(df[df['sentiment'] == 1])}")
    print(f"Negative: {len(df[df['sentiment'] == 0])}")

if __name__ == "__main__":
    main() 