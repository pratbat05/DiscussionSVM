import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def create_synthetic_github_data():
    """Create a large synthetic dataset that mimics real GitHub comments"""
    
    # Positive comment templates
    positive_templates = [
        "Thanks for the quick fix! This resolves the issue perfectly.",
        "Great work on the implementation! The code is clean and efficient.",
        "Awesome! This feature works exactly as expected.",
        "Excellent documentation, very helpful for understanding the API.",
        "Perfect! The performance improvement is significant.",
        "Love this solution! Much better than the previous approach.",
        "Fantastic job! The UI looks great and is very intuitive.",
        "Brilliant implementation! This makes development much easier.",
        "Outstanding work! The code is well-structured and maintainable.",
        "Incredible improvement! The speed boost is noticeable.",
        "Thanks for addressing this! The fix works perfectly.",
        "Great documentation! Very clear and comprehensive.",
        "Excellent performance! Much faster than before.",
        "Perfect solution! This resolves all my concerns.",
        "Amazing work! The feature is exactly what I needed.",
        "Wonderful implementation! Clean and efficient code.",
        "Superb job! The API is intuitive and well-designed.",
        "Outstanding contribution! This improves the project significantly.",
        "Fantastic fix! The issue is completely resolved.",
        "Excellent work! The code quality is top-notch."
    ]
    
    # Negative comment templates
    negative_templates = [
        "This is broken and crashes constantly. Needs immediate fixing.",
        "Terrible performance, the app is unusable with this bug.",
        "Buggy code that causes data loss. This is unacceptable.",
        "The documentation is horrible and impossible to understand.",
        "This implementation is a disaster. Nothing works as expected.",
        "Terrible user experience. The interface is confusing.",
        "Broken functionality that wastes hours of development time.",
        "The worst code I've ever seen. Needs complete rewrite.",
        "This is completely useless and doesn't solve the problem.",
        "Horrible performance issues. The app is too slow to use.",
        "Buggy and unreliable. Can't trust this implementation.",
        "Terrible error handling. Crashes on every edge case.",
        "The API is confusing and poorly designed.",
        "This feature is broken and doesn't work at all.",
        "Terrible documentation. No examples or clear explanations.",
        "The code is a mess and impossible to maintain.",
        "This is a nightmare to debug. Too many issues.",
        "Broken build process. Can't even compile the project.",
        "Terrible performance. Takes forever to load.",
        "This is completely broken. Don't use this version."
    ]
    
    # Neutral comment templates
    neutral_templates = [
        "I updated the configuration file as requested.",
        "The code has been refactored for better organization.",
        "Documentation has been updated with new examples.",
        "Added new test cases to improve coverage.",
        "Modified the build script to fix the deployment issue.",
        "Updated dependencies to their latest versions.",
        "Changed the default settings to match requirements.",
        "Added comments to the code for better understanding.",
        "Renamed the function for better clarity.",
        "Moved the file to a different directory structure.",
        "Updated the README with installation instructions.",
        "Added error handling for edge cases.",
        "Modified the API endpoint to accept new parameters.",
        "Updated the database schema for the new feature.",
        "Added logging statements for debugging purposes.",
        "Changed the file format to support new requirements.",
        "Updated the configuration to use environment variables.",
        "Added validation for user input data.",
        "Modified the algorithm to improve efficiency.",
        "Updated the documentation to reflect recent changes."
    ]
    
    # Mixed sentiment comments (more realistic)
    mixed_templates = [
        "The feature works but the documentation is poor and confusing.",
        "Good performance but needs better error handling for edge cases.",
        "Nice UI design but too many bugs in the implementation.",
        "Fast execution but the API is confusing and hard to use.",
        "Clean code structure but missing important features.",
        "Easy to use interface but crashes sometimes unexpectedly.",
        "Well documented but the performance is too slow for production.",
        "Good design principles but hard to configure properly.",
        "Reliable functionality but the interface is outdated.",
        "Functional code but needs optimization for better performance.",
        "The core feature works well but the error messages are unclear.",
        "Good implementation but the testing coverage is insufficient.",
        "Efficient algorithm but the code is difficult to read.",
        "Useful functionality but the setup process is complicated.",
        "Solid foundation but the documentation lacks examples.",
        "Working solution but the performance could be improved.",
        "Good approach but the error handling needs work.",
        "Effective implementation but the API design is confusing.",
        "Reliable system but the user interface needs improvement.",
        "Functional code but the deployment process is complex."
    ]
    
    # Generate synthetic data
    data = []
    
    # Generate positive comments
    for i in range(100):
        template = random.choice(positive_templates)
        # Add some variation
        if random.random() < 0.3:
            template += " 👍"
        if random.random() < 0.2:
            template += " Thanks again!"
        
        data.append({
            'repo': f'repo_{random.randint(1, 10)}',
            'comment_id': f'pos_{i}',
            'body': template,
            'created_at': (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
            'user': f'user_{random.randint(1, 50)}',
            'sentiment': 2
        })
    
    # Generate negative comments
    for i in range(100):
        template = random.choice(negative_templates)
        # Add some variation
        if random.random() < 0.3:
            template += " 👎"
        if random.random() < 0.2:
            template += " Please fix this ASAP!"
        
        data.append({
            'repo': f'repo_{random.randint(1, 10)}',
            'comment_id': f'neg_{i}',
            'body': template,
            'created_at': (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
            'user': f'user_{random.randint(1, 50)}',
            'sentiment': 0
        })
    
    # Generate neutral comments
    for i in range(100):
        template = random.choice(neutral_templates)
        data.append({
            'repo': f'repo_{random.randint(1, 10)}',
            'comment_id': f'neu_{i}',
            'body': template,
            'created_at': (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
            'user': f'user_{random.randint(1, 50)}',
            'sentiment': 1
        })
    
    # Generate mixed sentiment comments
    for i in range(100):
        template = random.choice(mixed_templates)
        data.append({
            'repo': f'repo_{random.randint(1, 10)}',
            'comment_id': f'mix_{i}',
            'body': template,
            'created_at': (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
            'user': f'user_{random.randint(1, 50)}',
            'sentiment': 1  # Mixed comments are typically neutral
        })
    
    # Shuffle the data
    random.shuffle(data)
    
    return pd.DataFrame(data)

def main():
    print("Creating synthetic GitHub comment dataset...")
    df = create_synthetic_github_data()
    
    # Save to data directory
    import os
    if not os.path.exists('data'):
        os.makedirs('data')
    
    df.to_csv('data/raw_comments.csv', index=False)
    print(f"Created dataset with {len(df)} comments")
    print(f"Positive: {len(df[df['sentiment'] == 2])}")
    print(f"Neutral: {len(df[df['sentiment'] == 1])}")
    print(f"Negative: {len(df[df['sentiment'] == 0])}")

if __name__ == "__main__":
    main() 