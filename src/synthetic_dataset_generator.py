import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

class SyntheticDatasetGenerator:
    def __init__(self):
        self.positive_templates = [
            "This is {adj}! {action} works perfectly.",
            "Great work on {feature}! {action} is much better now.",
            "Thanks for {action}! This is exactly what I needed.",
            "Excellent {feature}! {action} is working as expected.",
            "Love this {feature}! {action} is fantastic.",
            "Outstanding {action}! The {feature} is brilliant.",
            "This {feature} is amazing! {action} is spot on.",
            "Perfect implementation! {action} is working great.",
            "Awesome work on {feature}! {action} is clean and efficient.",
            "Incredible {action}! The {feature} is outstanding.",
            "This solution is brilliant! {action} is exactly right.",
            "Fantastic job! {action} is working perfectly.",
            "Wonderful {feature}! {action} is superb.",
            "This is excellent! {action} is working like a charm.",
            "Superb work on {feature}! {action} is flawless."
        ]
        
        self.negative_templates = [
            "This is {adj}! {action} is completely broken.",
            "Terrible {feature}! {action} crashes constantly.",
            "This {feature} is awful! {action} doesn't work at all.",
            "Horrible implementation! {action} is buggy.",
            "This is a disaster! {action} is unusable.",
            "The worst {feature} ever! {action} is terrible.",
            "This {action} is broken! {feature} needs fixing.",
            "Useless {feature}! {action} is completely wrong.",
            "This is terrible! {action} is a nightmare.",
            "Awful {feature}! {action} is frustrating.",
            "This {action} is horrible! {feature} is broken.",
            "Terrible work! {action} is completely useless.",
            "This is the worst! {action} is buggy and slow.",
            "Horrible {feature}! {action} is a mess.",
            "This {action} is awful! {feature} is broken."
        ]
        
        self.neutral_templates = [
            "Updated {feature} to {action}.",
            "Modified {feature} for {action}.",
            "Changed {feature} to {action}.",
            "Added {action} to {feature}.",
            "Removed {action} from {feature}.",
            "Refactored {feature} to {action}.",
            "Improved {feature} with {action}.",
            "Enhanced {feature} by {action}.",
            "Optimized {feature} for {action}.",
            "Fixed {feature} by {action}.",
            "Updated documentation for {feature}.",
            "Added tests for {feature}.",
            "Reviewed {feature} and {action}.",
            "Merged {feature} with {action}.",
            "Deployed {feature} with {action}."
        ]
        
        self.adjectives = {
            'positive': ['amazing', 'fantastic', 'excellent', 'brilliant', 'outstanding', 'superb', 'wonderful', 'perfect', 'awesome', 'incredible'],
            'negative': ['terrible', 'awful', 'horrible', 'terrible', 'disastrous', 'useless', 'broken', 'buggy', 'frustrating', 'annoying']
        }
        
        self.features = [
            'API', 'UI', 'backend', 'frontend', 'database', 'authentication', 'caching', 'logging', 'testing', 'deployment',
            'performance', 'security', 'documentation', 'error handling', 'validation', 'routing', 'middleware', 'configuration',
            'monitoring', 'analytics', 'search', 'notifications', 'file upload', 'image processing', 'data export', 'reporting'
        ]
        
        self.actions = [
            'works perfectly', 'is working great', 'performs well', 'handles errors', 'processes data', 'validates input',
            'caches results', 'logs events', 'authenticates users', 'authorizes access', 'sends notifications',
            'uploads files', 'downloads data', 'generates reports', 'creates backups', 'syncs data', 'updates records',
            'deletes entries', 'searches content', 'filters results', 'sorts data', 'paginates results', 'exports data',
            'imports data', 'validates forms', 'handles requests', 'processes responses', 'manages sessions', 'tracks metrics'
        ]
        
        self.repos = [
            'tensorflow/tensorflow', 'pytorch/pytorch', 'microsoft/vscode', 'django/django', 'flutter/flutter',
            'facebook/react', 'vuejs/vue', 'angular/angular', 'nodejs/node', 'python/cpython', 'microsoft/TypeScript',
            'rust-lang/rust', 'golang/go', 'kubernetes/kubernetes', 'docker/compose', 'hashicorp/terraform',
            'pandas-dev/pandas', 'numpy/numpy', 'scikit-learn/scikit-learn', 'matplotlib/matplotlib',
            'webpack/webpack', 'babel/babel', 'eslint/eslint', 'prettier/prettier', 'jestjs/jest'
        ]
        
        self.users = [
            'developer123', 'coder456', 'programmer789', 'dev_awesome', 'code_master', 'tech_guru', 'bug_hunter',
            'feature_dev', 'backend_pro', 'frontend_wiz', 'data_scientist', 'ml_engineer', 'devops_guy',
            'qa_tester', 'ui_designer', 'architect', 'senior_dev', 'junior_dev', 'team_lead', 'project_manager'
        ]
    
    def generate_positive_comment(self):
        """Generate a positive sentiment comment"""
        template = random.choice(self.positive_templates)
        adj = random.choice(self.adjectives['positive'])
        feature = random.choice(self.features)
        action = random.choice(self.actions)
        
        return template.format(adj=adj, feature=feature, action=action)
    
    def generate_negative_comment(self):
        """Generate a negative sentiment comment"""
        template = random.choice(self.negative_templates)
        adj = random.choice(self.adjectives['negative'])
        feature = random.choice(self.features)
        action = random.choice(self.actions)
        
        return template.format(adj=adj, feature=feature, action=action)
    
    def generate_neutral_comment(self):
        """Generate a neutral sentiment comment"""
        template = random.choice(self.neutral_templates)
        feature = random.choice(self.features)
        action = random.choice(self.actions)
        
        return template.format(feature=feature, action=action)
    
    def generate_mixed_comment(self):
        """Generate a mixed sentiment comment (realistic)"""
        positive_part = random.choice([
            "The feature works well",
            "Performance is good",
            "The code is clean",
            "Documentation is clear",
            "The API is intuitive"
        ])
        
        negative_part = random.choice([
            "but there are some bugs",
            "but it's a bit slow",
            "but the error handling could be better",
            "but the configuration is confusing",
            "but the documentation is incomplete"
        ])
        
        return f"{positive_part}, {negative_part}."
    
    def generate_realistic_comments(self, num_comments=2000):
        """Generate a realistic dataset of developer comments"""
        comments = []
        
        # Generate positive comments (30%)
        num_positive = int(num_comments * 0.3)
        for i in range(num_positive):
            comments.append({
                'repo': random.choice(self.repos),
                'comment_id': f"pos_{i}",
                'body': self.generate_positive_comment(),
                'created_at': self.generate_random_date(),
                'user': random.choice(self.users),
                'type': random.choice(['issue', 'pr']),
                'sentiment': 2
            })
        
        # Generate negative comments (30%)
        num_negative = int(num_comments * 0.3)
        for i in range(num_negative):
            comments.append({
                'repo': random.choice(self.repos),
                'comment_id': f"neg_{i}",
                'body': self.generate_negative_comment(),
                'created_at': self.generate_random_date(),
                'user': random.choice(self.users),
                'type': random.choice(['issue', 'pr']),
                'sentiment': 0
            })
        
        # Generate neutral comments (25%)
        num_neutral = int(num_comments * 0.25)
        for i in range(num_neutral):
            comments.append({
                'repo': random.choice(self.repos),
                'comment_id': f"neu_{i}",
                'body': self.generate_neutral_comment(),
                'created_at': self.generate_random_date(),
                'user': random.choice(self.users),
                'type': random.choice(['issue', 'pr']),
                'sentiment': 1
            })
        
        # Generate mixed comments (15%)
        num_mixed = num_comments - num_positive - num_negative - num_neutral
        for i in range(num_mixed):
            comments.append({
                'repo': random.choice(self.repos),
                'comment_id': f"mix_{i}",
                'body': self.generate_mixed_comment(),
                'created_at': self.generate_random_date(),
                'user': random.choice(self.users),
                'type': random.choice(['issue', 'pr']),
                'sentiment': 1  # Mixed comments tend to be neutral
            })
        
        # Shuffle the comments
        random.shuffle(comments)
        
        return pd.DataFrame(comments)
    
    def generate_random_date(self):
        """Generate a random date within the last year"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        random_number_of_days = random.randrange(days_between_dates)
        random_date = start_date + timedelta(days=random_number_of_days)
        
        return random_date.isoformat()
    
    def save_dataset(self, df, filename='synthetic_comments.csv'):
        """Save the generated dataset"""
        import os
        os.makedirs('data', exist_ok=True)
        output_path = os.path.join('data', filename)
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
        print(f"Total comments: {len(df)}")
        print(f"Sentiment distribution:")
        print(df['sentiment'].value_counts().sort_index())
        print(f"Repositories: {df['repo'].nunique()}")
        print(f"Users: {df['user'].nunique()}")

def main():
    print("Generating synthetic developer comments dataset...")
    
    generator = SyntheticDatasetGenerator()
    
    # Generate a large dataset
    df = generator.generate_realistic_comments(num_comments=3000)
    
    # Save the dataset
    generator.save_dataset(df)
    
    print("\nDataset generation completed!")
    print("This synthetic dataset contains realistic developer comments")
    print("that mimic the patterns and language used in real GitHub discussions.")

if __name__ == "__main__":
    main() 