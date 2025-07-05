import os
import requests
import pandas as pd
import time
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

class EnhancedGitHubDataCollector:
    def __init__(self):
        self.token = os.getenv('GITHUB_TOKEN')
        self.headers = {'Authorization': f'token {self.token}'} if self.token else {}
        self.base_url = 'https://api.github.com'
        
    def fetch_issue_comments(self, repos, max_comments_per_repo=500):
        """
        Fetch issue comments from specified GitHub repositories
        repos: list of strings in format 'owner/repo'
        """
        all_comments = []
        
        for repo in repos:
            print(f"Fetching comments from {repo}...")
            
            # Fetch both issue comments and pull request comments
            endpoints = [
                f"{self.base_url}/repos/{repo}/issues/comments",
                f"{self.base_url}/repos/{repo}/pulls/comments"
            ]
            
            for endpoint in endpoints:
                try:
                    params = {
                        'per_page': 100,  # Max per page
                        'sort': 'created',
                        'direction': 'desc'
                    }
                    
                    page = 1
                    comments_fetched = 0
                    
                    while comments_fetched < max_comments_per_repo // 2:
                        params['page'] = page
                        response = requests.get(endpoint, headers=self.headers, params=params)
                        
                        if response.status_code == 200:
                            comments = response.json()
                            if not comments:  # No more comments
                                break
                                
                            for comment in comments:
                                if comments_fetched >= max_comments_per_repo // 2:
                                    break
                                    
                                # Filter out very short or very long comments
                                body = comment.get('body', '')
                                if 10 <= len(body) <= 2000:  # Reasonable length
                                    all_comments.append({
                                        'repo': repo,
                                        'comment_id': comment['id'],
                                        'body': body,
                                        'created_at': comment['created_at'],
                                        'user': comment['user']['login'],
                                        'type': 'issue' if 'issues' in endpoint else 'pr'
                                    })
                                    comments_fetched += 1
                            
                            page += 1
                            # Rate limiting - be respectful to GitHub API
                            time.sleep(0.1)
                        else:
                            print(f"Error fetching from {endpoint}: {response.status_code}")
                            break
                            
                except Exception as e:
                    print(f"Error processing {repo}: {str(e)}")
                    continue
            
            print(f"Fetched {len([c for c in all_comments if c['repo'] == repo])} comments from {repo}")
            
            # Rate limiting between repositories
            time.sleep(1)
        
        return pd.DataFrame(all_comments)
    
    def add_synthetic_labels(self, df):
        """
        Add synthetic sentiment labels based on keywords and patterns
        This is a simple heuristic approach - in practice you'd want human labeling
        """
        def classify_sentiment(text):
            text_lower = text.lower()
            
            # Positive indicators
            positive_words = [
                'great', 'awesome', 'excellent', 'amazing', 'fantastic', 'perfect',
                'love', 'thanks', 'thank you', 'good', 'nice', 'wonderful', 'brilliant',
                'outstanding', 'superb', 'incredible', 'fabulous', 'terrific', 'splendid',
                'working', 'works', 'fixed', 'resolve', 'solved', 'helpful', 'useful',
                'improved', 'better', 'faster', 'cleaner', 'simpler', 'easier'
            ]
            
            # Negative indicators
            negative_words = [
                'broken', 'bug', 'crash', 'error', 'fail', 'failed', 'failing',
                'terrible', 'awful', 'horrible', 'bad', 'worst', 'useless', 'broken',
                'doesn\'t work', 'not working', 'issue', 'problem', 'buggy', 'slow',
                'confusing', 'difficult', 'hard', 'complicated', 'annoying', 'frustrating',
                'disappointing', 'poor', 'weak', 'terrible', 'awful'
            ]
            
            # Neutral indicators
            neutral_words = [
                'update', 'updated', 'change', 'changed', 'modify', 'modified',
                'add', 'added', 'remove', 'removed', 'test', 'testing', 'documentation',
                'comment', 'commented', 'review', 'reviewed', 'merge', 'merged',
                'commit', 'committed', 'push', 'pushed', 'pull', 'pulled'
            ]
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            neutral_count = sum(1 for word in neutral_words if word in text_lower)
            
            # Simple scoring system
            if positive_count > negative_count and positive_count > neutral_count:
                return 2  # Positive
            elif negative_count > positive_count and negative_count > neutral_count:
                return 0  # Negative
            else:
                return 1  # Neutral
        
        df['sentiment'] = df['body'].apply(classify_sentiment)
        return df
    
    def save_data(self, df, filename='enhanced_comments.csv'):
        """Save the collected data to CSV file"""
        os.makedirs('data', exist_ok=True)
        output_path = os.path.join('data', filename)
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
        print(f"Total comments collected: {len(df)}")
        print(f"Sentiment distribution:")
        print(df['sentiment'].value_counts().sort_index())

def main():
    # Expanded list of diverse, active repositories
    repos = [
        # Major frameworks and libraries
        'tensorflow/tensorflow',
        'pytorch/pytorch', 
        'microsoft/vscode',
        'django/django',
        'flutter/flutter',
        'facebook/react',
        'vuejs/vue',
        'angular/angular',
        'nodejs/node',
        'python/cpython',
        'microsoft/TypeScript',
        'rust-lang/rust',
        'golang/go',
        
        # Developer tools
        'kubernetes/kubernetes',
        'docker/compose',
        'hashicorp/terraform',
        'ansible/ansible',
        'jenkinsci/jenkins',
        'git/git',
        'github/gitignore',
        
        # Popular applications
        'mozilla/firefox',
        'atom/atom',
        'brave/brave-browser',
        'signalapp/Signal-Desktop',
        'telegramdesktop/tdesktop',
        'obsproject/obs-studio',
        
        # Data science and ML
        'pandas-dev/pandas',
        'numpy/numpy',
        'scikit-learn/scikit-learn',
        'matplotlib/matplotlib',
        'plotly/plotly.py',
        'streamlit/streamlit',
        'jupyter/notebook',
        
        # Web development
        'webpack/webpack',
        'babel/babel',
        'eslint/eslint',
        'prettier/prettier',
        'postcss/postcss',
        
        # Mobile development
        'facebook/react-native',
        'expo/expo',
        'ionic-team/ionic',
        
        # Backend and databases
        'mongodb/mongo',
        'redis/redis',
        'elastic/elasticsearch',
        'apache/kafka',
        
        # DevOps and infrastructure
        'hashicorp/vagrant',
        'prometheus/prometheus',
        'grafana/grafana',
        'istio/istio',
        
        # Testing and CI/CD
        'jestjs/jest',
        'cypress-io/cypress',
        'actions/runner',
        'travis-ci/travis-ci'
    ]

    print("Starting enhanced GitHub data collection...")
    print(f"Target repositories: {len(repos)}")
    
    collector = EnhancedGitHubDataCollector()
    
    # Check if GitHub token is available
    if not collector.token:
        print("Warning: No GitHub token found. Using unauthenticated requests (rate limited).")
        print("For better results, add GITHUB_TOKEN to your .env file")
    
    df = collector.fetch_issue_comments(repos, max_comments_per_repo=300)
    
    if not df.empty:
        print("\nAdding sentiment labels...")
        df = collector.add_synthetic_labels(df)
        collector.save_data(df)
        
        # Show some statistics
        print(f"\nDataset Statistics:")
        print(f"Total comments: {len(df)}")
        print(f"Unique repositories: {df['repo'].nunique()}")
        print(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")
        print(f"Average comment length: {df['body'].str.len().mean():.1f} characters")
    else:
        print("No data collected. Check your GitHub token and internet connection.")

if __name__ == "__main__":
    main() 