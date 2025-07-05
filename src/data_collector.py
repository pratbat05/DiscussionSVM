import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

class GitHubDataCollector:
    def __init__(self):
        self.token = os.getenv('GITHUB_TOKEN')
        self.headers = {'Authorization': f'token {self.token}'}
        self.base_url = 'https://api.github.com'

    def fetch_issue_comments(self, repos, max_comments_per_repo=200):
        """
        Fetch issue comments from specified GitHub repositories
        repos: list of strings in format 'owner/repo'
        """
        all_comments = []
        
        for repo in repos:
            print(f"Fetching comments from {repo}...")
            url = f"{self.base_url}/repos/{repo}/issues/comments"
            params = {
                'per_page': max_comments_per_repo,
                'sort': 'created',
                'direction': 'desc'
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                comments = response.json()
                for comment in comments:
                    all_comments.append({
                        'repo': repo,
                        'comment_id': comment['id'],
                        'body': comment['body'],
                        'created_at': comment['created_at'],
                        'user': comment['user']['login']
                    })
                print(f"Fetched {len(comments)} comments from {repo}")
            else:
                print(f"Error fetching comments from {repo}: {response.status_code}")

        return pd.DataFrame(all_comments)

    def save_data(self, df, filename='raw_comments.csv'):
        """Save the collected data to CSV file"""
        output_path = os.path.join('data', filename)
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")

def main():
    # List of diverse, active repositories for better training data
    repos = [
        # Popular frameworks and libraries
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
        
        # Developer tools and platforms
        'microsoft/TypeScript',
        'rust-lang/rust',
        'golang/go',
        'kubernetes/kubernetes',
        'docker/compose',
        'hashicorp/terraform',
        'ansible/ansible',
        'jenkinsci/jenkins',
        
        # Popular applications
        'mozilla/firefox',
        'atom/atom',
        'sublimehq/Packages',
        'brave/brave-browser',
        'signalapp/Signal-Desktop',
        'telegramdesktop/tdesktop',
        
        # Data science and ML
        'pandas-dev/pandas',
        'numpy/numpy',
        'scikit-learn/scikit-learn',
        'matplotlib/matplotlib',
        'plotly/plotly.py',
        'streamlit/streamlit'
    ]

    collector = GitHubDataCollector()
    df = collector.fetch_issue_comments(repos)
    collector.save_data(df)

if __name__ == "__main__":
    main() 