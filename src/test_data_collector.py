import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class TestDataCollector:
    def __init__(self):
        self.token = os.getenv('GITHUB_TOKEN')
        self.headers = {'Authorization': f'token {self.token}'} if self.token else {}
        self.base_url = 'https://api.github.com'
        
    def test_single_repo(self, repo):
        """Test fetching from a single repository"""
        print(f"Testing {repo}...")
        
        url = f"{self.base_url}/repos/{repo}/issues/comments"
        params = {'per_page': 5, 'sort': 'created', 'direction': 'desc'}
        
        response = requests.get(url, headers=self.headers, params=params)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            comments = response.json()
            print(f"Success! Found {len(comments)} comments")
            if comments:
                print(f"Sample comment: {comments[0]['body'][:100]}...")
            return True
        else:
            print(f"Error: {response.text}")
            return False

def main():
    print("Testing GitHub API access...")
    
    collector = TestDataCollector()
    
    # Test with a few popular repos
    test_repos = [
        'microsoft/vscode',
        'facebook/react',
        'tensorflow/tensorflow'
    ]
    
    for repo in test_repos:
        success = collector.test_single_repo(repo)
        if success:
            print(f"✅ {repo} - Working!")
        else:
            print(f"❌ {repo} - Failed!")
        print("-" * 50)

if __name__ == "__main__":
    main() 