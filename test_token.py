import os
import requests
from dotenv import load_dotenv

load_dotenv()

token = os.getenv('GITHUB_TOKEN')
print(f"Token loaded: {'Yes' if token else 'No'}")
if token:
    print(f"Token starts with: {token[:10]}...")

headers = {'Authorization': f'token {token}'} if token else {}

# Test with a simple public API call
response = requests.get('https://api.github.com/user', headers=headers)
print(f"Status code: {response.status_code}")

if response.status_code == 200:
    user_data = response.json()
    print(f"Authenticated as: {user_data.get('login', 'Unknown')}")
    print("Token is working!")
elif response.status_code == 401:
    print("Token is invalid or expired")
    print("Response:", response.text)
else:
    print(f"Unexpected response: {response.status_code}")
    print("Response:", response.text) 