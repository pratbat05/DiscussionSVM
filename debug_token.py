import os
from dotenv import load_dotenv

print("=== Debug Token Loading ===")

# Load .env file
load_dotenv()

# Get token
token = os.getenv('GITHUB_TOKEN')

print(f"Token exists: {token is not None}")
if token:
    print(f"Token length: {len(token)}")
    print(f"Token starts with: {token[:10]}")
    print(f"Token ends with: {token[-10:]}")
    print(f"Full token: {token}")
else:
    print("No token found!")

# Check if .env file exists
env_path = '.env'
print(f"\n.env file exists: {os.path.exists(env_path)}")

if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        content = f.read().strip()
        print(f".env file content: {content}")
        print(f".env file length: {len(content)}") 