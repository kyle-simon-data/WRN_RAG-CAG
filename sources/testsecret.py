import boto3
import requests

# --- Configuration ---
SECRET_NAME = "markdownscraper"     # Replace with your actual secret name
REGION_NAME = "us-east-1"      # Replace if needed

# --- Load token from AWS Secrets Manager ---
def get_github_token(secret_name, region_name):
    client = boto3.client("secretsmanager", region_name=region_name)
    response = client.get_secret_value(SecretId=secret_name)
    secret = response["SecretString"]

    # Debug and clean token
    print(f"Secret length: {len(secret)}")
    print(f"Starts with: {repr(secret[:10])}")

    # Clean up the token
    return secret.strip().strip('"').strip("'")

# --- Test GitHub API ---
def test_github_token(token):
    headers = {"Authorization": f"token {token}"}
    url = "https://api.github.com/user"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        print("✅ GitHub token works. User info:")
        print(response.json())
    else:
        print(f"❌ Failed to authenticate. Status code: {response.status_code}")
        print(response.json())

# --- Run ---
if __name__ == "__main__":
    token = get_github_token(SECRET_NAME, REGION_NAME)
    test_github_token(token)
