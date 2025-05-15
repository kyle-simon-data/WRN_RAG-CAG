import boto3
import requests
import os
import time

#Config
SECRET_NAME = "markdownscraper"
REGION_NAME = "us-east-1"
REPO_OWNER = "redcanaryco"
REPO_NAME = "atomic-red-team"
BRANCH = "master"
TARGET_DIR = "atomics"
EXCLUDE_SUBFOLDER = "atomics/Indexes"
LOCAL_SAVE_DIR = "data/atomics"

#get GitHub Token from AWS Secrets Manager ---
def get_github_token(secret_name, region_name):
    client = boto3.client("secretsmanager", region_name=region_name)
    response = client.get_secret_value(SecretId=secret_name)
    return response["SecretString"].strip()

#recursive capture of files
def get_markdown_file_paths(repo_owner, repo_name, branch, token):
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/trees/{branch}?recursive=1"
    headers = {"Authorization": f"token {token}"}
    response = requests.get(api_url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to fetch repo tree: {response.status_code}")
        print(response.json())
        return []

    all_files = response.json()["tree"]
    md_files = [
        f["path"] for f in all_files
        if f["path"].endswith(".md")
        and f["path"].startswith(f"{TARGET_DIR}/")
        and not f["path"].startswith(f"{EXCLUDE_SUBFOLDER}/")
    ]
    return md_files

#DL and save contents of .md files
def download_and_save_files(repo_owner, repo_name, branch, file_paths):
    os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)  # Create flat output directory once
    for path in file_paths:
        raw_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{path}"
        response = requests.get(raw_url)
        if response.status_code == 200:
            filename = os.path.basename(path)
            local_path = os.path.join(LOCAL_SAVE_DIR, filename)
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"Saved: {local_path}")
            time.sleep(1)  # Pause for 1 second between downloads
        else:
            print(f"Failed to download: {path} (Status code: {response.status_code})")


# run
if __name__ == "__main__":
    token = get_github_token(SECRET_NAME, REGION_NAME)
    md_file_paths = get_markdown_file_paths(REPO_OWNER, REPO_NAME, BRANCH, token)
    print(f"🔍 Found {len(md_file_paths)} Markdown files (excluding 'indexes/')")
    download_and_save_files(REPO_OWNER, REPO_NAME, BRANCH, md_file_paths)
