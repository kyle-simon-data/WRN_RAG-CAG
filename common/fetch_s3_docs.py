import boto3
import os

# Initialize your S3 client
s3 = boto3.client('s3')

BUCKET_NAME = 'cyberbot-rag-knowledgebase'

# Function to list objects in the bucket
def list_bucket_objects(bucket):
    response = s3.list_objects_v2(Bucket=bucket)
    return [obj['Key'] for obj in response.get('Contents', [])]

# Function to download a file
def download_file(bucket, key, download_path):
    s3.download_file(bucket, key, download_path)
    print(f"Downloaded {key} to {download_path}")

# Test listing files
files = list_bucket_objects(BUCKET_NAME)
print(f"Files in bucket: {files}")

# Test downloading and validating files
os.makedirs('data/downloads', exist_ok=True)

for file_key in files[:3]:  # Fetch first 3 files as samples
    local_path = os.path.join('data/downloads', os.path.basename(file_key))
    download_file(BUCKET_NAME, file_key, local_path)

    # Simple validation: check if file is non-empty
    if os.path.getsize(local_path) > 0:
        print(f"Validation passed for {file_key}")
    else:
        print(f"Validation failed for {file_key}")
