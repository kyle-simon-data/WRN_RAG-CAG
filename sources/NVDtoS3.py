#!/usr/bin/env python3

import requests
import json
import os
from datetime import datetime, timedelta
import time
import boto3
from botocore.exceptions import ClientError
from bs4 import BeautifulSoup
import shutil

def get_secret():
    """Retrieve NVD API Key from AWS Secrets Manager."""
    secret_name = "NVD_API_Key"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
        return get_secret_value_response['SecretString']
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

def save_cve_record(cve_record, output_dir):
    """Save a CVE record to a JSON file."""
    cve_id = cve_record['cve']['id']
    with open(f'{output_dir}/{cve_id}.json', 'w') as file:
        json.dump(cve_record, file, indent=2)

def retrieve_nvd_records(api_key, output_dir, days_back=120, severity="HIGH"):
    """Retrieve CVE records from NVD API."""
    # Define constants
    base_url = 'https://services.nvd.nist.gov/rest/json/cves/2.0'
    headers = {'apiKey': api_key}
    results_per_page = 100  # NVD's recommended value to balance the load
    sleep_time = 6  # Delay in seconds between requests

    # Create output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)

    # Format dates to ISO 8601 format
    start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')

    # Initialize pagination parameters
    start_index = 0
    total_results = 1  # Initialize to a non-zero value to enter the loop

    # Retrieve and save CVE records
    while start_index < total_results:
        params = {
            'startIndex': start_index,
            'resultsPerPage': results_per_page,
            'pubStartDate': start_date_str,
            'pubEndDate': end_date_str,
            'noRejected': None,
            'cvssV3Severity': severity
        }
        response = requests.get(base_url, headers=headers, params=params)
        
        if response.status_code == 200:
            try:
                data = response.json()
            except json.JSONDecodeError:
                print(f'Error decoding JSON response at index {start_index}.')
                print('Response text:', response.text)
                break
            
            # Update pagination info
            total_results = data.get('totalResults', 0)
            
            # Save each CVE record
            vulnerabilities = data.get('vulnerabilities', [])
            for cve in vulnerabilities:
                save_cve_record(cve, output_dir)
            
            print(f'Retrieved {len(vulnerabilities)} CVE records. Total so far: {start_index + len(vulnerabilities)}.')
            start_index += results_per_page
        else:
            print(f'Failed to retrieve data: {response.status_code}')
            print('Response text:', response.text)
            break
        
        # Sleep for 6 seconds before the next request
        time.sleep(sleep_time)

    print('Data retrieval and storage complete.')
    return output_dir

def clean_json_files(tobecleaned_directory, cleaned_directory):
    """Extract English descriptions and references from CVE JSON files."""
    # Create output directory if it doesn't exist
    if not os.path.exists(cleaned_directory):
        os.makedirs(cleaned_directory)

    for filename in os.listdir(tobecleaned_directory):
        if filename.endswith(".json"):
            input_filepath = os.path.join(tobecleaned_directory, filename)
            output_filepath = os.path.join(cleaned_directory, filename)
            
            with open(input_filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)

            cleaned_data = {
                "descriptions": [],
                "references": data.get("cve", {}).get("references", [])
            }
            
            # Filter descriptions where lang is "en"
            descriptions = data.get("cve", {}).get("descriptions", [])
            for description in descriptions:
                if description.get("lang") == "en":
                    cleaned_data["descriptions"].append(description["value"])

            # Save cleaned data to new JSON file in the output directory
            with open(output_filepath, 'w', encoding='utf-8') as file:
                json.dump(cleaned_data, file, indent=4)
    
    return cleaned_directory

def extract_exploit_urls_from_json(json_file):
    """Extract URLs tagged as 'exploit' from a JSON file."""
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        urls = []
        references = data.get('references', [])
        print(f"Processing {json_file}: Found {len(references)} references.")
        for ref in references:
            if 'tags' in ref and 'Exploit' in ref['tags']:
                urls.append(ref['url'])
                print(f"Found exploit URL: {ref['url']}")
        return urls

def scrape_url(url, output_file, delay=1):
    """Scrape content from a URL with a delay and save to a file."""
    if 'github.com' in url or 'chromium.org' in url:
        print(f"Skipping URL: {url}")
        return
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = soup.title.string if soup.title else 'No title found'
        print(f"Title: {title}")
        
        headings = [heading.get_text() for heading in soup.find_all(['h1', 'h2', 'h3'])]
        body = soup.get_text()
        
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"Title: {title}\n")
            f.write("Headings:\n")
            for heading in headings:
                f.write(f"{heading}\n")
            f.write("Body:\n")
            f.write(f"{body}\n\n")
    except requests.RequestException as e:
        print(f"Failed to retrieve {url}: {e}")
    finally:
        time.sleep(delay)  # Delay between requests

def process_exploit_urls(json_dir):
    """Process and scrape exploit URLs from all JSON files in a directory."""
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            json_file = os.path.join(json_dir, filename)
            exploit_urls = extract_exploit_urls_from_json(json_file)
            if exploit_urls:
                output_file = os.path.join(json_dir, f"{os.path.splitext(filename)[0]}.txt")
                print(f"Extracted {len(exploit_urls)} exploit URLs from {filename}.")
                
                # Ensure the output file is empty before writing
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write('')
                
                # Scrape content from all extracted "exploit" URLs
                for url in exploit_urls:
                    scrape_url(url, output_file)
            else:
                print(f"No exploit URLs found in {filename}.")

def get_bucket_name():
    """Return the S3 bucket name and region."""
    # Hardcoded bucket name and region as provided
    return "cyberbot-rag-knowledgebase", "us-east-1"

def upload_folder_to_s3(local_folder, bucket_name, region_name):
    """Upload all files in a folder to S3.
    
    Uses default AWS credentials configuration (environment variables, 
    ~/.aws/credentials, EC2 instance profile, etc.)
    """
    # Create an S3 client using default credentials and specified region
    s3_client = boto3.client('s3', region_name=region_name)
    
    for root, dirs, files in os.walk(local_folder):
        # Skip the '.ipynb_checkpoints' directory
        if '.ipynb_checkpoints' in dirs:
            dirs.remove('.ipynb_checkpoints')
        
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_folder)
            s3_path = relative_path.replace("\\", "/")  # Ensure S3 path uses forward slashes
            try:
                s3_client.upload_file(local_path, bucket_name, s3_path)
                print(f'Successfully uploaded {local_path} to s3://{bucket_name}/{s3_path}')
            except Exception as e:
                print(f'Failed to upload {local_path} to s3://{bucket_name}/{s3_path}: {e}')

def move_directories_to_archive(dir1, dir2, archive_dir):
    """Move directories to an archive folder."""
    # Ensure the archive directory exists
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
    
    # Move the first directory
    shutil.move(dir1, os.path.join(archive_dir, os.path.basename(dir1)))
    
    # Move the second directory
    shutil.move(dir2, os.path.join(archive_dir, os.path.basename(dir2)))

def main():
    # 1. Retrieve NVD records
    api_key = get_secret()
    output_dir = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    retrieve_nvd_records(api_key, output_dir)
    
    # 2. Clean JSON files
    cleaned_directory = f"{output_dir}_cleaned"
    clean_json_files(output_dir, cleaned_directory)
    
    # 3. Process exploit URLs
    process_exploit_urls(cleaned_directory)
    
    # 4. Upload to S3
    bucket_name, region_name = get_bucket_name()
    
    # Upload cleaned files to S3
    upload_folder_to_s3(cleaned_directory, bucket_name, region_name)
    
    # 5. Move directories to archive
    archive_dir = 'archive'
    move_directories_to_archive(output_dir, cleaned_directory, archive_dir)
    
    print("Process completed successfully.")

if __name__ == "__main__":
    main()