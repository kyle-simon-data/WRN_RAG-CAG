import requests
import json
import os
from datetime import datetime, timedelta
import time
from bs4 import BeautifulSoup
import boto3
from botocore.exceptions import ClientError


def get_secret():
    secret_name = "NVD_API_Key"
    region_name = "us-east-1"

    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)

    response = client.get_secret_value(SecretId=secret_name)
    secret = response['SecretString']
    return json.loads(secret)['NVD_API']  # <-- use the correct key name



def save_cve_record(cve_record, output_dir):
    """Save a CVE record to a JSON file."""
    cve_id = cve_record['cve']['id']
    with open(f'{output_dir}/{cve_id}.json', 'w') as file:
        json.dump(cve_record, file, indent=2)


def clean_json_files(tobecleaned_directory, cleaned_directory):
    """Clean and simplify JSON files retrieved from NVD API."""
    # Create output directory if it doesn't exist
    if not os.path.exists(cleaned_directory):
        os.makedirs(cleaned_directory)

    # Check if directory is empty
    if not os.listdir(tobecleaned_directory):
        print(f"Warning: No files found in {tobecleaned_directory}")
        return

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
    """Scrape content from a URL and save titles, headings, and body text to a file."""
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
    """Process each JSON file to extract and scrape exploit URLs."""
    # Check if directory exists and is not empty
    if not os.path.exists(json_dir) or not os.listdir(json_dir):
        print(f"Warning: No files found in {json_dir}")
        return

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
                
                # Scrape content from all extracted "exploit" URLs with a delay
                for url in exploit_urls:
                    scrape_url(url, output_file)
            else:
                print(f"No exploit URLs found in {filename}.")


def main():
    """Main function to retrieve, process, and analyze NVD CVE records."""
    # Configuration
    API_KEY = get_secret()
    BASE_URL = 'https://services.nvd.nist.gov/rest/json/cves/2.0'
    HEADERS = {'apiKey': API_KEY}
    RESULTS_PER_PAGE = 100 
    OUTPUT_DIR = 'data/NVD'
    SLEEP_TIME = 6

    # Define the date range for the past month
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Format dates for API request
    start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')

    print(f"Retrieving CVE records from {start_date_str} to {end_date_str}")

    # Step 1: Retrieve and store CVE records
    start_index = 0
    total_results = 1  # Initial value to enter the loop
    data_retrieved = False

    while start_index < total_results:
        params = {
            'startIndex': start_index,
            'resultsPerPage': RESULTS_PER_PAGE,
            'pubStartDate': start_date_str,
            'pubEndDate': end_date_str,
            'noRejected': None,  # Using None as in the original working code
            'cvssV3Severity': 'CRITICAL'
        }
        
        response = requests.get(BASE_URL, headers=HEADERS, params=params)
        
        if response.status_code == 200:
            try:
                data = response.json()
                data_retrieved = True
            except json.JSONDecodeError:
                print(f'Error decoding JSON response at index {start_index}.')
                print('Response text:', response.text)
                break

            total_results = data.get('totalResults', 0)
            vulnerabilities = data.get('vulnerabilities', [])
            
            if not vulnerabilities:
                print("No vulnerabilities found in this batch.")
                if total_results > 0:
                    # Continue to next batch
                    start_index += RESULTS_PER_PAGE
                    continue
                else:
                    # No results at all, break the loop
                    break
                
            for cve in vulnerabilities:
                save_cve_record(cve, OUTPUT_DIR)

            print(f'Retrieved {len(vulnerabilities)} CVE records. Total so far: {start_index + len(vulnerabilities)} of {total_results}.')
            start_index += RESULTS_PER_PAGE
                                
        else: 
            print(f'Failed to retrieve data: {response.status_code}')
            print('Response text:', response.text)
            break
            
        time.sleep(SLEEP_TIME)

    if data_retrieved:
        print('Data retrieval and storage complete.')
        
        # Step 2: Clean and simplify JSON files
        cleaned_directory = f"{OUTPUT_DIR}_cleaned"
        clean_json_files(OUTPUT_DIR, cleaned_directory)
        print(f'JSON cleaning complete. Cleaned files saved to {cleaned_directory}')

        # Step 3: Extract exploit URLs and scrape content
        process_exploit_urls(cleaned_directory)
        print('Exploit URL extraction and scraping complete.')
    else:
        print("No data was retrieved from the NVD API. Skipping processing steps.")


if __name__ == "__main__":
    main()