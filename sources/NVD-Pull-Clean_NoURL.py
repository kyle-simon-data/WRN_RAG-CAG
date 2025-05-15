import requests
import json
import os
from datetime import datetime, timedelta
import time
import boto3
from botocore.exceptions import ClientError


def get_secret():
    secret_name = "NVD_API_Key"
    region_name = "us-east-1"

    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)

    response = client.get_secret_value(SecretId=secret_name)
    secret = response['SecretString']
    return json.loads(secret)['NVD_API']  


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


# These functions have been removed as they were used for extracting exploit URLs


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
    else:
        print("No data was retrieved from the NVD API. Skipping processing steps.")


if __name__ == "__main__":
    main()