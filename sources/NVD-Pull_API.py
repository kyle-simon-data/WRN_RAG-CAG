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
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    secret = get_secret_value_response['SecretString']

# Configuration
API_KEY = get_secret()
BASE_URL = 'https://services.nvd.nist.gov/rest/json/cves/2.0'
HEADERS = {'apiKey': API_KEY}
RESULTS_PER_PAGE = 100 
OUTPUT_DIR = 'data/NVD'
SLEEP_TIME = 6

# Define the date range for the past month
end_date = datetime.utcnow()  # deprecated and needs updated
start_date = end_date - timedelta(days=30)

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Format dates for API request
start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')

def save_cve_record(cve_record, output_dir):
    """Save a CVE record to a JSON file."""
    cve_id = cve_record['cve']['id']
    with open(f'{output_dir}/{cve_id}.json', 'w') as file:
        json.dump(cve_record, file, indent=2)

def main():
    """Main function to retrieve and store CVE records."""
    start_index = 0
    total_results = 1  # Initial value to enter the loop

    while start_index < total_results:
        params = {
            'startIndex': start_index,
            'resultsPerPage': RESULTS_PER_PAGE,
            'pubStartDate': start_date_str,
            'pubEndDate': end_date_str,
            'noRejected': None,
            'cvssV3Severity': 'CRITICAL'
        }
        response = requests.get(BASE_URL, headers=HEADERS, params=params)

        if response.status_code == 200:
            try:
                data = response.json()
            except json.JSONDecodeError:  # Fixed typo in the original: JSONDecodeEror -> JSONDecodeError
                print(f'Error decoding JSON response at index {start_index}.')
                print('Response text:', response.text)
                break

            total_results = data.get('totalResults', 0)
            vulnerabilities = data.get('vulnerabilities', [])
            
            for cve in vulnerabilities:
                save_cve_record(cve, OUTPUT_DIR)

            print(f'Retrieved {len(vulnerabilities)} CVE records. Total so far: {start_index + len(vulnerabilities)}.')
            start_index += RESULTS_PER_PAGE
                                   
        else: 
            print(f'Failed to retrieve data: {response.status_code}')
            print('Response text:', response.text)
            break
            
        time.sleep(SLEEP_TIME)

    print('Data retrieval and storage complete.')

if __name__ == "__main__":
    main()