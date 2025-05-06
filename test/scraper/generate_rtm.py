import pandas as pd
import os
from datetime import datetime

def create_rtm():
    # Define the RTM columns
    columns = [
        'Test Case ID',
        'Test Case Description',
        'Test Steps',
        'Expected Result',
        'Actual Result',
        'Status',
        'Date Executed',
        'Comments'
    ]

    # Define test cases
    test_cases = [
        {
            'Test Case ID': 'TC001',
            'Test Case Description': 'Test clean_text function with extra spaces',
            'Test Steps': '1. Call clean_text with text containing multiple spaces\n2. Verify the output',
            'Expected Result': 'Text should be cleaned with single spaces between words',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        },
        {
            'Test Case ID': 'TC002',
            'Test Case Description': 'Test successful Wikipedia page scraping',
            'Test Steps': '1. Mock Wikipedia page response\n2. Call scrape_wikipedia_page\n3. Verify the output',
            'Expected Result': 'Should return cleaned text content from the page',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        },
        {
            'Test Case ID': 'TC003',
            'Test Case Description': 'Test Wikipedia page scraping with network error',
            'Test Steps': '1. Mock network error\n2. Call scrape_wikipedia_page\n3. Verify error handling',
            'Expected Result': 'Should return empty string on error',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        },
        {
            'Test Case ID': 'TC004',
            'Test Case Description': 'Test save_data function',
            'Test Steps': '1. Mock scrape_wikipedia_page\n2. Call save_data\n3. Verify file creation',
            'Expected Result': 'Should create a file with the scraped content',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        },
        {
            'Test Case ID': 'TC005',
            'Test Case Description': 'Test TOPICS list definition',
            'Test Steps': '1. Check TOPICS list type\n2. Verify list is not empty\n3. Verify all elements are strings',
            'Expected Result': 'TOPICS should be a non-empty list of strings',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        }
    ]

    # Create DataFrame
    df = pd.DataFrame(test_cases, columns=columns)

    # Create Excel file
    excel_path = os.path.join(os.path.dirname(__file__), 'scraper_rtm.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"RTM created at: {excel_path}")

def update_rtm_status(test_case_id, status, actual_result='', comments=''):
    """Update the status of a test case in the RTM"""
    excel_path = os.path.join(os.path.dirname(__file__), 'scraper_rtm.xlsx')
    
    if not os.path.exists(excel_path):
        print("RTM file not found!")
        return

    df = pd.read_excel(excel_path)
    
    # Find the test case and update its status
    mask = df['Test Case ID'] == test_case_id
    if not any(mask):
        print(f"Test case {test_case_id} not found!")
        return

    df.loc[mask, 'Status'] = status
    df.loc[mask, 'Actual Result'] = actual_result
    df.loc[mask, 'Comments'] = comments
    df.loc[mask, 'Date Executed'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Save the updated RTM
    df.to_excel(excel_path, index=False)
    print(f"Updated RTM for test case {test_case_id}")

if __name__ == '__main__':
    create_rtm() 