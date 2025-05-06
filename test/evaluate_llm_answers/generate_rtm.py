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
            'Test Case Description': 'Test BERTScore calculation',
            'Test Steps': '1. Mock BERTScore function\n2. Calculate scores for sample data\n3. Verify precision, recall, and F1 scores',
            'Expected Result': 'Should return correct precision, recall, and F1 scores for each answer pair',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        },
        {
            'Test Case ID': 'TC002',
            'Test Case Description': 'Test cosine similarity calculation',
            'Test Steps': '1. Mock sentence transformer\n2. Calculate embeddings\n3. Compute cosine similarity\n4. Verify results',
            'Expected Result': 'Should return valid cosine similarity scores between -1 and 1',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        },
        {
            'Test Case ID': 'TC003',
            'Test Case Description': 'Test fuzzy similarity calculation',
            'Test Steps': '1. Calculate fuzzy scores for answer pairs\n2. Verify score ranges',
            'Expected Result': 'Should return fuzzy match scores between 0 and 100',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        },
        {
            'Test Case ID': 'TC004',
            'Test Case Description': 'Test data loading and cleaning',
            'Test Steps': '1. Load sample CSV data\n2. Clean data by removing null values\n3. Verify data structure',
            'Expected Result': 'Should load and clean data correctly, maintaining required columns',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        },
        {
            'Test Case ID': 'TC005',
            'Test Case Description': 'Test output file generation',
            'Test Steps': '1. Add scores to DataFrame\n2. Generate detailed scores file\n3. Generate summary file\n4. Verify file creation',
            'Expected Result': 'Should create both detailed scores and summary files with correct data',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        }
    ]

    # Create DataFrame
    df = pd.DataFrame(test_cases, columns=columns)

    # Create Excel file
    excel_path = os.path.join(os.path.dirname(__file__), 'evaluate_llm_answers_rtm.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"RTM created at: {excel_path}")

def update_rtm_status(test_case_id, status, actual_result='', comments=''):
    """Update the status of a test case in the RTM"""
    excel_path = os.path.join(os.path.dirname(__file__), 'evaluate_llm_answers_rtm.xlsx')
    
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