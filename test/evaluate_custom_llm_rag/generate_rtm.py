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
            'Test Case Description': 'Test CSV path resolution',
            'Test Steps': '1. Mock Path object\n2. Call get_csv_path\n3. Verify path resolution',
            'Expected Result': 'Should return correct path to qa_with_context.csv',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        },
        {
            'Test Case ID': 'TC002',
            'Test Case Description': 'Test CSV data loading',
            'Test Steps': '1. Create test CSV file\n2. Call load_csv_data\n3. Verify data loading and error handling',
            'Expected Result': 'Should load data correctly and handle file not found errors',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        },
        {
            'Test Case ID': 'TC003',
            'Test Case Description': 'Test text truncation',
            'Test Steps': '1. Test short text\n2. Test long text\n3. Verify truncation behavior',
            'Expected Result': 'Should preserve short text and truncate long text with ellipsis',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        },
        {
            'Test Case ID': 'TC004',
            'Test Case Description': 'Test prompt building',
            'Test Steps': '1. Create test inputs\n2. Call build_prompt\n3. Verify prompt structure',
            'Expected Result': 'Should create properly formatted prompt with all required components',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        },
        {
            'Test Case ID': 'TC005',
            'Test Case Description': 'Test JSON extraction',
            'Test Steps': '1. Test valid JSON\n2. Test invalid JSON\n3. Test malformed JSON\n4. Verify extraction',
            'Expected Result': 'Should extract valid JSON and handle invalid/malformed JSON gracefully',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        },
        {
            'Test Case ID': 'TC006',
            'Test Case Description': 'Test row evaluation',
            'Test Steps': '1. Mock LLM response\n2. Test new row evaluation\n3. Test already evaluated row\n4. Verify results',
            'Expected Result': 'Should evaluate new rows and skip already evaluated ones',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        }
    ]

    # Create DataFrame
    df = pd.DataFrame(test_cases, columns=columns)

    # Create Excel file
    excel_path = os.path.join(os.path.dirname(__file__), 'evaluate_custom_llm_rag_rtm.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"RTM created at: {excel_path}")

def update_rtm_status(test_case_id, status, actual_result='', comments=''):
    """Update the status of a test case in the RTM"""
    excel_path = os.path.join(os.path.dirname(__file__), 'evaluate_custom_llm_rag_rtm.xlsx')
    
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