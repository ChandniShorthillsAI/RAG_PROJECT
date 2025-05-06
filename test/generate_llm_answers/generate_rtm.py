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
            'Test Case Description': 'Test stream_ollama function',
            'Test Steps': '1. Mock Ollama API response\n2. Test successful streaming\n3. Test error handling',
            'Expected Result': 'Should return concatenated response for success case and empty string for error case',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        },
        {
            'Test Case ID': 'TC002',
            'Test Case Description': 'Test get_done_questions function',
            'Test Steps': '1. Test with empty file\n2. Test with existing data',
            'Expected Result': 'Should return empty set for new file and set of questions for existing file',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        },
        {
            'Test Case ID': 'TC003',
            'Test Case Description': 'Test append_to_csv function',
            'Test Steps': '1. Test appending to empty file\n2. Test appending to existing file',
            'Expected Result': 'Should create new file with header and append data correctly',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        },
        {
            'Test Case ID': 'TC004',
            'Test Case Description': 'Test generate_llm_answers function',
            'Test Steps': '1. Mock LLM response\n2. Run generation process\n3. Verify output file',
            'Expected Result': 'Should process all questions and create output file with correct columns',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        }
    ]

    # Create DataFrame
    df = pd.DataFrame(test_cases, columns=columns)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

    # Save to Excel
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generate_llm_answers_rtm.xlsx')
    df.to_excel(output_file, index=False)
    print(f"✅ RTM created at: {output_file}")

def update_rtm_status(test_id, status, actual_result="", comments=""):
    """Update the status of a test case in the RTM"""
    rtm_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generate_llm_answers_rtm.xlsx')
    
    if not os.path.exists(rtm_file):
        print("❌ RTM file not found. Please create it first.")
        return
    
    # Read existing RTM
    df = pd.read_excel(rtm_file)
    
    # Update status
    mask = df['Test Case ID'] == test_id
    if mask.any():
        df.loc[mask, 'Status'] = status
        df.loc[mask, 'Actual Result'] = actual_result
        df.loc[mask, 'Comments'] = comments
        df.loc[mask, 'Date Executed'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save updated RTM
        df.to_excel(rtm_file, index=False)
        print(f"✅ Updated status for {test_id}")
    else:
        print(f"❌ Test case {test_id} not found in RTM")

if __name__ == '__main__':
    create_rtm() 