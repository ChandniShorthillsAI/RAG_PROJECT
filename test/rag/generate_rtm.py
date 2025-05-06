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
            'Test Case Description': 'Test text cleaning functionality',
            'Test Steps': '1. Create test text with citations and extra spaces\n2. Call clean_text function\n3. Verify the output',
            'Expected Result': 'Text should be cleaned of citations and extra spaces',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        },
        {
            'Test Case ID': 'TC002',
            'Test Case Description': 'Test document loading and chunking',
            'Test Steps': '1. Create test document\n2. Call load_and_chunk_text\n3. Verify document structure and content',
            'Expected Result': 'Should return list of Document objects with proper content and metadata',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        },
        {
            'Test Case ID': 'TC003',
            'Test Case Description': 'Test embedding and storing functionality',
            'Test Steps': '1. Create test documents\n2. Mock embedding model\n3. Call embed_and_store\n4. Verify FAISS operations',
            'Expected Result': 'Documents should be embedded and stored in FAISS index',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        },
        {
            'Test Case ID': 'TC004',
            'Test Case Description': 'Test context retrieval functionality',
            'Test Steps': '1. Mock FAISS and embedding model\n2. Call retrieve_context\n3. Verify retrieved content',
            'Expected Result': 'Should return concatenated context from top k documents',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        },
        {
            'Test Case ID': 'TC005',
            'Test Case Description': 'Test pipeline initialization',
            'Test Steps': '1. Initialize RagPipeline with test parameters\n2. Verify all components are properly initialized',
            'Expected Result': 'Pipeline should be initialized with correct parameters and components',
            'Actual Result': '',
            'Status': 'Not Executed',
            'Date Executed': '',
            'Comments': ''
        }
    ]

    # Create DataFrame
    df = pd.DataFrame(test_cases, columns=columns)

    # Create Excel file
    excel_path = os.path.join(os.path.dirname(__file__), 'rag_pipeline_rtm.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"RTM created at: {excel_path}")

def update_rtm_status(test_case_id, status, actual_result='', comments=''):
    """Update the status of a test case in the RTM"""
    excel_path = os.path.join(os.path.dirname(__file__), 'rag_pipeline_rtm.xlsx')
    
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