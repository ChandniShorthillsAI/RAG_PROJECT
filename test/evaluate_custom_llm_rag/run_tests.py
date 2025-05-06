import unittest
import sys
import os
from test_evaluate_custom_llm_rag import TestEvaluateCustomLLMRag
from generate_rtm import create_rtm, update_rtm_status

def run_tests():
    # Create RTM if it doesn't exist
    create_rtm()
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEvaluateCustomLLMRag)
    
    # Run tests and collect results
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Update RTM based on test results
    test_cases = {
        'test_get_csv_path': 'TC001',
        'test_load_csv_data': 'TC002',
        'test_truncate': 'TC003',
        'test_build_prompt': 'TC004',
        'test_extract_json': 'TC005',
        'test_evaluate_row': 'TC006'
    }
    
    # Update RTM for each test case
    for test_name, test_id in test_cases.items():
        test_result = next((result for result in results.failures + results.errors 
                          if result[0]._testMethodName == test_name), None)
        
        if test_result:
            status = 'Failed'
            actual_result = str(test_result[1])
            comments = 'Test failed'
        else:
            status = 'Passed'
            actual_result = 'Test passed successfully'
            comments = ''
        
        update_rtm_status(test_id, status, actual_result, comments)
    
    # Print summary
    print("\nTest Summary:")
    print(f"Total Tests: {results.testsRun}")
    print(f"Passed: {results.testsRun - len(results.failures) - len(results.errors)}")
    print(f"Failed: {len(results.failures)}")
    print(f"Errors: {len(results.errors)}")

if __name__ == '__main__':
    run_tests() 