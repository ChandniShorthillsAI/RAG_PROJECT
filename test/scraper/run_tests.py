import unittest
import sys
import os
from test_scraper import TestScraper
from generate_rtm import update_rtm_status

def run_tests():
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestScraper)
    
    # Create test runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run tests and collect results
    result = runner.run(suite)
    
    # Update RTM based on test results
    for test_case in result.testsRun:
        test_name = test_case._testMethodName
        test_id = f"TC00{test_name[-1]}"  # Extract test case ID from method name
        
        if test_case in result.failures:
            status = "Failed"
            actual_result = str(result.failures[0][1])
            comments = "Test failed"
        elif test_case in result.errors:
            status = "Error"
            actual_result = str(result.errors[0][1])
            comments = "Test error"
        else:
            status = "Passed"
            actual_result = "Test passed successfully"
            comments = ""
        
        update_rtm_status(test_id, status, actual_result, comments)

if __name__ == '__main__':
    run_tests() 