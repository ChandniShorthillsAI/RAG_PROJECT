import unittest
import sys
import os
from test_evaluate_llm_answers import TestEvaluateLLMAnswers
from generate_rtm import update_rtm_status, create_rtm

def run_tests():
    # First ensure RTM exists
    create_rtm()
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEvaluateLLMAnswers)
    
    # Create test runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run tests and collect results
    result = runner.run(suite)
    
    # Update RTM based on test results
    test_methods = [
        'test_bert_score_calculation',
        'test_cosine_similarity_calculation',
        'test_fuzzy_similarity_calculation',
        'test_data_loading_and_cleaning',
        'test_output_file_generation'
    ]
    
    for i, test_method in enumerate(test_methods, 1):
        test_id = f"TC00{i}"
        
        # Find the test result
        test_result = None
        for test, error in result.failures:
            if test._testMethodName == test_method:
                test_result = ('Failed', str(error), 'Test failed')
                break
                
        for test, error in result.errors:
            if test._testMethodName == test_method:
                test_result = ('Error', str(error), 'Test error')
                break
        
        if not test_result:
            test_result = ('Passed', 'Test passed successfully', '')
            
        status, actual_result, comments = test_result
        update_rtm_status(test_id, status, actual_result, comments)

if __name__ == '__main__':
    run_tests() 