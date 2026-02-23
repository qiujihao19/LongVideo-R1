#!/usr/bin/env python
import json
import requests
import fire
import logging
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_python(
    url: str = None,
    trajectory_id: str = "test-python-001",
):
    """Test Python code execution"""
    
    print("--- Testing 1 ---")
    action = """<think>dahdohawdaw</think><tool>video_qa((1,1,1),\"What shape does the protagonist in the video make when he spreads seasoning on the bread?\")</tool>"""
    print(_send_test_request(url, trajectory_id, action, "get_caption"))
    
    # print("--- Testing 2 ---")
    # action = """<think>dahdohawdaw</think><tool>get_caption((1,3,3))</tool>"""
    # print(_send_test_request(url, trajectory_id, action, "get_caption"))
    
    # print("--- Testing 3 ---")
    # action = """<think>dahdohawdaw</think><tool>get_caption((1,3))</tool>"""
    # print(_send_test_request(url, trajectory_id, action, "get_caption"))
    
    # print("--- Testing 4 ---")
    # action = """<think>dahdohawdaw</think><answer>A</answer>"""
    # print(_send_test_request(url, trajectory_id, action, "get_caption"))
    return True
    
    
def _send_test_request(url, trajectory_id, action, test_name):
    """Helper function to send test requests and process responses"""
    logger.info(f"Testing {test_name} code execution...")
    
    # Use server API
    payload = {
        "trajectory_ids": [trajectory_id],
        "actions": [action],
        "extra_fields": [{'video_uid':'N03ddXnwAls', 'data_source': 'videocaption_cgbench','width':5,'fps':25.0}]
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise exception for error status codes
        
        result = response.json()
        logger.info(f"Response received for {test_name} test")
        
        # Print observation
        if "observations" in result and len(result["observations"]) > 0:
            observation = result["observations"][0]
            logger.info(f"\n--- {test_name} Result ---\n{observation}\n")
        else:
            logger.error(f"No observation found in response for {test_name}")
        
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": str(e)}

def main():
    """Main entry point for the test script
    Run with:
        python -m verl_tool.servers.tests.test_python_code_tool python --url=http://localhost:5000/get_observation
    """
    fire.Fire({
        "python": test_python,
    })

if __name__ == "__main__":
    main()
