#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(__file__))

from cli_agent import generate_bash

def test_problematic_queries():
    test_cases = [
        "list all files in my parent directory",
        "show contents of parent directory", 
        "create a python file",
        "show free memory",
        "list files in current directory"
    ]
    
    print("Testing improved CLI agent...")
    print("=" * 50)
    
    for query in test_cases:
        print(f"\nQuery: {query}")
        try:
            result = generate_bash(query)
            print(f"Generated: {result}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_problematic_queries()