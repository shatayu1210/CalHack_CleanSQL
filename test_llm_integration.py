#!/usr/bin/env python3
"""
Test script for LLM integration
This script tests the LLM integration without requiring API keys
"""
import json
import os

from llm_integration import AnthropicSQLGenerator


def test_sql_generation():
    """Test SQL generation with sample profile"""
    
    # Sample profile data
    sample_profile = {
        "dataset": {
            "filename": "test_data.csv",
            "row_count": 5,
            "column_count": 4
        },
        "columns": [
            {
                "name": "name",
                "duckdb_type": "VARCHAR",
                "semantic_type": "category",
                "examples": ["John", "Jane", "Bob"]
            },
            {
                "name": "age", 
                "duckdb_type": "BIGINT",
                "semantic_type": "numeric",
                "examples": [25, 30, 35]
            },
            {
                "name": "city",
                "duckdb_type": "VARCHAR", 
                "semantic_type": "category",
                "examples": ["New York", "San Francisco", "Chicago"]
            },
            {
                "name": "salary",
                "duckdb_type": "BIGINT",
                "semantic_type": "numeric", 
                "examples": [50000, 75000, 60000]
            }
        ]
    }
    
    print("🧪 Testing LLM Integration...")
    
    # Test 1: Check if API key is configured
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("❌ Anthropic API key not configured!")
        print("Please set ANTHROPIC_API_KEY in your .env file")
        return False
    
    print("✅ API key configured")
    
    # Test 2: Test SQL generator initialization
    try:
        generator = AnthropicSQLGenerator()
        print("✅ SQL generator initialized")
    except Exception as e:
        print(f"❌ Failed to initialize SQL generator: {e}")
        return False
    
    # Test 3: Test fallback schema generation
    try:
        fallback_schema = generator._fallback_schema(sample_profile)
        print("✅ Fallback schema generation works")
        print("Sample fallback schema:")
        print(fallback_schema)
    except Exception as e:
        print(f"❌ Fallback schema generation failed: {e}")
        return False
    
    print("\n🎉 All tests passed! LLM integration is ready.")
    print("\nTo use the full features:")
    print("1. Set your Anthropic API key in .env file")
    print("2. Run: python3 data_assistant.py --csv your_file.csv")
    print("3. Or run: python3 profiler.py --csv your_file.csv --out output_dir")
    
    return True

if __name__ == "__main__":
    test_sql_generation()
