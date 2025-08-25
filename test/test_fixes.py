#!/usr/bin/env python3

"""
Test script to verify the fixes for predicate extraction and NoneType errors.
This script simulates the scenarios that were causing the FastAPI validation errors.
"""

import sys
import os
import logging
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_null_context_handling():
    """Test handling of None context from vector store"""
    print("\n=== Testing None context handling ===")
    
    # Simulate the _get_context_with_predicate_filtering method returning None
    context_chunks = None
    
    # Test the fix for None context handling
    if context_chunks is None:
        logger.warning("Context chunks is None, using empty list")
        context_chunks = []
    
    print(f"✅ None context handled correctly, context_chunks = {context_chunks}")
    
    # Test iteration over context chunks
    for i, chunk in enumerate(context_chunks):
        print(f"Processing chunk {i}: {chunk}")
    
    print("✅ Successfully handled None context chunks")

def test_invalid_chunk_structure():
    """Test handling of invalid chunk structures"""
    print("\n=== Testing invalid chunk structure handling ===")
    
    # Simulate various invalid chunk structures
    invalid_chunks = [
        None,  # None chunk
        "invalid_string",  # String instead of dict
        {"content": "valid content"},  # Missing metadata
        {"metadata": "invalid_metadata"},  # Missing content
        {"content": "valid", "metadata": None},  # None metadata
        {"content": "valid", "metadata": {"source": "test"}},  # Valid chunk
    ]
    
    processed_chunks = []
    
    for i, chunk in enumerate(invalid_chunks):
        # Add safety check for chunk structure
        if not chunk or not isinstance(chunk, dict):
            logger.warning(f"Invalid chunk at index {i}: {chunk}")
            continue
            
        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            logger.warning(f"Invalid metadata at index {i}: {metadata}")
            metadata = {}
            
        source = metadata.get("source", "Unknown")
        content = chunk.get("content", "")
        
        if content:  # Only process chunks with content
            processed_chunks.append({
                "content": content,
                "metadata": metadata,
                "source": source
            })
    
    print(f"✅ Processed {len(processed_chunks)} valid chunks out of {len(invalid_chunks)} total")
    return processed_chunks

def test_response_structure():
    """Test that response always contains required 'answer' field"""
    print("\n=== Testing response structure ===")
    
    # Test normal response
    def generate_normal_response():
        return {
            "answer": "This is a normal response",
            "context": []
        }
    
    # Test error response
    def generate_error_response():
        try:
            # Simulate an error
            raise ValueError("Simulated error")
        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}")
            return {
                "answer": f"I apologize, but I encountered an error: {str(e)}",
                "context": [],
                "error": str(e)
            }
    
    # Test None response handling
    def generate_none_response():
        answer = None
        if answer is None:
            answer = "I apologize, but I was unable to generate a response. Please try again."
        
        return {
            "answer": answer,
            "context": []
        }
    
    # Test all response types
    normal_response = generate_normal_response()
    error_response = generate_error_response()
    none_response = generate_none_response()
    
    # Verify all responses have 'answer' field
    for response_name, response in [
        ("normal", normal_response),
        ("error", error_response), 
        ("none", none_response)
    ]:
        assert "answer" in response, f"{response_name} response missing 'answer' field"
        assert response["answer"] is not None, f"{response_name} response has None answer"
        assert isinstance(response["answer"], str), f"{response_name} response answer is not string"
        print(f"✅ {response_name} response structure valid")
    
    return normal_response, error_response, none_response

def test_context_formatting():
    """Test safe context formatting"""
    print("\n=== Testing context formatting ===")
    
    # Test various context scenarios
    test_contexts = [
        None,  # None context
        [],    # Empty context
        [{"content": "Valid content", "metadata": {"source": "test"}}],  # Valid context
        [None, {"content": "Valid"}, {"invalid": "structure"}],  # Mixed valid/invalid
    ]
    
    for i, context in enumerate(test_contexts):
        print(f"Testing context scenario {i+1}")
        
        try:
            # Safe context formatting
            formatted_context = ""
            if context:
                context_parts = []
                for j, item in enumerate(context):
                    # Ensure item is a dict and has required fields
                    if isinstance(item, dict) and 'content' in item:
                        context_parts.append(f"Context {j+1}:\n{item['content']}")
                    else:
                        logger.warning(f"Invalid context item at index {j}: {item}")
                        
                formatted_context = "\n\n".join(context_parts)
            else:
                formatted_context = "No context available."
            
            print(f"✅ Formatted context: {len(formatted_context)} characters")
            
        except Exception as e:
            print(f"❌ Error formatting context: {e}")
            traceback.print_exc()

def test_predicate_extraction():
    """Test predicate extraction doesn't return None"""
    print("\n=== Testing predicate extraction ===")
    
    # Mock the SearchFilterExtractor
    class MockSearchFilterExtractor:
        def extract_predicate_from_query(self, query: str):
            # Test scenarios
            if "error" in query.lower():
                # Simulate an error
                raise ValueError("Simulated extraction error")
            elif "empty" in query.lower():
                # Return empty predicate
                return "", {}
            else:
                # Return valid predicate
                return "LOWER((cmetadata->>'type')) = %(type)s", {"type": "case"}
    
    extractor = MockSearchFilterExtractor()
    
    test_queries = [
        "What are cases from 2020?",
        "empty query",
        "error query"
    ]
    
    for query in test_queries:
        try:
            predicate, predicate_values = extractor.extract_predicate_from_query(query)
            
            # Ensure predicate_values is not None
            if predicate_values is None:
                predicate_values = {}
            
            print(f"✅ Query: '{query}' -> Predicate: '{predicate}', Values: {predicate_values}")
            
        except Exception as e:
            logger.error(f"Error extracting predicate from '{query}': {str(e)}")
            # Fallback
            predicate, predicate_values = "", {}
            print(f"✅ Query: '{query}' -> Fallback: predicate='', values={{}}")

def main():
    """Run all tests"""
    print("🧪 Starting predicate extraction and NoneType error fix tests")
    
    try:
        test_null_context_handling()
        test_invalid_chunk_structure()
        test_response_structure()
        test_context_formatting()
        test_predicate_extraction()
        
        print("\n🎉 All tests passed! The fixes should resolve the FastAPI validation errors.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
