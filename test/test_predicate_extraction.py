#!/usr/bin/env python3
"""
Simple test script to validate predicate extraction and PostgresVectorStore integration.
"""

import sys
import os

# Add the parent directory (project root) to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_filter_extractor import SearchFilterExtractor

def test_predicate_extraction():
    """Test predicate extraction with various query patterns."""
    
    extractor = SearchFilterExtractor()
    
    test_queries = [
        "Find NSW cases from 2023 about negligence",
        "Show me legislation from Victoria",
        "Federal court decisions in 2022",
        "Contract law cases",
        "Employment legislation since 2020",
        "Simple query without filters"
    ]
    
    print("🧪 Testing Predicate Extraction\n")
    
    for query in test_queries:
        print(f"Query: '{query}'")
        
        # Extract filters
        filters = extractor.extract_simple_filters_from_query(query)
        print(f"  Filters: {filters}")
        
        # Generate predicate
        predicate, values = extractor.extract_predicate_from_query(query)
        print(f"  Predicate: '{predicate}'")
        print(f"  Values: {values}")
        print()

def test_sql_predicate_format():
    """Test that the SQL predicate format is compatible with PGVector."""
    
    extractor = SearchFilterExtractor()
    
    print("🔍 Testing SQL Predicate Format\n")
    
    # Test a query that should generate multiple filters
    query = "Find NSW cases from 2023"
    predicate, values = extractor.extract_predicate_from_query(query)
    
    print(f"Query: '{query}'")
    print(f"Generated Predicate: '{predicate}'")
    print(f"Parameter Values: {values}")
    
    # Show what the final SQL would look like
    if predicate:
        sample_sql = f"""
        SELECT document, cmetadata, embedding <=> %s::vector(384) AS distance
        FROM langchain_pg_embedding
        WHERE 1=1 AND {predicate}
        ORDER BY embedding <=> %s::vector(384)
        LIMIT %s
        """
        print(f"\nSample SQL with predicate:")
        print(sample_sql)
        print(f"Parameters would be: [embedding_vector, embedding_vector, limit, {values}]")
    else:
        print("\nNo predicate generated - would use standard similarity search")

if __name__ == "__main__":
    print("=" * 60)
    print("PREDICATE EXTRACTION TEST")
    print("=" * 60)
    
    test_predicate_extraction()
    
    print("=" * 60)
    
    test_sql_predicate_format()
    
    print("=" * 60)
    print("✅ Test completed!")
