#!/usr/bin/env python3
"""
Performance test script for PostgresVectorStore
This script helps identify performance bottlenecks in the similarity search
"""

import time
import logging
import sys
import os

# Add the parent directory (project root) to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PostgresVectorStore import PostgresVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def test_performance():
    """Test the performance of PostgresVectorStore similarity search"""
    
    print("="*80)
    print("POSTGRESQL VECTOR STORE PERFORMANCE TEST")
    print("="*80)
    
    try:
        # Initialize the vector store
        print("\n1. Initializing PostgresVectorStore...")
        start_time = time.time()
        
        vector_store = PostgresVectorStore(collection_name="PDF Collection")
        
        init_duration = time.time() - start_time
        print(f"✓ Initialization completed in {init_duration:.2f} seconds")
        
        # Optimize database (create indexes)
        print("\n2. Optimizing database (creating indexes)...")
        start_time = time.time()
        
        vector_store.optimize_database()
        
        optimize_duration = time.time() - start_time
        print(f"✓ Database optimization completed in {optimize_duration:.2f} seconds")
        
        # Test similarity search
        test_queries = [
            "What is the Mabo case about?",
            "legal rights and land ownership",
            "High Court decision",
            "traditional land claims",
            "racial discrimination"
        ]
        
        print(f"\n3. Testing similarity search with {len(test_queries)} queries...")
        
        total_search_time = 0
        successful_searches = 0
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: '{query}'")
            start_time = time.time()
            
            try:
                results = vector_store.query_pdf_collection(query, n_results=3)
                search_duration = time.time() - start_time
                total_search_time += search_duration
                
                if results:
                    successful_searches += 1
                    print(f"   ✓ Found {len(results)} results in {search_duration:.2f} seconds")
                    
                    # Show first result preview
                    first_result = results[0]
                    content_preview = first_result['content'][:100] + "..." if len(first_result['content']) > 100 else first_result['content']
                    similarity_score = first_result['metadata'].get('similarity_score', 'N/A')
                    print(f"   → Top result (score: {similarity_score}): {content_preview}")
                else:
                    print(f"   ⚠ No results found in {search_duration:.2f} seconds")
                    
            except Exception as e:
                search_duration = time.time() - start_time
                print(f"   ✗ Error in {search_duration:.2f} seconds: {str(e)}")
        
        # Summary
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        print(f"Initialization time:     {init_duration:.2f} seconds")
        print(f"Database optimization:   {optimize_duration:.2f} seconds")
        print(f"Total search time:       {total_search_time:.2f} seconds")
        print(f"Average search time:     {total_search_time/len(test_queries):.2f} seconds")
        print(f"Successful searches:     {successful_searches}/{len(test_queries)}")
        print(f"Success rate:            {successful_searches/len(test_queries)*100:.1f}%")
        
        if total_search_time > 10:
            print("\n⚠ WARNING: Average search time is high. Consider:")
            print("  - Checking database indexes")
            print("  - Verifying network connectivity")
            print("  - Ensuring sufficient database resources")
            print("  - Checking embedding model performance")
        elif total_search_time < 3:
            print("\n✓ GOOD: Search performance is acceptable")
        
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_performance()
    exit(0 if success else 1)
