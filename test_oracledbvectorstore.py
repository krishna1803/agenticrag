import argparse
import json
from OracleDBVectorStore import OracleDBVectorStore
import time
import sys
import yaml
from pathlib import Path


def check_credentials():
    """Check if Oracle DB credentials are configured in config.yaml"""
    try:
        config_path = Path("config.yaml")
        if not config_path.exists():
            print("✗ config.yaml not found.")
            return False
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if not config:
            print("✗ config.yaml is empty or invalid YAML.")
            return False
            
        # Check for Oracle DB credentials
        if not config.get("ORACLE_DB_USERNAME"):
            print("✗ ORACLE_DB_USERNAME not found in config.yaml")
            return False
            
        if not config.get("ORACLE_DB_PASSWORD"):
            print("✗ ORACLE_DB_PASSWORD not found in config.yaml")
            return False
            
        if not config.get("ORACLE_DB_DSN"):
            print("✗ ORACLE_DB_DSN not found in config.yaml")
            return False
            
        print("✓ Oracle DB credentials found in config.yaml")
        return True
    except Exception as e:
        print(f"✗ Error checking credentials: {str(e)}")
        return False

def test_connection():
    """Test connection to Oracle DB"""
    print("Testing Oracle DB connection...")
    try:
        store = OracleDBVectorStore()
        print("✓ Connection successful!")
        return store
    except Exception as e:
        print(f"✗ Connection failed: {str(e)}")
        return None    
    
def check_collection_stats(store):
    """Check statistics for each collection including total chunks and latest insertion"""
    if not store:
        print("Skipping collection stats check as connection failed")
        return
    
    print("\n=== Collection Statistics ===")
    
    collections = {
        "PDF Collection": "pdf_documents",
        "Repository Collection": "repository_documents",
        "Web Knowledge Base": "web_documents",
        "General Knowledge": "general_knowledge"
    }
    
    for name, collection in collections.items():
        try:
            # Get total count
            count = store.get_collection_count(collection)
            print(f"\n{name}:")
            print(f"Total chunks: {count}")
            
            # Get latest insertion if collection is not empty
            if count > 0:
                latest = store.get_latest_chunk(collection)
                print("Latest chunk:")
                print(f"  Content: {latest['content'][:150]}..." if len(latest['content']) > 150 else f"  Content: {latest['content']}")
                
                # Print metadata
                if isinstance(latest['metadata'], str):
                    try:
                        metadata = json.loads(latest['metadata'])
                    except:
                        metadata = {"source": latest['metadata']}
                else:
                    metadata = latest['metadata']
                
                source = metadata.get('source', 'Unknown')
                print(f"  Source: {source}")
                
                # Print other metadata based on collection type
                if collection == "pdf_documents" and 'page' in metadata:
                    print(f"  Page: {metadata['page']}")
                elif collection == "repository_documents" and 'file_path' in metadata:
                    print(f"  File: {metadata['file_path']}")
                elif collection == "web_documents" and 'title' in metadata:
                    print(f"  Title: {metadata['title']}")
            else:
                print("No chunks found in this collection.")
                
        except Exception as e:
            print(f"Error checking {name}: {str(e)}")
                  

def check_similarity_search(store, query):
    """Perform a similarity search on the Oracle DB Vector Store"""
    if not store:
        print("Skipping similarity search as connection failed")
        return
    
    print(f"\n=== Similarity Search for Query: '{query}' ===")
    
    try:
        results = store.similarity_search(query, k=5)  # Get top 5 results
        if not results:
            print("No results found for the query.")
            return
        
        for i, result in enumerate(results):
            content = result['content'][:150] + '...' if len(result['content']) > 150 else result['content']
            print(f"Result {i+1}:")
            print(f"  Content: {content}")
            print(f"  Metadata: {result['metadata']}")
    except Exception as e:
        print(f"Error during similarity search: {str(e)}")
        

def main():
    parser = argparse.ArgumentParser(description="Test Oracle DB Vector Store")
    parser.add_argument("--query", default="machine learning", help="Query to use for testing")
    parser.add_argument("--stats-only", action="store_true", help="Only show collection statistics without inserting test data")
    
    args = parser.parse_args()
    
    print("=== Oracle DB Vector Store Test ===\n")
    
    # Check if oracledb is installed
    try:
        import oracledb
        print("✓ oracledb package is installed")
    except ImportError:
        print("✗ oracledb package is not installed.")
        print("Please install it with: pip install oracledb")
        sys.exit(1)
    
    # Check if sentence_transformers is installed
    try:
        import sentence_transformers
        print("✓ sentence_transformers package is installed")
    except ImportError:
        print("✗ sentence_transformers package is not installed.")
        print("Please install it with: pip install sentence-transformers")
        sys.exit(1)
    
    # Check if credentials are configured
    if not check_credentials():
        print("\n✗ Oracle DB credentials not properly configured in config.yaml")
        print("Please update config.yaml with the following:")
        print("""
        ORACLE_DB_USERNAME: ADMIN
        ORACLE_DB_PASSWORD: your_password_here
        ORACLE_DB_DSN: your_connection_string_here
        """)
        sys.exit(1)
    
    # Test connection
    store = test_connection()
    
    # Check collection statistics
    check_collection_stats(store) 
    
    # Perform similarity search if not just showing stats
    check_similarity_search(store, args.query)   
    
    print("\n=== Test Completed ===")

if __name__ == "__main__":
    main() 