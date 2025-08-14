from __future__ import annotations

import time
import array
import numpy as np
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_postgres import PGVector
from sqlalchemy import text, create_engine, inspect

from langchain_community.embeddings import HuggingFaceEmbeddings

#import psycopg2
#from psycopg2.extras import Json
import argparse
from pathlib import Path
import yaml
import json

#from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaEmbeddings

# Use consistent embedding model across the class
model_name = "sentence-transformers/all-MiniLM-L6-v2"
# Global embeddings instance - will be replaced by instance-specific one
embeddings = HuggingFaceEmbeddings(model_name=model_name)



# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class PostgresVectorStore(VectorStore):
    _LANGCHAIN_DEFAULT_COLLECTION_NAME = "PDF Collection"
    
    def __init__(
        self,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        embedding_function: Optional[Embeddings] = None,
        persist_directory: Optional[str] = None,
        client_settings: Optional[Any] = None,
        collection_metadata: Optional[Dict] = None,
        client: Optional[Any] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        verbose: Optional[bool] = False,
        vectorstore: Optional[Any] = None,
        cursor: Optional[Any] = None,
        connectionstring: Optional[str] = None,
    ) -> None:
        self.verbose = verbose
    

        self.collection_name = collection_name
        # Use consistent embedding model - same as the one used for storing embeddings
        self._embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.override_relevance_score_fn = relevance_score_fn
        
        # Load Postgres DB credentials from config_pg.yaml
        credentials = self._load_config()
        
        host = credentials.get("PG_HOST", "localhost")
        port = int(credentials.get("PG_PORT", "5432"))
        database = credentials.get("PG_DATABASE", "langchain")
        username = credentials.get("PG_USERNAME", "langchain")
        password = credentials.get("PG_PASSWORD", "langchain")

        if not host or not password:
            raise ValueError("PostgreSQL credentials not found in config_pg.yaml. Please set PG_HOST, PG_PORT, PG_DATABASE, PG_USERNAME, and PG_PASSWORD.")

        # Connect to the database
        try:
            #prepare connection string
            self.connectionstring = conn_string = f"postgresql+psycopg://{username}:{password}@{host}:{port}/{database}"  # Uses psycopg3!

            # Create engine with connection pooling for better performance
            self.engine = create_engine(
                self.connectionstring,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600
            )

            vector_store = PGVector(
                        embeddings=self._embedding_function,
                        collection_name=collection_name,
                        connection=self.connectionstring,
                        use_jsonb=True,
            ) 
            self.vectorstore = vector_store

            logging.info("PostgreSQL Collection Creation successful!")
            print("PostgreSQL Collection Creation successful!")
        except Exception as e:
            print("PostgreSQL Collection Creation failed!", e)
            raise

    def _load_config(self) -> Dict[str, str]:
        """Load configuration from config_pg.yaml"""
        try:
            config_path = Path("config_pg.yaml")
            if not config_path.exists():
                print("Warning: config.yaml not found. Using empty configuration.")
                return {}
                
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config if config else {}
        except Exception as e:
            print(f"Warning: Error loading config: {str(e)}")
            return {}
            
    def _sanitize_metadata(self, metadata: Dict) -> Dict:
        """Sanitize metadata to ensure all values are valid types for Oracle DB"""
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                # Convert list to string representation
                sanitized[key] = str(value)
            elif value is None:
                # Replace None with empty string
                sanitized[key] = ""
            else:
                # Convert any other type to string
                sanitized[key] = str(value)
        return sanitized
            
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        logging.info(f"add_texts")
        """Add texts to the vector store."""
        if not texts:
            return []
        # Prepare data for Postgres DB
        texts = list(texts)
        if metadatas is None:
            metadatas = [{}] * len(texts)
        elif len(metadatas) != len(texts):
            raise ValueError("Length of metadatas must match length of texts")  
        metadatas = [self._sanitize_metadata(m) for m in metadatas]
        ids = kwargs.get("ids", [f"text_{i}" for i in range(len(texts))])
        
        # Use the underlying connection (PGVector) to add texts directly
        self.connection.add_texts(texts, metadatas, **kwargs)
        
        logging.info(f"Added {len(texts)} texts to Postgres vector store")
        print(f"Added {len(texts)} texts to Postgres vector store")
        return ids                    
        
    def add_pdf_chunks(self, chunks: List[Dict[str, Any]], document_id: str):
        """Add chunks from a PDF document to the vector store"""
        if not chunks:
            return
        
        # Prepare data for Postgres DB
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [self._sanitize_metadata(chunk["metadata"]) for chunk in chunks]
        ids = [f"{document_id}_{i}" for i in range(len(chunks))]
        
        #Create Document Objects using texts and metadatas
        documents = [Document(page_content=text, metadata=metadata) for text, metadata in zip(texts, metadatas)]
        # Add documents to the vector store
        print("Adding documents to Postgres vector store...")
        print(f"Total documents to add: {len(documents)}")
        self.vectorstore.add_documents(documents, ids=ids)
        logging.info(f"Added {len(chunks)} chunks from document {document_id} to Postgres vector store")
        print(f"Added {len(chunks)} chunks from document {document_id} to Postgres vector store")
        
           
    @property
    def embeddings(self) -> Optional[Embeddings]:
        print("Getting embeddings from Postgres vector store...")
        return self._embedding_function

    def query_pdf_collection(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Query the PDF documents collection"""
        start_time = time.time()
        print("🔍 [Postgres] Querying PDF Collection")
        logging.info(f"Starting PDF collection query for: '{query[:50]}...' with n_results={n_results}")
        
        try:
            # Use your custom implementation directly instead of vectorstore's implementation
            formatted_results = self.similarity_search(query, k=n_results)
            
            duration = time.time() - start_time
            
            if not formatted_results:
                logging.warning(f"No results found for query in {duration:.2f}s")
                print("No results found for the query.")
                return []
                
            logging.info(f"🔍 [Postgres] Retrieved {len(formatted_results)} chunks from PDF Collection in {duration:.2f}s")
            print(f"🔍 [Postgres] Retrieved {len(formatted_results)} chunks from PDF Collection in {duration:.2f}s")
            return formatted_results
            
        except Exception as e:
            duration = time.time() - start_time
            logging.error(f"Error in query_pdf_collection after {duration:.2f}s: {str(e)}")
            print(f"Error in query_pdf_collection: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def query_general_collection(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        results = self.vectorstore.similarity_search(query, k=n_results)
        return self._convert_documents_to_dict(results)
    
    def query_repo_collection(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        results = self.vectorstore.similarity_search(query, k=n_results)
        return self._convert_documents_to_dict(results)

    def query_web_collection(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        results = self.vectorstore.similarity_search(query, k=n_results)
        return self._convert_documents_to_dict(results)
    
    def _convert_documents_to_dict(self, documents):
        """Convert LangChain Document objects to dictionaries compatible with the RAG agent"""
        result = []
        for doc in documents:
            result.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        return result 
     #
    # similarity_search
    #
    def similarity_search(
        self, query: str, k: int = 3
    ) -> List[Dict[str, Any]]:
        """Return docs most similar to query."""
        logging.info(f"Similarity Search for Postgres Vector Store")

        try:
            # First try using the built-in PGVector search which is more efficient
            documents = self.vectorstore.similarity_search(query, k=k)
            
            if documents:
                logging.info(f"🔍 [PostgresDB] Retrieved {len(documents)} chunks using PGVector search")
                return self._convert_documents_to_dict(documents)
            
            # Fallback to custom search if needed
            logging.info("No results from PGVector search, trying custom search...")
            return self._custom_similarity_search(query, k)
            
        except Exception as e:
            logging.error(f"Error in similarity_search: {str(e)}")
            # Fallback to custom search
            return self._custom_similarity_search(query, k)
    
    def _custom_similarity_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Custom similarity search implementation as fallback"""
        logging.info(f"Using custom similarity search for Postgres Vector Store")

        # Get the embedding for the query using the same model that created the embeddings
        query_embedding = self._embedding_function.embed_query(query)
        
        # Format the embedding vector as a PostgreSQL vector literal
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        embedding_dim = len(query_embedding)  # Get actual dimension

        formatted_results = []

        try:
            with self.engine.begin() as conn:
                
                print(f"Using embedding vector: {embedding_str[:50]}...")
                
                # First, try to determine the correct table structure
                # Check what tables exist for this collection
                collection_table = f"langchain_pg_embedding_{self.collection_name.lower().replace(' ', '_')}"
                
                # Try the standard PGVector table structure first
                search_query = f"""
                    SELECT document, cmetadata, embedding <=> :embedding_vector::vector({embedding_dim}) AS distance
                    FROM {collection_table}
                    ORDER BY distance
                    LIMIT :limit_count
                """

                print(f"Executing search query on table: {collection_table}")
                result = conn.execute(text(search_query), {"embedding_vector": embedding_str, "limit_count": k})
                rows = result.fetchall()
        
                # Format results
                formatted_results = []
                for row in rows:
                    content = row[0]
                    metadata = row[1] if row[1] else {}
                    distance = row[2]
                    
                    # Process metadata - either parse JSON or use as is
                    if isinstance(metadata, str):
                        try:
                            base_metadata = json.loads(metadata)
                        except json.JSONDecodeError:
                            base_metadata = {"raw_metadata": metadata}
                    elif metadata is None:
                        base_metadata = {}
                    else:
                        base_metadata = metadata
                    
                    # Add similarity score to metadata
                    enhanced_metadata = {
                        **base_metadata,
                        "similarity_score": float(distance)
                    }
                    
                    # Create the formatted result
                    result_item = {
                        "content": content,
                        "metadata": enhanced_metadata
                    }

                    print(f"Document Score: {distance}")
                    formatted_results.append(result_item)
                
                print(f"🔍 [PostgresDB] Retrieved {len(formatted_results)} chunks from PDF Collection")
                return formatted_results    
    
        except Exception as e:
            print(f"Error performing custom similarity search: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
        
    
     
    """def similarity_search(
        self, query: str, k: int = 3, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        results = self.vectorstore.similarity_search(query, k=k)
        print(f"Found {len(results)} results for query '{query}':")
        return results
    
    #Function to imlement similarity search with a retriever
    def similarity_search_with_retriever(self,query:str, k=3) -> List[Dict[str, Any]]:
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        results = retriever.invoke(query)
        print(f"Found {len(results)} results for query '{query}':")
        return results
    
    def as_retriever(self):
        # Return a retriever that uses this vector store for semantic search.
        from langchain.retrievers import VectorStoreRetriever
        
        return VectorStoreRetriever(
            vectorstore=self,
            search_type="similarity",
            search_kwargs={"k": 10}
        )"""
    
    @classmethod
    def from_texts(
        cls: Type[PostgresVectorStore],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> PostgresVectorStore:
        logging.info(f"from_texts")
        """Return VectorStore initialized from texts and embeddings."""
        raise NotImplementedError("from_texts method must be implemented...")

    def optimize_database(self):
        """Create indexes for better similarity search performance"""
        try:
            with self.engine.begin() as conn:
                collection_table = f"langchain_pg_embedding_{self.collection_name.lower().replace(' ', '_')}"
                
                # Create vector index using HNSW for fast similarity search
                index_query = f"""
                    CREATE INDEX IF NOT EXISTS {collection_table}_embedding_hnsw_idx 
                    ON {collection_table} 
                    USING hnsw (embedding vector_cosine_ops);
                """
                
                conn.execute(text(index_query))
                logging.info(f"Created HNSW index on {collection_table} for faster similarity search")
                
                # Create index on metadata for filtering
                metadata_index_query = f"""
                    CREATE INDEX IF NOT EXISTS {collection_table}_metadata_idx 
                    ON {collection_table} 
                    USING gin (cmetadata);
                """
                
                conn.execute(text(metadata_index_query))
                logging.info(f"Created GIN index on metadata for {collection_table}")
                
        except Exception as e:
            logging.warning(f"Could not create indexes: {str(e)}")

    def __del__(self):
        """Cleanup database connections when the object is destroyed"""
        try:
            if hasattr(self, 'engine'):
                self.engine.dispose()
                logging.info("Database connections disposed")
        except Exception as e:
            logging.warning(f"Error disposing database connections: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Manage Oracle DB vector store")
    parser.add_argument("--add", help="JSON file containing chunks to add")
    parser.add_argument("--query", help="Query to search for")
    
    args = parser.parse_args()
    store = PostgresVectorStore()
    
    if args.add:
        with open(args.add, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        store.add_pdf_chunks(chunks, document_id=args.add)
        print(f"✓ Added {len(chunks)} PDF chunks to Postgres vector store")
    elif args.query:
        results = store.similarity_search(args.query)
        if results:
            print(f"✓ Found {len(results)} results for query '{args.query}'")
            for i, result in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"Content: {result.page_content}")
                print(f"Metadata: {result.metadata}")
        else:
            print("No results found.")
        #Try with retriever
        retriever_results = store.similarity_search_with_retriever(args.query)
        if retriever_results:
            print(f"✓ Found {len(retriever_results)} results for query '{args.query}' using retriever")
            for i, result in enumerate(retriever_results):
                print(f"\nResult {i+1}:")
                print(f"Content: {result.page_content}")
                print(f"Metadata: {result.metadata}")
        else:
            print("No results found using retriever.")    
        
if __name__ == "__main__":
    main()