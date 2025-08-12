from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from sqlalchemy import text

from sqlalchemy import text, create_engine, inspect
import pandas as pd

# For running this code, you need to have a PostgreSQL instance running with pgvector enabled.
# Use the following command to start a PostgreSQL instance with pgvector:
#sudo docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16

embeddings = OllamaEmbeddings(model="llama3.3")

# See docker command above to launch a postgres instance with pgvector enabled.
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
collection_name = "PDF Collection"

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)


def add_docs():
    docs = [
    Document(
        page_content="there are cats in the pond",
        metadata={"id": 1, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="ducks are also found in the pond",
        metadata={"id": 2, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="fresh apples are available at the market",
        metadata={"id": 3, "location": "market", "topic": "food"},
    ),
    Document(
        page_content="the market also sells fresh oranges",
        metadata={"id": 4, "location": "market", "topic": "food"},
    ),
    Document(
        page_content="the new art exhibit is fascinating",
        metadata={"id": 5, "location": "museum", "topic": "art"},
    ),
    Document(
        page_content="a sculpture exhibit is also at the museum",
        metadata={"id": 6, "location": "museum", "topic": "art"},
    ),
    Document(
        page_content="a new coffee shop opened on Main Street",
        metadata={"id": 7, "location": "Main Street", "topic": "food"},
    ),
    Document(
        page_content="the book club meets at the library",
        metadata={"id": 8, "location": "library", "topic": "reading"},
    ),
    Document(
        page_content="the library hosts a weekly story time for kids",
        metadata={"id": 9, "location": "library", "topic": "reading"},
    ),
    Document(
        page_content="a cooking class for beginners is offered at the community center",
        metadata={"id": 10, "location": "community center", "topic": "classes"},
    ),
    ]
    
    # Add to vector store
    vector_store.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])
    print("Documents added successfully!")

    
#Function to implement similarity search
def similarity_search(query, k=3):
    results = vector_store.similarity_search(query, k=k)
    print(f"Found {len(results)} results for query '{query}':")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Content: {result.page_content}")
        print(f"Metadata: {result.metadata}")
    return results

#Function to imlement similarity search with a retriever
def similarity_search_with_retriever(query, k=3):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    results = retriever.invoke(query)
    print(f"Found {len(results)} results for query '{query}':")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Content: {result.page_content}")
        print(f"Metadata: {result.metadata}")
    return results

def query_postgres(query, k=3):
    """Query the PostgreSQL vector store directly using SQL"""
    with vector_store.client.engine.connect() as conn:
        sql = text(f"""
            SELECT id, page_content, metadata
            FROM {vector_store.collection_name}
            ORDER BY embedding <-> :query_embedding
            LIMIT :k
        """)
        results = conn.execute(sql, {"query_embedding": embeddings.embed_query(query), "k": k}).fetchall()
        
        # Format results
        formatted_results = []
        for row in results:
            formatted_results.append({
                "id": row.id,
                "content": row.page_content,
                "metadata": row.metadata
            })
        
        return formatted_results

def vector_similarity_search(query, collection_name="PDF Collection", k=3):
    """
    Perform similarity search directly using PostgreSQL pgvector extension.
    
    Args:
        query: The query string to search for
        collection_name: Name of the collection in the database
        k: Number of results to return
        
    Returns:
        List of documents with content and metadata
    """
    # Get the embedding for the query using the same model that created the embeddings
    query_embedding = embeddings.embed_query(query)
    
    # Create engine connection
    engine = create_engine(connection)
    
    # Format the embedding vector as a PostgreSQL vector literal
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    
    documents = []
    
    try:
        with engine.begin() as conn:
            # Step 1: Get the collection UUID for the given name
            collection_query = text("""
                SELECT uuid FROM public.langchain_pg_collection
                WHERE name = :collection_name LIMIT 1
            """)
            collection_result = conn.execute(collection_query, {"collection_name": collection_name})
            collection_row = collection_result.fetchone()
            
            if not collection_row:
                print(f"Collection '{collection_name}' not found")
                return []
                
            collection_uuid = collection_row[0]
            
            #collection_uuid = "74e2e511-2a14-425c-aa3d-56cdbb61ea2b"
            
            print(f"Collection UUID for '{collection_name}': {collection_uuid}")
            
            # Step 2: Perform the similarity search using vector operators
            #search_query = f"""
            #    SELECT e.document, e.cmetadata
            #    FROM public.langchain_pg_embedding e
            #    WHERE e.collection_id = :collection_id
            #    ORDER BY e.embedding <=> '{embedding_str}'::vector(384)
            #    LIMIT :k
            #"""
            search_query = text(f"""
                SELECT id, document, cmetadata, collection_id, 
                       embedding <=> '{embedding_str}'::vector(384) as distance
                FROM public.langchain_pg_embedding
                WHERE collection_id = :collection_id
                ORDER BY distance
                LIMIT :k
            """)
            
            #search_query = text(search_query)
            print(f"Executing search query: {search_query}")
            result = conn.execute(text(search_query), { "k": k})
            rows = result.fetchall()
        
            # Process the results
            #Retrieve the id, document content, and metadata
            for row in rows:
                document_content = row[1]
                print(f"Document content: {document_content}")
                # Handle metadata, if available
                metadata = row[2] if row[2] else {}
                print(f"Document metadata: {metadata}")
                documents.append(Document(
                    id=row[0],
                    page_content=document_content,
                    metadata=metadata
                ))
    
    except Exception as e:
        print(f"Error performing similarity search: {str(e)}")
    finally:
        engine.dispose()
    
    # Display results
    print(f"\nDirect Vector Search: Found {len(documents)} results for query '{query}':")
    for i, doc in enumerate(documents):
        print(f"\nResult {i+1}:")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
    
    return documents

def main():
    #add_docs()
    query = "What is Naman case"
    #similarity_search(query)
    #similarity_search_with_retriever(query)

    #query_postgres("What is Naman case?", k=3)
    
    print("Using vector_similarity_search:")
    vector_similarity_search(query)

if __name__ == "__main__":
    main()        
