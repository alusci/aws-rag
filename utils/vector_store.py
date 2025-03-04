from opensearchpy import OpenSearch
from langchain_community.vectorstores import OpenSearchVectorSearch
from .aws_clients import get_aws_auth, get_opensearch_connection_params
from .models import initialize_embeddings
from config import OPENSEARCH_URL, INDEX_NAME, VECTOR_DIMENSION

def get_opensearch_client():
    """Get OpenSearch client"""
    awsauth = get_aws_auth()
    connection_params = get_opensearch_connection_params()
    
    return OpenSearch(
        hosts=[OPENSEARCH_URL],
        http_auth=awsauth,
        **connection_params
    )

def get_index_config():
    """Get OpenSearch index configuration"""
    return {
        "settings": {
            "index": {
                "knn": True,
            }
        },
        "mappings": {
            "properties": {
                "vector_field": {
                    "type": "knn_vector",
                    "dimension": VECTOR_DIMENSION,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "nmslib"
                    }
                },
                "text": {"type": "text"},
                "metadata": {"type": "object"}
            }
        }
    }

def create_vector_index():
    """Create OpenSearch index with vector field configuration"""
    client = get_opensearch_client()
    
    # Delete index if it exists
    try:
        client.indices.delete(index=INDEX_NAME)
        print(f"Deleted existing index: {INDEX_NAME}")
    except:
        pass

    # Create new index
    try:
        response = client.indices.create(
            index=INDEX_NAME,
            body=get_index_config()
        )
        print(f"Successfully created index: {INDEX_NAME}")
        print(response)
    except Exception as e:
        print(f"Error creating index: {e}")

def get_vectorstore():
    """Get connection to the vector store"""
    embeddings = initialize_embeddings()
    awsauth = get_aws_auth()
    connection_params = get_opensearch_connection_params()
    
    return OpenSearchVectorSearch(
        index_name=INDEX_NAME,
        embedding_function=embeddings,
        opensearch_url=OPENSEARCH_URL,
        http_auth=awsauth,
        vector_field="vector_field",
        text_field="text",
        **connection_params
    ) 