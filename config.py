from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# AWS and OpenSearch Configuration
AWS_REGION = os.getenv('AWS_REGION')
OPENSEARCH_URL = os.getenv('OPENSEARCH_URL')
INDEX_NAME = os.getenv('INDEX_NAME')

# Model Configuration
EMBEDDING_MODEL_ID = os.getenv('EMBEDDING_MODEL_ID')
LLM_MODEL_ID = os.getenv('LLM_MODEL_ID')
VECTOR_DIMENSION = int(os.getenv('VECTOR_DIMENSION', 1024))  # Default to 1024 if not set

# Validate required environment variables
required_vars = [
    'AWS_REGION',
    'OPENSEARCH_URL',
    'INDEX_NAME',
    'EMBEDDING_MODEL_ID',
    'LLM_MODEL_ID'
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}") 