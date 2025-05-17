from langchain_aws import BedrockEmbeddings, BedrockLLM
from .aws_clients import initialize_bedrock_client
from config import EMBEDDING_MODEL_ID, LLM_MODEL_ID

def initialize_embeddings():
    """Initialize the embedding model"""
    bedrock_client = initialize_bedrock_client()
    return BedrockEmbeddings(
        client=bedrock_client,
        model_id=EMBEDDING_MODEL_ID
    )

def initialize_llm():
    """Initialize the LLM"""
    bedrock_client = initialize_bedrock_client()
    return BedrockLLM(
        client=bedrock_client,
        model_id=LLM_MODEL_ID,
        model_kwargs={
            "temperature": 0.0,
            "anthropic_version": "bedrock-2023-05-31"
        }
    ) 