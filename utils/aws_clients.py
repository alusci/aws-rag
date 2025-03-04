import boto3
from requests_aws4auth import AWS4Auth
from opensearchpy import RequestsHttpConnection

def initialize_bedrock_client():
    """Initialize AWS Bedrock client"""
    return boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1"
    )

def get_aws_auth():
    """Get AWS authentication object"""
    credentials = boto3.Session().get_credentials()
    region = 'us-east-1'
    service = 'aoss'
    
    return AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        service,
        session_token=credentials.token
    )

def get_opensearch_connection_params():
    """Get OpenSearch connection parameters"""
    return {
        "connection_class": RequestsHttpConnection,
        "use_ssl": True,
        "verify_certs": True,
        "ssl_assert_hostname": False,
        "ssl_show_warn": False
    } 