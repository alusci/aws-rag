from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from .vector_store import get_vectorstore

def load_and_split_documents(directory_path):
    """Load and split documents from a directory"""
    # Load documents
    loader = DirectoryLoader(directory_path)
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return text_splitter.split_documents(documents)

def index_documents(documents):
    """Index documents in the vector store"""
    vectorstore = get_vectorstore()
    vectorstore.add_documents(documents)
    return vectorstore 