from typing import List, Dict, Union
from langchain_core.documents import Document

def format_response(answer: str = "", sources: List[Document] = None) -> Dict[str, Union[str, List[Dict]]]:
    """Format the response into a consistent JSON structure"""
    if sources is None:
        sources = []
    
    formatted_sources = []
    for doc in sources:
        formatted_sources.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "metadata": {k:v for k,v in doc.metadata.items() if k != "source"}
        })
    
    return {
        "answer": answer,
        "sources": formatted_sources
    } 