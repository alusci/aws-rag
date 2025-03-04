import json
from utils.vector_store import get_vectorstore
from utils.models import initialize_llm
from utils.response_formatter import format_response
from langchain.chains import RetrievalQA

def qa_search(query, vectorstore):
    """Perform QA search with LLM"""
    llm = initialize_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    result = qa_chain({"query": query})
    return format_response(
        answer=result["result"],
        sources=result["source_documents"]
    )

def main():
    vectorstore = get_vectorstore()
    
    while True:
        print("\nEnter your query (or 'quit' to exit):")
        query = input()
        
        if query.lower() == 'quit':
            break
            
        print("\nChoose search type:")
        print("1. Similarity Search (just find similar documents)")
        print("2. QA Search (use LLM to answer question)")
        choice = input()
        
        if choice == "1":
            results = vectorstore.similarity_search(query, k=3)
            formatted_response = format_response(sources=results)
        
        elif choice == "2":
            formatted_response = qa_search(query, vectorstore)
        
        else:
            print("Invalid choice. Please choose 1 or 2.")
            continue
        
        print("\nResults:")
        print(json.dumps(formatted_response, indent=2))

if __name__ == "__main__":
    main() 