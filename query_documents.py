import json
from utils.vector_store import get_vectorstore
from utils.models import initialize_llm
from utils.response_formatter import format_response
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

def qa_search(query, vectorstore):
    """Perform QA search with LLM"""
    llm = initialize_llm()
    
    # Define the prompt template
    prompt = PromptTemplate.from_template(
        """Answer the question based on the following context:
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
    )
    
    # Create the retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create retrieval chain
    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | document_chain
        | StrOutputParser()
    )
    
    # Execute the chain
    result = retrieval_chain.invoke(query)
    
    # Get source documents for the response
    source_docs = retriever.get_relevant_documents(query)
    
    return format_response(
        answer=result,
        sources=source_docs
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