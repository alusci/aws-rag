from utils.document_processor import load_and_split_documents, index_documents

def main():
    # Load and split documents
    documents = load_and_split_documents("./documents")
    
    # Index documents
    vectorstore = index_documents(documents)
    print(f"Successfully indexed {len(documents)} document chunks")

if __name__ == "__main__":
    main() 