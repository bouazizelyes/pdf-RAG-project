# scripts/build_index.py
import json
import argparse
import config
from rag_system.processing import get_documents_from_sources
from rag_system.indexing import build_and_save_indexes
from langchain_core.documents import Document
from typing import List 
import shutil

def save_docs_to_json(docs: list[Document], path: str):
    """Saves a list of LangChain Documents to a JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump([doc.model_dump() for doc in docs], f, ensure_ascii=False, indent=2)

def display_chunk_examples(docs: List[Document]):
    """Prints the first 3 chunks to the console for verification."""
    print("\n--- Displaying First 3 Document Chunks ---")
    for i, doc in enumerate(docs[:3]):  # Fixed to show only 3 chunks as intended
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {doc.metadata.get('source', 'N/A')}")
        print(f"Content:\n{doc.page_content}")
        print("-" * 20)
        
def _perform_clean_build():
    """
    Deletes the generated directories to force a complete rebuild of all data.
    """
    print("--- 🚩 Force Rebuild Flag Detected. Cleaning old data... ---")

    # Delete the directory containing generated Markdown and JSON files
    if config.EXTRACTED_DATA_PATH.exists():
        shutil.rmtree(config.EXTRACTED_DATA_PATH)
        print(f"  -> Deleted directory: {config.EXTRACTED_DATA_PATH}")

    # Delete the directory containing the FAISS index and processed documents
    if config.INDEX_STORE_PATH.exists():
        shutil.rmtree(config.INDEX_STORE_PATH)
        print(f"  -> Deleted directory: {config.INDEX_STORE_PATH}")
    
    print("--- ✅ Cleaning complete. Starting fresh build. ---")

def main():
    """
    Main function for the preprocessing unit.
    It processes source files, creates chunks, and builds and saves all necessary indexes.
    """
    parser = argparse.ArgumentParser(description="Build the RAG system indexes from source documents.")
    parser.add_argument(
        "--force-rebuild",
        action="store_true", # This makes it a flag, e.g., --force-rebuild
        help="If set, deletes all previously extracted data and indexes before starting."
    )
    args = parser.parse_args()
    if args.force_rebuild:
        _perform_clean_build()  
    print("--- Running Index Building Pipeline ---")

    # Display device information
    print(f"Using device: {config.DEVICE} ({'CPU' if config.DEVICE.type == 'cpu' else 'GPU/MPS'})")

    # Step 1: Process source files into chunked documents
    documents = get_documents_from_sources()
    if not documents:
        print("❌ No documents were created. Exiting.")
        return
        
    display_chunk_examples(documents)
    # Step 2: Save the processed documents (needed for BM25)
    save_docs_to_json(documents, str(config.PROCESSED_DOCS_PATH))
    print(f"Saved {len(documents)} processed documents to '{config.PROCESSED_DOCS_PATH}'")

    # Step 3: Build and save the vector indexes
    build_and_save_indexes(documents)

    print("\n✅ Indexing pipeline complete. The system is ready for querying.")

if __name__ == "__main__":
    main()
