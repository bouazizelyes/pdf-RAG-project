# scripts/run_rag.py

import json
from langchain_core.documents import Document

# --- Local Imports from your RAG library ---
import config
from rag_system.retrieval import HybridRetriever
from rag_system.generation import AnswerGenerator

def main():
    """Main function to run the live RAG query system."""
    print("--- Initializing Live RAG System ---")

    # --- Step 1: Check if indexes exist ---
    if not config.FAISS_INDEX_PATH.exists() or not config.PROCESSED_DOCS_PATH.exists():
        print("❌ ERROR: Indexes not found!")
        print(f"Please run 'python scripts/build_index.py' first.")
        return

    # --- Step 2: Load documents and initialize retriever ---
    print("Loading pre-processed documents...")
    with open(config.PROCESSED_DOCS_PATH, 'r', encoding='utf-8') as f:
        documents = [Document(**d) for d in json.load(f)]
    
    retriever = HybridRetriever(documents=documents)

    # --- Step 3: Define query and retrieve documents ---
    user_query = "Quelles sont les directives pour la documentation technique et comment est-elle versionnée?"
    print(f"\n--- Processing Query: '{user_query}' ---")
    
    retrieved_docs = retriever.retrieve(query_text=user_query)

    if not retrieved_docs:
        print("\nNo relevant documents found. Cannot generate an answer.")
        return

    # --- Step 4: Initialize generator and produce the final answer ---
    generator = AnswerGenerator()
    
    print("\n--- Generating Final Answer ---")
    final_answer = generator.generate_answer(query=user_query, context_docs=retrieved_docs)

    print("\n\n================ FINAL ANSWER ================")
    print(final_answer)
    print("============================================")

    print("\n\n--- SOURCES USED ---")
    for doc in retrieved_docs:
        score = doc.metadata.get('rerank_score', 'N/A')
        print(f"- {doc.metadata['source']} (Chunk {doc.metadata['chunk_number']}, Score: {score:.4f})")

if __name__ == "__main__":
    main()
