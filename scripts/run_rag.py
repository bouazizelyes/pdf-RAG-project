# scripts/run_rag.py
import torch
import json
from langchain_core.documents import Document
from langchain_community.document_transformers import LongContextReorder

# --- Local Imports from your RAG library ---
import config
from config import DEBUG
from rag_system.retrieval import HybridRetriever
from rag_system.generation import AnswerGenerator

def main():
    # CPU compatibility: Only enable memory tracking on CUDA devices
    if config.DEVICE.type == "cuda":
        torch.cuda.memory._record_memory_history(enabled='all', context='all', stacks='all', max_entries=50_000)

    if DEBUG:
        """Main function to run the live RAG query system."""
        print("--- Initializing Live RAG System ---")
        print(f"Using device: {config.DEVICE} ({'CPU' if config.DEVICE.type == 'cpu' else 'GPU/MPS'})")

    # --- Step 1: Check if indexes exist ---
    if not config.FAISS_INDEX_PATH.exists() or not config.PROCESSED_DOCS_PATH.exists():
        print("❌ ERROR: Indexes not found!")
        print(f"Please run 'python scripts/build_index.py' first.")
        return

    # --- Step 2: Load documents and initialize retriever ---
    if DEBUG:
        print("Loading pre-processed documents...")
    with open(config.PROCESSED_DOCS_PATH, 'r', encoding='utf-8') as f:
        documents = [Document(**d) for d in json.load(f)]
    
    retriever = HybridRetriever(documents=documents)
    

    # --- Step 3: Define query and retrieve documents ---
    user_query = "Comment sécuriser les secrets dans un projet Symfony 6.4 ?"
    if DEBUG:
        print(f"\n--- Processing Query: '{user_query}' ---")
    
    # CPU compatibility: Only dump snapshots on CUDA devices
    if config.DEVICE.type == "cuda":
        torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
    
    retrieved_docs = retriever.retrieve(query_text=user_query)
    

    if not retrieved_docs:
        print("\nNo relevant documents found. Cannot generate an answer.")
        return
    
    # CPU compatibility: Only dump snapshots on CUDA devices
    if config.DEVICE.type == "cuda":
        torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
    
    # --- Step 4: Initialize generator and produce the final answer ---
    generator = AnswerGenerator()
    
    # CPU compatibility: Only dump snapshots on CUDA devices
    if config.DEVICE.type == "cuda":
        torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
    
    if DEBUG:
        print("\n--- Generating Final Answer ---")
    
    
    
    
    final_answer = generator.generate_answer(query=user_query, context_docs=retrieved_docs)
    
    # CPU compatibility: Only dump snapshots on CUDA devices
    if config.DEVICE.type == "cuda":
        torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
    
    if DEBUG:
        print("\n\n================ FINAL ANSWER ================")
        print(final_answer)
        print("============================================")
        print("\n\n--- SOURCES USED ---")
        
    for doc in retrieved_docs:
        score = doc.metadata.get('rerank_score', 'N/A')
        print(f"- {doc.metadata['source']} (Chunk {doc.metadata['chunk_number']}, Score: {score:.4f})")

if __name__ == "__main__":
    main()
