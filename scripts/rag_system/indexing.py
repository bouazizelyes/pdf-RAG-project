# scripts/rag_system/indexing.py
from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import config

def build_and_save_indexes(documents: List[Document]):
    """Builds and saves BM25 (via documents) and FAISS indexes."""
    if not documents:
        raise ValueError("Cannot build index from an empty list of documents.")

    print("\n--- Building and Saving Indexes ---")
    config.INDEX_STORE_PATH.mkdir(parents=True, exist_ok=True)

    # 1. BM25 is built in-memory later, but we save the documents it needs.
    # We already do this via PROCESSED_DOCS_PATH, so no extra action is needed.
    print("✅ Document list is ready for BM25.")

    # 2. Build and save the FAISS index.
    print(f"  -> Building FAISS index with '{config.EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL, model_kwargs={'device': config.DEVICE}
    )
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(str(config.FAISS_INDEX_PATH))
    print(f"✅ FAISS index saved to '{config.FAISS_INDEX_PATH}'")
