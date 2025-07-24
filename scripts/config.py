# scripts/config.py

from pathlib import Path
import torch

# --- Project & Data Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"

RAW_DATA_PATH = DATA_PATH / "raw"
EXTRACTED_DATA_PATH = DATA_PATH / "extracted"

PDF_SOURCE_DIR = DATA_PATH / "raw" / "Pdf"
MARKDOWN_DEST_DIR = EXTRACTED_DATA_PATH / "Markdown"
JSON_DEST_DIR = EXTRACTED_DATA_PATH / "Json"

# --- Index Storage Paths ---
INDEX_STORE_PATH = PROJECT_ROOT / "vector_store"
PROCESSED_DOCS_PATH = INDEX_STORE_PATH / "processed_documents.json"
FAISS_INDEX_PATH = INDEX_STORE_PATH / "faiss_index"

PDF_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
MARKDOWN_DEST_DIR.mkdir(parents=True, exist_ok=True) 
JSON_DEST_DIR.mkdir(parents=True, exist_ok=True)     
INDEX_STORE_PATH.mkdir(parents=True, exist_ok=True)

# --- Model & Device Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"  #'cross-encoder/ms-marco-MiniLM-L6-v2'
GENERATION_MODEL = "Qwen/Qwen3-1.7B"#"Qwen/Qwen2.5-1.5B-Instruct" # Using this as our SLM

# --- Processing & Chunking Configuration ---
OCR_LANGUAGE = "fr"
CHUNK_MIN_TOKENS = 100
CHUNK_MAX_TOKENS = 500

# --- RAG Pipeline Parameters ---
INITIAL_K_RETRIEVAL = 15
CANDIDATES_FOR_RERANKING = 70
TOP_K_RERANK = 5
RRF_K_CONSTANT = 60

# --- PROMPTS ---

# Prompt for generating multiple queries from a single user question
QUERY_REWRITE_PROMPT = """
<system>
Tu es un assistant expert en reformulation pour la recherche documentaire technique.
Réfléchis étape par étape à trois manières distinctes de poser la question :
1. Une formulation **générale** (pour capter l’intention large).  
2. Une formulation **très technique** (en utilisant un vocabulaire technique comme API/JSON).  
3. Une formulation **em mots simples** sans sacrifier la précision.

NE RETOURNE QUE LES 3 QUESTIONS REFORMULÉES, **une par ligne**, **sans** numérotation, sans préambule ,sans salutation.

Question originale : "{original_query}"

Reformulations :
</system>
"""

# Prompt for the final answer generation by the SLM
ANSWER_GENERATION_PROMPT = """
Tu es un assistant expert chargé de synthétiser des informations techniques de manière claire et concise.
En te basant STRICTEMENT sur le CONTEXTE fourni ci-dessous, réponds à la QUESTION de l'utilisateur.

Règles importantes :
1.  Ta réponse doit être uniquement basée sur les informations présentes dans le CONTEXTE.
2.  Si tu nùqs pqs répondu à la question en utilisant le CONTEXTE, dis-le clairement : "D'après les documents fournis, je ne peux pas répondre à cette question."
3.  Cite tes sources à la fin de chaque phrase en utilisant le format `[source: nom_du_fichier]`.
4.  La réponse doit être en français.


CONTEXTE:
---
{context}
---

QUESTION:
{query}

RÉPONSE:
"""
