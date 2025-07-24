# scripts/config.py

from pathlib import Path
import torch
import os

DEBUG = os.getenv("DEBUG_MODE", "0") == "1"

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
QUERY_REWRITE_PROMPT = [
    {
        "role": "system",
        "content": (
            "Tu es un assistant expert qui réécrit une question de 3 manières distinctes et complémentaires. "
            "Ta réponse doit contenir UNIQUEMENT les 3 phrases, une par ligne. "
            "N'ajoute AUCUN titre, AUCUN numéro, et AUCUNE explication."
        )
    },
    # --- DÉBUT DE L'EXEMPLE "ONE-SHOT" ---
    {
        "role": "user",
        "content": "Question originale : \"Comment configurer le pipeline de traitement des données ?\""
    },
    {
        "role": "assistant",
        "content": (
            "Bonnes pratiques pour la configuration du pipeline de traitement de données\n"
            "Guide d'intégration API et services pour le pipeline de données\n"
            "Étapes simples pour démarrer avec le processeur de données"
        )
    },
    # --- FIN DE L'EXEMPLE ---
    
    # La vraie question de l'utilisateur est insérée ici
    {
        "role": "user",
        "content": "Question originale : \"{original_query}\""
    }
]

# Prompt for the final answer generation by the SLM
ANSWER_GENERATION_PROMPT = [
    {
        "role": "system",
        "content": (
            "Tu es un assistant de Q&A factuel. Ta seule mission est de répondre à la question de l'utilisateur en te basant **exclusivement** sur les extraits de documents fournis dans le CONTEXTE."
            "\n\n**Règles impératives :**"
            "\n1. Ne jamais, sous aucun prétexte, utiliser des connaissances externes. Ta réponse doit être une synthèse directe du CONTEXTE."
            "\n2. Si les informations ne sont pas dans le CONTEXTE, ta seule et unique réponse doit être : **\"D'après les documents fournis, je ne peux pas répondre à cette question.\"**"
            "\n3. Cite tes sources à la fin de chaque information pertinente en utilisant le format `[source: nom_du_fichier]`."
            "\n4. Rédige en français, dans un style clair et direct."
        ),
    },
    {
        "role": "user",
        "content": (
            "CONTEXTE:\n"
            "{context}\n\n"
            "QUESTION:\n"
            "{query}"
        )
    }
]
