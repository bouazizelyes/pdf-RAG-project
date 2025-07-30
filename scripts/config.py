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

# --- Device Configuration ---
def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()

# --- CPU Acceleration Configuration ---
def configure_cpu_optimizations():
    """Configure CPU optimizations for better performance"""
    if DEVICE.type == "cpu":
        # Set number of threads for intra-op parallelism
        torch.set_num_threads(os.cpu_count() or 1)
        
        # Set number of threads for inter-op parallelism
        torch.set_num_interop_threads(min(2, os.cpu_count() or 1))
        
        # Enable MKL optimizations if available
        if torch.backends.mkldnn.is_available():
            torch.backends.mkldnn.enabled = True
            
        # Enable OpenMP optimizations
        os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)
        os.environ["MKL_NUM_THREADS"] = str(os.cpu_count() or 1)
        os.environ["NUMEXPR_NUM_THREADS"] = str(os.cpu_count() or 1)

# Apply CPU optimizations
configure_cpu_optimizations()

# --- Keep Your Original Models ---
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
GENERATION_MODEL = "Qwen/Qwen3-1.7B"

# --- CPU-Aware Model Loading Parameters ---
def get_model_loading_config():
    """Get model loading configuration based on device"""
    if DEVICE.type == "cpu":
        return {
            "torch_dtype": torch.float32,  # Use float32 for CPU
            "low_cpu_mem_usage": True,     # Reduce memory usage during loading
            "device_map": "auto" if hasattr(torch, 'cpu') else None  # Distribute if possible
        }
    else:
        return {
            "torch_dtype": torch.float16 if DEVICE.type == "cuda" else torch.float32,
            "device_map": None
        }

MODEL_LOADING_CONFIG = get_model_loading_config()

# --- Processing & Chunking Configuration ---
OCR_LANGUAGE = "fr"
CHUNK_MIN_TOKENS = 100
CHUNK_MAX_TOKENS = 500

# --- RAG Pipeline Parameters (CPU-optimized) ---
if DEVICE.type == "cpu":
    # Reduce computational load for CPU while keeping quality
    INITIAL_K_RETRIEVAL = 10  # Reduced from 15
    CANDIDATES_FOR_RERANKING = 30  # Reduced from 70
    TOP_K_RERANK = 5
    RRF_K_CONSTANT = 40  # Reduced from 60
    BATCH_SIZE = 2  # Small batch size for CPU
    MAX_TOKENS_PER_CHUNK = 400  # Reduce token count per chunk
else:
    # Original parameters for GPU
    INITIAL_K_RETRIEVAL = 15
    CANDIDATES_FOR_RERANKING = 70
    TOP_K_RERANK = 5
    RRF_K_CONSTANT = 60
    BATCH_SIZE = 8
    MAX_TOKENS_PER_CHUNK = 500

# --- CPU Memory Management ---
CPU_MAX_MODEL_SIZE_GB = 20 # Maximum model size to load (adjust based on your RAM)

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
