import sys
import os
import warnings

# Suppress harmless warnings from libraries
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

# --- Test Script ---

all_passed = True
print("🚀 Starting LLM Environment Test Suite...")
print(f"🐍 Using Python version: {sys.version.split()[0]}")
print("-" * 60)

# 1. Test PyTorch and CUDA
try:
    print("1. [CHECKING] PyTorch and CUDA...")
    import torch
    print(f"   - PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"   ✅ CUDA is available.")
        print(f"   - CUDA version: {torch.version.cuda}")
        print(f"   - GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("   ❌ CUDA not available. Model training will run on CPU.")
        # We don't mark this as a failure, as CPU-only is a valid setup
        # all_passed = False 
    print("-" * 60)
except ImportError as e:
    print(f"   ❌ FAILED: PyTorch not found. {e}")
    all_passed = False
    print("-" * 60)


# 2. Test Hugging Face Core Stack (Transformers, Accelerate, BitsAndBytes)
try:
    print("2. [CHECKING] Hugging Face Core (transformers, accelerate, bitsandbytes)...")
    import transformers
    import accelerate
    import bitsandbytes
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"   - Transformers version: {transformers.__version__}")
    print(f"   - Accelerate version: {accelerate.__version__}")
    print(f"   - BitsAndBytes version: {bitsandbytes.__version__}")

    # Test loading a small model with 4-bit quantization
    print("   - Attempting to load a small model in 4-bit (testing integration)...")
    model_id = "facebook/opt-125m"
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    
    # This single command tests transformers, accelerate, and bitsandbytes working together
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Quick inference test
    inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
    _ = model.generate(**inputs, max_new_tokens=5)
    
    print(f"   ✅ Successfully loaded '{model_id}' with 4-bit quantization.")
    print("-" * 60)
except Exception as e:
    print(f"   ❌ FAILED: Error in Hugging Face stack. {e}")
    all_passed = False
    print("-" * 60)


# 3. Test PEFT and TRL for Fine-tuning
try:
    print("3. [CHECKING] Fine-tuning libraries (peft, trl)...")
    import peft
    import trl
    print(f"   - PEFT version: {peft.__version__}")
    print(f"   - TRL version: {trl.__version__}")
    # Simple import check is sufficient as their functionality is tested above
    print("   ✅ PEFT and TRL are importable.")
    print("-" * 60)
except ImportError as e:
    print(f"   ❌ FAILED: Fine-tuning library not found. {e}")
    all_passed = False
    print("-" * 60)

# 4. Test RAG Stack (Sentence-Transformers and FAISS)
try:
    print("4. [CHECKING] RAG Stack (sentence-transformers, faiss)...")
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss

    print(f"   - Sentence-Transformers is installed.")
    print(f"   - FAISS-GPU version: {faiss.__version__}")

    print("   - Creating embeddings with SentenceTransformer...")
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = ['This is a test sentence.', 'FAISS is a vector search library.']
    embeddings = st_model.encode(sentences)
    print(f"   - Embedding shape: {embeddings.shape}")

    print("   - Building a FAISS index and searching...")
    d = embeddings.shape[1]  # dimension
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    
    # Search for the first sentence
    k = 2 # number of nearest neighbors
    xq = st_model.encode(['A test sentence for search.'])
    D, I = index.search(xq, k)
    
    assert I[0][0] == 0, "FAISS search failed"
    print("   ✅ Successfully created embeddings and performed a FAISS search.")
    print("-" * 60)
except Exception as e:
    print(f"   ❌ FAILED: Error in RAG stack. {e}")
    all_passed = False
    print("-" * 60)


# 5. Test High-Level Frameworks (LangChain, LlamaIndex)
try:
    print("5. [CHECKING] High-level frameworks (LangChain, LlamaIndex)...")
    import langchain
    import llama_index
    print(f"   - LangChain version: {langchain.__version__}")
    print(f"   - LlamaIndex version: {llama_index.__version__}")
    print("   ✅ LangChain and LlamaIndex are importable.")
    print("-" * 60)
except ImportError as e:
    print(f"   ❌ FAILED: High-level framework not found. {e}")
    all_passed = False
    print("-" * 60)


# 6. Test PDF Document Loader (PyMuPDF)
try:
    print("6. [CHECKING] PDF Loader (PyMuPDF)...")
    import fitz  # PyMuPDF is imported as fitz
    
    print(f"   - PyMuPDF (fitz) version: {fitz.__version__}")
    # Create a dummy in-memory PDF and read it
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Hello from PyMuPDF!")
    pdf_bytes = doc.write()
    doc.close()

    # Re-open from memory and check content
    doc_from_bytes = fitz.open("pdf", pdf_bytes)
    text = doc_from_bytes[0].get_text()

    assert "Hello from PyMuPDF!" in text, "PDF text extraction failed"
    print("   ✅ Successfully created and read a dummy PDF in memory.")
    print("-" * 60)
except Exception as e:
    print(f"   ❌ FAILED: PyMuPDF test failed. {e}")
    all_passed = False
    print("-" * 60)


# --- Final Summary ---
print("="*60)
if all_passed:
    print("🎉 CONGRATULATIONS! All checks passed. Your environment is ready!")
else:
    print("🔥 ATTENTION: Some checks failed. Please review the errors above.")
print("="*60)
