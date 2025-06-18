import sys
import os
import warnings

# Suppress harmless warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

all_passed = True
print("🚀 Starting LLM Environment Test Suite (v2)...")
print(f"🐍 Using Python version: {sys.version.split()[0]}")
print("-" * 60)

# 1. Test PyTorch and CUDA
try:
    print("1. [CHECKING] PyTorch and CUDA...")
    import torch
    print(f"   - PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"   ✅ CUDA is available. Found {gpu_count} GPU(s).")
        print(f"   - CUDA version: {torch.version.cuda}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"   - GPU {i}: {gpu_name} ({total_mem:.2f} GB VRAM)")
        # This is a key diagnostic step!
        torch.cuda.empty_cache() # Clear cache before tests
    else:
        print("   ❌ CUDA not available. Model training will run on CPU.")
    print("-" * 60)
except ImportError as e:
    print(f"   ❌ FAILED: PyTorch not found. {e}")
    all_passed = False; sys.exit(1)

# Function to print memory usage
def print_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**2)
        reserved = torch.cuda.memory_reserved(0) / (1024**2)
        print(f"   - VRAM Usage: Allocated={allocated:.2f} MB | Reserved={reserved:.2f} MB")

# 2. Hugging Face Core Stack
try:
    print("2. [CHECKING] Hugging Face Core (transformers, accelerate, bitsandbytes)...")
    import transformers, accelerate, bitsandbytes
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    print(f"   - Versions: transformers=={transformers.__version__}, accelerate=={accelerate.__version__}, bitsandbytes=={bitsandbytes.__version__}")
    print("   - Initial VRAM state:")
    print_gpu_memory_usage()
    
    print("   - Attempting to load 'facebook/opt-125m' in 4-bit...")
    model_id = "facebook/opt-125m"
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
    
    print("   - VRAM state after loading model:")
    print_gpu_memory_usage()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
    _ = model.generate(**inputs, max_new_tokens=5)
    
    print(f"   ✅ Successfully loaded and ran inference with '{model_id}'.")
    del model, tokenizer, inputs # Clean up memory
    torch.cuda.empty_cache()
    print("-" * 60)
except Exception as e:
    print(f"   ❌ FAILED: Error in Hugging Face stack. {e}")
    print("   - HINT: This often happens due to VRAM limitations or another process using the GPU.")
    all_passed = False
    print("-" * 60)

# 3. PEFT and TRL
try:
    print("3. [CHECKING] Fine-tuning libraries (peft, trl)...")
    import peft, trl
    print(f"   - Versions: peft=={peft.__version__}, trl=={trl.__version__}")
    print("   ✅ PEFT and TRL are importable.")
    print("-" * 60)
except ImportError as e:
    print(f"   ❌ FAILED: Fine-tuning library not found. {e}")
    all_passed = False

# 4. RAG Stack
try:
    print("4. [CHECKING] RAG Stack (sentence-transformers, faiss)...")
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss
    
    print(f"   - FAISS-GPU version: {faiss.__version__}")
    print("   - Loading 'all-MiniLM-L6-v2' SentenceTransformer...")
    st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
    
    print("   - VRAM state after loading SentenceTransformer:")
    print_gpu_memory_usage()

    sentences = ['This is a test sentence.', 'FAISS is a vector search library.']
    embeddings = st_model.encode(sentences)
    
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    xq = st_model.encode(['A test sentence for search.'])
    _, I = index.search(xq, 1)
    
    assert I[0][0] == 0, "FAISS search failed"
    print("   ✅ Successfully created embeddings and performed a FAISS search.")
    del st_model, index # Clean up memory
    torch.cuda.empty_cache()
    print("-" * 60)
except Exception as e:
    print(f"   ❌ FAILED: Error in RAG stack. {e}")
    print("   - HINT: If this is a CUDA error, it's likely a VRAM or resource conflict issue.")
    all_passed = False
    print("-" * 60)

# 5. High-Level Frameworks
try:
    print("5. [CHECKING] High-level frameworks (LangChain, LlamaIndex)...")
    import langchain
    # FIX: Import the 'core' module to get the version
    from llama_index.core import __version__ as llama_index_version
    
    print(f"   - LangChain version: {langchain.__version__}")
    print(f"   - LlamaIndex version: {llama_index_version}")
    print("   ✅ LangChain and LlamaIndex are importable.")
    print("-" * 60)
except ImportError as e:
    print(f"   ❌ FAILED: High-level framework not found. {e}")
    all_passed = False

# 6. PDF Loader
try:
    print("6. [CHECKING] PDF Loader (PyMuPDF)...")
    import fitz
    print(f"   - PyMuPDF (fitz) version: {fitz.__version__}")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Hello from PyMuPDF!")
    doc.close()
    print("   ✅ PyMuPDF is working.")
    print("-" * 60)
except Exception as e:
    print(f"   ❌ FAILED: PyMuPDF test failed. {e}")
    all_passed = False

# --- Final Summary ---
print("="*60)
if all_passed:
    print("🎉 CONGRATULATIONS! All checks passed. Your environment is ready!")
else:
    print("🔥 ATTENTION: Some checks failed. Please review the errors above.")
print("="*60)
