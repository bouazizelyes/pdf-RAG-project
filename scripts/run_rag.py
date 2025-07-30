# scripts/run_rag.py

import numpy as np
import argparse
import os
import json
import torch
import time
import psutil
import traceback
from datetime import datetime
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_transformers import LongContextReorder

# --- Local Imports from your RAG library ---
import config
from config import DEBUG
from rag_system.retrieval import HybridRetriever
from rag_system.generation import AnswerGenerator

# Parse arguments and set environment variables FIRST
def early_thread_setup():
    import sys
    # Parse command line to get thread count
    for i, arg in enumerate(sys.argv):
        if arg == "--cpu-threads" and i + 1 < len(sys.argv):
            num_threads = int(sys.argv[i + 1])
            break
    else:
        num_threads = 4  # default
    
    # Set ALL environment variables before any imports
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    
    if DEBUG:
        print(f"🔧 CPU threads configured for: {num_threads}")
    return num_threads

# Call this before ANY other imports
configured_threads = early_thread_setup()

class LogManager:
    def __init__(self, log_file=None):
        self.log_file = log_file or config.PROJECT_ROOT / "logs" / "rag_execution.log"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.session_data = {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "start_time": None,
            "end_time": None,
            "device": None,
            "cpu_threads": None,
            "execution_times": {},
            "query": None,
            "retrieved_documents": [],
            "final_answer": None,
            "performance_metrics": {},
            "system_info": {}
        }
    
    def log_system_info(self):
        """Log system information"""
        self.session_data["system_info"] = {
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "timestamp": datetime.now().isoformat()
        }
        
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            self.session_data["system_info"]["gpu"] = {
                "name": gpu_props.name,
                "total_memory_gb": round(gpu_props.total_memory / (1024**3), 2),
                "cuda_version": torch.version.cuda
            }
    
    def log_execution_step(self, step_name, duration, details=None):
        """Log execution step with timing"""
        if step_name not in self.session_data["execution_times"]:
            self.session_data["execution_times"][step_name] = []
        self.session_data["execution_times"][step_name].append({
            "duration": duration,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        })
    
    def log_query_info(self, query, retrieved_docs):
        """Log query and retrieved documents"""
        self.session_data["query"] = query
        self.session_data["retrieved_documents"] = [
            {
                "source": doc.metadata.get('source', 'N/A'),
                "chunk_number": doc.metadata.get('chunk_number', 'N/A'),
                "rerank_score": doc.metadata.get('rerank_score', 'N/A'),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            for doc in retrieved_docs
        ]
    
    def log_final_answer(self, answer):
        """Log final answer"""
        self.session_data["final_answer"] = answer
    
    def log_performance_metrics(self, total_time):
        """Log overall performance metrics"""
        self.session_data["performance_metrics"]["total_execution_time"] = total_time
        
        # Calculate total time for each step
        step_times = {}
        for step, times in self.session_data["execution_times"].items():
            step_times[step] = sum(t["duration"] for t in times)
        self.session_data["performance_metrics"]["step_times"] = step_times
    
    def save_log(self):
    """Save log to file in JSON format"""
    import numpy as np
    
    def convert_numpy(obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(item) for item in obj]
        return obj
    
    self.session_data["end_time"] = datetime.now().isoformat()
    
    # Convert all numpy types to Python native types
    self.session_data = convert_numpy(self.session_data)
    
    # Read existing logs
    logs = []
    if self.log_file.exists():
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
                # Convert existing logs too
                logs = convert_numpy(logs)
        except:
            logs = []
    
    # Append new session
    logs.append(self.session_data)
    
    # Save updated logs
    with open(self.log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2, default=str)
def print_help_extended():
    """Print extended help with environment variables and runtime options"""
    help_text = """
🚀 RAG SYSTEM - EXTENDED HELP
============================

COMMAND LINE ARGUMENTS:
----------------------
--cpu                  Force CPU usage even if GPU is available
--cpu-threads N        Number of CPU threads to use (default: 4)
--verbose              Show detailed execution information
--debug-output         Show internal model outputs (thinking process, etc.)
--log-file PATH        Path to log file for detailed execution logging
--help-extended        Show this extended help message

OUTPUT CONTROL:
---------------
Basic mode:     Only essential information displayed
--verbose:      Shows execution progress (✅ Model loaded, etc.)
--debug-output: Shows internal model outputs (thinking process, raw text)

ENVIRONMENT VARIABLES:
---------------------
CUDA_VISIBLE_DEVICES=-1        Force CPU mode
CUDA_VISIBLE_DEVICES=0         Use GPU (default if available)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  Fix memory fragmentation
OMP_NUM_THREADS=N              Set OpenMP threads
MKL_NUM_THREADS=N              Set Intel MKL threads
NUMEXPR_NUM_THREADS=N          Set NumExpr threads
DEBUG_MODE=1                   Enable debug output

RUNTIME OPTIMIZATION:
--------------------
GPU Memory Issues:
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export CUDA_VISIBLE_DEVICES=0
  python scripts/run_rag.py

CPU Performance:
  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4
  export NUMEXPR_NUM_THREADS=4
  export CUDA_VISIBLE_DEVICES=-1
  python scripts/run_rag.py --cpu-threads 4

Performance Testing:
  # Test different thread counts
  for threads in 1 2 4 8; do
    echo "Testing with $threads threads..."
    python scripts/run_rag.py --cpu --cpu-threads $threads --log-file logs/test_$threads.json
  done

EXAMPLES:
---------
# Basic GPU usage (if available)
python scripts/run_rag.py

# Force CPU with 4 threads
python scripts/run_rag.py --cpu --cpu-threads 4

# Verbose output with logging
python scripts/run_rag.py --verbose --log-file logs/session.json

# Debug mode with internal outputs
python scripts/run_rag.py --debug-output --verbose

# CPU mode with maximum performance
export OMP_NUM_THREADS=4 && export MKL_NUM_THREADS=4 && export NUMEXPR_NUM_THREADS=4
python scripts/run_rag.py --cpu --cpu-threads 4 --verbose --log-file logs/cpu_session.json

SYSTEM REQUIREMENTS:
-------------------
GPU: Minimum 4GB VRAM recommended for Qwen models
CPU: Multi-core recommended (4+ cores)
RAM: 8GB minimum, 16GB recommended
"""
    print(help_text)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run RAG system for question answering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_rag.py                           # Auto-detect GPU/CPU
  python scripts/run_rag.py --cpu --cpu-threads 4     # Force CPU mode
  python scripts/run_rag.py --verbose --log-file log.json  # Verbose with logging
  python scripts/run_rag.py --help-extended           # Show extended help
        """
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--cpu-threads", type=int, default=configured_threads, help="Number of CPU threads to use (default: %(default)s)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed execution information")
    parser.add_argument("--debug-output", action="store_true", help="Show internal model outputs (thinking process, etc.)")
    parser.add_argument("--log-file", type=str, help="Path to log file for detailed execution logging")
    parser.add_argument("--help-extended", action="store_true", help="Show extended help with environment variables and examples")
    args = parser.parse_args()
    
    # Show extended help if requested
    if args.help_extended:
        print_help_extended()
        return
    
    # Setup logging
    log_manager = LogManager(Path(args.log_file) if args.log_file else None)
    log_manager.session_data["start_time"] = datetime.now().isoformat()
    
    # Override device if --cpu flag is set
    if args.cpu:
        config.DEVICE = torch.device("cpu")
        if args.verbose:
            print("⚠️  FORCING CPU MODE")
        log_manager.session_data["cpu_threads"] = args.cpu_threads
    else:
        log_manager.session_data["cpu_threads"] = "N/A (GPU mode)"
    
    print(f"🚀 Using device: {config.DEVICE} ({'CPU' if config.DEVICE.type == 'cpu' else 'GPU/MPS'})")
    log_manager.session_data["device"] = config.DEVICE.type
    log_manager.log_system_info()
    
    # Start timing
    start_time = time.time()
    
    # CPU compatibility: Only enable memory tracking on CUDA devices
    if config.DEVICE.type == "cuda":
        try:
            torch.cuda.memory._record_memory_history(enabled='all', context='all', stacks='all', max_entries=50_000)
        except:
            pass

    if DEBUG or args.verbose:
        print("--- Initializing Live RAG System ---")

    # --- Step 1: Check if indexes exist ---
    if not config.FAISS_INDEX_PATH.exists() or not config.PROCESSED_DOCS_PATH.exists():
        print("❌ ERROR: Indexes not found!")
        print(f"Please run 'python scripts/build_index.py' first.")
        return

    # --- Step 2: Load documents and initialize retriever ---
    if DEBUG or args.verbose:
        print("Loading pre-processed documents...")
    load_start = time.time()
    with open(config.PROCESSED_DOCS_PATH, 'r', encoding='utf-8') as f:
        documents = [Document(**d) for d in json.load(f)]
    load_end = time.time()
    load_time = load_end - load_start
    log_manager.log_execution_step("document_loading", load_time, {"document_count": len(documents)})
    if DEBUG or args.verbose:
        print(f"⏱️  Document loading time: {load_time:.2f} seconds")

    retriever = HybridRetriever(documents=documents)

    # --- Step 3: Define query and retrieve documents ---
    user_query = "Comment sécuriser les secrets dans un projet Symfony 6.4 ?"
    if DEBUG or args.verbose:
        print(f"\n--- Processing Query: '{user_query}' ---")
    
    # CPU compatibility: Only dump snapshots on CUDA devices
    if config.DEVICE.type == "cuda":
        try:
            torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)
        except:
            pass
    
    retrieval_start = time.time()
    retrieved_docs = retriever.retrieve(query_text=user_query)
    retrieval_end = time.time()
    retrieval_time = retrieval_end - retrieval_start
    log_manager.log_execution_step("document_retrieval", retrieval_time, {"retrieved_count": len(retrieved_docs)})
    
    if DEBUG or args.verbose:
        print(f"⏱️  Retrieval time: {retrieval_time:.2f} seconds")

    if not retrieved_docs:
        print("\nNo relevant documents found. Cannot generate an answer.")
        return
    
    # Log retrieved documents
    log_manager.log_query_info(user_query, retrieved_docs)
    
    # CPU compatibility: Only dump snapshots on CUDA devices
    if config.DEVICE.type == "cuda":
        try:
            torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
        except:
            pass
    
    # --- Step 4: Initialize generator and produce the final answer ---
    if DEBUG or args.verbose:
        print("\n--- Generating Final Answer ---")
    generator = AnswerGenerator()
    
    # CPU compatibility: Only dump snapshots on CUDA devices
    if config.DEVICE.type == "cuda":
        try:
            torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
        except:
            pass
    
    generation_start = time.time()
    try:
        final_answer = generator.generate_answer(query=user_query, context_docs=retrieved_docs)
        generation_end = time.time()
        generation_time = generation_end - generation_start
        log_manager.log_execution_step("answer_generation", generation_time)
        log_manager.log_final_answer(final_answer)
        
        # CPU compatibility: Only dump snapshots on CUDA devices
        if config.DEVICE.type == "cuda":
            try:
                torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
            except:
                pass
        
        # End timing
        end_time = time.time()
        total_time = end_time - start_time
        log_manager.log_performance_metrics(total_time)
        
        # Save log
        log_manager.save_log()
        
        if DEBUG or args.verbose:
            print(f"⏱️  Generation time: {generation_time:.2f} seconds")
        
        print("\n\n================ FINAL ANSWER ================")
        print(final_answer)
        print("============================================")
        if DEBUG or args.verbose:
            print(f"\n⏱️  TOTAL EXECUTION TIME: {total_time:.2f} seconds")
        print("\n\n--- SOURCES USED ---")
        
        for doc in retrieved_docs:
            score = doc.metadata.get('rerank_score', 'N/A')
            print(f"- {doc.metadata['source']} (Chunk {doc.metadata['chunk_number']}, Score: {score:.4f})")
        
        if DEBUG or args.verbose:
            print(f"\n📊 PERFORMANCE SUMMARY:")
            print(f"   Document Loading: {load_time:.2f}s")
            print(f"   Document Retrieval: {retrieval_time:.2f}s")
            print(f"   Answer Generation: {generation_time:.2f}s")
            print(f"   🏁 Total Time: {total_time:.2f}s")
        
    except Exception as e:
        print(f"❌ ERROR during answer generation: {str(e)}")
        if DEBUG or args.verbose:
            print("🔍 Full traceback:")
            traceback.print_exc()
        print("\n⚠️  The system retrieved documents but failed to generate an answer.")
        print("   This might be due to memory constraints or model loading issues.")

if __name__ == "__main__":
    main()
