# scripts/rag_system/retrieval.py
import re
import torch
import gc
from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_community.document_transformers import LongContextReorder

import config
from config import DEBUG

class HybridRetriever:
    """
    A class to handle the full retrieval pipeline:
    - Loads indexes from disk.
    - Expands queries.
    - Performs hybrid search (BM25 + FAISS).
    - Fuses results with RRF.
    - Reranks the final candidates.
    """

    def __init__(self, documents: List[Document]):
        print("Initializing Hybrid Retriever...")
        self.documents = documents
        self.device = config.DEVICE
        self.llm_model, self.llm_tokenizer = None, None
        self.reranker = None
        
        # 1. Load FAISS index from disk
        print("  -> Loading FAISS index...")
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL, model_kwargs={'device': self.device}
        )
        vectorstore = FAISS.load_local(
            str(config.FAISS_INDEX_PATH), embeddings, allow_dangerous_deserialization=True
        )
        self.faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": config.INITIAL_K_RETRIEVAL})
        print("✅ FAISS index loaded.")
        
        # 2. Build BM25 retriever from documents
        print("  -> Building BM25 index...")
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = config.INITIAL_K_RETRIEVAL
        print("✅ BM25 index built.")
        
    
    def _load_expansion_llm(self):
        if self.llm_model is None:
            print("  -> Loading LLM for query expansion...")
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                config.GENERATION_MODEL, quantization_config=quant_config, device_map="auto"
            )
            self.llm_tokenizer = AutoTokenizer.from_pretrained(config.GENERATION_MODEL)
    """
    def _load_expansion_llm(self):
        if self.llm_model is None:
            print("  -> Loading LLM for query expansion...")
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                config.GENERATION_MODEL, device_map="auto"
            )
            self.llm_tokenizer = AutoTokenizer.from_pretrained(config.GENERATION_MODEL)
    """
    def _unload_expansion_llm(self):
        if self.llm_model is not None:
            del self.llm_model
            del self.llm_tokenizer
            self.llm_model, self.llm_tokenizer = None, None
            gc.collect()
            torch.cuda.empty_cache()
            print("     - Expansion LLM unloaded.")

    def _generate_queries(self, original_query: str) -> List[str]:
    
        messages = [
            {
                "role": msg["role"],
                "content": msg["content"].format(original_query=original_query)
            }
            for msg in config.QUERY_REWRITE_PROMPT
        ]
        #prompt = config.QUERY_REWRITE_PROMPT.format(original_query=original_query)
        text = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
            )
        inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7, # Un peu moins de créativité pour garder le sujet
                top_p=0.8,
                # Utiliser le token EOS standard du modèle est plus fiable
                eos_token_id=self.llm_tokenizer.eos_token_id 
            )

        # --- NOUVELLE LOGIQUE DE PARSING ---
        
        # 1. Isoler UNIQUEMENT le texte généré par le modèle
        generated_ids = outputs[0][input_length:]
        generated_text = self.llm_tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print(f"--- Raw Generated Text ---\n{generated_text}\n--------------------------")

        # 2. Nettoyer et extraire les requêtes ligne par ligne
        queries = []
        # Le modèle peut parfois ajouter des labels comme "Reformulations:", on les ignore
        clean_text = generated_text.split("Reformulations:")[-1]
        
        for line in clean_text.split('\n'):
            clean_line = line.strip()

            # Ignorer les lignes vides ou les artefacts de formatage
            if not clean_line or clean_line.startswith("`"):
                continue
                
            # Retirer les numérotations ou les tirets en début de ligne
            clean_line = re.sub(r'^\s*(\d+\.|\*|-)\s*', '', clean_line)

            queries.append(clean_line)
                
        # Libérer la mémoire
        del outputs
        torch.cuda.empty_cache() 
        
        # Renvoyer une liste unique avec la requête originale
        all_queries = [original_query] + queries
        final_queries = list(dict.fromkeys(all_queries))
        return final_queries

    def _reciprocal_rank_fusion(self, ranked_lists: List[List[Document]]) -> List[Document]:
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        for doc_list in ranked_lists:
            for rank, doc in enumerate(doc_list, 1):
                doc_id = f"{doc.metadata['source']}_{doc.metadata['chunk_number']}"
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                    doc_map[doc_id] = doc
                scores[doc_id] += 1.0 / (config.RRF_K_CONSTANT + rank)
        
        sorted_doc_ids = sorted(scores, key=scores.get, reverse=True)
        return [doc_map[id] for id in sorted_doc_ids]

    def retrieve(self, query_text: str) -> List[Document]:
        """The main method to perform the full retrieval and reranking process."""
        # 1. Expand Query
        try:
            self._load_expansion_llm()
            if DEBUG:
                print("\n--- Step 1: Expanding query ---")
            expanded_queries = self._generate_queries(query_text)
            print(f"     Expanded to: {expanded_queries}")
        finally:
            self._unload_expansion_llm()

        # 2. Hybrid Retrieval for each query
        if DEBUG:
            print("\n--- Step 2: Retrieving from BM25 and FAISS ---")
        all_ranked_lists = []
        for q in expanded_queries:
            all_ranked_lists.append(self.bm25_retriever.invoke(q))
            all_ranked_lists.append(self.faiss_retriever.invoke(q))
            
        # 3. Fuse Results
        if DEBUG:
            print("\n--- Step 3: Fusing results with RRF ---")
        fused_candidates = self._reciprocal_rank_fusion(all_ranked_lists)
        unique_docs_map = {f"{doc.metadata['source']}_{doc.metadata['chunk_number']}": doc for doc in fused_candidates}
        rerank_candidates = list(unique_docs_map.values())[:config.CANDIDATES_FOR_RERANKING]

        if not rerank_candidates: 
            return []

        # 4. Rerank with Cross-Encoder
        if DEBUG:
            print(f"\n--- Step 4: Reranking top {len(rerank_candidates)} candidates ---")
        with torch.no_grad():
            reranker = CrossEncoder(config.RERANKER_MODEL, max_length=512, device=self.device)
            print(f" reranker pad : {reranker.tokenizer.pad_token}   {reranker.tokenizer.pad_token}")
            reranker.tokenizer.pad_token = reranker.tokenizer.eos_token
            reranker.model.config.pad_token_id = reranker.tokenizer.pad_token_id
            try:
                sentence_pairs = [(query_text, doc.page_content) for doc in rerank_candidates]
                scores = reranker.predict(sentence_pairs, batch_size=2, show_progress_bar=True)
            
                for doc, score in zip(rerank_candidates, scores):
                    doc.metadata['rerank_score'] = score
                
                sorted_results = sorted(rerank_candidates, key=lambda d: d.metadata['rerank_score'], reverse=True)
                
                # On prend le top K documents *avant* de réorganiser
                top_k_docs = sorted_results[:config.TOP_K_RERANK]
                
                # 5. Réorganisation du contexte pour optimiser l'attention du LLM
                if DEBUG:
                    print(f"\n--- Étape 5: Réorganisation des {len(top_k_docs)} documents pour contrer le 'Lost in the Middle' ---")
                reorderer = LongContextReorder()
                reordered_docs = reorderer.transform_documents(top_k_docs)
            finally:
                del reranker
                torch.cuda.empty_cache()
                gc.collect()

            print("✅ Processus de récupération complet. Retour des documents réorganisés.")
            return reordered_docs  
