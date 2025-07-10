# scripts/rag_system/generation.py

import torch
import gc
from typing import List
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import config

class AnswerGenerator:
    """A class to handle the final answer generation using a powerful LLM."""

    def __init__(self):
        print("Initializing Answer Generator...")
        self.device = config.DEVICE
        self.model, self.tokenizer = None, None

    def _load_model(self):
        if self.model is None:
            print("  -> Loading SLM for answer generation...")
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.GENERATION_MODEL, quantization_config=quant_config, device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(config.GENERATION_MODEL)

    def _unload_model(self):
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model, self.tokenizer = None, None
            gc.collect()
            torch.cuda.empty_cache()
            print("     - Generation SLM unloaded.")

    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        """Generates a final answer based on the provided query and context documents."""
        try:
            self._load_model()
            
            # 1. Format the context string for the prompt
            context = "\n\n---\n\n".join([
                f"Source: {doc.metadata['source']}\n\n{doc.page_content}"
                for doc in context_docs
            ])
            
            # 2. Create the final prompt
            final_prompt = config.ANSWER_GENERATION_PROMPT.format(context=context, query=query)

            # 3. Generate the answer
            inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=1024, do_sample=True, temperature=0.2, top_p=0.95
                )
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the response part, after the prompt
            return answer.split("RÉPONSE:")[-1].strip()

        finally:
            self._unload_model()
