# scripts/rag_system/generation.py

import torch,re
import gc
from typing import List,Tuple
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import PreTrainedTokenizer
import config
from config import DEBUG

class AnswerGenerator:
    """A class to handle the final answer generation using a powerful LLM."""

    def __init__(self):
        if DEBUG:
            print("Initializing Answer Generator...")
        self.device = config.DEVICE
        self.model, self.tokenizer = None, None
    
    def _load_model(self):
        if self.model is None:
            if DEBUG:
                print("  -> Loading SLM for answer generation...")
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.GENERATION_MODEL, quantization_config=quant_config, device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(config.GENERATION_MODEL)
            

    """
    def _load_model(self):
        if self.model is None:
            print("  -> Loading SLM for answer generation...")
            self.model = AutoModelForCausalLM.from_pretrained(
                config.GENERATION_MODEL, device_map="auto", torch_dtype="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(config.GENERATION_MODEL)"""

    def _unload_model(self):
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model, self.tokenizer = None, None
            gc.collect()
            torch.cuda.empty_cache()
            if DEBUG:
                print("- Generation SLM unloaded.")




    # Définissez cette fonction utilitaire dans votre classe ou en dehors
    def _parse_qwen_thinking_output(self, generated_ids: List[int], tokenizer: PreTrainedTokenizer) -> Tuple[str, str]:
        """
        Parses the output from a Qwen3 model when enable_thinking=True is used.
        Returns a tuple of (thinking_content, final_content).
        """
        try:
            # L'ID du token </think> pour Qwen3 est 151668
            think_token_id = tokenizer.convert_tokens_to_ids("</think>")
            
            # Trouver l'index de la fin du bloc <think>
            index = len(generated_ids) - generated_ids[::-1].index(think_token_id)
        except (ValueError, IndexError):
            # Le tag </think> n'a pas été trouvé ou la liste est vide
            index = 0

        thinking_content = tokenizer.decode(generated_ids[:index], skip_special_tokens=True)
        final_content = tokenizer.decode(generated_ids[index:], skip_special_tokens=True)

        return thinking_content.strip(), final_content.strip()

    # Votre fonction de classe modifiée
    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        """Generates a final answer based on the provided query and context documents."""
        try:
            self._load_model()
            
            # 1. Format context
            context = "\n\n---\n\n".join([
                f"Source: {doc.metadata['source']}\n\n{doc.page_content}"
                for doc in context_docs
            ])
            
            # 2. Create messages for the prompt
            messages = [
                {"role": msg["role"], "content": msg["content"].format(context=context, query=query)}
                for msg in config.ANSWER_GENERATION_PROMPT
            ]
                
            # 3. Apply chat template with thinking enabled
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True 
            )
            
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            input_length = inputs.input_ids.shape[1]

            # 4. Generate output
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.6, 
                    top_p=0.95,
                    top_k=20,
                    min_p=0,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # 5. --- NOUVELLE LOGIQUE DE PARSING ---
            # Isoler uniquement les tokens générés
            generated_ids = outputs[0][input_length:].tolist()
            
            # Utiliser la fonction de parsing dédiée au mode "thinking"
            thinking_process, final_answer = self._parse_qwen_thinking_output(generated_ids, self.tokenizer)
            if DEBUG:
                print("============== THINKING PROCESS ==============")
                print(thinking_process)
                print("============================================")
            
            return final_answer

        finally:
            self._unload_model()
