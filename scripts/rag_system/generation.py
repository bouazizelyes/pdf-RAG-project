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
            
            # Use different loading strategy based on device
            if self.device.type == "cpu":
                # CPU-friendly loading without quantization
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.GENERATION_MODEL,
                    torch_dtype=torch.float32,  # Use float32 for CPU
                    low_cpu_mem_usage=True,      # Reduce memory usage
                    device_map="auto" if hasattr(torch, 'cpu') else None
                )
            else:
                # GPU loading with quantization (original approach)
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
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all operations to complete
            if DEBUG:
                print("- Generation SLM unloaded.")

    # Définissez cette fonction utilitaire dans votre classe ou en dehors
    def _parse_qwen_thinking_output(self, generated_ids: List[int], tokenizer: PreTrainedTokenizer) -> Tuple[str, str]:
        """
        Parses the output from a Qwen3 model when enable_thinking=True is used.
        Returns a tuple of (thinking_content, final_content).
        """
        try:
            # L'ID du token  </think> pour Qwen3 est 151668
            think_token_id = tokenizer.convert_tokens_to_ids("</think>")
            
            # Trouver l'index de la fin du bloc <think>
            index = len(generated_ids) - generated_ids[::-1].index(think_token_id)
        except (ValueError, IndexError):
            # Le tag  </think> n'a pas été trouvé ou la liste est vide
            index = 0

        thinking_content = tokenizer.decode(generated_ids[:index], skip_special_tokens=True)
        final_content = tokenizer.decode(generated_ids[index:], skip_special_tokens=True)

        return thinking_content.strip(), final_content.strip()

    # Votre fonction de classe modifiée
    # In your generation.py, add more debug prints:
    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        """Generates a final answer based on the provided query and context documents."""
        try:
            print("🔍 Starting answer generation...")
            self._load_model()
            print("✅ Model loaded successfully")
            
            # 1. Format context
            context = "\n\n---\n\n".join([
                f"Source: {doc.metadata['source']}\n\n{doc.page_content}"
                for doc in context_docs
            ])
            print(f"📄 Context formatted ({len(context)} characters)")
            
            # 2. Create messages for the prompt
            messages = [
                {"role": msg["role"], "content": msg["content"].format(context=context, query=query)}
                for msg in config.ANSWER_GENERATION_PROMPT
            ]
            print("💬 Messages created")
                
            # 3. Apply chat template with thinking enabled
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True 
            )
            print("📝 Chat template applied")
            
            # CPU optimization: limit input length to prevent memory issues
            inputs = self.tokenizer(
                [text], 
                return_tensors="pt", 
                truncation=True,
                max_length=2048 if self.device.type == "cpu" else None
            ).to(self.device)
            input_length = inputs.input_ids.shape[1]
            print(f"📥 Inputs prepared (length: {input_length})")

            # 4. Generate output with CPU-friendly parameters
            print("🚀 Generating response...")
            with torch.no_grad():
                generate_kwargs = {
                    "max_new_tokens": 512 if self.device.type == "cpu" else 512,  # Reduce for CPU
                    "do_sample": True,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "min_p":0,
                    "eos_token_id": self.tokenizer.eos_token_id
                    
                }
                
                # Add additional CPU optimizations
                if self.device.type == "cpu":
                    generate_kwargs.update({
                        "num_beams": 1,  # Disable beam search for speed
                        "early_stopping": True
                    })
                
                outputs = self.model.generate(**inputs, **generate_kwargs)
            print("✅ Generation completed")

            # 5. Parse the output
            generated_ids = outputs[0][input_length:].tolist()
            thinking_process, final_answer = self._parse_qwen_thinking_output(generated_ids, self.tokenizer)
            
            if DEBUG:
                print("============== THINKING PROCESS ==============")
                print(thinking_process)
                print("============================================")
            
            print("✅ Answer parsed successfully")
            return final_answer

        except Exception as e:
            print(f"❌ Error in generate_answer: {e}")
            import traceback
            traceback.print_exc()
            # Fallback response
            return "D'après les documents fournis, je ne peux pas répondre à cette question."
        finally:
            self._unload_model()
            print("🧹 Model unloaded")
