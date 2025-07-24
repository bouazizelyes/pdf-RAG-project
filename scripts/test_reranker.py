import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch_dtype = torch.float16  # Utilise FP16 pour stresser la VRAM plus efficacement

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "Qwen/Qwen3-Reranker-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch_dtype
).to(device).eval()

token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")

max_length = 8192

prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )
    return output

def process_inputs(pairs):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(device)
    return inputs

@torch.no_grad()
def compute_logits(inputs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

# --- 🚨 Stress test: Génère un batch énorme ---
task = 'Given a web search query, retrieve relevant passages that answer the query'

# ➜ Ici on génère un batch énorme (~500 pairs)
queries = ["What is the capital of China?"] * 50
documents = ["China is a big Country. It has a good capital for research."] * 50

pairs = [format_instruction(task, query, doc) for query, doc in zip(queries, documents)]

# Tokenize & forward
inputs = process_inputs(pairs)
scores = compute_logits(inputs)

print("scores: ", scores[:10])  # Affiche seulement les 10 premiers scores

# Nettoyage
del model
del tokenizer
torch.cuda.empty_cache()
print("Stress test done. Memory freed.")

