# --- Install/Import Dependencies ---
# pip install sentence-transformers torch scikit-learn tqdm bitsandbytes transformers

import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import collections
from torch.utils.data import Sampler
from transformers import Adafactor, AutoTokenizer, AutoModelForSequenceClassification
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Set environment variable before any PyTorch/CUDA imports if needed
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# --- Configuration ---
DATA_PATH = PROJECT_ROOT / "data/questions/mistral_generated_1.json"  # <-- CHANGE TO YOUR LOCAL FILE PATH
OUTPUT_DIR = "./output_model"          # <-- CHANGE TO YOUR DESIRED OUTPUT DIRECTORY
MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"

EPOCHS = 12
BATCH_SIZE = 8  # You might want to increase this depending on your GPU memory
VAL_SPLIT = 0.2

# Enable TF32 for faster training on Ampere GPUs (RTX 30xx, A100, etc.)
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("TF32 enabled.")

# --- Logging (Simple Print) ---
def log_info(message):
    print(message)

# --- Core Code ---

class GroupedTopicSampler(Sampler):
    """
    Custom Sampler for curriculum learning based on data order and topics.
    """
    def __init__(self, data_source, transition_epoch):
        self.data_source = data_source
        self.transition_epoch = transition_epoch
        self.epoch = 0
        self.topic_to_indices = collections.defaultdict(list)
        for idx, item in enumerate(data_source):
            topic_id = item.get('source_topic_id', f"topic_{idx // 10}")
            self.topic_to_indices[topic_id].append(idx)
        self.topic_ids = list(self.topic_to_indices.keys())

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        if self.epoch < self.transition_epoch:
            log_info(f"Sampler: Using STAGE 1 (Grouped by Topic) for epoch {self.epoch + 1}")
            shuffled_topic_ids = self.topic_ids.copy()
            random.shuffle(shuffled_topic_ids)
            indices = []
            for topic_id in shuffled_topic_ids:
                topic_indices = self.topic_to_indices[topic_id].copy()
                random.shuffle(topic_indices)
                indices.extend(topic_indices)
            yield from indices
        else:
            log_info(f"Sampler: Using STAGE 2 (Global Shuffle) for epoch {self.epoch + 1}")
            indices = list(range(len(self.data_source)))
            random.shuffle(indices)
            yield from indices

    def __len__(self):
        return len(self.data_source)

class CrossEncoderDataset(Dataset):
    """
    Dataset that samples negatives based on curriculum learning.
    Assumes data items have 'simple_negatives' (list) and 'hard_negatives' (list).
    """
    def __init__(self, data, epoch, total_epochs, num_simple_negs=10, num_hard_negs=3):
        self.data = data
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.hard_introduction_epoch = 3
        self.num_simple_negs = num_simple_negs
        self.num_hard_negs = num_hard_negs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        query = entry['query']
        positive = entry['positive']

        if self.epoch < self.hard_introduction_epoch:
            available_simple = entry.get('simple_negatives', [])
            sampled_negatives = random.sample(
                available_simple,
                min(self.num_simple_negs, len(available_simple))
            ) if available_simple else []
        else:
            available_simple = entry.get('simple_negatives', [])
            available_hard = entry.get('hard_negatives', [])
            sampled_simple = random.sample(
                available_simple,
                min(self.num_simple_negs, len(available_simple))
            ) if available_simple else []
            sampled_hard = random.sample(
                available_hard,
                min(self.num_hard_negs, len(available_hard))
            ) if available_hard else []
            sampled_negatives = sampled_simple + sampled_hard

        if not sampled_negatives:
             sampled_negatives = ["no_negative_placeholder"]

        return {
            'query': query,
            'positive': positive,
            'negatives': sampled_negatives
        }

# --- CRITICAL CHANGE: Simplified and Robust Loss Function ---
def compute_contrastive_loss(batch, model, tokenizer, device, current_epoch, total_epochs, max_length=512):
    """
    Computes contrastive loss using multiple negatives per query.
    Integrates scoring directly to ensure gradient flow.
    """
    queries = batch['query']
    positives = batch['positive']
    negatives_list = batch['negatives']
    batch_size = len(queries)

    if batch_size == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 1. Prepare all pairs for scoring
    all_pairs = []
    query_to_pair_indices = {}
    for i in range(batch_size):
        q = queries[i]
        p = positives[i]
        negs = negatives_list[i]
        pos_idx = len(all_pairs)
        all_pairs.append((q, p))
        neg_indices = []
        for n in negs:
            neg_idx = len(all_pairs)
            all_pairs.append((q, n))
            neg_indices.append(neg_idx)
        query_to_pair_indices[i] = {'pos': pos_idx, 'negs': neg_indices}

    if not all_pairs:
         return torch.tensor(0.0, device=device, requires_grad=True)

    # 2. Tokenize all pairs together
    texts_a = [pair[0] for pair in all_pairs] # queries
    texts_b = [pair[1] for pair in all_pairs] # passages (pos/neg)
    encoded = tokenizer(
        texts_a,
        texts_b,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    # --- CRITICAL: FORWARD PASS - GRADIENTS MUST FLOW HERE ---
    # Explicitly call the model's forward method
    outputs = model(**encoded) # This creates the computation graph
    all_scores_logits = outputs.logits # Shape: [B, 1] or [total_pairs, 1] depending on model
    # Ensure it's a flat tensor [total_pairs]
    all_scores = all_scores_logits.view(-1) # Shape: [total_number_of_pairs]

    # --- DEBUG: Check if scores require grad ---
    # print(f"[DEBUG] Scores require grad: {all_scores.requires_grad}")

    # 3. Calculate loss for each query
    losses = []
    for i in range(batch_size):
        pos_idx = query_to_pair_indices[i]['pos']
        neg_indices = query_to_pair_indices[i]['negs']
        if not neg_indices: continue
        pos_score = all_scores[pos_idx]
        neg_scores = all_scores[neg_indices]
        # Adaptive Margin
        margin = 1.0 + (current_epoch / total_epochs) * 0.5
        # Calculate loss for this query - Ensure operations support gradients
        # Use torch.clamp instead of relu for potentially better numerical stability (optional)
        # loss_per_query = torch.clamp(neg_scores - pos_score + margin, min=0.0)
        loss_per_query = torch.relu(neg_scores - pos_score + margin)
        mean_loss_per_query = torch.mean(loss_per_query)
        losses.append(mean_loss_per_query)

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Average loss across queries in the batch
    final_loss = torch.mean(torch.stack(losses))
    # --- DEBUG: Check if final loss requires grad ---
    # print(f"[DEBUG] Final Loss requires grad: {final_loss.requires_grad}")
    return final_loss


def train_cross_encoder(data_path, model_name, output_dir, epochs, batch_size, val_split):
    """Main training function."""
    log_info("Loading data...")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except FileNotFoundError:
        log_info(f"Error: Data file not found at {data_path}")
        return [], []
    except json.JSONDecodeError as e:
        log_info(f"Error: Failed to decode JSON from {data_path}: {e}")
        return [], []

    train_data, val_data = train_test_split(full_data, test_size=val_split, random_state=42)
    log_info(f"Data loaded. Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_info(f"Using device: {device}")

    # --- LOAD MODEL AND TOKENIZER DIRECTLY ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.to(device)

    # Handle DataParallel for multi-GPU setups
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    # Handle pad_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            if isinstance(model, torch.nn.DataParallel):
                 model.module.config.pad_token_id = tokenizer.eos_token_id
                 print(f"Set pad_token to eos_token (DataParallel): {tokenizer.eos_token}")
            else:
                model.config.pad_token_id = tokenizer.eos_token_id
                print(f"Set pad_token to eos_token: {tokenizer.eos_token}")
        else:
            pad_token = "<|pad|>"
            tokenizer.add_special_tokens({'pad_token': pad_token})
            if isinstance(model, torch.nn.DataParallel):
                model.module.config.pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
                model.module.resize_token_embeddings(len(tokenizer))
            else:
               model.config.pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
               model.resize_token_embeddings(len(tokenizer))
            print(f"Added new pad_token: {pad_token}")

    config_to_check = model.module.config if isinstance(model, torch.nn.DataParallel) else model.config
    if config_to_check.pad_token_id is None:
        print("Warning: pad_token_id is still None. Setting to 0 (might be unsafe).")
        config_to_check.pad_token_id = 0
    print(f"Tokenizer pad_token: {tokenizer.pad_token}, pad_token_id: {config_to_check.pad_token_id}")

    # Optimizer - Get parameters correctly
    model_params = model.parameters() if not isinstance(model, torch.nn.DataParallel) else model.module.parameters()
    optimizer = Adafactor(model_params, lr=1e-5, relative_step=False, scale_parameter=False, warmup_init=False)

    # Scheduler
    total_steps = epochs * (len(train_data) // batch_size) if batch_size > 0 else epochs * len(train_data)
    warmup_steps = int(0.1 * total_steps) if total_steps > 0 else 1

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step / warmup_steps) if warmup_steps > 0 else 1.0
        else:
            return 0.95 ** (current_step - warmup_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3
    train_losses = []
    val_losses = []

    # Curriculum learning setup
    transition_epoch = int(0.4 * epochs)
    train_sampler = GroupedTopicSampler(train_data, transition_epoch=transition_epoch)

    os.makedirs(output_dir, exist_ok=True)

    log_info("Starting training loop...")
    for epoch in range(epochs):
        log_info(f"--- Epoch {epoch+1}/{epochs} ---")

        # --- CRITICAL: Ensure model is in training mode ---
        model.train()
        train_sampler.set_epoch(epoch)
        train_dataset = CrossEncoderDataset(train_data, epoch, epochs, num_simple_negs=10, num_hard_negs=3)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=False
        )

        total_train_loss = 0
        train_steps = 0
        train_pbar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}", leave=False)
        for batch in train_pbar:
            optimizer.zero_grad()
            # --- CRITICAL: Pass tokenizer and ensure model.train() is active ---
            loss = compute_contrastive_loss(batch, model, tokenizer, device, epoch, epochs)
            # --- CRITICAL: Check if loss requires grad BEFORE calling backward ---
            if not loss.requires_grad:
                 print(f"[ERROR] Loss tensor does not require grad at epoch {epoch+1}. Skipping backward.")
                 # Optionally, raise an error to stop execution
                 # raise RuntimeError("Loss tensor does not require grad!")
                 # Or continue, but this indicates a fundamental problem
                 continue # Skip this step
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_steps += 1
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()
        avg_train_loss = total_train_loss / train_steps if train_steps > 0 else 0
        train_losses.append(avg_train_loss)

        # --- CRITICAL: Ensure model is in eval mode for validation ---
        model.eval()
        val_dataset = CrossEncoderDataset(val_data, epoch, epochs, num_simple_negs=10, num_hard_negs=3)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        total_val_loss = 0
        val_steps = 0
        val_pbar = tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}", leave=False)
        with torch.no_grad(): # No gradients needed during validation
            for batch in val_pbar:
                # Still compute loss for monitoring, but no backprop
                loss = compute_contrastive_loss(batch, model, tokenizer, device, epoch, epochs)
                total_val_loss += loss.item()
                val_steps += 1
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_val_loss = total_val_loss / val_steps if val_steps > 0 else 0
        val_losses.append(avg_val_loss)

        log_info(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save model correctly
            model_save_path = os.path.join(output_dir, "best_model")
            # Save using the underlying model and tokenizer
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            model_to_save.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            log_info(f"  -> New best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log_info(f"Stopping early at epoch {epoch+1} due to no improvement.")
                break

    log_info("Training completed.")
    return train_losses, val_losses

# --- Run Training ---
if __name__ == "__main__":
    log_info("Starting training process...")
    try:
        train_losses, val_losses = train_cross_encoder(
            DATA_PATH,
            MODEL_NAME,
            OUTPUT_DIR,
            EPOCHS,
            BATCH_SIZE,
            VAL_SPLIT
        )
        log_info("Training script finished. Check the output directory for the saved model.")
        log_info(f"Final Training Loss: {train_losses[-1] if train_losses else 'N/A'}")
        log_info(f"Best Validation Loss: {min(val_losses) if val_losses else 'N/A'}")
    except Exception as e:
        log_info(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

