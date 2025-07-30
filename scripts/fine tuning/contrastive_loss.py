# === Local Setup Instructions ===
# 1. Ensure Python and pip are installed.
# 2. Install required packages:
#    pip install torch sentence-transformers scikit-learn tqdm transformers
# 3. Place your training data file 'mistral_generated_1.json' in the same directory as this script,
#    or update DATA_PATH accordingly.
# 4. Run the script: python your_script_name.py

# --- Install/Import Dependencies ---
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
# Import CrossEncoder from sentence_transformers
from sentence_transformers import CrossEncoder
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import collections
from torch.utils.data import Sampler
# Import Adafactor from transformers.optimization
from transformers import Adafactor
from torch.cuda.amp import autocast, GradScaler
import os

# --- Configuration ---
# Update these paths for your local setup
# DATA_PATH = "mistral_generated_1.json" # If the file is in the same directory
DATA_PATH = "./mistral_generated_1.json" # Example relative path
OUTPUT_DIR = "./output_model_local"
MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"

# Training hyperparameters
EPOCHS = 15
BATCH_SIZE = 4 # Increased from 1 for efficiency
VAL_SPLIT = 0.2

# Enable TF32 for faster training on Ampere GPUs (optional, but often beneficial)
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8: # Check for Ampere+
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("TF32 enabled.")

# --- Logging (Simple Print) ---
def log_info(message):
    print(message)

# --- Core Code (Adapted for Local) ---

# (The core classes and functions remain the same)

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
            topic_id = item.get('source_topic_id', 'unknown')
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
        # --- Hard Negative Introduction Logic ---
        # Introduce hard negatives after epoch 3 (0-indexed, so epochs 0,1,2 are simple only)
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
            # Use only simple negatives for the first few epochs
            available_simple = entry.get('simple_negatives', [])
            sampled_negatives = random.sample(
                available_simple,
                min(self.num_simple_negs, len(available_simple))
            ) if available_simple else []
        else:
            # Introduce hard negatives from epoch 3 onwards
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

        # Ensure we have negatives
        if not sampled_negatives:
             sampled_negatives = ["no_negative_placeholder"]

        return {
            'query': query,
            'positive': positive,
            'negatives': sampled_negatives
        }


def get_scores(pairs, model, device):
    """Compute scores for a list of (query, passage) pairs."""
    if not pairs:
        return torch.empty(0, device=device)
    # Use the model's tokenizer and underlying model directly
    encoded = model.tokenizer(
        [q for q,_ in pairs],
        [p for _,p in pairs],
        padding=True,
        truncation=True,
        max_length=model.max_length,
        return_tensors="pt",
    ).to(device)
    
    outputs = model.model(**encoded)
    return outputs.logits.view(-1)


def contrastive_loss_fn_multi_neg(batch, model, device, current_epoch, total_epochs):
    """
    Computes contrastive loss using multiple negatives per query.
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

    # 2. Score all pairs in one go
    all_scores = get_scores(all_pairs, model, device)

    # 3. Calculate loss for each query
    losses = []
    for i in range(batch_size):
        pos_idx = query_to_pair_indices[i]['pos']
        neg_indices = query_to_pair_indices[i]['negs']
        if not neg_indices: continue # Skip if no negatives
        pos_score = all_scores[pos_idx]
        neg_scores = all_scores[neg_indices]
        # Adaptive Margin
        margin = 1.0 + (current_epoch / total_epochs) * 0.5
        loss_per_query = torch.mean(torch.relu(neg_scores - pos_score + margin))
        losses.append(loss_per_query)

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)

    final_loss = torch.mean(torch.stack(losses))
    return final_loss


def train_cross_encoder(data_path, model_name, output_dir, epochs, batch_size, val_split):
    """Main training function."""
    log_info("Loading data...")
    with open(data_path, 'r', encoding='utf-8') as f: # Specify encoding for robustness
        full_data = json.load(f)
    train_data, val_data = train_test_split(full_data, test_size=val_split, random_state=42)
    log_info(f"Data loaded. Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_info(f"Using device: {device}")

    # Initialize the CrossEncoder model
    model = CrossEncoder(model_name, max_length=512, trust_remote_code=True)
    model.model.to(device)

    # Handle multi-GPU (DataParallel)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model.model = torch.nn.DataParallel(model.model) # Wrap the underlying model

    # Handle pad_token
    if model.tokenizer.pad_token is None:
        # Check if eos_token exists and use it
        if model.tokenizer.eos_token:
            model.tokenizer.pad_token = model.tokenizer.eos_token
            # Access the config through .module if DataParallel is used
            if isinstance(model.model, torch.nn.DataParallel):
                model.model.module.config.pad_token_id = model.tokenizer.eos_token_id
                print(f"Set pad_token to eos_token (DataParallel): {model.tokenizer.eos_token}")
            else:
                model.model.config.pad_token_id = model.tokenizer.eos_token_id
                print(f"Set pad_token to eos_token: {model.tokenizer.eos_token}")
        else:
            # Fallback: Add a new pad token
            pad_token = "<|pad|>"
            model.tokenizer.add_special_tokens({'pad_token': pad_token})
            # Resize embeddings on the correct model instance
            if isinstance(model.model, torch.nn.DataParallel):
                model.model.module.config.pad_token_id = model.tokenizer.convert_tokens_to_ids(pad_token)
                model.model.module.resize_token_embeddings(len(model.tokenizer))
            else:
                model.model.config.pad_token_id = model.tokenizer.convert_tokens_to_ids(pad_token)
                model.model.resize_token_embeddings(len(model.tokenizer))
            print(f"Added new pad_token: {pad_token}")

    # Verify pad_token_id is set correctly
    config_to_check = model.model.module.config if isinstance(model.model, torch.nn.DataParallel) else model.model.config
    if config_to_check.pad_token_id is None:
        print("Warning: pad_token_id is still None. Setting to 0 (might be unsafe).")
        config_to_check.pad_token_id = 0
    print(f"Tokenizer pad_token: {model.tokenizer.pad_token}, pad_token_id: {config_to_check.pad_token_id}")

    # --- Optimizer Setup ---
    # Use Adafactor with automatic learning rate (relative_step=True) and remove the scheduler
    optimizer = Adafactor(
        model.model.parameters(), # Use model.model.parameters()
        relative_step=True,     # Enables automatic learning rate scheduling
        scale_parameter=True,
        warmup_init=True,
        weight_decay=1e-3 # Optional: Add weight decay if needed
    )

    # --- Early Stopping and Tracking ---
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3
    train_losses = []
    val_losses = []

    # --- Curriculum Learning Setup ---
    transition_epoch = int(0.5 * epochs)
    train_sampler = GroupedTopicSampler(train_data, transition_epoch=transition_epoch)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    log_info("Starting training loop...")
    for epoch in range(epochs):
        log_info(f"--- Epoch {epoch+1}/{epochs} ---")
        # --- Training Phase ---
        model.model.train()
        train_sampler.set_epoch(epoch)
        train_dataset = CrossEncoderDataset(train_data, epoch, epochs, num_simple_negs=10, num_hard_negs=3)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=False # Sampler handles shuffling
        )

        total_train_loss = 0
        train_steps = 0
        # Use tqdm for progress bar
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}", leave=False)
        for batch in progress_bar:
            optimizer.zero_grad()
            # Forward pass and loss calculation
            loss = contrastive_loss_fn_multi_neg(batch, model, device, epoch, epochs)
            # Backward pass
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_steps += 1
            # Update progress bar description with current loss
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_train_loss / train_steps if train_steps > 0 else 0
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.model.eval()
        val_dataset = CrossEncoderDataset(val_data, epoch, epochs, num_simple_negs=10, num_hard_negs=3)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        total_val_loss = 0
        val_steps = 0
        with torch.no_grad():
            # Use tqdm for validation progress bar
            for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}", leave=False):
                loss = contrastive_loss_fn_multi_neg(batch, model, device, epoch, epochs)
                total_val_loss += loss.item()
                val_steps += 1

        avg_val_loss = total_val_loss / val_steps if val_steps > 0 else 0
        val_losses.append(avg_val_loss)

        # --- Logging ---
        log_info(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | ")

        # --- Early Stopping Check ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the model
            if isinstance(model.model, torch.nn.DataParallel):
                # Save the underlying model (module) if DataParallel is used
                model.model.module.save_pretrained(output_dir)
                model.tokenizer.save_pretrained(output_dir)
                log_info(f"  -> New best model saved (DataParallel - underlying model & tokenizer) (Val Loss: {best_val_loss:.4f})")
            else:
                # Save normally if not using DataParallel
                # The sentence_transformers CrossEncoder save method handles this
                model.save(output_dir)
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
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        log_info(f"Error: Data file not found at {DATA_PATH}")
        log_info("Please ensure your 'mistral_generated_1.json' is in the correct location and DATA_PATH is set correctly.")
    else:
        log_info("Starting training process...")
        try:
            train_losses, val_losses = train_cross_encoder(
                DATA_PATH,
                MODEL_NAME,
                OUTPUT_DIR,
                EPOCHS,
                BATCH_SIZE,
                VAL_SPLIT,
            )
            log_info("Training script finished. Check the output directory for the saved model.")
            # Optional: Print final losses
            log_info(f"Final Training Loss: {train_losses[-1] if train_losses else 'N/A'}")
            log_info(f"Best Validation Loss: {min(val_losses) if val_losses else 'N/A'}")
        except Exception as e:
            log_info(f"An error occurred during training: {e}")
            import traceback
            traceback.print_exc()
