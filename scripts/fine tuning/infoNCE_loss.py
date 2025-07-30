# === Kaggle-Specific Setup ===
# 1. Upload your 'mistral_generated_1.json' file as a Kaggle dataset.
#    Let's assume the dataset is named 'my-training-data' after upload.
#    The file will then be accessible at: /kaggle/input/my-training-data/mistral_generated_1.json
# 2. Add the output directory to Kaggle's working directory: /kaggle/working/output_model

# --- Install/Import Dependencies ---
# These are usually pre-installed, but good to be sure
# !pip install sentence-transformers # Uncomment if needed and internet is enabled
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import CrossEncoder
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import collections
from torch.utils.data import Sampler
from transformers import Adafactor
import os # For handling paths
import torch.nn.functional as F # For InfoNCE loss (log_softmax)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data/questions/mistral_generated_1.json" # <-- CHANGE 'my-training-data' TO YOUR DATASET NAME
OUTPUT_DIR = "./output_model_temp_search"
MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"

# --- Hyperparameter Search Configuration ---
TEMP_SEARCH_DIR = os.path.join(OUTPUT_DIR, "temp_search_trials")
NUM_TEMP_TRIALS = 5  # Number of random temperature values to try
TEMP_MIN = 0.02     # Minimum temperature for random search
TEMP_MAX = 0.6      # Maximum temperature for random search
# Ensure TEMP_MIN < TEMP_MAX

# --- Short Training Run Configuration (for hyperparameter search) ---
SHORT_TRAIN_EPOCHS = 4 # Number of epochs for each hyperparameter trial

# --- Full Training Configuration (after finding best temp) ---
FULL_TRAIN_EPOCHS = 15
FULL_TRAIN_BATCH_SIZE = 6 # You might want to increase this depending on your GPU memory
VAL_SPLIT = 0.2

# --- Logging (Simple Print for Kaggle) ---
def log_info(message):
    print(message)

# --- Core Code (Adapted for Kaggle) ---

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

# --- NEW FUNCTION: InfoNCE Loss Implementation ---
def infonce_loss_fn_multi_neg(batch, model, device, temperature):
    """
    Computes InfoNCE loss using multiple negatives per query.
    Effectively treats each (query, positive) pair as a separate classification task
    where the positive is the correct class and all negatives for that query are incorrect classes.

    Args:
        batch: A batch from CrossEncoderDataset.
        model: The CrossEncoder model instance.
        device: The device (cuda/cpu) to run on.
        temperature: Temperature scaling factor (tau).

    Returns:
        torch.Tensor: The computed scalar loss.
    """
    queries = batch['query']
    positives = batch['positive']
    negatives_list = batch['negatives']
    batch_size = len(queries)

    if batch_size == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 1. Prepare all pairs for scoring (Positive first for each query)
    all_pairs = []
    query_group_boundaries = [0] # Track where each query's group of pairs starts in all_pairs

    for i in range(batch_size):
        q = queries[i]
        p = positives[i]
        negs = negatives_list[i]

        # Add positive pair first
        all_pairs.append((q, p))

        # Add negative pairs
        for n in negs:
            all_pairs.append((q, n))

        # Mark the end of this query's group (start of next query's group)
        query_group_boundaries.append(len(all_pairs))

    if not all_pairs:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 2. Score all pairs in one go
    encoded = model.tokenizer(
        [q for q, _ in all_pairs],
        [p for _, p in all_pairs],
        padding=True,
        truncation=True,
        max_length=model.max_length,
        return_tensors="pt",
    ).to(device)

    outputs = model.model(**encoded)
    all_scores_logits = outputs.logits.view(-1) # Shape: [total_number_of_pairs]

    # 3. Calculate InfoNCE loss for each query group
    losses = []
    for i in range(batch_size):
        start_idx = query_group_boundaries[i]
        end_idx = query_group_boundaries[i + 1]
        if start_idx >= end_idx:
            continue # Should not happen if there are negatives

        # Extract scores for this query's group
        group_scores = all_scores_logits[start_idx:end_idx] # Shape: [1 + num_negs_for_this_query]

        # Apply temperature scaling
        scaled_scores = group_scores / temperature

        # The first score in the group is the positive (by construction)
        # We want to maximize the probability assigned to the positive class (index 0)
        # Compute log-softmax over the group scores
        log_probs = F.log_softmax(scaled_scores, dim=0) # Shape: [1 + num_negs]

        # The log-probability of the positive class (index 0) is the loss contribution for this query
        # InfoNCE loss is the negative log-probability of the correct class
        loss_per_query = -log_probs[0] # Scalar
        losses.append(loss_per_query)

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Average loss across all queries in the batch
    final_loss = torch.mean(torch.stack(losses)) # Scalar
    return final_loss

# --- MODIFIED FUNCTION: Training Loop for Hyperparameter Search ---
def train_cross_encoder_temp_search(data_path, model_name, output_dir_base, num_trials, temp_min, temp_max, short_epochs, batch_size, val_split):
    """Performs random search for the InfoNCE temperature hyperparameter."""
    log_info("Loading data for hyperparameter search...")
    with open(data_path) as f:
        full_data = json.load(f)
    train_data, val_data = train_test_split(full_data, test_size=val_split, random_state=42)
    log_info(f"Data loaded for search. Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_info(f"Using device: {device}")

    results = []
    os.makedirs(output_dir_base, exist_ok=True)

    for trial in range(num_trials):
        # --- Randomly Sample Temperature ---
        # Sample uniformly from the log space for potentially better exploration
        # log_temp = random.uniform(np.log(temp_min), np.log(temp_max))
        # sampled_temp = np.exp(log_temp)
        # Or sample uniformly from the linear space:
        sampled_temp = random.uniform(temp_min, temp_max)
        log_info(f"--- Trial {trial+1}/{num_trials} | Temperature: {sampled_temp:.4f} ---")

        # Create a unique output directory for this trial
        trial_output_dir = os.path.join(output_dir_base, f"trial_{trial+1}_temp_{sampled_temp:.4f}")
        os.makedirs(trial_output_dir, exist_ok=True)

        # --- Initialize Model for Trial ---
        model = CrossEncoder(model_name, max_length=512)
        model.model.to(device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model.model = torch.nn.DataParallel(model.model)

        # Handle pad_token (same as before)
        if model.tokenizer.pad_token is None:
            if model.tokenizer.eos_token:
                model.tokenizer.pad_token = model.tokenizer.eos_token
                if isinstance(model.model, torch.nn.DataParallel):
                    model.model.module.config.pad_token_id = model.tokenizer.eos_token_id
                    print(f"Set pad_token to eos_token (DataParallel): {model.tokenizer.eos_token}")
                else:
                    model.model.config.pad_token_id = model.tokenizer.eos_token_id
                    print(f"Set pad_token to eos_token: {model.tokenizer.eos_token}")
            else:
                pad_token = "<|pad|>"
                model.tokenizer.add_special_tokens({'pad_token': pad_token})
                if isinstance(model.model, torch.nn.DataParallel):
                    model.model.module.config.pad_token_id = model.tokenizer.convert_tokens_to_ids(pad_token)
                    model.model.module.resize_token_embeddings(len(model.tokenizer))
                else:
                    model.model.config.pad_token_id = model.tokenizer.convert_tokens_to_ids(pad_token)
                    model.model.resize_token_embeddings(len(model.tokenizer))
                print(f"Added new pad_token: {pad_token}")

        config_to_check = model.model.module.config if isinstance(model.model, torch.nn.DataParallel) else model.model.config
        if config_to_check.pad_token_id is None:
            print("Warning: pad_token_id is still None. Setting to 0 (might be unsafe).")
            config_to_check.pad_token_id = 0
        print(f"Tokenizer pad_token: {model.tokenizer.pad_token}, pad_token_id: {config_to_check.pad_token_id}")

        optimizer = Adafactor(model.parameters(), lr=1e-5, relative_step=False, scale_parameter=False, warmup_init=False)

        # Simple scheduler for short runs (optional, or fixed LR)
        # For simplicity here, we can use a basic scheduler or even no scheduler for short runs
        total_steps = short_epochs * (len(train_data) // batch_size)
        warmup_steps = int(0.1 * total_steps) if total_steps > 0 else 1
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step / warmup_steps) if warmup_steps > 0 else 1.0
            else:
                return 0.95 ** (current_step - warmup_steps) # Simple decay
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # --- Curriculum Learning Setup (for short run) ---
        transition_epoch = int(0.4 * short_epochs)
        train_sampler = GroupedTopicSampler(train_data, transition_epoch=transition_epoch)

        best_trial_val_loss = float('inf')
        # --- Short Training Loop ---
        for epoch in range(short_epochs):
            log_info(f"  Trial {trial+1} | Epoch {epoch+1}/{short_epochs}")
            # --- Training Phase ---
            model.model.train()
            train_sampler.set_epoch(epoch)
            train_dataset = CrossEncoderDataset(train_data, epoch, short_epochs, num_simple_negs=10, num_hard_negs=3)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                shuffle=False
            )
            total_train_loss = 0
            train_steps = 0
            for batch in tqdm(train_dataloader, desc=f"  Train Trial {trial+1} Ep {epoch+1}", leave=False):
                optimizer.zero_grad()
                loss = infonce_loss_fn_multi_neg(batch, model, device, sampled_temp)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                train_steps += 1
            scheduler.step()

            # --- Validation Phase ---
            model.model.eval()
            val_dataset = CrossEncoderDataset(val_data, epoch, short_epochs, num_simple_negs=10, num_hard_negs=3)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            total_val_loss = 0
            val_steps = 0
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f"  Val Trial {trial+1} Ep {epoch+1}", leave=False):
                    loss = infonce_loss_fn_multi_neg(batch, model, device, sampled_temp)
                    total_val_loss += loss.item()
                    val_steps += 1
            avg_val_loss = total_val_loss / val_steps if val_steps > 0 else float('inf')
            log_info(f"  Trial {trial+1} | Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f}")

            # Track the best validation loss for this trial
            if avg_val_loss < best_trial_val_loss:
                best_trial_val_loss = avg_val_loss
                # Save the best model for this trial
                model.save(trial_output_dir)
                log_info(f"    -> Best model for trial {trial+1} saved (Val Loss: {best_trial_val_loss:.4f})")


        log_info(f"Trial {trial+1} finished. Best Val Loss: {best_trial_val_loss:.4f} @ Temp: {sampled_temp:.4f}")
        results.append({'temperature': sampled_temp, 'val_loss': best_trial_val_loss, 'trial_dir': trial_output_dir})

    # --- Select Best Hyperparameter ---
    if not results:
        raise RuntimeError("No hyperparameter trials completed successfully.")

    best_result = min(results, key=lambda x: x['val_loss'])
    log_info("\n--- Hyperparameter Search Completed ---")
    log_info(f"Best Temperature: {best_result['temperature']:.4f}")
    log_info(f"Best Val Loss: {best_result['val_loss']:.4f}")
    log_info("--------------------------------------\n")

    return best_result['temperature'], best_result['trial_dir'], train_data, val_data # Return data for potential reuse


# --- MODIFIED FUNCTION: Full Training using Found Temperature ---
def train_cross_encoder_full(data_path, model_name, output_dir, epochs, batch_size, val_split, best_temperature, train_data=None, val_data=None):
    """Main training function using the best found temperature."""
    log_info("Starting full training with best temperature...")
    # Re-load data if not provided (e.g., if not reusing from search)
    if train_data is None or val_data is None:
        log_info("Loading data for full training...")
        with open(data_path) as f:
            full_data = json.load(f)
        train_data, val_data = train_test_split(full_data, test_size=val_split, random_state=42)
    else:
        log_info("Reusing data splits from hyperparameter search.")

    log_info(f"Data loaded. Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_info(f"Using device: {device}")

    model = CrossEncoder(model_name, max_length=512)
    model.model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model.model = torch.nn.DataParallel(model.model)

    # Handle pad_token (same as before)
    if model.tokenizer.pad_token is None:
        if model.tokenizer.eos_token:
            model.tokenizer.pad_token = model.tokenizer.eos_token
            if isinstance(model.model, torch.nn.DataParallel):
                model.model.module.config.pad_token_id = model.tokenizer.eos_token_id
                print(f"Set pad_token to eos_token (DataParallel): {model.tokenizer.eos_token}")
            else:
                model.model.config.pad_token_id = model.tokenizer.eos_token_id
                print(f"Set pad_token to eos_token: {model.tokenizer.eos_token}")
        else:
            pad_token = "<|pad|>"
            model.tokenizer.add_special_tokens({'pad_token': pad_token})
            if isinstance(model.model, torch.nn.DataParallel):
                model.model.module.config.pad_token_id = model.tokenizer.convert_tokens_to_ids(pad_token)
                model.model.module.resize_token_embeddings(len(model.tokenizer))
            else:
                model.model.config.pad_token_id = model.tokenizer.convert_tokens_to_ids(pad_token)
                model.model.resize_token_embeddings(len(model.tokenizer))
            print(f"Added new pad_token: {pad_token}")

    config_to_check = model.model.module.config if isinstance(model.model, torch.nn.DataParallel) else model.model.config
    if config_to_check.pad_token_id is None:
        print("Warning: pad_token_id is still None. Setting to 0 (might be unsafe).")
        config_to_check.pad_token_id = 0
    print(f"Tokenizer pad_token: {model.tokenizer.pad_token}, pad_token_id: {config_to_check.pad_token_id}")

    optimizer = Adafactor(model.parameters(), lr=1e-5, relative_step=False, scale_parameter=False, warmup_init=False)

    # --- Learning Rate Scheduler (for full run) ---
    total_steps = epochs * (len(train_data) // batch_size)
    warmup_steps = int(0.1 * total_steps) if total_steps > 0 else 1
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step / warmup_steps) if warmup_steps > 0 else 1.0
        else:
            return 0.95 ** (current_step - warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Early Stopping and Tracking ---
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3
    train_losses = []
    val_losses = []

    # --- Curriculum Learning Setup (for full run) ---
    transition_epoch = int(0.4 * epochs)
    train_sampler = GroupedTopicSampler(train_data, transition_epoch=transition_epoch)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    log_info("Starting full training loop...")
    for epoch in range(epochs):
        log_info(f"--- Full Train Epoch {epoch+1}/{epochs} (Temp={best_temperature:.4f}) ---")

        # --- Training Phase ---
        model.model.train()
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
        for batch in tqdm(train_dataloader, desc=f"Full Train Epoch {epoch+1}", leave=False):
            optimizer.zero_grad()
            # --- Use InfoNCE Loss with Best Temperature ---
            loss = infonce_loss_fn_multi_neg(batch, model, device, best_temperature)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_steps += 1
        scheduler.step()
        avg_train_loss = total_train_loss / train_steps if train_steps > 0 else 0
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.model.eval()
        val_dataset = CrossEncoderDataset(val_data, epoch, epochs, num_simple_negs=10, num_hard_negs=3)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        total_val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Full Val Epoch {epoch+1}", leave=False):
                # --- Use InfoNCE Loss with Best Temperature ---
                loss = infonce_loss_fn_multi_neg(batch, model, device, best_temperature)
                total_val_loss += loss.item()
                val_steps += 1
        avg_val_loss = total_val_loss / val_steps if val_steps > 0 else 0
        val_losses.append(avg_val_loss)

        # --- Logging ---
        log_info(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # --- Early Stopping Check ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model.save(output_dir) # Save to main output directory
            log_info(f"  -> New best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log_info(f"Stopping early at epoch {epoch+1} due to no improvement.")
                break

    log_info("Full training completed.")
    return train_losses, val_losses

# --- Run Training with Hyperparameter Search ---
if __name__ == "__main__":
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        log_info(f"Error: Data file not found at {DATA_PATH}")
        log_info("Please ensure you have uploaded your 'mistral_generated_1.json' as a Kaggle dataset and updated the DATA_PATH correctly.")
    else:
        log_info("Starting hyperparameter search process...")
        try:
            # --- STEP 1: Hyperparameter Search ---
            best_temp, best_trial_dir, train_data_split, val_data_split = train_cross_encoder_temp_search(
                DATA_PATH,
                MODEL_NAME,
                TEMP_SEARCH_DIR, # Output dir for trials
                NUM_TEMP_TRIALS,
                TEMP_MIN,
                TEMP_MAX,
                SHORT_TRAIN_EPOCHS,
                FULL_TRAIN_BATCH_SIZE, # Use full batch size for search too, or define a separate one
                VAL_SPLIT
            )

            # --- STEP 2: Full Training with Best Hyperparameter ---
            log_info(f"\nStarting full training with best temperature: {best_temp:.4f}")
            final_train_losses, final_val_losses = train_cross_encoder_full(
                DATA_PATH,
                MODEL_NAME,
                OUTPUT_DIR, # Main output directory for final model
                FULL_TRAIN_EPOCHS,
                FULL_TRAIN_BATCH_SIZE,
                VAL_SPLIT,
                best_temp,
                train_data_split, # Reuse data splits
                val_data_split    # Reuse data splits
            )

            log_info("Training script with hyperparameter search finished.")
            log_info(f"Final Training Loss: {final_train_losses[-1] if final_train_losses else 'N/A'}")
            log_info(f"Best Final Validation Loss: {min(final_val_losses) if final_val_losses else 'N/A'}")
            

            with open(os.path.join(OUTPUT_DIR, "loss_history.json"), 'w') as f:
                json.dump({'final_train_losses': final_train_losses, 'final_val_losses': final_val_losses}, f)
            log_info("Loss history saved.")

        except Exception as e:
            log_info(f"An error occurred during the process: {e}")
            import traceback
            traceback.print_exc()

