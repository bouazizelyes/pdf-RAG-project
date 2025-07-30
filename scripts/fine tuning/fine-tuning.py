import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import CrossEncoder
from torch.optim import AdamW
from sklearn.model_selection import train_test_split # For splitting data
from tqdm.auto import tqdm 
from torch import nn


import collections
from torch.utils.data import Sampler

class GroupedTopicSampler(Sampler):
    """
    Custom Sampler to implement a two-stage curriculum learning based on data order.

    - Stage 1 (Early Epochs): Groups data by 'source_topic_id'. It shuffles the order
      of the topics, and also shuffles the items within each topic. This keeps
      related items together to encourage domain-specific learning.
    - Stage 2 (Later Epochs): Shuffles all data globally, mixing all topics together
      to encourage generalization and differentiation between topics.
    """
    def __init__(self, data_source, transition_epoch):
        self.data_source = data_source
        self.transition_epoch = transition_epoch
        self.epoch = 0

        # Pre-process to group indices by topic_id
        self.topic_to_indices = collections.defaultdict(list)
        for idx, item in enumerate(data_source):
            topic_id = item.get('source_topic_id', 'unknown') # Safely get topic_id
            self.topic_to_indices[topic_id].append(idx)

        self.topic_ids = list(self.topic_to_indices.keys())

    def set_epoch(self, epoch):
        """
        Called by the DataLoader to set the current epoch. This is crucial for our logic.
        """
        self.epoch = epoch

    def __iter__(self):
        if self.epoch < self.transition_epoch:
            # --- Stage 1: Grouped Shuffling ---
            print(f"\nSampler: Using STAGE 1 (Grouped by Topic) for epoch {self.epoch}")

            # 1. Shuffle the order of the topics themselves
            random.shuffle(self.topic_ids)

            # 2. Create the final list of indices by processing topic by topic
            indices = []
            for topic_id in self.topic_ids:
                # Get indices for this topic and shuffle them
                topic_indices = self.topic_to_indices[topic_id]
                random.shuffle(topic_indices)
                indices.extend(topic_indices)
            
            yield from indices

        else:
            # --- Stage 2: Global Shuffling ---
            print(f"\nSampler: Using STAGE 2 (Global Shuffle) for epoch {self.epoch}")

            # Create a list of all indices and shuffle it
            indices = list(range(len(self.data_source)))
            random.shuffle(indices)
            yield from indices

    def __len__(self):
        return len(self.data_source)

# --- Updated Dataset Class (No changes needed for loss tracking) ---
class CrossEncoderDataset(Dataset):
    def __init__(self, data, epoch, total_epochs):
        self.data = data
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.hard_introduction_epoch = int(0.3 * total_epochs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        query = entry['query']
        positive = entry['positive']

        if self.epoch < self.hard_introduction_epoch:
            negatives = entry['simple_negatives']
        else:
            negatives = entry['hard_negatives'] + entry['simple_negatives']

        neg = random.choice(negatives) if negatives else "no_negative"
        return {'query': query, 'positive': positive, 'negative': neg}

def get_scores(pairs, model, device):
    """
    pairs: List[Tuple[str,str]] of (query, passage)
    model: sentence_transformers.CrossEncoder
    """
    # 1) Tokenize into tensors (with padding!) and move to device
    encoded = model.tokenizer(
        [q for q,_ in pairs],
        [p for _,p in pairs],
        padding=True,
        truncation=True,
        max_length=model.max_length,
        return_tensors="pt",
    ).to(device)

    # 2) Forward through the underlying HF model
    outputs = model.model(**encoded)
    # logits: Tensor of shape (batch_size, 1), with requires_grad=True
    return outputs.logits.view(-1)


# --- Training Function with Validation Tracking ---
def train_cross_encoder(data_path, model_name, output_dir, epochs=10, batch_size=16, val_split=0.2):
    """Fine-tune cross-encoder with curriculum learning, LR scheduling, and validation."""
    with open(data_path) as f:
        full_data = json.load(f)

    # Split data into training and validation sets
    train_data, val_data = train_test_split(full_data, test_size=val_split, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossEncoder(
    model_name,
    max_length=512,
    # Add quantization_config for 4-bit loading
    model_kwargs={
        "load_in_4bit": True,
        "device_map": "auto", # Let accelerate handle device placement
        "quantization_config": {
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        },
    },
    # The classification head must remain in full precision for stability
    automodel_args={'trust_remote_code': True}
)

    # After loading, you must prepare the model for QLoRA-style training
    # This freezes most of the model but makes the trainable parts memory-efficient
    from peft import prepare_model_for_kbit_training
    model.model = prepare_model_for_kbit_training(model.model)
 
    # The CrossEncoder adds a final linear layer called 'score' or 'classifier'
    # We need to make sure this layer is trainable and in the correct dtype
    for name, param in model.model.named_parameters():
        if "score" in name or "classifier" in name or "lora" in name:
            param.requires_grad = True
            
    model.tokenizer.pad_token = model.tokenizer.eos_token
    model.model.config.pad_token_id = model.tokenizer.eos_token_id

    optimizer = AdamW(model.parameters(), lr=1e-5)

    def lr_lambda(current_step):
        total_steps = epochs * (len(train_data) // batch_size)
        warmup_steps = int(0.1 * total_steps)
        if current_step < warmup_steps:
            return current_step / warmup_steps
        return 0.95 ** (current_step - warmup_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3 # Stop if val loss doesn't improve for 3 epochs

    # Store losses for plotting
    train_losses = []
    val_losses = []
    
    transition_epoch = int(0.4 * epochs) 
    train_sampler = GroupedTopicSampler(train_data, transition_epoch=transition_epoch)

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_sampler.set_epoch(epoch)
        
        # Your CrossEncoderDataset is created per-epoch to handle the simple/hard negative curriculum
        train_dataset = CrossEncoderDataset(train_data, epoch, epochs)
        
        # --- NEW: Use the sampler in the DataLoader ---
        # NOTE: When a sampler is provided, shuffle MUST be False.
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler, 
            shuffle=False # This is now controlled by the sampler!
        )

        total_train_loss = 0
        train_steps = 0
        
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            loss = contrastive_loss_fn(batch, model, device, epoch, epochs)
            # ... (rest of the training step is the same) ...
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_steps += 1
            
        scheduler.step()
        avg_train_loss = total_train_loss / train_steps
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        val_dataset = CrossEncoderDataset(val_data, epoch, epochs) # Curriculum logic still applies if desired
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        total_val_loss = 0
        val_steps = 0
        with torch.no_grad(): # Disable gradient calculation for validation
            for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}"):
                 # Use the SAME loss function for validation
                loss = contrastive_loss_fn(batch, model, device, epoch, epochs)
                total_val_loss += loss.item()
                val_steps += 1

        avg_val_loss = total_val_loss / val_steps
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- Early Stopping Check ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            model.save(output_dir)
            print(f"  -> New best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Stopping early at epoch {epoch+1} due to no improvement in validation loss.")
                break

    # Optional: Plot training & validation loss curves
    # import matplotlib.pyplot as plt
    # plt.plot(train_losses, label='Training Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.title('Training and Validation Loss')
    # plt.show()

    return train_losses, val_losses # Return for potential external plotting

# --- Loss Functions ---
def contrastive_loss_fn(batch, model, device, epoch, total_epochs):
    # Build your pairs
    features     = list(zip(batch['query'],     batch['positive']))
    neg_features = list(zip(batch['query'],     batch['negative']))

    # Compute scores with gradients
    p_scores = get_scores(features,     model, device)
    n_scores = get_scores(neg_features, model, device)

    # Adaptive margin
    margin = 1.0 + (epoch / total_epochs) * 0.5

    # Hinge‐style loss
    loss = torch.mean(torch.nn.functional.relu(n_scores - p_scores + margin))
    return loss

def batch_hard_triplet_loss_fn(batch, model, device, margin=1.0):
    """Batch Hard Triplet Loss (if you sample multiple negatives per query)."""
    # This example assumes you modify data loading to provide multiple negatives per item
    queries = batch['query']
    positives = batch['positive']
    negatives_list = batch['negatives'] # List of lists of negatives per query

    all_scores = []
    labels = [] # 1 for positive pairs, 0 for negative pairs
    for i, query in enumerate(queries):
         # Score positive
        pos_score = model([query, positives[i]]).to(device)
        all_scores.append(pos_score)
        labels.append(1)

        # Score all negatives for this query
        for neg in negatives_list[i]:
            neg_score = model([query, neg]).to(device)
            all_scores.append(neg_score)
            labels.append(0)

    scores = torch.stack(all_scores)
    labels = torch.tensor(labels, device=device)

    # Find hardest positive and negative for each query in the batch
    # Requires reshaping/grouping scores by query if processing batch-wise
    # Simplified version: Calculate pairwise loss for demonstration
    # A more robust implementation would group by query first.
    # This is a basic approximation:
    pos_mask = (labels == 1)
    neg_mask = (labels == 0)

    hardest_pos = torch.masked_select(scores, pos_mask).max() # Simplified
    hardest_neg = torch.masked_select(scores, neg_mask).min() # Simplified

    loss = torch.nn.functional.relu(hardest_neg - hardest_pos + margin)
    return loss

def mse_loss_fn(batch, model, device):
    """MSE-based ranking loss (less common but possible)."""
    features = [[q, p] for q, p in zip(batch['query'], batch['positive'])]
    neg_features = [[q, n] for q, n in zip(batch['query'], batch['negative'])]

    p_scores = model(features).to(device)
    n_scores = model(neg_features).to(device)

    # Assume ideal score for positive is 1.0 and for negative is 0.0
    target_pos = torch.ones_like(p_scores)
    target_neg = torch.zeros_like(n_scores)

    mse_pos = torch.nn.functional.mse_loss(p_scores, target_pos)
    mse_neg = torch.nn.functional.mse_loss(n_scores, target_neg)

    # Combine losses (e.g., average)
    loss = (mse_pos + mse_neg) / 2.0
    return loss


# --- Example Usage ---
train_losses, val_losses = train_cross_encoder(
    "mistral_generated_1.json",
    "Qwen/Qwen3-Reranker-0.6B",
    "output_model",
    epochs=15,
    batch_size=1    
)
# Ensure you call the desired loss function inside the training/validation loops
# e.g., replace `contrastive_loss_fn` calls with `mse_loss_fn` if using MSE.
