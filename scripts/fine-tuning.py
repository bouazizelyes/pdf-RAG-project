import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import CrossEncoder
from torch.optim import AdamW
from sklearn.model_selection import train_test_split # For splitting data

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

# --- Training Function with Validation Tracking ---
def train_cross_encoder(data_path, model_name, output_dir, epochs=10, batch_size=16, val_split=0.2):
    """Fine-tune cross-encoder with curriculum learning, LR scheduling, and validation."""
    with open(data_path) as f:
        full_data = json.load(f)

    # Split data into training and validation sets
    train_data, val_data = train_test_split(full_data, test_size=val_split, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossEncoder(model_name, max_length=512).to(device)

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

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_dataset = CrossEncoderDataset(train_data, epoch, epochs)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        total_train_loss = 0
        train_steps = 0
        for batch in train_dataloader:
             # Use the chosen loss function (e.g., contrastive_loss_fn)
            loss = contrastive_loss_fn(batch, model, device, epoch, epochs)

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
            for batch in val_dataloader:
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
    """Contrastive Loss with Adaptive Margin."""
    features = [[q, p] for q, p in zip(batch['query'], batch['positive'])]
    neg_features = [[q, n] for q, n in zip(batch['query'], batch['negative'])]

    p_scores = model(features).to(device)
    n_scores = model(neg_features).to(device)

    margin = 1.0 + (epoch / total_epochs) * 0.5
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
# train_losses, val_losses = train_cross_encoder(
#     "data.json",
#     "Qwen/Qwen3-Reranker-0.6B",
#     "output_model",
#     epochs=15,
#     batch_size=16
# )
# Ensure you call the desired loss function inside the training/validation loops
# e.g., replace `contrastive_loss_fn` calls with `mse_loss_fn` if using MSE.
