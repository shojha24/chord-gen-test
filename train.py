import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional
from mambaformer_model import build_chordformer 
from preprocessing import PreprocessingConfig, create_dataloaders
from sklearn.metrics import classification_report, recall_score, accuracy_score


"""
# --- 1. Placeholder Dataset (Unchanged) ---
class ChordFormerDataset(Dataset):
    def __init__(self, num_songs=100, seq_len=1000, feature_dim=252, output_dims=None):
        if output_dims is None:
            # [root_triad, bass, 7th, 9th, 11th, 13th]
            self.output_dims = [85, 13, 4, 4, 3, 3] 
        self.num_songs = num_songs
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        print(f"Created a dummy dataset with {num_songs} items.")

    def __len__(self):
        return self.num_songs

    def __getitem__(self, idx):
        cqt_segment = torch.randn(self.seq_len, self.feature_dim)
        target_labels = [
            torch.randint(0, dim, (self.seq_len,)) for dim in self.output_dims
        ]
        return cqt_segment, target_labels
"""

# --- Custom Loss Function ---
class ChordFormerLoss(nn.Module):
    """
    Computes the weighted cross-entropy loss across all 6 chord component heads.
    """
    def __init__(self, class_weights: Optional[List[torch.Tensor]] = None, ignore_index: int = -100):
        super(ChordFormerLoss, self).__init__()
        
        # Using nn.ModuleList automatically handles pushing the loss functions 
        # (and their internal weight tensors) to the correct GPU/device
        self.loss_functions = nn.ModuleList()
        
        if class_weights:
            for weights in class_weights:
                self.loss_functions.append(nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index))
        else:
            for _ in range(6):  
                self.loss_functions.append(nn.CrossEntropyLoss(ignore_index=ignore_index))

    def forward(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]) -> torch.Tensor:
        total_loss = 0.0
        
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            # Flatten predictions: (batch, seq_len, num_classes) -> (batch * seq_len, num_classes)
            pred_flat = pred.view(-1, pred.size(-1))
            # Flatten targets: (batch, seq_len) -> (batch * seq_len)
            target_flat = target.view(-1)
            # Accumulate the loss for this head
            loss_fn = self.loss_functions[i]
            total_loss += loss_fn(pred_flat, target_flat)
            
        return total_loss


def compute_class_weights(
    dataset, 
    output_dims: List[int], 
    gamma: float = 0.5, 
    w_max: float = 10.0, 
    eps: float = 1e-6
) -> List[torch.Tensor]:
    """
    Computes class weights according to the ChordFormer paper's formula:
    w = min( (n_m / max_n)^(-gamma), w_max )
    """
    print(f"Calculating class frequencies for re-weighting (gamma={gamma}, w_max={w_max})...")
    # 1. Tally the exact frequency of every class for all 6 heads
    counts = [torch.zeros(dim, dtype=torch.float64) for dim in output_dims]
    
    for _, labels in dataset:
        for i, head_labels in enumerate(labels):
            # Flatten the labels to 1D
            flat_labels = head_labels.view(-1)
            
            # FIXED: Filter out the -100 padding tokens before counting!
            valid_labels = flat_labels[flat_labels >= 0]
            
            # Count only the valid classes
            bincount = torch.bincount(valid_labels, minlength=output_dims[i]).to(torch.float64)
            counts[i] += bincount

    # 2. Apply the paper's specific bounding formula
    weights: List[torch.Tensor] = []
    
    for i, count in enumerate(counts):
        # Find the maximum class frequency for this specific head
        max_count = count.max() 
        if max_count == 0:
            # Fallback if a head is completely empty (shouldn't happen with real data)
            weights.append(torch.ones(output_dims[i], dtype=torch.float32))
            continue
            
        # (n_m / max_n)
        ratio = count / max_count
        # Add epsilon to prevent 0^(-gamma), which evaluates to infinity
        ratio = torch.clamp(ratio, min=eps)
        # (ratio)^(-gamma)
        w = ratio ** (-gamma)
        # Clamp to w_max
        w = torch.clamp(w, max=w_max)
        weights.append(w.to(torch.float32))
        
    return weights


def build_crf_transition_matrix(num_classes: int, penalty: float) -> torch.Tensor:
    """
    Builds a fixed transition log-probability matrix based on Eq. 12 from the paper.
    log_trans[i, j] = 0 if i == j else -penalty
    """
    # Initialize matrix with the penalty for changing states
    trans_log = torch.full((num_classes, num_classes), -penalty, dtype=torch.float32)
    # Zero penalty for staying in the exact same state (diagonal)
    trans_log.fill_diagonal_(0.0)
    
    return trans_log


def viterbi_decode_crf(logits: torch.Tensor, penalty: float = 2.0) -> torch.Tensor:
    """
    Vectorized Viterbi decoding to process the entire batch simultaneously.
    """
    batch, seq_len, num_classes = logits.shape
    device = logits.device
    
    # Convert raw logits to log probabilities (observation potentials)
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Build fixed transition matrix: (num_classes, num_classes)
    trans_log = build_crf_transition_matrix(num_classes, penalty).to(device) 
    
    # Initialize scores for the whole batch at t=0
    # Shape: (batch, num_classes)
    score = log_probs[:, 0, :] 
    backpointers = []
    
    # Forward Pass: Sequential over time, but parallel over the batch
    for t in range(1, seq_len):
        # Broadcast scores and transitions to compute all possible paths
        # score.unsqueeze(2): (batch, num_classes_prev, 1)
        # trans_log.unsqueeze(0): (1, num_classes_prev, num_classes_current)
        # next_score: (batch, num_classes_prev, num_classes_current)
        next_score = score.unsqueeze(2) + trans_log.unsqueeze(0)
        
        # Max over the previous states (dim=1)
        # Returns best scores and best states of shape (batch, num_classes_current)
        best_prev_score, best_prev_state = torch.max(next_score, dim=1)
        
        # Add the observation potentials for time t
        score = best_prev_score + log_probs[:, t, :]
        backpointers.append(best_prev_state)
        
    # Backtracking Pass
    decoded = torch.zeros((batch, seq_len), dtype=torch.long, device=device)
    
    # Find the best final state for each sequence in the batch
    last_state = torch.argmax(score, dim=1) # Shape: (batch,)
    decoded[:, seq_len - 1] = last_state
    
    # Backtrack sequentially over time
    for t in range(seq_len - 2, -1, -1):
        # Use torch.gather to fetch the backpointer for the specific last_state of each batch item
        # backpointers[t] is (batch, num_classes), last_state.unsqueeze(1) is (batch, 1)
        last_state = backpointers[t].gather(1, last_state.unsqueeze(1)).squeeze(1)
        decoded[:, t] = last_state
        
    return decoded

# --- Validation and Evaluation Functions ---
def run_validation(model, val_dataloader, device, loss_fn):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            cqt_segments, target_labels = batch
            cqt_segments = cqt_segments.to(device)
            target_labels = [label.to(device) for label in target_labels]
            predictions = model(cqt_segments)
            loss = loss_fn(predictions, target_labels)
            total_val_loss += loss.item()
    avg_loss = total_val_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


# --- Validation and Evaluation Functions ---
def run_evaluation(model, test_dataloader, device, loss_fn, crf_penalty=None):
    """
    Runs final evaluation on the test set, outputting detailed precision, recall, 
    and F1-scores to diagnose class imbalance.
    """
    model.eval()
    total_loss = 0
    all_preds = [[] for _ in range(6)]
    all_crf_preds = [[] for _ in range(6)] if crf_penalty is not None else None
    all_targets = [[] for _ in range(6)]

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Final Evaluation"):
            cqt_segments, target_labels = batch
            cqt_segments = cqt_segments.to(device)
            target_labels = [label.to(device) for label in target_labels]

            predictions = model(cqt_segments)
            loss = loss_fn(predictions, target_labels)
            total_loss += loss.item()

            for i in range(6):
                # Standard Argmax
                predicted_classes = torch.argmax(predictions[i], dim=-1)
                all_preds[i].extend(predicted_classes.view(-1).cpu().numpy())
                all_targets[i].extend(target_labels[i].view(-1).cpu().numpy())

                # CRF Smoothed Decoding
                if all_crf_preds is not None:
                    decoded = viterbi_decode_crf(predictions[i], penalty=crf_penalty)
                    all_crf_preds[i].extend(decoded.view(-1).cpu().numpy())

    avg_loss = total_loss / len(test_dataloader)
    print(f"\n--- Final Test Set Evaluation ---")
    print(f"Test Loss: {avg_loss:.4f}\n")

    head_names = ["Root/Triad", "Bass", "7th", "9th", "11th", "13th"]
    
    for i in range(6):
        targets_np = all_targets[i]
        # Evaluate using CRF predictions if available, otherwise raw argmax
        preds_to_use = all_crf_preds[i] if all_crf_preds is not None else all_preds[i]
        
        # Frame-wise accuracy is standard accuracy
        acc_frame = accuracy_score(targets_np, preds_to_use)
        
        # Class-wise accuracy is equivalent to macro-averaged recall
        acc_class = recall_score(targets_np, preds_to_use, average='macro', zero_division=0)
        
        print(f"=== Head {i+1}: {head_names[i]} ===")
        print(f"Frame-wise Accuracy (acc_frame): {acc_frame:.4f}")
        print(f"Class-wise Accuracy (acc_class): {acc_class:.4f}")
        print("Detailed Report per Class:")
        
        # Zero division is set to 0 to prevent warnings if a rare class is never predicted
        report = classification_report(targets_np, preds_to_use, zero_division=0, digits=4)
        print(report)
        print("-" * 50)
        
    return avg_loss


# --- Main Training Pipeline ---
def train_model(
    max_epochs=200, # Failsafe limit, training will likely stop before this
    lr=1e-3, 
    batch_size=48,
    experiment_name="runs/chordformer_final",
    dataset_root="bello_dataset",
    segment_seconds=10.0,
    max_songs=None,
    use_cache=True,
    refresh_cache=False,
    cache_dir=".cache/chordformer",
    use_class_weights=True,
    crf_penalty=2.0, 
):
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    Path("chordformer_models").mkdir(parents=True, exist_ok=True)

    dataset_cfg = PreprocessingConfig(
        dataset_root=dataset_root,
        segment_seconds=segment_seconds,
        max_songs=max_songs,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        cache_dir=cache_dir,
    )

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(dataset_cfg, batch_size=batch_size)
    print(f"Data split: {len(train_dataloader.dataset)} train, {len(val_dataloader.dataset)} validation, {len(test_dataloader.dataset)} test samples.")

    output_dims = [85, 13, 4, 4, 3, 3]
    class_weights = compute_class_weights(
        train_dataloader.dataset, 
        output_dims, 
        gamma=0.5, 
        w_max=10.0
    ) if use_class_weights else None

    model = build_chordformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # UPDATED: Patience changed from 3 to 5 to match the paper
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    loss_fn = ChordFormerLoss(class_weights=class_weights).to(device)
    writer = SummaryWriter(experiment_name)
    global_step = 0
    
    # UPDATED: Loop over max_epochs, but rely on the early stopping condition
    for epoch in range(max_epochs):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{max_epochs}")
        
        for batch in batch_iterator:
            cqt_segments, target_labels = batch
            cqt_segments = cqt_segments.to(device)
            target_labels = [label.to(device) for label in target_labels]
            
            predictions = model(cqt_segments)
            loss = loss_fn(predictions, target_labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_iterator.set_postfix(loss=loss.item())
            writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1

        val_loss = run_validation(model, val_dataloader, device, loss_fn)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.flush()
        
        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        model_filename = f"chordformer_models/epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), model_filename)

        # UPDATED: Check learning rate for early stopping
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1} completed. Val Loss: {val_loss:.4f} | Current LR: {current_lr}")
        
        if current_lr < 1e-6:
            print(f"\nLearning rate has dropped below 1e-6 (Current LR: {current_lr}).")
            print("Training concluded as per paper's early stopping criteria.")
            break

    writer.close()
    print("\nTraining finished.")
    
    run_evaluation(model, test_dataloader, device, loss_fn, crf_penalty=crf_penalty)


if __name__ == "__main__":
    # train_model()

    
    # To run final evaluation on one of the saved models instead of running the full training loop, you can use the following code snippet. 
    # Make sure to adjust the model path and dataset configuration as needed.
    # The test set this is run on should be the same one used during training for a valid evaluation.
    # This should be the case because the dataset will be cached in .cache/chordformer with the same splits.

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = build_chordformer().to(device)
    model.load_state_dict(torch.load("chordformer_models/3-19_epoch_33_best.pt", map_location=device))
    dataset_cfg = PreprocessingConfig(
        dataset_root="bello_dataset",
        segment_seconds=10.0,
        max_songs=None,
        use_cache=True,
        refresh_cache=False,
        cache_dir=".cache/chordformer",
    )
    _, _, test_dataloader = create_dataloaders(dataset_cfg, batch_size=48)
    loss_fn = ChordFormerLoss().to(device)  # Use unweighted loss for evaluation
    run_evaluation(model, test_dataloader, device, loss_fn, crf_penalty=2.0)

