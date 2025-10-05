import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional
from model import build_chordformer 


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

# --- 2. Custom Loss Function (Unchanged) ---
class ChordFormerLoss(nn.Module):
    def __init__(self, class_weights: Optional[List[torch.Tensor]] = None):
        super(ChordFormerLoss, self).__init__()
        self.class_weights = class_weights
        self.loss_functions = []
        if self.class_weights:
            for weights in self.class_weights:
                self.loss_functions.append(nn.CrossEntropyLoss(weight=weights))
        else:
            self.loss_functions.append(nn.CrossEntropyLoss())

    def forward(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]):
        total_loss = 0.0
        device = predictions[0].device
        if self.class_weights:
            for i, loss_fn in enumerate(self.loss_functions):
                if loss_fn.weight.device != device:
                    self.loss_functions[i].weight = self.class_weights[i].to(device)
        
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            pred_flat = pred.view(-1, pred.size(-1))
            target_flat = target.view(-1)
            loss_fn = self.loss_functions[i] if self.class_weights else self.loss_functions[0]
            total_loss += loss_fn(pred_flat, target_flat)
            
        return total_loss

# --- 3. Validation and Evaluation Functions (Updated) ---
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

def run_evaluation(model, test_dataloader, device, loss_fn):
    """
    Runs final evaluation on the test set for a multi-head model.
    """
    model.eval()
    total_loss = 0
    # Store predictions and targets for each of the 6 heads
    all_preds = [[] for _ in range(6)]
    all_targets = [[] for _ in range(6)]

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Final Evaluation"):
            cqt_segments, target_labels = batch
            cqt_segments = cqt_segments.to(device)
            target_labels = [label.to(device) for label in target_labels]

            predictions = model(cqt_segments)
            loss = loss_fn(predictions, target_labels)
            total_loss += loss.item()

            # Get argmax for each head and store results
            for i in range(6):
                predicted_classes = torch.argmax(predictions[i], dim=-1)
                all_preds[i].extend(predicted_classes.view(-1).cpu().numpy())
                all_targets[i].extend(target_labels[i].view(-1).cpu().numpy())

    avg_loss = total_loss / len(test_dataloader)
    print(f"\n--- Final Test Set Evaluation ---")
    print(f"Test Loss: {avg_loss:.4f}")

    # Calculate and print accuracy for each head
    head_names = ["Root/Triad", "Bass", "7th", "9th", "11th", "13th"]
    accuracies = []
    for i in range(6):
        preds_np = torch.tensor(all_preds[i])
        targets_np = torch.tensor(all_targets[i])
        accuracy = (preds_np == targets_np).float().mean().item()
        accuracies.append(accuracy)
        print(f"  - Accuracy (Head {i+1} - {head_names[i]}): {accuracy:.4f}")
    
    avg_accuracy = sum(accuracies) / len(accuracies)
    print(f"Average Head Accuracy: {avg_accuracy:.4f}")
    print("---------------------------------")
    # For a full implementation, you would combine the head predictions into final
    # chord labels and use mir_eval for official metrics like WCSR.
    return avg_loss, accuracies


# --- 4. Main Training Pipeline (Updated) ---
def train_model(
    num_epochs=50, 
    lr=1e-3, 
    batch_size=24,
    experiment_name="runs/chordformer_final"
):
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    Path("chordformer_models").mkdir(parents=True, exist_ok=True)

    # UPDATED: Setup all three datasets
    dataset = ChordFormerDataset()
    total_len = len(dataset)
    train_len = int(0.6 * total_len)
    val_len = int(0.2 * total_len)
    test_len = total_len - train_len - val_len
    
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_len, val_len, test_len], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print(f"Data split: {len(train_ds)} train, {len(val_ds)} validation, {len(test_ds)} test samples.")

    model = build_chordformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)
    loss_fn = ChordFormerLoss(class_weights=None).to(device)
    
    writer = SummaryWriter(experiment_name)
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
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
        scheduler.step(val_loss)

        model_filename = f"chordformer_models/epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), model_filename)

    writer.close()
    print("\nTraining finished.")
    
    # UPDATED: Run final evaluation on the held-out test set
    run_evaluation(model, test_dataloader, device, loss_fn)


if __name__ == "__main__":
    train_model()
