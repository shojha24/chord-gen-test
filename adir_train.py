import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# Import dependencies directly from existing modules
from mambaformer_model import build_chordformer 
from preprocessing import PreprocessingConfig, create_dataloaders

# Import the helper functions and loss class directly from train.py
from train import ChordFormerLoss, run_validation, run_evaluation

def compute_batch_diversity(features: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int = -100, max_samples: int = 500):
    """
    Calculates the batch-wise Diversity (D) for each class using the ADIR formula.
    Includes random sub-sampling to prevent O(N^3) matrix inversion bottlenecks on sequential data.
    """
    batch_diversities = {c: 0.0 for c in range(num_classes)}
    
    valid_mask = targets != ignore_index
    valid_features = features[valid_mask]
    valid_targets = targets[valid_mask]

    unique_classes = torch.unique(valid_targets)

    for c in unique_classes:
        c = c.item()
        
        class_mask = valid_targets == c
        f_c = valid_features[class_mask] 
        original_n_c = f_c.size(0)
        
        if original_n_c <= 1:
            batch_diversities[c] = float(original_n_c)
            continue
            
        # --- THE FIX: Sub-sample if the class dominates the batch ---
        if original_n_c > max_samples:
            # Randomly select max_samples indices
            indices = torch.randperm(original_n_c, device=features.device)[:max_samples]
            f_c = f_c[indices]
            n_c = max_samples
        else:
            n_c = original_n_c
            
        # Centralize features
        mu = f_c.mean(dim=0, keepdim=True)
        f_c_centered = f_c - mu
        
        # Compute Correlation Matrix (P) via Cosine Similarity
        f_c_norm = F.normalize(f_c_centered, p=2, dim=1)
        P = torch.matmul(f_c_norm, f_c_norm.T) 
        
        # Add epsilon to diagonal for invertibility
        epsilon = 1e-4
        P = P + torch.eye(n_c, device=P.device) * epsilon
        
        try:
            P_inv = torch.linalg.inv(P)
            diversity = P_inv.sum().item()
            
            # Scale the diversity back up if we sub-sampled!
            # If 500 samples yielded a diversity of 40, and we actually had 5000 samples,
            # the projected diversity for the full set is 40 * (5000/500) = 400.
            if original_n_c > max_samples:
                scaling_factor = original_n_c / max_samples
                diversity = diversity * scaling_factor
                
            batch_diversities[c] = max(diversity, 1e-6)
            
        except torch._C._LinAlgError:
            # Fallback
            batch_diversities[c] = float(original_n_c)

    return batch_diversities


def train_adir(
    checkpoint_path="chordformer_models/3-2_epoch_31_best.pt",
    save_dir="chordformer_models/adir_finetuned",
    experiment_name="runs/chordformer_adir",
    epochs=50,
    stage_2_lr=1e-4, 
    batch_size=48,
    dataset_root="bello_dataset",
    segment_seconds=10.0,
    crf_penalty=2.0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Setup dataloaders exactly as they are in train.py
    dataset_cfg = PreprocessingConfig(
        dataset_root=dataset_root,
        segment_seconds=segment_seconds,
        use_cache=True,
        cache_dir=".cache/chordformer",
    )
    train_loader, val_loader, test_loader = create_dataloaders(dataset_cfg, batch_size=batch_size)
    
    # FIXED: Ensure output_dims perfectly matches checkpoint and train.py
    output_dims = [85, 13, 4, 4, 3, 3] 
    
    # 2. Build and Load Model
    model = build_chordformer().to(device)
    print(f"Loading Stage 1 weights from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # 3. Setup Optimizer, Scheduler, and Tensorboard
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage_2_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    writer = SummaryWriter(experiment_name)
    
    # We can reuse the original ChordFormerLoss. We initialize it with no weights for Epoch 1.
    loss_fn = ChordFormerLoss(class_weights=None).to(device)
    
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        
        # Accumulators for the overall diversity of this epoch
        epoch_diversities = [torch.zeros(dim, dtype=torch.float32, device=device) for dim in output_dims]
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch in train_pbar:
            audio_cqt, targets = batch
            audio_cqt = audio_cqt.to(device)
            targets = [t.to(device) for t in targets]
            
            optimizer.zero_grad()
            
            # Extract both latents and predictions (requires the small tweak to the model's forward method)
            latents, preds = model(audio_cqt, return_latents=True)
            flat_latents = latents.view(-1, latents.size(-1))
            
            loss = loss_fn(preds, targets)
            loss.backward()
            
            # Apply gradient clipping identically to train.py
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1
            
            # --- ADIR Calculation ---
            with torch.no_grad():
                for i in range(6):
                    flat_targets = targets[i].view(-1)
                    batch_div_dict = compute_batch_diversity(flat_latents, flat_targets, output_dims[i])
                    for c, d_val in batch_div_dict.items():
                        epoch_diversities[i][c] += d_val
                        
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})


        # --- Dynamic Weight Update for Next Epoch ---
        new_class_weights = []
        w_max = 10.0  # THE CEILING: Matches baseline's maximum penalty
        
        for i in range(6):
            D = epoch_diversities[i]
            # Replace 0s with 1s to prevent division by zero for absent classes
            D = torch.where(D == 0, torch.ones_like(D), D) 
            
            # Formula: w_c = (1 / D_c) / sum(1 / D_i) * C
            inv_D = 1.0 / D
            sum_inv_D = inv_D.sum()
            weight = (inv_D / sum_inv_D) * output_dims[i]
            
            # --- NEW: CLAMP THE WEIGHTS ---
            weight = torch.clamp(weight, max=w_max)
            
            new_class_weights.append(weight.to(device))
            
        # Swap out the loss function entirely with the newly calculated ADIR weights
        loss_fn = ChordFormerLoss(class_weights=new_class_weights).to(device)

        # --- Validation Loop (Reusing train.py function) ---
        val_loss = run_validation(model, val_loader, device, loss_fn)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.flush()
                
        avg_train_loss = total_train_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "adir_best.pt"))
            print("--> Saved new best model!")
            
        # Early Stopping Logic
        if current_lr < 1e-6:
            print(f"\nLearning rate has dropped below 1e-6. Concluding fine-tuning.")
            break

    writer.close()
    
    print("\n--- Training Complete. Running Test Evaluation ---")
    model.load_state_dict(torch.load(os.path.join(save_dir, "adir_best.pt"), map_location=device))
    
    # Revert to unweighted cross-entropy so the final test loss metric is comparable to your baseline
    eval_loss_fn = ChordFormerLoss(class_weights=None).to(device)
    run_evaluation(model, test_loader, device, eval_loss_fn, crf_penalty=crf_penalty)

if __name__ == "__main__":
    # train_adir()

    
    # To run final evaluation on one of the saved models instead of running the full training loop, you can use the following code snippet. 
    # Make sure to adjust the model path and dataset configuration as needed.
    # The test set this is run on should be the same one used during training for a valid evaluation.
    # This should be the case because the dataset will be cached in .cache/chordformer with the same splits.

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = build_chordformer().to(device)
    model.load_state_dict(torch.load("chordformer_models/adir_finetuned/adir_best.pt", map_location=device))
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