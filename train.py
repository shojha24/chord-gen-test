import warnings
import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader

from dataset import ChordMatchedDataset
from model import build_transformer

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pathlib import Path
import os

import config

def run_validation(model, val_dataloader, device, loss_fn, num_classes):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            encoder_input = batch['feature'].to(device)
            target = batch['target'].to(device)
            mask = batch['mask'].to(device)  # Don't forget the mask here too!

            # Forward pass
            encoder_output = model.encode(encoder_input, src_key_padding_mask=~mask)
            logits = model.project(encoder_output)
            
            # Calculate loss (ignoring padding)
            loss = loss_fn(logits.view(-1, num_classes), target.view(-1))
            total_loss += loss.item() * encoder_input.size(0) # Multiply by batch size

            # Calculate accuracy (ignoring padding)
            predicted_classes = torch.argmax(logits, dim=-1)
            
            # Only compare where the target is not the ignore_index (-1)
            valid_preds = predicted_classes[target != -1]
            valid_targets = target[target != -1]
            
            total_correct += (valid_preds == valid_targets).sum().item()
            total_samples += valid_targets.numel()

    avg_loss = total_loss / len(val_dataloader.dataset)
    accuracy = total_correct / total_samples
    print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy
 


def get_model(device, src_seq_len, hop_length=config.HOP_LENGTH, sample_rate=config.SAMPLE_RATE, d_model=config.D_MODEL, num_classes=config.NUM_CLASSES, n_bins=config.N_BINS):
    model = build_transformer(src_seq_len, hop_length, sample_rate, d_model, num_classes, n_bins)
    model.to(device)
    return model

def get_ds(mix_path, annotation_path, batch_size=config.BATCH_SIZE, sample_rate=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH, n_mels=config.N_MELS, n_fft=config.N_FFT, n_files=config.N_FILES):
    dataset = ChordMatchedDataset(mix_path, annotation_path, sample_rate, hop_length, n_mels, n_fft, n_files)

    train_ds_size = int(len(dataset) * 0.9)
    val_ds_size = len(dataset) - train_ds_size
    train_ds, val_ds = random_split(dataset, [train_ds_size, val_ds_size])

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return dataset, train_dataloader, val_dataloader

def train_model(mix_path="dataset\\mixes", annotation_path="dataset\\annotations", experiment_name="runs/music_transformer", num_epochs=config.NUM_EPOCHS, lr=config.LEARNING_RATE, num_classes=config.NUM_CLASSES):
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    Path(f"music_models").mkdir(parents=True, exist_ok=True)

    dataset, train_dataloader, val_dataloader = get_ds(mix_path, annotation_path)
    
    model = get_model(device, dataset.seq_len)
    model = model.to(device)

    writer = SummaryWriter(experiment_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    initial_epoch = 0
    global_step = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1).to(device)

    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    for epoch in range(initial_epoch, num_epochs):
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in batch_iterator:
            model.train()

            encoder_input = batch['feature'].to(device)
            target = batch['target'].to(device)
            mask = batch['mask'].to(device)
            
            # Check for NaN in input data
            if torch.isnan(encoder_input).any():
                print("NaN detected in encoder_input")
                continue
            if torch.isnan(target).any():
                print("NaN detected in target")
                continue
            
            encoder_output = model.encode(encoder_input, src_key_padding_mask=mask)
            
            # Check encoder output
            if torch.isnan(encoder_output).any():
                print("NaN detected in encoder_output")
                break
                
            logits = model.project(encoder_output)
            
            # Check logits
            if torch.isnan(logits).any():
                print("NaN detected in logits")
                break

            loss = loss_fn(logits.view(-1, num_classes), target.view(-1))
            
            # Check loss
            if torch.isnan(loss):
                print(f"NaN loss detected at step {global_step}")
                print(f"Logits stats: min={logits.min()}, max={logits.max()}, mean={logits.mean()}")
                print(f"Target stats: min={target.min()}, max={target.max()}")
                break

            batch_iterator.set_postfix(loss=loss.item())
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.flush()

            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        # ... inside the epoch loop, after validation
        val_loss, val_acc = run_validation(model, val_dataloader, device, loss_fn, config.NUM_CLASSES) # Assumes run_validation returns metrics
        writer.add_scalar('Loss/validation', val_loss, global_step)
        writer.add_scalar('Accuracy/validation', val_acc, global_step)

        # Step the scheduler based on the validation loss
        scheduler.step(val_loss) 

        model_filename = f"music_models/epoch_{epoch + 1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train_model()