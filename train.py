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

def run_validation(model, val_ds, device):
    model.eval()
    count = 0

    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except Exception as e:
        console_width = 80 

    with torch.no_grad():
        for batch in tqdm(val_ds, desc="Validation", ncols=console_width):
            count += 1
            encoder_input = batch['feature'].to(device)  # (batch_size, seq_len, 141)

            encoder_output = model.encode(encoder_input)
            logits = model.project(encoder_output)

            # retrieve the predicted class with the highest probability
            predicted_classes = torch.argmax(logits, dim=-1)
            target = batch['target'].to(device)
            target = target.view(-1)
            predicted_classes = predicted_classes.view(-1)

            print(f"{f'TARGET: ':>12}{target}")
            print(f"{f'PREDICTED: ':>12}{predicted_classes}")     


def get_model(device, src_seq_len, hop_length=1024, sample_rate=11025, d_model=16, num_classes=25, n_bins=13):
    model = build_transformer(src_seq_len, hop_length, sample_rate, d_model, num_classes, n_bins)
    model.to(device)
    return model

def get_ds(mix_path, annotation_path, batch_size=4, sample_rate=11025, hop_length=1024, n_mels=64, n_fft=2048, n_files=1000):
    dataset = ChordMatchedDataset(mix_path, annotation_path, sample_rate, hop_length, n_mels, n_fft, n_files)

    train_ds_size = int(len(dataset) * 0.9)
    val_ds_size = len(dataset) - train_ds_size
    train_ds, val_ds = random_split(dataset, [train_ds_size, val_ds_size])

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return dataset, train_dataloader, val_dataloader

def train_model(mix_path="dataset\\mixes", annotation_path="dataset\\annotations", experiment_name="runs/music_transformer", num_epochs=20, lr=1e-4, num_classes=25):
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    Path(f"music_models").mkdir(parents=True, exist_ok=True)

    dataset, train_dataloader, val_dataloader = get_ds(mix_path, annotation_path)
    
    model = get_model(device, dataset.seq_len)
    model = model.to(device)

    writer = SummaryWriter(experiment_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
            
            # Check for NaN in input data
            if torch.isnan(encoder_input).any():
                print("NaN detected in encoder_input")
                continue
            if torch.isnan(target).any():
                print("NaN detected in target")
                continue
            
            encoder_output = model.encode(encoder_input)
            
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

        #run_validation(model, val_dataloader, device)

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