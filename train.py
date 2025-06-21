import warnings
import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader

from dataset import ChordMatchedDataset
from model import build_transformer

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pathlib import Path

def get_model(device, src_seq_len, hop_length=1024, sample_rate=11025, d_model=32, num_classes=24, n_bins=13):
    model = build_transformer(src_seq_len, hop_length, sample_rate, d_model, num_classes, n_bins)
    model.to(device)
    return model

def get_ds(mix_path, annotation_path, batch_size=4, sample_rate=11025, hop_length=1024, n_mels=64, n_fft=2048):
    dataset = ChordMatchedDataset(mix_path, annotation_path, sample_rate, hop_length, n_mels, n_fft)

    train_ds_size = int(len(dataset) * 0.9)
    val_ds_size = len(dataset) - train_ds_size
    train_ds, val_ds = random_split(dataset, [train_ds_size, val_ds_size])

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return dataset, train_dataloader, val_dataloader

def train_model(mix_path="dataset\\mixes", annotation_path="dataset\\annotations", experiment_name="runs/music_transformer", num_epochs=20, lr=1e-4):
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

    for epoch in range(initial_epoch, num_epochs):
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in batch_iterator:
            model.train()

            encoder_input = batch['feature'].to(device)  # (batch_size, seq_len, 141)
            target = batch['target'].to(device)
            
            # Create padding mask - True for padded positions
            # Check if entire feature vector is -1 (padded)
            src_key_padding_mask = (encoder_input == -1).all(dim=-1)  # (batch_size, seq_len)
            
            # Pass the padding mask to the model
            encoder_output = model.encode(encoder_input, src_key_padding_mask)
            logits = model.project(encoder_output)

            loss = loss_fn(logits.view(-1, 24), target.view(-1))
            batch_iterator.set_postfix(loss=loss.item())

            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

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