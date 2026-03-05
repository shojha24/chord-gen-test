import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, recall_score, accuracy_score
from mambaformer_model import build_chordformer
from preprocessing import PreprocessingConfig, create_dataloaders

def evaluate_ensemble(
    ce_model_path="chordformer_models/3-2_epoch_31_best.pt",
    adir_model_path="chordformer_models/adir_finetuned/adir_best.pt",
    alpha=0.5  # 0.5 means a perfect 50/50 blend. Increase to favor common chords, decrease to favor rare.
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Baseline CE Model
    model_ce = build_chordformer().to(device)
    model_ce.load_state_dict(torch.load(ce_model_path, map_location=device))
    model_ce.eval()

    # Load ADIR Model
    model_adir = build_chordformer().to(device)
    model_adir.load_state_dict(torch.load(adir_model_path, map_location=device))
    model_adir.eval()

    # Load Test Data
    dataset_cfg = PreprocessingConfig(use_cache=True, cache_dir=".cache/chordformer")
    _, _, test_loader = create_dataloaders(dataset_cfg, batch_size=48)

    all_preds = [[] for _ in range(6)]
    all_targets = [[] for _ in range(6)]

    print(f"Running Ensemble Evaluation (Alpha={alpha})...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            audio, targets = batch
            audio = audio.to(device)
            
            # Get raw logits from both models
            logits_ce = model_ce(audio)
            logits_adir = model_adir(audio, return_latents=False)

            for i in range(6):
                # THE ENSEMBLE: Blend the logits mathematically
                blended_logits = (alpha * logits_ce[i]) + ((1.0 - alpha) * logits_adir[i])
                
                preds = torch.argmax(blended_logits, dim=-1)
                
                all_preds[i].extend(preds.view(-1).cpu().numpy())
                all_targets[i].extend(targets[i].view(-1).cpu().numpy())

    # Evaluate Head 1
    targets_np = all_targets[0]
    preds_np = all_preds[0]
    
    acc_frame = accuracy_score(targets_np, preds_np)
    acc_class = recall_score(targets_np, preds_np, average='macro', zero_division=0)
    
    print("\n=== ENSEMBLE RESULTS (HEAD 1: Root/Triad) ===")
    print(f"Frame-wise Accuracy: {acc_frame:.4f}")
    print(f"Class-wise Accuracy: {acc_class:.4f}")
    print("\nDetailed Report:")
    print(classification_report(targets_np, preds_np, zero_division=0, digits=4))

if __name__ == "__main__":
    evaluate_ensemble(alpha=0.5)