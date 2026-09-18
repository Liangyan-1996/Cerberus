# -*- coding: utf-8 -*-

import torch
import numpy as np
import pyfaidx
import argparse
from scipy.ndimage import gaussian_filter1d

from se_data_utils import dna_one_hot
from se_model_arch import BorzoiSEAdapter

# =============================================================================
# Config & Helpers
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_MODEL_ID = "pretrained_models/borzoi-replicate-0"

def load_trained_model(checkpoint_path, num_cell_types=None):
    """
    Loads architecture and weights. 
    [Modified]: Automatically infers num_cell_types from checkpoint if not provided.
    """
    print(f"Loading Weights from {checkpoint_path}...")
    # Load state_dict to CPU first to inspect shapes
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # [Logic]: Auto-detect num_cell_types from the last layer's weight shape
    # The cluster_head is a Sequential block. The last Conv1d is at index 4.
    # Key: "cluster_head.4.weight" -> Shape: [num_cell_types, 256, 1]
    if num_cell_types is None:
        try:
            detected_types = state_dict["cluster_head.4.weight"].shape[0]
            print(f"[Info] Auto-detected num_cell_types from checkpoint: {detected_types}")
            num_cell_types = detected_types
        except KeyError:
            # Fallback for safety, though unlikely if checkpoint is valid
            raise ValueError("Could not infer num_cell_types from checkpoint. Please specify --num_cell_types manually.")

    print(f"Initializing Architecture from {HF_MODEL_ID}...")
    model = BorzoiSEAdapter(pretrained_model_name_or_path=HF_MODEL_ID, num_cell_types=num_cell_types)

    # Load weights into the model
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    
    try:
        model = torch.compile(model)
        print("Model compiled with torch.compile.")
    except Exception as e:
        print(f"Skipping torch.compile: {e}")
        
    return model

def post_process_predictions(cluster_probs, se_probs, sigma=1.0):
    """
    Applies Gaussian smoothing to remove speckle noise.
    Final Score = P(Cluster) * P(SE|Cluster)
    """
    p_cluster = cluster_probs.float().cpu().numpy()
    p_se_cond = se_probs.float().cpu().numpy()
    
    # 1. Joint Probability
    raw_se_prob = p_cluster * p_se_cond
    
    # 2. Gaussian Smoothing (along length axis)
    smoothed_se_prob = gaussian_filter1d(raw_se_prob, sigma=sigma, axis=-1)
    
    return smoothed_se_prob

def predict_region(model, genome, chrom, start, end):
    """
    Predicts SE probability for a specific region.
    """
    input_len = 524288
    
    center = (start + end) // 2
    inp_start = center - input_len // 2
    inp_end = center + input_len // 2
    
    # Fetch sequence as string using pyfaidx
    seq_str = genome[chrom][inp_start:inp_end].seq
    
    seq_bytes = seq_str.upper().encode('utf-8')
    seq_arr = np.frombuffer(seq_bytes, dtype=np.int8)
    
    # Convert to tensor
    seq_tensor = dna_one_hot(seq_arr).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        # [Modified]: Use BFloat16
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            cluster_logits, se_logits = model(seq_tensor)
            p_cluster = torch.sigmoid(cluster_logits)
            p_se = torch.sigmoid(se_logits)
    
    final_probs = post_process_predictions(p_cluster, p_se, sigma=args.smooth_sigma)
    
    return final_probs

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SE Prediction Inference")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth model')
    parser.add_argument('--fasta', type=str, required=True, help='Path to genome fasta')
    parser.add_argument('--chrom', type=str, default='chr8')
    parser.add_argument('--start', type=int, default=127735434)
    parser.add_argument('--end', type=int, default=127740434)
    parser.add_argument('--num_cell_types', type=int, default=None, help='(Optional) Auto-inferred if not set')
    parser.add_argument('--smooth_sigma', type=float, default=1.0, help='Gaussian smoothing sigma')
    
    args = parser.parse_args()
    genome = pyfaidx.Fasta(args.fasta)
    model = load_trained_model(args.checkpoint, args.num_cell_types)
    print(f"Predicting {args.chrom}:{args.start}-{args.end}...")
    probs = predict_region(model, genome, args.chrom, args.start, args.end)
    print(f"Output Shape: {probs.shape}")
    print(f"Max SE Probability: {probs.max():.4f}")