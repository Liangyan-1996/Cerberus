import torch
from torch.utils.data import DataLoader

import os
import argparse
import random
import pickle
import pandas as pd
import numpy as np
from se_model_arch import BorzoiSEAdapter
from se_train_utils import build_dataset, evaluate_unified, plot_pr_curve_from_data, plot_per_cell_performance

CONFIGS = {
    'human': {
        'fasta': '/public/home/liunangroup/liangyan/project/SE_model/data/genome/hg38.fa',
        'chrom_sizes': '/public/home/liunangroup/liangyan/project/SE_model/data/genomesize/hg38.chrom.sizes',
        'pool_file': '/public/home/liunangroup/liangyan/project/SE_model/data/AllSample.hg38.AnchorPool.parquet',
        'blacklist': '/public/home/liunangroup/liangyan/project/SE_model/data/blacklist/hg38.blacklist.bed',
        'label_cache': '/public/home/liunangroup/liangyan/project/SE_model/data/AllSample.hg38.SE.label_cache.pkl',
    },
    'mouse': {
        'fasta': '/public/home/liunangroup/liangyan/project/SE_model/data/genome/mm10.fa',
        'chrom_sizes': '/public/home/liunangroup/liangyan/project/SE_model/data/genomesize/mm10.chrom.sizes',
        'pool_file': '/public/home/liunangroup/liangyan/project/SE_model/data/AllSample.mm10.AnchorPool.parquet',
        'blacklist': '/public/home/liunangroup/liangyan/project/SE_model/data/blacklist/mm10.blacklist.bed',
        'label_cache': '/public/home/liunangroup/liangyan/project/SE_model/data/AllSample.mm10.SE.label_cache.pkl',
    }
}

HF_MODEL_ID = '/public/home/liunangroup/liangyan/project/SE_model/pretrained_models/borzoi-replicate-0'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 12
SEED = 42

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(species, checkpoint_path, output_dir, pos_weight_cluster, pos_weight_se, lambda_cluster, fix_jitters, chrs=None):
    seed_all(SEED)
    print(f"\n[Evaluation] Starting Standalone Evaluation for {species.upper()}")
    paths = CONFIGS[species]
    
    if output_dir is None: output_dir = os.path.dirname(checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Reproduce Test Split
    chrom_sizes = {line.split()[0]: int(line.split()[1]) for line in open(paths['chrom_sizes'])}
    std_chroms = sorted([c for c in chrom_sizes.keys() if ('_' not in c) and ('chrM' not in c) and (c != 'chrX') and (c != 'chrY')])
    random.shuffle(std_chroms)
    test_chroms = std_chroms[2:4] if chrs is None else chrs
    print(f"Reproduced Test Chromosomes: {test_chroms}")
    print("Building Test Dataset...")
    test_dataset = build_dataset(test_chroms, paths, chrom_sizes, False, None, fix_jitters)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True, prefetch_factor=2, persistent_workers=False)
    
    anchors_df = pd.DataFrame(test_dataset.sampler.current_epoch_anchors,columns=['chrom','center','jitter'])
    # anchors_df = anchors_df.sort_values(by=['chrom','center'])
    # !!! do not sort anchors_df !!!
    # the order of anchors_df must be the same as the order of predictions_df (npy)
    anchors_df.to_csv(f"{output_dir}/anchors_testing.csv", index=False)
    
    # 2. Init Model & Load Weights
    meta_cache = paths['label_cache'].replace('.pkl', '.metadata.pkl')
    with open(meta_cache, 'rb') as f: num_cell_types = len(pickle.load(f))
    print(f"Loading Model from {checkpoint_path}...")
    model = BorzoiSEAdapter(HF_MODEL_ID, num_cell_types)
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    
    # 3. Unified Evaluation (Global + Per-Cell)
    print("Running Unified Evaluation (Global + Per-Cell)...")
    weight_c_tensor = torch.tensor(pos_weight_cluster, device=DEVICE, dtype=torch.float32)
    weight_se_tensor = torch.tensor(pos_weight_se, device=DEVICE, dtype=torch.float32)
    global_res, per_cell_df = evaluate_unified(model, test_loader, DEVICE, meta_cache, weight_c_tensor, weight_se_tensor, lambda_cluster, calc_per_cell=True, save_dir=output_dir)
    
    #  Global evaluation metrics
    print(f"Final Test: total loss={global_res['total_loss']:.4f} | cluster loss={global_res['cluster_loss']:.4f} | se loss={global_res['se_loss']:.4f}")
    print(f"Final Test SE AUPRC (Exact): {global_res['auprc_se']:.4f}")
    print(f"Final Test Cluster AUPRC (Exact): {global_res['auprc_cluster']:.4f}")
    print(f"Best F1 Cluster: {global_res['best_f1_cluster']:.4f}")
    print(f"Best F1 Cluster Thr: {global_res['best_thresh_cluster']:.4f}")
    print(f"Best F1 SE: {global_res['best_f1_se']:.4f}")
    print(f"Best F1 SE Thr: {global_res['best_thresh_se']:.4f}")
    
    if global_res['curve_data']:
        glob_eval_path = os.path.join(output_dir, 'eval_global_pr_curve.png')
        plot_pr_curve_from_data(global_res['curve_data'], global_res['auprc_cluster'], global_res['auprc_se'], glob_eval_path)
    
    #  Per-cell evaluation metrics
    csv_path = os.path.join(output_dir, 'eval_per_cell_metrics.csv')
    per_cell_df.to_csv(csv_path, index=False)
    
    per_cell_eval_path = os.path.join(output_dir, 'eval_per_cell_performance.png')
    plot_per_cell_performance(per_cell_df, per_cell_eval_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--species', type=str, default='human', choices=['human', 'mouse'])
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the trained .pth file")
    parser.add_argument('--output_dir', type=str, default=None, help="Directory to save results (default: same as checkpoint)")
    parser.add_argument('--pos_weight_cluster', type=float, default=1.0, help="Positive weight for cluster loss")
    parser.add_argument('--pos_weight_se', type=float, default=1.0, help="Positive weight for SE loss")
    parser.add_argument('--lambda_cluster', type=float, default=0.5, help="Weight for cluster loss in total loss")
    parser.add_argument('--fix_jitters', type=int, nargs='+', default=[-175000, 0, 175000], help="List of jitters to fix during testing (e.g., --fix_jitters -175000 0 175000)")
    parser.add_argument('--chrs', type=str, nargs='+', default=None, help="List of chromosomes to evaluate, separated by space (default: None)")
    args = parser.parse_args()
    
    main(args.species, args.checkpoint, args.output_dir, args.pos_weight_cluster, args.pos_weight_se, args.lambda_cluster, args.fix_jitters, args.chrs)
