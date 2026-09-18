import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import pickle
import threading
from tqdm import tqdm
from se_data_utils import GenomicSampler, GenomicDataset

def build_dataset(chrom_subset, paths, chrom_sizes, is_train=True, samples_se=None, fixed_jitters=None):
    # 1. Config Sampler based on mode
    if samples_se is None:
        samples_se = 200 if is_train else 1000000 # 200 for train, full for eval
    
    sampler = GenomicSampler(pool_path=paths['pool_file'], chrom_sizes_dict=chrom_sizes, chrom_subset_list=chrom_subset, blacklist_path=paths['blacklist'], samples_per_cell_se=samples_se)
    # 2. Config Jitter logic
    dataset = GenomicDataset(sampler=sampler, label_cache_path=paths['label_cache'], fasta_path=paths['fasta'], chrom_sizes=chrom_sizes, fixed_jitters=fixed_jitters)
    
    return dataset

# Loss Functions
def hierarchical_loss(cluster_logits:torch.Tensor, se_logits:torch.Tensor, targets:torch.Tensor, weight_c_tensor:torch.Tensor, weight_se_tensor:torch.Tensor, lambda_cluster=0.5):
    # Cluster Loss
    # values > 0.5 indicate cluster regions (0.75 or 1.0 or 2.0 -> 1.0)
    target_cluster = torch.clamp(targets, 0.0, 1.0)
    loss_cluster = nn.BCEWithLogitsLoss(pos_weight=weight_c_tensor)(cluster_logits, target_cluster)
    # SE Loss
    # values > 1.5 indicate super enhancer regions (0/0.75/1.0 -> 0.0, 1.75/2.0 -> >0)
    target_se = torch.clamp(targets - 1.0, 0.0, 1.0)
    loss_se = nn.BCEWithLogitsLoss(pos_weight=weight_se_tensor)(se_logits, target_se)
    # Weighted Sum
    loss = lambda_cluster * loss_cluster + loss_se
    
    return {'total': loss, 'cluster': loss_cluster, 'se': loss_se}

# Metric Utilities
def fast_auprc_torch(y_true, y_score):
    if y_true.numel() == 0: return None, None, None, 0.0
    
    desc_indices = torch.argsort(y_score, descending=True)
    y_score_sorted = y_score[desc_indices]
    y_true_sorted = y_true[desc_indices]
    
    tp_cumsum = torch.cumsum(y_true_sorted, dim=0).float()
    fp_cumsum = torch.cumsum(1 - y_true_sorted, dim=0).float()
    total_pos = tp_cumsum[-1]
    
    if total_pos == 0: return None, None, None, 0.0
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recall = tp_cumsum / (total_pos + 1e-10)
    auprc = torch.sum(precision * y_true_sorted) / total_pos
    
    return precision, recall, y_score_sorted, auprc.item()

def find_optimal_threshold(precision, recall, thresholds):
    if precision is None: return 0.5, 0.0
    
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = torch.argmax(f1_scores)
    best_f1 = f1_scores[best_idx].item()
    
    idx = min(best_idx, len(thresholds)-1)
    best_thresh = thresholds[idx].item()
    
    return best_thresh, best_f1

def subsample_curve(rec, prec, device, max_points=5000):
    if rec is None: return (None, None)
    
    length = len(rec)
    if length > max_points:
        indices = torch.linspace(0, length-1, steps=max_points).long()
        indices = torch.clamp(indices, 0, length - 1).to(device)
        return (rec[indices].cpu().numpy(), prec[indices].cpu().numpy())
    
    return (rec.cpu().numpy(), prec.cpu().numpy())

# Plotting Utilities
def plot_pr_curve_from_data(curve_data, auprc_c, auprc_s, save_path):
    """
    Plot precision-recall curves for cluster and super enhancer predictions
    Includes baseline comparison for SE prediction
    """
    plt.figure(figsize=(8, 6))
    rec_c, prec_c = curve_data['cluster']
    rec_s, prec_s = curve_data['se']
    
    if rec_c is not None:
        plt.plot(rec_c, prec_c, color='purple', lw=2, label=f'Cluster (AP={auprc_c:.3f})')
    if rec_s is not None:
        plt.plot(rec_s, prec_s, color='orange', lw=2, label=f'Super Enhancer (AP={auprc_s:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axhline(y=curve_data.get('baseline_se', 0.0), color='orange', linestyle='--', alpha=0.5, label='SE Baseline')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()

def plot_training_history(history, save_path):
    """
    Plot training history with 4 subplots: total loss, cluster loss, SE loss, and SE AUPRC
    Shows both training and validation curves for loss metrics
    """
    epochs = range(1, len(history['train_total_loss']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics_map = [('total_loss', 'Total Loss'), ('cluster_loss', 'Cluster Loss'), ('se_loss', 'SE Loss'), ('auprc_se', 'SE AUPRC (Val)')]
    for idx, (key, title) in enumerate(metrics_map):
        ax = axes[idx // 2, idx % 2]
        if key.startswith('auprc'):
            ax.plot(epochs, history[f'val_{key}'], 'g-s', label='Val')  # only validation for AUPRC
        else:
            ax.plot(epochs, history[f'train_{key}'], 'b-o', label='Train')
            ax.plot(epochs, history[f'val_{key}'], 'g-s', label='Val')
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close(fig)

def plot_per_cell_performance(df, save_path):
    """
    Plot per-cell-type performance analysis
    Left: Boxplot comparing cluster vs SE AUPRC distributions
    Right: Scatter plot showing relationship between SE abundance and performance
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.boxplot(df[['Cluster AUPRC','SE AUPRC']].melt(var_name='Metrics', value_name='AUPRC'), x = 'Metrics', y = 'AUPRC', hue='Metrics', ax=ax[0],palette={'Cluster AUPRC':'purple','SE AUPRC':'orange'}, width=0.5)
    ax[0].set_title('Performance Distribution across Cell Types')
    ax[0].set_ylabel('AUPRC')
    ax[0].grid(axis='y', alpha=0.3)
    sns.scatterplot(x='Num SE', y='SE AUPRC', data=df, alpha=0.6, color='orange', ax=ax[1])
    ax[1].set_xlabel('Number of SE Bins')
    ax[1].set_ylabel('SE AUPRC')
    ax[1].set_title('SE AUPRC vs Data Abundance')
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close(fig)

# [Evaluation] Saving Results to Disk
def _save_worker(save_dir, labels, cluster_probs, se_probs, mode='raw'):
    try:
        if mode == 'raw':
            base_name = os.path.join(save_dir, 'test_preds')
            np.save(f"{base_name}.labels.npy", labels)
            np.save(f"{base_name}.cluster_probs.npy", cluster_probs)
            np.save(f"{base_name}.se_probs.npy", se_probs)
            print(f"[Background] Saved RAW .npy files to {base_name}.*.npy")
        elif mode == 'compressed':
            np.savez_compressed(os.path.join(save_dir, 'test_preds.npz'), labels=labels, cluster_probs=cluster_probs, se_probs=se_probs)
            print(f"[Background] Saved COMPRESSED .npz to {os.path.join(save_dir, 'test_preds.npz')}")
        else:
            raise ValueError(f"Unknown mode: {mode}")
    except Exception as e:
        print(f"[Background] Error during saving: {e}")

def dispatch_save_task(save_dir, y_true, y_c, y_s, mode='raw'):
    labels_np, c_probs_np, s_probs_np = y_true.numpy(), y_c.numpy(), y_s.numpy()
    args = (save_dir, labels_np, c_probs_np, s_probs_np, mode)
    t = threading.Thread(target=_save_worker, args=args, daemon=True)
    t.start()

# [Evaluation] Inference Loop
def run_inference_loop(model, dataloader, device, weight_c_tensor, weight_se_tensor, lambda_cluster):
    all_targets_cpu, all_p_cluster_cpu, all_p_se_cpu = [], [], []
    metrics_accum = {'total_loss': 0.0, 'cluster_loss': 0.0, 'se_loss': 0.0}
    steps = 0
    model.eval()
    
    with torch.no_grad():
        for seqs, labels in tqdm(dataloader, desc="Inference"):
            seqs, labels = seqs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                # 1. Forward
                cluster_logits, se_logits = model(seqs)
                # 2. Loss Calculation
                loss_dict = hierarchical_loss(cluster_logits, se_logits, labels, weight_c_tensor, weight_se_tensor, lambda_cluster)
                
                for k in metrics_accum: 
                    metrics_accum[k] += loss_dict[k.replace('_loss', '')].item()
                
                steps += 1
                
                # 3. Sigmoid & Quantization for CPU Storage
                p_c = torch.sigmoid(cluster_logits)
                p_se = torch.sigmoid(se_logits)
                
            # [B, C, L] -> [B, L, C] -> [-1, C]
            C = labels.shape[1]
            # Labels to int8 & Probs to float16
            labels_cpu = (labels * 4).round().to(torch.int8).permute(0, 2, 1).reshape(-1, C).cpu()
            p_c_cpu = p_c.permute(0, 2, 1).reshape(-1, C).to(torch.float16).cpu()
            p_se_cpu = p_se.permute(0, 2, 1).reshape(-1, C).to(torch.float16).cpu()
            
            all_targets_cpu.append(labels_cpu)
            all_p_cluster_cpu.append(p_c_cpu)
            all_p_se_cpu.append(p_se_cpu)
    
    del seqs, labels, cluster_logits, se_logits
    torch.cuda.empty_cache()
    
    # [Evaluation] Concatenating data on CPU
    y_true_all = torch.cat(all_targets_cpu, dim=0)
    y_score_c_all = torch.cat(all_p_cluster_cpu, dim=0)
    y_score_s_all = torch.cat(all_p_se_cpu, dim=0)
    
    avg_loss = {k: v/steps for k, v in metrics_accum.items()}
    
    return y_true_all, y_score_c_all, y_score_s_all, avg_loss

# [Evaluation] Calculating Global Metrics
def calc_global_metrics(y_true_all, y_score_c_all, y_score_s_all, avg_loss, device, sampling_rate=None):
    # Flatten on CPU
    y_true_flat_cpu = y_true_all.reshape(-1)
    y_score_c_flat_cpu = y_score_c_all.reshape(-1)
    y_score_s_flat_cpu = y_score_s_all.reshape(-1)
    
    total_points = y_true_flat_cpu.numel()
    
    MAX_SAMPLES = 30_000_000
    target_samples = MAX_SAMPLES
    if sampling_rate is not None: target_samples = min(int(total_points * sampling_rate), MAX_SAMPLES)
    
    # Deterministic Sampling
    g_cpu = torch.Generator()
    g_cpu.manual_seed(42)
    indices = torch.randint(0, total_points, (target_samples,), generator=g_cpu)
    
    # Move subset to GPU & Binarize
    y_true_flat = (y_true_flat_cpu[indices].to(device).float() / 4.0)
    y_score_c_flat = y_score_c_flat_cpu[indices].to(device).float()
    y_score_s_flat = y_score_s_flat_cpu[indices].to(device).float()
    
    # Calculate Metrics (GPU)
    prec_c, rec_c, thresh_c, auprc_c = fast_auprc_torch((y_true_flat > 0.5).float(), y_score_c_flat)
    prec_s, rec_s, thresh_s, auprc_s = fast_auprc_torch((y_true_flat > 1.5).float(), y_score_s_flat)
    
    best_t_cluster, best_f1_cluster, best_t_se, best_f1_se = 0.5, 0.0, 0.5, 0.0
    if prec_c is not None: best_t_cluster, best_f1_cluster = find_optimal_threshold(prec_c, rec_c, thresh_c)
    if prec_s is not None: best_t_se, best_f1_se = find_optimal_threshold(prec_s, rec_s, thresh_s)
    
    # Result
    results = avg_loss.copy()
    baseline_acc_se = torch.mean((y_true_flat > 1.5).float()).item()
    cluster_curve = subsample_curve(rec_c, prec_c, 'cpu')
    se_curve = subsample_curve(rec_s, prec_s, 'cpu')
    curve_data = {'cluster': cluster_curve, 'se': se_curve, 'baseline_se': baseline_acc_se}
    
    results.update({
        'auprc_cluster': auprc_c, 'auprc_se': auprc_s,
        'best_f1_cluster': best_f1_cluster, 'best_thresh_cluster': best_t_cluster,
        'best_f1_se': best_f1_se, 'best_thresh_se': best_t_se,
        'curve_data': curve_data
    })
    
    return results

# [Evaluation] Calculating Per-Cell Metrics
def calc_per_cell_metrics(y_true_all, y_score_c_all, y_score_s_all, id_to_idx_path, device):
    with open(id_to_idx_path, 'rb') as f: id_to_idx = pickle.load(f)
    idx_to_id = {v: k for k, v in id_to_idx.items()}
    num_cell_types = len(id_to_idx)
    results_list = []
    
    for idx in tqdm(range(num_cell_types), desc="Per-Cell"):
        cell_name = idx_to_id[idx]
        
        # Column Slice -> GPU -> Binarize
        y_true_col = (y_true_all[:, idx].to(device, non_blocking=True).float() / 4.0)
        y_score_c_col = y_score_c_all[:, idx].to(device, non_blocking=True).float()
        y_score_s_col = y_score_s_all[:, idx].to(device, non_blocking=True).float()
        
        _, _, _, ap_c = fast_auprc_torch((y_true_col > 0.5).float(), y_score_c_col)
        _, _, _, ap_s = fast_auprc_torch((y_true_col > 1.5).float(), y_score_s_col)
        
        num_se = int((y_true_col > 1.5).sum().item())
        results_list.append({'Cell Type': cell_name, 'Cluster AUPRC': ap_c, 'SE AUPRC': ap_s, 'Num SE': num_se})
        
    return pd.DataFrame(results_list).sort_values(by='SE AUPRC', ascending=True).reset_index(drop=True)

# Main Evaluation Wrapper
def evaluate_unified(model, dataloader, device, id_to_idx_path, weight_c_tensor, weight_se_tensor, lambda_cluster=0.5, calc_per_cell=False, save_dir=None):
    # 1. Run Inference
    y_true_all, y_score_c_all, y_score_s_all, avg_loss = run_inference_loop(model, dataloader, device, weight_c_tensor, weight_se_tensor, lambda_cluster)
    
    # 2. Optional: Save Raw Predictions (for debugging or ensemble)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        dispatch_save_task(save_dir, y_true_all, y_score_c_all, y_score_s_all, mode='raw')
    
    # 3. Calculate Global Metrics
    global_res = calc_global_metrics(y_true_all, y_score_c_all, y_score_s_all, avg_loss, device, sampling_rate=0.1)
    
    # 4. Calculate Per-Cell Metrics (Conditional)
    df_metrics = None
    if calc_per_cell:
        df_metrics = calc_per_cell_metrics(y_true_all, y_score_c_all, y_score_s_all, id_to_idx_path, device)
    
    return global_res, df_metrics
