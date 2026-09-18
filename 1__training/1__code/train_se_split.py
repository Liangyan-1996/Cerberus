import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

import os
import argparse
import random
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np

from se_model_arch import BorzoiSEAdapter
from se_train_utils import build_dataset, hierarchical_loss, evaluate_unified, plot_training_history, plot_pr_curve_from_data

CONFIGS = {
    'human': {
        'fasta': '/public/home/liunangroup/liangyan/project/SE_model/data/genome/hg38.fa',
        'chrom_sizes': '/public/home/liunangroup/liangyan/project/SE_model/data/genomesize/hg38.chrom.sizes',
        'pool_file': '/public/home/liunangroup/liangyan/project/SE_model/data/AllSample.hg38.AnchorPool.parquet',
        'blacklist': '/public/home/liunangroup/liangyan/project/SE_model/data/blacklist/hg38.blacklist.bed',
        'label_cache': '/public/home/liunangroup/liangyan/project/SE_model/data/AllSample.hg38.SE.label_cache.pkl',
        'model_save_prefix': 'Cerberus_human'
        },
    'mouse': {
        'fasta': '/public/home/liunangroup/liangyan/project/SE_model/data/genome/mm10.fa',
        'chrom_sizes': '/public/home/liunangroup/liangyan/project/SE_model/data/genomesize/mm10.chrom.sizes',
        'pool_file': '/public/home/liunangroup/liangyan/project/SE_model/data/AllSample.mm10.AnchorPool.parquet',
        'blacklist': '/public/home/liunangroup/liangyan/project/SE_model/data/blacklist/mm10.blacklist.bed',
        'label_cache': '/public/home/liunangroup/liangyan/project/SE_model/data/AllSample.mm10.SE.label_cache.pkl',
        'model_save_prefix': 'Cerberus_mouse'
        }
    }

HF_MODEL_ID = '/public/home/liunangroup/liangyan/project/SE_model/pretrained_models/borzoi-replicate-0'
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 1
NUM_EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

def seed_all(seed=42):
    print(f"Setting random seed to {seed}...")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    # ==========
    parser = argparse.ArgumentParser()
    parser.add_argument('--species', type=str, default='human', choices=['human', 'mouse'])
    
    parser.add_argument('--lr_head', type=float, default=2e-4, help='Learning rate for the head')
    parser.add_argument('--lr_borzoi_ratio', type=float, default=0.01, help='Backbone LR = Head LR * ratio')
    parser.add_argument('--pos_weight_cluster', type=float, default=1.0, help='Weight for positive cluster samples in Loss')
    parser.add_argument('--pos_weight_se', type=float, default=1.0, help='Weight for positive SE samples in Loss')
    parser.add_argument('--lambda_cluster', type=float, default=0.5, help='Weight for cluster loss')
    parser.add_argument('--head_dropout', type=float, default=0.5, help='Dropout rate for the head')
    parser.add_argument('--block_dropout', type=float, default=0.2, help='Dropout rate for the blocks')
    parser.add_argument('--estimated_stop_epoch', type=int, default=6, help='Estimated stop epoch for training')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
    
    parser.add_argument('--unfreeze_attn_layers', type=int, default=0, help='Number of transformer layers to unfreeze')
    parser.add_argument('--unfreeze_decoder', action='store_true', help='Whether to unfreeze the decoder')
    
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to .pth checkpoint to load weights from')
    parser.add_argument('--save_dir', type=str, default='model_output', help='Output directory')
    parser.add_argument('--val_chroms', type=str, nargs='+', default=None,
                        help='Val chromosomes (e.g. chr9 chr14); default: random split')
    parser.add_argument('--test_chroms', type=str, nargs='+', default=None,
                        help='Reserved test chromosomes (e.g. chr3 chr11)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override BATCH_SIZE (default code constant)')
    args = parser.parse_args()

    global BATCH_SIZE
    if args.batch_size:
        BATCH_SIZE = args.batch_size
    print(f"[Config] lr_head={args.lr_head} lr_borzoi_ratio={args.lr_borzoi_ratio} "
          f"lambda_cluster={args.lambda_cluster} "
          f"pos_weight_cluster={args.pos_weight_cluster} "
          f"pos_weight_se={args.pos_weight_se} "
          f"est_stop={args.estimated_stop_epoch} patience={args.patience} "
          f"unfreeze_attn={args.unfreeze_attn_layers} "
          f"unfreeze_decoder={args.unfreeze_decoder} batch={BATCH_SIZE}")
    
    # ==========
    seed_all(SEED)
    print(f"=== Starting Training for Species: {args.species} ===")
    paths = CONFIGS[args.species]
    lr_head = args.lr_head
    lr_borzoi = lr_head * args.lr_borzoi_ratio
    weight_c_tensor = torch.tensor(args.pos_weight_cluster, device=DEVICE, dtype=torch.float32)
    weight_se_tensor = torch.tensor(args.pos_weight_se, device=DEVICE, dtype=torch.float32)
    lambda_cluster = args.lambda_cluster
    head_dropout = args.head_dropout
    block_dropout = args.block_dropout
    estimated_stop_epoch = args.estimated_stop_epoch
    patience = args.patience
    
    unfreeze_attn_layers = args.unfreeze_attn_layers
    unfreeze_decoder = True if unfreeze_attn_layers > 0 else args.unfreeze_decoder
    save_dir = f"{args.save_dir}"
    os.makedirs(save_dir, exist_ok=True)
    
    chrom_sizes = {line.split()[0]: int(line.split()[1]) for line in open(paths['chrom_sizes'])}
    std_chroms = sorted([c for c in chrom_sizes.keys() if ('_' not in c) and ('chrM' not in c) and (c != 'chrX') and (c != 'chrY')])
    if args.val_chroms and args.test_chroms:
        val_chroms = list(args.val_chroms)
        test_chroms = list(args.test_chroms)
        excluded = set(val_chroms) | set(test_chroms)
        train_chroms = [c for c in std_chroms if c not in excluded]
    else:
        random.shuffle(std_chroms)
        val_chroms = std_chroms[:2]
        test_chroms = std_chroms[2:4]
        train_chroms = std_chroms[4:]
    
    print(f"Train: {len(train_chroms)} chroms, Val: {val_chroms}, Test (Reserved): {test_chroms}")
    
    # Build Loaders
    print("Building training & validation Dataset...")
    train_dataset = build_dataset(train_chroms, paths, chrom_sizes, True, 200, None)
    val_dataset = build_dataset(val_chroms, paths, chrom_sizes, False, None, [0])
    # test_dataset = build_dataset(test_chroms, paths, chrom_sizes, False, None, fix_jitters=[-150000, 0, 150000])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True, prefetch_factor=2, persistent_workers=False)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True, prefetch_factor=2, persistent_workers=False)
    
    # ==========
    meta_cache = paths['label_cache'].replace('.pkl', '.metadata.pkl')
    with open(meta_cache, 'rb') as f: id_to_idx = pickle.load(f)
    num_cell_types = len(id_to_idx)
    model = BorzoiSEAdapter(HF_MODEL_ID, num_cell_types, unfreeze_attn_layers=unfreeze_attn_layers, unfreeze_decoder=unfreeze_decoder, head_dropout=head_dropout, block_dropout=block_dropout).to(DEVICE)
    
    # Load Checkpoint for 2nd-Stage fine-tuning
    if args.load_checkpoint:
        print(f"--> Loading full model checkpoint from: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=DEVICE, weights_only=True)
        
        new_state_dict = {}
        for k, v in checkpoint.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v
        
        load_result = model.load_state_dict(new_state_dict, strict=False)
        
        if len(load_result.missing_keys) > 0:
            print(f"    [Warning] Missing keys (init randomly): {load_result.missing_keys[:5]} ...")
        if len(load_result.unexpected_keys) > 0:
            print(f"    [Warning] Unexpected keys (ignored): {load_result.unexpected_keys[:5]} ...")
        
        print("--> Checkpoint loaded successfully.")
    
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        
        if "borzoi" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    print(f"Optimizing: {len(backbone_params)} backbone tensors, {len(head_params)} head tensors.")
    
    optimizer = optim.AdamW([{'params': head_params, 'lr': lr_head, 'weight_decay': 1e-2}, {'params': backbone_params, 'lr': lr_borzoi, 'weight_decay': 1e-6}])
    num_update_steps = len(train_dataset) * estimated_stop_epoch // (BATCH_SIZE * GRAD_ACCUM_STEPS)
    # num_update_steps = (len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM_STEPS)) * NUM_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.1 * num_update_steps), num_update_steps)
    
    history = {'train_total_loss': [], 'train_cluster_loss': [], 'train_se_loss': [], 'val_total_loss': [], 'val_cluster_loss': [], 'val_se_loss': [], 'val_auprc_cluster': [], 'val_auprc_se': []}
    best_val_auprc = 0.0
    
    # ==========
    # Training
    if estimated_stop_epoch > NUM_EPOCHS:
        train_epoch = estimated_stop_epoch
    else:
        train_epoch = NUM_EPOCHS
    
    for epoch in range(train_epoch):
        train_dataset.sampler.reshuffle()
        anchors_df = pd.DataFrame(train_dataset.sampler.current_epoch_anchors,columns=['chrom','center','jitter'])
        anchors_df = anchors_df.sort_values(by=['chrom','center'])
        anchors_df.to_csv(f"{save_dir}/anchors_epoch_{epoch+1}.csv", index=False)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True, prefetch_factor=2, persistent_workers=False, worker_init_fn=worker_init_fn)
        
        # ==========
        model.train()
        epoch_losses = {'total': 0, 'cluster': 0, 'se': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for step, (seqs, labels) in enumerate(pbar):
            seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                c_logits, s_logits = model(seqs)
                loss_d = hierarchical_loss(c_logits, s_logits, labels, weight_c_tensor, weight_se_tensor, lambda_cluster)
                loss = loss_d['total'] / GRAD_ACCUM_STEPS
            loss.backward()
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            for k in epoch_losses: epoch_losses[k] += loss_d[k].item()
            
            pbar.set_postfix({'T_Loss': f"{epoch_losses['total']/(step+1):.4f}", 'C_Loss': f"{epoch_losses['cluster']/(step+1):.4f}", 'SE_Loss': f"{epoch_losses['se']/(step+1):.4f}"})
        
        # ==========
        val_res, _ = evaluate_unified(model, val_loader, DEVICE, meta_cache, weight_c_tensor, weight_se_tensor, lambda_cluster, calc_per_cell=False, save_dir=None)
        for k in ['total', 'cluster', 'se']:
            history[f'train_{k}_loss'].append(epoch_losses[k] / (len(train_loader)))
            history[f'val_{k}_loss'].append(val_res[f'{k}_loss'])
        history['val_auprc_cluster'].append(val_res['auprc_cluster'])
        history['val_auprc_se'].append(val_res['auprc_se'])
        
        print(f"Epoch {epoch+1} T: total loss={history['train_total_loss'][-1]:.4f} | cluster loss={history['train_cluster_loss'][-1]:.4f} | se loss={history['train_se_loss'][-1]:.4f}")
        print(f"Epoch {epoch+1} V: total loss={history['val_total_loss'][-1]:.4f} | cluster loss={history['val_cluster_loss'][-1]:.4f} | se loss={history['val_se_loss'][-1]:.4f}")
        print(f"Epoch {epoch+1} V: cluster AUPRC={history['val_auprc_cluster'][-1]:.4f} | se AUPRC={history['val_auprc_se'][-1]:.4f}")
        
        if val_res['auprc_se'] > best_val_auprc:
            best_val_auprc = val_res['auprc_se']
            best_model_path = os.path.join(save_dir, f"{paths['model_save_prefix']}_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print("--> Best Model Saved!")
            
            if val_res['curve_data']:
                plot_pr_curve_from_data(val_res['curve_data'], val_res['auprc_cluster'], val_res['auprc_se'], os.path.join(save_dir, 'best_val_pr_curve.png'))
            
            early_stop_counter = 0
        
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("--> Early stopping triggered!")
                break
    
    plot_training_history(history, os.path.join(save_dir, 'training_curves.png'))

if __name__ == "__main__":
    main()
