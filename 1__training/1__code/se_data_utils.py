import torch
import numpy as np
import pandas as pd

import pickle
import os
import random
from tqdm import tqdm

from torch.utils.data import Dataset

INPUT_LEN = 524288
OUTPUT_LEN = 409600
BG_JITTER = 50000
MAX_JITTER = 150000
RESOLUTION = 256
HALF_INPUT = INPUT_LEN // 2

# =============================================================================
# 1. Helper: Fast DNA One-hot Encoding
# =============================================================================
def dna_one_hot(seq_arr):
    SEQ_MAP = np.zeros((256, 4), dtype=np.float32)
    SEQ_MAP[65] = [1, 0, 0, 0]  # A
    SEQ_MAP[67] = [0, 1, 0, 0]  # C
    SEQ_MAP[71] = [0, 0, 1, 0]  # G
    SEQ_MAP[84] = [0, 0, 0, 1]  # T
    one_hot_L4 = SEQ_MAP[seq_arr] 
    one_hot_4L = np.ascontiguousarray(one_hot_L4.T)    
    return torch.from_numpy(one_hot_4L)

# =============================================================================
# 2. Label Generator (With Soft Labeling)
# =============================================================================
def generate_label_from_cache(chrom, window_center, label_cache, num_cell_types):
    seq_start = window_center - HALF_INPUT
    seq_end = window_center + HALF_INPUT
    total_bins = INPUT_LEN // RESOLUTION
    target_bins = OUTPUT_LEN // RESOLUTION
    crop_offset = (total_bins - target_bins) // 2
    full_mask = np.zeros((num_cell_types, total_bins), dtype=np.float32) # float32 for soft labels
    # generate labels for each cell type
    for ct_idx in range(num_cell_types):
        if (ct_idx not in label_cache) or (chrom not in label_cache[ct_idx]): continue
        ct_data = label_cache[ct_idx][chrom]
        # --- A. Cluster Label ---
        clusters = ct_data.get('Cluster', [])
        if len(clusters) > 0:
            c_starts = clusters[:, 0]; c_ends = clusters[:, 1]
            in_window = (c_starts < seq_end) & (c_ends > seq_start) # indicator
            relevant = clusters[in_window]
            
            for row in relevant:
                bin_s = max(0, int((row[0] - seq_start) / RESOLUTION))
                bin_e = min(total_bins, int(np.ceil((row[1] - seq_start) / RESOLUTION)))
                
                if bin_e > bin_s:
                    full_mask[ct_idx, bin_s:bin_e] = 1.0
                    full_mask[ct_idx, bin_s] = 0.75 # Cluster start edge
                    if bin_e - 1 > bin_s: full_mask[ct_idx, bin_e - 1] = 0.75 # Cluster end edge
        
        # --- B. SuperEnhancer Label ---
        ses = ct_data.get('SuperEnhancer', [])
        if len(ses) > 0:
            s_starts = ses[:, 0]; s_ends = ses[:, 1]
            in_window = (s_starts < seq_end) & (s_ends > seq_start) # indicator
            relevant = ses[in_window]
            
            for row in relevant:
                bin_s = max(0, int((row[0] - seq_start) / RESOLUTION))
                bin_e = min(total_bins, int(np.ceil((row[1] - seq_start) / RESOLUTION)))
                
                if bin_e > bin_s:
                    full_mask[ct_idx, bin_s:bin_e] = 2.0
                    full_mask[ct_idx, bin_s] = 1.75 # SE start edge
                    if bin_e - 1 > bin_s: full_mask[ct_idx, bin_e - 1] = 1.75 # SE end edge
        
    # Center Crop
    cropped_mask = full_mask[:, crop_offset : crop_offset + target_bins]
    return torch.from_numpy(cropped_mask)

# =============================================================================
# 2. Sampler
# =============================================================================
class GenomicSampler:
    def __init__(self, pool_path, chrom_sizes_dict, chrom_subset_list, blacklist_path=None, samples_per_cell_se=200):
        
        self.chrom_sizes = chrom_sizes_dict
        self.chrom_subset = chrom_subset_list
        self.samples_se = samples_per_cell_se
        
        print(f"[Sampler] Loading Pool from {pool_path}...")
        self.full_pool = pd.read_parquet(pool_path)    
        self.full_pool = self.full_pool[self.full_pool['chrom'].isin(self.chrom_subset)]
        
        self.blacklist = None
        if blacklist_path and os.path.exists(blacklist_path):
            print(f"[Sampler] Loading Blacklist...")
            self.blacklist = pd.read_table(blacklist_path, header=None, names=['chr', 'start', 'end'])
            if self.chrom_subset:
                self.blacklist = self.blacklist[self.blacklist['chr'].isin(self.chrom_subset)]
        
        self.candidates = {} 
        self._preprocess_all_candidates()
        
        self.bg_pool = []
        self._precalculate_background_pool()
        
        self.current_epoch_anchors = []
        self.reshuffle()

    def _merge_and_split(self, df_subset):
        """
        静态处理函数：输入 DataFrame -> Merge -> Split -> 返回 Block 列表
        Block 格式: (chrom, center, jitter_radius)
        """
        if df_subset.empty: return []
        
        # 1. Merge (基于 pandas 的高效操作)
        intervals = df_subset[['chrom', 'start', 'end']].sort_values(['chrom', 'start']).values.tolist()
        merged = []
        if intervals:
            curr_chrom, curr_start, curr_end = intervals[0]
            for i in range(1, len(intervals)):
                chrom, start, end = intervals[i]
                if chrom == curr_chrom and start <= curr_end + 12500: # Distance 12.5kb
                    curr_end = max(curr_end, end)
                else:
                    merged.append((curr_chrom, curr_start, curr_end))
                    curr_chrom, curr_start, curr_end = chrom, start, end
            merged.append((curr_chrom, curr_start, curr_end))
        
        # 2. Split (长片段切分)
        anchors = []
        for chrom, start, end in merged:
            length = end - start
            if length > INPUT_LEN:
                num_blocks = int(np.ceil(length / INPUT_LEN))
                block_size = length / num_blocks
                for i in range(num_blocks):
                    b_start = start + i * block_size
                    b_end = start + (i + 1) * block_size
                    center = int((b_start + b_end) / 2)
                    jitter = int(block_size / 2)
                    anchors.append((chrom, center, jitter))
            else:
                center = int((start + end) / 2)
                # jitter = int(length / 2)
                jitter = MAX_JITTER
                anchors.append((chrom, center, jitter))
        
        return anchors
    
    def _preprocess_all_candidates(self):
        """一次性遍历所有细胞类型，生成“原子化”的训练候选项"""
        unique_cells = self.full_pool['id'].unique()
        print(f"[Init] Pre-processing regions for {len(unique_cells)} cell types...")
        
        for cell_id in tqdm(unique_cells, desc="Building Universe"):
            self.candidates[cell_id] = {}
            cell_mask = self.full_pool['id'] == cell_id
            # SE
            se_df = self.full_pool[cell_mask & (self.full_pool['isSuper'] == 1)]
            self.candidates[cell_id]['SE'] = self._merge_and_split(se_df)
            # Cluster
            cl_df = self.full_pool[cell_mask & (self.full_pool['isSuper'] == 0)]
            self.candidates[cell_id]['Cluster'] = self._merge_and_split(cl_df)
    
    def _precalculate_background_pool(self, pool_size=500000):
        """预生成大量合法的背景点，避免训练时卡顿"""
        print(f"[Init] Pre-generating {pool_size} background anchors...")
        valid_anchors = []
        chrom_keys = list(self.chrom_subset)
        total_len = sum(self.chrom_sizes[c] for c in chrom_keys)
        probs = [self.chrom_sizes[c]/total_len for c in chrom_keys]
        
        pbar = tqdm(total=pool_size, desc="Gen BG")
        while len(valid_anchors) < pool_size:
            batch_size = 50000
            batch_chroms = np.random.choice(chrom_keys, size=batch_size, p=probs)
            batch_centers = []
            for c in batch_chroms:
                size = self.chrom_sizes[c]
                batch_centers.append(np.random.randint(HALF_INPUT, size - HALF_INPUT))
            batch_centers = np.array(batch_centers)
            # filter
            is_bad = self._is_blacklisted_batch(batch_chroms, batch_centers)
            # valid anchors
            for c, center in zip(batch_chroms[~is_bad], batch_centers[~is_bad]):
                valid_anchors.append((c, center, BG_JITTER))
            
            pbar.update(len(valid_anchors) - pbar.n)
            if len(valid_anchors) > pool_size:
                valid_anchors = valid_anchors[:pool_size]
        
        self.bg_pool = np.array(valid_anchors, dtype=object)
    
    def reshuffle(self):
        """极速 Reshuffle: 只做随机抽样，不做几何计算"""
        fg_anchors = []
        
        # foreground sampling
        for cell_id in self.candidates:
            # A. SE Sampling
            se_candidates = self.candidates[cell_id]['SE']
            if not se_candidates: continue
            
            n_se = min(len(se_candidates), self.samples_se)
            curr_se = random.sample(se_candidates, n_se)
            fg_anchors.extend(curr_se)
            
            # B. Cluster Sampling (Balance with SE)
            cl_candidates = self.candidates[cell_id]['Cluster']
            if not cl_candidates: continue
            
            n_cl = min(len(cl_candidates), n_se) # SE:Cluster = 1:1
            curr_cl = random.sample(cl_candidates, n_cl)
            fg_anchors.extend(curr_cl)
        
        # print(f"[reshuffle] Before deduplication: {len(fg_anchors)} foreground anchors")
        
        # deduplication
        # remove duplicates sampled from different cell types
        # fg_anchors = list(set(fg_anchors))
        # print(f"[Init] After deduplication: {len(fg_anchors)} foreground anchors")
        
        # background Sampling
        num_bg = len(fg_anchors) // 2
        bg_indices = np.random.choice(len(self.bg_pool), size=num_bg, replace=False)
        bg_anchors = self.bg_pool[bg_indices].tolist()
        # print(f"[reshuffle] Generated {num_bg} background anchors")
        
        # finalize
        self.current_epoch_anchors = fg_anchors + bg_anchors
        # print(f"[reshuffle] Total anchors in epoch: {len(self.current_epoch_anchors)}")
        random.shuffle(self.current_epoch_anchors)
        
    def _is_blacklisted_batch(self, chroms, centers):
        """批量检查生成的背景点是否在黑名单中"""
        if self.blacklist is None:
            return np.zeros(len(chroms), dtype=bool)
        
        is_bad = np.zeros(len(chroms), dtype=bool)
        unique_chroms = np.unique(chroms)
        for chrom in unique_chroms:
            bl_chrom = self.blacklist[self.blacklist['chr'] == chrom]
            if bl_chrom.empty: continue
            
            idx_in_batch = np.where(chroms == chrom)[0]
            points = centers[idx_in_batch]
            bl_starts = bl_chrom['start'].values
            bl_ends = bl_chrom['end'].values
            
            for i, p in zip(idx_in_batch, points):
                if np.any((bl_starts <= p) & (bl_ends >= p)):
                    is_bad[i] = True
        return is_bad
    
    def __len__(self):
        return len(self.current_epoch_anchors)
    
    def __getitem__(self, idx):
        return self.current_epoch_anchors[idx]

# =============================================================================
# 3. Dataset
# =============================================================================
class GenomicDataset(Dataset):
    def __init__(self, sampler, label_cache_path, fasta_path, chrom_sizes, fixed_jitters=None):
        self.sampler = sampler
        self.fasta_path = fasta_path
        self.chrom_sizes = chrom_sizes
        self.fixed_jitters = fixed_jitters
        self.is_train = fixed_jitters is None
        
        with open(label_cache_path, 'rb') as f: self.label_cache = pickle.load(f) # label cache
        
        meta_path = label_cache_path.replace('.pkl', '.metadata.pkl') # meta
        with open(meta_path, 'rb') as f: self.id_to_idx = pickle.load(f)
        self.num_cell_types = len(self.id_to_idx)
        
        self.genome = self._load_or_create_mmap(fasta_path, chrom_sizes) # genome memmap
    
    def _load_or_create_mmap(self, fasta_path, chrom_sizes):
        genome_dict = {}
        temp_fasta = None
        for chrom in chrom_sizes.keys():
            npy_file = f"{fasta_path}.{chrom}.npy"
            if not os.path.exists(npy_file):
                print(f"Generating .npy for {chrom} (First time only)...")
                if temp_fasta is None:import pyfaidx; temp_fasta = pyfaidx.Fasta(fasta_path)
                seq_str = str(temp_fasta[chrom])
                seq_bytes = seq_str.upper().encode('utf-8')
                seq_arr = np.frombuffer(seq_bytes, dtype=np.int8)
                np.save(npy_file, seq_arr)
            genome_dict[chrom] = np.load(npy_file, mmap_mode='r')
        return genome_dict
    
    def __len__(self):
        num_blocks = len(self.sampler) * len(self.fixed_jitters) if self.fixed_jitters else len(self.sampler)
        return num_blocks
    
    def __getitem__(self, idx):
        # 1. Anchor & Jitter
        if self.is_train:
            sampler_idx = idx
            chrom, center, sample_max_jitter = self.sampler[sampler_idx]
            jitter = random.randint(-sample_max_jitter, sample_max_jitter) if sample_max_jitter > 0 else 0
        else:
            sampler_idx = idx // len(self.fixed_jitters)
            jitter_idx = idx % len(self.fixed_jitters)
            jitter = self.fixed_jitters[jitter_idx]
            chrom, center, _ = self.sampler[sampler_idx]
        # 2. Window Coordinates
        shifted_center = center + jitter
        chrom_len = self.chrom_sizes[chrom]
        min_safe = HALF_INPUT; max_safe = chrom_len - HALF_INPUT
        if shifted_center < min_safe: shifted_center = min_safe
        if shifted_center > max_safe: shifted_center = max_safe
        start = shifted_center - HALF_INPUT; end = shifted_center + HALF_INPUT
        # 3. Retrieve Sequence
        seq_arr = self.genome[chrom][start:end]
        one_hot = dna_one_hot(seq_arr)
        assert one_hot.shape[-1] == INPUT_LEN
        # 4. Generate Labels
        labels = generate_label_from_cache(chrom, shifted_center, self.label_cache, self.num_cell_types)
        # 5. Augmentation (training)
        if self.is_train and random.random() > 0.5:
            one_hot = torch.flip(one_hot, dims=[1])
            one_hot = one_hot[[3, 2, 1, 0], :]
            labels = torch.flip(labels, dims=[1])
        return one_hot, labels