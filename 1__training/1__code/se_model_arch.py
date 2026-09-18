import sys
import os
import torch
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from borzoi_pytorch.pytorch_borzoi_model import Borzoi
from borzoi_pytorch.config_borzoi import BorzoiConfig
from borzoi_pytorch.pytorch_borzoi_utils import TargetLengthCrop

# dilation conv block
class ResBlock1d(nn.Module):
    def __init__(self, channels, dilation, dropout_rate, kernel_size=3):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding='same', dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout_rate),
            nn.Conv1d(channels, channels, kernel_size, padding='same', dilation=dilation),
            nn.BatchNorm1d(channels)
            )
        self.act = nn.GELU(approximate='tanh')
    
    def forward(self, x):
        return self.act(x + self.convs(x))

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        # channel attention block
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1) # [bs, channel, len] > [bs, channel, 1]
        self.fc = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.GELU(approximate='tanh'),
            nn.Conv1d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
            )
    
    def forward(self, x):
        y = self.fc(self.avg_pool(x))
        return x * y

class BorzoiSEAdapter(nn.Module):
    def __init__(self, pretrained_model_path, num_cell_types, freeze_backbone=True, unfreeze_attn_layers=0, unfreeze_decoder=False, head_dropout=0.5, block_dropout=0.2):
        """
        Args:
            pretrained_model_path: Path to the pretrained Borzoi model. REQUIRED.
            num_cell_types: Number of output classes (tissues). REQUIRED.
            freeze_backbone: Whether to freeze Borzoi parameters. Default: True.
            unfreeze_last_n_layers: Number of last Transformer blocks to unfreeze. Default: 0.
        """
        super().__init__()
        
        self.freeze_backbone = freeze_backbone
        self.unfreeze_attn_layers = unfreeze_attn_layers
        if unfreeze_attn_layers > 0:
            unfreeze_decoder = True # decoder is always unfrozen when attn layers are unfrozen
        
        weight_path = os.path.join(pretrained_model_path, 'pytorch_model.bin')
        if os.path.exists(weight_path):
            print(f"Loading Pretrained Borzoi from {pretrained_model_path}...")
            config = BorzoiConfig.from_pretrained(pretrained_model_path)
            self.borzoi = Borzoi(config)
            state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
            self.borzoi.load_state_dict(state_dict)
            # configure
            self.borzoi.config.return_center_bins_only = False
            self.borzoi.crop = TargetLengthCrop(16384 - 32)
        
        if self.freeze_backbone:
            print("Freezing Borzoi Backbone Parameters...")
            for param in self.borzoi.parameters():
                param.requires_grad = False
            self.borzoi.eval()
        
        if self.unfreeze_attn_layers > 0:
            total_blocks = len(self.borzoi.transformer)
            start_idx = total_blocks - self.unfreeze_attn_layers
            if start_idx < 0: start_idx = 0
            
            print(f"--> UNFREEZING the last {self.unfreeze_attn_layers} transformer layers...")
            
            for i in range(start_idx, total_blocks):
                block = self.borzoi.transformer[i]
                for param in block.parameters():
                    param.requires_grad = True
        
        if unfreeze_decoder:
            print("--> UNFREEZING decoder layers (Upsampling/Separable Convs)...")
            decoder_layers = [self.borzoi.upsampling_unet1, self.borzoi.separable1, self.borzoi.upsampling_unet0, self.borzoi.separable0, self.borzoi.horizontal_conv1, self.borzoi.horizontal_conv0]
            for layer in decoder_layers:
                for param in layer.parameters():
                    param.requires_grad = True
        
        input_dim = self.borzoi.config.dim
        self.output_bins = 1600
        
        self.downsampler = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=9, stride=8, padding=1, groups=input_dim),
            nn.BatchNorm1d(input_dim),
            nn.GELU(approximate='tanh')
            )
        
        self.cluster_head = nn.Sequential(
            nn.Conv1d(input_dim, 512, kernel_size=1, padding='same'),
            nn.BatchNorm1d(512), nn.GELU(approximate='tanh'), nn.Dropout(head_dropout),
            ResBlock1d(512, 1, block_dropout),
            ResBlock1d(512, 2, block_dropout),
            ResBlock1d(512, 4, block_dropout),
            ResBlock1d(512, 8, block_dropout),
            ResBlock1d(512, 16, block_dropout),
            ResBlock1d(512, 32, block_dropout),
            nn.Conv1d(512, num_cell_types, kernel_size=1)
            )
        
        se_input_dim = input_dim + num_cell_types
        self.fusion_attention = ChannelAttention(se_input_dim, reduction=16)
        
        self.se_head = nn.Sequential(
            nn.Conv1d(se_input_dim, 512, kernel_size=3, padding='same'), 
            nn.BatchNorm1d(512), nn.GELU(approximate='tanh'), nn.Dropout(head_dropout),
            ResBlock1d(512, 1, block_dropout),
            ResBlock1d(512, 2, block_dropout),
            ResBlock1d(512, 4, block_dropout),
            ResBlock1d(512, 8, block_dropout),
            nn.Conv1d(512, 256, kernel_size=1, padding='same'),
            nn.BatchNorm1d(256), nn.GELU(approximate='tanh'), nn.Dropout(head_dropout),
            nn.Conv1d(256, num_cell_types, kernel_size=1)
            )
    
    def forward(self, x):
        features = self.borzoi.get_embs_after_crop(x)
        features_low = self.downsampler(features) # [bs, 1536, 16352] -> [bs, 1536, 2044]
        cluster_logits_full = self.cluster_head(features_low) # [bs, 1536, 2044] > [bs, NumTypes, 2044]
        se_input = torch.cat([features_low, cluster_logits_full], dim=1) # [bs, 1536+NumTypes, 2044]
        se_input = self.fusion_attention(se_input)
        se_logits_full = self.se_head(se_input) # [bs, 1536+NumTypes, 2044] > [bs, NumTypes, 2044]
        
        current_bins = se_logits_full.shape[-1]
        start = (current_bins - self.output_bins) // 2
        end = start + self.output_bins
        cluster_logits_out = cluster_logits_full[:, :, start:end]
        se_logits_out = se_logits_full[:, :, start:end] # [bs, NumTypes, 2044] > [bs, NumTypes, 1600]
        
        return cluster_logits_out, se_logits_out
    
    def train(self, mode=True):
        super().train(mode)
        
        if self.freeze_backbone:
            self.borzoi.eval()
            
            if mode and self.unfreeze_attn_layers > 0:
                total_blocks = len(self.borzoi.transformer)
                start_idx = total_blocks - self.unfreeze_attn_layers
                if start_idx < 0: start_idx = 0
                # open dropout in unfreezed Transformer layers
                for i in range(start_idx, total_blocks):
                    self.borzoi.transformer[i].train()
