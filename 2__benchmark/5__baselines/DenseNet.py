import torch
import torch.nn as nn
import torch.nn.functional as F
from CBAM import CBAMLayer


# 1D Transition Layer
class TransitionLayer1D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super(TransitionLayer1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        return x


# 1D Dense Block
class DenseBlock1D(nn.Module):
    def __init__(self, in_channels, layers, growth_rate, dropout_rate=None):
        super(DenseBlock1D, self).__init__()
        self.layers = layers
        self.growth_rate = growth_rate
        self.dropout_rate = dropout_rate

        # Define the convolution layers for the DenseBlock
        self.conv_layers = nn.ModuleList([self._conv_factory(in_channels + i * growth_rate) for i in range(layers)])

    def _conv_factory(self, in_channels):
        layers = [
            nn.Conv1d(in_channels, self.growth_rate, kernel_size=61, padding=30, bias=False),
            nn.BatchNorm1d(self.growth_rate),
            nn.ReLU(inplace=True),
        ]
        if self.dropout_rate:
            layers.append(nn.Dropout(self.dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = [x]
        for layer in self.conv_layers:
            out = layer(torch.cat(features, 1))  # Concatenate all previous feature maps
            features.append(out)
        return torch.cat(features, 1)  # Concatenate all features at the end


# Optimized DenseNet Model for 1D Data
class DenseNet1D(nn.Module):
    def __init__(self, in_channels=64, growth_rate=32, layers_per_block=4, dropout_rate=0.2, num_blocks=2):
        super(DenseNet1D, self).__init__()

        # Initial Convolution Layer
        self.initial_conv = nn.Conv1d(in_channels, growth_rate, kernel_size=81, stride=1, padding=40, bias=False)
        self.initial_batch_norm = nn.BatchNorm1d(growth_rate)
        self.initial_relu = nn.ReLU(inplace=True)

        # Define Dense Blocks and Transition Layers
        num_features = growth_rate
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        self.cbam_attentions = nn.ModuleList()

        # Create Dense Blocks and Transition Layers
        for i in range(num_blocks):
            # Create a Dense Block
            self.dense_blocks.append(DenseBlock1D(num_features, layers_per_block, growth_rate, dropout_rate))
            num_features += layers_per_block * growth_rate  # Update the number of features after each DenseBlock

            # Create a Transition Layer after each Dense Block (except the last one)
            if i < num_blocks - 1:
                self.cbam_attentions.append(
                    CBAMLayer(channel=num_features)
                )
                self.transitions.append(TransitionLayer1D(num_features, num_features // 2, dropout_rate))
                num_features = num_features // 2  # Reduce the number of features after the transition

        # Fully connected layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # Initial Convolution
        x = self.initial_conv(x)
        x = self.initial_batch_norm(x)
        x = self.initial_relu(x)

        # Pass through each Dense Block and Transition Layer
        for i in range(len(self.dense_blocks)):
            x = self.dense_blocks[i](x)
            if i < len(self.transitions):

                x = self.cbam_attentions[i](x)
                x = self.transitions[i](x)

        x = self.global_pool(x)
        return x




def calculate_num_features(growth_rate, layers_per_block, num_blocks):

    num_features = growth_rate  


    for block_idx in range(num_blocks):

        num_features += layers_per_block * growth_rate


        if block_idx < num_blocks - 1:
            num_features = num_features // 2

    return num_features
