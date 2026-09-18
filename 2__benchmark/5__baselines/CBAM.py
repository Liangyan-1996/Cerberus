import torch
import torch.nn as nn
from parms_settings import settings
import torch.nn.functional as F

args = settings()
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=args.reduction, spatial_kernel=args.spatial_kernel):
        super(CBAMLayer, self).__init__()


        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # shared MLP
        self.mlp = nn.Sequential(

            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv1d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv1d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.weight_channel = nn.Parameter(torch.randn(1))
        self.weight_spatial = nn.Parameter(torch.randn(1))

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x1 = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x2 = spatial_out * x

        x = torch.sigmoid(self.weight_channel) * x1 + torch.sigmoid(self.weight_spatial) * x2

        return x

