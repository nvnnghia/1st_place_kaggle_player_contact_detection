import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from torch.nn.parameter import Parameter
from resnet3d_csn import ResNet3dCSN

class NModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.backbone = ResNet3dCSN(
            pretrained2d=False,
            in_channels = 3,
            pretrained=None,
            depth=int(cfg.model_name[1:-2]),
            with_pool2=False,
            bottleneck_mode=cfg.model_name[-2:],
            norm_eval=False,
            zero_init_residual=False)

        self.final = nn.Linear(2048+1024, out_features=1)
        if cfg.pool_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1)) #if pool =="avg" else 
        else:
            self.avg_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        bs = x.size(0)

        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1, 1)[:, :, :, :, :]
        x = self.backbone(x)
        # print(x[-2].shape, x[-1].shape)
        # x = x[-1]
        # x = self.avg_pool(x)

        x_fast = self.avg_pool(x[-2])
        x_slow = self.avg_pool(x[-1])

        x = torch.cat((x_slow, x_fast), dim=1)

        x = self.dropout(x)
        x = x.flatten(1)
        x = self.final(x)
        return {
        'out1':x,
        'emb':x
        }