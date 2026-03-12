import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from typing import List
from torch import Tensor
import copy
from mmcv.cnn import build_norm_layer
from math import log

from mmdet.registry import MODELS
from mmengine.model import BaseModule

class Conv_Extra(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channel, 64, 1),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, channel, 1),
                                   build_norm_layer(norm_layer, channel)[1])
    def forward(self, x):
        return self.block(x)

class Scharr(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(Scharr, self).__init__()
        scharr_x = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        scharr_y = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.conv_x = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_y = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_x.weight.data = scharr_x.repeat(channel, 1, 1, 1)
        self.conv_y.weight.data = scharr_y.repeat(channel, 1, 1, 1)
        
        # 🚀 手术 1：解封边缘算子，允许随石头纹理微调
        self.conv_x.weight.requires_grad = True
        self.conv_y.weight.requires_grad = True
        
        self.norm = build_norm_layer(norm_layer, channel)[1]
        self.act = act_layer()
        self.conv_extra = Conv_Extra(channel, norm_layer, act_layer)

    def forward(self, x):
        edges_x = self.conv_x(x)
        edges_y = self.conv_y(x)
        scharr_edge = torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-6)
        scharr_edge = self.act(self.norm(scharr_edge))
        out = self.conv_extra(x + scharr_edge)
        return out

class Gaussian(nn.Module):
    def __init__(self, dim, size, sigma, norm_layer, act_layer, feature_extra=True):
        super().__init__()
        self.feature_extra = feature_extra
        gaussian = self.gaussian_kernel(size, sigma)
        
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)
        
        # 🚀 手术 1：解封高斯平滑算子
        self.gaussian.weight.requires_grad = True
        
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.act = act_layer()
        if feature_extra == True:
            self.conv_extra = Conv_Extra(dim, norm_layer, act_layer)

    def forward(self, x):
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))
        if self.feature_extra == True:
            out = self.conv_extra(x + gaussian)
        else:
            out = gaussian
        return out
    
    def gaussian_kernel(self, size: int, sigma: float):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
             for y in range(-size // 2 + 1, size // 2 + 1)
             ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()

class DRFD(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.dim = dim
        self.outdim = dim * 2
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv_c = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=2, padding=1, groups=dim * 2)
        self.act_c = act_layer()
        self.norm_c = build_norm_layer(norm_layer, dim * 2)[1]
        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm_m = build_norm_layer(norm_layer, dim * 2)[1]
        self.fusion = nn.Conv2d(dim * 4, self.outdim, kernel_size=1, stride=1)
        self.gaussian = Gaussian(self.outdim, 5, 0.5, norm_layer, act_layer, feature_extra=False)
        self.norm_g = build_norm_layer(norm_layer, self.outdim)[1]

    def forward(self, x):
        x = self.conv(x)
        gaussian = self.gaussian(x)
        x = self.norm_g(x + gaussian)
        max_feat = self.norm_m(self.max_m(x))
        conv_feat = self.norm_c(self.act_c(self.conv_c(x)))
        x = torch.cat([conv_feat, max_feat], dim=1)
        x = self.fusion(x)
        return x

class LFEA(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(LFEA, self).__init__()
        self.channel = channel
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, dilation=1, bias=False),
            build_norm_layer(norm_layer, channel)[1],
            act_layer())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = build_norm_layer(norm_layer, channel)[1]

    def forward(self, c, att):
        att = c * att + c
        att = self.conv2d(att)
        wei = self.avg_pool(att)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = self.norm(c + att * wei)
        return x

class LFE_Module(nn.Module):
    def __init__(self, dim, stage, mlp_ratio, drop_path, act_layer, norm_layer):
        super().__init__()
        self.stage = stage
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_layer = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            build_norm_layer(norm_layer, mlp_hidden_dim)[1],
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)]
        self.mlp = nn.Sequential(*mlp_layer)
        self.LFEA = LFEA(dim, norm_layer, act_layer)
        if stage == 0:
            self.Scharr_edge = Scharr(dim, norm_layer, act_layer)
        else:
            self.gaussian = Gaussian(dim, 5, 1.0, norm_layer, act_layer)
        self.norm = build_norm_layer(norm_layer, dim)[1]

    def forward(self, x: Tensor) -> Tensor:
        if self.stage == 0:
            att = self.Scharr_edge(x)
        else:
            att = self.gaussian(x)
        x_att = self.LFEA(x, att)
        x = x + self.norm(self.drop_path(self.mlp(x_att)))
        return x

class BasicStage(nn.Module):
    def __init__(self, dim, stage, depth, mlp_ratio, drop_path, norm_layer, act_layer):
        super().__init__()
        blocks_list = [
            LFE_Module(dim=dim, stage=stage, mlp_ratio=mlp_ratio, drop_path=drop_path[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ]
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)

class LoGFilter(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, sigma, norm_layer, act_layer):
        super(LoGFilter, self).__init__()
        self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3)
        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax)
        kernel = (xx**2 + yy**2 - 2 * sigma**2) / (2 * math.pi * sigma**4) * torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel - kernel.mean()
        kernel = kernel / (kernel.abs().sum() + 1e-8)
        log_kernel = kernel.unsqueeze(0).unsqueeze(0)
        
        self.LoG = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2), groups=out_c, bias=False)
        self.LoG.weight.data = log_kernel.repeat(out_c, 1, 1, 1)
        
        # 🚀 手术 1：解封 LoG 算子
        self.LoG.weight.requires_grad = True
        
        self.act = act_layer()
        self.norm1 = build_norm_layer(norm_layer, out_c)[1]
        self.norm2 = build_norm_layer(norm_layer, out_c)[1]
    
    def forward(self, x):
        x = self.conv_init(x)
        LoG = self.LoG(x)
        LoG_edge = self.act(self.norm1(LoG))
        x = self.norm2(x + LoG_edge)
        return x

class Stem(nn.Module):
    def __init__(self, in_chans, stem_dim, act_layer, norm_layer):
        super().__init__()
        out_c14 = int(stem_dim / 4)
        out_c12 = int(stem_dim / 2)
        self.Conv_D = nn.Sequential(
            nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14),
            nn.Conv2d(out_c12, out_c12, kernel_size=3, stride=2, padding=1, groups=out_c12),
            build_norm_layer(norm_layer, out_c12)[1])
        self.LoG = LoGFilter(in_chans, out_c14, 7, 1.0, norm_layer, act_layer)
        self.gaussian = Gaussian(out_c12, 9, 0.5, norm_layer, act_layer)
        self.norm = build_norm_layer(norm_layer, out_c12)[1]
        self.drfd = DRFD(out_c12, norm_layer, act_layer)

    def forward(self, x):
        x = self.LoG(x)
        x = self.Conv_D(x)
        x = self.norm(x + self.gaussian(x))
        x = self.drfd(x)
        return x

@MODELS.register_module()
class LWEGNet(BaseModule):
    def __init__(self,
                 in_chans=3,
                 stem_dim=32,
                 depths=(1, 4, 4, 2),
                 norm_layer=dict(type='BN', requires_grad=True),
                 act_layer=nn.ReLU,
                 mlp_ratio=2.,
                 drop_path_rate=0.1,
                 fork_feat=True,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)

        self.num_stages = len(depths)
        self.Stem = Stem(in_chans=in_chans, stem_dim=stem_dim, act_layer=act_layer, norm_layer=norm_layer)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        stages_list = []
        for i_stage in range(self.num_stages):
            stage = BasicStage(dim=int(stem_dim * 2 ** i_stage), stage=i_stage, depth=depths[i_stage],
                               mlp_ratio=mlp_ratio, drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                               norm_layer=norm_layer, act_layer=act_layer)
            stages_list.append(stage)
            if i_stage < self.num_stages - 1:
                stages_list.append(DRFD(dim=int(stem_dim * 2 ** i_stage), norm_layer=norm_layer, act_layer=act_layer))
        self.stages = nn.Sequential(*stages_list)

        self.fork_feat = fork_feat
        self.out_indices = [0, 2, 4, 6]
        
        # 🚀 手术 2：为每个输出分支挂载升维补偿器 (1x1 Conv)
        self.compensators = nn.ModuleList()
        for i_emb, i_layer in enumerate(self.out_indices):
            layer = build_norm_layer(norm_layer, int(stem_dim * 2 ** i_emb))[1]
            setattr(self, f'norm{i_layer}', layer)
            
            in_c = int(stem_dim * 2 ** i_emb)
            out_c = in_c * 2 # 将通道翻倍: 32->64, 64->128, 128->256, 256->512
            self.compensators.append(
                nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False),
                    build_norm_layer(norm_layer, out_c)[1],
                    act_layer()
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        x = self.Stem(x)
        outs = []
        comp_idx = 0
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                # 🚀 经过升维补偿器放大通道
                x_out = self.compensators[comp_idx](x_out)
                outs.append(x_out)
                comp_idx += 1
        return tuple(outs)
