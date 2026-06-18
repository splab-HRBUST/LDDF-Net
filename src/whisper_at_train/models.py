import numpy as np
import torch
import math
import torch.nn.functional as F
from torch import nn

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class SENet(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SENet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.compress = nn.Sequential(
            nn.Conv2d(channel, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.expand = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x)
        x = self.compress(x)
        x = self.expand(x)
        return x


class TimeAttention(nn.Module):

    def __init__(self, time_steps, reduction=16):
        super(TimeAttention, self).__init__()
        self.time_steps = time_steps
        self.reduction = reduction

        # 1D全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # 通过全连接层学习时间注意力
        self.fc = nn.Sequential(
            nn.Linear(time_steps, time_steps // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(time_steps // reduction, time_steps, bias=False),
            nn.Sigmoid()  # 输出注意力系数
        )

    def forward(self, x):
        b, c, t, d = x.size()  # [batch, channels, time_steps, feature_dim]
        # print(f"x的形状: {x.shape}")  # 打印 x 的形状

        # 在时间维度上进行池化
        x_avg = self.avg_pool(x.view(b * c, t, d)).view(b, c, -1)  # [batch, channels, 1]

        # 通过全连接层计算时间注意力
        time_attention_weights = self.fc(x_avg.squeeze(-1))  # [batch, channels, time_steps]

        # 调整时间注意力权重的维度，以便能够广播到 x
        # time_attention_weights 需要变成 [batch, channels, time_steps, 1] 以便与 x 广播
        time_attention_weights = time_attention_weights.unsqueeze(3)  # [batch, channels, time_steps, 1]

        output = x * time_attention_weights

        return output

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_kernel_size=3, stride=1, relu=True,
                 only_cheap=False):
        super(GhostModule, self).__init__()
        init_channels = int(out_channels / ratio)
        new_channels = out_channels - init_channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_kernel_size, 1, dw_kernel_size // 2, groups=init_channels,
                      bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

        self.only_cheap = only_cheap

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        if self.only_cheap:
            return x2
        else:
            x1_split = torch.chunk(x1, chunks=x1.shape[1], dim=1)
            x2_split = torch.chunk(x2, chunks=x2.shape[1], dim=1)
            min_len = min(len(x1_split), len(x2_split))
            concat_list = [torch.cat([x1_split[i], x2_split[-(i + 1)]], dim=1) for i in range(min_len)]
            return torch.cat(concat_list, dim=1)

class TLTR(nn.Module):
    def __init__(self, label_dim=527, n_layer=32, rep_dim=1280, time_steps=100, mode='ghost', reduction=8, drop=0.):
        super().__init__()
        self.mode = mode
        self.n_layer = n_layer
        self.rep_dim = rep_dim
        self.label_dim = label_dim

        if mode == 'ghost':
            self.ghost_layer1 = GhostModule(in_channels=n_layer, out_channels=16, only_cheap=True)
            # 定义通道注意力模块
            self.se_branch = SENet(channel=n_layer)
            # 定义时间注意力模块
            self.time_attention = TimeAttention(time_steps, reduction=reduction)
            # 先归一化两个分支
            self.norm1 = nn.BatchNorm2d(8)  # 对 GhostModule 的输出使用 BatchNorm
            self.norm2 = nn.BatchNorm2d(8)  # 对 SENet 的输出使用 BatchNorm

            # 固定的加权系数 alpha_ghost_se 仍然为固定权重
            self.alpha_ghost_se = 0.6999  # 固定权重，用于 GhostModule 和 SENet 的加权融合

            self.norm_out = nn.BatchNorm2d(8)

            self.ghost_layer2 = GhostModule(in_channels=rep_dim, out_channels=rep_dim)

            # 新增的线性层用于对时间注意力模块输出的通道进行降维
            self.time_attention_fc = nn.Conv2d(n_layer, 8, kernel_size=1)  # 将通道数降至8

            # 在时间注意力输出后加入归一化
            self.time_attention_norm = nn.LayerNorm([8, time_steps])  # 使用 LayerNorm 对最后两个维度进行归一化

            self.mlp_layer = nn.Sequential(
                nn.LayerNorm(rep_dim),
                nn.Linear(rep_dim, self.label_dim)
            )

            # 为时间和通道的加权部分引入可训练的自适应参数
            self.alpha_time = nn.Parameter(torch.tensor(0.5999))  # 自适应权重，用于时间注意力模块输出与加权后的融合

    def forward(self, audio_rep):
        # 应用时间注意力
        x = self.time_attention(audio_rep)

        # 对时间注意力输出的通道进行降维处理
        x_time_attention = self.time_attention_fc(x)

        # 在拼接之前对时间注意力输出进行归一化
        x_time_attention = self.time_attention_norm(x_time_attention.permute(0, 3, 1, 2))  # 调整维度顺序，便于 LayerNorm

        # 对GhostLayer1和SE进行处理
        x_ghost = self.ghost_layer1(audio_rep)
        x_se = self.se_branch(audio_rep)

        # 先归一化再加权
        x_ghost = self.norm1(x_ghost)
        x_se = self.norm2(x_se)

        # 使用固定加权系数 alpha_ghost_se 加权融合 GhostModule 和 SENet 输出
        x_ghost_se = self.alpha_ghost_se * x_ghost + (1 - self.alpha_ghost_se) * x_se

        # 使用自适应加权系数 alpha_time 加权融合 GhostModule + SENet 输出和时间注意力模块输出
        x = self.alpha_time * x_ghost_se + (1 - self.alpha_time) * x_time_attention.permute(0, 2, 3, 1)
        x = self.norm_out(x)

        # 对加权后的输出进行处理
        x = x.permute(0, 3, 2, 1)  # [B,D,C,T]
        x = self.ghost_layer2(x)
        x = x.permute(0, 3, 2, 1)
        x = torch.mean(x, dim=1)
        x = torch.mean(x, dim=1)
        x = self.mlp_layer(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn([2, 32, 25, 1280]).to(device)
    m = TLTR(mode='ghost')
    m.to(device)
    output = m(x)
    print(output.shape)