import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from model.pvt_v2 import pvt_v2_b2
from torch.nn.parameter import Parameter
import math
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from timm.models import create_model
from mmcv.cnn import build_norm_layer


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.PReLU()
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class PatchwiseTokenReEmbedding:
    @staticmethod
    def encode(x, nh, ph, pw):
        return rearrange(x, "b (nh hd) (nhp ph) (nwp pw) -> b nh (hd ph pw) (nhp nwp)", nh=nh, ph=ph, pw=pw)

    @staticmethod
    def decode(x, nhp, ph, pw):
        return rearrange(x, "b nh (hd ph pw) (nhp nwp) -> b (nh hd) (nhp ph) (nwp pw)", nhp=nhp, ph=ph, pw=pw)


class SpatialViewAttn(nn.Module):
    def __init__(self, dim, p, nh=2):
        super().__init__()
        self.p = p
        self.nh = nh
        self.scale = (dim // nh * self.p ** 2) ** -0.5

        self.to_q = nn.Conv2d(dim, dim, 1)
        self.to_kv = nn.Conv2d(dim, dim * 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, q, kv=None, need_weights: bool = False):
        if kv is None:
            kv = q
        N, C, H, W = q.shape

        q = self.to_q(q)
        k, v = torch.chunk(self.to_kv(kv), 2, dim=1)

        # multi-head patch-wise token re-embedding (PTRE)
        q = PatchwiseTokenReEmbedding.encode(q, nh=self.nh, ph=self.p, pw=self.p)
        k = PatchwiseTokenReEmbedding.encode(k, nh=self.nh, ph=self.p, pw=self.p)
        v = PatchwiseTokenReEmbedding.encode(v, nh=self.nh, ph=self.p, pw=self.p)

        qk = torch.einsum("bndx, bndy -> bnxy", q, k) * self.scale
        qk = qk.softmax(-1)
        qkv = torch.einsum("bnxy, bndy -> bndx", qk, v)

        qkv = PatchwiseTokenReEmbedding.decode(qkv, nhp=H // self.p, ph=self.p, pw=self.p)

        x = self.proj(qkv)
        if not need_weights:
            return x
        else:
            # average attention weights over heads
            return x, qk.mean(dim=1)


class ChannelViewAttn(nn.Module):
    def __init__(self, dim, nh):
        super().__init__()
        self.nh = nh
        self.to_q = nn.Conv2d(dim, dim, 1)
        self.to_kv = nn.Conv2d(dim, dim * 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, q, kv=None):
        if kv is None:
            kv = q
        B, C, H, W = q.shape

        q = self.to_q(q)
        k, v = torch.chunk(self.to_kv(kv), 2, dim=1)
        q = q.reshape(B, self.nh, C // self.nh, H * W)
        k = k.reshape(B, self.nh, C // self.nh, H * W)
        v = v.reshape(B, self.nh, C // self.nh, H * W)

        q = q * (q.shape[-1] ** (-0.5))
        qk = q @ k.transpose(-2, -1)
        qk = qk.softmax(dim=-1)
        qkv = qk @ v

        qkv = qkv.reshape(B, C, H, W)
        x = self.proj(qkv)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, p, nh=4, ffn_expand=1):
        super().__init__()
        self.rgb_norm2 = nn.BatchNorm2d(dim // 2)
        self.depth_norm2 = nn.BatchNorm2d(dim // 2)

        self.depth_to_rgb_sa = SpatialViewAttn(dim // 2, p=p, nh=nh)
        self.depth_to_rgb_ca = ChannelViewAttn(dim // 2, nh=nh)
        self.rgb_alpha = nn.Parameter(data=torch.zeros(1))
        self.rgb_beta = nn.Parameter(data=torch.zeros(1))

        self.rgb_to_depth_sa = SpatialViewAttn(dim // 2, p=p, nh=nh)
        self.rgb_to_depth_ca = ChannelViewAttn(dim // 2, nh=nh)
        self.depth_alpha = nn.Parameter(data=torch.zeros(1))
        self.depth_beta = nn.Parameter(data=torch.zeros(1))
        self.conv1 = Conv(dim, dim // 2, kernel_size=1)
        self.conv2 = Conv(dim, dim // 2, kernel_size=1)

    def forward(self, rgb, depth):
        rgb = self.conv1(rgb)
        depth = self.conv2(depth)
        normed_rgb = self.rgb_norm2(rgb)
        normed_depth = self.depth_norm2(depth)
        transd_rgb = self.rgb_alpha.sigmoid() * self.depth_to_rgb_sa(
            normed_rgb, normed_depth
        ) + self.rgb_beta.sigmoid() * self.depth_to_rgb_ca(normed_rgb, normed_depth)
        rgb_rgbd = rgb + transd_rgb
        transd_depth = self.depth_alpha.sigmoid() * self.rgb_to_depth_sa(
            normed_depth, normed_rgb
        ) + self.depth_beta.sigmoid() * self.rgb_to_depth_ca(normed_depth, normed_rgb)
        depth_rgbd = depth + transd_depth

        rgbd = torch.cat([rgb_rgbd, depth_rgbd], dim=1)
        return rgbd


class last_conv(nn.Module):
    def __init__(self, out_dim):
        super(last_conv, self).__init__()

        self.proj_1 = ConvBNReLU(64, out_dim, kernel_size=3, stride=1)
        self.proj_2 = ConvBNReLU(128, out_dim, kernel_size=3, stride=1)
        self.proj_3 = ConvBNReLU(320, out_dim, kernel_size=3, stride=1)
        self.proj_4 = ConvBNReLU(512, out_dim, kernel_size=3, stride=1)

    def forward(self, x4, x3, x2, x1):
        x4 = self.proj_4(x4)
        x3 = self.proj_3(x3)
        x2 = self.proj_2(x2)
        x1 = self.proj_1(x1)

        return x4, x3, x2, x1


class HFF_high(nn.Module):
    def __init__(self, channel):
        super(HFF_high, self).__init__()
        self.relu = nn.PReLU()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2 * channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(2 * channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(2 * channel, channel, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(channel)
        self.bn2 = nn.BatchNorm2d(channel)
        self.bn3 = nn.BatchNorm2d(channel)
        self.bn4 = nn.BatchNorm2d(channel)

        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x1, x2, x3, x4):
        x1 = self.relu(self.bn1(self.conv1(self.upsample(x1))))
        x1 = self.maxpool(self.upsample(x1))

        x2 = torch.cat((x1, x2), 1)
        x2 = self.relu(self.bn2(self.conv2(x2)))
        x2 = x2.mul(x1)
        x2 = self.upsample(self.maxpool(self.upsample(x2)))

        x3 = torch.cat((x2, x3), 1)
        x3 = self.relu(self.bn3(self.conv3(x3)))
        x3 = x3.mul(x2)
        x3 = self.upsample(self.maxpool(self.upsample(x3)))

        x4 = torch.cat((x3, x4), 1)
        x4 = self.relu(self.bn4(self.conv4(x4)))
        x4 = x4.mul(x3)
        x4 = self.upsample(self.maxpool(self.upsample(x4)))

        return x1, x2, x3, x4


class SPMCNet_VGG(nn.Module):
    def __init__(self, channel=32):
        super(SPMCNet_VGG, self).__init__()
        self.backbone_rgb = pvt_v2_b2()
        self.backbone_depth = pvt_v2_b2()
        self.backbone_rgb.load_state_dict(torch.load('./model/pvt_v2_b2.pth'), strict=False)
        self.backbone_depth.load_state_dict(torch.load('./model/pvt_v2_b2.pth'), strict=False)

        # 解码
        self.hff = HFF_high(channel=64)
        self.hff_d = HFF_high(channel=64)

        self.conv_pre_f4 = Conv(64, 1, 1, stride=1)
        self.conv_pre_f3 = Conv(64, 1, 1, stride=1)
        self.conv_pre_f2 = Conv(64, 1, 1, stride=1)
        self.conv_pre_f1 = Conv(64, 1, 1, stride=1)
        self.conv_pre_d = Conv(64, 1, 1, stride=1)
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.last_c = last_conv(out_dim=64)
        self.last_d = last_conv(out_dim=64)

        # 融合
        self.ca1 = CrossAttention(64, p=11)
        self.ca2 = CrossAttention(128, p=11)
        self.ca3 = CrossAttention(320, p=11)
        self.ca4 = CrossAttention(512, p=11)

        self.conv4 = ConvBNReLU(320, 128)
        self.conv3 = ConvBNReLU(128, 64)
        self.conv2 = ConvBNReLU(64, 32)
        self.conv1 = ConvBNReLU(32, 1)

    def forward(self, x_rgb, x_t):
        F1, F2, F3, F4 = self.backbone_rgb(x_rgb)
        F1_d, F2_d, F3_d, F4_d = self.backbone_depth(x_t)

        fusion1 = self.ca1(F1, F1_d)
        fusion2 = self.ca2(F2, F2_d)
        fusion3 = self.ca3(F3, F3_d)
        fusion4 = self.ca4(F4, F4_d)

        # 统一通道
        F4_Sea, F3_Sea, F2_Sea, F1_Sea = self.last_c(fusion4, fusion3, fusion2, fusion1)

        F4_d_Sea, F3_d_Sea, F2_d_Sea, F1_d_Sea = self.last_d(F4_d, F3_d, F2_d, F1_d)

        # 解码
        y4, y3, y2, y1 = self.hff(F4_Sea, F3_Sea, F2_Sea, F1_Sea)
        y4 = self.conv_pre_f4(y4)
        y3 = self.conv_pre_f3(y3)
        y2 = self.conv_pre_f2(y2)
        y1 = self.conv_pre_f1(y1)

        d4, d3, d2, d1 = self.hff_d(F4_d_Sea, F3_d_Sea, F2_d_Sea, F1_d_Sea)
        d1 = self.conv_pre_d(d1)

        return self.upsample2(y1), self.upsample4(y2), self.upsample8(y3), self.upsample16(y4), self.upsample2(d1)
