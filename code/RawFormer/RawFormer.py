import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import math
from torch.utils import data
from collections import OrderedDict
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import pickle
import random
from math import sqrt
import os

class RawFormer(nn.Module):
    # 主干网络
    def __init__(self, d_args, device):
        super(RawFormer, self).__init__()
        # win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
        #  dropout=0.0, activation='gelu'
        self.device=device
        # # Encoding
        # self.embedding = DataEmbedding(enc_in, d_model, dropout)
        self.win_size = d_args.win_size
        self.enc_in = d_args.enc_in
        self.c_out = d_args.c_out
        self.n_heads = d_args.n_heads
        self.e_layers = d_args.e_layers
        self.d_ff = d_args.d_ff
        self.d_model = d_args.d_model
        self.activation = 'gelu'
        self.dropout = 0.0
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        RawAttention(self.win_size, False, attention_dropout=self.dropout),
                                  self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        self.projection = nn.Linear(self.d_model, self.c_out, bias=True)

    def forward(self):
        # enc_out = self.embedding(x)
        enc_out = self.encoder(enc_out)
        enc_out = self.projection(enc_out)
        return enc_out  # [B, L, D]



class Encoder(nn.Module):
    # 编码器模块
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        for attn_layer in self.attn_layers:
            x  = attn_layer(x, attn_mask=attn_mask)
            
        if self.norm is not None:
            x = self.norm(x)

        return x


class EncoderLayer(nn.Module):
    # 注意力计算层
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        # 前向过程
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn, mask

class AttentionLayer(nn.Module):
    # 注意力计算过程
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        out, series, prior = self.inner_attention(
            queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out)



class RawAttention(nn.Module):
    # 注意力计算最底层
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0):
        super(RawAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size

        self.distances = torch.zeros((window_size, window_size)).cuda()
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)

    def forward(self, queries, keys, values, attn_mask):
        # 输入: qkv矩阵; 输出:V
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        # scale =1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            # default True
            if attn_mask is None:
                # default None
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores
        attn = self.dropout(torch.softmax(attn, dim=-1))

        V = torch.einsum("bhls,bshd->blhd", attn, values)
        # 输出V
        return V.contiguous()


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(
                mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask
