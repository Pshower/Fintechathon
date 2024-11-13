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
from einops import rearrange, reduce, repeat
import warnings
warnings.filterwarnings('ignore')

class RawFormer(nn.Module):
    # 主干网络
    def __init__(self, d_args, device):
        super(RawFormer, self).__init__()
        # win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
        #  dropout=0.0, activation='gelu'
        self.device=device
        # # Encoding
        # self.embedding = DataEmbedding(enc_in, d_model, dropout)
        self.win_size = d_args['win_size']
        self.enc_in = d_args['enc_in']
        self.c_out = d_args["c_out"]
        self.n_heads = d_args['n_heads']
        self.e_layers = d_args['e_layers']
        self.d_ff = d_args['d_ff']
        self.d_model = d_args['d_model']
        self.activation = 'gelu'
        self.dropout = 0.0
        # embedding
        self.Sinc_conv = SincConv(device=self.device,
			out_channels = d_args['filts'],
			kernel_size = d_args['first_conv'],
                        in_channels = d_args['in_channels'],freq_scale='Mel'
        )
        self.first_bn = nn.BatchNorm1d(num_features = d_args['filts'])
        self.selu = nn.SELU(inplace=True)
        self.embedding = LinearEmbedding(in_size=21490, out_size=self.d_model, d_model=self.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        RawAttention(False, attention_dropout=self.dropout),
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

    def forward(self, x, y=None, is_test=False):
        # enc_out = self.embedding(x)
        # [32, 64000]
        batches, length = x.shape 
        x = x.view(batches, 1, length)
        x = self.Sinc_conv(x)
        # print("x shape after Sinc_conv:", x.shape)
        # [32, 512, 64472]
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.first_bn(x)
        x = self.selu(x)
        # print("x shape after selu:", x.shape)
        # [32, 512, 21490]
        x = self.embedding(x)
        # print("x shape after embedding:", x.shape)
        # [32, 512, 512]

        x = self.encoder(x)
        # print("encoder res shape:",x.shape)
        # [32, 512, 512]
        x =  torch.mean(x, dim=1)
        # print(x.shape)
        y = self.projection(x)
        # print("output shape:", y.shape)
        return y  # [B, 2]


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
        new_x = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        # 前向过程
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)

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

    def forward(self, queries_x, keys_x, values_x, attn_mask):
        B, L, _ = queries_x.shape
        _, S, _ = keys_x.shape
        H = self.n_heads
        # print("input shape:", queries_x.shape)
        # print("self.query_projection:", self.query_projection)
        # print("heads:", self.n_heads)
        
        
        queries = self.query_projection(queries_x).view(B, L, H, -1)
        keys = self.key_projection(keys_x).view(B, S, H, -1)
        values = self.value_projection(values_x).view(B, S, H, -1)
        # print("Q, K, V shape:", queries.shape, keys.shape, values.shape)
        
        out  = self.inner_attention(
            queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out)

class RawAttention(nn.Module):
    # 注意力计算底层
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.0):
        super(RawAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

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


class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)


    def __init__(self, device,out_channels, kernel_size,in_channels=1,sample_rate=16000,
                 stride=1, padding=0, dilation=1, bias=False, groups=1,freq_scale='Mel'):

        super(SincConv,self).__init__()


        if in_channels != 1:
            
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)
        
        self.out_channels = out_channels+1
        self.kernel_size = kernel_size
        self.sample_rate=sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1

        self.device=device   
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')
        
        
        # initialize filterbanks using Mel scale
        NFFT = 512
        f=int(self.sample_rate/2)*np.linspace(0,1,int(NFFT/2)+1)


        if freq_scale == 'Mel':
            fmel=self.to_mel(f) # Hz to mel conversion
            fmelmax=np.max(fmel)
            fmelmin=np.min(fmel)
            filbandwidthsmel=np.linspace(fmelmin,fmelmax,self.out_channels+2)
            filbandwidthsf=self.to_hz(filbandwidthsmel) # Mel to Hz conversion
            self.freq=filbandwidthsf[:self.out_channels]

        elif freq_scale == 'Inverse-mel':
            fmel=self.to_mel(f) # Hz to mel conversion
            fmelmax=np.max(fmel)
            fmelmin=np.min(fmel)
            filbandwidthsmel=np.linspace(fmelmin,fmelmax,self.out_channels+2)
            filbandwidthsf=self.to_hz(filbandwidthsmel) # Mel to Hz conversion
            self.mel=filbandwidthsf[:self.out_channels]
            self.freq=np.abs(np.flip(self.mel)-1) ## invert mel scale

        
        else:
            fmelmax=np.max(f)
            fmelmin=np.min(f)
            filbandwidthsmel=np.linspace(fmelmin,fmelmax,self.out_channels+2)
            self.freq=filbandwidthsmel[:self.out_channels]
        
        self.hsupp=torch.arange(-(self.kernel_size-1)/2, (self.kernel_size-1)/2+1)
        self.band_pass=torch.zeros(self.out_channels-1,self.kernel_size)
    
       
        
    def forward(self,x):
        for i in range(len(self.freq)-1):
            fmin=self.freq[i]
            fmax=self.freq[i+1]
            # print("fmin, fmax:",fmin, fmax) # 0.0 13.701590719550216
            # print("self.sample_rate:",self.sample_rate)
            # print("self.hsupp:",self.hsupp)
            # a = 2*fmax*self.hsupp/self.sample_rate
            # print("2*fmax*self.hsupp/self.sample_rate:",a)
            # c= np.array([1,2,3,4])
            # print(c)
            # d = torch.Tensor([1,2,3,4])
            # print(d)
            # print("np.sinc(d.numpy()):",np.sinc(d.numpy()))
            # print(np.sinc(d))
            # print("np.sinc(a):", np.sinc(a))
            # print("np.sinc(2*fmax*self.hsupp/self.sample_rate):",np.sinc(2*fmax*self.hsupp.numpy()/self.sample_rate.numpy()))
            # print("2*fmax/self.sample_rate)*np.sinc(2*fmax*self.hsupp/self.sample_rate:",2*fmax/self.sample_rate)*np.sinc(2*fmax*self.hsupp.numpy()/self.sample_rate)
            # Numpy is not available?
            # print("np.sinc(2*fmax*self.hsupp/self.sample_rate) is:", np.sinc(2*fmax*self.hsupp/self.sample_rate))
            # 计算数学上的sinc函数，即"sin(x) / x"（当x不为0时），在x等于0时，sinc(0)定义为1
            hHigh=(2*fmax/self.sample_rate)*np.sinc(2*fmax*self.hsupp/self.sample_rate)
            hLow=(2*fmin/self.sample_rate)*np.sinc(2*fmin*self.hsupp/self.sample_rate)
            hideal=hHigh-hLow
            
            self.band_pass[i,:]=Tensor(np.hamming(self.kernel_size))*Tensor(hideal)
        
        band_pass_filter=self.band_pass.to(self.device)

        self.filters = (band_pass_filter).view(self.out_channels-1, 1, self.kernel_size)
        
        return F.conv1d(x, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1)



# class TokenEmbedding(nn.Module):
#     def __init__(self, c_in, d_model):
#         super(TokenEmbedding, self).__init__()
#         padding = 1 if torch.__version__ >= '1.5.0' else 2
#         self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,kernel_size=3, 
#                                    padding=padding, padding_mode='circular', bias=False)
        
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

#     def forward(self, x):
#         x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
#         return x


# class DataEmbedding(nn.Module):
#     def __init__(self, c_in, d_model, dropout=0.0):
#         # 对输入数据进行embedding
#         super(DataEmbedding, self).__init__()
#         self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
#         self.position_embedding = PositionalEmbedding(d_model=d_model)

#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, x):
#         x = self.value_embedding(x) + self.position_embedding(x)
#         return self.dropout(x)


# class PatchEmbedding(nn.Module):
#     def __init__(self, patchsize, d_model, dropout=0.01):
#         # 对输入数据进行patch化，再进行embedding
#         super(PatchEmbedding, self).__init__()
#         self.patchsize = patchsize
#         self.embedding_patch_size = DataEmbedding(self.patchsize, d_model, dropout)
    
#     def forward(self, x):
#         x_patch_size = x
#         # Batch channel win_size
#         print("input shape:", x_patch_size.shape)
#         # [32, 512, 21490]
        
#         x_patch_size = rearrange(
#             x_patch_size, 'b m (n p) -> (b m) n p', p=self.patchsize)
#         print("rearranged shape:", x_patch_size.shape)
#         # [32*512, 2149, 10]
#         x_patch_size = self.embedding_patch_size(x_patch_size)
#         print("shape after embedding:", x_patch_size.shape)
#         # []

#         # series_patch_size = reduce(
#         #     series_patch_size, '(b reduce_b) l m n-> b l m n', 'mean', reduce_b=self.channel)
#         # print("shape after reduce:", shape:)

#         return x_patch_size

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        # if isinstance(max_len, torch.Tensor):
        #     max_len = max_len.item()
        # if isinstance(d_model, torch.Tensor):
        #     d_model = d_model.item()

        pe = torch.zeros(max_len, d_model)
        pe.require_grad = False

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        
        # return self.pe[:, :x.size(1)]
        return self.pe[:, :x.shape[1]]
        
class LinearEmbedding(nn.Module):
    def __init__(self, in_size, out_size, d_model, dropout=0.01):
        # 对输入数据进行patch化，再进行embedding
        super(LinearEmbedding, self).__init__()
        self.Linear_layer = nn.Linear(in_size, out_size)
        self.BN = nn.Sequential(nn.BatchNorm1d(out_size),
                                 nn.ReLU())
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.input_length = in_size
    
    def forward(self, x):
        # 进行长度补齐处理
        if x.shape[-1] < self.input_length:
            padding_size = self.input_length - x.shape[-1]
            x = F.pad(x, (0, padding_size), "constant", 0)
        elif x.shape[-1] > self.input_length:
            x = x[:,:, :self.input_length]


        Linear_embedding = self.BN(self.Linear_layer(x))
        # print("shape of Linear_embedding:", Linear_embedding.shape)
        Pos_embedding = self.position_embedding(Linear_embedding)
        # print("Pos_embedding:", Pos_embedding.shape)
        return Linear_embedding + Pos_embedding


if __name__ == '__main__':
    # 输出模型信息
    import os
    import yaml
    dir_yaml = os.path.splitext('model_config_RawFormer')[0] + '.yaml'
    with open(dir_yaml, 'r') as f_yaml:
        parser1 = yaml.load(f_yaml)
        # print(parser1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    # print(parser1['model'])
    # print(parser1['model']['win_size'])

    test_embedding_input = torch.rand((32, 512, 21490))
    embedding = LinearEmbedding(21490, 512, 512)
    embedding_res = embedding(test_embedding_input)

    test_input = torch.rand((32, 64600)).to(device)
    # print('test_input shape:', test_input.shape)
    # batches, length = test_input.shape
    # x = test_input.view(batches, 1, length)
    # print("x.shape is ",x.shape)
    model_1gpu = RawFormer(parser1['model'], device)
    model_1gpu.to(device)
    # # print(model_1gpu)
    test_res = model_1gpu(test_input)

    
    # embedding = PatchEmbedding(patchsize=10, d_model=512)
    # embedding_res = embedding(test_embedding_input)

    # check = torch.zeros(5000, 500)
    # print(check.shape)



    

