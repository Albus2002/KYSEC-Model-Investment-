import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# 定义网络结构
class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FullyConnectedNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

#带dropout和注意力机制的模型
class MLPWithDropout(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(MLPWithDropout, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.dropout(x)
        x = self.fc5(x)
        return x
    

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        # 注意力得分向量，用于计算每个输入特征的权重
        self.score_vector = nn.Parameter(torch.Tensor(input_dim,1))
        nn.init.kaiming_uniform_(self.score_vector, a=1)  # 使用He初始化

    def forward(self, x):
        # 计算注意力得分
        attention_scores = torch.matmul(x, self.score_vector)
        attention_weights = F.softmax(attention_scores, dim=1)
        # 应用注意力权重
        weighted_features = x * attention_weights.unsqueeze(-1)
        return weighted_features.sum(dim=1)
    
class MLPWithAttention(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLPWithAttention, self).__init__()
        # 自注意力层
        self.attention = SelfAttention(input_size)
        # 定义第一个隐藏层
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        # 定义第二个隐藏层
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        # 定义输出层
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])

        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])

        self.fc5 = nn.Linear(hidden_sizes[3], output_size)

    def forward(self, x):
        # 通过自注意力层
        x = x.unsqueeze(1)
        x = self.attention(x)
        x = x.squeeze(1)
        # 通过第一个隐藏层然后应用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 通过第二个隐藏层然后应用ReLU激活函数
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # 通过输出层
        x = self.fc5(x)
        return x


class TokenMixingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, patch_size):
        super().__init__()
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, input_dim)
        
        self.patch_embedding = nn.Parameter(torch.randn(1, patch_size, input_dim))
        
    def forward(self, x):
        # Token Mixing
        patchEmb = self.patch_embedding.repeat(x.size(0), 1, 1)
        x = torch.cat([x, patchEmb], dim=1)
        x = self.mlp1(x)
        x = F.relu(x)
        x = self.mlp2(x)
        return x

class ChannelMixingLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        # Channel Mixing
        x = x.flatten(start_dim=2)
        x = self.mlp(x)
        x = x.view_as(x)
        return x

class MLPMixer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, depth=4, patch_size=6):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.layers = nn.ModuleList()
        for _ in range(self.depth):
            self.layers.append(TokenMixingLayer(hidden_dim, hidden_dim, patch_size))
            self.layers.append(ChannelMixingLayer(hidden_dim))
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Flatten the input for the input embedding
        x = x.view(x.size(0), -1, self.input_dim)
        
        # Input embedding
        x = self.input_embedding(x)
        
        # MLP-Mixer layers
        for layer in self.layers:
            x = layer(x)
        
        # Global pooling and output
        x = torch.mean(x, dim=1)  # Assuming we want to use the mean of the features
        x = self.output_layer(x)
        
        return x

# BiAGRU模型

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)

        # 初始化注意力权重
        nn.init.xavier_uniform_(self.att_weights.data)

    def forward(self, x):
        # x.shape: (batch_size, seq_len, hidden_size)
        scores = torch.bmm(x, self.att_weights.permute(1, 0).unsqueeze(0).repeat(x.size(0), 1, 1))
        # scores.shape: (batch_size, seq_len, 1)
        scores = self.softmax(scores.squeeze(2))
        # scores.shape: (batch_size, seq_len)
        weighted_output = torch.bmm(x.transpose(1, 2), scores.unsqueeze(2)).squeeze(2)
        # weighted_output.shape: (batch_size, hidden_size)
        return weighted_output, scores

class BiAGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super(BiAGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 定义双向GRU层
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        
        # 定义注意力层
        self.attention = Attention(hidden_size * 2)  # 因为是双向，所以尺寸是hidden_size的两倍
        
        # 定义输出层
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 同样是因为双向

    def forward(self, x):
        # x.shape: (batch_size, seq_len, input_size)
        gru_out, _ = self.gru(x)
        # gru_out.shape: (batch_size, seq_len, hidden_size * 2)
        attn_out, attn_scores = self.attention(gru_out)
        # attn_out.shape: (batch_size, hidden_size * 2)
        output = self.fc(attn_out)
        # output.shape: (batch_size, output_size)
        return output

import math
import numpy as np
from math import sqrt
#transformer-base
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dytpe=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

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

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(MultiHeadAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

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

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn



class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        a = self.value_embedding(x)
        b = self.position_embedding(x)
        x = a+b

        return self.dropout(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, dropout, activation):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=ff_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        x = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        x = self.dropout(self.conv2(x).transpose(-1, 1))
        return x

class MultiheadFeedForward(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout, activation):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.mhfw = nn.ModuleList([FeedForward(d_model=self.head_dim, ff_dim=ff_dim, dropout=dropout, activation=activation) for i in range(self.n_heads)])

    def forward(self, x): # [bs, seq_len, d_model]
        bs = x.shape[0]
        input = x.reshape(bs, -1, self.n_heads, self.head_dim) # [bs, seq_len, n_heads, head_dim]
        outputs = []
        for i in range(self.n_heads):
            outputs.append(self.mhfw[i](input[:, :, i, :])) # [bs, seq_len, head_dim]
        outputs = torch.stack(outputs, dim=-2).reshape(bs, -1, self.d_model) # [bs, seq_len, n_heads, head_dim]
        return outputs


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, n_heads=8, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.mhfw = MultiheadFeedForward(d_model=d_model, n_heads=n_heads, ff_dim=d_ff, dropout=dropout, activation=activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.mhfw(y)

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff, n_heads=8,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.mhfw = MultiheadFeedForward(d_model=d_model, n_heads=n_heads, ff_dim=d_ff, dropout=dropout, activation=activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.mhfw(y)

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x

class Transformer_base(nn.Module):
    def __init__(self, enc_in, dec_in, c_out = 1,
                d_model=128, n_heads=4, e_layers=2, d_layers=1, d_ff=256, 
                dropout=0.1, activation='gelu', output_attention=False):
        super(Transformer_base, self).__init__()

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    n_heads,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    n_heads,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        
        self.projection_decoder = nn.Linear(d_model, c_out, bias=True)
    

        
    def forward(self, x_enc, x_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc)
        dec_out = self.dec_embedding(x_dec)

        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        
        
        output = self.projection_decoder(dec_out)


        return enc_out, dec_out, output

