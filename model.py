import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x 的维度是 (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    if dropout:
        attn = dropout(attn)
    return attn @ value, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        # query, key, value 的维度: (batch_size, seq_len, d_model)
        query = self.linear_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        # q, k, v 变换后的维度: (batch_size, h, seq_len, d_k)
        if mask is not None:
            # mask 的输入维度可能是:
            # - (batch_size, 1, seq_len) 用于 padding mask
            # - (1, seq_len, seq_len) 用于 subsequent mask
            # - (batch_size, seq_len, seq_len) 用于组合 mask
            if mask.dim() == 3 and mask.size(1) == 1:
                # padding mask: (batch_size, 1, seq_len) -> (batch_size, 1, 1, seq_len)
                mask = mask.unsqueeze(1)
            elif mask.dim() == 3 and mask.size(0) == 1:
                # subsequent mask: (1, seq_len, seq_len) -> (1, 1, seq_len, seq_len)
                mask = mask.unsqueeze(1)
            elif mask.dim() == 3:
                # combined mask: (batch_size, seq_len, seq_len) -> (batch_size, 1, seq_len, seq_len)
                mask = mask.unsqueeze(1)
            # 掩码维度经过处理后可以正确广播到 (batch_size, h, seq_len, seq_len)
        x, _ = attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linear_out(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class AddNorm(nn.Module):
    def __init__(self, size, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        # 注意这里的顺序是 "pre-norm"，先 norm 再进子层，但返回的是 x + sublayer(norm(x))
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = nn.ModuleList([
            AddNorm(d_model, dropout),
            AddNorm(d_model, dropout)
        ])
    def forward(self, x, mask):
        # self-attention 子层
        x = self.sublayers[0](x, lambda y: self.self_attn(y, y, y, mask))
        # feed-forward 子层
        x = self.sublayers[1](x, self.feed_forward)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, cross_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayers = nn.ModuleList([
            AddNorm(d_model, dropout),
            AddNorm(d_model, dropout),
            AddNorm(d_model, dropout)
        ])
    def forward(self, x, memory, src_mask, tag_mask):
        # Masked self-attention 子层
        x = self.sublayers[0](x, lambda y: self.self_attn(y, y, y, tag_mask))
        # Cross-attention 子层 (Query from decoder, Key/Value from encoder)
        x = self.sublayers[1](x, lambda y: self.cross_attn(y, memory, memory, src_mask))
        # Feed-forward 子层
        x = self.sublayers[2](x, self.feed_forward)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab, tag_vocab, d_model=512, N=6, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.src_embed = nn.Sequential(
            Embeddings(src_vocab, d_model),
            PositionalEncoding(d_model)
        )
        self.tag_embed = nn.Sequential(
            Embeddings(tag_vocab, d_model),
            PositionalEncoding(d_model)
        )
        attn = lambda: MultiHeadAttention(h, d_model, dropout)
        ff = lambda: FeedForward(d_model, d_ff, dropout)
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, attn(), ff(), dropout) for _ in range(N)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, attn(), attn(), ff(), dropout) for _ in range(N)
        ])
        self.out = nn.Linear(d_model, tag_vocab)
    def encode(self, src, src_mask):
        """编码器方法"""
        x = self.src_embed(src)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x
    def decode(self, tag, memory, src_mask, tag_mask):
        """解码器方法"""
        x = self.tag_embed(tag)
        for layer in self.decoder:
            x = layer(x, memory, src_mask, tag_mask)
        return x
    def forward(self, src, tag, src_mask=None, tag_mask=None):
        memory = self.encode(src, src_mask)
        out = self.decode(tag, memory, src_mask, tag_mask)
        return self.out(out)
