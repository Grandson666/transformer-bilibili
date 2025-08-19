import torch
import torch.nn as nn

def create_padding_mask(seq, pad_token):
    # 创建填充掩码
    # 输入 seq 的维度是 (batch_size, seq_len)
    # 返回的 mask 维度是 (batch_size, 1, seq_len)，以便于广播
    return (seq != pad_token).unsqueeze(1)

def create_subsequent_mask(size):
    # 创建后续掩码，用于解码器自注意力
    # size 是序列长度
    # 返回一个下三角矩阵，对角线以下都是 1
    # 创建下三角矩阵 (包含对角线)
    attn_shape = (1, size, size)
    subsequent_mask = torch.tril(torch.ones(attn_shape)).type(torch.bool)
    return subsequent_mask
