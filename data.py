import torch
from utils import create_padding_mask, create_subsequent_mask

# 定义特殊符号的索引
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

def create_copy_data_batch(batch_size, max_len, vocab_size, device):
    # 为复制任务生成一个批次的数据
    # 随机生成序列长度 (为了让模型能处理变长序列)
    seq_len = torch.randint(1, max_len - 1, (1,)).item()
    # 随机生成数据 (词汇从 3 开始，因为 0,1,2 是特殊符号)
    # 维度: (batch_size, seq_len)
    data = torch.randint(3, vocab_size, (batch_size, seq_len))
    # 源序列 (Encoder Input)
    # 直接使用生成的数据，并用 PAD_TOKEN 填充到 max_len
    src = torch.full((batch_size, max_len), PAD_TOKEN)
    src[:, :seq_len] = data
    # 目标序列 (Decoder Input & Ground Truth)
    # 解码器的输入是 <sos> + sequence
    tgt_input = torch.full((batch_size, max_len), PAD_TOKEN)
    tgt_input[:, 0] = SOS_TOKEN
    tgt_input[:, 1:seq_len+1] = data
    # 用于计算损失的真实标签是 sequence + <eos>
    tgt_output = torch.full((batch_size, max_len), PAD_TOKEN)
    tgt_output[:, :seq_len] = data
    tgt_output[:, seq_len] = EOS_TOKEN
    # 创建掩码
    # 源序列的填充掩码
    src_mask = create_padding_mask(src, PAD_TOKEN)
    # 目标序列的掩码需要结合填充掩码和后续掩码
    tgt_padding_mask = create_padding_mask(tgt_input, PAD_TOKEN)
    tgt_subsequent_mask = create_subsequent_mask(max_len)
    # 两个掩码通过 & 操作合并
    tgt_mask = tgt_padding_mask & tgt_subsequent_mask
    return src.to(device), tgt_input.to(device), tgt_output.to(device), src_mask.to(device), tgt_mask.to(device)
