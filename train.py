import torch
import torch.nn as nn
from model import Transformer
from data import create_copy_data_batch, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
from utils import create_padding_mask, create_subsequent_mask
import time

def train(model, optimizer, criterion, epochs=10, steps_per_epoch=100):
    # 训练模型
    print("----开始训练----")
    # 设置为训练模式
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        start_time = time.time()
        for _ in range(steps_per_epoch):
            # 生成数据
            src, tgt_input, tgt_output, src_mask, tgt_mask = create_copy_data_batch(
                BATCH_SIZE, MAX_LEN, VOCAB_SIZE, device
            )
            # 模型前向传播
            out = model(src, tgt_input, src_mask, tgt_mask)
            # 计算损失
            # out 的维度是 (batch_size, max_len, vocab_size)
            # tgt_output 的维度是 (batch_size, max_len)
            # 需要将它们 reshape 以符合 CrossEntropyLoss 的要求
            loss = criterion(out.view(-1, VOCAB_SIZE), tgt_output.view(-1))
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / steps_per_epoch
        epoch_time = time.time() - start_time
        print(f"Epoch: {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
    print("----训练完成----\n")

def run_example(model, device):
    # 运行一个例子来检验模型效果
    # 设置为评估模式
    model.eval()
    # 创建一个测试序列
    test_seq = torch.tensor([[5, 8, 2, 7, 4, 6]], device=device)
    # 源序列和掩码
    src = torch.full((1, MAX_LEN), PAD_TOKEN, device=device)
    src[0, :test_seq.size(1)] = test_seq
    src_mask = create_padding_mask(src, PAD_TOKEN)
    # 编码器只需要运行一次
    with torch.no_grad():
        memory = model.encode(src, src_mask)
    # 解码器自回归生成
    # 从 <sos> token 开始
    tgt_input = torch.tensor([[SOS_TOKEN]], device=device)
    print(f"输入序列: {test_seq.cpu().numpy()[0]}")
    print("模型输出: ", end="")
    for _ in range(MAX_LEN -1):
        with torch.no_grad():
            # 创建目标序列的掩码
            tgt_mask = create_subsequent_mask(tgt_input.size(1)).to(device)
            # 解码
            # src_mask 在 cross-attention 中不需要了
            out = model.decode(tgt_input, memory, None, tgt_mask)
            # 获取最后一个时间步的 logits，并找到概率最大的 token
            prob = model.out(out[:, -1])
            next_token = prob.argmax(dim=-1).item()
            print(f"{next_token} ", end="")
            # 如果是 <eos>，则停止生成
            if next_token == EOS_TOKEN:
                break
            # 将新生成的 token 添加到解码器输入中，准备下一次循环
            next_token_tensor = torch.tensor([[next_token]], device=device)
            tgt_input = torch.cat([tgt_input, next_token_tensor], dim=1)
    print("\n")

if __name__ == '__main__':
    # 超参数定义
    VOCAB_SIZE = 11                # 词汇表示例大小 (0-10)
    D_MODEL = 512                  # 嵌入维度
    N_LAYERS = 3                   # Encoder/Decoder 层数 (为了快速训练，设为3)
    HEADS = 8                      # 多头注意力头数
    D_FF = 2048                    # 前馈网络隐藏层维度
    MAX_LEN = 15                   # 序列最大长度
    BATCH_SIZE = 64                # 批次大小
    EPOCHS = 20                    # 迭代次数
    LEARNING_RATE = 0.0001         # 学习率
    # 设备、模型、优化器、损失函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model = Transformer(
        src_vocab=VOCAB_SIZE, 
        tag_vocab=VOCAB_SIZE, 
        d_model=D_MODEL, 
        N=N_LAYERS, 
        h=HEADS, 
        d_ff=D_FF
    ).to(device)
    # 忽略 PAD_TOKEN，不计算它的损失
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 训练和评估
    train(model, optimizer, criterion, epochs=EPOCHS)
    run_example(model, device)
