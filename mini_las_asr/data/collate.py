"""
collate.py

负责：定义 DataLoader 的自定义 collate_fn，把多个可变长度样本打包成 batch 张量。

关键工作：
    - 对 log-Mel 特征按最长序列进行 padding
    - 对字符 id 序列按最长目标长度进行 padding
    - 保留每个样本的真实长度（input_lengths, target_lengths）

整体数据流（以一个 batch 为例）：
    Dataset.__getitem__:
        第 i 条样本 ->
            input_i: (T_in_i, n_mels)
            target_i: (T_out_i,)
            input_length_i: int
            target_length_i: int
    collate_fn:
        ├── 对所有 i 的 input_i 找最大长度 max_T_in
        ├── 对所有 i 的 target_i 找最大长度 max_T_out
        ├── 在时间维度对 input_i 做 padding -> (B, max_T_in, n_mels)
        ├── 在序列维度对 target_i 做 padding -> (B, max_T_out)
        └── 记录 input_lengths, target_lengths -> (B,)

    结果 batch_dict:
        "inputs": (B, max_T_in, n_mels)
        "input_lengths": (B,)
        "targets": (B, max_T_out)
        "target_lengths": (B,)
"""

from typing import List, Dict, Any
import torch


def collate_fn(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    """
    Purpose:
        - 将 Dataset 返回的多个样本组合成一个 batch
        - 解决“输入序列长度不同”的问题，让模型可以在 GPU 上一次性处理多个样本

    Inputs:
        - batch: List[Dict]，每个元素是 dataset.__getitem__ 返回的样本
          每个样本中至少有：
            - "input": (T_in_i, n_mels)
            - "input_length": int
            - "target": (T_out_i,)
            - "target_length": int
        - pad_id: 用于填充 target 序列的 pad token id

    Outputs:
        - batch_dict: Dict[str, torch.Tensor]，包含：
            - "inputs": (batch_size, max_T_in, n_mels)
            - "input_lengths": (batch_size,)
            - "targets": (batch_size, max_T_out)
            - "target_lengths": (batch_size,)

    Tensor shapes（构造过程）:
        假设 batch 内有 B 条样本：

        单条样本：
            input_i: (T_in_i, n_mels)
            target_i: (T_out_i,)

        第一步：找最大长度
            max_T_in  = max(T_in_1, ..., T_in_B)
            max_T_out = max(T_out_1, ..., T_out_B)

        第二步：构造承载张量
            inputs:  zeros or padding, 形状 (B, max_T_in, n_mels)
            targets: full of pad_id, 形状 (B, max_T_out)

        第三步：逐样本拷贝
            对第 i 个样本：
                inputs[i, :T_in_i, :]  = input_i
                targets[i, :T_out_i]   = target_i
                input_lengths[i]       = T_in_i
                target_lengths[i]      = T_out_i

        最终：
            inputs: (B, max_T_in, n_mels)   -> 喂给 Encoder
            input_lengths: (B,)             -> 用于 pack_padded_sequence 或 mask
            targets: (B, max_T_out)         -> 用于 Decoder 训练 + 计算交叉熵
            target_lengths: (B,)            -> 可用于截断损失或统计

    Common mistakes:
        - padding 时 pad 的维度搞错（如在频率维度 pad，而不是时间维度）
            正确：在时间维度 pad，使每条样本时间步数一致；
        - 忘记记录真实长度，导致 encoder/decoder 无法 mask 掉 padding 部分
            后果：模型会把全 0 或 pad token 当成真实数据学习
        - input_lengths / target_lengths 的数据类型不对（最好是 long）
        - 混淆 "B, T, ..." 的维度顺序，导致后续 LSTM 前向时报错

    Relationship to the next step in the pipeline:
        - train.py 会在每个训练 step 中调用 DataLoader，
          而 DataLoader 内部会使用本 collate_fn 生成 batch：
              for batch in dataloader:
                  inputs = batch["inputs"]         # (B, max_T_in, n_mels)
                  input_lengths = batch["input_lengths"]   # (B,)
                  targets = batch["targets"]       # (B, max_T_out)
                  target_lengths = batch["target_lengths"] # (B,)
        - 这些 batch 张量会直接传入 LASModel.forward():
              logits = model(inputs, input_lengths, targets, ...)
        - LASModel 和 Encoder/Decoder 会利用 input_lengths / target_lengths
          来构造 mask 或使用 pack_padded_sequence，避免在 pad 上浪费计算。
    """
    # TODO:
    # 1. 从 batch 中取出所有 input, input_length, target, target_length
    # 2. 找到 max_T_in, max_T_out
    # 3. 创建全 0 或 pad_id 的张量，并把每个样本拷贝进去
    # 4. 小心确保 dtype：
    #       - inputs: float32
    #       - targets: long（int64），方便用于 nn.CrossEntropyLoss
    # 5. 返回一个字典
    pass


