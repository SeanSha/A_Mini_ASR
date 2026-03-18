"""
decoder.py

负责：LAS 中的“Attend and Spell”部分。

功能：
    - 在每个解码时间步，读取上一个时间步生成的字符（或真值字符，用于教师强制），
      以及上一个时间步的隐藏状态
    - 调用 attention 计算上下文向量 context
    - 基于当前输入字符 embedding、上一隐藏状态和 context，更新 LSTM 状态并预测下一个字符分布
"""

from typing import Tuple
import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    LSTM 解码器 + Bahdanau 注意力。
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        decoder_hidden_dim: int,
        encoder_hidden_dim: int,
        attention_module: nn.Module,
        dropout: float,
    ):
        """
        Purpose:
            - 定义解码器的 embedding 层、LSTM 层和输出层
            - 接收来自编码器的表示和注意力模块，在字符级别逐步生成输出

        Inputs:
            - vocab_size: 词表大小（包括所有字符和特殊 token）
            - embedding_dim: 字符 embedding 的维度
            - decoder_hidden_dim: 解码器 LSTM 的隐藏维度
            - encoder_hidden_dim: 编码器输出的维度（通常为 2 * hidden_dim）
            - attention_module: BahdanauAttention 实例，用于计算 context
            - dropout: dropout 概率

        Outputs:
            - 初始化后的 Decoder 模块

        Tensor shapes:
            - 初始化阶段不涉及张量运算

        Common mistakes:
            - 忽略了 context 与 decoder hidden 的拼接/组合方式
            - embedding 层输入和输出的维度对应关系搞错
            - 输出层没有把维度映射到 vocab_size 上

        Relationship to the next step in the pipeline:
            - las.py 中会把 Encoder 和 Decoder 组合在一起，实现完整的 seq2seq 前向过程
            - train.py 将调用 LASModel 的 forward，进而使用 Decoder 的 step-by-step 计算
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.attention = attention_module

        # TODO: 定义 embedding 层、LSTM、输出线性层等
        self.embedding = None
        self.lstm = None
        self.fc_out = None
        self.dropout = None

    def forward_step(
        self,
        prev_tokens: torch.Tensor,
        prev_hidden: torch.Tensor,
        prev_cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Purpose:
            - 执行“单个时间步”的解码计算
            - 这是理解解码器工作机制的关键函数：
                1. 嵌入上一个时间步输入字符
                2. 结合 encoder_outputs 通过注意力得到 context
                3. 把 embedding + context 输入 LSTM，更新 hidden/cell
                4. 用新的 hidden+context 预测当前时间步字符分布

        Inputs:
            - prev_tokens: (B,)
              上一个时间步的输入 token id（训练时通常是真实字符，推理时是上一步预测结果）
            - prev_hidden: (num_layers, B, decoder_hidden_dim)
            - prev_cell: (num_layers, B, decoder_hidden_dim)
            - encoder_outputs: (B, T_in, encoder_hidden_dim)
            - encoder_mask: (B, T_in)，可选，用于屏蔽 padding

        Outputs:
            - logits: (B, vocab_size)
              当前时间步对每个 token 的预测得分（还没有 softmax）
            - hidden: (num_layers, B, decoder_hidden_dim)
            - cell: (num_layers, B, decoder_hidden_dim)
            - attention_weights: (B, T_in)

        Tensor shapes:
            - embedding(prev_tokens): (B, embedding_dim)
            - decoder_hidden 当前步: (B, decoder_hidden_dim)（注意可能需要从多层中取 top layer）
            - attention 结果 context: (B, encoder_hidden_dim)
            - LSTM 输入：通常是 embedding 和 context 的拼接，形状 (B, embedding_dim + encoder_hidden_dim)
            - LSTM 输出 hidden: (num_layers, B, decoder_hidden_dim)
            - logits: (B, vocab_size)

        Common mistakes:
            - 没有区分 training 时的教师强制输入与 inference 时的自回归输入
            - prev_tokens 的形状不对（例如 (B, 1) 而不是 (B,)）
            - attention 使用的 hidden 状态选错层（多层 LSTM 时要选最顶层或按约定来）

        Relationship to the next step in the pipeline:
            - 在 Decoder 的整体 forward 中，会在一个 for 循环中多次调用 forward_step
              来生成整个输出序列的 logits
        """
        # TODO:
        # 1. 对 prev_tokens 做 embedding
        # 2. 从 prev_hidden 中取出当前用于 attention 的隐藏状态
        # 3. 调用 self.attention 计算 context 和 attention_weights
        # 4. 把 embedding + context 输入 LSTM，得到新的 hidden, cell
        # 5. 使用一个线性层把 hidden + context 映射到 vocab_size，拿到 logits
        pass

    def forward(
        self,
        targets: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_mask: torch.Tensor,
        teacher_forcing_ratio: float,
        sos_id: int,
        eos_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Purpose:
            - 逐步解码整条目标序列，在训练阶段通常使用教师强制（Teacher Forcing）
            - 输出整个序列上每个时间步的 logits（用于计算交叉熵损失）

        Inputs:
            - targets: (B, T_out)
              真实的目标 token 序列，包括或不包括 <sos>/<eos>，取决于你的设计
            - encoder_outputs: (B, T_in, encoder_hidden_dim)
            - encoder_mask: (B, T_in)
            - teacher_forcing_ratio: float，[0,1]
              - 在每个时间步，决定是用真实 token 作为下一个步输入，还是用模型预测的 token
            - sos_id: int，起始 token id
            - eos_id: int，结束 token id（可能在推理中更重要）

        Outputs:
            - all_logits: (B, T_out, vocab_size)
              整个输出序列的 logits（每个时间步一个分布）
            - all_attention_weights: (B, T_out, T_in)
              每个解码步对应一组 attention 权重（用于可视化）

        Tensor shapes:
            - 初始时，你会先构造一个 (B,) 的当前输入 token（通常是 sos_id）
            - 每次 forward_step 生成 (B, vocab_size) 的 logits
            - 堆叠 T_out 次后，得到 (B, T_out, vocab_size)

        Common mistakes:
            - 没有正确处理 teacher_forcing_ratio 逻辑（导致永远 teacher forcing 或永远自回归）
            - 忘记限制最大解码长度，导致训练或推理时出现无限循环
            - 忘记记录 attention 权重，导致无法绘制对齐图

        Relationship to the next step in the pipeline:
            - las.py 中的 LASModel.forward 会调用本函数，
              并对返回的 logits 计算交叉熵损失
            - visualize.py 会使用 all_attention_weights 来绘制注意力对齐图
        """
        # TODO:
        # 1. 初始化 hidden, cell（通常从 encoder 的 final hidden 变换得到）
        # 2. 以 sos_id 作为第一个解码步的输入
        # 3. 循环 T_out 次，每次调用 forward_step
        # 4. 根据教师强制比率选择当前步输入是真实 target 还是预测 token
        # 5. 收集每个时间步的 logits 和 attention_weights
        pass


