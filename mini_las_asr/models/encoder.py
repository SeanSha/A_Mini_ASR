"""
encoder.py

负责：实现 LAS 中的“Listen”部分，也就是双向 LSTM 编码器（BiLSTM）。

整体功能（从数据流角度）：
    单条样本：
        log-Mel 频谱: (T_in, n_mels)
            ├── 加入 batch 维度
            ▼
        inputs: (B, T_in, n_mels)  # B 是 batch_size
            ├── BiLSTM（多层、双向）
            ▼
        encoder_outputs: (B, T_in, 2 * hidden_dim)
            - 每个时间步 t 上的向量，综合了“过去”和“未来”的上下文信息
        encoder_final_hidden: (num_layers * 2, B, hidden_dim)（或你自定义的形状）
            - 编码器看到整条序列后的状态摘要，可用来初始化解码器

为什么需要“整个输出序列”而不只是最后一个 hidden？
    - 注意力（Attention）的核心思想：Decoder 在每个解码步 t_dec，
      都可以“从所有输入时间步 t_enc 中选择最相关的一些信息”。
    - 这要求我们保留“每个输入时间步的 BiLSTM 表示”：
          encoder_outputs[b, t_enc, :]  就是第 b 条样本、输入时间步 t_enc 的编码表示。
    - BahdanauAttention 会把：
          当前 decoder_hidden (B, D_dec)
          和整条 encoder_outputs (B, T_in, D_enc)
      放在一起，算一组对时间维度的权重 α[b, t_enc]，再加权求和出一个 context 向量。

直观比喻：
    - encoder_outputs 是“把整段语音切成很多小时间块，每块都用一个向量来表示”；
    - attention 每一步就是问：“我现在要写的这个字符，应该主要参考哪几个时间块？”；
    - 所以，如果你只保留最后一个 hidden state，而丢掉中间所有时间步的输出，
      attention 在时间轴上就无路可走了——因为它已经“看不到每一帧的信息分布”。
"""

from typing import Tuple
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    LAS 中的双向 LSTM 编码器。

    从张量视角可以这样理解 Encoder：

        输入：inputs: (B, T_in, input_dim)
            - B: batch_size
            - T_in: 这一 batch 中最长的时间步数（短的已经 pad）
            - input_dim: 每一帧的特征维度（在本项目中 = n_mels）

        内部（以双向单层 LSTM 为例）：
            - 前向 LSTM：从 t = 0 -> T_in-1，生成 h_fwd[b, t, :]
            - 反向 LSTM：从 t = T_in-1 -> 0，生成 h_bwd[b, t, :]
            - 对于每个时间步 t，把两个方向的隐藏态拼接：
                  h_enc[b, t, :] = [h_fwd[b, t, :], h_bwd[b, t, :]]
                于是维度从 hidden_dim -> 2 * hidden_dim

        输出：
            - encoder_outputs: (B, T_in, 2 * hidden_dim)
                每个 (b, t) 对应输入第 b 条样本在时间步 t 的“上下文表示”
            - encoder_final_hidden:
                通常形状为 (num_layers * 2, B, hidden_dim)
                代表所有层、两个方向在“序列末端”的隐藏态。

    对 attention 的重要性：
        - encoder_outputs 提供了一个“时间轴上的记忆库”：
              memory[b, t_enc, :] = encoder_outputs[b, t_enc, :]
        - 在注意力计算中，decoder 会拿当前的 decoder_hidden[b, :]
          去和 memory[b, :, :] 的每一个时间步做匹配打分；
        - 如果没有这条 (T_in 维度的) 序列，注意力就没法在时间维度上“扫描与加权”，
          只能变成类似普通 seq2seq 里“只看最后 hidden”的结构，表达能力差很多。
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        """
        Purpose:
            - 定义 BiLSTM 编码器的结构（层数、隐藏维度、dropout 等）
            - 明确“输入帧向量序列 -> 编码后帧向量序列”的映射关系
            - 为后续的注意力和解码器提供一个“可被查询的时间序列记忆库”

        Inputs:
            - input_dim: 输入特征维度（通常等于 n_mels）
            - hidden_dim: LSTM 隐藏维度（单向的）
            - num_layers: LSTM 堆叠层数
            - dropout: 层与层之间的 dropout 概率

        Outputs:
            - 初始化后的 Encoder 模块实例

        Tensor shapes:
            - 此处不进行前向计算，只是保存参数，但要在脑中固定几个关键维度：
                - inputs:           (B, T_in, input_dim)
                - encoder_outputs:  (B, T_in, 2 * hidden_dim)
                - encoder_hidden:   (num_layers * 2, B, hidden_dim)

            - 其中：
                - input_dim 通常等于 n_mels
                - hidden_dim 是单向 LSTM 的隐藏维度
                - 双向后，时间步上的输出维度会翻倍成 2 * hidden_dim

        Common mistakes:
            - 忘记设置 batch_first=True，导致维度顺序不一致
                正确：LSTM 接收 (B, T_in, input_dim)，而不是 (T_in, B, input_dim)
            - 使用单向 LSTM 而非双向 LSTM（BiLSTM），丢失“反向上下文信息”
            - 忽略 audio 序列的 padding mask，导致编码器把 pad 部分也当成有效语音
            - 对 hidden_dim 与 2 * hidden_dim 混淆，后续 attention / decoder 连不上

        Relationship to the next step in the pipeline:
            - 在 las.py 中，Encoder 会被调用：
                  encoder_outputs, encoder_hidden = encoder(inputs, input_lengths)
            - encoder_outputs: 会被 BahdanauAttention 当作“记忆序列 memory”
            - encoder_hidden:  常用于设定 Decoder 的初始 hidden/cell，
                               告诉 Decoder：“现在你已经听完整段语音的大致内容了”
        """
        super().__init__()

        # TODO: 定义 BiLSTM 层和可选的线性变换等
        self.lstm = None  # 占位
        self.dropout = None

    def forward(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Purpose:
            - 对 batch 内的 log-Mel 特征序列进行编码
            - 将“原始帧级特征序列”转换为“带有上下文的帧级表示序列”
            - 处理可变长度序列（利用 input_lengths 做 pack/pad 或 mask）

        Inputs:
            - inputs: torch.Tensor, 形状 (B, T_in, input_dim)
              - B: batch_size
              - T_in: 当前 batch 中最长的时间长度（短序列右侧 padding）
              - input_dim: 特征维度（= n_mels）
              - 举例：
                    B = 3
                    三条语音的帧数分别为 100, 80, 60
                    则 T_in = 100，后两条在 t >= 80 / 60 之后是 padding
            - input_lengths: torch.Tensor, 形状 (B,)
              - 每个样本的真实帧数（不含 pad）
              - 上例中 input_lengths = [100, 80, 60]

        Outputs:
            - encoder_outputs: torch.Tensor, 形状 (B, T_in, 2 * hidden_dim)
                - 第 b 条样本在时间步 t 的编码向量：
                      encoder_outputs[b, t, :] ∈ R^{2 * hidden_dim}
                - 这个向量是“在看过这条语音的前后若干帧之后，模型对这一个时间位置的理解”
            - encoder_final_hidden: torch.Tensor, 形状视设计而定
                - 通常为 (num_layers * 2, B, hidden_dim)
                - 包含了每一层、每个方向在“序列末端”的隐藏态
                - 你可以直接或经过线性变换，把它喂给 Decoder 作为初始状态

        Tensor shapes（更细）:
            - 输入：
                  inputs:         (B, T_in, input_dim)
                  input_lengths:  (B,)
            - pack 之后（逻辑上）：
                  packed_inputs:  按真实长度压缩，喂给 LSTM，省掉 pad 位置计算
            - BiLSTM 内部单向隐藏维度为 hidden_dim，
              双向输出在时间维度上拼接得到：
                  outputs:        (B, T_in, 2 * hidden_dim)
            - 最终返回：
                  encoder_outputs      = outputs
                  encoder_final_hidden = (num_layers * 2, B, hidden_dim)

        Common mistakes:
            - 没有正确处理 padding，导致 encoder 把 padding 当作真实语音：
                - 后果：attention 可能在这些“空白帧”上给出很大权重
            - 忘记使用 pack_padded_sequence / pad_packed_sequence，
              或者使用了，但 seq_len 维度顺序写错
            - 双向 LSTM 输出的两个方向连接时张量维度不清楚，
              把 (2 * hidden_dim) 当成 hidden_dim 用到 Attention / Decoder 里
            - 忽略 input_lengths 在后续 attention mask 中的作用

        Relationship to the next step in the pipeline:
            - attention.BahdanauAttention 会把 encoder_outputs 当作“记忆序列 memory”：
                  memory = encoder_outputs  # (B, T_in, D_enc)
            - decoder.Decoder 在每个解码时间步 t_dec 会：
                1. 拿当前 decoder_hidden[b, :] 作为 query
                2. 对 memory[b, :, :] 的每个时间步做打分，得到 attention_weights[b, :, :]
                3. 对 memory 加权求和得到 context[b, :]
            - 如果 encoder_outputs 的时间维信息被破坏（比如顺序错乱或 pad 未 mask），
              注意力就无法正确“在语音的时间轴上移动焦点”，
              也就很难学出合理的“听->写”对齐关系。
        """
        # TODO:
        # 1. 使用 input_lengths 对 inputs 进行 pack（可选）
        # 2. 通过 BiLSTM
        # 3. 还原为 pad 后的序列
        # 4. 整理最终的隐藏状态作为 encoder_final_hidden
        pass


