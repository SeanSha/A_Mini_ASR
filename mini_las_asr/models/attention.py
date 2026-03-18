"""
attention.py

负责：实现 Bahdanau（Additive）注意力机制。

在 LAS 中，注意力层的作用，可以按“每个解码时间步”拆成一个固定流程：

    第 t_dec 个解码步时：
        1. 取当前解码器隐藏态：decoder_hidden_t  (B, D_dec)
        2. 取整条编码器输出序列：encoder_outputs (B, T_in, D_enc)
        3. 对每个时间步 t_enc 计算一个 score：
               score[b, t_enc] = v^T * tanh(W_enc * h_enc[b, t_enc] + W_dec * h_dec[b])
        4. 对 score 在时间维 T_in 上做 softmax，得到 attention_weights[b, t_enc]
        5. 用 attention_weights 对 encoder_outputs 加权求和，得到 context_t (B, D_enc)
        6. 将 context_t 交给 Decoder，用来预测当前字符。

从直观角度看：
    - encoder_outputs[b, :, :] 是“一整条语音时间轴上的记忆向量序列”；
    - decoder_hidden[b, :] 是“当前要写的这个字符的内部状态”； 
    - attention 就是在问：“现在要写这个字符时，这条语音里哪些时间片最相关？”，
      然后用 softmax 权重把这些时间片的表示融合成一个 context 向量。
"""

from typing import Tuple
import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    """
    Bahdanau 注意力：score(h_t_dec, h_t_enc) = v^T * tanh(W_enc * h_t_enc + W_dec * h_t_dec)

    其中：
        - h_t_dec: 当前解码时间步的 decoder hidden（query）
        - h_t_enc: 各个编码时间步的 encoder output（keys / values）
        - v, W_enc, W_dec: 可训练参数

    与“点积注意力”的区别：
        - 点积：score = <W_q h_dec, W_k h_enc>
        - Bahdanau：先把两个向量投到同一中间空间 attention_dim 上，加和后再过 tanh，再和 v 做点积；
        - 好处：可以学习到更灵活的非线性匹配函数，而不仅仅是简单相似度。
    """

    def __init__(
        self,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        attention_dim: int,
    ):
        """
        Purpose:
            - 定义 Bahdanau 注意力中的线性层和可训练向量 v
            - 约定好“编码器表示维度”和“解码器隐藏维度”，以及其中间的 attention_dim
            - 为后续 step-by-step 的注意力计算准备参数容器

        Inputs:
            - encoder_hidden_dim: 编码器输出维度 D_enc
                - 如果 Encoder 是 BiLSTM，通常 = 2 * encoder_hidden_dim（单向）
            - decoder_hidden_dim: 解码器隐藏维度 D_dec
            - attention_dim: 内部对齐空间维度 D_att（W_enc 和 W_dec 把向量映射到这里）

        Outputs:
            - 初始化后的 BahdanauAttention 模块，内部大致包含：
                - W_enc: 线性层，用于映射 encoder_outputs: (B, T_in, D_enc) -> (B, T_in, D_att)
                - W_dec: 线性层，用于映射 decoder_hidden: (B, D_dec)      -> (B, D_att)
                - v:     一个长度为 D_att 的向量（也可以实现为 Linear(D_att, 1)）

        Tensor shapes（参数侧）:
            - W_enc.weight: (D_att, D_enc)
            - W_dec.weight: (D_att, D_dec)
            - v:            (D_att,) 或等价形状

        Common mistakes:
            - 混淆 encoder_hidden_dim 与 decoder_hidden_dim 的含义
            - attention_dim 取值太小/太大，没有考虑 D_enc 与 D_dec 的规模
            - 线性层输入的维度不匹配（尤其是在双向 LSTM 输出时，D_enc = 2 * hidden_dim）

        Relationship to the next step in the pipeline:
            - decoder.Decoder 在每个时间步都会调用本模块：
                  context_t, attn_weights_t = attention(decoder_hidden_t, encoder_outputs, encoder_mask)
            - 得到的 context_t 会与 decoder 当前的输出状态拼接，用于预测下一个字符。
        """
        super().__init__()

        # TODO: 按照 Bahdanau 注意力公式定义线性层和向量 v
        self.W_enc = None
        self.W_dec = None
        self.v = None

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Purpose:
            - 计算“当前解码时间步”的注意力权重和上下文向量 context
            - 直观理解：告诉模型“现在要写这个字符时，该主要听输入语音的哪一段”

        Inputs:
            - decoder_hidden: (B, decoder_hidden_dim)
              当前时间步 decoder 的隐藏状态（通常取最顶层 LSTM 的 hidden）
            - encoder_outputs: (B, T_in, encoder_hidden_dim)
              整条输入语音在每个时间步的编码器输出序列：
                  encoder_outputs[b, t_enc, :] = h_enc_t_enc
            - encoder_mask: (B, T_in)，可选
              - 值为 0/1 或 True/False
              - 用于在计算注意力时屏蔽掉 padding 部分

        Outputs:
            - context: torch.Tensor, (B, encoder_hidden_dim)
              - 对 encoder_outputs 在时间维度按 attention 权重加权求和后的结果
              - 可以理解为“当前解码步视角下，从整段语音摘要出的一个向量”
            - attention_weights: torch.Tensor, (B, T_in)
              - 对每个时间步的注意力权重（softmax 结果）
              - 可以画成一条对齐曲线或热力图的一行

        Tensor shapes（一步一步）:

            1. 线性变换到 attention 空间：
                - enc_proj = W_enc(encoder_outputs)
                    输入： (B, T_in, D_enc)
                    输出： (B, T_in, D_att)
                - dec_proj = W_dec(decoder_hidden)
                    输入： (B, D_dec)
                    输出： (B, D_att)
                - 为了和 enc_proj 相加，需要把 dec_proj 扩展时间维：
                    dec_proj_expanded: (B, 1, D_att) -> (B, T_in, D_att)

            2. 计算“匹配能量”（energy / score）：
                - energy_tilde = tanh(enc_proj + dec_proj_expanded)
                      形状： (B, T_in, D_att)
                - 对 D_att 维度与向量 v 做“点积”：
                      score[b, t_enc] = v^T * energy_tilde[b, t_enc, :]
                      得到：
                      scores: (B, T_in)

            3. 使用 mask 并做 softmax：
                - 如果有 encoder_mask，则对 pad 位置的 scores 设为 -inf
                - attention_weights = softmax(scores, dim=-1)
                      形状： (B, T_in)

            4. 计算上下文向量 context：
                - 将 attention_weights 视为对时间维 T_in 的权重：
                      context[b, :] = Σ_t_enc attention_weights[b, t_enc] * encoder_outputs[b, t_enc, :]
                - 得到：
                      context: (B, D_enc)

        Common mistakes:
            - 忘记在有 mask 的地方对 padding 位置设为 -inf 再 softmax，
              导致模型在纯 padding 上分配了不少注意力
            - 把时间维度和 batch 维度搞混（例如错误地在 batch 维度上 softmax）
            - decoder_hidden 的形状不对，没有从 (num_layers, B, D) 压缩成 (B, D)
            - 误把 D_att 当成 encoder_hidden_dim 或 decoder_hidden_dim 使用，维度对不上

        Relationship to the next step in the pipeline（按解码时间步看）:
            对于每个解码时间步 t_dec：
                1. Decoder 先根据前一时刻 hidden/cell 和当前输入 token 走一步 LSTM，
                   得到新的 decoder_hidden_t
                2. 调用 BahdanauAttention.forward(...)，获取：
                       context_t, attention_weights_t
                3. 将 context_t 与当前 decoder 的输出拼接，
                   通过线性层得到对下一个字符的 logits 分布
                4. attention_weights_t 会被收集起来，在可视化时画成一张
                   (T_out, T_in) 的对齐热力图。

        Step-by-step 总结（一个解码步）:
            1) 输入：decoder_hidden_t, encoder_outputs, encoder_mask
            2) 投影到统一 attention 空间：enc_proj, dec_proj
            3) 相加 + tanh + v^T -> scores (B, T_in)
            4) 掩蔽 pad，softmax -> attention_weights (B, T_in)
            5) 用 attention_weights 对 encoder_outputs 加权求和 -> context (B, D_enc)
        """
        # TODO:
        # 1. 使用线性层将 encoder_outputs 和 decoder_hidden 映射到同一 attention 空间
        # 2. 求和 + tanh，再与 v 点积，得到对每个时间步的 score
        # 3. 应用 mask（如果有），再做 softmax 得到 attention_weights
        # 4. 按权重对 encoder_outputs 加权求和，得到 context
        pass


