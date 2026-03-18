"""
las.py

负责：整合 Encoder、Decoder 和 Attention，形成完整的 LAS 模型。

整体流程：
    inputs (log-Mel 特征) + input_lengths
        -> Encoder
        -> EncoderOutputs + EncoderHidden
        -> Decoder (多步调用 + Attention)
        -> logits over vocabulary
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn

from .encoder import Encoder
from .attention import BahdanauAttention
from .decoder import Decoder


class LASModel(nn.Module):
    """
    Listen, Attend and Spell 模型整体封装。
    """

    def __init__(self, config, vocab_size: int):
        """
        Purpose:
            - 根据 Config 初始化 Encoder、Attention、Decoder 三大组件
            - 把模型结构与配置文件解耦，方便调整超参数

        Inputs:
            - config: Config 实例，包含各种维度与超参数
            - vocab_size: 词表大小

        Outputs:
            - 初始化好的 LASModel，可以在 train.py 中直接使用

        Tensor shapes:
            - 初始化阶段不涉及实际张量运算

        Common mistakes:
            - Encoder 输出维度与 Attention / Decoder 输入维度不匹配
            - 忘记把模型搬到 GPU (config.device) 上
            - 把 vocab_size 传错（与实际词表大小不一致）

        Relationship to the next step in the pipeline:
            - train.py 会在每个 batch 中调用 model(inputs, input_lengths, targets, target_lengths, ...)
            - decode.py 会在推理阶段调用 model 的“仅解码”接口
        """
        super().__init__()

        self.config = config
        self.vocab_size = vocab_size

        # TODO: 初始化 Encoder、Attention、Decoder
        self.encoder = Encoder(
            input_dim=config.encoder_input_dim,
            hidden_dim=config.encoder_hidden_dim,
            num_layers=config.encoder_num_layers,
            dropout=config.dropout,
        )

        self.attention = BahdanauAttention(
            encoder_hidden_dim=2 * config.encoder_hidden_dim,
            decoder_hidden_dim=config.decoder_hidden_dim,
            attention_dim=config.attention_dim,
        )

        self.decoder = Decoder(
            vocab_size=vocab_size,
            embedding_dim=config.decoder_embedding_dim,
            decoder_hidden_dim=config.decoder_hidden_dim,
            encoder_hidden_dim=2 * config.encoder_hidden_dim,
            attention_module=self.attention,
            dropout=config.dropout,
        )

        # TODO: 可能还需要一个线性层把 encoder 最终状态映射到 decoder 初始 hidden/cell

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
        targets: torch.Tensor,
        teacher_forcing_ratio: float,
        sos_id: int,
        eos_id: int,
    ) -> Dict[str, Any]:
        """
        Purpose:
            - 定义训练阶段的整体前向过程
            - 将“音频特征序列”和“目标字符序列”映射到“每个时间步的 logits”

        Inputs:
            - inputs: (B, T_in, n_mels)
            - input_lengths: (B,)
            - targets: (B, T_out)
            - teacher_forcing_ratio: float
            - sos_id, eos_id: int，用于 decoder

        Outputs:
            - output_dict: Dict，包含：
                - "logits": (B, T_out, vocab_size)
                - "attention_weights": (B, T_out, T_in)
                - （可选）"encoder_outputs" 等

        Tensor shapes:
            - encoder_outputs: (B, T_in, 2 * encoder_hidden_dim)
            - decoder 输出 logits: (B, T_out, vocab_size)

        Common mistakes:
            - 忘记对输入进行长度排序 / pack，导致 LSTM 处理错误
            - encoder 与 decoder 之间的 hidden 映射不清晰（维度不匹配）
            - 忘记在损失计算时对 padding 部分 mask 掉或 ignore_index

        Relationship to the next step in the pipeline:
            - train.py 会拿到 logits 并计算交叉熵损失
            - visualize.py 会利用 attention_weights 绘制注意力对齐图
        """
        # TODO:
        # 1. 使用 Encoder 对 inputs 编码，得到 encoder_outputs, encoder_hidden
        # 2. 基于 encoder_outputs 构造 encoder_mask (B, T_in)
        # 3. 把 encoder_hidden 映射为 decoder 的初始 hidden/cell
        # 4. 调用 self.decoder.forward(...) 得到 logits 和 attention_weights
        # 5. 打包成字典返回
        pass

    def greedy_decode(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
        max_decode_len: int,
        sos_id: int,
        eos_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Purpose:
            - 在推理阶段进行简单的贪心解码（每步选取概率最大的 token）
            - 用于快速得到预测序列，配合 CER 评估模型性能

        Inputs:
            - inputs: (B, T_in, n_mels)
            - input_lengths: (B,)
            - max_decode_len: int，最大解码步数，防止死循环
            - sos_id, eos_id: int

        Outputs:
            - predictions: (B, T_pred)
              每条样本的预测 id 序列（已经截断到 eos 或 max_decode_len）
            - all_attention_weights: (B, T_pred, T_in)
              每步的注意力权重，用于可视化

        Tensor shapes:
            - 和训练阶段类似，只是现在不使用 targets 和 teacher forcing
            - 每步都基于上一步预测结果输入 decoder

        Common mistakes:
            - 没有在遇到 eos_id 时提前停止生成，导致输出全是 pad
            - 没有记录 attention 权重，后面无法画对齐图
            - 忘记关闭 dropout 或切换到 eval 模式

        Relationship to the next step in the pipeline:
            - decode.py 会调用本函数得到预测结果
            - metrics.py 使用 predictions 与参考 labels 计算 CER
            - visualize.py 使用 all_attention_weights 绘制注意力热力图
        """
        # TODO:
        # 1. 使用 Encoder 编码
        # 2. 初始化 decoder 的 hidden/cell
        # 3. 构造初始输入 token (sos_id)
        # 4. while 循环或 for 循环，重复调用 decoder.forward_step
        # 5. 记录每步预测的 token 和 attention 权重
        pass


