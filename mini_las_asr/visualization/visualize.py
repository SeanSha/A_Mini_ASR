"""
visualize.py

负责：可视化注意力权重（attention map），帮助直观理解“听写对齐”。

典型做法：
    - 对于一条样本，拿到 (T_out, T_in) 的 attention 矩阵
    - 使用 matplotlib 画成热力图
    - x 轴：输入时间帧；y 轴：输出字符位置；
      颜色：注意力权重大小
"""

from typing import List
import torch


def plot_attention(
    attention_weights: torch.Tensor,
    input_tokens: List[str] = None,
    output_tokens: List[str] = None,
    save_path: str = None,
):
    """
    Purpose:
        - 绘制单条样本的注意力热力图
        - 帮助你判断模型是否学会了合理的时间对齐关系

    Inputs:
        - attention_weights: torch.Tensor, 形状 (T_out, T_in)
          - 通常来自 LASModel.greedy_decode 或 Decoder.forward 的输出
        - input_tokens: List[str], 可选
          - 对应输入时间帧的文本标签，通常没有（语音是连续的）
          - 可以选择不传或传一些辅助标记（如时间戳）
        - output_tokens: List[str], 可选
          - 解码出的字符序列，用于标注 y 轴
        - save_path: str, 可选
          - 若提供，则将图像保存到该路径，否则可以直接显示

    Outputs:
        - 无直接返回值（可能返回 matplotlib 的 figure 对象，视你实现而定）

    Tensor shapes:
        - attention_weights: (T_out, T_in)
          - 注意和训练阶段 (B, T_out, T_in) 的 batch 维度不同
          - 可视化时常常只画一条样本

    Common mistakes:
        - 把 (T_in, T_out) 搞反，导致图像轴颠倒
        - 没有对 attention_weights 做 detach / 转到 CPU 就直接绘图
        - 在有 batch 维度的 attention_weights 上没有先选某一条样本

    Relationship to the next step in the pipeline:
        - 可视化不是训练流程中的一部分，但它是“理解模型”的重要工具
        - 你可以在训练过程中定期选取几条样本画图，观察模型学习情况
    """
    # TODO:
    # 1. 将 attention_weights 从 GPU 移到 CPU
    # 2. 使用 matplotlib 的 imshow 或 matshow 绘制热力图
    # 3. 设置 x/y 轴标签，若 output_tokens 提供则标在 y 轴
    # 4. 保存或显示图像
    pass


