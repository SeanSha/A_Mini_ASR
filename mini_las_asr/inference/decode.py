"""
decode.py

负责：推理阶段的解码逻辑包装。

用途：
    - 从训练好的模型和一个 DataLoader 得到预测文本
    - 可选择 greedy 或 beam search（本项目从 greedy 开始）
"""

from typing import List, Dict, Any
import torch
from torch.utils.data import DataLoader

from ..models.las import LASModel
from ..data.vocab import CharVocab


def greedy_decode_dataloader(
    model: LASModel,
    dataloader: DataLoader,
    vocab: CharVocab,
    device: str,
) -> List[Dict[str, Any]]:
    """
    Purpose:
        - 对整个 DataLoader 中的样本进行贪心解码
        - 返回每条样本的预测文本、参考文本以及注意力权重等信息

    Inputs:
        - model: 训练好的 LASModel
        - dataloader: 测试或验证集的 DataLoader
        - vocab: CharVocab，用于 ids_to_text
        - device: "cpu" 或 "cuda"

    Outputs:
        - results: List[Dict]，每个元素包含：
            - "ref": str，参考文本
            - "hyp": str，预测文本
            - "attention": torch.Tensor, (T_pred, T_in)
              （可选）某些实现中可能只保存部分样本的 attention

    Tensor shapes:
        - dataloader 提供：
            - inputs: (B, T_in, n_mels)
            - input_lengths: (B,)
            - targets: (B, T_out)
            - target_lengths: (B,)
        - 模型 greedy_decode 输出：
            - predictions: (B, T_pred)
            - attention_weights: (B, T_pred, T_in)

    Common mistakes:
        - 忘记切换到 eval 模式和关闭梯度
        - 把 targets 当成 hyp，或反之
        - 在 ids_to_text 时没有去掉 <sos>/<eos>/<pad>

    Relationship to the next step in the pipeline:
        - evaluation.metrics 会对这些结果计算 CER
        - visualization.visualize 可以从某条结果中取 attention 来绘制注意力图
    """
    # TODO:
    # 1. model.eval(), torch.no_grad()
    # 2. 遍历 dataloader，调用 model.greedy_decode
    # 3. 使用 vocab.ids_to_text 将预测和参考都转为字符串
    # 4. 将结果以 dict 形式存入列表
    pass


