"""
train.py

负责：训练循环。

职责：
    - 构建 DataLoader
    - 构建模型、优化器、损失函数
    - 迭代 epoch 与 batch：
        - 前向传播
        - 计算损失（交叉熵）
        - 反向传播 + 参数更新
    - （可选）在验证集上评估 CER
"""

from typing import Dict, Any
import torch
from torch.utils.data import DataLoader

from ..config.config import Config
from ..data.vocab import CharVocab
from ..data.dataset import LJSpeechDataset
from ..data.collate import collate_fn
from ..models.las import LASModel
from ..evaluation.metrics import compute_cer


def train_one_epoch(
    model: LASModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    config: Config,
    vocab: CharVocab,
) -> Dict[str, Any]:
    """
    Purpose:
        - 执行一次完整的训练 epoch
        - 在所有训练 batch 上：
            - 前向传播 -> 计算损失 -> 反向传播 -> 更新参数

    Inputs:
        - model: LASModel 实例
        - dataloader: 训练集的 DataLoader
        - optimizer: 优化器（如 Adam）
        - criterion: 损失函数（如 CrossEntropyLoss）
        - config: Config 实例
        - vocab: CharVocab（可选，用于日志可视化）

    Outputs:
        - epoch_stats: Dict，包含本 epoch 的统计信息
            - "loss": float，平均训练损失
            - （可选）"cer": float，若在训练集上也计算 CER

    Tensor shapes:
        - dataloader 提供 batch 字典：
            - inputs: (B, T_in, n_mels)
            - input_lengths: (B,)
            - targets: (B, T_out)
            - target_lengths: (B,)
        - model.forward 输出：
            - logits: (B, T_out, vocab_size)

    Common mistakes:
        - 忘记调用 model.train()，导致 dropout/bn 行为不正确
        - 在计算损失时没有对 padding 部分设为 ignore_index
        - 没有做梯度裁剪，LSTM 训练中容易出现梯度爆炸

    Relationship to the next step in the pipeline:
        - main.py 会在训练阶段多次调用 train_one_epoch
        - 你可以在每个 epoch 后调用 validate() 评估验证集 CER
    """
    # TODO:
    # 1. 切换 model.train()
    # 2. 遍历 dataloader，逐 batch 前向、反向、更新
    # 3. 对 logits 和 targets 计算交叉熵损失（注意 reshape + ignore_index）
    # 4. 累加损失并取平均
    pass


def validate(
    model: LASModel,
    dataloader: DataLoader,
    config: Config,
    vocab: CharVocab,
) -> Dict[str, Any]:
    """
    Purpose:
        - 在验证集上评估模型性能（主要是 CER）
        - 不进行反向传播，只做前向推理

    Inputs:
        - model: 训练好的或正在训练的 LASModel
        - dataloader: 验证集 DataLoader
        - config: Config
        - vocab: CharVocab，用于 ids_to_text

    Outputs:
        - val_stats: Dict，包含：
            - "cer": float
            - （可选）"loss": float

    Tensor shapes:
        - 与训练阶段类似，只是这里通常会用 greedy_decode 或 beam search 得到预测序列

    Common mistakes:
        - 忘记调用 model.eval() 和 torch.no_grad()
        - 混淆训练阶段 forward（需要 targets）与推理阶段 forward（只输入 acoustics）
        - 在计算 CER 时忘记清洗掉 <sos>/<eos>/<pad> token

    Relationship to the next step in the pipeline:
        - main.py 会在每个 epoch 后调用 validate，观察 CER 变化
        - 根据验证 CER 决定是否保存模型或调整超参数
    """
    # TODO:
    # 1. 切换 model.eval()
    # 2. 关闭梯度计算
    # 3. 对每个 batch 调用 model.greedy_decode 得到预测 id 序列
    # 4. 通过 vocab.ids_to_text 转成字符串
    # 5. 调用 compute_cer(ref_texts, hyp_texts)
    pass


