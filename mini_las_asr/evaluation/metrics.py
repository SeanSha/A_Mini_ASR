"""
metrics.py

负责：计算评估指标，特别是字符错误率（CER）。

CER 定义：
    CER = (插入数 + 删除数 + 替换数) / 参考文本总字符数

需要先将预测文本和参考文本对齐（使用编辑距离），然后统计三种错误类型。
"""

from typing import List, Tuple


def levenshtein_distance(ref: List[str], hyp: List[str]) -> Tuple[int, int, int]:
    """
    Purpose:
        - 计算两条字符序列之间的编辑距离，并统计插入、删除、替换的数量
        - 这是 CER 的基础

    Inputs:
        - ref: 参考字符序列，例如 list("hello")
        - hyp: 预测字符序列，例如 list("hallo")

    Outputs:
        - (num_sub, num_ins, num_del):
            - num_sub: 替换次数
            - num_ins: 插入次数
            - num_del: 删除次数

    Tensor shapes:
        - 输入输出都为 Python list / int，不涉及张量

    Common mistakes:
        - 把 Levenshtein 距离的 DP 表索引搞混
        - 忽略 ref 或 hyp 为空串的边界情况
        - 把 CER 定义成 (sub+ins+del)/len(hyp)，而不是 /len(ref)

    Relationship to the next step in the pipeline:
        - compute_cer 会基于本函数的输出计算最终的 CER 值
    """
    # TODO: 实现标准的 Levenshtein 编辑距离算法（基于动态规划）
    pass


def compute_cer(ref_texts: List[str], hyp_texts: List[str]) -> float:
    """
    Purpose:
        - 在一批样本上计算平均 CER
        - 评估整个 ASR 系统的性能

    Inputs:
        - ref_texts: List[str]，参考文本列表
        - hyp_texts: List[str]，模型预测文本列表
          两者长度应当一致

    Outputs:
        - cer: float，字符错误率

    Tensor shapes:
        - 不涉及张量，只是字符串列表

    Common mistakes:
        - 忽略大小写或空格处理的一致性，导致 CER 不可比
        - 把所有句子字符数之和用错（应当是总的参考字符数）

    Relationship to the next step in the pipeline:
        - training.train 在每个 epoch 结束时可以用 compute_cer 在验证集上评估模型
        - 训练完成后，你可以使用 CER 来判断是否需要改进模型结构或训练策略
    """
    # TODO:
    # 1. 遍历所有样本，调用 levenshtein_distance
    # 2. 积累 sub, ins, del 与参考字符总数
    # 3. 返回 (sub+ins+del)/总参考字符数
    pass


