"""
vocab.py

这个文件用于管理字符级词表（vocabulary）：
- 把字符映射到整数 id（char -> index）
- 把整数 id 映射回字符（index -> char）
- 处理特殊 token（<pad>, <sos>, <eos>, <unk>）

在 LAS 中，解码器输出的是序列 token（这里是字符），
因此词表是“解码器输入/输出张量”与“人类可读字符串”之间的桥梁。

从数据流角度看：
    原始文本字符串 (str)
        ├── text_to_ids()
        ▼
    目标 id 序列 (List[int])
        ├── collate_fn 中 padding，变成 (B, T_out) 的 long 张量
        ▼
    解码器输入/输出张量 (targets)

    预测阶段：
        模型输出的预测 id 序列 (List[int] 或 (T_pred,) 张量)
        ├── ids_to_text()
        ▼
        最终识别文本 (str)
"""

from typing import List, Dict


class CharVocab:
    """
    一个简单的字符级词表类。

    注意：这里不实现任何真正逻辑，只给出接口和注释，引导你理解。
    """

    def __init__(
        self,
        chars: List[str],
        pad_token: str = "<pad>",
        sos_token: str = "<sos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
    ):
        """
        Purpose:
            - 初始化字符词表，构建 char -> id 与 id -> char 的映射
            - 把特殊 token 放在固定或约定的位置（例如 0 是 <pad> 等）
            - 为后续的数据预处理、解码提供统一的索引方式

        Inputs:
            - chars: 普通字符列表，比如 ['a', 'b', 'c', ..., ' ']
              （不包括特殊 token，特殊 token 单独指定）
            - pad_token, sos_token, eos_token, unk_token: 特殊 token 名称

        Outputs:
            - 内部保存：
              - self.char2id: Dict[str, int]
              - self.id2char: Dict[int, str]
              - 各种特殊 token 的 id：pad_id, sos_id, eos_id, unk_id

        Tensor shapes:
            - 这里不直接涉及张量形状，但后续会影响：
              - 解码器输入张量形状为 (batch_size, tgt_len)
              - 每个位置是一个 int，对应词表中的某个 token id

        Common mistakes:
            - 忘记把 <pad>, <sos>, <eos>, <unk> 放入词表，导致索引出错
            - 不统一 special token 的 id，导致不同模块之间不兼容
            - 对 LJSpeech 文本预处理时，字符集合没对齐（大小写、空格等）

        Relationship to the next step in the pipeline:
            - 在 dataset.py 中，会用到 CharVocab 来把原始文本转成 id 序列：
                  text (str) -> text_to_ids(text) -> List[int]
            - 在 collate.py 中，多个 List[int] 会被拼成 (B, T_out) 的张量
            - 在 decode.py 中，会用到 CharVocab 来把预测的 id 序列转回字符串：
                  List[int] -> ids_to_text(...) -> str
        """
        # TODO: 在这里初始化各种映射与特殊 token id
        # 数据流示意：
        #   1. 把 special tokens 放在前面（比如 0~3）
        #   2. 把普通字符依次添加进去
        #   3. 建立 char2id / id2char 两个方向的映射
        self.char2id: Dict[str, int] = {}
        self.id2char: Dict[int, str] = {}
        self.pad_id: int = 0
        self.sos_id: int = 0
        self.eos_id: int = 0
        self.unk_id: int = 0

        # TODO: 填充映射表
        # 建议你在实现时打印出：
        #   - 词表大小
        #   - 前几个 token 及其 id
        # 帮助你理解后续张量中每个 int 对应的字符。
        pass

    def text_to_ids(self, text: str) -> List[int]:
        """
        Purpose:
            - 把一个字符串（文本）转成字符 id 序列
            - 用于训练数据准备和解码评价等环节

        Inputs:
            - text: 例如 "hello world"
              - 通常来自 LJSpeech 的 metadata 中的一行文本
              - 你可以根据 config 决定是否转小写、是否保留标点等

        Outputs:
            - ids: List[int]，例如 [12, 5, 7, ...]
              - 每个 int 对应词表中的一个 token
              - 是否在最前后添加 <sos>/<eos>，由你的设计决定

        Tensor shapes:
            - 函数本身返回 Python list: 长度为 tgt_len
            - 在 dataset.__getitem__ 中，会把 List[int] 转成 1D 张量：
                  target: (tgt_len,)
            - 在 collate_fn 中，会把多个样本的 1D 张量拼成 2D：
                  targets: (batch_size, max_tgt_len)

        Common mistakes:
            - 没有处理未知字符（用 <unk> 替代）
            - 忘记在句子前后添加 <sos> 与 <eos>（根据你的设计）
            - 和模型约定的特殊 token 使用不一致（例如模型假设有 <sos>，但预处理没加）
            - 把空格等正常字符不小心当成要丢弃的东西，导致解码结果缺少空格

        Relationship to the next step in the pipeline:
            - dataset.py 在加载每一行文本时，会调用这个函数生成目标 id 序列：
                  text -> ids (List[int]) -> torch.tensor(ids)
            - 这些 1D target 张量会在 collate.py 中组成 batch：
                  (tgt_len,) -> 拼成 (B, max_tgt_len)
            - 解码器在训练阶段会直接使用这个 batch 张量作为目标序列。
        """
        # TODO: 实现字符串到 id 序列的转换逻辑
        # 典型流程：
        #   1. 可选的预处理（lowercase、去掉多余空格）
        #   2. 对每个字符查 self.char2id，查不到的用 unk_id
        #   3. 可选：前后加 sos_id 和 eos_id
        pass

    def ids_to_text(self, ids: List[int]) -> str:
        """
        Purpose:
            - 把字符 id 序列还原成可读的字符串
            - 用于推理（decode），把模型输出转成最终识别结果

        Inputs:
            - ids: List[int]，通常来自模型预测的 argmax 或 beam search
              - 你可能会先从预测张量中取出一条序列：
                    predictions: (B, T_pred)
                    -> predictions[i]: (T_pred,)
                    -> 转成 List[int] 再传进来

        Outputs:
            - text: 对应的字符串，比如 "hello world"
              - 通常会去掉 <sos>/<eos>/<pad> 等特殊 token 之后再拼接

        Tensor shapes:
            - 这里输入是一个一维的 Python list (长度为 T_pred)
            - 在调用本函数之前，通常的数据流是：
                1. 模型输出 logits: (B, T_pred, vocab_size)
                2. 对 vocab 维度做 argmax -> (B, T_pred)
                3. 选取其中一条 -> (T_pred,) -> List[int]

        Common mistakes:
            - 没有去掉 <sos>/<eos>/<pad> 这些特殊 token
            - 解码时错把 <pad> 当普通字符留下来
            - 对于连续相同的 token（如延长音），不清楚要不要合并（本项目可以简单保留）

        Relationship to the next step in the pipeline:
            - decode.py 会利用本函数把预测出的 id 序列转换为文本：
                  ids -> text
            - metrics.py 会拿这些预测文本与参考文本计算 CER：
                  ref_texts, hyp_texts -> compute_cer
            - visualize.py 在画 attention 图时，可能会把 text 的字符标在 y 轴上。
        """
        # TODO: 实现 id 序列到字符串的转换逻辑
        # 典型流程：
        #   1. 遍历 ids，把 pad_id / sos_id / eos_id 过滤掉（或在遇到 eos_id 时停止）
        #   2. 用 self.id2char 把 id 转成字符
        #   3. 拼接成字符串
        pass


