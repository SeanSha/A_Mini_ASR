"""
拷贝自原 data/vocab.py，仅更换包名为 data_preprocessing。
详尽注释见文件内部。
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
        """
        self.char2id: Dict[str, int] = {}
        self.id2char: Dict[int, str] = {}
        self.pad_id: int = 0
        self.sos_id: int = 0
        self.eos_id: int = 0
        self.unk_id: int = 0

        # TODO: 详细逻辑参考原注释
        pass

    def text_to_ids(self, text: str) -> List[int]:
        """参见原 data/vocab.py 中的详细注释。"""
        pass

    def ids_to_text(self, ids: List[int]) -> str:
        """参见原 data/vocab.py 中的详细注释。"""
        pass


