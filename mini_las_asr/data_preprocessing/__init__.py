"""
data_preprocessing 子包

用于存放所有“数据准备 / 预处理”相关模块：
- vocab: 字符级词表（CharVocab）
- audio_utils: 音频加载与 log-Mel 特征提取
- dataset: LJSpeechDataset，将元数据组织成样本
- collate: 自定义 collate_fn，用于 DataLoader 打包 batch

从宏观上看，这个子包负责把：
    “(路径, 文本)” 这样的原始信息
转成：
    “(log-Mel 特征张量, 文本 id 序列张量)” 这样的模型可用张量。
"""


