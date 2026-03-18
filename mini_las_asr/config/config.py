"""
config.py

集中管理整个 Mini LAS ASR 项目的超参数与路径配置。
（内容拷贝自原 mini_las_asr/config.py）
"""


class Config:
    """
    一个简单的配置类，用于保存各种超参数与路径设置。

    你可以在这里放：
    - 数据路径（LJSpeech 根目录、预处理后的特征保存路径等）
    - 音频与特征相关设置（采样率、FFT 长度、Mel 滤波器数量、帧长、帧移等）
    - 模型结构相关设置（编码器/解码器隐藏维度、层数、dropout、多头数等）
    - 训练相关设置（batch_size、学习率、最大训练轮数、梯度裁剪阈值等）
    - 解码相关设置（beam size、最大解码长度、EOS token 等）

    你不需要在这里实现任何逻辑，只需要定义字段，并在其他模块导入使用即可。
    """

    def __init__(self):
        # =========================
        # 数据与路径相关配置
        # =========================
        # LJSpeech 数据集根目录
        self.data_root = "path/to/LJSpeech-1.1"

        # 例如音频文件所在文件夹，metadata.csv 所在路径等
        self.metadata_path = "path/to/LJSpeech-1.1/metadata.csv"

        # 预处理后的特征（如 log-Mel 频谱）缓存保存目录
        self.feature_cache_dir = "path/to/feature_cache"

        # =========================
        # 音频与特征提取相关配置
        # =========================
        # 音频采样率（LJSpeech 默认是 22050 Hz）
        self.sample_rate = 22050

        # STFT 相关参数
        self.n_fft = 1024
        self.win_length = 1024
        self.hop_length = 256

        # Mel 滤波器数量，即 log-Mel 频谱的频率维度大小
        self.n_mels = 80

        # 最长音频时长（秒），用于可选的裁剪或过滤
        self.max_audio_duration = 10.0

        # =========================
        # 文本与词表相关配置
        # =========================
        # 是否只使用小写字母
        self.lowercase = True

        # 是否在字符级别加入空格、标点等
        # 这些具体细节在 vocab.py 中说明
        # 这里只是一个开关或占位符
        self.use_punctuation = False

        # 特殊 token
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        # =========================
        # 模型结构相关配置
        # =========================
        # 编码器输入维度：等于 log-Mel 频谱的 Mel 维度 n_mels
        self.encoder_input_dim = self.n_mels

        # 编码器隐藏维度（双向 LSTM 的隐藏维度）
        self.encoder_hidden_dim = 256

        # 编码器层数
        self.encoder_num_layers = 3

        # 解码器嵌入维度（字符 embedding 维度）
        self.decoder_embedding_dim = 256

        # 解码器 LSTM 隐藏维度
        self.decoder_hidden_dim = 256

        # 注意力层维度（Bahdanau attention 内部的“对齐模型”维度）
        self.attention_dim = 256

        # dropout 概率
        self.dropout = 0.1

        # =========================
        # 训练相关配置
        # =========================
        self.batch_size = 16
        self.num_epochs = 50
        self.learning_rate = 1e-3
        self.gradient_clip = 5.0

        # =========================
        # 解码与推理配置
        # =========================
        self.beam_size = 1  # 从最简单的贪心解码开始
        self.max_decode_len = 200  # 最大字符长度，防止死循环

        # 是否在训练中使用教师强制（Teacher Forcing），以及比率
        self.teacher_forcing_ratio = 0.9

        # 设备设置（可以在 main.py 中根据是否有 GPU 覆盖）
        self.device = "cuda"  # 或 "cpu"


# 你可以选择在这里创建一个全局默认配置实例：
# config = Config()
# 也可以在 main.py 或 train.py 中手动创建。


