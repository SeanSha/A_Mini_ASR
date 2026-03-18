"""
config.py

集中管理整个 Mini LAS ASR 项目的**超参数与路径配置**。

这个文件本身不做任何“运算”，只是一个“配置仓库”：

- 数据路径：LJSpeech 根目录、metadata.csv、特征缓存目录等；
- 特征相关：采样率、FFT 长度、hop_length、Mel 维度等；
- 模型结构：encoder/decoder 的 hidden_dim、层数、attention_dim 等；
- 训练超参：batch_size、learning_rate、epoch 数、gradient clip 等；
- 解码相关：beam size、最大解码长度、teacher forcing 等。

学习建议：
- 先**把路径类参数改成你自己机器上真实存在的路径**；
- 再大致理解每一类参数会影响哪个模块（data / models / training）。
"""


class Config:
    """
    一个简单的配置类，用于保存各种超参数与路径设置。

    使用方式（典型）：
    - 在 `main.py` / `train.py` 中：
        from mini_las_asr.config.config import Config
        cfg = Config()
    - 然后：
        - 数据模块用 cfg.data_root, cfg.sample_rate, cfg.n_mels
        - 模型模块用 cfg.encoder_hidden_dim, cfg.attention_dim
        - 训练模块用 cfg.batch_size, cfg.learning_rate 等

    这里**不需要实现任何复杂逻辑**，只需要定义字段。
    但你可以适当加上一些小工具方法（例如打印摘要），帮助自己理解。
    """

    def __init__(self):
        """
        初始化配置对象。

        可以先把默认值写成“一个你打算使用的合理初值”，
        然后再逐渐根据自己机器 / 数据路径进行修改。
        """

        # =========================
        # 数据与路径相关配置
        # =========================
        # TODO: 1. 把下面三个路径改成你本机真实存在的路径
        # TODO: 2. 建议先手动下载/解压 LJSpeech，然后对齐这里
        # 例如：
        #   data_root = "/Users/你的用户名/data/LJSpeech-1.1"
        #   metadata_path = data_root + "/metadata.csv"
        #   feature_cache_dir = "/Users/你的用户名/data/mini_las_features"

        # LJSpeech 数据集根目录（里面应该有 wavs/ 和 metadata.csv）
        self.data_root = "path/to/LJSpeech-1.1"

        # metadata.csv 的完整路径（通常就在 data_root 下面）
        self.metadata_path = "path/to/LJSpeech-1.1/metadata.csv"

        # 预处理后的特征（如 log-Mel 频谱）缓存保存目录
        self.feature_cache_dir = "path/to/feature_cache"

        # =========================
        # 音频与特征提取相关配置
        # =========================
        # 音频采样率（LJSpeech 默认是 22050 Hz）
        # 影响：waveform 的采样点数、STFT 频率刻度
        self.sample_rate = 22050

        # STFT 相关参数：
        # - n_fft: FFT 窗长度
        # - win_length: 实际窗口长度（通常等于 n_fft）
        # - hop_length: 帧移（步长），控制时间分辨率 T_in
        self.n_fft = 1024
        self.win_length = 1024
        self.hop_length = 256

        # Mel 滤波器数量，即 log-Mel 频谱的频率维度大小
        # 后续特征张量的 shape 通常是 (T_in, n_mels)
        self.n_mels = 80

        # 最长音频时长（秒），用于可选的裁剪或过滤
        # 例如，你可以在 Dataset 里丢掉超过这个长度的样本。
        self.max_audio_duration = 10.0

        # =========================
        # 文本与词表相关配置
        # =========================
        # 是否把文本统一转成小写（在 vocab/text_to_ids 里可以用到）
        self.lowercase = True

        # 是否在字符级别加入标点等
        # 具体行为由 vocab.py 决定，这里只是一个“开关型”配置
        self.use_punctuation = False

        # 特殊 token 的字符串形式
        # 在 vocab.py 里，你会把这些字符串映射到具体的整数 id
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        # =========================
        # 模型结构相关配置
        # =========================
        # 编码器输入维度：等于 log-Mel 频谱的 Mel 维度 n_mels
        # Encoder 输入张量 shape 通常是 (B, T_in, encoder_input_dim)
        self.encoder_input_dim = self.n_mels

        # 编码器隐藏维度（双向 LSTM 的隐藏维度）
        # Encoder 输出张量 shape 通常是 (B, T_in, 2 * encoder_hidden_dim)
        self.encoder_hidden_dim = 256

        # 编码器层数（LSTM 堆叠层数）
        self.encoder_num_layers = 3

        # 解码器嵌入维度（字符 embedding 维度）
        self.decoder_embedding_dim = 256

        # 解码器 LSTM 隐藏维度
        self.decoder_hidden_dim = 256

        # 注意力层维度（Bahdanau attention 内部的“对齐模型”维度）
        self.attention_dim = 256

        # dropout 概率（通常在 LSTM / Linear 层之间使用）
        self.dropout = 0.1

        # =========================
        # 训练相关配置
        # =========================
        # 一个 batch 里包含多少条样本
        self.batch_size = 16

        # 最多训练多少个 epoch（轮次）
        self.num_epochs = 50

        # 初始学习率
        self.learning_rate = 1e-3

        # 梯度裁剪阈值（防止梯度爆炸）
        self.gradient_clip = 5.0

        # =========================
        # 解码与推理配置
        # =========================
        # beam_size = 1 时，相当于最简单的贪心解码
        self.beam_size = 1

        # 最大解码步数（字符长度），防止模型一直不产生 <eos> 而死循环
        self.max_decode_len = 200

        # 是否在训练中使用教师强制（Teacher Forcing），以及比率
        # teacher_forcing_ratio = 1.0 表示始终用真实 target 作为下一步输入
        self.teacher_forcing_ratio = 0.9

        # 设备设置：可以在 main.py 中根据是否有 GPU 覆盖
        # TODO: 你可以在 main.py 里写逻辑：
        #   device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cuda"  # 或 "cpu"

    # TODO（进阶练习，非必须）:
    # 你可以给 Config 加一个“小工具方法”，比如：
    # def summary(self):
    #     """
    #     打印出当前配置中最重要的几项，方便快速检查。
    #     """
    #     # 在这里用 print(...) 把关键信息打印出来
    #     raise NotImplementedError


# 你可以选择在别的文件中这样使用：
#   from mini_las_asr.config.config import Config
#   cfg = Config()
#   print(cfg.sample_rate, cfg.n_mels)
#
# 如果你希望在整个项目中共享一个“全局默认配置”，
# 也可以在这里取消下面一行的注释（看个人习惯）：
# config = Config()

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


