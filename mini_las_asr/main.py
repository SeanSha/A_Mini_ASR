"""
main.py

负责：整个项目的入口脚本。

典型流程：
    - 解析命令行参数或直接使用 Config
    - 初始化 Config、Vocab、Dataset、DataLoader
    - 初始化 LASModel、优化器、损失函数
    - 根据模式选择：训练 / 评估 / 解码 / 可视化
"""

from .config.config import Config
from .data.vocab import CharVocab
from .data.dataset import LJSpeechDataset
from .data.collate import collate_fn
from .models.las import LASModel
from .training.train import train_one_epoch, validate
from .inference.decode import greedy_decode_dataloader
from .visualization.visualize import plot_attention


def main():
    """
    Purpose:
        - 作为项目的统一入口，串联所有模块
        - 提供简单的命令行界面，指定训练或解码等操作

    Inputs:
        - 通常通过 argparse 从命令行读取参数（例如 --mode train / decode）

    Outputs:
        - 无直接返回值，通过日志与模型文件等形式对外输出结果

    Tensor shapes:
        - 不直接涉及张量计算，但会负责构建所有张量流动的组件

    Common mistakes:
        - 在这里写太多逻辑，导致 main 变得复杂难懂
        - 没有清晰区分“训练模式”和“推理模式”的代码路径
        - 忘记将所有相关模块迁移到相同的 device 上（cpu/gpu 混用出错）

    Relationship to the next step in the pipeline:
        - main 是用户使用本项目的入口
        - 本文件将所有前面构建的骨架模块串联起来，形成完整的 ASR 系统
    """
    # TODO:
    # 1. 创建 Config 实例
    # 2. 构建 CharVocab（可以从训练集文本统计得到）
    # 3. 创建 Dataset 和 DataLoader
    # 4. 初始化 LASModel、优化器和损失函数
    # 5. 根据 mode 决定是训练、验证还是仅解码/可视化
    pass


if __name__ == "__main__":
    main()


