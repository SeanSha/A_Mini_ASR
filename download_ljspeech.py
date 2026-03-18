"""
一个用于下载 LJSpeech-1.1 数据集的脚本骨架。

目标（第一版）：
1. 能够从官方 URL 下载压缩包到本地某个目录。
2. 能够把压缩包解压出来，得到包含 wavs/ 和 metadata.csv 的目录。
3. 整个过程里，多打印一些信息，让自己知道代码跑到哪一步了。

注意：这个文件目前大部分地方都是 TODO，需要你一点点填上去。
"""

# ==== 导入标准库 ====
# os / pathlib: 处理路径
# typing.Optional: 表示参数可以是 None 或 Path
# tarfile: 处理 .tar/.tar.bz2 解压
# （requests 等第三方库暂时先不导入，等你实现下载函数时自己决定用什么）

# TODO: 导入 os 模块，用于路径相关操作（例如 os.path.exists）
# import os
import os

# TODO: 从 pathlib 模块导入 Path 类，方便操作路径
# from pathlib import Path
from pathlib import Path

# TODO: 从 typing 模块导入 Optional 类型（Optional[Path] 表示"要么是 Path，要么是 None"）
# from typing import Optional
from typing import Optional

# ==== 常量定义 ====
# 官方 LJSpeech-1.1 下载地址（暂时先相信它是对的）
LJSPEECH_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"

# 压缩包文件名（我们希望下载到本地时就叫这个名字）
LJSPEECH_ARCHIVE_NAME = "LJSpeech-1.1.tar.bz2"

# 解压后文件夹的名字（一般是 LJSpeech-1.1 这个目录，里面有 wavs/ 和 metadata.csv）
LJSPEECH_DIR_NAME = "LJSpeech-1.1"


# ==== 辅助函数 1：确保目录存在 ====
def ensure_dir(path):
    """
    确保某个目录存在：

    - 如果目录已经存在，什么都不做。
    - 如果目录不存在，就创建这个目录（包括中间的父目录）。

    参数
    ----
    path: 可以是字符串（str），也可以是 Path 对象。
    """

    # 提示 1：如果传进来的是字符串，你可以用 Path(path) 把它变成 Path 对象
    # 提示 2：Path 对象有一个 .mkdir 方法，可以创建目录，参数 parents=True, exist_ok=True

    # TODO:
    # 1. 把 path 转成 Path 对象（如果已经是 Path，也可以直接用）
    path_obj = Path(path)
    # 2. 调用 .mkdir(...) 创建目录
    path_obj.mkdir(parents=True, exist_ok=True)
    # 3. 打印一句话，比如：print(f"确保目录存在: {path_obj}")
    print(f"Ensure directory exists: {path_obj}")


# ==== 辅助函数 2：下载文件 ====
def download_file(url, dst_path):
    """
    从指定 URL 下载文件到本地的 dst_path。

    参数
    ----
    url:      字符串，远程文件的 URL
    dst_path: 本地保存路径，可以是字符串，也可以是 Path
    """

    # 提示 1：进度条使用 tqdm 库（pip install tqdm）
    #   tqdm 可以把任何迭代过程包装成带进度条的版本，支持百分比、速度、剩余时间显示。
    #   下载场景的惯用模式：
    #     from tqdm import tqdm
    #     with tqdm(...) as bar:
    #         # 每写入一个 chunk，就调用 bar.update(len(chunk)) 推进进度
    #
    # 提示 2：配合 requests 的 stream=True 可以按块读取文件，避免一次性占用大量内存
    #   resp = requests.get(url, stream=True)
    #   total_size = int(resp.headers.get("Content-Length", 0))  # 文件总字节数
    #   chunk_size = 1024 * 8  # 每次读取 8 KB
    #
    # 提示 3：tqdm 初始化时常用参数
    #   total=total_size  -> 告诉进度条总量是多少（单位：字节）
    #   unit="B"          -> 显示单位为 Byte
    #   unit_scale=True   -> 自动换算成 KB / MB
    #   desc=文件名        -> 进度条左边的说明文字

    # TODO:
    # 1. 把 dst_path 转成 Path 对象
    dst_path = Path(dst_path)
    # 2. 如果文件已经存在，print 一句"已存在，跳过下载"，然后 return
    if dst_path.exists():
        print(f"File already exists: {dst_path}")
        return
    # 3. 否则，用 requests.get(url, stream=True) 发起请求，读取响应头中的 Content-Length
    else:
        print(f"Downloading: {dst_path.name}")
        # TODO: import requests，发起请求，获取 resp
        import requests
        resp = requests.get(url, stream=True)
        # TODO: 从 resp.headers.get("Content-Length", 0) 获取文件总大小 total_size
        total_size = int(resp.headers.get("Content-Length", 0))

        # 4. 以二进制写入模式打开 dst_path，用 tqdm 包裹写入循环
        #    tqdm 是专门做进度条显示的库（pip install tqdm），支持百分比、速度、剩余时间
        
        with open(dst_path, "wb") as f:
            # TODO: from tqdm import tqdm
            from tqdm import tqdm
            # TODO: with tqdm(total=total_size, unit="B", unit_scale=True, desc=dst_path.name) as bar:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=dst_path.name) as bar:
            # TODO:     for chunk in resp.iter_content(chunk_size=1024 * 8):
                for chunk in resp.iter_content(chunk_size=1024 * 8):
                    # TODO:         f.write(chunk)
                    f.write(chunk)
                    # TODO:         bar.update(len(chunk))
                    bar.update(len(chunk))

        # 5. 下载完成后，打印"下载完成: 文件路径"
        print(f"Download completed: {dst_path}")


# ==== 辅助函数 3：解压压缩包 ====
def extract_archive(archive_path, target_root):
    """
    解压 LJSpeech 的 .tar.bz2 压缩包到指定目录。

    参数
    ----
    archive_path: 压缩包路径（字符串或 Path）
    target_root:  目标根目录（字符串或 Path），
                  解压后会在这里出现一个 LJSPEECH_DIR_NAME 的子目录。
    """

    # 提示 1：使用标准库 tarfile
    #   import tarfile
    #   with tarfile.open(archive_path, "r:bz2") as tar:
    #       tar.extractall(target_root)
    #
    # 提示 2：注意路径也可以先转成 Path 再转成 str

    # TODO:
    # 1. 把 archive_path / target_root 都转成 Path 对象
    archive_path = Path(archive_path)
    target_root = Path(target_root)
    # 2. 打印一条解压开始的信息
    print(f"Extracting archive: {archive_path} to {target_root}")
    # 3. 使用 tarfile 打开并 extractall 到 target_root
    import tarfile
    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(target_root)
    # 4. 打印一条解压完成的信息
    print(f"Archive extracted to: {target_root}")


# ==== 高层封装：准备 LJSpeech 数据 ====
def prepare_ljspeech(data_root=None):
    """
    高层函数：负责"整体流程"：

    1. 确定 data_root 目录（数据存放的根目录）。
       - 如果传入了 data_root 参数，就用它；
       - 如果没有传入，可以暂时写死一个默认目录，比如 "./data"。
       （后面你可以再改成从 Config 里读）

    2. 确保 data_root 目录存在。

    3. 确定压缩包路径 archive_path = data_root / LJSPEECH_ARCHIVE_NAME

    4. 如果解压后的目录已经存在（data_root / LJSPEECH_DIR_NAME）：
       - 说明以前已经下载并解压过了，可以直接返回这个目录。

    5. 否则：
       - 调用 download_file 下载压缩包
       - 调用 extract_archive 解压

    6. 返回解压后的目录路径（Path 对象）。
    """

    # TODO:
    # 1. 如果 data_root 是 None，就把它设成 Path("./data")
    if data_root is None:
        data_root = Path("./data")
    # 2. 调用 ensure_dir(data_root)
    ensure_dir(data_root)
    # 3. 计算 archive_path 和 ljspeech_dir（都用 Path 拼接）
    archive_path = data_root / LJSPEECH_ARCHIVE_NAME
    ljspeech_dir = data_root / LJSPEECH_DIR_NAME
    # 4. 如果 ljspeech_dir 已存在，打印一条提示并返回
    if ljspeech_dir.exists():
        print(f"LJSpeech directory already exists: {ljspeech_dir}")
        return ljspeech_dir
    # 5. 如果不存在：
    else:
        #    - 调用 download_file(LJSPEECH_URL, archive_path)
        download_file(LJSPEECH_URL, archive_path)
        #    - 调用 extract_archive(archive_path, data_root)
        extract_archive(archive_path, data_root)
        # 6. 最后返回 ljspeech_dir
        return ljspeech_dir


# ==== 脚本入口 ====
def main():
    """
    这个函数是脚本的入口：

    当你在命令行里执行：
        python download_ljspeech.py

    Python 会从文件顶部开始执行，
    遇到 `if __name__ == "__main__": main()` 时，就会调用这里的 main()。
    """

    # TODO:
    # 1. 调用 prepare_ljspeech()，拿到 ljspeech_dir
    ljspeech_dir = prepare_ljspeech()
    # 2. 根据 ljspeech_dir 拼出 metadata.csv 和 wavs 目录的路径
    metadata_path = ljspeech_dir / "metadata.csv"
    wavs_dir = ljspeech_dir / "wavs"
    # 3. 打印这两个路径，确认你知道数据在哪里
    print(f"metadata path: {metadata_path}")
    print(f"wavs dir: {wavs_dir}")


# 只有当这个文件被"直接运行"时，才会进入这里；
# 如果是被其他文件 import，不会执行 main()。
if __name__ == "__main__":
    main()
