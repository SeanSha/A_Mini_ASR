### Mini LAS 项目实现顺序总览（从易到难）

> 这个文件是给你自己看的“学习路线图”。  
> 你可以每完成一个 Step，就在旁边打勾或写日期，方便以后接着往下做。

---

## Step 1：配置与词表（最简单，无深度学习）

### Step 1.1：`config/config.py`（Config）

- **要完成什么**
  - 把 `Config` 里的路径改成你本地真实的：
    - `data_root`, `metadata_path`, `feature_cache_dir` 等。
- **要理解什么**
  - 每个配置参数影响哪一层：
    - `sample_rate, n_fft, hop_length, n_mels` → `data/audio_utils.py` 里的特征形状。
    - `encoder_hidden_dim, decoder_hidden_dim, attention_dim` → `models/encoder.py`, `models/attention.py`, `models/decoder.py` 的维度。
    - `batch_size, learning_rate, teacher_forcing_ratio` → `training/train.py` 的训练行为。
- **怎么测试**
  - 在 Python REPL 中：
    ```python
    from mini_las_asr.config.config import Config
    cfg = Config()
    print(cfg.sample_rate, cfg.n_mels, cfg.encoder_hidden_dim)
    ```
  - 能正常创建实例、访问字段，即通过。

---

### Step 1.2：`data/vocab.py`（CharVocab）

- **要完成什么**
  - 在 `__init__` 中：
    - 把 `<pad>`, `<sos>`, `<eos>`, `<unk>` 放入词表，分配固定 id。
    - 把普通字符 `chars` 依次加入，填好 `char2id` / `id2char`。
  - 在 `text_to_ids` 中：
    - 可选预处理（如转小写、去多余空格）。
    - 字符映射到 id（未知字符用 `unk_id`）。
    - 视设计决定是否前后加 `<sos>/<eos>`。
  - 在 `ids_to_text` 中：
    - 过滤掉 `<pad>/<sos>/<eos>` 等特殊 token，其他 id 转回字符并拼接成字符串。
- **要理解什么**
  - 文本 ↔ id 序列 的双向关系：
    - 训练：`text -> ids -> tensor (B, T_out)` 喂给 Decoder。
    - 推理：`预测 ids -> text` 用于 CER 与查看结果。
  - 特殊 token 的语义及作用范围。
- **怎么测试**
  ```python
  from mini_las_asr.data.vocab import CharVocab
  chars = list("abcdefghijklmnopqrstuvwxyz '")
  vocab = CharVocab(chars)
  ids = vocab.text_to_ids("hello world")
  print(ids)
  print(vocab.ids_to_text(ids))
  ```
  - 看输出是否合理（有无 `<unk>`、是否基本还原原句）。

---

## Step 2：音频与特征（从 wav 到 log-Mel）

### Step 2.1：`data/audio_utils.py`

- **要完成什么**
  - `load_wav(path, sample_rate)`：
    - 用 `librosa` 或 `torchaudio` 读取 wav。
    - 按需重采样、归一化到 `[-1, 1]`。
  - `waveform_to_logmel(...)`：
    - waveform → STFT → Mel 滤波 → 对数 → `(T_in, n_mels)`。
  - `compute_feature_length(num_samples, hop_length)`：
    - 使用 `T_in ≈ num_samples // hop_length` 估算帧数。
- **要理解什么**
  - 波形 `(num_samples,)` 到 log-Mel `(T_in, n_mels)` 的**每一步形状变化**。
  - `hop_length` 如何控制 T_in 的大小。
- **怎么测试**
  ```python
  from mini_las_asr.config.config import Config
  from mini_las_asr.data.audio_utils import load_wav, waveform_to_logmel
  cfg = Config()
  wav = load_wav("某个 LJSpeech wav 路径", cfg.sample_rate)
  print(wav.shape)
  feat = waveform_to_logmel(
      wav, cfg.sample_rate, cfg.n_fft, cfg.win_length, cfg.hop_length, cfg.n_mels
  )
  print(feat.shape)  # 期望 (T_in, n_mels)
  ```
  - 观察 `T_in` 是否大致符合你预期（和音频时长成比例）。

---

## Step 3：Dataset 与 Collate（从单样本到 batch）

### Step 3.1：`data/dataset.py`（LJSpeechDataset）

- **要完成什么**
  - `__init__`：
    - 打开 `metadata.csv`，解析出 `(audio_path, text)`，存入 `self.samples`。
  - `__len__`：
    - 返回 `len(self.samples)`。
  - `__getitem__`：
    - `audio_path, text` →
      `load_wav` →
      `waveform_to_logmel` → `input: (T_in, n_mels)`。
    - `vocab.text_to_ids(text)` → `target: (T_out,)`。
    - 返回 dict：`{"input", "input_length", "target", "target_length", "text"}`。
- **要理解什么**
  - 单条样本的数据流：`metadata 行 -> 路径 + 文本 -> waveform -> log-Mel -> ids -> tensor`。
  - T_in / T_out 在不同样本之间是**变长**的。
- **怎么测试**
  ```python
  from mini_las_asr.config.config import Config
  from mini_las_asr.data.vocab import CharVocab
  from mini_las_asr.data.dataset import LJSpeechDataset

  cfg = Config()
  chars = list("abcdefghijklmnopqrstuvwxyz '")
  vocab = CharVocab(chars)
  ds = LJSpeechDataset(cfg.metadata_path, cfg.data_root, vocab, cfg)

  sample = ds[0]
  for k, v in sample.items():
      print(k, type(v), getattr(v, "shape", None))
  ```
  - 查看 `input` 与 `target` 的形状和 dtype 是否正确。

---

### Step 3.2：`data/collate.py`（collate_fn）

- **要完成什么**
  - 根据 batch 中每条样本的 `T_in_i`、`T_out_i`：
    - 求 `max_T_in`，构造 `inputs: (B, max_T_in, n_mels)`。
    - 求 `max_T_out`，构造 `targets: (B, max_T_out)`，用 `pad_id` 填充。
  - 同时返回 `input_lengths`, `target_lengths`。
- **要理解什么**
  - 变长序列是如何通过 padding + length 向量**变成可以进 GPU 的 batch 张量**的。
  - padding 只加在时间/序列维度，不改频率维。
- **怎么测试**
  ```python
  from torch.utils.data import DataLoader
  from mini_las_asr.data.collate import collate_fn

  dl = DataLoader(ds, batch_size=4, collate_fn=lambda b: collate_fn(b, vocab.pad_id))
  batch = next(iter(dl))
  for k, v in batch.items():
      print(k, v.shape, v.dtype)
  ```
  - 检查四个张量的形状是否为预期 `(B, max_T_in, n_mels)` 等。

---

## Step 4：Encoder & Attention（表征 + 对齐的核心）

### Step 4.1：`models/encoder.py`（BiLSTM Encoder）

- **要完成什么**
  - `__init__` 中定义双向 LSTM：`nn.LSTM(..., bidirectional=True, batch_first=True, num_layers=...)`。
  - `forward` 中：
    - 使用 `input_lengths` 做 pack → LSTM → pad 回来。
    - 返回：
      - `encoder_outputs: (B, T_in, 2 * hidden_dim)`
      - `encoder_final_hidden: (num_layers * 2, B, hidden_dim)`（或你设计的形状）。
- **要理解什么**
  - 双向 LSTM 如何把 `(B, T_in, D_in)` 变为 `(B, T_in, D_enc)`。
  - pack/pad 的作用：不在 padding 上浪费计算。
  - 为什么需要**整条 encoder_outputs 序列**供 attention 使用（而不仅仅是最后一个 hidden）。
- **怎么测试**
  ```python
  import torch
  from mini_las_asr.models.encoder import Encoder

  B, T_in, D_in = 3, 100, 80
  x = torch.randn(B, T_in, D_in)
  lengths = torch.tensor([100, 80, 60])

  enc = Encoder(input_dim=D_in, hidden_dim=256, num_layers=2, dropout=0.1)
  out, h = enc(x, lengths)
  print(out.shape)  # (B, T_in, 2*hidden_dim)
  print(h.shape)    # 视你的设计而定
  ```

---

### Step 4.2：`models/attention.py`（BahdanauAttention）

- **要完成什么**
  - 在 `__init__` 中定义：
    - `W_enc: Linear(D_enc -> D_att)`
    - `W_dec: Linear(D_dec -> D_att)`
    - `v: Linear(D_att -> 1)` 或 `nn.Parameter(D_att,)`。
  - 在 `forward` 中完整实现**一个解码步的注意力**：
    1. `enc_proj = W_enc(encoder_outputs)` → `(B, T_in, D_att)`
    2. `dec_proj = W_dec(decoder_hidden)` → `(B, D_att)` → broadcast 到 `(B, T_in, D_att)`
    3. `energy_tilde = tanh(enc_proj + dec_proj_expanded)`
    4. `scores = v(energy_tilde)` → `(B, T_in)`
    5. 应用 `encoder_mask`，再 softmax：`attention_weights: (B, T_in)`
    6. 加权求和：`context: (B, D_enc)`。
- **要理解什么**
  - Decoder hidden（query）、Encoder outputs（keys/values）、score、attention weights、context 的关系。
  - 注意力不是“选一个时间步”，而是对所有时间步做 softmax 的凸组合。
- **怎么测试**
  ```python
  import torch
  from mini_las_asr.models.attention import BahdanauAttention

  B, T_in, D_enc, D_dec, D_att = 2, 5, 512, 256, 128
  enc_out = torch.randn(B, T_in, D_enc)
  dec_h = torch.randn(B, D_dec)
  mask = torch.ones(B, T_in, dtype=torch.bool)

  attn = BahdanauAttention(D_enc, D_dec, D_att)
  ctx, alpha = attn(dec_h, enc_out, mask)
  print(ctx.shape, alpha.shape)      # (B, D_enc), (B, T_in)
  print(alpha.sum(dim=-1))          # 每个样本上应接近 1
  ```

---

## Step 5：Decoder & LASModel（完整前向）

### Step 5.1：`models/decoder.py`（先实现 forward_step）

- **要完成什么**
  - `__init__` 中：
    - `embedding`, `lstm`, `fc_out`, `dropout`。
  - `forward_step`：
    - `prev_tokens` → `embedding: (B, E)`
    - 从 `prev_hidden` 里取 top layer hidden 作为 `decoder_hidden`；
    - 调 `attention(decoder_hidden, encoder_outputs, encoder_mask)` → `context`;
    - 拼接 `[embedding, context]` → LSTM → 新 hidden/cell；
    - 用 LSTM 输出（或 hidden_top + context）过 `fc_out` → `logits: (B, vocab_size)`。
- **要理解什么**
  - 单个时间步的 Decoder + Attention 计算顺序。
  - Embedding、上下文向量、LSTM 输出如何一起作用于预测 logits。
- **怎么测试**
  - 构造随机张量跑一两步，检查维度不报错即可。

---

### Step 5.2：`Decoder.forward` & `models/las.py`（训练前向）

- **要完成什么**
  - `Decoder.forward`：
    - 循环解码长度 `T_out`，在循环内反复调用 `forward_step`；
    - 根据 `teacher_forcing_ratio` 决定下一个时间步的输入是：
      - 真实 target token，还是
      - 当前步 argmax 出来的预测 token。
  - `LASModel.forward`：
    - Encoder 前向 → `encoder_outputs`, `encoder_hidden`, `encoder_mask`；
    - 把 `encoder_hidden` 转成 Decoder 初始 hidden/cell（简单线性层或拼接均可）；
    - 调 `decoder.forward(...)` → `logits`, `all_attention_weights`。
- **要理解什么**
  - 整个训练前向流程如何从 `(inputs, input_lengths, targets)` 变成 `logits (B, T_out, V)`。
  - Teacher forcing 的概念和实现方式。
- **怎么测试**
  ```python
  import torch
  from mini_las_asr.models.las import LASModel
  from mini_las_asr.config.config import Config

  cfg = Config()
  vocab_size = 40
  model = LASModel(cfg, vocab_size)

  B, T_in, T_out = 2, 50, 20
  x = torch.randn(B, T_in, cfg.n_mels)
  len_x = torch.tensor([50, 40])
  y = torch.randint(0, vocab_size, (B, T_out))

  out = model(x, len_x, y, teacher_forcing_ratio=1.0, sos_id=1, eos_id=2)
  print(out["logits"].shape)  # (B, T_out, vocab_size)
  ```

---

## Step 6：训练、推理、评估与可视化

### Step 6.1：`training/train.py`（train_one_epoch, validate）

- **要完成什么**
  - `train_one_epoch`：
    - 标准 PyTorch 训练循环（model.train() / optimizer.zero_grad() / backward / step / clip）。
    - loss 使用 `ignore_index=pad_id`。
  - `validate`：
    - `model.eval()`, `torch.no_grad()`；
    - 使用 `model.greedy_decode` 得到预测 id；
    - 通过 `vocab.ids_to_text` 转成字符串；
    - 调 `compute_cer` 计算 CER。
- **要理解什么**
  - 如何正确 reshape logits / targets（如 `(B*T, V)` 与 `(B*T,)`）。
  - dropout / eval-mode 的区别。
- **怎么测试**
  - 只跑 1–2 个 epoch，看 loss 是否 **大致下降**，即使 CER 还很差也没关系。

---

### Step 6.2：`inference/decode.py`（greedy_decode_dataloader）

- **要完成什么**
  - 对整个 DataLoader：
    - 调 `model.greedy_decode` 得到 `pred_ids`；
    - 从 batch 或 dataset 取参考文本（ref）；
    - 用 `vocab.ids_to_text` 得到 `ref`, `hyp` 字符串；
    - 存入结果列表。
- **要理解什么**
  - 训练阶段 forward（需要 targets）与推理阶段 greedy_decode（不需要 targets）的区别。
  - `<sos>/<eos>` 在解码时的起止作用。
- **怎么测试**
  - 训练一点点之后，对验证集 decode 几条，肉眼看 `ref` vs `hyp`。

---

### Step 6.3：`evaluation/metrics.py`（CER）

- **要完成什么**
  - `levenshtein_distance`：标准 DP，实现插入/删除/替换统计。
  - `compute_cer`：对一批句子累加 sub+ins+del，除以**总参考字符数**。
- **要理解什么**
  - CER 的定义与 WER 的区别（按字符计）。
  - 空串、完全不匹配等边界情况如何处理。
- **怎么测试**
  - 用几个手写的例子验证 CER 是否和你手算的一致。

---

### Step 6.4：`visualization/visualization.py`（plot_attention）

- **要完成什么**
  - 从 `(T_out, T_in)` 的 attention matrix 画出热力图；
  - x 轴：输入时间步；y 轴：输出字符位置（可选标文字）。
- **要理解什么**
  - 如何从 batch attention `(B, T_out, T_in)` 中选取某一条 `(T_out, T_in)`。
  - `.detach().cpu().numpy()` 的基本用法。
- **怎么测试**
  - 训练一段时间后，对一条样本画出 attention 图，观察是否大致呈“斜线”形。

---

## Step 7：`main.py`（项目入口）

- **要完成什么**
  - 写一个简单的 `main()`：
    - 创建 `Config / Vocab / Dataset / DataLoader / LASModel / Optimizer / Criterion`；
    - 根据硬编码或命令行参数选择模式：`train / decode / visualize`。
- **要理解什么**
  - 从“外部用户角度”看，这个项目是怎么被使用的。
- **怎么测试**
  - 一开始可以只支持 `mode="train"`，确认能跑训练循环；
  - 后面再逐步加 `decode`、`visualize` 模式。


