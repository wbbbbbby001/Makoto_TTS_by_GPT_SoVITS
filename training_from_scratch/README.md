# 🏗️ GPT-SoVITS 从零训练指南

> 用你自己的音频数据，从零训练一个角色语音克隆模型。

本指南假设你只有一段原始人声音频（如 `dataset/vocals.wav`），带你走完 GPT-SoVITS 的完整训练流水线。最终产出可直接用于推理的微调权重。

---

## 📖 目录

- [准备工作](#准备工作)
- [第一步：音频预处理](#第一步音频预处理)
- [第二步：数据集制作](#第二步数据集制作)
- [第三步：GPT 训练 (Stage 1)](#第三步gpt-训练-stage-1)
- [第四步：SoVITS v2 训练 (Stage 2)](#第四步sovits-v2-训练-stage-2)
- [第五步：LoRA v3 微调 (Stage 3)](#第五步lora-v3-微调-stage-3)
- [第六步：推理测试](#第六步推理测试)
- [训练配置说明](#训练配置说明)
- [目录结构对照](#目录结构对照)

---

## 准备工作

### 系统要求

| 组件 | 最低要求 | 本项目实际 |
|------|---------|-----------|
| GPU | NVIDIA 6 GB VRAM | RTX 5060 Laptop (8 GB) |
| 内存 | 16 GB RAM | 16 GB |
| 磁盘 | 20 GB 空闲 | 项目约占用 5 GB（含预训练模型） |
| OS | Windows / Linux | Windows 11 |
| Python | 3.9 - 3.11 | 3.9.13 |
| PyTorch | 2.0+ CUDA | 2.8.0+cu128 |

### 安装 GPT-SoVITS

```bash
# 1. 克隆官方框架
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载预训练基座模型
# 方式 A: 使用框架自带脚本
python download.py

# 方式 B: 手动下载后放入 GPT_SoVITS/pretrained_models/
# 需要: chinese-hubert-base, chinese-roberta-wwm-ext-large,
#       gsv-v2final-pretrained/ (s1bert25hz-5kh-longer, s2G2333k, s2D2333k),
#       s2Gv3.pth (用于 LoRA v3 阶段)
```

### 准备你的音频

将你的人声音频文件放到 `dataset/vocals.wav`。

**音频要求**：
- 格式：WAV，单声道，建议 32000 Hz
- 时长：至少 3-5 分钟（越长越好，50 条切片约需 10 分钟原始音频）
- 内容：目标角色的干净人声（无背景音乐、无多人对话）
- 语言：支持中日英多语种（日语需要额外的发音词典）

> 📌 本仓库提供了一个**日语动漫角色（橘真琴）的示例音频**作为参考：`dataset/vocals.wav`（约 10 分钟人声，50 段有效对白）。你可以直接用这个文件跑通完整流程，也可以替换成你自己的音频。

---

## 第一步：音频预处理

将原始人声切分为短片段（1-5 秒），并去除背景噪声。

### 启动 WebUI

```bash
cd GPT-SoVITS
python webui.py
```

浏览器打开 `http://localhost:9874`。

### 1A. UVR5 人声分离（可选）

如果你的音频已经是人声干音（无伴奏/无背景音），可跳过此步。

在 WebUI 中进入 **UVR5 伴奏分离** 标签页：
1. 选择模型：`HP5_only_main_vocal`
2. 输入音频：`dataset/vocals.wav`
3. 点击"转换"

输出在 `output/uvr5_opt/`。

### 1B. 音频切片

在 WebUI 中进入 **语音切分** 标签页：
1. 输入路径：`output/uvr5_opt/vocals.wav_10.wav`（或你的原始人声）
2. 调整参数（默认值通常可用）：
   - `threshold`: -34 dB（静音判定阈值）
   - `min_length`: 1000 ms（最短片段）
   - `min_interval`: 300 ms（最短间隔）
   - `max_sil_kept`: 500 ms（保留句尾静音）
3. 点击"开始语音切分"

输出在 `output/slicer_opt/`，文件名格式：`xxx_XXXXXXXX_YYYYYYYY.wav`。

### 1C. 降噪

在 WebUI 中进入 **语音降噪** 标签页：
1. 输入路径：`output/slicer_opt/`
2. 点击"开始语音降噪"

输出在 `output/denoise_opt/`。

---

## 第二步：数据集制作

将切分后的音频转换为训练所需的特征和标注。

### 2A. ASR 语音识别

在 WebUI 中进入 **语音文本校对** 标签页：
1. 选择 ASR 模型：`Faster-Whisper Large v3`（支持日语/多语种）
2. 音频路径：`output/denoise_opt/`
3. 点击"开始语音识别"

输出在 `output/asr_opt/`，生成 `.list` 文件。

### 2B. 数据集特征提取

在 WebUI 中进入 **数据集制作** 标签页：
1. 选择数据集名称：例如 `makoto`
2. 音频路径：`output/slicer_opt/`（注意：用切片音频而非降噪音频）
3. 标注文件：`output/asr_opt/denoise_opt.list`
4. 点击"一键三连"或分别执行 BERT 特征提取 → HuBERT 特征提取 → 语义令牌提取

输出目录结构：
```
logs/makoto/
├── 2-name2text.txt      # 音素序列标注
├── 3-bert/              # BERT 语义特征
├── 4-cnhubert/          # HuBERT 音频特征
├── 5-wav32k/            # 重采样至 32kHz 的音频
├── 6-name2semantic.tsv   # 语义令牌序列
└── config.json           # 自动生成的训练配置
```

> 📌 确认 `logs/makoto/2-name2text.txt` 有内容（条数 = 切片数），`4-cnhubert/` 下有 `.pt` 文件。如果这两个为空，后续训练会直接报错。

---

## 第三步：GPT 训练 (Stage 1)

### 目标

训练 AR Transformer 模型，学习从**文本音素 → 语义令牌**的映射。

### 训练配置

复制本仓库的 `configs/tmp_s1.yaml` 并根据你的情况修改：

```yaml
# configs/s1_train.yaml — GPT 阶段训练配置

train:
  batch_size: 4              # 根据 GPU 显存调整（4→2→1）
  epochs: 15                 # 训练轮数
  save_every_n_epoch: 5      # 每隔 N 轮保存一次
  precision: "16-mixed"      # 混合精度训练
  exp_name: "my_voice"       # ★ 你的角色/模型名称
  half_weights_save_dir: "GPT_weights_v2"  # 权重输出目录

model:
  n_layer: 24                # Transformer 层数
  embedding_dim: 512
  hidden_dim: 512
  head: 16                   # 多头注意力头数
  phoneme_vocab_size: 732    # 音素词汇量（日语+多语种）

data:
  max_sec: 54                # 最大音频长度（秒）
  num_workers: 4             # DataLoader 工作线程

# ★ 预训练基座
pretrained_s1: "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"

# ★ 训练数据路径
output_dir: "logs/my_voice/logs_s1_v2"
train_semantic_path: "logs/my_voice/6-name2semantic.tsv"
train_phoneme_path: "logs/my_voice/2-name2text.txt"
```

### 启动训练

```bash
cd GPT_SoVITS
python s1_train.py -c ../path/to/configs/s1_train.yaml
```

### 训练监控

```bash
# 另开终端，查看 TensorBoard
tensorboard --logdir logs/my_voice/logs_s1_v2
```

浏览器打开 `http://localhost:6006`。

### 预期结果

- 15 epochs、batch_size=4、50 条数据的情况下，约需 5-15 分钟（取决于 GPU）
- 输出权重：`GPT_weights_v2/my_voice-e5.ckpt`、`-e10.ckpt`、`-e15.ckpt`

| 参考值（RTX 5060 Laptop） | |
|---|---|
| 每 epoch 耗时 | ~30 秒 |
| 15 epochs 总耗时 | ~8 分钟 |
| 权重文件大小 | ~148 MB / 个 |

---

## 第四步：SoVITS v2 训练 (Stage 2)

### 目标

训练 VITS 变体模型，学习从**语义令牌 → Mel 频谱 → 音频波形**的映射。

### 训练配置

复制本仓库的 `configs/tmp_s2.json` 并根据你的情况修改：

```json
{
  "train": {
    "epochs": 8,
    "batch_size": 4,
    "learning_rate": 0.0001,
    "fp16_run": true,
    "save_every_epoch": 4,
    "if_save_every_weights": true,
    "if_save_latest": true,
    "pretrained_s2G": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
    "pretrained_s2D": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth",
    "gpu_numbers": "0"
  },
  "data": {
    "exp_dir": "logs/my_voice",
    "sampling_rate": 32000,
    "n_mel_channels": 128,
    "n_speakers": 300
  },
  "model": {
    "inter_channels": 192,
    "hidden_channels": 192,
    "n_layers": 6,
    "n_heads": 2,
    "semantic_frame_rate": "25hz",
    "version": "v2"
  },
  "s2_ckpt_dir": "logs/my_voice",
  "save_weight_dir": "SoVITS_weights_v2",
  "name": "my_voice",
  "version": "v2"
}
```

### 启动训练

```bash
cd GPT_SoVITS
python s2_train.py -c ../path/to/configs/s2_train.json
```

### 预期结果

- 8 epochs、batch_size=4、50 条数据的情况下，约需 15-30 分钟
- 输出权重：`SoVITS_weights_v2/my_voice_e4_s100.pth`、`my_voice_e8_s200.pth`

| 参考值（RTX 5060 Laptop） | |
|---|---|
| 每 epoch (25 steps) | ~90 秒 |
| 8 epochs 总耗时 | ~12 分钟 |
| 权重文件大小 | ~81 MB / 个 |

---

## 第五步：LoRA v3 微调 (Stage 3)

### 目标

在 v3 架构（`SynthesizerTrnV3`）上使用 LoRA（Low-Rank Adaptation）进行高效微调，进一步提升音质。

### 关键配置修改

在 SoVITS v2 的 `configs/s2_train.json` 基础上，修改以下 4 个字段：

| 字段 | v2 全量训练值 | v3 LoRA 训练值 |
|------|-------------|---------------|
| `train.pretrained_s2G` | `gsv-v2final-pretrained/s2G2333k.pth` | `GPT_SoVITS/pretrained_models/s2Gv3.pth` |
| `save_weight_dir` | `SoVITS_weights_v2` | `SoVITS_weights_v3` |
| `model.version` | `v2` | `v3` |
| `train.save_every_epoch` | `4` | `1`（确保每轮都保存） |
| `train.epochs` | `8` | `2`（LoRA 收敛快，2 轮即可） |

### 启动训练

```bash
cd GPT_SoVITS
python s2_train_v3_lora.py -c ../path/to/configs/s2_lora_train.json
```

### 预期结果

- 2 epochs、batch_size=4 的情况下，约需 1-2 分钟
- 输出权重：`SoVITS_weights_v3/my_voice_e1_s25_l32.pth`、`my_voice_e2_s50_l32.pth`

### 关于 LoRA 权重的加载

LoRA 权重使用了自定义文件头格式（`b'03'` 标记），保存时通过 `my_save2()` 写入。加载时必须使用配套函数：

```python
from process_ckpt import load_sovits_new

# ✅ 正确：用框架自带的加载器
ckpt = load_sovits_new("SoVITS_weights_v3/my_voice_e2_s50_l32.pth")
print(ckpt["lora_rank"])  # 32
print(len(ckpt["weight"])) # 258

# ❌ 错误：直接 torch.load 会报 UnpicklingError
# torch.load("SoVITS_weights_v3/my_voice_e2_s50_l32.pth")  # 报错!
```

---

## 第六步：推理测试

### Python 脚本推理

```python
import sys, os, yaml, time, soundfile as sf

os.chdir("GPT-SoVITS")
sys.path.insert(0, ".")
sys.path.insert(0, "GPT_SoVITS")

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

# 1. 配置模型路径
with open("GPT_SoVITS/configs/tts_infer.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
config["custom"]["t2s_weights_path"] = "GPT_weights_v2/my_voice-e15.ckpt"
config["custom"]["vits_weights_path"] = "SoVITS_weights_v2/my_voice_e8_s200.pth"
config["custom"]["version"] = "v2"
config["custom"]["device"] = "cuda"
config["custom"]["is_half"] = True

# 2. 初始化
tts = TTS(TTS_Config(config))

# 3. 准备参考音频和标注
ref_audio = "output/denoise_opt/xxx.wav"  # 选一个切片作为参考
prompt_text = ""  # 从 logs/my_voice/2-name2text.txt 中获取

# 4. 推理
inputs = {
    "text": "你好，这是我的语音克隆测试。",
    "text_lang": "zh",  # 或 "ja" / "en"
    "ref_audio_path": ref_audio,
    "prompt_text": prompt_text,
    "prompt_lang": "zh",
    "top_k": 15,
    "temperature": 1.0,
}

gen = tts.run(inputs)
sr, audio = None, None
for chunk in gen:
    if isinstance(chunk, tuple) and len(chunk) == 2:
        sr, audio = chunk

sf.write("test_output.wav", audio, sr)
print(f"✅ 生成完毕: {len(audio)/sr:.1f}秒 @ {sr}Hz")
```

### WebUI 推理

```bash
python webui.py
# 或
python GPT_SoVITS/inference_webui.py
```

---

## 训练配置说明

### 内存/显存不足时的调整

| 问题 | 解决方案 |
|------|---------|
| GPU OOM（Out of Memory） | `batch_size`: 4 → 2 → 1 |
| 显存仍然不足 | 启用 `grad_ckpt: true`（用时间换空间） |
| 系统内存不足 | `num_workers`: 4 → 1 |
| 训练太慢 | 减小 `max_sec`（如 54 → 30） |
| 过拟合（数据集小） | 减少 `epochs`，或增加 `dropout`（如 0 → 0.1） |

### 训练阶段对照表

```
音频预处理（UVR5→切片→降噪）
    │
    ├── 数据集制作（ASR→BERT→HuBERT→语义令牌）
    │
    ├── [Stage 1] GPT 训练 (Text → Semantic Tokens)
    │       输出: GPT_weights_v2/<name>-e*.ckpt
    │
    ├── [Stage 2] SoVITS v2 训练 (Semantic → Audio)
    │       输出: SoVITS_weights_v2/<name>_e*_s*.pth
    │
    └── [Stage 3] LoRA v3 微调 (可选, 增强音质)
            输出: SoVITS_weights_v3/<name>_e*_s*_l32.pth
```

### 项目文件夹结构（训练完成后）

```
your_project/
├── dataset/
│   └── vocals.wav                    # 原始人声音频
├── output/
│   ├── slicer_opt/                   # 切片后音频
│   ├── denoise_opt/                  # 降噪后音频
│   └── asr_opt/                      # ASR 转录结果
├── logs/<name>/
│   ├── 2-name2text.txt               # 音素标注
│   ├── 6-name2semantic.tsv           # 语义令牌
│   ├── 3-bert/                       # BERT 特征
│   ├── 4-cnhubert/                   # HuBERT 特征
│   ├── 5-wav32k/                     # 重采样音频
│   └── config.json                   # 自动生成配置
├── GPT_weights_v2/                   # ★ Stage 1 产出
│   └── <name>-e*.ckpt               #     GPT 微调权重
├── SoVITS_weights_v2/                # ★ Stage 2 产出
│   └── <name>_e*_s*.pth             #     SoVITS v2 微调权重
└── SoVITS_weights_v3/                # ★ Stage 3 产出（可选）
    └── <name>_e*_s*_l32.pth          #     LoRA v3 权重
```

---

## 目录结构对照

本指南对应的文件夹内容：

```
training_from_scratch/
├── README.md                  # 本文件
├── dataset/
│   └── vocals.wav             # 示例原始音频（橘真琴日语人声，10分钟，40 MB）
└── configs/
    ├── s1_train.yaml          # GPT 训练配置模板
    └── s2_train.json          # SoVITS v2 + LoRA v3 训练配置模板
```

**如何使用本文件夹**：
1. 将本文件夹拷贝到 GPT-SoVITS 框架同级目录
2. 替换 `dataset/vocals.wav` 为你自己的音频
3. 修改 `configs/` 中的配置（将 `makoto` 改为你的角色名）
4. 按本指南从头执行训练流程

> 📌 本仓库根目录的 `README_MAKOTO.md` 提供了橘真琴的**预训练权重和快速推理指南**。如果你只想使用已训练好的模型进行推理，请参考那个文件。本指南适用于想用自己的数据从零训练的场景。

---

## 常见问题

### Q: 中文/英文语音支持吗？

支持。GPT-SoVITS 对中日英三语都有良好支持。训练时无需额外配置，只需确保你的训练音频是目标语言。推理时在 `text_lang` 中指定 `"zh"` / `"ja"` / `"en"`。

### Q: 最少需要多少音频数据？

建议至少 3-5 分钟有效人声（约 30-50 个切片）。数据越多质量越好。本项目示例为约 10 分钟 / 50 个切片。

### Q: 训练到一半断了怎么办？

GPT-SoVITS 支持自动 resume。重新运行训练命令即可从最近的 checkpoint 继续。Stage 1 的 checkpoint 在 `logs/<name>/logs_s1_v2/ckpt/`，Stage 2/3 的 checkpoint 在 `logs/<name>/logs_s2_v3_lora_32/` 等目录。

### Q: 可以用 CPU 训练吗？

可以，但会非常慢（Stage 1 可能从 8 分钟变成 8 小时）。建议至少使用带有 CUDA 的 NVIDIA GPU。

### Q: 框架和依赖版本有要求吗？

本指南基于 GPT-SoVITS v3lora (2025-02-28) 版本。关键依赖：
- Python 3.9
- PyTorch 2.0+ CUDA
- peft（用于 LoRA 训练）
- pytorch_lightning（Stage 1）

如遇到版本兼容问题，可参考根目录 [README_MAKOTO.md](../README_MAKOTO.md) 的已验证环境信息。

---

## 参考

- [GPT-SoVITS 官方仓库](https://github.com/RVC-Boss/GPT-SoVITS)
- [本仓库主 README（预训练模型 + 推理）](../README_MAKOTO.md)
- [验证报告](../VERIFICATION_REPORT.md)
- [Free! 橘真琴 Wiki](https://zh.wikipedia.org/wiki/Free!)
